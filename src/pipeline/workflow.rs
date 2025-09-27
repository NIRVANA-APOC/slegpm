use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use indexmap::IndexSet;
use petgraph::algo::isomorphism::is_isomorphic_matching;
use rayon::prelude::*;

use crate::graph::{EdgeAttributes, GraphInstance, NodeAttributes};
use crate::pipeline::candidate::{AnchorSelector, CandidateGenerator, SubgraphExtractor};
use crate::pipeline::preprocess::{GraphPreprocessor, PreprocessedGraph};
use crate::spectral::{spectral_distance, SpectralProfile};
use crate::wl::weisfeiler_lehman_isomorphic;

#[derive(Debug, Clone)]
pub struct WorkflowConfig {
    pub anchor_count: Option<usize>,
    pub epsilon: f64,
    pub wl_iterations: usize,
    pub spectral_dim: usize,
}

impl Default for WorkflowConfig {
    fn default() -> Self {
        Self {
            anchor_count: None,
            epsilon: 1e-3,
            wl_iterations: 3,
            spectral_dim: 16,
        }
    }
}

pub struct MatchingWorkflow {
    config: WorkflowConfig,
    pattern: Arc<PreprocessedGraph>,
    target: Arc<PreprocessedGraph>,
}

impl MatchingWorkflow {
    pub fn new(
        config: WorkflowConfig,
        pattern: impl Into<Arc<PreprocessedGraph>>,
        target: impl Into<Arc<PreprocessedGraph>>,
    ) -> Self {
        Self {
            config,
            pattern: pattern.into(),
            target: target.into(),
        }
    }

    pub fn execute(&self) -> Result<MatchingSummary> {
        let online_start = Instant::now();
        let pattern_size = self.pattern.node_count();

        let anchors = AnchorSelector::select(self.pattern.graph(), self.config.anchor_count);
        let candidate_ids = CandidateGenerator::filter_candidates(
            self.pattern.graph(),
            self.target.graph(),
            &anchors,
            self.config.epsilon,
        )?;

        let subgraphs =
            SubgraphExtractor::extract(self.target.graph(), &candidate_ids, pattern_size)?;

        let pattern_spectral = self.pattern.spectral()?;
        let pattern_arc = Arc::clone(&self.pattern);
        let pattern_fingerprint = Arc::new(PatternFingerprint::new(
            pattern_arc.graph(),
            self.config.epsilon,
        ));
        let config = self.config.clone();

        let evaluation_start = Instant::now();
        let accumulator = subgraphs
            .into_par_iter()
            .map(|candidate| {
                let fingerprint = Arc::clone(&pattern_fingerprint);
                evaluate_candidate(
                    pattern_arc.as_ref(),
                    fingerprint.as_ref(),
                    Arc::clone(&pattern_spectral),
                    candidate,
                    &config,
                )
            })
            .fold(WorkflowAccumulator::default, |mut acc, eval| {
                acc.consume(eval);
                acc
            })
            .reduce(WorkflowAccumulator::default, WorkflowAccumulator::combine);
        let matching_duration = evaluation_start.elapsed();

        let mut stats = accumulator.stats;
        let raw_matches = accumulator.matches;

        let mut unique = Vec::new();
        let mut seen = IndexSet::new();
        for graph in raw_matches {
            let graph_ref = graph.as_ref();
            let mut ids: Vec<_> = graph_ref.reverse_lookup.values().cloned().collect();
            ids.sort();
            if seen.insert(ids) {
                unique.push(graph);
            }
        }
        stats.matches = unique.len();

        let online_duration = online_start.elapsed();
        Ok(MatchingSummary {
            matches: unique,
            stats,
            online_duration,
            matching_duration,
        })
    }
}

#[derive(Clone, Debug)]
struct PatternFingerprint {
    edge_count: usize,
    degrees: Vec<usize>,
    signature_counts: HashMap<(Option<String>, Option<i64>), usize>,
    scale: f64,
}

impl PatternFingerprint {
    fn new(pattern: &GraphInstance, epsilon: f64) -> Self {
        let scale = quantization_scale(epsilon);
        Self {
            edge_count: pattern.edge_count(),
            degrees: degree_multiset(pattern),
            signature_counts: node_signature_counts(pattern, scale),
            scale,
        }
    }
}

fn evaluate_candidate(
    pattern: &PreprocessedGraph,
    pattern_fingerprint: &PatternFingerprint,
    pattern_spectral: Arc<SpectralProfile>,
    candidate: GraphInstance,
    config: &WorkflowConfig,
) -> CandidateEvaluation {
    if !fingerprint_compatible(pattern_fingerprint, &candidate) {
        return CandidateEvaluation::fingerprint_reject();
    }

    let Ok(candidate_processed) = GraphPreprocessor::from_instance(candidate, config.spectral_dim)
    else {
        return CandidateEvaluation::failed();
    };

    let Ok(candidate_spectral) = candidate_processed.spectral() else {
        return CandidateEvaluation::failed();
    };

    let distance = spectral_distance(pattern_spectral.as_ref(), candidate_spectral.as_ref());
    if distance > config.epsilon {
        return CandidateEvaluation::spectral_reject();
    }

    let wl_pass = weisfeiler_lehman_isomorphic(
        pattern.graph(),
        candidate_processed.graph(),
        config.wl_iterations,
    );

    if !wl_pass {
        return CandidateEvaluation::wl_reject(true);
    }

    let pattern_graph = pattern.graph();
    let candidate_graph = candidate_processed.graph();
    let vf2_success = is_isomorphic_matching(
        &pattern_graph.graph,
        &candidate_graph.graph,
        |a: &NodeAttributes, b: &NodeAttributes| node_match(a, b, config.epsilon),
        |a: &EdgeAttributes, b: &EdgeAttributes| edge_match(a, b, config.epsilon),
    );

    CandidateEvaluation::vf2_result(
        candidate_processed.instance_arc(),
        true,
        wl_pass,
        vf2_success,
    )
}

fn fingerprint_compatible(
    pattern: &PatternFingerprint,
    candidate: &GraphInstance,
) -> bool {
    if pattern.edge_count != candidate.edge_count() {
        return false;
    }

    if pattern.degrees != degree_multiset(candidate) {
        return false;
    }

    let candidate_counts = node_signature_counts(candidate, pattern.scale);
    pattern.signature_counts == candidate_counts
}

fn degree_multiset(graph: &GraphInstance) -> Vec<usize> {
    let mut degrees: Vec<_> = graph
        .graph
        .node_indices()
        .map(|node| graph.graph.neighbors(node).count())
        .collect();
    degrees.sort_unstable();
    degrees
}

fn quantization_scale(epsilon: f64) -> f64 {
    if epsilon > 0.0 {
        (1.0 / epsilon).max(1.0)
    } else {
        1e6
    }
}

fn node_signature_counts(
    graph: &GraphInstance,
    scale: f64,
) -> HashMap<(Option<String>, Option<i64>), usize> {
    let mut counts: HashMap<(Option<String>, Option<i64>), usize> = HashMap::new();
    for node in graph.graph.node_weights() {
        let weight_key = node.weight.map(|w| (w * scale).round() as i64);
        let key = (node.label.clone(), weight_key);
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

fn node_match(left: &NodeAttributes, right: &NodeAttributes, epsilon: f64) -> bool {
    if left.label != right.label {
        return false;
    }
    match (left.weight, right.weight) {
        (Some(l), Some(r)) => (l - r).abs() <= epsilon,
        (None, None) => true,
        _ => false,
    }
}

fn edge_match(left: &EdgeAttributes, right: &EdgeAttributes, epsilon: f64) -> bool {
    match (left.weight, right.weight) {
        (Some(l), Some(r)) => (l - r).abs() <= epsilon,
        (None, None) => true,
        _ => false,
    }
}

struct CandidateEvaluation {
    graph: Option<Arc<GraphInstance>>,
    spectral_pass: bool,
    wl_pass: bool,
    vf2_calls: usize,
    fingerprint_reject: bool,
}

impl CandidateEvaluation {
    fn failed() -> Self {
        Self {
            graph: None,
            spectral_pass: false,
            wl_pass: false,
            vf2_calls: 0,
            fingerprint_reject: false,
        }
    }

    fn spectral_reject() -> Self {
        Self {
            graph: None,
            spectral_pass: false,
            wl_pass: false,
            vf2_calls: 0,
            fingerprint_reject: false,
        }
    }

    fn wl_reject(spectral_pass: bool) -> Self {
        Self {
            graph: None,
            spectral_pass,
            wl_pass: false,
            vf2_calls: 0,
            fingerprint_reject: false,
        }
    }

    fn fingerprint_reject() -> Self {
        Self {
            graph: None,
            spectral_pass: false,
            wl_pass: false,
            vf2_calls: 0,
            fingerprint_reject: true,
        }
    }

    fn vf2_result(
        graph: Arc<GraphInstance>,
        spectral_pass: bool,
        wl_pass: bool,
        success: bool,
    ) -> Self {
        Self {
            graph: if success { Some(graph) } else { None },
            spectral_pass,
            wl_pass,
            vf2_calls: 1,
            fingerprint_reject: false,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct WorkflowStats {
    pub candidate_count: usize,
    pub spectral_pass: usize,
    pub wl_pass: usize,
    pub vf2_calls: usize,
    pub matches: usize,
    pub fingerprint_rejects: usize,
}

pub struct MatchingSummary {
    pub matches: Vec<Arc<GraphInstance>>,
    pub stats: WorkflowStats,
    pub online_duration: Duration,
    pub matching_duration: Duration,
}

#[derive(Default)]
struct WorkflowAccumulator {
    matches: Vec<Arc<GraphInstance>>,
    stats: WorkflowStats,
}

impl WorkflowAccumulator {
    fn consume(&mut self, eval: CandidateEvaluation) {
        self.stats.candidate_count += 1;
        if eval.fingerprint_reject {
            self.stats.fingerprint_rejects += 1;
        }
        if eval.spectral_pass {
            self.stats.spectral_pass += 1;
        }
        if eval.wl_pass {
            self.stats.wl_pass += 1;
        }
        self.stats.vf2_calls += eval.vf2_calls;
        if let Some(graph) = eval.graph {
            self.matches.push(graph);
        }
    }

    fn combine(mut self, other: Self) -> Self {
        self.matches.extend(other.matches);
        self.stats.candidate_count += other.stats.candidate_count;
        self.stats.fingerprint_rejects += other.stats.fingerprint_rejects;
        self.stats.spectral_pass += other.stats.spectral_pass;
        self.stats.wl_pass += other.stats.wl_pass;
        self.stats.vf2_calls += other.stats.vf2_calls;
        self
    }
}
