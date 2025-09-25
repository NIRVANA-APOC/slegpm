use std::time::{Duration, Instant};

use anyhow::Result;
use log::{debug, info};

use crate::graph::model::GraphInstance;
use crate::pipeline::candidate::{AnchorSelector, CandidateGenerator, SubgraphExtractor};
use crate::pipeline::matching::{CandidateBundle, MatchingOrchestrator};
use crate::pipeline::preprocess::PreprocessedGraph;

/// High-level configuration for running the online workflow.
#[derive(Debug, Clone)]
pub struct WorkflowConfig {
    pub anchor_count: Option<usize>,
    pub epsilon: f64,
    pub dominance_threshold: f64,
}

impl Default for WorkflowConfig {
    fn default() -> Self {
        Self {
            anchor_count: None,
            epsilon: 1e-6,
            dominance_threshold: 0.9,
        }
    }
}

/// Entry point struct orchestrating all modules (online stage only).
pub struct MatchingWorkflow {
    pub config: WorkflowConfig,
    pub pattern: PreprocessedGraph,
    pub target: PreprocessedGraph,
}

impl MatchingWorkflow {
    pub fn new(
        config: WorkflowConfig,
        pattern: PreprocessedGraph,
        target: PreprocessedGraph,
    ) -> Self {
        Self {
            config,
            pattern,
            target,
        }
    }

    pub fn execute(&self) -> Result<MatchingSummary> {
        info!("Online workflow started");
        let online_start = Instant::now();

        info!("Selecting anchors");
        let anchors =
            AnchorSelector::select_anchors(&self.pattern.instance, self.config.anchor_count);
        debug!("Anchors selected: {:?}", anchors);

        info!("Filtering candidate nodes");
        let candidates = CandidateGenerator::filter_candidates(
            &self.pattern.instance,
            &self.target.instance,
            &anchors,
            self.config.epsilon,
        )?;
        info!("Candidate nodes retained: {}", candidates.len());

        info!("Extracting candidate subgraphs");
        let subgraphs = SubgraphExtractor::extract_subgraphs(
            &self.target.instance,
            &candidates,
            self.pattern.node_count(),
        )?;
        info!("Candidate subgraphs produced: {}", subgraphs.len());

        info!("Refining candidates via spectral + WL screening");
        let mut bundles = MatchingOrchestrator::refine_candidates(
            &self.pattern.spectral,
            subgraphs,
            self.config.epsilon,
        )?;
        info!("Bundles after spectral + WL screening: {}", bundles.len());

        info!("Evaluating dominance constraints");
        MatchingOrchestrator::evaluate_dominance_parallel(
            &self.pattern.spectral,
            self.pattern.node_count(),
            &mut bundles,
            self.config.epsilon,
            self.config.dominance_threshold,
        );

        let mut audit_log = Vec::new();
        let dominant_candidates: Vec<CandidateBundle> = bundles
            .into_iter()
            .filter(|bundle| match &bundle.dominance {
                Some(report) => {
                    audit_log.push(report.explanation.clone());
                    report.is_dominant
                }
                None => false,
            })
            .collect();
        info!(
            "Dominant candidates ready for iterative matching: {}",
            dominant_candidates.len()
        );

        info!("Running iterative matching pipeline");
        let (matches, matching_duration, matching_notes) =
            MatchingOrchestrator::run_iterative_matching(
                &self.pattern.instance,
                &self.pattern.spectral,
                dominant_candidates,
                self.config.epsilon,
            )?;
        audit_log.extend(matching_notes);
        info!("Exact matches found: {}", matches.len());
        info!("Matching phase completed in {:?}", matching_duration);
        let online_duration = online_start.elapsed();
        info!("Online workflow completed in {:?}", online_duration);

        Ok(MatchingSummary {
            matches,
            audit_log,
            online_duration,
            matching_duration,
        })
    }
}

/// Aggregated output of the workflow.
pub struct MatchingSummary {
    pub matches: Vec<GraphInstance>,
    pub audit_log: Vec<String>,
    pub online_duration: Duration,
    pub matching_duration: Duration,
}
