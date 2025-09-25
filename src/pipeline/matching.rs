use anyhow::Result;
use indexmap::IndexSet;
use log::{debug, trace};
use petgraph::algo::isomorphism::{is_isomorphic_matching, is_isomorphic_subgraph_matching};
use petgraph::prelude::NodeIndex;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use crate::graph::construction::GraphLoader;
use crate::graph::model::{EdgeAttributes, GraphInstance, NodeAttributes};
use crate::spectral::features::SpectralProfile;
use crate::spectral::{NormalizedLaplacianBuilder, SpectralVectorExtractor};
use crate::verify::dominance::DominanceReport;
use crate::verify::{DominanceChecker, InterlacingVerifier};

const WL_ITERATIONS: usize = 3;
const NEAR_PATTERN_RATIO: f64 = 1.5;
const MAX_PRECISE_PER_CANDIDATE: usize = 32;
const MAX_ANCHOR_SHRINK_ATTEMPTS: usize = 4;
const COARSE_WL_DEFICIT_RATIO: f64 = 0.2;
const VF2_DEFICIT_FALLBACK_RATIO: f64 = 0.25;
const SPECTRAL_DISTANCE_RATIO: f64 = 1.5;
const SPECTRAL_EQUIVALENCE_ABS: f64 = 1e-2;

#[derive(Debug, Clone)]
pub struct CandidateBundle {
    pub graph: GraphInstance,
    pub spectral: SpectralProfile,
    pub wl_signature: WlSignature,
    pub dominance: Option<DominanceReport>,
}

#[derive(Debug, Default)]
struct PreciseFilterStats {
    total: usize,
    dedup: usize,
    too_small: usize,
    wl_mismatch: usize,
    wl_deficit: usize,
    spectral_fail: usize,
    accepted: usize,
    sample_wl_mismatch: Option<String>,
}

#[derive(Debug, Clone)]
pub struct WlSignature {
    histogram: BTreeMap<u64, usize>,
}

impl WlSignature {
    fn sorted_counts(&self) -> Vec<usize> {
        let mut counts: Vec<usize> = self.histogram.values().copied().collect();
        counts.sort_unstable_by(|a, b| b.cmp(a));
        counts
    }

    fn equivalent(&self, other: &Self) -> bool {
        self.sorted_counts() == other.sorted_counts()
    }

    fn deficit_stats(&self, other: &Self) -> (usize, usize) {
        let mut available = self.sorted_counts();
        let mut required = other.sorted_counts();
        let max_len = available.len().max(required.len());
        available.resize(max_len, 0);
        required.resize(max_len, 0);

        let mut missing_total = 0;
        let mut deficit_slots = 0;
        for (avail, need) in available.iter().zip(required.iter()) {
            if avail < need {
                missing_total += need - avail;
                deficit_slots += 1;
            }
        }
        (missing_total, deficit_slots)
    }
}

/// Combine spectral pruning with iterative matching refinement.
pub struct MatchingOrchestrator;

impl MatchingOrchestrator {
    pub fn refine_candidates(
        pattern_profile: &SpectralProfile,
        subgraphs: Vec<GraphInstance>,
        epsilon: f64,
    ) -> Result<Vec<CandidateBundle>> {
        let results: Vec<Result<Option<CandidateBundle>>> = subgraphs
            .into_par_iter()
            .map(|candidate| -> Result<Option<CandidateBundle>> {
                let laplacian = NormalizedLaplacianBuilder::build(&candidate)?;
                let spectral = SpectralVectorExtractor::compute_profile(&laplacian)?;
                if !spectral_dominates(&spectral, pattern_profile, epsilon) {
                    return Ok(None);
                }
                let wl_signature = compute_wl_signature(&candidate, 0);
                Ok(Some(CandidateBundle {
                    graph: candidate,
                    spectral,
                    wl_signature,
                    dominance: None,
                }))
            })
            .collect();

        let mut bundles = Vec::new();
        for res in results {
            if let Some(bundle) = res? {
                bundles.push(bundle);
            }
        }
        debug!(
            "Bundles retained after spectral/WL refinement: {}",
            bundles.len()
        );
        Ok(bundles)
    }

    pub fn run_iterative_matching(
        pattern: &GraphInstance,
        pattern_profile: &SpectralProfile,
        refined: Vec<CandidateBundle>,
        epsilon: f64,
    ) -> Result<(Vec<GraphInstance>, Duration, Vec<String>)> {
        let start = Instant::now();
        let pattern_signature_coarse = compute_wl_signature(pattern, 0);
        let pattern_signature_precise = compute_wl_signature(pattern, WL_ITERATIONS);
        let pattern_size = pattern.graph.node_count();

        let results: Vec<Result<(Vec<GraphInstance>, Vec<String>)>> = refined
            .into_par_iter()
            .map(|bundle| -> Result<(Vec<GraphInstance>, Vec<String>)> {
                let mut notes = Vec::new();
                let CandidateBundle {
                    graph,
                    spectral,
                    wl_signature,
                    ..
                } = bundle;

                let candidate_size = graph.graph.node_count();
                notes.push(format!(
                    "Candidate |V|={}, |E|={}",
                    candidate_size,
                    graph.graph.edge_count()
                ));

                if candidate_size < pattern_size {
                    notes.push("Rejected: candidate smaller than pattern".to_string());
                    return Ok((Vec::new(), notes));
                }

                if !spectral_dominates(&spectral, pattern_profile, epsilon) {
                    notes.push("Rejected: spectral dominance violated".to_string());
                    return Ok((Vec::new(), notes));
                }

                let (missing_total, deficit_slots) =
                    wl_signature.deficit_stats(&pattern_signature_coarse);
                let wl_ratio = if pattern_size > 0 {
                    missing_total as f64 / pattern_size as f64
                } else {
                    0.0
                };

                if missing_total == 0 {
                    notes.push("Coarse WL deficits: none".to_string());
                } else {
                    notes.push(format!(
                        "Coarse WL deficits: missing {} nodes across {} slots (ratio {:.3})",
                        missing_total,
                        deficit_slots,
                        wl_ratio
                    ));
                    if wl_ratio > COARSE_WL_DEFICIT_RATIO && candidate_size > pattern_size {
                        notes.push("Rejected: coarse WL deficit ratio too high".to_string());
                        return Ok((Vec::new(), notes));
                    }
                }

                let threshold = (pattern_size as f64 * NEAR_PATTERN_RATIO).ceil() as usize;
                let mut precise_candidates = if candidate_size <= threshold {
                    notes.push("Candidate within VF2 threshold; using directly".to_string());
                    vec![graph.clone()]
                } else {
                    notes.push(format!(
                        "Generating precise candidates from |V|={} (> threshold {})",
                        candidate_size, threshold
                    ));
                    generate_precise_candidates(&graph, pattern_size)?
                };

                if !precise_candidates
                    .iter()
                    .any(|candidate_graph| candidate_graph.graph.node_count() == candidate_size)
                {
                    precise_candidates.push(graph.clone());
                }

                notes.push(format!(
                    "Precise candidates generated: {}",
                    precise_candidates.len()
                ));

                if precise_candidates.is_empty() {
                    return Ok((Vec::new(), notes));
                }

                let (refined_precise, precise_stats) = evaluate_precise_candidates(
                    pattern_profile,
                    &pattern_signature_precise,
                    precise_candidates,
                    epsilon,
                )?;
                notes.push(format!(
                    "Precise candidates after spectral/WL checks: {} of {} (dedup {}, <pattern {}, WL mismatch {}, WL deficit {}, spectral {})",
                    refined_precise.len(),
                    precise_stats.total,
                    precise_stats.dedup,
                    precise_stats.too_small,
                    precise_stats.wl_mismatch,
                    precise_stats.wl_deficit,
                    precise_stats.spectral_fail
                ));
                if let Some(detail) = &precise_stats.sample_wl_mismatch {
                    notes.push(format!("WL mismatch sample: {}", detail));
                }

                let mut matches = Vec::new();
                for candidate_graph in refined_precise {
                    if run_vf2(pattern, &candidate_graph, epsilon) {
                        notes.push(format!(
                            "VF2 success on candidate |V|={}, |E|={}",
                            candidate_graph.graph.node_count(),
                            candidate_graph.graph.edge_count()
                        ));
                        matches.push(candidate_graph);
                    } else {
                        notes.push(format!(
                            "VF2 rejection on candidate |V|={}, |E|={}",
                            candidate_graph.graph.node_count(),
                            candidate_graph.graph.edge_count()
                        ));
                    }
                }

                if matches.is_empty() && wl_ratio <= VF2_DEFICIT_FALLBACK_RATIO {
                    if run_vf2(pattern, &graph, epsilon) {
                        notes.push("VF2 fallback on original candidate succeeded".to_string());
                        matches.push(graph.clone());
                    } else {
                        notes.push("VF2 fallback on original candidate failed".to_string());
                    }
                }

                Ok((matches, notes))
            })
            .collect();

        let mut matches = Vec::new();
        let mut audit = Vec::new();
        for res in results {
            let (mut candidate_matches, mut notes) = res?;
            matches.append(&mut candidate_matches);
            audit.append(&mut notes);
        }

        let duration = start.elapsed();
        Ok((matches, duration, audit))
    }

    pub fn evaluate_dominance_parallel(
        pattern_profile: &SpectralProfile,
        pattern_size: usize,
        bundles: &mut [CandidateBundle],
        epsilon: f64,
        threshold: f64,
    ) {
        bundles.par_iter_mut().for_each(|bundle| {
            let candidate_size = bundle.graph.graph.node_count();
            let interlacing = InterlacingVerifier::verify(
                &bundle.spectral,
                pattern_profile,
                candidate_size,
                pattern_size,
                epsilon,
            );
            let factor = if candidate_size > 0 {
                pattern_size as f64 / candidate_size as f64
            } else {
                1.0
            };
            let report = DominanceChecker::assess(
                interlacing.is_valid,
                &bundle.spectral,
                pattern_profile,
                factor,
                threshold,
                epsilon,
            );
            bundle.dominance = Some(report);
        });
    }
}

fn spectral_dominates(
    candidate: &SpectralProfile,
    pattern: &SpectralProfile,
    epsilon: f64,
) -> bool {
    let pattern_vals: Vec<f64> = pattern.eigenvalues.iter().copied().collect();
    let candidate_vals: Vec<f64> = candidate.eigenvalues.iter().copied().collect();
    if candidate_vals.len() < pattern_vals.len() {
        return false;
    }

    for (idx, pattern_value) in pattern_vals.iter().copied().enumerate() {
        if let Some(&candidate_value) = candidate_vals.get(idx) {
            if candidate_value + epsilon < pattern_value {
                trace!(
                    "Spectral dominance violated at index {}: candidate {:.6} < pattern {:.6}",
                    idx, candidate_value, pattern_value
                );
                return false;
            }
        } else {
            return false;
        }
    }

    candidate.norm_l2 + epsilon >= pattern.norm_l2 && candidate.trace + epsilon >= pattern.trace
}

fn spectral_distance(candidate: &SpectralProfile, pattern: &SpectralProfile) -> Option<f64> {
    let candidate_slice = candidate.eigenvalues.as_slice()?;
    let pattern_slice = pattern.eigenvalues.as_slice()?;
    let len = candidate_slice.len().min(pattern_slice.len());
    if len == 0 {
        return Some(0.0);
    }

    let mut sum_sq = 0.0;
    for idx in 0..len {
        let diff = candidate_slice[idx] - pattern_slice[idx];
        sum_sq += diff * diff;
    }
    Some(sum_sq.sqrt())
}

fn compute_wl_signature(graph: &GraphInstance, iterations: usize) -> WlSignature {
    let mut colors: indexmap::IndexMap<NodeIndex, u64> = indexmap::IndexMap::new();
    for node_idx in graph.graph.node_indices() {
        if let Some(attrs) = graph.graph.node_weight(node_idx) {
            let degree = graph.graph.neighbors(node_idx).count();
            colors.insert(node_idx, initial_color(attrs, degree));
        }
    }

    for _ in 0..iterations {
        let mut next = indexmap::IndexMap::with_capacity(colors.len());
        for (node_idx, color) in colors.iter() {
            let mut neighbor_colors: Vec<u64> = graph
                .graph
                .neighbors(*node_idx)
                .filter_map(|neighbor| colors.get(&neighbor).copied())
                .collect();
            neighbor_colors.sort_unstable();

            let mut hasher = DefaultHasher::new();
            color.hash(&mut hasher);
            neighbor_colors.hash(&mut hasher);
            next.insert(*node_idx, hasher.finish());
        }
        colors = next;
    }

    let mut histogram = BTreeMap::new();
    for color in colors.values() {
        *histogram.entry(*color).or_insert(0) += 1;
    }
    WlSignature { histogram }
}

fn initial_color(attrs: &NodeAttributes, degree: usize) -> u64 {
    let mut hasher = DefaultHasher::new();
    if let Some(label) = &attrs.label {
        label.hash(&mut hasher);
    }
    if let Some(weight) = attrs.weight {
        ((weight * 1e6).round() as i64).hash(&mut hasher);
    }
    degree.hash(&mut hasher);
    attrs.extra.len().hash(&mut hasher);
    hasher.finish()
}

fn generate_precise_candidates(
    graph: &GraphInstance,
    pattern_size: usize,
) -> Result<Vec<GraphInstance>> {
    if pattern_size == 0 {
        return Ok(Vec::new());
    }
    if graph.graph.node_count() < pattern_size {
        return Ok(Vec::new());
    }
    if graph.graph.node_count() == pattern_size {
        return Ok(vec![graph.clone()]);
    }

    let mut results = Vec::new();
    let mut seen = HashSet::new();

    let anchor_nodes = anchor_seed_nodes(graph, MAX_ANCHOR_SHRINK_ATTEMPTS);
    for anchor in anchor_nodes {
        if results.len() >= MAX_PRECISE_PER_CANDIDATE {
            break;
        }
        let nodes = bfs_collect(graph, anchor, pattern_size);
        if nodes.len() != pattern_size {
            continue;
        }
        let node_ids = node_indices_to_ids(graph, &nodes);
        insert_subgraph(graph, node_ids, &mut seen, &mut results)?;
    }

    for start in graph.graph.node_indices() {
        if results.len() >= MAX_PRECISE_PER_CANDIDATE {
            break;
        }
        let visited = bfs_collect(graph, start, pattern_size * 2);
        let indices: Vec<NodeIndex> = visited.into_iter().collect();
        if indices.len() < pattern_size {
            continue;
        }
        let max_start = indices.len().saturating_sub(pattern_size);
        for offset in 0..=max_start {
            if results.len() >= MAX_PRECISE_PER_CANDIDATE {
                break;
            }
            let mut node_ids = IndexSet::new();
            for idx in &indices[offset..offset + pattern_size] {
                if let Some(id) = graph.reverse_lookup.get(idx) {
                    node_ids.insert(id.clone());
                }
            }
            if node_ids.len() != pattern_size {
                continue;
            }
            insert_subgraph(graph, node_ids, &mut seen, &mut results)?;
        }
    }

    Ok(results)
}

fn anchor_seed_nodes(graph: &GraphInstance, attempts: usize) -> Vec<NodeIndex> {
    let mut nodes: Vec<NodeIndex> = graph.graph.node_indices().collect();
    nodes.sort_by_key(|&idx| Reverse(graph.graph.neighbors(idx).count()));
    nodes.truncate(attempts);
    nodes
}

fn node_indices_to_ids(graph: &GraphInstance, indices: &IndexSet<NodeIndex>) -> IndexSet<String> {
    let mut ids = IndexSet::new();
    for idx in indices.iter() {
        if let Some(id) = graph.reverse_lookup.get(idx) {
            ids.insert(id.clone());
        }
    }
    ids
}

fn insert_subgraph(
    graph: &GraphInstance,
    node_ids: IndexSet<String>,
    seen: &mut HashSet<String>,
    results: &mut Vec<GraphInstance>,
) -> Result<()> {
    let mut signature_parts: Vec<&str> = node_ids.iter().map(String::as_str).collect();
    signature_parts.sort_unstable();
    let signature = signature_parts.join("|");
    if !seen.insert(signature) {
        return Ok(());
    }
    if results.len() >= MAX_PRECISE_PER_CANDIDATE {
        return Ok(());
    }
    let subgraph = GraphLoader::induced_subgraph(graph, &node_ids)?;
    results.push(subgraph);
    Ok(())
}

fn bfs_collect(graph: &GraphInstance, start: NodeIndex, limit: usize) -> IndexSet<NodeIndex> {
    let mut visited = IndexSet::new();
    let mut queue = VecDeque::new();
    visited.insert(start);
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        if visited.len() >= limit {
            break;
        }
        for neighbor in graph.graph.neighbors(node) {
            if visited.insert(neighbor) {
                queue.push_back(neighbor);
                if visited.len() >= limit {
                    break;
                }
            }
        }
    }

    let mut trimmed = IndexSet::new();
    for node in visited.into_iter().take(limit) {
        trimmed.insert(node);
    }
    trimmed
}

fn evaluate_precise_candidates(
    pattern_profile: &SpectralProfile,
    pattern_signature: &WlSignature,
    candidates: Vec<GraphInstance>,
    epsilon: f64,
) -> Result<(Vec<GraphInstance>, PreciseFilterStats)> {
    let mut seen = HashSet::new();
    let mut refined = Vec::new();
    let mut stats = PreciseFilterStats::default();
    let pattern_size: usize = pattern_signature.histogram.values().copied().sum();

    for candidate in candidates {
        stats.total += 1;
        let mut ids: Vec<String> = candidate.reverse_lookup.values().cloned().collect();
        ids.sort();
        let signature = ids.join("|");
        if !seen.insert(signature) {
            stats.dedup += 1;
            continue;
        }

        let candidate_size = candidate.graph.node_count();
        if candidate_size < pattern_size {
            stats.too_small += 1;
            debug!(
                "Skipping precise candidate |V|={} < pattern |V|={}",
                candidate_size, pattern_size
            );
            continue;
        }

        let candidate_signature = compute_wl_signature(&candidate, WL_ITERATIONS);

        if candidate_size == pattern_size {
            if !candidate_signature.equivalent(pattern_signature) {
                stats.wl_mismatch += 1;
                if stats.sample_wl_mismatch.is_none() {
                    stats.sample_wl_mismatch = Some(format!(
                        "pattern {:?} | candidate {:?}",
                        pattern_signature.histogram,
                        candidate_signature.histogram
                    ));
                }
                if stats.wl_mismatch <= 3 {
                    debug!("Pattern WL histogram: {:?}", pattern_signature.histogram);
                    debug!("Candidate WL histogram: {:?}", candidate_signature.histogram);
                }
                debug!(
                    "Precise WL mismatch for |V|={} candidate; rejecting",
                    candidate_size
                );
                continue;
            }
        } else {
            let (missing_total, deficit_slots) =
                candidate_signature.deficit_stats(pattern_signature);
            if missing_total > 0 {
                stats.wl_deficit += 1;
                debug!(
                    "Larger candidate |V|={} missing {} nodes across {} WL slots; rejecting",
                    candidate_size,
                    missing_total,
                    deficit_slots
                );
                continue;
            }
        }

        let laplacian = NormalizedLaplacianBuilder::build(&candidate)?;
        let candidate_profile = SpectralVectorExtractor::compute_profile(&laplacian)?;

        let spectral_ok = if candidate_size == pattern_size {
            spectral_equivalent(&candidate_profile, pattern_profile, epsilon)
        } else {
            spectral_dominates(&candidate_profile, pattern_profile, epsilon)
                && spectral_distance(&candidate_profile, pattern_profile)
                    .map(|distance| {
                        let bound =
                            (pattern_profile.norm_l2 * SPECTRAL_DISTANCE_RATIO).max(epsilon);
                        distance <= bound
                    })
                    .unwrap_or(true)
        };

        if !spectral_ok {
            stats.spectral_fail += 1;
            debug!("Spectral check failed for precise candidate |V|={}", candidate_size);
            continue;
        }

        stats.accepted += 1;
        refined.push(candidate);
    }

    debug!(
        "Precise candidates retained: {} of {} (dedup {}, too-small {}, WL mismatch {}, WL deficit {}, spectral {})",
        stats.accepted,
        stats.total,
        stats.dedup,
        stats.too_small,
        stats.wl_mismatch,
        stats.wl_deficit,
        stats.spectral_fail
    );
    Ok((refined, stats))
}

fn spectral_equivalent(candidate: &SpectralProfile, pattern: &SpectralProfile, epsilon: f64) -> bool {
    let len = pattern.eigenvalues.len();
    if candidate.eigenvalues.len() < len {
        return false;
    }

    let distance = spectral_distance(candidate, pattern).unwrap_or(f64::INFINITY);
    let tolerance = epsilon * (len.max(1) as f64).sqrt() + SPECTRAL_EQUIVALENCE_ABS;
    if distance > tolerance {
        return false;
    }

    let norm_tolerance = epsilon * (1.0 + pattern.norm_l2.abs()) + SPECTRAL_EQUIVALENCE_ABS;
    if (candidate.norm_l2 - pattern.norm_l2).abs() > norm_tolerance {
        return false;
    }

    let trace_tolerance = epsilon * (1.0 + pattern.trace.abs()) + SPECTRAL_EQUIVALENCE_ABS;
    if (candidate.trace - pattern.trace).abs() > trace_tolerance {
        return false;
    }

    true
}

fn run_vf2(pattern: &GraphInstance, candidate: &GraphInstance, epsilon: f64) -> bool {
    if pattern.graph.node_count() == candidate.graph.node_count() {
        return is_isomorphic_matching(
            &pattern.graph,
            &candidate.graph,
            |a: &NodeAttributes, b: &NodeAttributes| node_compatible(a, b, epsilon),
            |a: &EdgeAttributes, b: &EdgeAttributes| edge_compatible(a, b, epsilon),
        );
    }

    is_isomorphic_subgraph_matching(
        &pattern.graph,
        &candidate.graph,
        |a: &NodeAttributes, b: &NodeAttributes| node_compatible(a, b, epsilon),
        |a: &EdgeAttributes, b: &EdgeAttributes| edge_compatible(a, b, epsilon),
    )
}
fn node_compatible(a: &NodeAttributes, b: &NodeAttributes, epsilon: f64) -> bool {
    match (&a.label, &b.label) {
        (Some(lhs), Some(rhs)) if lhs != rhs => return false,
        (Some(_), None) | (None, Some(_)) => return false,
        _ => {}
    }

    match (a.weight, b.weight) {
        (Some(lhs), Some(rhs)) if (lhs - rhs).abs() > epsilon => return false,
        (Some(_), None) | (None, Some(_)) => return false,
        _ => {}
    }

    true
}

fn edge_compatible(a: &EdgeAttributes, b: &EdgeAttributes, epsilon: f64) -> bool {
    match (a.weight, b.weight) {
        (Some(lhs), Some(rhs)) => (lhs - rhs).abs() <= epsilon,
        (Some(_), None) | (None, Some(_)) => false,
        _ => true,
    }
}
