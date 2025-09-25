use std::cmp::Ordering;
use std::collections::VecDeque;

use anyhow::{Result, anyhow};
use indexmap::IndexSet;
use log::debug;
use petgraph::visit::IntoNodeReferences;
use rayon::prelude::*;

use crate::graph::construction::GraphLoader;
use crate::graph::model::{GraphId, GraphInstance};

/// Select anchor nodes within the pattern graph using degree and attribute heuristics.
pub struct AnchorSelector;

impl AnchorSelector {
    pub fn select_anchors(pattern: &GraphInstance, k: Option<usize>) -> IndexSet<GraphId> {
        let node_count = pattern.graph.node_count();
        if node_count == 0 {
            return IndexSet::new();
        }
        let target_k = k.unwrap_or_else(|| node_count.min(3).max(1));

        let mut ranked: Vec<(f64, GraphId)> = pattern
            .graph
            .node_references()
            .filter_map(|(idx, attrs)| {
                let id = pattern.reverse_lookup.get(&idx)?.clone();
                let degree = pattern.graph.neighbors(idx).count() as f64;
                let label_bonus = if attrs.label.is_some() { 0.1 } else { 0.0 };
                let weight_bonus = attrs.weight.unwrap_or(0.0);
                Some((degree + label_bonus + weight_bonus, id))
            })
            .collect();

        ranked.sort_by(|(score_a, _), (score_b, _)| {
            score_b.partial_cmp(score_a).unwrap_or(Ordering::Equal)
        });

        ranked
            .into_iter()
            .take(target_k.min(node_count))
            .map(|(_, id)| id)
            .collect()
    }
}

#[derive(Clone)]
struct AnchorDescriptor {
    label: Option<String>,
    weight: Option<f64>,
    degree: usize,
}

/// Generate candidate nodes in the target graph for each anchor.
pub struct CandidateGenerator;

impl CandidateGenerator {
    pub fn filter_candidates(
        pattern: &GraphInstance,
        target: &GraphInstance,
        anchors: &IndexSet<GraphId>,
        epsilon: f64,
    ) -> Result<IndexSet<GraphId>> {
        let mut descriptors = Vec::new();
        for anchor_id in anchors {
            let anchor_idx = pattern
                .node_lookup
                .get(anchor_id)
                .ok_or_else(|| anyhow!("Anchor '{}' not present in pattern graph", anchor_id))?;
            let anchor_attrs = pattern
                .graph
                .node_weight(*anchor_idx)
                .ok_or_else(|| anyhow!("Missing node weight for anchor '{}'", anchor_id))?;
            let descriptor = AnchorDescriptor {
                label: anchor_attrs.label.clone(),
                weight: anchor_attrs.weight,
                degree: pattern.graph.neighbors(*anchor_idx).count(),
            };
            descriptors.push(descriptor);
        }

        let candidate_lists: Vec<Vec<GraphId>> = descriptors
            .par_iter()
            .map(|descriptor| {
                target
                    .graph
                    .node_references()
                    .filter_map(|(target_idx, target_attrs)| {
                        if !label_matches(
                            descriptor.label.as_deref(),
                            target_attrs.label.as_deref(),
                        ) {
                            return None;
                        }
                        if !weight_matches(descriptor.weight, target_attrs.weight, epsilon) {
                            return None;
                        }
                        let target_degree = target.graph.neighbors(target_idx).count();
                        if target_degree < descriptor.degree {
                            return None;
                        }
                        target.reverse_lookup.get(&target_idx).cloned()
                    })
                    .collect()
            })
            .collect();

        let mut candidates = IndexSet::new();
        for list in candidate_lists {
            for id in list {
                candidates.insert(id);
            }
        }

        debug!("Candidates after filtering: {}", candidates.len());
        Ok(candidates)
    }
}

/// Produce induced subgraphs around candidate nodes with dynamic radii.
pub struct SubgraphExtractor;

impl SubgraphExtractor {
    pub fn extract_subgraphs(
        target: &GraphInstance,
        candidates: &IndexSet<GraphId>,
        pattern_size: usize,
    ) -> Result<Vec<GraphInstance>> {
        if pattern_size == 0 {
            return Ok(Vec::new());
        }
        let radius = ((pattern_size as f64).log2().ceil() as usize).max(1);

        let results: Vec<Result<Option<GraphInstance>>> = candidates
            .par_iter()
            .map(|candidate_id| -> Result<Option<GraphInstance>> {
                let start_idx = target.node_lookup.get(candidate_id).ok_or_else(|| {
                    anyhow!("Candidate '{}' not found in target graph", candidate_id)
                })?;

                let mut visited = IndexSet::new();
                let mut queue = VecDeque::new();
                queue.push_back((*start_idx, 0usize));
                visited.insert(*start_idx);

                while let Some((node_idx, depth)) = queue.pop_front() {
                    if depth >= radius {
                        continue;
                    }
                    for neighbor in target.graph.neighbors(node_idx) {
                        if visited.insert(neighbor) {
                            queue.push_back((neighbor, depth + 1));
                        }
                    }
                    if visited.len() >= pattern_size * 2 {
                        break;
                    }
                }

                let mut node_ids = IndexSet::new();
                for idx in visited.iter() {
                    if let Some(id) = target.reverse_lookup.get(idx) {
                        node_ids.insert(id.clone());
                    }
                }

                if node_ids.len() >= pattern_size {
                    let subgraph = GraphLoader::induced_subgraph(target, &node_ids)?;
                    Ok(Some(subgraph))
                } else {
                    Ok(None)
                }
            })
            .collect();

        let mut subgraphs = Vec::new();
        for res in results {
            if let Some(graph) = res? {
                subgraphs.push(graph);
            }
        }
        debug!("Subgraphs extracted: {}", subgraphs.len());
        Ok(subgraphs)
    }
}

fn label_matches(left: Option<&str>, right: Option<&str>) -> bool {
    match (left, right) {
        (Some(l), Some(r)) => l == r,
        (Some(_), None) => false,
        _ => true,
    }
}

fn weight_matches(left: Option<f64>, right: Option<f64>, epsilon: f64) -> bool {
    match (left, right) {
        (Some(l), Some(r)) => (l - r).abs() <= epsilon,
        (Some(_), None) => false,
        _ => true,
    }
}
