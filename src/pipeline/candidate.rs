use std::collections::VecDeque;

use anyhow::{anyhow, Result};
use indexmap::IndexSet;
use petgraph::prelude::NodeIndex;
use rayon::prelude::*;

use crate::graph::{GraphId, GraphInstance, GraphLoader};

pub struct AnchorSelector;

impl AnchorSelector {
    pub fn select(pattern: &GraphInstance, limit: Option<usize>) -> Vec<GraphId> {
        if pattern.node_count() == 0 {
            return Vec::new();
        }
        let count = limit.unwrap_or_else(|| pattern.node_count().min(3));
        let mut ranked: Vec<(usize, GraphId)> = pattern
            .graph
            .node_indices()
            .filter_map(|idx| {
                let degree = pattern.graph.neighbors(idx).count();
                let attrs = pattern.graph.node_weight(idx)?;
                let label_bonus = attrs.label.as_ref().map(|_| 1).unwrap_or_default();
                let weight_bonus = attrs.weight.map(|w| (w * 10.0) as i64).unwrap_or_default();
                let score = degree * 10 + label_bonus + weight_bonus as usize;
                let id = pattern.reverse_lookup.get(&idx)?.clone();
                Some((score, id))
            })
            .collect();
        ranked.sort_by(|a, b| b.0.cmp(&a.0));
        ranked
            .into_iter()
            .take(count.max(1))
            .map(|(_, id)| id)
            .collect()
    }
}

pub struct CandidateGenerator;

impl CandidateGenerator {
    pub fn filter_candidates(
        pattern: &GraphInstance,
        target: &GraphInstance,
        anchors: &[GraphId],
        epsilon: f64,
    ) -> Result<IndexSet<GraphId>> {
        if anchors.is_empty() {
            return Ok(IndexSet::new());
        }

        let descriptors: Vec<_> = anchors
            .iter()
            .map(|anchor_id| -> Result<_> {
                let index = pattern
                    .node_lookup
                    .get(anchor_id)
                    .ok_or_else(|| anyhow!("Unknown anchor '{anchor_id}' in pattern"))?;
                let attrs = pattern
                    .graph
                    .node_weight(*index)
                    .ok_or_else(|| anyhow!("Missing attributes for anchor '{anchor_id}'"))?;
                Ok((
                    attrs.label.clone(),
                    attrs.weight,
                    pattern.graph.neighbors(*index).count(),
                ))
            })
            .collect::<Result<_>>()?;

        let candidate_sets: Vec<_> = descriptors
            .par_iter()
            .map(|(label, weight, degree)| {
                target
                    .graph
                    .node_indices()
                    .filter_map(|idx| {
                        let attrs = target.graph.node_weight(idx)?;
                        if !label_matches(label.as_ref(), attrs.label.as_ref()) {
                            return None;
                        }
                        if !weight_matches(*weight, attrs.weight, epsilon) {
                            return None;
                        }
                        if target.graph.neighbors(idx).count() < *degree {
                            return None;
                        }
                        target.reverse_lookup.get(&idx).cloned()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let mut merged = IndexSet::new();
        for set in candidate_sets {
            for id in set {
                merged.insert(id);
            }
        }

        Ok(merged)
    }
}

pub struct SubgraphExtractor;

impl SubgraphExtractor {
    pub fn extract(
        target: &GraphInstance,
        candidates: &IndexSet<GraphId>,
        pattern_size: usize,
    ) -> Result<Vec<GraphInstance>> {
        if pattern_size == 0 {
            return Ok(Vec::new());
        }

        let radius = (pattern_size as f64).log2().ceil() as usize;
        let max_nodes = pattern_size.saturating_mul(2).max(pattern_size);
        let candidate_ids: Vec<GraphId> = candidates.iter().cloned().collect();

        let subgraphs: Vec<_> = candidate_ids
            .par_iter()
            .filter_map(|id| {
                let idx = target.node_lookup.get(id)?;
                let collected = bfs_collect(target, *idx, max_nodes, radius.max(1));
                if collected.len() < pattern_size {
                    return None;
                }
                let ids: Vec<GraphId> = collected
                    .into_iter()
                    .filter_map(|node| target.reverse_lookup.get(&node).cloned())
                    .take(pattern_size)
                    .collect();
                if ids.len() < pattern_size {
                    return None;
                }
                Some(GraphLoader::induced_subgraph(target, &ids))
            })
            .collect();

        let mut results = Vec::new();
        for graph in subgraphs {
            results.push(graph?);
        }
        Ok(results)
    }
}

fn bfs_collect(
    graph: &GraphInstance,
    start: NodeIndex,
    max_nodes: usize,
    radius: usize,
) -> IndexSet<NodeIndex> {
    let mut visited = IndexSet::new();
    let mut queue = VecDeque::new();
    visited.insert(start);
    queue.push_back((start, 0usize));

    while let Some((node, depth)) = queue.pop_front() {
        if visited.len() >= max_nodes {
            break;
        }
        if depth >= radius {
            continue;
        }
        for neighbor in graph.graph.neighbors(node) {
            if visited.insert(neighbor) {
                queue.push_back((neighbor, depth + 1));
                if visited.len() >= max_nodes {
                    break;
                }
            }
        }
    }

    visited
}

fn label_matches(left: Option<&String>, right: Option<&String>) -> bool {
    match (left, right) {
        (Some(lhs), Some(rhs)) => lhs == rhs,
        (Some(_), None) => false,
        _ => true,
    }
}

fn weight_matches(left: Option<f64>, right: Option<f64>, epsilon: f64) -> bool {
    match (left, right) {
        (Some(lhs), Some(rhs)) => (lhs - rhs).abs() <= epsilon,
        (Some(_), None) => false,
        _ => true,
    }
}
