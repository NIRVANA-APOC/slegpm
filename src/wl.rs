use std::collections::HashMap;

use indexmap::{map::Entry, IndexMap};
use petgraph::visit::NodeIndexable;

use crate::graph::GraphInstance;

pub fn weisfeiler_lehman_isomorphic(
    pattern: &GraphInstance,
    candidate: &GraphInstance,
    iterations: usize,
) -> bool {
    if pattern.node_count() != candidate.node_count() {
        return false;
    }
    if pattern.node_count() == 0 {
        return true;
    }

    let mut pattern_colors = initial_colors(pattern);
    let mut candidate_colors = initial_colors(candidate);

    for _ in 0..iterations.max(1) {
        pattern_colors = refine_colors(pattern, &pattern_colors);
        candidate_colors = refine_colors(candidate, &candidate_colors);
        if !multiset_equivalent(&pattern_colors, &candidate_colors) {
            return false;
        }
    }

    true
}

fn initial_colors(graph: &GraphInstance) -> Vec<u64> {
    let mut palette: IndexMap<(Option<String>, Option<i64>, usize), u64> = IndexMap::new();
    let mut colors = Vec::with_capacity(graph.node_count());

    for node in graph.graph.node_indices() {
        let attrs = graph.graph.node_weight(node).expect("node present");
        let label = attrs.label.clone();
        let weight = attrs.weight.map(|w| (w * 1e6) as i64);
        let degree = graph.graph.neighbors(node).count();
        let key = (label, weight, degree);
        let next_value = palette.len() as u64 + 1;
        let color = match palette.entry(key) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => *entry.insert(next_value),
        };
        colors.push(color);
    }

    colors
}

fn refine_colors(graph: &GraphInstance, colors: &[u64]) -> Vec<u64> {
    let mut palette: IndexMap<Vec<u64>, u64> = IndexMap::new();
    let mut next_colors = Vec::with_capacity(colors.len());

    for node in graph.graph.node_indices() {
        let idx = graph.graph.to_index(node);
        let mut signature = Vec::new();
        signature.push(colors[idx]);
        let mut neighbor_colors: Vec<u64> = graph
            .graph
            .neighbors(node)
            .map(|neighbor| colors[graph.graph.to_index(neighbor)])
            .collect();
        neighbor_colors.sort_unstable();
        signature.extend(neighbor_colors);
        let next_value = palette.len() as u64 + 1;
        let color = match palette.entry(signature) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => *entry.insert(next_value),
        };
        next_colors.push(color);
    }

    next_colors
}

fn multiset_equivalent(left: &[u64], right: &[u64]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut freq_left = HashMap::new();
    for value in left {
        *freq_left.entry(*value).or_insert(0usize) += 1;
    }
    let mut freq_right = HashMap::new();
    for value in right {
        *freq_right.entry(*value).or_insert(0usize) += 1;
    }
    freq_left == freq_right
}
