use std::collections::HashMap;

use anyhow::Result;
use nalgebra::DMatrix;
use petgraph::visit::EdgeRef;

use crate::graph::model::GraphInstance;

/// Responsible for constructing the normalized Laplacian matrix \( \hat{L} = D^{-1/2} L D^{-1/2} \).
pub struct NormalizedLaplacianBuilder;

impl NormalizedLaplacianBuilder {
    pub fn build(graph: &GraphInstance) -> Result<DMatrix<f64>> {
        let node_count = graph.graph.node_count();
        if node_count == 0 {
            return Ok(DMatrix::zeros(0, 0));
        }

        let mut index_map = HashMap::new();
        for (row, node_idx) in graph.graph.node_indices().enumerate() {
            index_map.insert(node_idx, row);
        }

        let mut adjacency = DMatrix::zeros(node_count, node_count);

        for edge in graph.graph.edge_references() {
            let source = index_map[&edge.source()];
            let target = index_map[&edge.target()];
            let weight = edge.weight().weight.unwrap_or(1.0).max(0.0); // ensure non-negative

            adjacency[(source, target)] += weight;
            if !graph.directed {
                adjacency[(target, source)] += weight;
            }
        }

        // Degree matrix
        let mut degrees = vec![0.0f64; node_count];
        for i in 0..node_count {
            degrees[i] = adjacency.row(i).iter().copied().sum::<f64>();
        }

        // Laplacian L = D - A
        let mut laplacian = DMatrix::zeros(node_count, node_count);
        for i in 0..node_count {
            laplacian[(i, i)] = degrees[i];
        }
        laplacian -= adjacency;

        // Normalized Laplacian \hat{L} = D^{-1/2} L D^{-1/2}
        let mut normalized = DMatrix::zeros(node_count, node_count);
        for i in 0..node_count {
            let di = degrees[i];
            for j in 0..node_count {
                let dj = degrees[j];
                if di == 0.0 || dj == 0.0 {
                    continue;
                }
                let scale = 1.0f64 / (di.sqrt() * dj.sqrt());
                normalized[(i, j)] = laplacian[(i, j)] * scale;
            }
        }

        for i in 0..node_count {
            if degrees[i] > 0.0 {
                normalized[(i, i)] = 1.0;
            }
        }

        Ok(normalized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::model::{EdgeAttributes, GraphInstance, LabeledGraph, NodeAttributes};
    use indexmap::IndexMap;
    use proptest::prelude::*;

    fn line_graph() -> GraphInstance {
        let mut graph = LabeledGraph::with_capacity(3, 4);
        let mut node_lookup = IndexMap::new();
        let mut reverse_lookup = IndexMap::new();
        for id in ["a", "b", "c"] {
            let attrs = NodeAttributes {
                label: Some(id.to_string()),
                weight: None,
                extra: IndexMap::new(),
            };
            let node_idx = graph.add_node(attrs);
            node_lookup.insert(id.to_string(), node_idx);
            reverse_lookup.insert(node_idx, id.to_string());
        }
        let mut add_edge = |u: &str, v: &str| {
            let edge = EdgeAttributes {
                weight: Some(1.0),
                extra: IndexMap::new(),
            };
            let u_idx = node_lookup[u];
            let v_idx = node_lookup[v];
            graph.add_edge(u_idx, v_idx, edge.clone());
            graph.add_edge(v_idx, u_idx, edge);
        };
        add_edge("a", "b");
        add_edge("b", "c");
        GraphInstance {
            graph,
            node_lookup,
            reverse_lookup,
            graph_attributes: IndexMap::new(),
            directed: false,
        }
    }

    #[test]
    fn normalized_laplacian_is_symmetric_with_unit_diagonal() {
        let graph = line_graph();
        let lap = NormalizedLaplacianBuilder::build(&graph).expect("laplacian");
        for row in 0..lap.nrows() {
            assert!((lap[(row, row)] - 1.0).abs() < 1e-9 || lap[(row, row)].abs() < 1e-9);
            for col in 0..lap.ncols() {
                let diff = (lap[(row, col)] - lap[(col, row)]).abs();
                assert!(diff < 1e-9, "matrix must be symmetric");
            }
        }
    }

    fn random_graph_strategy() -> impl Strategy<Value = GraphInstance> {
        (2_usize..=5).prop_flat_map(|size| {
            let edge_count = size * (size - 1) / 2;
            prop::collection::vec(prop::bool::ANY, edge_count)
                .prop_map(move |edges| build_graph(size, &edges))
        })
    }

    fn build_graph(size: usize, edges: &[bool]) -> GraphInstance {
        let mut graph = LabeledGraph::with_capacity(size, size * size);
        let mut node_lookup = IndexMap::new();
        let mut reverse_lookup = IndexMap::new();
        for idx in 0..size {
            let attrs = NodeAttributes {
                label: Some(format!("v{idx}")),
                weight: None,
                extra: IndexMap::new(),
            };
            let node_idx = graph.add_node(attrs);
            let id = format!("v{idx}");
            node_lookup.insert(id.clone(), node_idx);
            reverse_lookup.insert(node_idx, id);
        }
        let mut edge_iter = edges.iter();
        for i in 0..size {
            for j in (i + 1)..size {
                if *edge_iter.next().unwrap_or(&false) {
                    let edge = EdgeAttributes {
                        weight: Some(1.0),
                        extra: IndexMap::new(),
                    };
                    let u = node_lookup[&format!("v{i}")];
                    let v = node_lookup[&format!("v{j}")];
                    graph.add_edge(u, v, edge.clone());
                    graph.add_edge(v, u, edge);
                }
            }
        }
        GraphInstance {
            graph,
            node_lookup,
            reverse_lookup,
            graph_attributes: IndexMap::new(),
            directed: false,
        }
    }

    proptest! {
        #[test]
        fn normalized_laplacian_diagonal_bounded(graph in random_graph_strategy()) {
            let lap = NormalizedLaplacianBuilder::build(&graph).expect("laplacian");
            for row in 0..lap.nrows() {
                prop_assert!(lap[(row, row)] <= 1.0 + 1e-9);
                for col in 0..lap.ncols() {
                    let diff = (lap[(row, col)] - lap[(col, row)]).abs();
                    prop_assert!(diff < 1e-9);
                }
            }
        }
    }
}
