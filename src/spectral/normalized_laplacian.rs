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
