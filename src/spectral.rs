use std::cmp::Ordering;

use anyhow::{anyhow, Result};
use indexmap::IndexMap;
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use petgraph::visit::EdgeRef;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::graph::{EdgeAttributes, GraphInstance};

const DENSE_THRESHOLD: usize = 2048;
const POWER_ITERATIONS: usize = 128;
const POWER_TOLERANCE: f64 = 1e-6;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LaplacianMatrix {
    Dense(DenseLaplacian),
    Sparse(SparseLaplacian),
}

impl LaplacianMatrix {
    pub fn node_count(&self) -> usize {
        match self {
            LaplacianMatrix::Dense(dense) => dense.size,
            LaplacianMatrix::Sparse(sparse) => sparse.size,
        }
    }

    pub fn as_dense_matrix(&self) -> Option<DMatrix<f64>> {
        match self {
            LaplacianMatrix::Dense(dense) => Some(dense.to_matrix()),
            LaplacianMatrix::Sparse(_) => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseLaplacian {
    pub size: usize,
    pub data: Vec<f64>,
}

impl DenseLaplacian {
    fn to_matrix(&self) -> DMatrix<f64> {
        DMatrix::from_row_slice(self.size, self.size, &self.data)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseLaplacian {
    pub size: usize,
    pub normalized_rows: Vec<Vec<(usize, f64)>>,
    pub zero_degree: usize,
}

impl SparseLaplacian {
    fn multiply_normalized(&self, vector: &[f64]) -> Vec<f64> {
        self.normalized_rows
            .par_iter()
            .map(|row| row.iter().map(|(j, weight)| weight * vector[*j]).sum())
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralProfile {
    pub eigenvalues: Vec<f64>,
    pub norm_l2: f64,
    pub trace: f64,
}

impl SpectralProfile {
    pub fn from_eigenvalues(mut eigenvalues: Vec<f64>) -> Self {
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let norm_l2 = eigenvalues.iter().map(|v| v * v).sum::<f64>().sqrt();
        let trace = eigenvalues.iter().sum();
        Self {
            eigenvalues,
            norm_l2,
            trace,
        }
    }
}

pub fn normalized_laplacian(graph: &GraphInstance) -> Result<LaplacianMatrix> {
    let n = graph.node_count();
    if n == 0 {
        return Ok(LaplacianMatrix::Dense(DenseLaplacian {
            size: 0,
            data: Vec::new(),
        }));
    }

    let mut index_map: IndexMap<_, _> = IndexMap::new();
    for (row, node) in graph.graph.node_indices().enumerate() {
        index_map.insert(node, row);
    }

    let mut adjacency: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for edge in graph.graph.edge_references() {
        let Some(&row) = index_map.get(&edge.source()) else {
            continue;
        };
        let Some(&col) = index_map.get(&edge.target()) else {
            continue;
        };
        if !graph.directed && row > col {
            continue;
        }
        let weight = edge.weight().weight.unwrap_or(1.0).max(0.0);
        adjacency[row].push((col, weight));
        if !graph.directed && row != col {
            adjacency[col].push((row, weight));
        }
    }

    let degrees: Vec<f64> = adjacency
        .par_iter()
        .map(|edges| edges.iter().map(|(_, w)| *w).sum::<f64>())
        .collect();
    let inv_sqrt: Vec<f64> = degrees
        .iter()
        .map(|d| if *d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();
    let zero_degree = degrees.iter().filter(|d| **d == 0.0).count();

    if n <= DENSE_THRESHOLD {
        let mut normalized = vec![0.0; n * n];
        normalized
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, row)| {
                let scale_i = inv_sqrt[i];
                let mut loop_adjust = 0.0f64;
                for &(j, weight) in &adjacency[i] {
                    let scale_j = inv_sqrt[j];
                    if i == j {
                        loop_adjust += weight * scale_i * scale_j;
                        continue;
                    }
                    if scale_i == 0.0 || scale_j == 0.0 {
                        continue;
                    }
                    row[j] = -weight * scale_i * scale_j;
                }
                row[i] = if scale_i == 0.0 {
                    0.0
                } else {
                    1.0 - loop_adjust
                };
            });
        return Ok(LaplacianMatrix::Dense(DenseLaplacian {
            size: n,
            data: normalized,
        }));
    }

    let normalized_rows: Vec<Vec<(usize, f64)>> = adjacency
        .into_par_iter()
        .enumerate()
        .map(|(i, edges)| {
            let scale_i = inv_sqrt[i];
            edges
                .into_iter()
                .filter_map(|(j, weight)| {
                    let scale_j = inv_sqrt[j];
                    if scale_i == 0.0 || scale_j == 0.0 {
                        None
                    } else {
                        Some((j, weight * scale_i * scale_j))
                    }
                })
                .collect()
        })
        .collect();

    Ok(LaplacianMatrix::Sparse(SparseLaplacian {
        size: n,
        normalized_rows,
        zero_degree,
    }))
}

pub fn spectral_profile(matrix: &LaplacianMatrix, max_dim: usize) -> Result<SpectralProfile> {
    match matrix {
        LaplacianMatrix::Dense(dense) => {
            let dense_matrix = dense.to_matrix();
            if dense_matrix.nrows() != dense_matrix.ncols() {
                return Err(anyhow!("Spectral profile expects square matrix"));
            }
            if dense_matrix.is_empty() {
                return Ok(SpectralProfile::from_eigenvalues(Vec::new()));
            }
            let eigen = SymmetricEigen::new(dense_matrix);
            let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
            eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let dim = max_dim.min(eigenvalues.len()).max(1);
            eigenvalues.truncate(dim);
            Ok(SpectralProfile::from_eigenvalues(eigenvalues))
        }
        LaplacianMatrix::Sparse(sparse) => {
            let mut eigenvalues = approximate_sparse_eigenvalues(sparse, max_dim.max(1));
            eigenvalues.truncate(max_dim.max(1));
            Ok(SpectralProfile::from_eigenvalues(eigenvalues))
        }
    }
}

pub fn spectral_distance(a: &SpectralProfile, b: &SpectralProfile) -> f64 {
    let len = a.eigenvalues.len().min(b.eigenvalues.len());
    let mut distance = 0.0;
    for i in 0..len {
        let diff = a.eigenvalues[i] - b.eigenvalues[i];
        distance += diff * diff;
    }
    (distance / len.max(1) as f64).sqrt()
}

pub fn rayleigh_quotient(matrix: &LaplacianMatrix, vector: &DVector<f64>) -> Result<f64> {
    match matrix {
        LaplacianMatrix::Dense(dense) => {
            let mat = dense.to_matrix();
            if vector.len() != mat.nrows() {
                return Err(anyhow!("Vector length must match matrix order"));
            }
            let numerator = vector.transpose() * &mat * vector;
            let denominator = vector.transpose() * vector;
            Ok((numerator[(0, 0)] / denominator[(0, 0)]).abs())
        }
        LaplacianMatrix::Sparse(sparse) => {
            if vector.len() != sparse.size {
                return Err(anyhow!("Vector length must match matrix order"));
            }
            let normalized = sparse.multiply_normalized(vector.as_slice());
            let vtv = vector.dot(vector);
            let vtn = vector
                .iter()
                .zip(normalized.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
            if vtv.abs() <= f64::EPSILON {
                return Err(anyhow!("Vector norm is zero"));
            }
            Ok(((vtv - vtn) / vtv).abs())
        }
    }
}

pub fn edge_weight(edge: &EdgeAttributes) -> f64 {
    edge.weight.unwrap_or(1.0)
}

fn approximate_sparse_eigenvalues(sparse: &SparseLaplacian, max_dim: usize) -> Vec<f64> {
    if sparse.size == 0 {
        return Vec::new();
    }
    let mut eigenvalues = Vec::with_capacity(max_dim);
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(max_dim);
    let mut seed = 1u64;

    for _ in 0..max_dim {
        seed = seed.wrapping_mul(48271).wrapping_add(1);
        if let Some((lambda, vector)) = power_iteration(sparse, &basis, seed) {
            eigenvalues.push((1.0 - lambda).clamp(0.0, 2.0));
            basis.push(vector);
        } else {
            break;
        }
    }

    while eigenvalues.len() < sparse.zero_degree {
        eigenvalues.push(0.0);
    }

    if eigenvalues.is_empty() {
        eigenvalues.push(0.0);
    }

    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    eigenvalues
}

fn power_iteration(
    sparse: &SparseLaplacian,
    basis: &[Vec<f64>],
    seed: u64,
) -> Option<(f64, Vec<f64>)> {
    let n = sparse.size;
    if n == 0 {
        return None;
    }

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut vector: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0f64..1.0f64)).collect();
    if !orthonormalize(&mut vector, basis) {
        return None;
    }
    normalize(&mut vector)?;

    let mut previous_lambda = 0.0;
    for _ in 0..POWER_ITERATIONS {
        let mut next = sparse.multiply_normalized(&vector);
        if !orthonormalize(&mut next, basis) {
            return None;
        }
        normalize(&mut next)?;
        let lambda = dot(&next, &sparse.multiply_normalized(&next));
        if (lambda - previous_lambda).abs() <= POWER_TOLERANCE {
            return Some((lambda, next));
        }
        previous_lambda = lambda;
        vector = next;
    }

    let lambda = dot(&vector, &sparse.multiply_normalized(&vector));
    Some((lambda, vector))
}

fn orthonormalize(vector: &mut [f64], basis: &[Vec<f64>]) -> bool {
    for other in basis {
        let projection = dot(vector, other);
        vector
            .iter_mut()
            .zip(other.iter())
            .for_each(|(v, o)| *v -= projection * o);
    }
    vector.iter().any(|v| v.abs() > 1e-8)
}

fn normalize(vector: &mut [f64]) -> Option<f64> {
    let norm = vector.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm <= f64::EPSILON {
        return None;
    }
    vector.iter_mut().for_each(|v| *v /= norm);
    Some(norm)
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right.iter()).map(|(l, r)| l * r).sum()
}
