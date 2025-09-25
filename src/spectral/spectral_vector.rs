use anyhow::{Result, anyhow};
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use ndarray::Array1;
use rayon::prelude::*;

use crate::spectral::features::SpectralProfile;

/// Extracts the leading spectral components used for dominance checks.
pub struct SpectralVectorExtractor;

impl SpectralVectorExtractor {
    pub fn compute_profile(matrix: &DMatrix<f64>) -> Result<SpectralProfile> {
        if matrix.nrows() != matrix.ncols() {
            return Err(anyhow!("Spectral extraction expects a square matrix"));
        }
        if matrix.nrows() == 0 {
            return Ok(SpectralProfile::new(Array1::zeros(0)));
        }

        let symmetric = 0.5 * (matrix + matrix.transpose());
        let size = symmetric.nrows();
        let k = recommended_k(size);

        let eigenvalues = if size <= 256 {
            dense_eigendecomposition(&symmetric)
        } else {
            lanczos_smallest_eigenvalues(&symmetric, k)
        }?;

        let trimmed = eigenvalues.into_iter().take(k).collect::<Vec<_>>();
        let eigen_array = Array1::from(trimmed);
        Ok(SpectralProfile::new(eigen_array))
    }
}

fn dense_eigendecomposition(matrix: &DMatrix<f64>) -> Result<Vec<f64>> {
    let eigen = SymmetricEigen::new(matrix.clone());
    let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(eigenvalues)
}

fn lanczos_smallest_eigenvalues(matrix: &DMatrix<f64>, k: usize) -> Result<Vec<f64>> {
    let n = matrix.nrows();
    let max_iterations = (n * 3).max(1000);
    let seeds: Vec<u64> = (0..8).collect();

    let candidates: Vec<Vec<f64>> = seeds
        .into_par_iter()
        .filter_map(|seed| lanczos_iteration(matrix, k, max_iterations, seed).ok())
        .collect();

    candidates
        .into_iter()
        .filter(|eigs| eigs.len() == k)
        .min_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| anyhow!("Lanczos iteration failed to converge"))
}

fn lanczos_iteration(
    matrix: &DMatrix<f64>,
    k: usize,
    max_iterations: usize,
    seed: u64,
) -> Result<Vec<f64>> {
    use rand::prelude::*;
    use rand_xoshiro::Xoshiro256PlusPlus;

    let n = matrix.nrows();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut v = DVector::from_iterator(n, (0..n).map(|_| rng.r#gen::<f64>()));
    let norm = v.norm();
    if norm == 0.0 {
        return Err(anyhow!("Random vector sampled as zero"));
    }
    v /= norm;

    let mut alpha = Vec::with_capacity(k);
    let mut beta: Vec<f64> = Vec::with_capacity(k);
    let mut w = matrix * &v;
    let mut prev_v = DVector::zeros(n);

    for _ in 0..max_iterations {
        let a = v.dot(&w);
        alpha.push(a);
        w -= a * &v;
        if let Some(last_beta) = beta.last().copied() {
            w -= last_beta * &prev_v;
        }
        let b = w.norm();
        if b < 1e-10 {
            break;
        }
        beta.push(b);
        prev_v = v.clone();
        v = &w / b;
        w = matrix * &v;
        if alpha.len() >= k {
            break;
        }
    }

    if alpha.len() < k {
        return Err(anyhow!(
            "Lanczos iterations produced insufficient alpha coefficients"
        ));
    }

    let t = build_tridiagonal(&alpha, &beta);
    let eigen = SymmetricEigen::new(t);
    let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(eigenvalues.into_iter().take(k).collect())
}

fn build_tridiagonal(alpha: &[f64], beta: &[f64]) -> DMatrix<f64> {
    let n = alpha.len();
    let mut matrix = DMatrix::zeros(n, n);
    for i in 0..n {
        matrix[(i, i)] = alpha[i];
        if i + 1 < n {
            let b = beta.get(i).copied().unwrap_or(0.0);
            matrix[(i, i + 1)] = b;
            matrix[(i + 1, i)] = b;
        }
    }
    matrix
}

fn recommended_k(n: usize) -> usize {
    let base = (n as f64).max(1.0);
    let value = base.log2().ceil() as usize;
    value.max(1).min(n)
}
