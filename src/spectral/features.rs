use ndarray::Array1;

/// Lightweight container for spectral vectors and auxiliary metrics.
#[derive(Debug, Clone)]
pub struct SpectralProfile {
    pub eigenvalues: Array1<f64>,
    pub norm_l2: f64,
    pub trace: f64,
}

impl SpectralProfile {
    pub fn new(eigenvalues: Array1<f64>) -> Self {
        let norm_l2 = eigenvalues.mapv(|v| v * v).sum().sqrt();
        let trace = eigenvalues.sum();
        Self {
            eigenvalues,
            norm_l2,
            trace,
        }
    }
}
