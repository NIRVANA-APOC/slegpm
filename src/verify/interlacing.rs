use std::fmt::Write;

use crate::spectral::features::SpectralProfile;

/// Implements the Cauchy interlacing verification between pattern and candidate spectra.
pub struct InterlacingVerifier;

pub struct InterlacingResult {
    pub is_valid: bool,
    pub explanation: String,
}

impl InterlacingVerifier {
    #[allow(clippy::too_many_arguments)]
    pub fn verify(
        super_graph: &SpectralProfile,
        sub_graph: &SpectralProfile,
        n: usize,
        m: usize,
        epsilon: f64,
    ) -> InterlacingResult {
        let mut explanation = String::new();
        let mut is_valid = true;

        let super_values = match super_graph.eigenvalues.as_slice() {
            Some(slice) => slice,
            None => {
                return InterlacingResult {
                    is_valid: false,
                    explanation: "Super graph eigenvalues are not contiguous".to_string(),
                };
            }
        };
        let sub_values = match sub_graph.eigenvalues.as_slice() {
            Some(slice) => slice,
            None => {
                return InterlacingResult {
                    is_valid: false,
                    explanation: "Sub graph eigenvalues are not contiguous".to_string(),
                };
            }
        };

        if n < m {
            return InterlacingResult {
                is_valid: false,
                explanation: format!(
                    "Invalid dimensions: super graph order {} < sub graph order {}",
                    n, m
                ),
            };
        }

        if sub_values.len() < m {
            return InterlacingResult {
                is_valid: false,
                explanation: format!(
                    "Insufficient subgraph eigenvalues: have {}, need {}",
                    sub_values.len(),
                    m
                ),
            };
        }

        if super_values.len() < n {
            is_valid = false;
            let _ = writeln!(
                explanation,
                "Super graph eigenvalues truncated: have {}, expected at least {}",
                super_values.len(),
                n
            );
        }

        let limit = m.min(sub_values.len());
        for i in 0..limit {
            let lower_idx = i;
            let upper_idx = i + n - m;
            if upper_idx >= super_values.len() {
                is_valid = false;
                let _ = writeln!(
                    explanation,
                    "i={}: upper index {} out of bounds for super graph spectrum size {}",
                    i,
                    upper_idx,
                    super_values.len()
                );
                break;
            }

            let lower = super_values[lower_idx];
            let upper = super_values[upper_idx];
            let value = sub_values[i];
            let within_lower = lower - epsilon <= value;
            let within_upper = value <= upper + epsilon;
            let satisfied = within_lower && within_upper;

            if !satisfied {
                is_valid = false;
            }

            let _ = writeln!(
                explanation,
                "i={}: {:.6} <= {:.6} <= {:.6}: {}",
                i, lower, value, upper, satisfied
            );
        }

        InterlacingResult {
            is_valid,
            explanation,
        }
    }
}
