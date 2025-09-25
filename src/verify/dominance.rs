use std::fmt::Write;

use crate::spectral::features::SpectralProfile;

const DOMINANCE_RATIO: f64 = 0.6;

#[derive(Debug, Clone)]
pub struct DominanceReport {
    pub is_dominant: bool,
    pub explanation: String,
}

pub struct DominanceChecker;

impl DominanceChecker {
    #[allow(clippy::too_many_arguments)]
    pub fn assess(
        interlacing_valid: bool,
        super_graph: &SpectralProfile,
        sub_graph: &SpectralProfile,
        factor: f64,
        threshold: f64,
        epsilon: f64,
    ) -> DominanceReport {
        let mut explanation = String::new();
        let mut satisfied = 0.0;
        let mut total_checks = 0.0;

        // Record interlacing result
        total_checks += 1.0;
        if interlacing_valid {
            satisfied += 1.0;
            let _ = writeln!(explanation, "Interlacing: pass");
        } else {
            let _ = writeln!(explanation, "Interlacing: fail");
        }

        let scaled_norm = super_graph.norm_l2 * factor;
        let norm_check = super_graph.norm_l2 + epsilon >= sub_graph.norm_l2 * DOMINANCE_RATIO
            || sub_graph.norm_l2 <= scaled_norm + epsilon;
        total_checks += 1.0;
        if norm_check {
            satisfied += 1.0;
        }
        let _ = writeln!(
            explanation,
            "Norm(H)={:.6} vs Norm(G)={:.6} (scaled {:.6}): {}",
            sub_graph.norm_l2, super_graph.norm_l2, scaled_norm, norm_check
        );

        let scaled_trace = super_graph.trace * factor;
        let trace_check = super_graph.trace + epsilon >= sub_graph.trace * DOMINANCE_RATIO
            || sub_graph.trace <= scaled_trace + epsilon;
        total_checks += 1.0;
        if trace_check {
            satisfied += 1.0;
        }
        let _ = writeln!(
            explanation,
            "Trace(H)={:.6} vs Trace(G)={:.6} (scaled {:.6}): {}",
            sub_graph.trace, super_graph.trace, scaled_trace, trace_check
        );

        let ratio = if total_checks > 0.0 {
            satisfied / total_checks
        } else {
            0.0
        };
        let dominance = ratio >= threshold;
        let _ = writeln!(
            explanation,
            "Satisfied ratio {:.2} (threshold {:.2})",
            ratio, threshold
        );
        let _ = writeln!(explanation, "Dominance: {}", dominance);

        DominanceReport {
            is_dominant: dominance,
            explanation,
        }
    }
}
