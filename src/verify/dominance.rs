use std::fmt::Write;

use crate::spectral::features::SpectralProfile;

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

        let norm_check = sub_graph.norm_l2 <= super_graph.norm_l2 * factor + epsilon;
        total_checks += 1.0;
        if norm_check {
            satisfied += 1.0;
        }
        let _ = writeln!(
            explanation,
            "Norm(H)={:.6} <= Norm(G)*factor={:.6}: {}",
            sub_graph.norm_l2,
            super_graph.norm_l2 * factor,
            norm_check
        );

        let trace_check = sub_graph.trace <= super_graph.trace * factor + epsilon;
        total_checks += 1.0;
        if trace_check {
            satisfied += 1.0;
        }
        let _ = writeln!(
            explanation,
            "Trace(H)={:.6} <= Trace(G)*factor={:.6}: {}",
            sub_graph.trace,
            super_graph.trace * factor,
            trace_check
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
