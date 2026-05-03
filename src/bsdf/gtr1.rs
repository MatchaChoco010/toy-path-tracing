// GTR1 (Berry / γ=1) microfacet distribution helpers from
// Burley 2012 "Physically Based Shading at Disney" Appendix B.
//
// Used by the Disney BRDF clearcoat lobe. The α parameter here is the Disney
// `α_cc = lerp(0.1, 0.001, clearcoatGloss)` and is always isotropic.

use std::f32::consts::{PI, TAU};

use glam::{Vec2, Vec3};

const ALPHA_MIN: f32 = 1.0e-3;
const ALPHA_MAX: f32 = 1.0 - 1.0e-4;

/// `D_GTR1(θ_h)` — paper Eq. (4). At the α → 1 limit GTR1 collapses to the
/// uniform spherical distribution `1/π`; the same fallback is used by the
/// WDAS reference implementation to keep the formula numerically safe.
pub fn d_gtr1(cos_theta_h: f32, alpha: f32) -> f32 {
    if cos_theta_h <= 0.0 {
        return 0.0;
    }
    let alpha = alpha.clamp(ALPHA_MIN, 1.0);
    if alpha >= 1.0 {
        return 1.0 / PI;
    }
    let alpha2 = alpha * alpha;
    let denom = PI * alpha2.ln() * (1.0 + (alpha2 - 1.0) * cos_theta_h * cos_theta_h);
    if denom == 0.0 {
        return 0.0;
    }
    (alpha2 - 1.0) / denom
}

/// PDF of the GTR1 sampler over half-vectors: `D · cos(θ_h)`.
pub fn pdf_h_gtr1(cos_theta_h: f32, alpha: f32) -> f32 {
    if cos_theta_h <= 0.0 {
        return 0.0;
    }
    d_gtr1(cos_theta_h, alpha) * cos_theta_h
}

/// Inverse-CDF GTR1 half-vector sampler (paper Eq. 5).
///
/// `us` is two i.i.d. uniforms in `[0, 1)`. Returns the half-vector in
/// tangent space (z-up). The view direction is not needed for GTR1 sampling
/// itself; the caller reflects `wo` through the returned half-vector.
pub fn sample_h_gtr1(alpha: f32, us: Vec2) -> Vec3 {
    let alpha = alpha.clamp(ALPHA_MIN, ALPHA_MAX);
    let alpha2 = alpha * alpha;
    let phi = TAU * us.x;
    let cos2_theta = ((1.0 - alpha2.powf(1.0 - us.y)) / (1.0 - alpha2)).clamp(0.0, 1.0);
    let cos_theta = cos2_theta.sqrt();
    let sin_theta = (1.0 - cos2_theta).max(0.0).sqrt();
    Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use glam::{Vec2, Vec3};

    use super::{d_gtr1, pdf_h_gtr1, sample_h_gtr1};

    const Z_SAMPLES: usize = 512;
    const PHI_SAMPLES: usize = 256;

    fn integrate_hemisphere(f: impl Fn(Vec3) -> f32) -> f32 {
        let dz = 1.0 / Z_SAMPLES as f32;
        let dphi = TAU / PHI_SAMPLES as f32;
        let domega = dz * dphi;
        let mut acc = 0.0;
        for zi in 0..Z_SAMPLES {
            let z = (zi as f32 + 0.5) * dz;
            let r = (1.0 - z * z).max(0.0).sqrt();
            for pi in 0..PHI_SAMPLES {
                let phi = (pi as f32 + 0.5) * dphi;
                let w = Vec3::new(r * phi.cos(), r * phi.sin(), z);
                acc += f(w);
            }
        }
        acc * domega
    }

    #[test]
    fn d_gtr1_is_normalized_over_projected_hemisphere() {
        // α below ~0.1 puts the lobe inside z ∈ [0.999, 1], which the
        // regular grid integrator cannot resolve. Narrower lobes are
        // covered by `sample_h_gtr1_distribution_matches_pdf_via_histogram`.
        for &alpha in &[0.15_f32, 0.25, 0.5, 0.85] {
            let integral = integrate_hemisphere(|w| d_gtr1(w.z, alpha) * w.z);
            assert!(
                (integral - 1.0).abs() < 5.0e-3,
                "alpha={alpha}, integral={integral}"
            );
        }
    }

    #[test]
    fn pdf_h_gtr1_is_normalized() {
        for &alpha in &[0.15_f32, 0.25, 0.5, 0.85] {
            let integral = integrate_hemisphere(|w| pdf_h_gtr1(w.z, alpha));
            assert!(
                (integral - 1.0).abs() < 5.0e-3,
                "alpha={alpha}, integral={integral}"
            );
        }
    }

    #[test]
    fn d_gtr1_collapses_to_one_over_pi_when_alpha_is_one() {
        let v = d_gtr1(0.5, 1.0);
        assert!((v - 1.0 / std::f32::consts::PI).abs() < 1.0e-6);
    }

    #[test]
    fn sample_h_gtr1_returns_upper_hemisphere_unit_vector() {
        for y in 0..8 {
            for x in 0..8 {
                let us = Vec2::new((x as f32 + 0.5) / 8.0, (y as f32 + 0.5) / 8.0);
                let h = sample_h_gtr1(0.1, us);
                assert!(h.is_finite());
                assert!(h.z > 0.0);
                assert!((h.length() - 1.0).abs() < 1.0e-4);
            }
        }
    }

    #[test]
    fn sample_h_gtr1_distribution_matches_pdf_via_histogram() {
        // Bin sampled half-vectors by cos(theta_h) and compare frequencies
        // with integrated pdf over each bin. This catches major errors in the
        // inverse CDF formula.
        let alpha = 0.4_f32;
        let bin_count = 16;
        let sample_count = 200_000;
        let mut histogram = vec![0u32; bin_count];

        for n in 0..sample_count {
            // Stratified pseudo-random pair.
            let u1 = ((n as f32) * 0.123_456_7).fract();
            let u2 = ((n as f32) * 0.987_654_3 + 0.314_159_2).fract();
            let h = sample_h_gtr1(alpha, Vec2::new(u1, u2));
            let bin = ((h.z.clamp(0.0, 1.0)) * bin_count as f32) as usize;
            histogram[bin.min(bin_count - 1)] += 1;
        }

        // Expected probability per bin = ∫_{bin} pdf_h · 2π · sin θ dθ
        // = ∫_{cos_lo}^{cos_hi} 2π · pdf_h(cos) d(cos).
        // We approximate with mid-point rule.
        let mut diffs_sum_sq = 0.0;
        for (bin, &count) in histogram.iter().enumerate() {
            let lo = bin as f32 / bin_count as f32;
            let hi = (bin + 1) as f32 / bin_count as f32;
            let mid = 0.5 * (lo + hi);
            let expected_pdf = pdf_h_gtr1(mid, alpha) * std::f32::consts::TAU * (hi - lo);
            let observed = count as f32 / sample_count as f32;
            diffs_sum_sq += (observed - expected_pdf).powi(2);
        }

        // Loose tolerance — this is a coarse histogram with stratified pairs,
        // not an iid Monte Carlo run.
        assert!(
            diffs_sum_sq < 5.0e-3,
            "histogram mismatch: sum_sq={diffs_sum_sq}"
        );
    }
}
