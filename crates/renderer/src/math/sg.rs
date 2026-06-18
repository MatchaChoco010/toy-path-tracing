// Spherical Gaussian (SG) primitives used by the hierarchical light tree
// importance approximation [Tokuyoshi et al. 2024].
//
// The numerically stable forms follow the open-source VSGL reference code
// [Tokuyoshi 2024, https://github.com/yusuketokuyoshi/VSGL] and the
// supplementary documents of:
//
//   - Tokuyoshi 2022, "Accurate Diffuse Lighting from Spherical Gaussian
//     Lights" (SIGGRAPH '22 Posters).
//   - Tokuyoshi et al. 2024, "Hierarchical Light Sampling with Accurate
//     Spherical Gaussian Lighting" (SIGGRAPH Asia '24).
//
// An SG (a.k.a. von Mises-Fisher kernel scaled by an amplitude) is
//   g(o; xi, kappa) = exp(kappa * (o . xi) - kappa)
// with axis xi in S^2 and sharpness kappa >= 0. We carry the amplitude in
// log-space for numerical stability:
//   amplitude * g(o; xi, kappa) = exp(log_amplitude) * g(o; xi, kappa).

use std::f32::consts::PI;

use glam::Vec3;

const FLT_MIN_POSITIVE: f32 = f32::MIN_POSITIVE;
// Clamp on vMF sharpness to avoid overflow when computing exp(-kappa) etc.
// The same threshold is used in VSGL (`SGLIGHT_SHARPNESS_MAX = 2^41`).
pub const SG_SHARPNESS_MAX: f32 = 2.199_023_3e12; // 2^41

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SgLobe {
    pub axis: Vec3,
    pub sharpness: f32,
    pub log_amplitude: f32,
}

impl SgLobe {
    pub fn new(axis: Vec3, sharpness: f32, log_amplitude: f32) -> Self {
        Self {
            axis,
            sharpness,
            log_amplitude,
        }
    }
}

// (e^x - 1) / x with cancellation-of-rounding-errors trick.
// [Higham 2002, Accuracy and Stability of Numerical Algorithms, §1.14.1, p.19]
//
// A naive `(x.exp() - 1.0) / x` loses precision near x = 0; this form is
// well-defined for the entire range. Used to evaluate (1 - e^-kappa)/kappa for
// SG hemispherical / full-sphere integrals when kappa is small.
pub fn expm1_over_x(x: f32) -> f32 {
    let u = x.exp();
    if u == 1.0 {
        return 1.0;
    }
    let y = u - 1.0;
    if x.abs() < 1.0 { y / u.ln() } else { y / x }
}

// erf approximation matching VSGL Math.hlsli. This is good to ~1e-7 and
// avoids pulling libm. Mirrors the polynomial split at |x| > 1.
pub fn erf(x: f32) -> f32 {
    if x.abs() >= 4.0 {
        return 1.0_f32.copysign(x);
    }

    if x.abs() > 1.0 {
        const A1: f32 = 1.628_459_5;
        const A2: f32 = 9.156_747e-1;
        const A3: f32 = 1.543_293_9e-1;
        const A4: f32 = -3.517_598_3e-2;
        const A5: f32 = 5.667_955_6e-3;
        const A6: f32 = -5.648_746e-4;
        const A7: f32 = 2.589_076_8e-5;
        let a = x.abs();
        let y = 1.0
            - ((((((((A7 * a + A6) * a + A5) * a + A4) * a + A3) * a + A2) * a + A1) * a)
                .exp2_negative());
        return y.copysign(x);
    }

    const A1: f32 = std::f32::consts::FRAC_2_SQRT_PI;
    const A2: f32 = -3.761_23e-1;
    const A3: f32 = 1.127_992_2e-1;
    const A4: f32 = -2.670_306_5e-2;
    const A5: f32 = 4.907_355_6e-3;
    const A6: f32 = -5.588_531_5e-4;
    let x2 = x * x;
    (((((A6 * x2 + A5) * x2 + A4) * x2 + A3) * x2 + A2) * x2 + A1) * x
}

trait Exp2Negative {
    fn exp2_negative(self) -> f32;
}

impl Exp2Negative for f32 {
    // 2^(-x) using the platform exp2; written this way to mirror the VSGL
    // call structure where the polynomial value is negated before exp2.
    fn exp2_negative(self) -> f32 {
        (-self).exp2()
    }
}

// 1 - erf(x).
//
// Naively `1 - erf(x)` loses precision for large positive x because erf(x)
// approaches 1; that is acceptable here because the only consumer
// (`sg_clamped_cosine_product_integral_over_pi`) clamps the lerp factor with
// `f32::EPSILON / 2` to absorb that loss, matching the VSGL conservative path.
pub fn erfc(x: f32) -> f32 {
    1.0 - erf(x)
}

// Exact integral of an SG over the sphere:
//   integral = 2 pi (1 - e^{-2 kappa}) / kappa.
// Numerically stable form via expm1_over_x; reduces to 4 pi as kappa -> 0.
pub fn sg_integral(sharpness: f32) -> f32 {
    4.0 * PI * expm1_over_x(-2.0 * sharpness)
}

// Approximate solution for the SG integral. Valid when sharpness is not small
// (>= ~0.5). Use the exact form by default; this is provided for parity with
// the VSGL reference and for places where sharpness is guaranteed large.
pub fn sg_approx_integral(sharpness: f32) -> f32 {
    2.0 * PI / sharpness
}

// Numerically stable product of two SGs.
//   g(.; a1, k1) * g(.; a2, k2) = exp(log_amp) g(.; (k1 a1 + k2 a2)/k3, k3)
// where k3 = ||k1 a1 + k2 a2||. The naive log_amp = k3 - k1 - k2 cancels for
// large sharpnesses; we use the form
//   log_amp = -k1 k2 ||a1 - a2||^2 / (k3 + k1 + k2)
// (see VSGL `SphericalGaussian.hlsli::SGProduct`).
pub fn sg_product(axis1: Vec3, sharpness1: f32, axis2: Vec3, sharpness2: f32) -> SgLobe {
    let axis = axis1 * sharpness1 + axis2 * sharpness2;
    let sharpness = axis.length();
    let d = axis1 - axis2;
    let len2 = d.dot(d);
    let denom = (sharpness + sharpness1 + sharpness2).max(FLT_MIN_POSITIVE);
    let log_amplitude = -sharpness1 * sharpness2 * len2 / denom;
    let normalized_axis = if sharpness > FLT_MIN_POSITIVE {
        axis / sharpness
    } else {
        Vec3::Z
    };
    SgLobe {
        axis: normalized_axis,
        sharpness,
        log_amplitude,
    }
}

// Interpolation factor v in [0, 1] for the upper-hemisphere integral of an SG
// [Tokuyoshi 2022, Eq. 3]. cosine = xi . n.
pub fn sg_normalized_hemispherical_integral(cosine: f32, sharpness: f32) -> f32 {
    const A: f32 = 0.651_732_88;
    const B: f32 = 1.341_828;
    const C: f32 = 7.221_669;
    let steepness = sharpness * ((0.5 * sharpness + A) / ((sharpness + B) * sharpness + C)).sqrt();
    let cosine = cosine.clamp(-1.0, 1.0);
    let denom = erf(steepness);
    let numer = erf(steepness * cosine);
    let v = if denom.abs() > 0.0 {
        0.5 + 0.5 * (numer / denom)
    } else {
        // sharpness -> 0; SG approaches uniform, hemispherical fraction = 1/2.
        0.5
    };
    v.clamp(0.0, 1.0)
}

// Approximate hemispherical integral of an SG divided by 2 pi.
//   ∫ H(o.n) g(o; xi, kappa) do / (2 pi)
//   = lerp(e^-kappa, 1, v) * (1 - e^-kappa)/kappa
// where v is the interpolation factor from [Tokuyoshi 2022].
pub fn sg_hemispherical_integral_over_two_pi(cosine: f32, sharpness: f32) -> f32 {
    let v = sg_normalized_hemispherical_integral(cosine, sharpness);
    let e = (-sharpness).exp();
    (e + (1.0 - e) * v) * expm1_over_x(-sharpness)
}

pub fn sg_hemispherical_integral(cosine: f32, sharpness: f32) -> f32 {
    2.0 * PI * sg_hemispherical_integral_over_two_pi(cosine, sharpness)
}

// Hemispherical integral of a *normalized* SG (a.k.a. vMF). Returns a value
// in [0, 1]. Used as the filtered visibility V in the glossy SG lighting
// approximation [Tokuyoshi et al. 2024, §5.2].
pub fn vmf_hemispherical_integral(cosine: f32, sharpness: f32) -> f32 {
    let v = sg_normalized_hemispherical_integral(cosine, sharpness);
    let e = (-sharpness).exp();
    let num = e + (1.0 - e) * v;
    num / (e + 1.0)
}

// Banerjee 2005's vMF sharpness estimator: given the average direction length
//   r = ||mean of N unit vectors in R^3||,
// returns the maximum-likelihood sharpness lambda that fits the data:
//   lambda(r) = r * (3 - r^2) / (1 - r^2).
// As r -> 1 (perfectly aligned), lambda -> infinity; as r -> 0 (uniform on the
// sphere), lambda -> 0. We clamp r to [0, 1) and the result to SG_SHARPNESS_MAX
// to avoid overflows when fitting tightly clustered emitters.
pub fn vmf_axis_length_to_sharpness(axis_length: f32) -> f32 {
    let r = axis_length.clamp(0.0, 1.0);
    if r >= 1.0 {
        return SG_SHARPNESS_MAX;
    }
    let one_minus = 1.0 - r * r;
    if one_minus <= 0.0 {
        return SG_SHARPNESS_MAX;
    }
    let lambda = r * (3.0 - r * r) / one_minus;
    lambda.clamp(0.0, SG_SHARPNESS_MAX)
}

// Inverse of `vmf_axis_length_to_sharpness`. Solves the cubic
//   x^3 - s x^2 - 3 x + s = 0 with x in [0, 1] and s in [0, infty).
// Implementation: numerically stable cubic formula from
// [Peters, "How to solve a cubic equation, revisited"]. Mirrors VSGL.
pub fn vmf_sharpness_to_axis_length(sharpness: f32) -> f32 {
    if sharpness >= 33_554_432.0 {
        // 2^25; matches VSGL's overflow shortcut.
        return 1.0;
    }
    if sharpness <= 0.0 {
        return 0.0;
    }
    let a = sharpness / 3.0;
    let b = a * a * a;
    let c = (1.0 + 3.0 * (a * a) * (1.0 + a * a)).sqrt();
    let theta = c.atan2(b) / 3.0;
    let d = -2.0 * (PI / 6.0 - theta).sin();
    ((1.0 + a * a).sqrt() * d + a).clamp(0.0, 1.0)
}

// B_hat(kappa) / (2 pi) = (e^-kappa - 1 + kappa) / kappa^2.
// Stable Taylor approximation for small kappa to avoid catastrophic
// cancellation [Tokuyoshi et al. 2024, Sup. Listing 5].
pub fn upper_sg_clamped_cosine_integral_over_two_pi(sharpness: f32) -> f32 {
    if sharpness <= 0.5 {
        let s = sharpness;
        return (((((((-1.0 / 362_880.0) * s + 1.0 / 40_320.0) * s - 1.0 / 5_040.0) * s
            + 1.0 / 720.0)
            * s
            - 1.0 / 120.0)
            * s
            + 1.0 / 24.0)
            * s
            - 1.0 / 6.0)
            * s
            + 0.5;
    }
    ((-sharpness).exp_m1() + sharpness) / (sharpness * sharpness)
}

// B_check(kappa) / (2 pi) = e^-kappa (1 - e^-kappa - kappa e^-kappa) / kappa^2.
// Stable Taylor approximation for small kappa [Tokuyoshi et al. 2024,
// Sup. Listing 6].
pub fn lower_sg_clamped_cosine_integral_over_two_pi(sharpness: f32) -> f32 {
    let e = (-sharpness).exp();
    if sharpness <= 0.5 {
        let s = sharpness;
        return e
            * (((((((((1.0 / 403_200.0) * s - 1.0 / 45_360.0) * s + 1.0 / 5_760.0) * s
                - 1.0 / 840.0)
                * s
                + 1.0 / 144.0)
                * s
                - 1.0 / 30.0)
                * s
                + 1.0 / 8.0)
                * s
                - 1.0 / 3.0)
                * s
                + 0.5);
    }
    e * (-(-sharpness).exp_m1() - sharpness * e) / (sharpness * sharpness)
}

// Approximate product integral of an SG and a clamped cosine, divided by pi:
//   sg_clamped_cosine_product_integral_over_pi(z, kappa)
//     = (1/pi) ∫ g(o; xi, kappa) max(o.n, 0) do,    z = xi . n.
//
// Returns a value >= 0 and satisfies the unbiased-sampling constraint
// [Tokuyoshi et al. 2024, Sec. 4]. We use the conservative-clamp form (matching
// the VSGL public implementation) because our `erfc` reduces to `1 - erf` which
// can lose precision for large arguments.
pub fn sg_clamped_cosine_product_integral_over_pi(z: f32, sharpness: f32) -> f32 {
    const A: f32 = 2.736_083;
    const B: f32 = 17.021_297;
    const C: f32 = 4.010_082_7;
    const D: f32 = 15.219_156;
    const E: f32 = 76.087_9;
    let s = sharpness;
    let t = s * (0.5 * ((s + A) * s + B) / (((s + C) * s + D) * s + E)).sqrt();
    let tz = t * z;
    const INV_SQRT_PI: f32 = 0.564_189_6;
    const CLAMP_MIN: f32 = 0.5 * f32::EPSILON;
    let lerp_factor = (0.5 * (z * erfc(-tz) + erfc(t))
        - 0.5 * INV_SQRT_PI * (-tz * tz).exp() * (t * t * (z * z - 1.0)).exp_m1()
            / t.max(FLT_MIN_POSITIVE))
    .clamp(CLAMP_MIN, 1.0);
    let lower = lower_sg_clamped_cosine_integral_over_two_pi(sharpness);
    let upper = upper_sg_clamped_cosine_integral_over_two_pi(sharpness);
    2.0 * (lower + (upper - lower) * lerp_factor)
}

pub fn sg_clamped_cosine_product_integral(z: f32, sharpness: f32) -> f32 {
    PI * sg_clamped_cosine_product_integral_over_pi(z, sharpness)
}

// Rec. 709 / sRGB linear luminance. Used to fold spectral importances into a
// scalar so that `pmf` and `pdf` are well-defined regardless of channel.
pub fn luminance(rgb: Vec3) -> f32 {
    0.2126 * rgb.x + 0.7152 * rgb.y + 0.0722 * rgb.z
}

// Merge two clusters' (mu, sigma_s^2, nu_bar, Phi) parameters into a parent
// cluster as in [Tokuyoshi et al. 2024, Eqs. 2, 4, 5]. `nu_bar` is the
// flux-weighted *un-normalized* vMF axis (length encodes sharpness once
// converted via Banerjee).
//
// Returns (mu, sigma_s^2, nu_bar, Phi). Caller converts `nu_bar` to (nu, lambda)
// at the end of the bottom-up pass via `vmf_axis_length_to_sharpness`.
pub fn merge_cluster_params(
    flux_left: f32,
    mu_left: Vec3,
    sigma_s2_left: f32,
    nu_bar_left: Vec3,
    flux_right: f32,
    mu_right: Vec3,
    sigma_s2_right: f32,
    nu_bar_right: Vec3,
) -> (Vec3, f32, Vec3, f32) {
    let total_flux = flux_left + flux_right;
    if total_flux <= 0.0 {
        return (Vec3::ZERO, 0.0, Vec3::ZERO, 0.0);
    }
    let w_l = flux_left / total_flux;
    let w_r = flux_right / total_flux;
    let mu = w_l * mu_left + w_r * mu_right;
    let mu_diff = mu_left - mu_right;
    let sigma_s2 = w_l * sigma_s2_left + w_r * sigma_s2_right + w_l * w_r * mu_diff.dot(mu_diff);
    let nu_bar = w_l * nu_bar_left + w_r * nu_bar_right;
    (mu, sigma_s2, nu_bar, total_flux)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_utils::approx_f as close;

    #[test]
    fn expm1_over_x_handles_zero_and_unity() {
        assert!(close(expm1_over_x(0.0), 1.0, 1e-6));
        // (e - 1) / 1 ≈ 1.71828
        assert!(close(expm1_over_x(1.0), std::f32::consts::E - 1.0, 1e-5));
    }

    #[test]
    fn expm1_over_x_handles_small_negative_values() {
        // The naive `(x.exp()-1)/x` form is unstable for tiny x (this is why we
        // need Higham's reformulation in the first place). Compare against an
        // f64 reference instead.
        let x = -1.0e-4_f32;
        let expected = (((x as f64).exp() - 1.0) / x as f64) as f32;
        assert!(
            close(expm1_over_x(x), expected, 1.0e-5),
            "expm1_over_x={} expected~{} (Higham form)",
            expm1_over_x(x),
            expected
        );
    }

    #[test]
    fn erf_matches_known_values() {
        // erf(0) = 0, erf(1) ≈ 0.8427, erf(2) ≈ 0.9953
        assert!(close(erf(0.0), 0.0, 1e-6));
        assert!(close(erf(1.0), 0.842_700_8, 1e-4));
        assert!(close(erf(2.0), 0.995_322_3, 1e-4));
        assert!(close(erf(-1.0), -0.842_700_8, 1e-4));
        assert!(close(erf(5.0), 1.0, 1e-6));
    }

    #[test]
    fn sg_integral_reduces_to_4pi_at_zero() {
        assert!(close(sg_integral(0.0), 4.0 * PI, 1e-4));
    }

    #[test]
    fn sg_integral_matches_2pi_over_kappa_for_large_kappa() {
        let k = 100.0_f32;
        let approx = 2.0 * PI / k;
        // 4pi expm1_over_x(-2k) = 4pi (e^-2k - 1)/(-2k) ~= 4pi/(2k) = 2pi/k.
        assert!(close(sg_integral(k), approx, 1e-3));
    }

    #[test]
    fn sg_product_with_identical_axes_collapses() {
        // Same axis: product axis stays the same, sharpness adds, log_amp = 0.
        let p = sg_product(Vec3::Z, 5.0, Vec3::Z, 3.0);
        assert!(p.axis.abs_diff_eq(Vec3::Z, 1e-5));
        assert!(close(p.sharpness, 8.0, 1e-5));
        assert!(close(p.log_amplitude, 0.0, 1e-5));
    }

    #[test]
    fn sg_product_with_opposite_axes_has_zero_sharpness() {
        // Equal but opposite axes with equal sharpness cancel.
        let p = sg_product(Vec3::Z, 5.0, -Vec3::Z, 5.0);
        assert!(close(p.sharpness, 0.0, 1e-5));
        // log_amp = -k1 k2 ||a1-a2||^2 / (k3+k1+k2) = -25*4/(0+10) = -10.
        assert!(close(p.log_amplitude, -10.0, 1e-5));
    }

    #[test]
    fn sg_normalized_hemispherical_integral_axis_perpendicular_is_half() {
        // For any sharpness, axis perpendicular to the surface (cos = 0) splits
        // the SG mass evenly between the two hemispheres, so v = 1/2.
        for &k in &[0.5_f32, 5.0, 50.0, 500.0] {
            let v = sg_normalized_hemispherical_integral(0.0, k);
            assert!(close(v, 0.5, 1e-5), "k={}: v={}", k, v);
        }
    }

    #[test]
    fn sg_normalized_hemispherical_integral_axis_at_pole_for_sharp() {
        // Sharp SG aligned with the pole: nearly all mass above => v ≈ 1.
        let v = sg_normalized_hemispherical_integral(1.0, 100.0);
        assert!(v > 0.99);
    }

    #[test]
    fn sg_normalized_hemispherical_integral_below_pole_is_zero_for_sharp() {
        let v = sg_normalized_hemispherical_integral(-1.0, 100.0);
        assert!(v < 0.01);
    }

    #[test]
    fn sg_hemispherical_integral_over_two_pi_for_zero_sharpness_is_half() {
        // Uniform SG: half goes above the hemisphere, integral = 2pi.
        let v = sg_hemispherical_integral_over_two_pi(0.0, 0.0);
        assert!(close(v, 1.0, 1e-3)); // (g(.; xi, 0) = 1, integral = 2pi -> /2pi = 1).
    }

    #[test]
    fn vmf_axis_length_to_sharpness_monotone() {
        let l1 = vmf_axis_length_to_sharpness(0.1);
        let l2 = vmf_axis_length_to_sharpness(0.5);
        let l3 = vmf_axis_length_to_sharpness(0.9);
        assert!(l1 < l2);
        assert!(l2 < l3);
    }

    #[test]
    fn vmf_round_trip_preserves_axis_length() {
        for &r in &[0.1_f32, 0.3, 0.5, 0.7, 0.9, 0.99] {
            let lambda = vmf_axis_length_to_sharpness(r);
            let r2 = vmf_sharpness_to_axis_length(lambda);
            assert!(
                (r - r2).abs() < 5e-3,
                "round trip r={} lambda={} -> r={} (delta {})",
                r,
                lambda,
                r2,
                r - r2
            );
        }
    }

    #[test]
    fn upper_clamped_cosine_integral_at_zero_is_pi() {
        // ∫ g(o; n, 0) max(o.n, 0) do = ∫ max(o.n, 0) do = pi.
        // Divided by 2 pi -> 0.5.
        assert!(close(
            upper_sg_clamped_cosine_integral_over_two_pi(0.0),
            0.5,
            1e-4
        ));
    }

    #[test]
    fn lower_clamped_cosine_integral_at_zero_is_pi() {
        // SG with axis at -n at kappa=0 is uniform: ∫ max(o.n, 0) do = pi.
        // /2pi -> 0.5.
        assert!(close(
            lower_sg_clamped_cosine_integral_over_two_pi(0.0),
            0.5,
            1e-4
        ));
    }

    #[test]
    fn sg_clamped_cosine_product_integral_at_axis_aligned_kappa_zero() {
        // kappa = 0 SG is uniform; ∫ uniform * max(cos, 0) do = pi.
        // The function returns the integral / pi.
        let i = sg_clamped_cosine_product_integral_over_pi(1.0, 0.0);
        assert!(close(i, 1.0, 1e-3));
    }

    #[test]
    fn sg_clamped_cosine_product_integral_is_nonnegative() {
        for &z in &[-1.0_f32, -0.5, 0.0, 0.5, 1.0] {
            for &k in &[0.0_f32, 0.1, 1.0, 10.0, 100.0, 1000.0] {
                let v = sg_clamped_cosine_product_integral_over_pi(z, k);
                assert!(v >= 0.0, "negative integral at z={}, kappa={}: {}", z, k, v);
            }
        }
    }

    #[test]
    fn sg_clamped_cosine_product_integral_is_increasing_in_z() {
        // For fixed kappa, the integral grows monotonically with z = xi . n.
        let kappa = 50.0;
        let mut prev = sg_clamped_cosine_product_integral_over_pi(-1.0, kappa);
        for i in 1..=20 {
            let z = -1.0 + 2.0 * (i as f32) / 20.0;
            let curr = sg_clamped_cosine_product_integral_over_pi(z, kappa);
            assert!(
                curr >= prev - 1e-6,
                "non-monotonic at z={}: {} -> {}",
                z,
                prev,
                curr
            );
            prev = curr;
        }
    }

    #[test]
    fn vmf_hemispherical_integral_in_unit_interval() {
        for &z in &[-1.0_f32, -0.5, 0.0, 0.5, 1.0] {
            for &k in &[0.0_f32, 0.5, 5.0, 50.0, 500.0] {
                let v = vmf_hemispherical_integral(z, k);
                assert!(
                    (0.0..=1.0).contains(&v),
                    "vmf hemi out of range at z={}, k={}: {}",
                    z,
                    k,
                    v
                );
            }
        }
    }

    #[test]
    fn merge_cluster_params_recovers_left_when_right_has_zero_flux() {
        let (mu, sigma2, nu_bar, flux) = merge_cluster_params(
            2.0,
            Vec3::new(1.0, 2.0, 3.0),
            0.5,
            Vec3::new(0.0, 0.0, 0.5),
            0.0,
            Vec3::ZERO,
            0.0,
            Vec3::ZERO,
        );
        assert!(mu.abs_diff_eq(Vec3::new(1.0, 2.0, 3.0), 1e-5));
        assert!(close(sigma2, 0.5, 1e-5));
        assert!(nu_bar.abs_diff_eq(Vec3::new(0.0, 0.0, 0.5), 1e-5));
        assert!(close(flux, 2.0, 1e-5));
    }

    #[test]
    fn merge_cluster_params_combines_two_equal_weight_points() {
        // Two point clusters at (-1,0,0) and (+1,0,0) with equal flux:
        // mu = 0, sigma_s^2 = (0+0) + 0.5*0.5*||(2,0,0)||^2 = 0.25*4 = 1.
        let (mu, sigma2, _nu, flux) = merge_cluster_params(
            1.0,
            Vec3::new(-1.0, 0.0, 0.0),
            0.0,
            Vec3::ZERO,
            1.0,
            Vec3::new(1.0, 0.0, 0.0),
            0.0,
            Vec3::ZERO,
        );
        assert!(mu.abs_diff_eq(Vec3::ZERO, 1e-5));
        assert!(close(sigma2, 1.0, 1e-5));
        assert!(close(flux, 2.0, 1e-5));
    }

    #[test]
    fn luminance_matches_srgb_coefficients() {
        let g = luminance(Vec3::new(0.0, 1.0, 0.0));
        let r = luminance(Vec3::new(1.0, 0.0, 0.0));
        let b = luminance(Vec3::new(0.0, 0.0, 1.0));
        assert!(close(r, 0.2126, 1e-4));
        assert!(close(g, 0.7152, 1e-4));
        assert!(close(b, 0.0722, 1e-4));
    }
}
