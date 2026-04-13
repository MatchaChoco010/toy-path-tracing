use std::f32::consts::TAU;

use glam::{Vec2, Vec3};

use super::{BsdfFlags, BsdfSample};

const MIN_ALPHA: f32 = 1.0e-4;
const EFFECTIVELY_SMOOTH_ALPHA: f32 = 1.0e-3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConductorGgxBsdf {
    base_color: Vec3,
    alpha_x: f32,
    alpha_y: f32,
}

impl ConductorGgxBsdf {
    pub fn new(base_color: Vec3, alpha_x: f32, alpha_y: f32) -> Self {
        Self {
            base_color: base_color.clamp(Vec3::ZERO, Vec3::ONE),
            alpha_x: alpha_x.max(MIN_ALPHA),
            alpha_y: alpha_y.max(MIN_ALPHA),
        }
    }

    pub fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if !is_upper_hemisphere(wo) || !is_upper_hemisphere(wi) || self.effectively_smooth() {
            return Vec3::ZERO;
        }

        let Some(wm) = half_vector(wo, wi) else {
            return Vec3::ZERO;
        };

        let cos_theta_o = wo.z.max(0.0);
        let cos_theta_i = wi.z.max(0.0);
        if cos_theta_o <= 0.0 || cos_theta_i <= 0.0 {
            return Vec3::ZERO;
        }

        let d = ggx_d(wm, self.alpha_x, self.alpha_y);
        if d <= 0.0 {
            return Vec3::ZERO;
        }

        let g = ggx_g2_height_correlated(wo, wi, self.alpha_x, self.alpha_y);
        if g <= 0.0 {
            return Vec3::ZERO;
        }

        let f = schlick_fresnel(self.base_color, wo.dot(wm).abs());
        f * (d * g / (4.0 * cos_theta_o * cos_theta_i))
    }

    pub fn pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        if !is_upper_hemisphere(wo) || !is_upper_hemisphere(wi) || self.effectively_smooth() {
            return 0.0;
        }

        let Some(wm) = half_vector(wo, wi) else {
            return 0.0;
        };

        let pdf_wm = pdf_wm_bounded_vndf(wo, wm, self.alpha_x, self.alpha_y);
        let jacobian = reflection_jacobian(wo, wm);
        pdf_wm * jacobian
    }

    pub fn sample(&self, wo: Vec3, us: Vec2) -> Option<BsdfSample> {
        if !is_upper_hemisphere(wo) {
            return None;
        }

        if self.effectively_smooth() {
            return self.sample_smooth(wo);
        }

        let wm = sample_wm_bounded_vndf(wo, self.alpha_x, self.alpha_y, us)?;
        let wi = reflect_local(wo, wm);
        if !is_upper_hemisphere(wi) {
            return None;
        }

        let pdf_wm = pdf_wm_bounded_vndf(wo, wm, self.alpha_x, self.alpha_y);
        let jacobian = reflection_jacobian(wo, wm);
        let pdf = pdf_wm * jacobian;
        if pdf <= 0.0 {
            return None;
        }

        let g1 = ggx_g1(wo, self.alpha_x, self.alpha_y);
        if g1 <= 0.0 {
            return None;
        }

        let g2 = ggx_g2_height_correlated(wo, wi, self.alpha_x, self.alpha_y);
        let vndf_pdf_wm = visible_normal_pdf_wm(wo, wm, self.alpha_x, self.alpha_y);
        if vndf_pdf_wm <= 0.0 {
            return None;
        }

        // For bounded VNDF sampling:
        // weight = f * cos(theta_i) / pdf
        //        = F * (G / G1(wo)) * (p_v(wm | wo) / p_b(wm | wo))
        let f = schlick_fresnel(self.base_color, wo.dot(wm).abs());
        let weight_scale = (g2 / g1) * (vndf_pdf_wm / pdf_wm);
        let weight = f * weight_scale;

        Some(BsdfSample {
            weight,
            wi,
            pdf,
            flags: BsdfFlags::GLOSSY | BsdfFlags::REFLECTION,
        })
    }

    fn sample_smooth(&self, wo: Vec3) -> Option<BsdfSample> {
        let wi = Vec3::new(-wo.x, -wo.y, wo.z).normalize_or_zero();
        if !is_upper_hemisphere(wi) {
            return None;
        }

        let weight = schlick_fresnel(self.base_color, wi.z.abs());

        Some(BsdfSample {
            weight,
            wi,
            pdf: 1.0,
            flags: BsdfFlags::DELTA | BsdfFlags::REFLECTION,
        })
    }

    fn effectively_smooth(&self) -> bool {
        self.alpha_x.max(self.alpha_y) < EFFECTIVELY_SMOOTH_ALPHA
    }
}

pub fn schlick_fresnel(f0: Vec3, cos_theta: f32) -> Vec3 {
    let cos_theta = cos_theta.clamp(0.0, 1.0);
    let one_minus_cos_theta = 1.0 - cos_theta;
    f0 + (Vec3::ONE - f0) * one_minus_cos_theta.powi(5)
}

fn ggx_d(wm: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
    if wm.z <= 0.0 {
        return 0.0;
    }

    let term = wm.x * wm.x / (alpha_x * alpha_x) + wm.y * wm.y / (alpha_y * alpha_y) + wm.z * wm.z;
    let denom = std::f32::consts::PI * alpha_x * alpha_y * term * term;
    if denom <= 0.0 { 0.0 } else { 1.0 / denom }
}

fn ggx_lambda(w: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
    let cos_theta = w.z.abs();
    if cos_theta <= 0.0 {
        return f32::INFINITY;
    }

    let term = 1.0
        + (alpha_x * alpha_x * w.x * w.x + alpha_y * alpha_y * w.y * w.y) / (cos_theta * cos_theta);
    0.5 * (-1.0 + term.sqrt())
}

fn ggx_g1(w: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
    let lambda = ggx_lambda(w, alpha_x, alpha_y);
    if !lambda.is_finite() {
        return 0.0;
    }
    1.0 / (1.0 + lambda)
}

fn ggx_g2_height_correlated(wo: Vec3, wi: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
    let lambda_o = ggx_lambda(wo, alpha_x, alpha_y);
    let lambda_i = ggx_lambda(wi, alpha_x, alpha_y);
    if !lambda_o.is_finite() || !lambda_i.is_finite() {
        return 0.0;
    }
    1.0 / (1.0 + lambda_o + lambda_i)
}

fn visible_normal_pdf_wm(wo: Vec3, wm: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
    if !is_upper_hemisphere(wo) || wm.z <= 0.0 {
        return 0.0;
    }

    let d = ggx_d(wm, alpha_x, alpha_y);
    let g1 = ggx_g1(wo, alpha_x, alpha_y);
    let dot_wm_wo = wo.dot(wm).max(0.0);
    if d <= 0.0 || g1 <= 0.0 || dot_wm_wo <= 0.0 {
        return 0.0;
    }

    d * g1 * dot_wm_wo / wo.z
}

fn reflection_jacobian(wo: Vec3, wm: Vec3) -> f32 {
    let denom = 4.0 * wo.dot(wm).abs();
    if denom <= 0.0 { 0.0 } else { 1.0 / denom }
}

fn half_vector(wo: Vec3, wi: Vec3) -> Option<Vec3> {
    let wm = (wo + wi).normalize_or_zero();
    if wm.length_squared() == 0.0 {
        return None;
    }

    Some(if wm.z < 0.0 { -wm } else { wm })
}

fn reflect_local(wo: Vec3, wm: Vec3) -> Vec3 {
    (-wo + 2.0 * wo.dot(wm) * wm).normalize_or_zero()
}

fn is_upper_hemisphere(w: Vec3) -> bool {
    w.z > 0.0
}

pub fn sample_wm_bounded_vndf(wo: Vec3, alpha_x: f32, alpha_y: f32, us: Vec2) -> Option<Vec3> {
    if wo.z <= 0.0 {
        return None;
    }

    let alpha_x = alpha_x.max(MIN_ALPHA);
    let alpha_y = alpha_y.max(MIN_ALPHA);
    let wo_std = Vec3::new(alpha_x * wo.x, alpha_y * wo.y, wo.z).normalize_or_zero();
    if wo_std.length_squared() == 0.0 {
        return None;
    }

    let phi = TAU * us.x;
    let a = alpha_x.min(alpha_y).clamp(0.0, 1.0);
    let s = 1.0 + wo.truncate().length();
    let a2 = a * a;
    let s2 = s * s;
    let k = (1.0 - a2) * s2 / (s2 + a2 * wo.z * wo.z);
    let lower_bound = if wo.z > 0.0 { -k * wo_std.z } else { -wo_std.z };
    let z = lower_bound.mul_add(us.y, 1.0 - us.y);
    let sin_theta = (1.0 - z * z).clamp(0.0, 1.0).sqrt();
    let o_std = Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), z);
    let wm_std = wo_std + o_std;
    let wm = Vec3::new(alpha_x * wm_std.x, alpha_y * wm_std.y, wm_std.z).normalize_or_zero();

    if wm.length_squared() == 0.0 || wm.z <= 0.0 {
        return None;
    }

    Some(wm)
}

pub fn pdf_wm_bounded_vndf(wo: Vec3, wm: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
    if wo.z <= 0.0 || wm.z <= 0.0 {
        return 0.0;
    }

    let alpha_x = alpha_x.max(MIN_ALPHA);
    let alpha_y = alpha_y.max(MIN_ALPHA);
    let a = alpha_x.min(alpha_y).clamp(0.0, 1.0);
    let s = 1.0 + wo.truncate().length();
    let a2 = a * a;
    let s2 = s * s;
    let k = (1.0 - a2) * s2 / (s2 + a2 * wo.z * wo.z);
    let t =
        (alpha_x * alpha_x * wo.x * wo.x + alpha_y * alpha_y * wo.y * wo.y + wo.z * wo.z).sqrt();
    let denom = k * wo.z + t;
    if denom <= 0.0 {
        return 0.0;
    }

    2.0 * ggx_d(wm, alpha_x, alpha_y) * wo.dot(wm).max(0.0) / denom
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use glam::{Vec2, Vec3};

    use crate::bsdf::{BsdfFlags, ConductorGgxBsdf};

    use super::{ggx_d, pdf_wm_bounded_vndf, schlick_fresnel, visible_normal_pdf_wm};

    const HEMISPHERE_Z_SAMPLES: usize = 256;
    const HEMISPHERE_PHI_SAMPLES: usize = 256;

    fn integrate_hemisphere(f: impl Fn(Vec3) -> f32) -> f32 {
        let dz = 1.0 / HEMISPHERE_Z_SAMPLES as f32;
        let dphi = TAU / HEMISPHERE_PHI_SAMPLES as f32;
        let domega = dz * dphi;
        let mut integral = 0.0;

        for z_index in 0..HEMISPHERE_Z_SAMPLES {
            let z = (z_index as f32 + 0.5) * dz;
            let r = (1.0 - z * z).max(0.0).sqrt();

            for phi_index in 0..HEMISPHERE_PHI_SAMPLES {
                let phi = (phi_index as f32 + 0.5) * dphi;
                let w = Vec3::new(r * phi.cos(), r * phi.sin(), z);
                integral += f(w);
            }
        }

        integral * domega
    }

    fn integrate_hemisphere_vec3(f: impl Fn(Vec3) -> Vec3) -> Vec3 {
        let dz = 1.0 / HEMISPHERE_Z_SAMPLES as f32;
        let dphi = TAU / HEMISPHERE_PHI_SAMPLES as f32;
        let domega = dz * dphi;
        let mut integral = Vec3::ZERO;

        for z_index in 0..HEMISPHERE_Z_SAMPLES {
            let z = (z_index as f32 + 0.5) * dz;
            let r = (1.0 - z * z).max(0.0).sqrt();

            for phi_index in 0..HEMISPHERE_PHI_SAMPLES {
                let phi = (phi_index as f32 + 0.5) * dphi;
                let w = Vec3::new(r * phi.cos(), r * phi.sin(), z);
                integral += f(w);
            }
        }

        integral * domega
    }

    #[test]
    fn schlick_matches_f0_at_normal_incidence_and_one_at_grazing() {
        let f0 = Vec3::new(0.2, 0.5, 0.8);

        assert!(schlick_fresnel(f0, 1.0).abs_diff_eq(f0, 1.0e-6));
        assert!(schlick_fresnel(f0, 0.0).abs_diff_eq(Vec3::ONE, 1.0e-6));
    }

    #[test]
    fn smooth_conductor_behaves_like_delta_reflection() {
        let bsdf = ConductorGgxBsdf::new(Vec3::new(0.7, 0.5, 0.3), 1.0e-4, 1.0e-4);
        let wo = Vec3::new(0.3, -0.4, 0.8660254).normalize();

        let sample = bsdf
            .sample(wo, Vec2::splat(0.5))
            .expect("expected smooth reflection sample");

        assert!(
            sample
                .wi
                .abs_diff_eq(Vec3::new(-wo.x, -wo.y, wo.z).normalize(), 1.0e-6)
        );
        assert_eq!(sample.pdf, 1.0);
        assert_eq!(sample.flags, BsdfFlags::DELTA | BsdfFlags::REFLECTION);
        assert_eq!(bsdf.eval(wo, sample.wi), Vec3::ZERO);
        assert_eq!(bsdf.pdf(wo, sample.wi), 0.0);
    }

    #[test]
    fn rough_sample_matches_eval_cos_over_pdf() {
        let bsdf = ConductorGgxBsdf::new(Vec3::new(0.8, 0.6, 0.2), 0.35, 0.2);
        let wo = Vec3::new(0.2, -0.3, 0.9327379).normalize();

        let sample = bsdf
            .sample(wo, Vec2::new(0.37, 0.82))
            .expect("expected glossy reflection sample");

        let f = bsdf.eval(wo, sample.wi);
        let expected_weight = f * (sample.wi.z / sample.pdf);

        assert_eq!(sample.flags, BsdfFlags::GLOSSY | BsdfFlags::REFLECTION);
        assert!(sample.wi.z > 0.0);
        assert!(sample.pdf > 0.0);
        assert!(sample.weight.abs_diff_eq(expected_weight, 1.0e-5));
    }

    #[test]
    fn bounded_pdf_over_half_vectors_is_positive_for_valid_configuration() {
        let wo = Vec3::new(0.1, 0.2, 0.9746794).normalize();
        let wm = Vec3::new(0.05, -0.15, 0.9874209).normalize();
        let pdf = pdf_wm_bounded_vndf(wo, wm, 0.4, 0.25);

        assert!(pdf > 0.0);
    }

    #[test]
    fn eval_and_pdf_reject_lower_hemisphere_directions() {
        let bsdf = ConductorGgxBsdf::new(Vec3::ONE, 0.3, 0.3);

        assert_eq!(bsdf.eval(Vec3::Z, Vec3::NEG_Z), Vec3::ZERO);
        assert_eq!(bsdf.pdf(Vec3::Z, Vec3::NEG_Z), 0.0);
        assert!(bsdf.sample(Vec3::NEG_Z, Vec2::splat(0.5)).is_none());
    }

    #[test]
    fn ggx_ndf_is_normalized_over_projected_hemisphere() {
        let configs = [(0.2, 0.2), (0.35, 0.7), (0.8, 0.15)];

        for (alpha_x, alpha_y) in configs {
            let integral = integrate_hemisphere(|wm| ggx_d(wm, alpha_x, alpha_y) * wm.z);

            assert!(integral.is_finite());
            assert!(
                (integral - 1.0).abs() < 5.0e-3,
                "alpha_x={alpha_x}, alpha_y={alpha_y}, integral={integral}"
            );
        }
    }

    #[test]
    fn visible_normal_distribution_is_normalized() {
        let configs = [
            (Vec3::Z, 0.2, 0.2),
            (Vec3::new(0.3, -0.4, 0.8660254).normalize(), 0.35, 0.7),
            (Vec3::new(0.8, 0.0, 0.6).normalize(), 0.8, 0.15),
        ];

        for (wo, alpha_x, alpha_y) in configs {
            let integral =
                integrate_hemisphere(|wm| visible_normal_pdf_wm(wo, wm, alpha_x, alpha_y));

            assert!(integral.is_finite());
            assert!(
                (integral - 1.0).abs() < 5.0e-3,
                "wo={wo:?}, alpha_x={alpha_x}, alpha_y={alpha_y}, integral={integral}"
            );
        }
    }

    #[test]
    fn white_furnace_does_not_increase_energy() {
        let cases = [
            (Vec3::Z, 0.2, 0.2),
            (Vec3::new(0.3, -0.4, 0.8660254).normalize(), 0.35, 0.35),
            (Vec3::new(0.3, -0.4, 0.8660254).normalize(), 0.35, 0.7),
            (Vec3::new(0.8, 0.0, 0.6).normalize(), 0.7, 0.2),
        ];

        for (wo, alpha_x, alpha_y) in cases {
            let bsdf = ConductorGgxBsdf::new(Vec3::ONE, alpha_x, alpha_y);
            let reflected_energy = integrate_hemisphere_vec3(|wi| bsdf.eval(wo, wi) * wi.z);

            assert!(reflected_energy.is_finite());
            assert!(
                reflected_energy.min_element() >= -1.0e-4,
                "wo={wo:?}, alpha_x={alpha_x}, alpha_y={alpha_y}, reflected_energy={reflected_energy:?}"
            );
            assert!(
                reflected_energy.max_element() <= 1.0 + 5.0e-3,
                "wo={wo:?}, alpha_x={alpha_x}, alpha_y={alpha_y}, reflected_energy={reflected_energy:?}"
            );
        }
    }

    #[test]
    fn bounded_vndf_sampling_produces_finite_upper_hemisphere_samples_for_typical_inputs() {
        let cases = [
            (Vec3::new(0.3, -0.4, 0.8660254).normalize(), 0.2, 0.2),
            (Vec3::new(-0.2, 0.3, 0.9327379).normalize(), 0.35, 0.2),
        ];

        for (wo, alpha_x, alpha_y) in cases {
            let bsdf = ConductorGgxBsdf::new(Vec3::new(0.9, 0.7, 0.4), alpha_x, alpha_y);

            for y in 0..4 {
                for x in 0..4 {
                    let us = Vec2::new((x as f32 + 0.5) / 4.0, (y as f32 + 0.5) / 4.0);
                    let sample = bsdf
                        .sample(wo, us)
                        .expect("expected bounded VNDF sampling to produce a reflection sample");

                    assert!(sample.wi.is_finite());
                    assert!(sample.weight.is_finite());
                    assert!(sample.pdf.is_finite());
                    assert!(sample.wi.z > 0.0);
                    assert!(sample.pdf > 0.0);
                    assert_eq!(sample.flags, BsdfFlags::GLOSSY | BsdfFlags::REFLECTION);
                }
            }
        }
    }
}
