use glam::{Vec2, Vec3};

use crate::math::schlick_fresnel;

use super::smith_ggx::{
    EFFECTIVELY_SMOOTH_ALPHA, MIN_ALPHA, ggx_g1, ggx_g2_height_correlated, is_upper_hemisphere,
    pdf_wm_bounded_vndf, pdf_wm_vndf, reflect_local, reflection_half_vector,
    sample_wm_bounded_vndf,
};
use super::{BsdfFlags, BsdfSample};

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

        let Some(wm) = reflection_half_vector(wo, wi) else {
            return Vec3::ZERO;
        };

        let cos_theta_o = wo.z.max(0.0);
        let cos_theta_i = wi.z.max(0.0);
        if cos_theta_o <= 0.0 || cos_theta_i <= 0.0 {
            return Vec3::ZERO;
        }

        let d = crate::bsdf::smith_ggx::ggx_d(wm, self.alpha_x, self.alpha_y);
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

        let Some(wm) = reflection_half_vector(wo, wi) else {
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
        let vndf_pdf_wm = pdf_wm_vndf(wo, wm, self.alpha_x, self.alpha_y);
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
            eta: 1.0,
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
            eta: 1.0,
        })
    }

    fn effectively_smooth(&self) -> bool {
        self.alpha_x.max(self.alpha_y) < EFFECTIVELY_SMOOTH_ALPHA
    }
}

fn reflection_jacobian(wo: Vec3, wm: Vec3) -> f32 {
    let denom = 4.0 * wo.dot(wm).abs();
    if denom <= 0.0 { 0.0 } else { 1.0 / denom }
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};

    use crate::bsdf::{BsdfFlags, ConductorGgxBsdf};
    use crate::math::schlick_fresnel;

    const HEMISPHERE_Z_SAMPLES: usize = 256;
    const HEMISPHERE_PHI_SAMPLES: usize = 256;

    fn integrate_hemisphere_vec3(f: impl Fn(Vec3) -> Vec3) -> Vec3 {
        let dz = 1.0 / HEMISPHERE_Z_SAMPLES as f32;
        let dphi = std::f32::consts::TAU / HEMISPHERE_PHI_SAMPLES as f32;
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
    fn eval_and_pdf_reject_lower_hemisphere_directions() {
        let bsdf = ConductorGgxBsdf::new(Vec3::ONE, 0.3, 0.3);

        assert_eq!(bsdf.eval(Vec3::Z, Vec3::NEG_Z), Vec3::ZERO);
        assert_eq!(bsdf.pdf(Vec3::Z, Vec3::NEG_Z), 0.0);
        assert!(bsdf.sample(Vec3::NEG_Z, Vec2::splat(0.5)).is_none());
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
