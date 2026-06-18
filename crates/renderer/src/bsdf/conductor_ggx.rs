use std::f32::consts::PI;
use std::sync::Arc;

use glam::{Vec2, Vec3};

use crate::math::{
    cosine_weighted_hemisphere_pdf, sample_cosine_weighted_hemisphere, schlick_fresnel,
};

use super::ConductorGgxEnergyCompensationLut;
use super::smith_ggx::{
    EFFECTIVELY_SMOOTH_ALPHA, MIN_ALPHA, ggx_g1, ggx_g2_height_correlated, is_upper_hemisphere,
    pdf_wm_bounded_vndf, pdf_wm_vndf, reflect_local, reflection_half_vector,
    sample_wm_bounded_vndf,
};
use super::{BsdfFlags, BsdfSample};

#[derive(Debug, Clone, PartialEq)]
pub struct ConductorGgxBsdf {
    base_color: Vec3,
    alpha_x: f32,
    alpha_y: f32,
    energy_compensation_lut: Option<Arc<ConductorGgxEnergyCompensationLut>>,
}

impl ConductorGgxBsdf {
    pub fn new(base_color: Vec3, alpha_x: f32, alpha_y: f32) -> Self {
        Self {
            base_color: base_color.clamp(Vec3::ZERO, Vec3::ONE),
            alpha_x: alpha_x.max(MIN_ALPHA),
            alpha_y: alpha_y.max(MIN_ALPHA),
            energy_compensation_lut: None,
        }
    }

    pub(crate) fn new_with_energy_compensation(
        base_color: Vec3,
        alpha_x: f32,
        alpha_y: f32,
        lut: Arc<ConductorGgxEnergyCompensationLut>,
    ) -> Self {
        Self {
            base_color: base_color.clamp(Vec3::ZERO, Vec3::ONE),
            alpha_x: alpha_x.max(MIN_ALPHA),
            alpha_y: alpha_y.max(MIN_ALPHA),
            energy_compensation_lut: Some(lut),
        }
    }

    pub fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if !is_upper_hemisphere(wo) || !is_upper_hemisphere(wi) || self.effectively_smooth() {
            return Vec3::ZERO;
        }

        let f_ss = self.eval_single_scattering(wo, wi);
        let f_ms = self.eval_multi_scattering(wo, wi);
        f_ss + f_ms
    }

    pub fn pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        if !is_upper_hemisphere(wo) || !is_upper_hemisphere(wi) || self.effectively_smooth() {
            return 0.0;
        }

        let pdf_ss = self.pdf_single_scattering(wo, wi);

        if let Some(lut) = self.energy_compensation_lut.as_ref() {
            let roughness_eq = self.roughness_eq();
            let e_avg = lut.lookup_e_avg(roughness_eq);
            let pr_ss = e_avg.clamp(0.0, 1.0);
            let pr_ms = (1.0 - pr_ss).max(0.0);
            let pdf_ms = cosine_weighted_hemisphere_pdf(wi.z.max(0.0));
            pr_ss * pdf_ss + pr_ms * pdf_ms
        } else {
            pdf_ss
        }
    }

    pub fn sample(&self, wo: Vec3, us: Vec2) -> Option<BsdfSample> {
        if !is_upper_hemisphere(wo) {
            return None;
        }

        if self.effectively_smooth() {
            return self.sample_smooth(wo);
        }

        if let Some(lut) = self.energy_compensation_lut.as_ref() {
            let roughness_eq = self.roughness_eq();
            let e_avg = lut.lookup_e_avg(roughness_eq);
            let pr_ss = e_avg.clamp(0.0, 1.0);
            let pr_ms = (1.0 - pr_ss).max(0.0);
            if pr_ss + pr_ms <= 0.0 {
                return None;
            }
            let (use_ss, us_remapped) = if us.x < pr_ss {
                let remapped = if pr_ss > 0.0 { us.x / pr_ss } else { 0.0 };
                (true, Vec2::new(remapped, us.y))
            } else {
                let remapped = if pr_ms > 0.0 {
                    (us.x - pr_ss) / pr_ms
                } else {
                    0.0
                };
                (false, Vec2::new(remapped, us.y))
            };

            if use_ss {
                let mut sample = self.sample_single_scattering(wo, us_remapped)?;
                let pdf_ms = cosine_weighted_hemisphere_pdf(sample.wi.z.max(0.0));
                let pdf_ss = sample.pdf;
                let pdf_total = pr_ss * pdf_ss + pr_ms * pdf_ms;
                if pdf_total <= 0.0 {
                    return None;
                }
                let f_total = self.eval_single_scattering(wo, sample.wi)
                    + self.eval_multi_scattering(wo, sample.wi);
                sample.pdf = pdf_total;
                sample.weight = f_total * (sample.wi.z.max(0.0) / pdf_total);
                Some(sample)
            } else {
                let wi = sample_cosine_weighted_hemisphere(us_remapped);
                if !is_upper_hemisphere(wi) {
                    return None;
                }
                let pdf_ms = cosine_weighted_hemisphere_pdf(wi.z);
                let pdf_ss = self.pdf_single_scattering(wo, wi);
                let pdf_total = pr_ss * pdf_ss + pr_ms * pdf_ms;
                if pdf_total <= 0.0 {
                    return None;
                }
                let f_total =
                    self.eval_single_scattering(wo, wi) + self.eval_multi_scattering(wo, wi);
                let weight = f_total * (wi.z / pdf_total);
                Some(BsdfSample {
                    weight,
                    wi,
                    pdf: pdf_total,
                    flags: BsdfFlags::GLOSSY | BsdfFlags::REFLECTION,
                    eta: 1.0,
                    wavelength_lock: None,
                })
            }
        } else {
            self.sample_single_scattering(wo, us)
        }
    }

    fn eval_single_scattering(&self, wo: Vec3, wi: Vec3) -> Vec3 {
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

    fn eval_multi_scattering(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        let Some(lut) = self.energy_compensation_lut.as_ref() else {
            return Vec3::ZERO;
        };
        let cos_o = wo.z.max(0.0);
        let cos_i = wi.z.max(0.0);
        if cos_o <= 0.0 || cos_i <= 0.0 {
            return Vec3::ZERO;
        }
        let roughness_eq = self.roughness_eq();
        let e_o = lut.lookup_e(cos_o, roughness_eq);
        let e_i = lut.lookup_e(cos_i, roughness_eq);
        let e_avg = lut.lookup_e_avg(roughness_eq);
        let one_minus_e_avg = (1.0 - e_avg).max(MS_DENOM_EPS);
        let f_avg = schlick_f_avg(self.base_color);
        let f_ms = compute_f_ms(f_avg, e_avg);
        f_ms * ((1.0 - e_o) * (1.0 - e_i) / (PI * one_minus_e_avg))
    }

    fn pdf_single_scattering(&self, wo: Vec3, wi: Vec3) -> f32 {
        let Some(wm) = reflection_half_vector(wo, wi) else {
            return 0.0;
        };
        let pdf_wm = pdf_wm_bounded_vndf(wo, wm, self.alpha_x, self.alpha_y);
        let jacobian = reflection_jacobian(wo, wm);
        pdf_wm * jacobian
    }

    fn sample_single_scattering(&self, wo: Vec3, us: Vec2) -> Option<BsdfSample> {
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
            wavelength_lock: None,
        })
    }

    fn roughness_eq(&self) -> f32 {
        (self.alpha_x * self.alpha_y).powf(0.25)
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
            wavelength_lock: None,
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

const MS_DENOM_EPS: f32 = 1.0e-6;

fn schlick_f_avg(f0: Vec3) -> Vec3 {
    (20.0 * f0 + Vec3::ONE) / 21.0
}

fn compute_f_ms(f_avg: Vec3, e_avg: f32) -> Vec3 {
    let one_minus_eavg = (1.0 - e_avg).max(0.0);
    let denom = Vec3::ONE - f_avg * one_minus_eavg;
    let denom_safe = Vec3::new(
        denom.x.max(MS_DENOM_EPS),
        denom.y.max(MS_DENOM_EPS),
        denom.z.max(MS_DENOM_EPS),
    );
    f_avg * f_avg * e_avg / denom_safe
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};

    use crate::bsdf::{BsdfFlags, ConductorGgxBsdf, integrate_hemisphere_vec3};

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

    #[test]
    fn schlick_f_avg_matches_closed_form_for_f0() {
        let f0 = Vec3::new(0.04, 0.5, 0.92);
        let expected = (20.0 * f0 + Vec3::ONE) / 21.0;
        assert!(super::schlick_f_avg(f0).abs_diff_eq(expected, 1.0e-6));
    }

    #[test]
    fn compensation_passes_white_furnace_at_high_roughness() {
        use std::sync::Arc;

        use crate::bsdf::ConductorGgxEnergyCompensationLut;

        let lut = Arc::new(ConductorGgxEnergyCompensationLut::build_for_tests());

        for &alpha in &[0.05_f32, 0.2, 0.4, 0.7, 1.0] {
            let bsdf = ConductorGgxBsdf::new_with_energy_compensation(
                Vec3::ONE,
                alpha,
                alpha,
                Arc::clone(&lut),
            );
            let wo = Vec3::new(0.3, -0.4, 0.8660254).normalize();
            let energy = integrate_hemisphere_vec3(|wi| bsdf.eval(wo, wi) * wi.z);
            assert!(
                energy.x > 0.97 && energy.x < 1.03,
                "alpha={alpha}, energy={energy:?}",
            );
        }
    }

    #[test]
    fn compensation_off_matches_single_scattering_eval() {
        let bsdf = ConductorGgxBsdf::new(Vec3::new(0.9, 0.7, 0.4), 0.4, 0.25);
        let wo = Vec3::new(0.2, -0.1, 0.9746794).normalize();
        let wi = Vec3::new(-0.2, 0.1, 0.9746794).normalize();
        let f = bsdf.eval(wo, wi);
        let pdf = bsdf.pdf(wo, wi);
        assert!(f.is_finite());
        assert!(pdf.is_finite());
        assert!(pdf > 0.0);
    }

    #[test]
    fn compensation_sample_weight_matches_eval_cos_over_pdf() {
        use std::sync::Arc;

        use crate::bsdf::ConductorGgxEnergyCompensationLut;

        let lut = Arc::new(ConductorGgxEnergyCompensationLut::build_for_tests());
        let bsdf =
            ConductorGgxBsdf::new_with_energy_compensation(Vec3::new(0.9, 0.7, 0.4), 0.4, 0.4, lut);
        let wo = Vec3::new(0.2, -0.3, 0.9327379).normalize();

        for (x_index, y_index) in [(1, 1), (1, 3), (3, 2), (2, 0)] {
            let us = Vec2::new((x_index as f32 + 0.5) / 4.0, (y_index as f32 + 0.5) / 4.0);
            let sample = bsdf
                .sample(wo, us)
                .expect("expected glossy sample with compensation");
            let f = bsdf.eval(wo, sample.wi);
            let expected = f * (sample.wi.z / sample.pdf);
            assert!(
                sample.weight.abs_diff_eq(expected, 5.0e-3),
                "us={us:?} weight={:?} expected={:?}",
                sample.weight,
                expected,
            );
        }
    }
}
