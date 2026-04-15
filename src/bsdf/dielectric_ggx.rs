use glam::{Vec2, Vec3};

use crate::math::{fresnel_dielectric, refract};

use super::smith_ggx::{
    EFFECTIVELY_SMOOTH_ALPHA, MIN_ALPHA, ggx_d, ggx_g1, ggx_g2_height_correlated,
    is_upper_hemisphere, pdf_wm_vndf, reflect_local, reflection_half_vector, sample_wm_vndf,
};
use super::{BsdfFlags, BsdfSample};

const DENOM_EPS: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DielectricGgxBsdf {
    color: Vec3,
    eta: f32,
    alpha_x: f32,
    alpha_y: f32,
    thin: bool,
    front_face: bool,
}

impl DielectricGgxBsdf {
    pub fn new(
        color: Vec3,
        eta: f32,
        alpha_x: f32,
        alpha_y: f32,
        thin: bool,
        front_face: bool,
    ) -> Self {
        Self {
            color,
            eta,
            alpha_x: alpha_x.max(MIN_ALPHA),
            alpha_y: alpha_y.max(MIN_ALPHA),
            thin,
            front_face,
        }
    }

    fn effectively_smooth(&self) -> bool {
        self.alpha_x.max(self.alpha_y) < EFFECTIVELY_SMOOTH_ALPHA
    }

    fn fresnel_interface(&self) -> (f32, f32) {
        if self.front_face {
            (1.0, self.eta)
        } else {
            (self.eta, 1.0)
        }
    }

    fn eta_rel(&self) -> f32 {
        // eta_rel = n_o / n_i; matches math::refract's `eta` parameter convention.
        if self.front_face {
            1.0 / self.eta
        } else {
            self.eta
        }
    }

    pub fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if !is_upper_hemisphere(wo) || self.eta <= 0.0 {
            return Vec3::ZERO;
        }
        if self.thin || self.effectively_smooth() {
            return Vec3::ZERO;
        }

        let (eta_i, eta_t) = self.fresnel_interface();
        let eta_rel = self.eta_rel();

        if wo.z * wi.z > 0.0 {
            // Reflection branch.
            let Some(wm) = reflection_half_vector(wo, wi) else {
                return Vec3::ZERO;
            };
            let cos_wo_wm = wo.dot(wm);
            if cos_wo_wm <= 0.0 {
                return Vec3::ZERO;
            }
            let cos_o = wo.z.abs();
            let cos_i = wi.z.abs();
            if cos_o <= 0.0 || cos_i <= 0.0 {
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
            let f = fresnel_dielectric(cos_wo_wm, eta_i, eta_t);
            Vec3::splat(d * g * f / (4.0 * cos_o * cos_i))
        } else if wo.z * wi.z < 0.0 {
            // Transmission branch. Generalized half vector.
            let wm_unnorm = eta_rel * wo + wi;
            if wm_unnorm.length_squared() < 1.0e-12 {
                return Vec3::ZERO;
            }
            let mut wm = wm_unnorm.normalize();
            if wm.z < 0.0 {
                wm = -wm;
            }
            let cos_wo_wm = wo.dot(wm);
            if cos_wo_wm <= 0.0 {
                return Vec3::ZERO;
            }
            let cos_wi_wm = wi.dot(wm);
            let den = cos_wi_wm + eta_rel * cos_wo_wm;
            if den.abs() < DENOM_EPS {
                return Vec3::ZERO;
            }
            let cos_o = wo.z.abs();
            let cos_i = wi.z.abs();
            if cos_o <= 0.0 || cos_i <= 0.0 {
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
            let f = fresnel_dielectric(cos_wo_wm, eta_i, eta_t);
            let radiance_scale = 1.0 / (eta_rel * eta_rel);
            let numerator = d * (1.0 - f) * g * (cos_wi_wm * cos_wo_wm).abs();
            let denom = den * den * cos_o * cos_i;
            if denom <= 0.0 {
                return Vec3::ZERO;
            }
            self.color * (radiance_scale * numerator / denom)
        } else {
            Vec3::ZERO
        }
    }

    pub fn pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        if !is_upper_hemisphere(wo) || self.eta <= 0.0 {
            return 0.0;
        }
        if self.thin || self.effectively_smooth() {
            return 0.0;
        }

        let (eta_i, eta_t) = self.fresnel_interface();
        let eta_rel = self.eta_rel();

        if wo.z * wi.z > 0.0 {
            // Reflection.
            let Some(wm) = reflection_half_vector(wo, wi) else {
                return 0.0;
            };
            let cos_wo_wm = wo.dot(wm);
            if cos_wo_wm <= 0.0 {
                return 0.0;
            }
            let f = fresnel_dielectric(cos_wo_wm, eta_i, eta_t);
            let pdf_wm = pdf_wm_vndf(wo, wm, self.alpha_x, self.alpha_y);
            if pdf_wm <= 0.0 {
                return 0.0;
            }
            f * pdf_wm / (4.0 * cos_wo_wm)
        } else if wo.z * wi.z < 0.0 {
            // Transmission.
            let mut wm_unnorm = eta_rel * wo + wi;
            if wm_unnorm.length_squared() < 1.0e-12 {
                return 0.0;
            }
            let mut wm = wm_unnorm.normalize();
            if wm.z < 0.0 {
                wm = -wm;
                wm_unnorm = -wm_unnorm;
            }
            let _ = wm_unnorm;
            let cos_wo_wm = wo.dot(wm);
            if cos_wo_wm <= 0.0 {
                return 0.0;
            }
            let cos_wi_wm = wi.dot(wm);
            let den = cos_wi_wm + eta_rel * cos_wo_wm;
            if den.abs() < DENOM_EPS {
                return 0.0;
            }
            let f = fresnel_dielectric(cos_wo_wm, eta_i, eta_t);
            let pdf_wm = pdf_wm_vndf(wo, wm, self.alpha_x, self.alpha_y);
            if pdf_wm <= 0.0 {
                return 0.0;
            }
            (1.0 - f) * pdf_wm * cos_wi_wm.abs() / (den * den)
        } else {
            0.0
        }
    }

    pub fn sample(&self, wo: Vec3, uc: f32, us: Vec2) -> Option<BsdfSample> {
        if !is_upper_hemisphere(wo) || self.eta <= 0.0 {
            return None;
        }

        if self.thin {
            return self.sample_thin_delta(wo, uc);
        }

        if self.effectively_smooth() {
            return self.sample_smooth_delta(wo, uc);
        }

        self.sample_rough(wo, uc, us)
    }

    fn sample_thin_delta(&self, wo: Vec3, uc: f32) -> Option<BsdfSample> {
        let mut reflectance = fresnel_dielectric(wo.z.abs(), 1.0, self.eta);
        let transmittance = 1.0 - reflectance;
        let denom = 1.0 - reflectance * reflectance;

        if denom <= 0.0 {
            reflectance = 1.0;
        } else {
            reflectance += transmittance * transmittance * reflectance / denom;
        }

        let transmittance = (1.0 - reflectance).max(0.0);
        let (pr, pt) = normalized_probabilities(reflectance, transmittance)?;

        let reflect = uc < pr;
        let (wi, pdf, weight, flags) = if reflect {
            (
                reflected_direction(wo),
                pr,
                Vec3::ONE,
                BsdfFlags::DELTA | BsdfFlags::REFLECTION,
            )
        } else {
            (
                -wo,
                pt,
                self.color,
                BsdfFlags::DELTA | BsdfFlags::TRANSMISSION,
            )
        };

        Some(BsdfSample {
            weight,
            wi,
            pdf,
            flags,
        })
    }

    fn sample_smooth_delta(&self, wo: Vec3, uc: f32) -> Option<BsdfSample> {
        let (eta_i, eta_t) = self.fresnel_interface();
        let eta_rel = self.eta_rel();
        let reflectance = fresnel_dielectric(wo.z.abs(), eta_i, eta_t);
        let transmission_direction = refract(wo, eta_rel);
        let transmittance = if transmission_direction.is_some() {
            1.0 - reflectance
        } else {
            0.0
        };
        let (pr, pt) = normalized_probabilities(reflectance, transmittance)?;

        if uc < pr {
            return Some(BsdfSample {
                weight: Vec3::ONE,
                wi: reflected_direction(wo),
                pdf: pr,
                flags: BsdfFlags::DELTA | BsdfFlags::REFLECTION,
            });
        }

        let wi = transmission_direction?;
        let radiance_scale = 1.0 / (eta_rel * eta_rel);

        Some(BsdfSample {
            weight: self.color * radiance_scale,
            wi,
            pdf: pt,
            flags: BsdfFlags::DELTA | BsdfFlags::TRANSMISSION,
        })
    }

    fn sample_rough(&self, wo: Vec3, uc: f32, us: Vec2) -> Option<BsdfSample> {
        let (eta_i, eta_t) = self.fresnel_interface();
        let eta_rel = self.eta_rel();

        let wm = sample_wm_vndf(wo, self.alpha_x, self.alpha_y, us)?;
        let cos_wo_wm = wo.dot(wm);
        if cos_wo_wm <= 0.0 {
            return None;
        }
        let f = fresnel_dielectric(cos_wo_wm, eta_i, eta_t);
        let g1 = ggx_g1(wo, self.alpha_x, self.alpha_y);
        if g1 <= 0.0 {
            return None;
        }

        if uc < f {
            // Reflection branch.
            let wi = reflect_local(wo, wm);
            if !is_upper_hemisphere(wi) {
                return None;
            }
            let g2 = ggx_g2_height_correlated(wo, wi, self.alpha_x, self.alpha_y);
            let pdf_wm = pdf_wm_vndf(wo, wm, self.alpha_x, self.alpha_y);
            if pdf_wm <= 0.0 {
                return None;
            }
            let pdf = f * pdf_wm / (4.0 * cos_wo_wm);
            if pdf <= 0.0 {
                return None;
            }
            let weight = Vec3::splat(g2 / g1);
            Some(BsdfSample {
                weight,
                wi,
                pdf,
                flags: BsdfFlags::GLOSSY | BsdfFlags::REFLECTION,
            })
        } else {
            // Transmission branch.
            let wi = refract_about_wm(wo, wm, eta_rel)?;
            if wi.z >= 0.0 {
                return None;
            }
            let cos_wi_wm = wi.dot(wm);
            let den = cos_wi_wm + eta_rel * cos_wo_wm;
            if den.abs() < DENOM_EPS {
                return None;
            }
            let g2 = ggx_g2_height_correlated(wo, wi, self.alpha_x, self.alpha_y);
            let pdf_wm = pdf_wm_vndf(wo, wm, self.alpha_x, self.alpha_y);
            if pdf_wm <= 0.0 {
                return None;
            }
            let pdf = (1.0 - f) * pdf_wm * cos_wi_wm.abs() / (den * den);
            if pdf <= 0.0 {
                return None;
            }
            let radiance_scale = 1.0 / (eta_rel * eta_rel);
            let weight = self.color * (radiance_scale * g2 / g1);
            Some(BsdfSample {
                weight,
                wi,
                pdf,
                flags: BsdfFlags::GLOSSY | BsdfFlags::TRANSMISSION,
            })
        }
    }
}

fn reflected_direction(wo: Vec3) -> Vec3 {
    Vec3::new(-wo.x, -wo.y, wo.z).normalize_or_zero()
}

fn normalized_probabilities(reflectance: f32, transmittance: f32) -> Option<(f32, f32)> {
    let reflection = reflectance.max(0.0);
    let transmission = transmittance.max(0.0);
    let sum = reflection + transmission;
    if sum <= 0.0 {
        return None;
    }
    Some((reflection / sum, transmission / sum))
}

fn refract_about_wm(wo: Vec3, wm: Vec3, eta: f32) -> Option<Vec3> {
    let cos_o = wo.dot(wm);
    if cos_o <= 0.0 {
        return None;
    }
    let sin2_t = eta * eta * (1.0 - cos_o * cos_o).max(0.0);
    if sin2_t >= 1.0 {
        return None;
    }
    let cos_t = (1.0 - sin2_t).max(0.0).sqrt();
    let wi = (-eta * wo + (eta * cos_o - cos_t) * wm).normalize_or_zero();
    if wi.length_squared() == 0.0 {
        return None;
    }
    Some(wi)
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};

    use crate::{
        bsdf::{BsdfFlags, DielectricGgxBsdf},
        math::{fresnel_dielectric, refract},
    };

    #[test]
    fn effectively_smooth_returns_delta_reflection_at_normal_incidence() {
        let bsdf = DielectricGgxBsdf::new(Vec3::ONE, 1.5, 1.0e-4, 1.0e-4, false, true);
        let wo = Vec3::Z;
        let sample = bsdf
            .sample(wo, 0.01, Vec2::splat(0.5))
            .expect("expected a delta reflection sample");

        assert_eq!(sample.wi, Vec3::Z);
        assert_eq!(sample.weight, Vec3::ONE);
        assert!((sample.pdf - 0.04).abs() < 1.0e-4);
        assert_eq!(sample.flags, BsdfFlags::DELTA | BsdfFlags::REFLECTION);
        assert_eq!(bsdf.eval(wo, sample.wi), Vec3::ZERO);
        assert_eq!(bsdf.pdf(wo, sample.wi), 0.0);
    }

    #[test]
    fn effectively_smooth_returns_delta_transmission_with_radiance_scaling() {
        let color = Vec3::new(0.3, 0.5, 0.7);
        let eta = 1.5;
        let bsdf = DielectricGgxBsdf::new(color, eta, 1.0e-4, 1.0e-4, false, true);
        let wo = Vec3::new(0.3, -0.4, 0.8660254).normalize();
        let sample = bsdf
            .sample(wo, 0.99, Vec2::splat(0.5))
            .expect("expected a delta transmission sample");
        let expected_wi = refract(wo, 1.0 / eta).expect("expected refraction");

        assert!(sample.wi.abs_diff_eq(expected_wi, 1.0e-6));
        assert!(sample.weight.abs_diff_eq(color * (eta * eta), 1.0e-6));
        assert_eq!(sample.flags, BsdfFlags::DELTA | BsdfFlags::TRANSMISSION);
        assert!((sample.pdf - (1.0 - fresnel_dielectric(wo.z, 1.0, eta))).abs() < 1.0e-5);
    }

    #[test]
    fn effectively_smooth_exiting_glass_scales_radiance_down() {
        let color = Vec3::new(0.3, 0.5, 0.7);
        let eta = 1.5;
        let bsdf = DielectricGgxBsdf::new(color, eta, 1.0e-4, 1.0e-4, false, false);
        let wo = Vec3::Z;
        let sample = bsdf
            .sample(wo, 0.99, Vec2::splat(0.5))
            .expect("expected a delta transmission sample");

        assert!(sample.weight.abs_diff_eq(color / (eta * eta), 1.0e-6));
        assert_eq!(sample.wi, -Vec3::Z);
        assert_eq!(sample.flags, BsdfFlags::DELTA | BsdfFlags::TRANSMISSION);
    }

    #[test]
    fn effectively_smooth_falls_back_to_reflection_under_total_internal_reflection() {
        let bsdf = DielectricGgxBsdf::new(Vec3::ONE, 1.5, 1.0e-4, 1.0e-4, false, false);
        let wo = Vec3::new(0.8, 0.0, 0.6).normalize();
        let sample = bsdf
            .sample(wo, 0.9, Vec2::splat(0.5))
            .expect("expected total internal reflection");

        assert!(sample.wi.abs_diff_eq(Vec3::new(-0.8, 0.0, 0.6), 1.0e-6));
        assert_eq!(sample.weight, Vec3::ONE);
        assert_eq!(sample.pdf, 1.0);
        assert_eq!(sample.flags, BsdfFlags::DELTA | BsdfFlags::REFLECTION);
    }

    #[test]
    fn thin_uses_aggregate_reflectance_and_flips_direction_on_transmission() {
        let color = Vec3::new(0.3, 0.5, 0.7);
        let eta = 1.5;
        let bsdf = DielectricGgxBsdf::new(color, eta, 0.1, 0.1, true, true);
        let wo = Vec3::new(0.3, -0.4, 0.8660254).normalize();
        let sample = bsdf
            .sample(wo, 0.9, Vec2::splat(0.5))
            .expect("expected a thin transmission sample");
        let base = fresnel_dielectric(wo.z, 1.0, eta);
        let expected_reflectance = base + (1.0 - base).powi(2) * base / (1.0 - base * base);

        assert!(sample.wi.abs_diff_eq(-wo, 1.0e-6));
        assert!(sample.weight.abs_diff_eq(color, 1.0e-6));
        assert_eq!(sample.flags, BsdfFlags::DELTA | BsdfFlags::TRANSMISSION);
        assert!((sample.pdf - (1.0 - expected_reflectance)).abs() < 1.0e-5);
    }

    #[test]
    fn thin_ignores_roughness_and_stays_delta() {
        let bsdf = DielectricGgxBsdf::new(Vec3::ONE, 1.5, 0.9, 0.9, true, true);
        let wo = Vec3::new(0.2, -0.1, 0.9746794).normalize();
        let sample = bsdf
            .sample(wo, 0.5, Vec2::new(0.3, 0.7))
            .expect("expected a thin delta sample");

        assert!(sample.flags.contains(BsdfFlags::DELTA));
        assert!(!sample.flags.contains(BsdfFlags::GLOSSY));
    }

    #[test]
    fn rough_reflection_sample_matches_eval_cos_over_pdf() {
        let bsdf = DielectricGgxBsdf::new(Vec3::ONE, 1.5, 0.3, 0.2, false, true);
        let wo = Vec3::new(0.2, -0.3, 0.9327379).normalize();

        // With uc = 0.0 we always pick the reflection branch.
        let sample = bsdf
            .sample(wo, 0.0, Vec2::new(0.35, 0.72))
            .expect("expected a reflection sample");

        assert!(
            sample
                .flags
                .contains(BsdfFlags::GLOSSY | BsdfFlags::REFLECTION)
        );
        assert!(sample.wi.z > 0.0);
        assert!(sample.pdf > 0.0);

        let f = bsdf.eval(wo, sample.wi);
        let expected = f * (sample.wi.z.abs() / sample.pdf);
        assert!(sample.weight.abs_diff_eq(expected, 5.0e-4));
    }

    #[test]
    fn rough_transmission_sample_matches_eval_cos_over_pdf() {
        let bsdf = DielectricGgxBsdf::new(Vec3::new(0.5, 0.7, 0.9), 1.5, 0.3, 0.2, false, true);
        let wo = Vec3::new(0.2, -0.3, 0.9327379).normalize();

        // With uc = 1.0 we always pick the transmission branch.
        let sample = bsdf
            .sample(wo, 0.999, Vec2::new(0.4, 0.6))
            .expect("expected a transmission sample");

        assert!(
            sample
                .flags
                .contains(BsdfFlags::GLOSSY | BsdfFlags::TRANSMISSION)
        );
        assert!(sample.wi.z < 0.0);
        assert!(sample.pdf > 0.0);

        let f = bsdf.eval(wo, sample.wi);
        let expected = f * (sample.wi.z.abs() / sample.pdf);
        assert!(sample.weight.abs_diff_eq(expected, 5.0e-4));
    }

    #[test]
    fn sample_returns_none_for_lower_hemisphere_wo() {
        let bsdf = DielectricGgxBsdf::new(Vec3::ONE, 1.5, 0.3, 0.3, false, true);

        assert!(bsdf.sample(-Vec3::Z, 0.5, Vec2::splat(0.5)).is_none());
        assert!(
            DielectricGgxBsdf::new(Vec3::ONE, 0.0, 0.3, 0.3, false, true)
                .sample(Vec3::Z, 0.5, Vec2::splat(0.5))
                .is_none()
        );
    }

    #[test]
    fn eval_returns_zero_when_wi_is_on_wrong_side_of_expected_event() {
        let bsdf = DielectricGgxBsdf::new(Vec3::ONE, 1.5, 0.3, 0.3, false, true);
        // wi == wo.z small positive but non-physical reflection pair; eval uses
        // generalized half vector logic and should not produce energy.
        assert_eq!(bsdf.eval(Vec3::Z, Vec3::ZERO), Vec3::ZERO);
    }
}
