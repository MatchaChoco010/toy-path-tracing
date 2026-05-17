use std::f32::consts::PI;

use glam::{Vec2, Vec3};

use crate::math::{cosine_weighted_hemisphere_pdf, sample_cosine_weighted_hemisphere};

use super::{BsdfFlags, BsdfSample};

const MIN_SHEEN_ROUGHNESS: f32 = 0.07;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SheenBsdf {
    color: Vec3,
    inv_alpha: f32,
}

impl SheenBsdf {
    pub fn new(color: Vec3, roughness: f32) -> Self {
        let r = roughness.clamp(MIN_SHEEN_ROUGHNESS, 1.0);
        let alpha = r * r;
        Self {
            color,
            inv_alpha: 1.0 / alpha,
        }
    }

    pub fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return Vec3::ZERO;
        }
        let h = (wo + wi).normalize_or_zero();
        if h.length_squared() == 0.0 {
            return Vec3::ZERO;
        }
        let d = sheen_d(h.z, self.inv_alpha);
        let v = sheen_v(wo.z, wi.z);
        self.color * (d * v)
    }

    pub fn pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return 0.0;
        }
        cosine_weighted_hemisphere_pdf(wi.z)
    }

    pub fn sample(&self, wo: Vec3, us: Vec2) -> Option<BsdfSample> {
        if wo.z <= 0.0 {
            return None;
        }
        let wi = sample_cosine_weighted_hemisphere(us);
        let pdf = cosine_weighted_hemisphere_pdf(wi.z);
        if pdf <= 0.0 {
            return None;
        }
        let f = self.eval(wo, wi);
        let weight = f * (wi.z / pdf);
        Some(BsdfSample {
            weight,
            wi,
            pdf,
            flags: BsdfFlags::GLOSSY | BsdfFlags::REFLECTION,
            eta: 1.0,
            wavelength_lock: None,
        })
    }
}

fn sheen_d(cos_theta_h: f32, inv_alpha: f32) -> f32 {
    let cos2 = (cos_theta_h * cos_theta_h).clamp(0.0, 1.0);
    let sin2 = (1.0 - cos2).max(0.0);
    (2.0 + inv_alpha) * sin2.powf(0.5 * inv_alpha) / (2.0 * PI)
}

fn sheen_v(cos_o: f32, cos_i: f32) -> f32 {
    let denom = 4.0 * (cos_o + cos_i - cos_o * cos_i).max(1.0e-4);
    1.0 / denom
}

pub fn sheen_directional_albedo_estimate(roughness: f32, cos_theta_o: f32, samples: usize) -> f32 {
    let r = roughness.clamp(MIN_SHEEN_ROUGHNESS, 1.0);
    let alpha = r * r;
    let inv_alpha = 1.0 / alpha;
    let cos_o = cos_theta_o.clamp(0.0, 1.0);
    let sin_o = (1.0 - cos_o * cos_o).max(0.0).sqrt();
    let wo = Vec3::new(sin_o, 0.0, cos_o);
    if wo.z <= 0.0 {
        return 0.0;
    }
    let inv_pdf = 2.0 * PI;
    let mut acc = 0.0_f32;
    for index in 0..samples {
        let u = (index as f32 + 0.5) / samples as f32;
        let v = radical_inverse_vdc(index as u32);
        let z = u;
        let r_xy = (1.0 - z * z).max(0.0).sqrt();
        let phi = std::f32::consts::TAU * v;
        let wi = Vec3::new(r_xy * phi.cos(), r_xy * phi.sin(), z);
        if wi.z <= 0.0 {
            continue;
        }
        let h = (wo + wi).normalize_or_zero();
        if h.length_squared() == 0.0 {
            continue;
        }
        let d = sheen_d(h.z, inv_alpha);
        let v_term = sheen_v(wo.z, wi.z);
        acc += d * v_term * wi.z * inv_pdf;
    }
    (acc / samples as f32).clamp(0.0, 1.0)
}

fn radical_inverse_vdc(bits: u32) -> f32 {
    bits.reverse_bits() as f32 * 2.328_306_4e-10
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};

    use super::SheenBsdf;

    #[test]
    fn upper_hemisphere_only() {
        let bsdf = SheenBsdf::new(Vec3::ONE, 0.3);
        assert_eq!(bsdf.eval(Vec3::Z, -Vec3::Z), Vec3::ZERO);
        assert_eq!(bsdf.eval(-Vec3::Z, Vec3::Z), Vec3::ZERO);
    }

    #[test]
    fn sample_weight_matches_eval_cos_over_pdf() {
        let bsdf = SheenBsdf::new(Vec3::new(0.8, 0.6, 0.4), 0.4);
        let wo = Vec3::new(0.2, -0.1, 0.9746794).normalize();
        let sample = bsdf.sample(wo, Vec2::new(0.5, 0.7)).unwrap();
        let f = bsdf.eval(wo, sample.wi);
        let expected = f * (sample.wi.z / sample.pdf);
        assert!(sample.weight.abs_diff_eq(expected, 1.0e-4));
    }

    #[test]
    fn tangent_aligned_half_vector_dominates_normal_aligned() {
        let bsdf = SheenBsdf::new(Vec3::ONE, 0.3);
        let wo = Vec3::new(0.7, 0.0, (1.0_f32 - 0.49).sqrt()).normalize();
        let wi_retro = wo;
        let wi_normal = Vec3::Z;
        let f_retro = bsdf.eval(wo, wi_retro).x;
        let f_normal = bsdf.eval(wo, wi_normal).x;
        assert!(f_retro > f_normal);
    }
}
