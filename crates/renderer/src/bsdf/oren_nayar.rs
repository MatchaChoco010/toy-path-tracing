use std::f32::consts::PI;

use glam::{Vec2, Vec3};

use crate::math::{cosine_weighted_hemisphere_pdf, sample_cosine_weighted_hemisphere};

use super::{BsdfFlags, BsdfSample};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrenNayarBsdf {
    rho: Vec3,
    a: f32,
    b: f32,
}

impl OrenNayarBsdf {
    pub fn new(rho: Vec3, roughness: f32) -> Self {
        let sigma = roughness.clamp(0.0, 1.0);
        let sigma2 = sigma * sigma;
        let a = 1.0 - 0.5 * sigma2 / (sigma2 + 0.33);
        let b = 0.45 * sigma2 / (sigma2 + 0.09);
        Self { rho, a, b }
    }

    pub fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return Vec3::ZERO;
        }
        self.rho * (self.shape(wo, wi) / PI)
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
        let weight = self.rho * self.shape(wo, wi);
        Some(BsdfSample {
            weight,
            wi,
            pdf,
            pdf_rev: self.pdf(wi, wo),
            flags: BsdfFlags::DIFFUSE | BsdfFlags::REFLECTION,
            eta: 1.0,
            wavelength_lock: None,
        })
    }

    fn shape(&self, wo: Vec3, wi: Vec3) -> f32 {
        let sin_theta_o = (1.0 - wo.z * wo.z).max(0.0).sqrt();
        let sin_theta_i = (1.0 - wi.z * wi.z).max(0.0).sqrt();
        let cos_phi_diff = if sin_theta_o > 1.0e-4 && sin_theta_i > 1.0e-4 {
            let inv = 1.0 / (sin_theta_o * sin_theta_i);
            (wo.x * wi.x + wo.y * wi.y) * inv
        } else {
            0.0
        };
        let cos_phi_diff = cos_phi_diff.clamp(-1.0, 1.0).max(0.0);

        let (sin_alpha, tan_beta) = if wo.z.abs() < wi.z.abs() {
            let cos_beta = wi.z.abs().max(1.0e-4);
            (sin_theta_o, sin_theta_i / cos_beta)
        } else {
            let cos_beta = wo.z.abs().max(1.0e-4);
            (sin_theta_i, sin_theta_o / cos_beta)
        };

        self.a + self.b * cos_phi_diff * sin_alpha * tan_beta
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use glam::{Vec2, Vec3};

    use super::OrenNayarBsdf;

    #[test]
    fn lambert_reduction_when_roughness_zero() {
        let bsdf = OrenNayarBsdf::new(Vec3::splat(0.7), 0.0);
        let wo = Vec3::new(0.2, 0.3, 0.9327379).normalize();
        let wi = Vec3::new(-0.1, 0.4, 0.910).normalize();
        let f = bsdf.eval(wo, wi);
        assert!(f.abs_diff_eq(Vec3::splat(0.7) / PI, 1.0e-5));
    }

    #[test]
    fn upper_hemisphere_only() {
        let bsdf = OrenNayarBsdf::new(Vec3::ONE, 0.5);
        assert_eq!(bsdf.eval(Vec3::Z, -Vec3::Z), Vec3::ZERO);
        assert_eq!(bsdf.eval(-Vec3::Z, Vec3::Z), Vec3::ZERO);
        assert_eq!(bsdf.pdf(Vec3::Z, -Vec3::Z), 0.0);
    }

    #[test]
    fn sample_weight_matches_eval_cos_over_pdf() {
        let bsdf = OrenNayarBsdf::new(Vec3::new(0.7, 0.5, 0.3), 0.6);
        let wo = Vec3::new(0.2, -0.1, 0.9746794).normalize();
        let sample = bsdf.sample(wo, Vec2::new(0.37, 0.82)).unwrap();
        let f = bsdf.eval(wo, sample.wi);
        let expected = f * (sample.wi.z / sample.pdf);
        assert!(sample.weight.abs_diff_eq(expected, 1.0e-5));
    }

    #[test]
    fn energy_below_unity_in_white_furnace() {
        let bsdf = OrenNayarBsdf::new(Vec3::ONE, 1.0);
        let wo = Vec3::new(0.0, 0.0, 1.0);
        let dz = 1.0 / 64.0;
        let dphi = std::f32::consts::TAU / 64.0;
        let mut energy = Vec3::ZERO;
        for zi in 0..64 {
            let z = (zi as f32 + 0.5) * dz;
            let r = (1.0 - z * z).max(0.0).sqrt();
            for pi in 0..64 {
                let phi = (pi as f32 + 0.5) * dphi;
                let wi = Vec3::new(r * phi.cos(), r * phi.sin(), z);
                energy += bsdf.eval(wo, wi) * wi.z * dz * dphi;
            }
        }
        assert!(energy.max_element() <= 1.0 + 1.0e-3);
    }
}
