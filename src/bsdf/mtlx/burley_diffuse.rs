use std::f32::consts::PI;

use glam::{Vec2, Vec3};

use crate::bsdf::{BsdfFlags, OrenNayarBsdf};

use super::closure::MtlxLobeSample;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BurleyDiffuseBsdf {
    pub weight: f32,
    pub color: Vec3,
    pub roughness: f32,
}

impl BurleyDiffuseBsdf {
    pub fn new(weight: f32, color: Vec3, roughness: f32) -> Self {
        Self {
            weight: weight.clamp(0.0, 1.0),
            color: color.max(Vec3::ZERO),
            roughness: roughness.clamp(0.0, 1.0),
        }
    }

    pub fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return Vec3::ZERO;
        }
        let weighted = self.color * self.weight;
        if self.roughness <= 1.0e-6 {
            weighted / PI
        } else {
            OrenNayarBsdf::new(weighted, self.roughness).eval(wo, wi)
        }
    }

    pub fn pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return 0.0;
        }
        wi.z / PI
    }

    pub fn sample(&self, wo: Vec3, us: Vec2) -> Option<MtlxLobeSample> {
        if wo.z <= 0.0 {
            return None;
        }
        let phi = 2.0 * PI * us.x;
        let cos_theta = us.y.sqrt();
        let sin_theta = (1.0 - us.y).max(0.0).sqrt();
        let wi = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);
        let pdf = wi.z / PI;
        if pdf <= 0.0 {
            return None;
        }
        let f = self.eval(wo, wi);
        Some(MtlxLobeSample {
            weight: f * wi.z / pdf,
            wi_local: wi,
            pdf,
            flags: BsdfFlags::DIFFUSE | BsdfFlags::REFLECTION,
            eta: 1.0,
        })
    }

    pub fn directional_albedo(&self, _wo: Vec3) -> Vec3 {
        self.color * self.weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upper_hemisphere_only() {
        let b = BurleyDiffuseBsdf::new(1.0, Vec3::ONE, 0.5);
        assert_eq!(b.eval(Vec3::Z, -Vec3::Z), Vec3::ZERO);
    }

    #[test]
    fn smooth_lambert_limit_at_zero_roughness_normal_incidence() {
        let b = BurleyDiffuseBsdf::new(1.0, Vec3::splat(0.6), 0.0);
        let v = b.eval(Vec3::Z, Vec3::Z);
        assert!((v.x - 0.6 / PI).abs() < 1.0e-6);
    }

    #[test]
    fn sample_weight_matches_f_cos_over_pdf() {
        let b = BurleyDiffuseBsdf::new(1.0, Vec3::splat(0.6), 0.5);
        let wo = Vec3::new(0.0, 0.0, 1.0);
        let s = b.sample(wo, Vec2::new(0.3, 0.7)).unwrap();
        let f = b.eval(wo, s.wi_local);
        let expected = f * s.wi_local.z / s.pdf;
        assert!(s.weight.abs_diff_eq(expected, 1.0e-5));
    }

    #[test]
    fn rough_burley_matches_mdl_diffuse_reflection_layer() {
        let b = BurleyDiffuseBsdf::new(0.8, Vec3::new(0.6, 0.4, 0.2), 0.65);
        let wo = Vec3::new(0.2, -0.3, 0.9327379).normalize();
        let wi = Vec3::new(-0.25, 0.4, 0.881759).normalize();
        let expected = OrenNayarBsdf::new(b.color * b.weight, b.roughness).eval(wo, wi);
        assert!(b.eval(wo, wi).abs_diff_eq(expected, 1.0e-6));
    }
}
