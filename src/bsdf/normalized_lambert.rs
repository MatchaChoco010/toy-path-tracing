use std::f32::consts::PI;

use glam::{Vec2, Vec3};

use crate::math::{cosine_weighted_hemisphere_pdf, sample_cosine_weighted_hemisphere};

use super::{BsdfFlags, BsdfSample};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizedLambertBsdf {
    rho: Vec3,
}

impl NormalizedLambertBsdf {
    pub fn new(rho: Vec3) -> Self {
        Self { rho }
    }

    pub fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return Vec3::ZERO;
        }

        self.rho / PI
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
        let pdf = self.pdf(wo, wi);

        if pdf <= 0.0 {
            return None;
        }

        // For cosine-weighted Lambert sampling, f * cos(theta) / pdf simplifies to rho.
        let weight = self.rho;

        Some(BsdfSample {
            weight,
            wi,
            pdf,
            flags: BsdfFlags::DIFFUSE | BsdfFlags::REFLECTION,
            eta: 1.0,
        })
    }
}
#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use glam::{Vec2, Vec3};

    use crate::bsdf::{BsdfFlags, NormalizedLambertBsdf};

    #[test]
    fn eval_returns_rho_over_pi_for_upper_hemisphere_directions() {
        let bsdf = NormalizedLambertBsdf::new(Vec3::new(0.3, 0.5, 0.7));

        let f = bsdf.eval(Vec3::Z, Vec3::new(0.2, 0.3, 0.9327379).normalize());

        assert!(f.abs_diff_eq(Vec3::new(0.3, 0.5, 0.7) / PI, 1.0e-6));
    }

    #[test]
    fn eval_returns_zero_for_lower_hemisphere_directions() {
        let bsdf = NormalizedLambertBsdf::new(Vec3::ONE);

        assert_eq!(bsdf.eval(Vec3::Z, -Vec3::Z), Vec3::ZERO);
        assert_eq!(bsdf.eval(-Vec3::Z, Vec3::Z), Vec3::ZERO);
    }

    #[test]
    fn pdf_matches_cosine_weighted_hemisphere_density() {
        let bsdf = NormalizedLambertBsdf::new(Vec3::ONE);
        let wi = Vec3::new(0.2, 0.3, 0.9327379).normalize();

        let pdf = bsdf.pdf(Vec3::Z, wi);

        assert!((pdf - wi.z / PI).abs() < 1.0e-6);
    }

    #[test]
    fn pdf_returns_zero_for_lower_hemisphere_directions() {
        let bsdf = NormalizedLambertBsdf::new(Vec3::ONE);

        assert_eq!(bsdf.pdf(Vec3::Z, -Vec3::Z), 0.0);
        assert_eq!(bsdf.pdf(-Vec3::Z, Vec3::Z), 0.0);
    }

    #[test]
    fn cosine_weighted_sample_returns_rho_as_weight() {
        let bsdf = NormalizedLambertBsdf::new(Vec3::new(0.3, 0.5, 0.7));

        let sample = bsdf
            .sample(Vec3::Z, Vec2::new(0.25, 0.75))
            .expect("expected a valid sample");

        assert!(sample.weight.abs_diff_eq(Vec3::new(0.3, 0.5, 0.7), 1.0e-6));
        assert!(sample.pdf > 0.0);
        assert!(sample.wi.z > 0.0);
        assert_eq!(sample.flags, BsdfFlags::DIFFUSE | BsdfFlags::REFLECTION);
    }

    #[test]
    fn sample_returns_none_for_lower_hemisphere_outgoing_direction() {
        let bsdf = NormalizedLambertBsdf::new(Vec3::ONE);

        assert!(bsdf.sample(-Vec3::Z, Vec2::splat(0.5)).is_none());
    }
}
