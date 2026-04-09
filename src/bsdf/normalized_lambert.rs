use std::f32::consts::PI;

use glam::{Vec2, Vec3};

use crate::math::{cosine_weighted_hemisphere_pdf, sample_cosine_weighted_hemisphere};

use super::BsdfSample;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizedLambertBsdf {
    rho: Vec3,
}

impl NormalizedLambertBsdf {
    pub fn new(rho: Vec3) -> Self {
        Self { rho }
    }

    pub fn sample(&self, wo: Vec3, us: Vec2) -> Option<BsdfSample> {
        if wo.z <= 0.0 {
            return None;
        }

        let wi = sample_cosine_weighted_hemisphere(us);
        let cos_theta = wi.z.max(0.0);
        let pdf = cosine_weighted_hemisphere_pdf(cos_theta);

        if pdf <= 0.0 {
            return None;
        }

        let f = self.rho / PI;
        let weight = f * (cos_theta / pdf);

        Some(BsdfSample { weight, wi, pdf })
    }
}
#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};

    use super::NormalizedLambertBsdf;

    #[test]
    fn cosine_weighted_sample_returns_rho_as_weight() {
        let bsdf = NormalizedLambertBsdf::new(Vec3::new(0.3, 0.5, 0.7));

        let sample = bsdf
            .sample(Vec3::Z, Vec2::new(0.25, 0.75))
            .expect("expected a valid sample");

        assert!(sample.weight.abs_diff_eq(Vec3::new(0.3, 0.5, 0.7), 1.0e-6));
        assert!(sample.pdf > 0.0);
        assert!(sample.wi.z > 0.0);
    }

    #[test]
    fn sample_returns_none_for_lower_hemisphere_outgoing_direction() {
        let bsdf = NormalizedLambertBsdf::new(Vec3::ONE);

        assert!(bsdf.sample(-Vec3::Z, Vec2::splat(0.5)).is_none());
    }
}
