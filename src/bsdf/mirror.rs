use glam::Vec3;

use super::{BsdfFlags, BsdfSample};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MirrorBsdf {
    color: Vec3,
}

impl MirrorBsdf {
    pub fn new(color: Vec3) -> Self {
        Self { color }
    }

    pub fn eval(&self, _wo: Vec3, _wi: Vec3) -> Vec3 {
        Vec3::ZERO
    }

    pub fn pdf(&self, _wo: Vec3, _wi: Vec3) -> f32 {
        0.0
    }

    pub fn sample(&self, wo: Vec3) -> Option<BsdfSample> {
        if wo.z <= 0.0 {
            return None;
        }

        let wi = Vec3::new(-wo.x, -wo.y, wo.z).normalize_or_zero();

        Some(BsdfSample {
            weight: self.color,
            wi,
            pdf: 1.0,
            flags: BsdfFlags::DELTA | BsdfFlags::REFLECTION,
        })
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::bsdf::{BsdfFlags, MirrorBsdf};

    #[test]
    fn eval_and_pdf_are_zero() {
        let bsdf = MirrorBsdf::new(Vec3::new(0.3, 0.5, 0.7));

        assert_eq!(bsdf.eval(Vec3::Z, Vec3::Z), Vec3::ZERO);
        assert_eq!(bsdf.pdf(Vec3::Z, Vec3::Z), 0.0);
    }

    #[test]
    fn sample_returns_perfect_mirror_reflection_with_color_weight() {
        let color = Vec3::new(0.3, 0.5, 0.7);
        let bsdf = MirrorBsdf::new(color);
        let wo = Vec3::new(0.3, -0.4, 0.8660254).normalize();

        let sample = bsdf.sample(wo).expect("expected a valid sample");

        let expected_wi = Vec3::new(-wo.x, -wo.y, wo.z).normalize();
        assert!(sample.wi.abs_diff_eq(expected_wi, 1.0e-6));
        assert_eq!(sample.weight, color);
        assert_eq!(sample.pdf, 1.0);
        assert_eq!(sample.flags, BsdfFlags::DELTA | BsdfFlags::REFLECTION);
    }

    #[test]
    fn sample_returns_none_for_lower_hemisphere_outgoing_direction() {
        let bsdf = MirrorBsdf::new(Vec3::ONE);

        assert!(bsdf.sample(-Vec3::Z).is_none());
    }
}
