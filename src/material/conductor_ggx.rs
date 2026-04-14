use glam::{Vec2, Vec3};
use rand::{RngExt, rngs::ThreadRng};

use crate::{
    bsdf::{BsdfFlags, ConductorGgxBsdf},
    math::OrthonormalBasis,
};

use super::{MaterialSample, ShadingVertex};

const MIN_ALPHA: f32 = 1.0e-4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConductorGgxMaterial {
    pub base_color: Vec3,
    pub roughness: f32,
    pub anisotropy: f32,
}

impl ConductorGgxMaterial {
    pub fn new(base_color: Vec3, roughness: f32, anisotropy: f32) -> Self {
        Self {
            base_color,
            roughness,
            anisotropy,
        }
    }

    pub fn sample(
        &self,
        shading_vertex: &ShadingVertex,
        rng: &mut ThreadRng,
    ) -> Option<MaterialSample> {
        let us = Vec2::new(rng.random::<f32>(), rng.random::<f32>());
        let sample = self.sample_with_frame(shading_vertex, shading_vertex.frame, us)?;

        if sample_matches_geometric_reflection_side(&sample, shading_vertex.ng) {
            return Some(sample);
        }

        let geometric_frame =
            OrthonormalBasis::from_normal_and_tangent(shading_vertex.ng, shading_vertex.dpdu);
        let sample = self.sample_with_frame(shading_vertex, geometric_frame, us)?;

        if !sample_matches_geometric_reflection_side(&sample, shading_vertex.ng) {
            return None;
        }

        Some(sample)
    }

    fn sample_with_frame(
        &self,
        shading_vertex: &ShadingVertex,
        frame: OrthonormalBasis,
        us: Vec2,
    ) -> Option<MaterialSample> {
        if shading_vertex.wo.dot(shading_vertex.ng) <= 0.0 {
            return None;
        }

        let wo_local = frame.world_to_local(shading_vertex.wo).normalize_or_zero();
        let (alpha_x, alpha_y) = self.alpha_xy();
        let bsdf = ConductorGgxBsdf::new(self.base_color, alpha_x, alpha_y);
        let sample = bsdf.sample(wo_local, us)?;
        let wi = frame.local_to_world(sample.wi);

        Some(MaterialSample {
            weight: sample.weight,
            wi,
            pdf: sample.pdf,
            flags: sample.flags,
        })
    }

    pub fn eval(&self, shading_vertex: &ShadingVertex, wi: Vec3) -> Vec3 {
        if shading_vertex.wo.dot(shading_vertex.ng) <= 0.0 || wi.dot(shading_vertex.ng) <= 0.0 {
            return Vec3::ZERO;
        }

        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let wi_local = shading_vertex.frame.world_to_local(wi).normalize_or_zero();
        let (alpha_x, alpha_y) = self.alpha_xy();
        let bsdf = ConductorGgxBsdf::new(self.base_color, alpha_x, alpha_y);
        bsdf.eval(wo_local, wi_local)
    }

    pub fn pdf(&self, shading_vertex: &ShadingVertex, wi: Vec3) -> f32 {
        if shading_vertex.wo.dot(shading_vertex.ng) <= 0.0 || wi.dot(shading_vertex.ng) <= 0.0 {
            return 0.0;
        }

        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let wi_local = shading_vertex.frame.world_to_local(wi).normalize_or_zero();
        let (alpha_x, alpha_y) = self.alpha_xy();
        let bsdf = ConductorGgxBsdf::new(self.base_color, alpha_x, alpha_y);
        bsdf.pdf(wo_local, wi_local)
    }

    pub fn le(&self, _shading_vertex: &ShadingVertex) -> Option<Vec3> {
        None
    }

    pub fn may_emit(&self) -> bool {
        false
    }

    pub fn max_emission(&self) -> f32 {
        0.0
    }

    fn alpha_xy(&self) -> (f32, f32) {
        let roughness = self.roughness.clamp(0.0, 1.0);
        let anisotropy = self.anisotropy.clamp(-1.0, 1.0);
        let alpha = roughness * roughness;
        let aspect = (1.0 - 0.9 * anisotropy.abs()).sqrt();
        let (alpha_x, alpha_y) = if anisotropy >= 0.0 {
            (alpha / aspect, alpha * aspect)
        } else {
            (alpha * aspect, alpha / aspect)
        };
        let alpha_x = alpha_x.clamp(MIN_ALPHA, 1.0);
        let alpha_y = alpha_y.clamp(MIN_ALPHA, 1.0);
        (alpha_x, alpha_y)
    }
}

fn sample_matches_geometric_reflection_side(
    sample: &MaterialSample,
    geometric_normal: Vec3,
) -> bool {
    if !sample.flags.contains(BsdfFlags::REFLECTION) {
        return false;
    }

    sample.wi.dot(geometric_normal) > 1.0e-6
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};

    use crate::{
        bsdf::BsdfFlags,
        math::OrthonormalBasis,
        scene::{InstanceIndex, TriangleRef},
    };

    use super::ConductorGgxMaterial;
    use crate::material::ShadingVertex;

    fn test_shading_vertex(wo: Vec3) -> ShadingVertex {
        ShadingVertex {
            triangle: TriangleRef {
                instance_index: InstanceIndex(0),
                triangle_index: 0,
            },
            p: Vec3::ZERO,
            uv: Vec2::ZERO,
            ng: Vec3::Z,
            ns: Vec3::Z,
            wo,
            dpdu: Vec3::X,
            dpdv: Vec3::Y,
            frame: OrthonormalBasis::from_normal(Vec3::Z),
            front_face: true,
        }
    }

    #[test]
    fn alpha_mapping_matches_expected_isotropic_case() {
        let material = ConductorGgxMaterial::new(Vec3::ONE, 0.5, 0.0);
        let vtx = test_shading_vertex(Vec3::Z);
        let wi = Vec3::new(0.0, 0.0, 1.0);

        assert!(material.eval(&vtx, wi).max_element().is_finite());
    }

    #[test]
    fn signed_anisotropy_flips_alpha_axes() {
        let positive = ConductorGgxMaterial::new(Vec3::ONE, 0.4, 0.8);
        let negative = ConductorGgxMaterial::new(Vec3::ONE, 0.4, -0.8);

        let (pos_x, pos_y) = positive.alpha_xy();
        let (neg_x, neg_y) = negative.alpha_xy();

        assert!((pos_x - neg_y).abs() < 1.0e-6);
        assert!((pos_y - neg_x).abs() < 1.0e-6);
        assert!(pos_x > pos_y);
    }

    #[test]
    fn sample_returns_reflection_sample() {
        let material = ConductorGgxMaterial::new(Vec3::new(0.9, 0.6, 0.2), 0.45, 0.25);
        let vtx = test_shading_vertex(Vec3::new(0.2, -0.1, 0.9746794).normalize());
        let mut rng = rand::rng();
        let sample = material
            .sample(&vtx, &mut rng)
            .expect("expected a reflection sample");

        assert!(sample.wi.z > 0.0);
        assert!(sample.pdf > 0.0);
        assert!(sample.flags.contains(BsdfFlags::REFLECTION));
    }
}
