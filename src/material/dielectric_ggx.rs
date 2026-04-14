use glam::{Vec2, Vec3};
use rand::{RngExt, rngs::ThreadRng};

use crate::{
    bsdf::{BsdfFlags, DielectricGgxBsdf},
    math::OrthonormalBasis,
};

use super::{MaterialSample, ShadingVertex};

const MIN_ALPHA: f32 = 1.0e-4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DielectricGgxMaterial {
    pub color: Vec3,
    pub eta: f32,
    pub roughness: f32,
    pub anisotropy: f32,
    pub thin: bool,
}

impl DielectricGgxMaterial {
    pub fn new(color: Vec3, eta: f32, roughness: f32, anisotropy: f32, thin: bool) -> Self {
        Self {
            color,
            eta,
            roughness,
            anisotropy,
            thin,
        }
    }

    pub fn sample(
        &self,
        shading_vertex: &ShadingVertex,
        rng: &mut ThreadRng,
    ) -> Option<MaterialSample> {
        let uc = rng.random::<f32>();
        let us = Vec2::new(rng.random::<f32>(), rng.random::<f32>());
        let sample = self.sample_with_frame(shading_vertex, shading_vertex.frame, uc, us)?;

        if sample_matches_geometric_side(&sample, shading_vertex.ng) {
            return Some(sample);
        }

        // Shading-normal interpolation can route the sample to the wrong side
        // of the surface; retry with the geometric frame in that case.
        let geometric_frame = OrthonormalBasis::from_normal(shading_vertex.ng);
        let sample = self.sample_with_frame(shading_vertex, geometric_frame, uc, us)?;

        if !sample_matches_geometric_side(&sample, shading_vertex.ng) {
            return None;
        }

        Some(sample)
    }

    fn sample_with_frame(
        &self,
        shading_vertex: &ShadingVertex,
        frame: OrthonormalBasis,
        uc: f32,
        us: Vec2,
    ) -> Option<MaterialSample> {
        let wo_local = frame.world_to_local(shading_vertex.wo).normalize_or_zero();
        let (alpha_x, alpha_y) = self.alpha_xy();
        let bsdf = DielectricGgxBsdf::new(
            self.color,
            self.eta,
            alpha_x,
            alpha_y,
            self.thin,
            shading_vertex.front_face,
        );
        let sample = bsdf.sample(wo_local, uc, us)?;
        let wi = frame.local_to_world(sample.wi);

        Some(MaterialSample {
            weight: sample.weight,
            wi,
            pdf: sample.pdf,
            flags: sample.flags,
        })
    }

    pub fn eval(&self, shading_vertex: &ShadingVertex, wi: Vec3) -> Vec3 {
        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let wi_local = shading_vertex.frame.world_to_local(wi).normalize_or_zero();
        let (alpha_x, alpha_y) = self.alpha_xy();
        let bsdf = DielectricGgxBsdf::new(
            self.color,
            self.eta,
            alpha_x,
            alpha_y,
            self.thin,
            shading_vertex.front_face,
        );
        bsdf.eval(wo_local, wi_local)
    }

    pub fn pdf(&self, shading_vertex: &ShadingVertex, wi: Vec3) -> f32 {
        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let wi_local = shading_vertex.frame.world_to_local(wi).normalize_or_zero();
        let (alpha_x, alpha_y) = self.alpha_xy();
        let bsdf = DielectricGgxBsdf::new(
            self.color,
            self.eta,
            alpha_x,
            alpha_y,
            self.thin,
            shading_vertex.front_face,
        );
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

fn sample_matches_geometric_side(sample: &MaterialSample, geometric_normal: Vec3) -> bool {
    let side = sample.wi.dot(geometric_normal);
    let epsilon = 1.0e-6;

    if sample.flags.contains(BsdfFlags::TRANSMISSION) {
        return side < -epsilon;
    }
    if sample.flags.contains(BsdfFlags::REFLECTION) {
        return side > epsilon;
    }
    true
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};

    use crate::{
        bsdf::BsdfFlags,
        material::ShadingVertex,
        math::OrthonormalBasis,
        scene::{InstanceIndex, TriangleRef},
    };

    use super::DielectricGgxMaterial;

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
    fn alpha_mapping_matches_isotropic_case() {
        let material = DielectricGgxMaterial::new(Vec3::ONE, 1.5, 0.5, 0.0, false);
        let (alpha_x, alpha_y) = material.alpha_xy();

        assert!((alpha_x - 0.25).abs() < 1.0e-6);
        assert!((alpha_y - 0.25).abs() < 1.0e-6);
    }

    #[test]
    fn signed_anisotropy_flips_alpha_axes() {
        let positive = DielectricGgxMaterial::new(Vec3::ONE, 1.5, 0.4, 0.8, false);
        let negative = DielectricGgxMaterial::new(Vec3::ONE, 1.5, 0.4, -0.8, false);
        let (pos_x, pos_y) = positive.alpha_xy();
        let (neg_x, neg_y) = negative.alpha_xy();

        assert!((pos_x - neg_y).abs() < 1.0e-6);
        assert!((pos_y - neg_x).abs() < 1.0e-6);
        assert!(pos_x > pos_y);
    }

    #[test]
    fn sample_returns_reflection_or_transmission_flag() {
        let material =
            DielectricGgxMaterial::new(Vec3::new(0.85, 0.95, 0.95), 1.5, 0.3, 0.0, false);
        let vtx = test_shading_vertex(Vec3::new(0.2, -0.1, 0.9746794).normalize());
        let mut rng = rand::rng();

        let mut saw_reflection = false;
        let mut saw_transmission = false;
        for _ in 0..256 {
            if let Some(sample) = material.sample(&vtx, &mut rng) {
                if sample.flags.contains(BsdfFlags::REFLECTION) {
                    saw_reflection = true;
                }
                if sample.flags.contains(BsdfFlags::TRANSMISSION) {
                    saw_transmission = true;
                }
                if saw_reflection && saw_transmission {
                    break;
                }
            }
        }

        assert!(saw_reflection, "expected at least one reflection sample");
        assert!(saw_transmission, "expected at least one transmission sample");
    }

    #[test]
    fn sample_at_back_face_returns_some() {
        let material = DielectricGgxMaterial::new(Vec3::ONE, 1.5, 0.3, 0.0, false);
        let mut vtx = test_shading_vertex(Vec3::Z);
        vtx.front_face = false;
        let mut rng = rand::rng();

        let sample = material
            .sample(&vtx, &mut rng)
            .expect("expected a back-face sample");
        assert!(sample.flags.intersects(BsdfFlags::REFLECTION | BsdfFlags::TRANSMISSION));
    }
}
