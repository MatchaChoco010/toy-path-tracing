use glam::Vec3;
use rand::rngs::ThreadRng;

use crate::bsdf::MirrorBsdf;

use super::{MaterialSample, ShadingVertex};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MirrorMaterial {
    pub color: Vec3,
}

impl MirrorMaterial {
    pub fn new(color: Vec3) -> Self {
        Self { color }
    }

    pub fn sample(
        &self,
        shading_vertex: &ShadingVertex,
        _rng: &mut ThreadRng,
    ) -> Option<MaterialSample> {
        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let bsdf = MirrorBsdf::new(self.color);
        let sample = bsdf.sample(wo_local)?;
        let wi = shading_vertex.frame.local_to_world(sample.wi);

        Some(MaterialSample {
            weight: sample.weight,
            wi,
            pdf: sample.pdf,
            flags: sample.flags,
        })
    }

    pub fn eval(&self, _shading_vertex: &ShadingVertex, _wi: Vec3) -> Vec3 {
        Vec3::ZERO
    }

    pub fn pdf(&self, _shading_vertex: &ShadingVertex, _wi: Vec3) -> f32 {
        0.0
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
}
