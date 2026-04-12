use glam::{Vec2, Vec3};

use crate::bsdf::GlassBsdf;

use super::{MaterialSample, ShadingVertex};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlassMaterial {
    pub eta: f32,
    pub color: Vec3,
    pub thin: bool,
}

impl GlassMaterial {
    pub fn new(eta: f32, color: Vec3, thin: bool) -> Self {
        Self { eta, color, thin }
    }

    pub fn sample(&self, shading_vertex: &ShadingVertex, us: Vec2) -> Option<MaterialSample> {
        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let bsdf = GlassBsdf::new(self.eta, self.color, self.thin, shading_vertex.front_face);
        let sample = bsdf.sample(wo_local, us)?;
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
