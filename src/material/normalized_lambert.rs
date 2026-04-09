use glam::{Vec2, Vec3};

use crate::bsdf::NormalizedLambertBsdf;

use super::{MaterialSample, ShadingVertex};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizedLambertMaterial {
    pub rho: Vec3,
}

impl NormalizedLambertMaterial {
    pub fn new(rho: Vec3) -> Self {
        Self { rho }
    }

    pub fn sample(&self, shading_vertex: &ShadingVertex, us: Vec2) -> Option<MaterialSample> {
        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let bsdf = NormalizedLambertBsdf::new(self.rho);
        let sample = bsdf.sample(wo_local, us)?;
        let wi = shading_vertex.frame.local_to_world(sample.wi);

        Some(MaterialSample {
            weight: sample.weight,
            wi,
            pdf: sample.pdf,
        })
    }

    pub fn eval(&self, shading_vertex: &ShadingVertex, wi: Vec3) -> Vec3 {
        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let wi_local = shading_vertex.frame.world_to_local(wi).normalize_or_zero();
        let bsdf = NormalizedLambertBsdf::new(self.rho);
        bsdf.eval(wo_local, wi_local)
    }

    pub fn pdf(&self, shading_vertex: &ShadingVertex, wi: Vec3) -> f32 {
        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let wi_local = shading_vertex.frame.world_to_local(wi).normalize_or_zero();
        let bsdf = NormalizedLambertBsdf::new(self.rho);
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
}
