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

    pub fn sample(
        &self,
        shading_vertex: &ShadingVertex,
        us: Vec2,
        wo: Vec3,
    ) -> Option<MaterialSample> {
        let wo_local = shading_vertex.frame.world_to_local(wo).normalize_or_zero();
        let bsdf = NormalizedLambertBsdf::new(self.rho);
        let sample = bsdf.sample(wo_local, us)?;
        let wi = shading_vertex.frame.local_to_world(sample.wi);

        Some(MaterialSample {
            weight: sample.weight,
            wi,
            pdf: sample.pdf,
        })
    }

    pub fn le(&self, _shading_vertex: &ShadingVertex) -> Option<Vec3> {
        None
    }
}
