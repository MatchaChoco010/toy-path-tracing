use glam::Vec3;
use rand::{RngExt, rngs::ThreadRng};

use crate::{
    bsdf::{BsdfFlags, GlassBsdf},
    math::OrthonormalBasis,
};

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

    pub fn sample(
        &self,
        shading_vertex: &ShadingVertex,
        rng: &mut ThreadRng,
    ) -> Option<MaterialSample> {
        let uc = rng.random::<f32>();
        let sample = self.sample_with_frame(shading_vertex, shading_vertex.frame, uc)?;

        if sample_matches_geometric_side(&sample, shading_vertex.ng) {
            return Some(sample);
        }

        // Shading normals can bend a transmission or reflection event across
        // the actual surface. Fall back to the geometric frame only in that
        // case so smooth normal interpolation still shapes the result when the
        // sampled direction is physically plausible.
        let geometric_frame = OrthonormalBasis::from_normal(shading_vertex.ng);
        let sample = self.sample_with_frame(shading_vertex, geometric_frame, uc)?;

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
    ) -> Option<MaterialSample> {
        let wo_local = frame.world_to_local(shading_vertex.wo).normalize_or_zero();
        let bsdf = GlassBsdf::new(self.eta, self.color, self.thin, shading_vertex.front_face);
        let sample = bsdf.sample(wo_local, uc)?;
        let wi = frame.local_to_world(sample.wi);

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
