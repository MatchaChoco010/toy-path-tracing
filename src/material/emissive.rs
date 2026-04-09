use glam::{Vec2, Vec3};

use super::{MaterialSample, ShadingVertex};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmissiveMaterial {
    pub color: Vec3,
    pub strength: f32,
}

impl EmissiveMaterial {
    pub fn new(color: Vec3, strength: f32) -> Self {
        Self { color, strength }
    }

    pub fn sample(&self, _shading_vertex: &ShadingVertex, _us: Vec2) -> Option<MaterialSample> {
        None
    }

    pub fn eval(&self, _shading_vertex: &ShadingVertex, _wi: Vec3) -> Vec3 {
        Vec3::ZERO
    }

    pub fn pdf(&self, _shading_vertex: &ShadingVertex, _wi: Vec3) -> f32 {
        0.0
    }

    pub fn le(&self, _shading_vertex: &ShadingVertex) -> Option<Vec3> {
        Some(self.color * self.strength)
    }

    pub fn may_emit(&self) -> bool {
        true
    }

    pub fn max_emission(&self) -> f32 {
        (self.color * self.strength).max_element().max(0.0)
    }
}
