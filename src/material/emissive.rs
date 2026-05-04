use std::sync::Arc;

use glam::Vec3;
use rand::rngs::ThreadRng;

use crate::color::srgb_to_linear;

use super::{MaterialSample, ScalarTexture, ShadingVertex, Texture};

#[derive(Debug, Clone, PartialEq)]
pub struct EmissiveMaterial {
    pub color: Vec3,
    pub strength: f32,
    pub color_texture: Option<Arc<Texture>>,
    pub opacity: f32,
    pub opacity_texture: Option<Arc<ScalarTexture>>,
}

impl EmissiveMaterial {
    pub fn new(color: Vec3, strength: f32) -> Self {
        Self {
            color,
            strength,
            color_texture: None,
            opacity: 1.0,
            opacity_texture: None,
        }
    }

    pub fn with_color_texture(mut self, texture: Arc<Texture>) -> Self {
        self.color_texture = Some(texture);
        self
    }

    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    pub fn with_opacity_texture(mut self, texture: Arc<ScalarTexture>) -> Self {
        self.opacity_texture = Some(texture);
        self
    }

    pub fn sample(
        &self,
        _shading_vertex: &ShadingVertex,
        _rng: &mut ThreadRng,
    ) -> Option<MaterialSample> {
        None
    }

    pub fn eval(
        &self,
        _shading_vertex: &ShadingVertex,
        _wi: Vec3,
        _internal_rng: &mut ThreadRng,
    ) -> Vec3 {
        Vec3::ZERO
    }

    pub fn pdf(&self, _shading_vertex: &ShadingVertex, _wi: Vec3) -> f32 {
        0.0
    }

    pub fn le(&self, shading_vertex: &ShadingVertex) -> Option<Vec3> {
        let texture_factor = self
            .color_texture
            .as_ref()
            .map(|texture| {
                texture.sample_filtered(
                    shading_vertex.uv,
                    shading_vertex.uv_dx(),
                    shading_vertex.uv_dy(),
                )
            })
            .unwrap_or(Vec3::ONE);
        Some(srgb_to_linear(self.color) * self.strength * texture_factor)
    }

    pub fn may_emit(&self) -> bool {
        true
    }

    pub fn max_emission(&self) -> f32 {
        let texture_factor = self
            .color_texture
            .as_ref()
            .map(|texture| texture.max_value())
            .unwrap_or(1.0);
        ((srgb_to_linear(self.color) * self.strength).max_element() * texture_factor).max(0.0)
    }

    pub fn opacity_at_uv(&self, shading_vertex: &ShadingVertex) -> f32 {
        let texture_factor = self
            .opacity_texture
            .as_ref()
            .map(|texture| {
                texture.sample_filtered(
                    shading_vertex.uv,
                    shading_vertex.uv_dx(),
                    shading_vertex.uv_dy(),
                )
            })
            .unwrap_or(1.0);
        (self.opacity * texture_factor).clamp(0.0, 1.0)
    }

    pub fn has_alpha_test(&self) -> bool {
        self.opacity < 1.0 || self.opacity_texture.is_some()
    }

    pub fn any_hit(&self, shading_vertex: &ShadingVertex, u: f32) -> bool {
        let alpha = self.opacity_at_uv(shading_vertex);
        alpha >= 1.0 || u < alpha
    }
}
