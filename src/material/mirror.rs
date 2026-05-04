use std::{path::Path, sync::Arc};

use glam::Vec3;
use rand::rngs::ThreadRng;

use crate::{bsdf::MirrorBsdf, color::srgb_to_linear};

use super::{
    GEOMETRIC_NORMAL_COS_EPSILON, MaterialSample, NormalMap, ScalarTexture, ShadingVertex, Texture,
    TextureColorSpace, normal_map::load_optional_normal_map, texture::load_optional_color_texture,
};

#[derive(Debug, Clone, PartialEq)]
pub struct MirrorMaterial {
    pub color: Vec3,
    pub color_texture: Option<Arc<Texture>>,
    pub normal_map: Option<NormalMap>,
    pub normal_strength: f32,
    pub opacity: f32,
    pub opacity_texture: Option<Arc<ScalarTexture>>,
}

impl MirrorMaterial {
    pub fn new(color: Vec3) -> Self {
        Self {
            color,
            color_texture: None,
            normal_map: None,
            normal_strength: 1.0,
            opacity: 1.0,
            opacity_texture: None,
        }
    }

    pub fn with_color_texture(mut self, texture: Arc<Texture>) -> Self {
        self.color_texture = Some(texture);
        self
    }

    pub fn with_normal_map(mut self, normal_map: NormalMap) -> Self {
        self.normal_map = Some(normal_map);
        self
    }

    pub fn with_normal_strength(mut self, strength: f32) -> Self {
        self.normal_strength = strength;
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

    pub fn try_new_with_texture_path(
        color: Vec3,
        color_texture_path: Option<&Path>,
        normal_map_path: Option<&Path>,
    ) -> image::ImageResult<Self> {
        Ok(Self {
            color,
            color_texture: load_optional_color_texture(
                color_texture_path,
                TextureColorSpace::Srgb,
            )?,
            normal_map: load_optional_normal_map(normal_map_path)?,
            normal_strength: 1.0,
            opacity: 1.0,
            opacity_texture: None,
        })
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

    pub(crate) fn prepare_shading_vertex(&self, shading_vertex: &ShadingVertex) -> ShadingVertex {
        self.normal_map
            .as_ref()
            .map(|normal_map| normal_map.apply(shading_vertex, self.normal_strength))
            .unwrap_or(*shading_vertex)
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
        let bsdf = MirrorBsdf::new(self.color_at(shading_vertex));
        let sample = bsdf.sample(wo_local)?;
        let wi = shading_vertex.frame.local_to_world(sample.wi);

        let sample = MaterialSample {
            weight: sample.weight,
            wi,
            pdf: sample.pdf,
            flags: sample.flags,
            eta: sample.eta,
            cone_spread: 0.0,
            wavelength_lock: None,
        };

        if sample.wi.dot(shading_vertex.ng) <= GEOMETRIC_NORMAL_COS_EPSILON {
            return None;
        }

        Some(sample)
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

    fn color_at(&self, shading_vertex: &ShadingVertex) -> Vec3 {
        srgb_to_linear(self.color)
            * self
                .color_texture
                .as_ref()
                .map(|texture| {
                    texture.sample_filtered(
                        shading_vertex.uv,
                        shading_vertex.uv_dx(),
                        shading_vertex.uv_dy(),
                    )
                })
                .unwrap_or(Vec3::ONE)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use glam::{Vec2, Vec3};

    use crate::{
        material::{MirrorMaterial, ShadingVertex, Texture},
        math::OrthonormalBasis,
        scene::{InstanceIndex, TriangleRef},
    };

    fn test_shading_vertex(uv: Vec2) -> ShadingVertex {
        ShadingVertex {
            triangle: TriangleRef {
                instance_index: InstanceIndex(0),
                triangle_index: 0,
            },
            p: Vec3::ZERO,
            uv,
            dudx: 0.0,
            dvdx: 0.0,
            dudy: 0.0,
            dvdy: 0.0,
            ng: Vec3::Z,
            ns: Vec3::Z,
            wo: Vec3::Z,
            dpdu: Vec3::X,
            dpdv: Vec3::Y,
            dpdx: Vec3::ZERO,
            dpdy: Vec3::ZERO,
            dndu: Vec3::ZERO,
            dndv: Vec3::ZERO,
            frame: OrthonormalBasis::from_normal(Vec3::Z),
            front_face: true,
            wavelength_lock: None,
        }
    }

    #[test]
    fn texture_modulates_color() {
        let material = MirrorMaterial {
            color: Vec3::ONE,
            color_texture: Some(Arc::new(Texture::from_pixels(
                1,
                1,
                vec![Vec3::new(0.2, 0.4, 0.6)],
            ))),
            normal_map: None,
            normal_strength: 1.0,
            opacity: 1.0,
            opacity_texture: None,
        };
        let vtx = test_shading_vertex(Vec2::ZERO);
        let mut rng = rand::rng();
        let sample = material
            .sample(&vtx, &mut rng)
            .expect("expected a valid mirror sample");

        assert!(sample.weight.abs_diff_eq(Vec3::new(0.2, 0.4, 0.6), 1.0e-6));
    }

    #[test]
    fn sample_returns_none_when_shading_normal_reflects_below_geometry() {
        let material = MirrorMaterial::new(Vec3::ONE);
        let mut vtx = test_shading_vertex(Vec2::ZERO);
        let ns = Vec3::new(0.8660254, 0.0, 0.5).normalize();
        vtx.ns = ns;
        vtx.frame = OrthonormalBasis::from_normal_and_tangent(ns, vtx.dpdu);
        let mut rng = rand::rng();

        assert!(material.sample(&vtx, &mut rng).is_none());
    }
}
