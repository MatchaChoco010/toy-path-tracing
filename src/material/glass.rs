use std::{path::Path, sync::Arc};

use glam::Vec3;
use rand::{RngExt, rngs::ThreadRng};

use crate::bsdf::{BsdfFlags, GlassBsdf};

use super::{
    GEOMETRIC_NORMAL_COS_EPSILON, MaterialSample, NormalMap, ShadingVertex, Texture,
    TextureColorSpace, normal_map::load_optional_normal_map, texture::load_optional_texture,
};

#[derive(Debug, Clone, PartialEq)]
pub struct GlassMaterial {
    pub eta: f32,
    pub color: Vec3,
    pub color_texture: Option<Arc<Texture>>,
    pub thin: bool,
    pub normal_map: Option<NormalMap>,
    pub normal_strength: f32,
    pub opacity: f32,
    pub opacity_texture: Option<Arc<Texture>>,
}

impl GlassMaterial {
    pub fn new(eta: f32, color: Vec3, thin: bool) -> Self {
        Self {
            eta,
            color,
            color_texture: None,
            thin,
            normal_map: None,
            normal_strength: 1.0,
            opacity: 1.0,
            opacity_texture: None,
        }
    }

    pub fn try_new_with_texture_path(
        eta: f32,
        color: Vec3,
        color_texture_path: Option<&Path>,
        normal_map_path: Option<&Path>,
        thin: bool,
    ) -> image::ImageResult<Self> {
        Ok(Self {
            eta,
            color,
            color_texture: load_optional_texture(color_texture_path, TextureColorSpace::Srgb)?,
            thin,
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
                texture.sample_scalar_filtered(
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
        rng: &mut ThreadRng,
    ) -> Option<MaterialSample> {
        let uc = rng.random::<f32>();
        let sample = self.sample_impl(shading_vertex, uc)?;

        let wi_side = sample.wi.dot(shading_vertex.ng);
        if sample.flags.contains(BsdfFlags::REFLECTION) && wi_side <= GEOMETRIC_NORMAL_COS_EPSILON {
            return None;
        }
        if sample.flags.contains(BsdfFlags::TRANSMISSION)
            && wi_side >= -GEOMETRIC_NORMAL_COS_EPSILON
        {
            return None;
        }

        Some(sample)
    }

    fn sample_impl(&self, shading_vertex: &ShadingVertex, uc: f32) -> Option<MaterialSample> {
        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let bsdf = GlassBsdf::new(
            self.eta,
            self.color_at(shading_vertex),
            self.thin,
            shading_vertex.front_face,
        );
        let sample = bsdf.sample(wo_local, uc)?;
        let wi = shading_vertex.frame.local_to_world(sample.wi);

        Some(MaterialSample {
            weight: sample.weight,
            wi,
            pdf: sample.pdf,
            flags: sample.flags,
            eta: sample.eta,
            cone_spread: 0.0,
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

    fn color_at(&self, shading_vertex: &ShadingVertex) -> Vec3 {
        self.color
            * self
                .color_texture
                .as_ref()
                .map(|texture| {
                    texture.sample_rgb_filtered(
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
        material::{GlassMaterial, ShadingVertex, Texture},
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
        }
    }

    #[test]
    fn texture_modulates_transmission_color() {
        let material = GlassMaterial {
            eta: 1.5,
            color: Vec3::new(0.5, 0.5, 0.5),
            color_texture: Some(Arc::new(Texture::from_pixels(
                1,
                1,
                vec![Vec3::new(0.2, 0.4, 0.6)],
            ))),
            thin: false,
            normal_map: None,
            normal_strength: 1.0,
            opacity: 1.0,
            opacity_texture: None,
        };
        let vtx = test_shading_vertex(Vec2::ZERO);
        let sample = material
            .sample_impl(&vtx, 1.0)
            .expect("expected a transmission sample");
        let radiance_scale = 1.0 / (1.0 / material.eta).powi(2);

        assert!(
            sample
                .weight
                .abs_diff_eq(Vec3::new(0.1, 0.2, 0.3) * radiance_scale, 1.0e-6)
        );
    }
}
