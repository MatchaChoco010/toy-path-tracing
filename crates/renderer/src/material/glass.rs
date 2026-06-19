use std::{path::Path, sync::Arc};

use glam::Vec3;

use crate::{
    bsdf::{BsdfFlags, GlassBsdf, TransportMode},
    sampler::{AuxRng, MaterialSampleRandoms},
};

use super::{
    GEOMETRIC_NORMAL_COS_EPSILON, MaterialSample, NormalMap, ScalarTexture, ShadingVertex, Texture,
    TextureColorSpace, modified_bsdf_sample_weight, normal_map::load_optional_normal_map,
    texture::load_optional_color_texture,
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
    pub opacity_texture: Option<Arc<ScalarTexture>>,
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
        eta: f32,
        color: Vec3,
        color_texture_path: Option<&Path>,
        normal_map_path: Option<&Path>,
        thin: bool,
        ocio: &crate::color::OcioColorPipeline,
    ) -> image::ImageResult<Self> {
        Ok(Self {
            eta,
            color,
            color_texture: load_optional_color_texture(
                color_texture_path,
                TextureColorSpace::Srgb,
                ocio,
            )?,
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

    pub(crate) fn prepare_shading_vertex(&self, shading_vertex: &mut ShadingVertex) {
        if let Some(normal_map) = self.normal_map.as_ref() {
            normal_map.apply(shading_vertex, self.normal_strength);
        }
    }

    pub fn sample(
        &self,
        shading_vertex: &ShadingVertex,
        randoms: &MaterialSampleRandoms,
        _aux_rng: &mut AuxRng,
        mode: TransportMode,
    ) -> Option<MaterialSample> {
        let uc = randoms.u_lobe;
        let sample = self.sample_impl(shading_vertex, uc, mode)?;

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

    fn sample_impl(
        &self,
        shading_vertex: &ShadingVertex,
        uc: f32,
        mode: TransportMode,
    ) -> Option<MaterialSample> {
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
        let sample = bsdf.sample(wo_local, uc, mode)?;
        let wi = shading_vertex.frame.local_to_world(sample.wi);

        Some(MaterialSample {
            weight: modified_bsdf_sample_weight(
                shading_vertex,
                wi,
                sample.weight,
                sample.flags,
                mode,
            ),
            wi,
            pdf: sample.pdf,
            pdf_rev: sample.pdf_rev,
            flags: sample.flags,
            eta: sample.eta,
            cone_spread: 0.0,
            wavelength_lock: None,
        })
    }

    pub fn eval(
        &self,
        _shading_vertex: &ShadingVertex,
        _wi: Vec3,
        _aux_rng: &mut AuxRng,
        _mode: TransportMode,
    ) -> Vec3 {
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
        bsdf::TransportMode,
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
            path_throughput: Vec3::ONE,
            wavelength_lock: None,
            object_to_world: glam::Mat4::IDENTITY,
            world_to_object: glam::Mat4::IDENTITY,
            object_normal_to_world: glam::Mat3::IDENTITY,
            mtlx_regs: None,
            mtlx_dalbedo: None,
            mtlx_precomputed_for: None,
        }
    }

    #[test]
    fn texture_modulates_transmission_color() {
        let material = GlassMaterial {
            eta: 1.5,
            color: Vec3::ONE,
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
            .sample_impl(&vtx, 1.0, TransportMode::Radiance)
            .expect("expected a transmission sample");
        let radiance_scale = 1.0 / (1.0 / material.eta).powi(2);

        assert!(
            sample
                .weight
                .abs_diff_eq(Vec3::new(0.2, 0.4, 0.6) * radiance_scale, 1.0e-3)
        );
    }
}
