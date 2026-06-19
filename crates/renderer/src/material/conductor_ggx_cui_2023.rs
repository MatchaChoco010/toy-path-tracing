use std::{path::Path, sync::Arc};

use glam::Vec3;

use crate::{
    bsdf::{BsdfFlags, ConductorGgxCui2023Bsdf, TransportMode},
    sampler::{AuxRng, MaterialSampleRandoms},
};

use super::{
    GEOMETRIC_NORMAL_COS_EPSILON, MaterialSample, NormalMap, ScalarTexture, ShadingVertex, Texture,
    TextureColorSpace, modified_bsdf_eval, modified_bsdf_sample_weight,
    normal_map::load_optional_normal_map,
    texture::{load_optional_color_texture, load_optional_scalar_texture},
};

const MIN_ALPHA: f32 = 1.0e-4;

#[derive(Debug, Clone, PartialEq)]
pub struct ConductorGgxCui2023Material {
    pub base_color: Vec3,
    pub base_color_texture: Option<Arc<Texture>>,
    pub roughness: f32,
    pub roughness_texture: Option<Arc<ScalarTexture>>,
    pub anisotropy: f32,
    pub normal_map: Option<NormalMap>,
    pub normal_strength: f32,
    pub opacity: f32,
    pub opacity_texture: Option<Arc<ScalarTexture>>,
}

impl ConductorGgxCui2023Material {
    pub fn new(base_color: Vec3, roughness: f32, anisotropy: f32) -> Self {
        Self {
            base_color,
            base_color_texture: None,
            roughness,
            roughness_texture: None,
            anisotropy,
            normal_map: None,
            normal_strength: 1.0,
            opacity: 1.0,
            opacity_texture: None,
        }
    }

    pub fn with_base_color_texture(mut self, texture: Arc<Texture>) -> Self {
        self.base_color_texture = Some(texture);
        self
    }

    pub fn with_roughness_texture(mut self, texture: Arc<ScalarTexture>) -> Self {
        self.roughness_texture = Some(texture);
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

    pub fn try_new_with_texture_paths(
        base_color: Vec3,
        roughness: f32,
        anisotropy: f32,
        base_color_texture_path: Option<&Path>,
        roughness_texture_path: Option<&Path>,
        normal_map_path: Option<&Path>,
        ocio: &crate::color::OcioColorPipeline,
    ) -> image::ImageResult<Self> {
        Ok(Self {
            base_color,
            base_color_texture: load_optional_color_texture(
                base_color_texture_path,
                TextureColorSpace::Srgb,
                ocio,
            )?,
            roughness,
            roughness_texture: load_optional_scalar_texture(roughness_texture_path)?,
            anisotropy,
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
        _randoms: &MaterialSampleRandoms,
        aux_rng: &mut AuxRng,
        mode: TransportMode,
    ) -> Option<MaterialSample> {
        if shading_vertex.wo.dot(shading_vertex.ng) <= 0.0 {
            return None;
        }

        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let roughness = self.roughness_at(shading_vertex);
        let (alpha_x, alpha_y) = self.alpha_xy_from_roughness(roughness);
        let bsdf =
            ConductorGgxCui2023Bsdf::new(self.base_color_at(shading_vertex), alpha_x, alpha_y);
        let sample = bsdf.sample(wo_local, aux_rng)?;
        let wi = shading_vertex.frame.local_to_world(sample.wi);

        if wi.dot(shading_vertex.ng) <= GEOMETRIC_NORMAL_COS_EPSILON {
            return None;
        }

        let cone_spread = if sample.flags.contains(BsdfFlags::GLOSSY) {
            2.0 * roughness.clamp(0.0, 1.0)
        } else {
            0.0
        };

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
            cone_spread,
            wavelength_lock: None,
        })
    }

    pub fn eval(
        &self,
        shading_vertex: &ShadingVertex,
        wi: Vec3,
        aux_rng: &mut AuxRng,
        mode: TransportMode,
    ) -> Vec3 {
        if shading_vertex.wo.dot(shading_vertex.ng) <= 0.0 || wi.dot(shading_vertex.ng) <= 0.0 {
            return Vec3::ZERO;
        }

        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let wi_local = shading_vertex.frame.world_to_local(wi).normalize_or_zero();
        let (alpha_x, alpha_y) = self.alpha_xy_at(shading_vertex);
        let bsdf =
            ConductorGgxCui2023Bsdf::new(self.base_color_at(shading_vertex), alpha_x, alpha_y);
        modified_bsdf_eval(
            shading_vertex,
            wi,
            bsdf.eval(wo_local, wi_local, aux_rng),
            mode,
        )
    }

    pub fn pdf(&self, shading_vertex: &ShadingVertex, wi: Vec3) -> f32 {
        if shading_vertex.wo.dot(shading_vertex.ng) <= 0.0 || wi.dot(shading_vertex.ng) <= 0.0 {
            return 0.0;
        }

        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let wi_local = shading_vertex.frame.world_to_local(wi).normalize_or_zero();
        let (alpha_x, alpha_y) = self.alpha_xy_at(shading_vertex);
        let bsdf =
            ConductorGgxCui2023Bsdf::new(self.base_color_at(shading_vertex), alpha_x, alpha_y);
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

    pub fn light_tree_precompute(
        &self,
        shading_vertex: &ShadingVertex,
    ) -> Option<crate::light_tree::LightTreePrecompute> {
        let rho = crate::math::sg::luminance(self.base_color_at(shading_vertex));
        let alpha = self.alpha_xy_at(shading_vertex);
        let glossy = crate::light_tree::make_glossy_lobe(
            rho,
            shading_vertex.frame,
            shading_vertex.wo,
            alpha.0,
            alpha.1,
        )?;
        Some(crate::light_tree::LightTreePrecompute {
            p: shading_vertex.p,
            n: shading_vertex.ns,
            frame: shading_vertex.frame,
            diffuse: None,
            glossy: Some(glossy),
            btdf: None,
        })
    }

    pub fn light_tree_importance(
        &self,
        precompute: &crate::light_tree::LightTreePrecompute,
        w: f32,
        lobe: &crate::math::sg::SgLobe,
    ) -> f32 {
        precompute.glossy.map_or(0.0, |g| {
            crate::light_tree::glossy_importance(g, precompute.frame, precompute.n, w, lobe)
        })
    }

    fn alpha_xy_at(&self, shading_vertex: &ShadingVertex) -> (f32, f32) {
        self.alpha_xy_from_roughness(self.roughness_at(shading_vertex))
    }

    fn alpha_xy_from_roughness(&self, roughness: f32) -> (f32, f32) {
        let roughness = roughness.clamp(0.0, 1.0);
        let anisotropy = self.anisotropy.clamp(-1.0, 1.0);
        let alpha = roughness * roughness;
        let aspect = (1.0 - 0.9 * anisotropy.abs()).sqrt();
        let (alpha_x, alpha_y) = if anisotropy >= 0.0 {
            (alpha / aspect, alpha * aspect)
        } else {
            (alpha * aspect, alpha / aspect)
        };
        let alpha_x = alpha_x.clamp(MIN_ALPHA, 1.0);
        let alpha_y = alpha_y.clamp(MIN_ALPHA, 1.0);
        (alpha_x, alpha_y)
    }

    fn base_color_at(&self, shading_vertex: &ShadingVertex) -> Vec3 {
        self.base_color
            * self
                .base_color_texture
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

    fn roughness_at(&self, shading_vertex: &ShadingVertex) -> f32 {
        self.roughness
            * self
                .roughness_texture
                .as_ref()
                .map(|texture| {
                    texture.sample_filtered(
                        shading_vertex.uv,
                        shading_vertex.uv_dx(),
                        shading_vertex.uv_dy(),
                    )
                })
                .unwrap_or(1.0)
    }
}
