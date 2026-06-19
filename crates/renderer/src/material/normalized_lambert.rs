use std::{path::Path, sync::Arc};

use glam::Vec3;

use crate::{
    bsdf::{NormalizedLambertBsdf, TransportMode},
    sampler::{AuxRng, MaterialSampleRandoms},
};

use super::{
    GEOMETRIC_NORMAL_COS_EPSILON, MaterialSample, NormalMap, ScalarTexture, ShadingVertex, Texture,
    TextureColorSpace, modified_bsdf_eval, modified_bsdf_sample_weight,
    normal_map::load_optional_normal_map, texture::load_optional_color_texture,
};

const DIFFUSE_CONE_SPREAD: f32 = 0.5;

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedLambertMaterial {
    pub rho: Vec3,
    pub rho_texture: Option<Arc<Texture>>,
    pub normal_map: Option<NormalMap>,
    pub normal_strength: f32,
    pub opacity: f32,
    pub opacity_texture: Option<Arc<ScalarTexture>>,
}

impl NormalizedLambertMaterial {
    pub fn new(rho: Vec3) -> Self {
        Self {
            rho,
            rho_texture: None,
            normal_map: None,
            normal_strength: 1.0,
            opacity: 1.0,
            opacity_texture: None,
        }
    }

    pub fn with_rho_texture(mut self, texture: Arc<Texture>) -> Self {
        self.rho_texture = Some(texture);
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
        rho: Vec3,
        rho_texture_path: Option<&Path>,
        normal_map_path: Option<&Path>,
        ocio: &crate::color::OcioColorPipeline,
    ) -> image::ImageResult<Self> {
        Ok(Self {
            rho,
            rho_texture: load_optional_color_texture(
                rho_texture_path,
                TextureColorSpace::Srgb,
                ocio,
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
        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let us = randoms.u_dir;
        let bsdf = NormalizedLambertBsdf::new(self.rho_at(shading_vertex));
        let sample = bsdf.sample(wo_local, us)?;
        let wi = shading_vertex.frame.local_to_world(sample.wi);

        let sample = MaterialSample {
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
            cone_spread: DIFFUSE_CONE_SPREAD,
            wavelength_lock: None,
        };

        if sample.wi.dot(shading_vertex.ng) <= GEOMETRIC_NORMAL_COS_EPSILON {
            return None;
        }

        Some(sample)
    }

    pub fn eval(
        &self,
        shading_vertex: &ShadingVertex,
        wi: Vec3,
        _aux_rng: &mut AuxRng,
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
        let bsdf = NormalizedLambertBsdf::new(self.rho_at(shading_vertex));
        modified_bsdf_eval(shading_vertex, wi, bsdf.eval(wo_local, wi_local), mode)
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
        let bsdf = NormalizedLambertBsdf::new(self.rho_at(shading_vertex));
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
        let rho = crate::math::sg::luminance(self.rho_at(shading_vertex));
        if rho <= 0.0 {
            return None;
        }
        Some(crate::light_tree::LightTreePrecompute {
            p: shading_vertex.p,
            n: shading_vertex.ns,
            frame: shading_vertex.frame,
            diffuse: Some(crate::light_tree::DiffuseLobePrecompute { rho }),
            glossy: None,
            btdf: None,
        })
    }

    pub fn light_tree_importance(
        &self,
        precompute: &crate::light_tree::LightTreePrecompute,
        w: f32,
        lobe: &crate::math::sg::SgLobe,
    ) -> f32 {
        precompute.diffuse.map_or(0.0, |d| {
            crate::light_tree::diffuse_importance(d, precompute.n, w, lobe)
        })
    }

    fn rho_at(&self, shading_vertex: &ShadingVertex) -> Vec3 {
        self.rho
            * self
                .rho_texture
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
    use std::{f32::consts::PI, sync::Arc};

    use glam::{Vec2, Vec3};

    use crate::{
        bsdf::TransportMode,
        material::{NormalizedLambertMaterial, ShadingVertex, Texture},
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
    fn texture_modulates_rho() {
        let material = NormalizedLambertMaterial {
            rho: Vec3::ONE,
            rho_texture: Some(Arc::new(Texture::from_pixels(
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

        assert!(
            material
                .eval(
                    &vtx,
                    Vec3::Z,
                    &mut crate::sampler::AuxRng::default(),
                    TransportMode::Radiance
                )
                .abs_diff_eq(Vec3::new(0.2, 0.4, 0.6) / PI, 1.0e-3)
        );
    }

    #[test]
    fn none_texture_keeps_existing_rho() {
        let ocio = crate::color::OcioColorPipeline::new(
            crate::color::DEFAULT_OCIO_CONFIG,
            Some(crate::color::DEFAULT_RENDERING_SPACE.to_string()),
            crate::color::DEFAULT_TEXTURE_COLOR_SPACE,
        )
        .expect("default OCIO config");
        let material =
            NormalizedLambertMaterial::try_new_with_texture_path(Vec3::ONE, None, None, &ocio)
                .expect("None texture should not try to load an image");

        assert_eq!(material, NormalizedLambertMaterial::new(Vec3::ONE));
    }
}
