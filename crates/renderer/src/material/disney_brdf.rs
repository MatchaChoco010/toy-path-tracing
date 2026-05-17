use std::{path::Path, sync::Arc};

use glam::Vec3;
use rand::rngs::ThreadRng;

use crate::{
    bsdf::{BsdfFlags, DisneyBrdfBsdf},
    light_tree::{
        DiffuseLobePrecompute, LightTreePrecompute, diffuse_importance, glossy_importance,
        make_glossy_lobe, merge_glossy_roughness,
    },
    math::sg,
};

use super::{
    GEOMETRIC_NORMAL_COS_EPSILON, MaterialSample, NormalMap, ScalarTexture, ShadingVertex, Texture,
    TextureColorSpace,
    normal_map::load_optional_normal_map,
    texture::{load_optional_color_texture, load_optional_scalar_texture},
};

const GLOSSY_DIFFUSE_CONE_SPREAD: f32 = 0.5;

#[derive(Debug, Clone, PartialEq)]
pub struct DisneyBrdfMaterial {
    pub base_color: Vec3,
    pub metallic: f32,
    pub subsurface: f32,
    pub specular: f32,
    pub specular_tint: f32,
    pub roughness: f32,
    pub anisotropic: f32,
    pub sheen: f32,
    pub sheen_tint: f32,
    pub clearcoat: f32,
    pub clearcoat_gloss: f32,
    pub base_color_texture: Option<Arc<Texture>>,
    pub metallic_texture: Option<Arc<ScalarTexture>>,
    pub roughness_texture: Option<Arc<ScalarTexture>>,
    pub normal_map: Option<NormalMap>,
    pub normal_strength: f32,
    pub opacity: f32,
    pub opacity_texture: Option<Arc<ScalarTexture>>,
}

impl DisneyBrdfMaterial {
    pub fn new(base_color: Vec3) -> Self {
        Self {
            base_color,
            metallic: 0.0,
            subsurface: 0.0,
            specular: 0.5,
            specular_tint: 0.0,
            roughness: 0.5,
            anisotropic: 0.0,
            sheen: 0.0,
            sheen_tint: 0.5,
            clearcoat: 0.0,
            clearcoat_gloss: 1.0,
            base_color_texture: None,
            metallic_texture: None,
            roughness_texture: None,
            normal_map: None,
            normal_strength: 1.0,
            opacity: 1.0,
            opacity_texture: None,
        }
    }

    pub fn with_metallic(mut self, metallic: f32) -> Self {
        self.metallic = metallic;
        self
    }

    pub fn with_subsurface(mut self, subsurface: f32) -> Self {
        self.subsurface = subsurface;
        self
    }

    pub fn with_specular(mut self, specular: f32) -> Self {
        self.specular = specular;
        self
    }

    pub fn with_specular_tint(mut self, specular_tint: f32) -> Self {
        self.specular_tint = specular_tint;
        self
    }

    pub fn with_roughness(mut self, roughness: f32) -> Self {
        self.roughness = roughness;
        self
    }

    pub fn with_anisotropic(mut self, anisotropic: f32) -> Self {
        self.anisotropic = anisotropic;
        self
    }

    pub fn with_sheen(mut self, sheen: f32) -> Self {
        self.sheen = sheen;
        self
    }

    pub fn with_sheen_tint(mut self, sheen_tint: f32) -> Self {
        self.sheen_tint = sheen_tint;
        self
    }

    pub fn with_clearcoat(mut self, clearcoat: f32) -> Self {
        self.clearcoat = clearcoat;
        self
    }

    pub fn with_clearcoat_gloss(mut self, gloss: f32) -> Self {
        self.clearcoat_gloss = gloss;
        self
    }

    pub fn with_base_color_texture(mut self, texture: Arc<Texture>) -> Self {
        self.base_color_texture = Some(texture);
        self
    }

    pub fn with_metallic_texture(mut self, texture: Arc<ScalarTexture>) -> Self {
        self.metallic_texture = Some(texture);
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
        metallic: f32,
        roughness: f32,
        base_color_texture_path: Option<&Path>,
        metallic_texture_path: Option<&Path>,
        roughness_texture_path: Option<&Path>,
        normal_map_path: Option<&Path>,
        ocio: &crate::color::OcioColorPipeline,
    ) -> image::ImageResult<Self> {
        Ok(Self {
            base_color,
            metallic,
            roughness,
            base_color_texture: load_optional_color_texture(
                base_color_texture_path,
                TextureColorSpace::Srgb,
                ocio,
            )?,
            metallic_texture: load_optional_scalar_texture(metallic_texture_path)?,
            roughness_texture: load_optional_scalar_texture(roughness_texture_path)?,
            normal_map: load_optional_normal_map(normal_map_path)?,
            ..Self::new(base_color)
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

    fn make_bsdf(&self, shading_vertex: &ShadingVertex) -> DisneyBrdfBsdf {
        DisneyBrdfBsdf::new(
            self.base_color_at(shading_vertex),
            self.metallic_at(shading_vertex),
            self.subsurface,
            self.specular,
            self.specular_tint,
            self.roughness_at(shading_vertex),
            self.anisotropic,
            self.sheen,
            self.sheen_tint,
            self.clearcoat,
            self.clearcoat_gloss,
        )
    }

    pub fn sample(
        &self,
        shading_vertex: &ShadingVertex,
        rng: &mut ThreadRng,
    ) -> Option<MaterialSample> {
        if shading_vertex.wo.dot(shading_vertex.ng) <= 0.0 {
            return None;
        }
        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        if wo_local.z <= 0.0 {
            return None;
        }

        let bsdf = self.make_bsdf(shading_vertex);
        let sample = bsdf.sample(wo_local, rng)?;

        let wi = shading_vertex.frame.local_to_world(sample.wi);
        if wi.dot(shading_vertex.ng) <= GEOMETRIC_NORMAL_COS_EPSILON {
            return None;
        }

        let cone_spread = if sample.flags.contains(BsdfFlags::GLOSSY) {
            2.0 * self.roughness_at(shading_vertex).clamp(0.0, 1.0)
        } else {
            GLOSSY_DIFFUSE_CONE_SPREAD
        };

        Some(MaterialSample {
            weight: sample.weight,
            wi,
            pdf: sample.pdf,
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
        _internal_rng: &mut ThreadRng,
    ) -> Vec3 {
        if shading_vertex.wo.dot(shading_vertex.ng) <= 0.0 || wi.dot(shading_vertex.ng) <= 0.0 {
            return Vec3::ZERO;
        }
        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let wi_local = shading_vertex.frame.world_to_local(wi).normalize_or_zero();
        if wo_local.z <= 0.0 || wi_local.z <= 0.0 {
            return Vec3::ZERO;
        }
        self.make_bsdf(shading_vertex).eval(wo_local, wi_local)
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
        if wo_local.z <= 0.0 || wi_local.z <= 0.0 {
            return 0.0;
        }
        self.make_bsdf(shading_vertex).pdf(wo_local, wi_local)
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
    ) -> Option<LightTreePrecompute> {
        let base_color = self
            .base_color_at(shading_vertex)
            .clamp(Vec3::ZERO, Vec3::ONE);
        let metallic = self.metallic_at(shading_vertex).clamp(0.0, 1.0);
        let roughness = self.roughness_at(shading_vertex).clamp(0.0, 1.0);
        let specular = self.specular.clamp(0.0, 1.0);
        let specular_tint = self.specular_tint.clamp(0.0, 1.0);
        let anisotropic = self.anisotropic.clamp(0.0, 1.0);
        let clearcoat = self.clearcoat.clamp(0.0, 1.0);
        let clearcoat_gloss = self.clearcoat_gloss.clamp(0.0, 1.0);

        let rho_d = sg::luminance(base_color) * (1.0 - metallic);
        let diffuse = if rho_d > 0.0 {
            Some(DiffuseLobePrecompute { rho: rho_d })
        } else {
            None
        };

        let lum = sg::luminance(base_color);
        let c_tint = if lum > 0.0 {
            base_color / lum
        } else {
            Vec3::ONE
        };
        let dielectric_f0 = 0.08 * specular * Vec3::ONE.lerp(c_tint, specular_tint);
        let c_spec0 = dielectric_f0.lerp(base_color, metallic);
        let rho_s_primary = sg::luminance(c_spec0).max(0.0);
        let rho_s_cc = 0.25 * clearcoat * 0.04;

        let alpha = roughness * roughness;
        let aspect = (1.0 - 0.9 * anisotropic).sqrt();
        let alpha_primary_x = (alpha / aspect).max(1.0e-3);
        let alpha_primary_y = (alpha * aspect).max(1.0e-3);
        let alpha_cc = (0.1 * (1.0 - clearcoat_gloss) + 0.001 * clearcoat_gloss).max(1.0e-3);

        let glossy = match (rho_s_primary > 0.0, rho_s_cc > 0.0) {
            (true, true) => {
                let (alpha_x, alpha_y) = merge_glossy_roughness(
                    rho_s_primary,
                    (alpha_primary_x, alpha_primary_y),
                    rho_s_cc,
                    (alpha_cc, alpha_cc),
                );
                make_glossy_lobe(
                    rho_s_primary + rho_s_cc,
                    shading_vertex.frame,
                    shading_vertex.wo,
                    alpha_x,
                    alpha_y,
                )
            }
            (true, false) => make_glossy_lobe(
                rho_s_primary,
                shading_vertex.frame,
                shading_vertex.wo,
                alpha_primary_x,
                alpha_primary_y,
            ),
            (false, true) => make_glossy_lobe(
                rho_s_cc,
                shading_vertex.frame,
                shading_vertex.wo,
                alpha_cc,
                alpha_cc,
            ),
            (false, false) => None,
        };

        if diffuse.is_none() && glossy.is_none() {
            return None;
        }
        Some(LightTreePrecompute {
            p: shading_vertex.p,
            n: shading_vertex.ns,
            frame: shading_vertex.frame,
            diffuse,
            glossy,
            btdf: None,
        })
    }

    pub fn light_tree_importance(
        &self,
        precompute: &LightTreePrecompute,
        w: f32,
        lobe: &sg::SgLobe,
    ) -> f32 {
        let mut imp = 0.0;
        if let Some(d) = precompute.diffuse {
            imp += diffuse_importance(d, precompute.n, w, lobe);
        }
        if let Some(g) = precompute.glossy {
            imp += glossy_importance(g, precompute.frame, precompute.n, w, lobe);
        }
        imp.max(0.0)
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

    fn metallic_at(&self, shading_vertex: &ShadingVertex) -> f32 {
        self.metallic
            * self
                .metallic_texture
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

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};

    use crate::{
        material::ShadingVertex,
        math::OrthonormalBasis,
        scene::{InstanceIndex, TriangleRef},
    };

    use super::DisneyBrdfMaterial;

    fn test_shading_vertex(wo: Vec3) -> ShadingVertex {
        ShadingVertex {
            triangle: TriangleRef {
                instance_index: InstanceIndex(0),
                triangle_index: 0,
            },
            p: Vec3::ZERO,
            uv: Vec2::ZERO,
            dudx: 0.0,
            dvdx: 0.0,
            dudy: 0.0,
            dvdy: 0.0,
            ng: Vec3::Z,
            ns: Vec3::Z,
            wo,
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
    fn default_material_evaluates_to_finite_positive_response_for_normal_incidence() {
        let material = DisneyBrdfMaterial::new(Vec3::new(0.82, 0.67, 0.16));
        let vtx = test_shading_vertex(Vec3::Z);
        let mut rng = rand::rng();

        let f = material.eval(&vtx, Vec3::Z, &mut rng);
        assert!(f.is_finite());
        assert!(f.x > 0.0 && f.y > 0.0 && f.z > 0.0);
    }

    #[test]
    fn metallic_one_zero_subsurface_evaluates_with_specular_only() {
        let material = DisneyBrdfMaterial::new(Vec3::new(0.95, 0.78, 0.35))
            .with_metallic(1.0)
            .with_roughness(0.3);
        let vtx = test_shading_vertex(Vec3::new(0.3, -0.4, 0.866_025_4).normalize());
        let mut rng = rand::rng();

        let wi = Vec3::new(-vtx.wo.x, -vtx.wo.y, vtx.wo.z).normalize();
        let f = material.eval(&vtx, wi, &mut rng);
        assert!(f.is_finite());
        assert!(f.length() > 0.0);
    }

    #[test]
    fn light_tree_precompute_is_some_for_default() {
        let material = DisneyBrdfMaterial::new(Vec3::new(0.82, 0.67, 0.16));
        let vtx = test_shading_vertex(Vec3::Z);
        let pre = material.light_tree_precompute(&vtx);
        assert!(pre.is_some());
        let pre = pre.unwrap();
        assert!(pre.diffuse.is_some());
        assert!(pre.glossy.is_some());
        assert!(pre.btdf.is_none());
    }

    #[test]
    fn light_tree_precompute_pure_metallic_drops_diffuse() {
        let material = DisneyBrdfMaterial::new(Vec3::new(0.95, 0.78, 0.35))
            .with_metallic(1.0)
            .with_roughness(0.3);
        let vtx = test_shading_vertex(Vec3::Z);
        let pre = material
            .light_tree_precompute(&vtx)
            .expect("precompute should be Some for metallic case");
        assert!(pre.diffuse.is_none());
        assert!(pre.glossy.is_some());
    }
}
