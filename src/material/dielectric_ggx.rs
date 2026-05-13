use std::{path::Path, sync::Arc};

use glam::{Vec2, Vec3};
use rand::{RngExt, rngs::ThreadRng};

use crate::{
    bsdf::{BsdfFlags, DielectricGgxBsdf, DielectricGgxEnergyCompensationLut},
    color::srgb_to_linear,
};

use super::{
    GEOMETRIC_NORMAL_COS_EPSILON, MaterialSample, NormalMap, ScalarTexture, ShadingVertex, Texture,
    TextureColorSpace,
    normal_map::load_optional_normal_map,
    texture::{load_optional_color_texture, load_optional_scalar_texture},
};

const MIN_ALPHA: f32 = 1.0e-4;

#[derive(Debug, Clone, PartialEq)]
pub struct DielectricGgxMaterial {
    pub color: Vec3,
    pub color_texture: Option<Arc<Texture>>,
    pub eta: f32,
    pub roughness: f32,
    pub roughness_texture: Option<Arc<ScalarTexture>>,
    pub anisotropy: f32,
    pub thin: bool,
    pub normal_map: Option<NormalMap>,
    pub normal_strength: f32,
    pub opacity: f32,
    pub opacity_texture: Option<Arc<ScalarTexture>>,
    pub energy_compensation: bool,
    pub(crate) energy_compensation_lut: Option<Arc<DielectricGgxEnergyCompensationLut>>,
}

impl DielectricGgxMaterial {
    pub fn new(color: Vec3, eta: f32, roughness: f32, anisotropy: f32, thin: bool) -> Self {
        Self {
            color,
            color_texture: None,
            eta,
            roughness,
            roughness_texture: None,
            anisotropy,
            thin,
            normal_map: None,
            normal_strength: 1.0,
            opacity: 1.0,
            opacity_texture: None,
            energy_compensation: false,
            energy_compensation_lut: None,
        }
    }

    pub fn with_energy_compensation(mut self) -> Self {
        self.energy_compensation = true;
        self
    }

    pub(crate) fn install_energy_compensation_lut(
        &mut self,
        lut: Arc<DielectricGgxEnergyCompensationLut>,
    ) {
        self.energy_compensation_lut = Some(lut);
    }

    pub fn with_color_texture(mut self, texture: Arc<Texture>) -> Self {
        self.color_texture = Some(texture);
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
        color: Vec3,
        eta: f32,
        roughness: f32,
        anisotropy: f32,
        thin: bool,
        color_texture_path: Option<&Path>,
        roughness_texture_path: Option<&Path>,
        normal_map_path: Option<&Path>,
    ) -> image::ImageResult<Self> {
        Ok(Self {
            color,
            color_texture: load_optional_color_texture(
                color_texture_path,
                TextureColorSpace::Srgb,
            )?,
            eta,
            roughness,
            roughness_texture: load_optional_scalar_texture(roughness_texture_path)?,
            anisotropy,
            thin,
            normal_map: load_optional_normal_map(normal_map_path)?,
            normal_strength: 1.0,
            opacity: 1.0,
            opacity_texture: None,
            energy_compensation: false,
            energy_compensation_lut: None,
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
        rng: &mut ThreadRng,
    ) -> Option<MaterialSample> {
        let uc = rng.random::<f32>();
        let us = Vec2::new(rng.random::<f32>(), rng.random::<f32>());
        let sample = self.sample_impl(shading_vertex, uc, us)?;

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
        us: Vec2,
    ) -> Option<MaterialSample> {
        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let roughness = self.roughness_at(shading_vertex);
        let (alpha_x, alpha_y) = self.alpha_xy_from_roughness(roughness);
        let bsdf = self.make_bsdf(
            self.color_at(shading_vertex),
            alpha_x,
            alpha_y,
            shading_vertex.front_face,
        );
        let sample = bsdf.sample(wo_local, uc, us)?;
        let wi = shading_vertex.frame.local_to_world(sample.wi);
        let cone_spread = if sample.flags.contains(BsdfFlags::GLOSSY) {
            2.0 * roughness.clamp(0.0, 1.0)
        } else {
            0.0
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
        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let wi_local = shading_vertex.frame.world_to_local(wi).normalize_or_zero();
        let wo_side = shading_vertex.wo.dot(shading_vertex.ng);
        let wi_side = wi.dot(shading_vertex.ng);
        if wo_side <= GEOMETRIC_NORMAL_COS_EPSILON {
            return Vec3::ZERO;
        }
        if wi_local.z > 0.0 && wi_side <= GEOMETRIC_NORMAL_COS_EPSILON {
            return Vec3::ZERO;
        }
        if wi_local.z < 0.0 && wi_side >= -GEOMETRIC_NORMAL_COS_EPSILON {
            return Vec3::ZERO;
        }
        if wi_local.z == 0.0 {
            return Vec3::ZERO;
        }

        let (alpha_x, alpha_y) = self.alpha_xy_at(shading_vertex);
        let bsdf = self.make_bsdf(
            self.color_at(shading_vertex),
            alpha_x,
            alpha_y,
            shading_vertex.front_face,
        );
        bsdf.eval(wo_local, wi_local)
    }

    pub fn pdf(&self, shading_vertex: &ShadingVertex, wi: Vec3) -> f32 {
        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let wi_local = shading_vertex.frame.world_to_local(wi).normalize_or_zero();
        let wo_side = shading_vertex.wo.dot(shading_vertex.ng);
        let wi_side = wi.dot(shading_vertex.ng);
        if wo_side <= GEOMETRIC_NORMAL_COS_EPSILON {
            return 0.0;
        }
        if wi_local.z > 0.0 && wi_side <= GEOMETRIC_NORMAL_COS_EPSILON {
            return 0.0;
        }
        if wi_local.z < 0.0 && wi_side >= -GEOMETRIC_NORMAL_COS_EPSILON {
            return 0.0;
        }
        if wi_local.z == 0.0 {
            return 0.0;
        }

        let (alpha_x, alpha_y) = self.alpha_xy_at(shading_vertex);
        let bsdf = self.make_bsdf(
            self.color_at(shading_vertex),
            alpha_x,
            alpha_y,
            shading_vertex.front_face,
        );
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

    #[cfg(test)]
    fn alpha_xy(&self) -> (f32, f32) {
        self.alpha_xy_from_roughness(self.roughness)
    }

    /// Per-shading-point precompute for the hierarchical light tree.
    /// Dielectric GGX has a glossy reflection lobe and a transmission lobe
    /// (Tokuyoshi 2024 "Proxy A": treat the rough refraction cone as a
    /// glossy SG kernel pivoted around the perfect refraction direction).
    /// Both contribute additively to the importance.
    pub fn light_tree_precompute(
        &self,
        shading_vertex: &ShadingVertex,
    ) -> Option<crate::light_tree::LightTreePrecompute> {
        let alpha = self.alpha_xy_at(shading_vertex);
        let cos_o = shading_vertex.wo.dot(shading_vertex.ns).abs().max(1.0e-4);
        let f_avg = crate::math::fresnel_dielectric(cos_o, 1.0, self.eta);
        let rho_color = crate::math::sg::luminance(self.color_at(shading_vertex));
        let rho_refl = rho_color * f_avg;
        let rho_trans = rho_color * (1.0 - f_avg);
        let glossy = crate::light_tree::make_glossy_lobe(
            rho_refl,
            shading_vertex.frame,
            shading_vertex.wo,
            alpha.0,
            alpha.1,
        );
        let btdf = if rho_trans > 0.0 {
            let eta_rel = if shading_vertex.front_face {
                1.0 / self.eta
            } else {
                self.eta
            };
            crate::light_tree::make_btdf_lobe(
                rho_trans,
                shading_vertex.frame,
                shading_vertex.wo,
                alpha.0,
                alpha.1,
                eta_rel,
            )
        } else {
            None
        };

        let diffuse = if let Some(lut) = self.energy_compensation_lut.as_ref() {
            let eta_o = if shading_vertex.front_face {
                self.eta
            } else {
                1.0 / self.eta
            };
            let roughness_eq = (alpha.0 * alpha.1).powf(0.25);
            let e_avg = lut.lookup_e_avg(roughness_eq, eta_o);
            let f_avg_o = if eta_o >= 1.0 {
                ((eta_o - 1.0) / (4.08567 + 1.00071 * eta_o)).clamp(0.0, 1.0)
            } else {
                (0.997118 + 0.1014 * eta_o
                    - 0.965241 * eta_o * eta_o
                    - 0.130607 * eta_o * eta_o * eta_o)
                    .clamp(0.0, 1.0)
            };
            let rho_ms_r = rho_color * f_avg_o * (1.0 - e_avg).max(0.0);
            if rho_ms_r > 0.0 {
                Some(crate::light_tree::DiffuseLobePrecompute { rho: rho_ms_r })
            } else {
                None
            }
        } else {
            None
        };

        if glossy.is_none() && btdf.is_none() && diffuse.is_none() {
            return None;
        }
        Some(crate::light_tree::LightTreePrecompute {
            p: shading_vertex.p,
            n: shading_vertex.ns,
            frame: shading_vertex.frame,
            diffuse,
            glossy,
            btdf,
        })
    }

    pub fn light_tree_importance(
        &self,
        precompute: &crate::light_tree::LightTreePrecompute,
        w: f32,
        lobe: &crate::math::sg::SgLobe,
    ) -> f32 {
        let mut imp = 0.0;
        if let Some(g) = precompute.glossy {
            imp += crate::light_tree::glossy_importance(g, precompute.frame, precompute.n, w, lobe);
        }
        if let Some(b) = precompute.btdf {
            imp += crate::light_tree::btdf_importance(b, precompute.frame, precompute.n, w, lobe);
        }
        if let Some(d) = precompute.diffuse {
            imp += crate::light_tree::diffuse_importance(d, precompute.n, w, lobe);
        }
        imp.max(0.0)
    }

    fn make_bsdf(
        &self,
        color: Vec3,
        alpha_x: f32,
        alpha_y: f32,
        front_face: bool,
    ) -> DielectricGgxBsdf {
        if !self.thin
            && let Some(lut) = self.energy_compensation_lut.as_ref()
        {
            DielectricGgxBsdf::new_with_energy_compensation(
                color,
                self.eta,
                alpha_x,
                alpha_y,
                self.thin,
                front_face,
                Arc::clone(lut),
            )
        } else {
            DielectricGgxBsdf::new(color, self.eta, alpha_x, alpha_y, self.thin, front_face)
        }
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
    use std::sync::Arc;

    use glam::{Vec2, Vec3};

    use crate::{
        bsdf::BsdfFlags,
        material::{ScalarTexture, ShadingVertex, Texture},
        math::OrthonormalBasis,
        scene::{InstanceIndex, TriangleRef},
    };

    use super::DielectricGgxMaterial;

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
    fn alpha_mapping_matches_isotropic_case() {
        let material = DielectricGgxMaterial::new(Vec3::ONE, 1.5, 0.5, 0.0, false);
        let (alpha_x, alpha_y) = material.alpha_xy();

        assert!((alpha_x - 0.25).abs() < 1.0e-6);
        assert!((alpha_y - 0.25).abs() < 1.0e-6);
    }

    #[test]
    fn signed_anisotropy_flips_alpha_axes() {
        let positive = DielectricGgxMaterial::new(Vec3::ONE, 1.5, 0.4, 0.8, false);
        let negative = DielectricGgxMaterial::new(Vec3::ONE, 1.5, 0.4, -0.8, false);
        let (pos_x, pos_y) = positive.alpha_xy();
        let (neg_x, neg_y) = negative.alpha_xy();

        assert!((pos_x - neg_y).abs() < 1.0e-6);
        assert!((pos_y - neg_x).abs() < 1.0e-6);
        assert!(pos_x > pos_y);
    }

    #[test]
    fn sample_returns_reflection_or_transmission_flag() {
        let material =
            DielectricGgxMaterial::new(Vec3::new(0.85, 0.95, 0.95), 1.5, 0.3, 0.0, false);
        let vtx = test_shading_vertex(Vec3::new(0.2, -0.1, 0.9746794).normalize());
        let mut rng = rand::rng();

        let mut saw_reflection = false;
        let mut saw_transmission = false;
        for _ in 0..256 {
            if let Some(sample) = material.sample(&vtx, &mut rng) {
                if sample.flags.contains(BsdfFlags::REFLECTION) {
                    saw_reflection = true;
                }
                if sample.flags.contains(BsdfFlags::TRANSMISSION) {
                    saw_transmission = true;
                }
                if saw_reflection && saw_transmission {
                    break;
                }
            }
        }

        assert!(saw_reflection, "expected at least one reflection sample");
        assert!(
            saw_transmission,
            "expected at least one transmission sample"
        );
    }

    #[test]
    fn sample_at_back_face_returns_some() {
        let material = DielectricGgxMaterial::new(Vec3::ONE, 1.5, 0.3, 0.0, false);
        let mut vtx = test_shading_vertex(Vec3::Z);
        vtx.front_face = false;
        let mut rng = rand::rng();

        let sample = material
            .sample(&vtx, &mut rng)
            .expect("expected a back-face sample");
        assert!(
            sample
                .flags
                .intersects(BsdfFlags::REFLECTION | BsdfFlags::TRANSMISSION)
        );
    }

    #[test]
    fn textures_modulate_color_and_roughness() {
        let material = DielectricGgxMaterial {
            color: Vec3::ONE,
            color_texture: Some(Arc::new(Texture::from_pixels(
                1,
                1,
                vec![Vec3::new(0.2, 0.4, 0.6)],
            ))),
            eta: 1.5,
            roughness: 0.8,
            roughness_texture: Some(Arc::new(ScalarTexture::from_pixels(1, 1, vec![0.5]))),
            anisotropy: 0.0,
            thin: false,
            normal_map: None,
            normal_strength: 1.0,
            opacity: 1.0,
            opacity_texture: None,
            energy_compensation: false,
            energy_compensation_lut: None,
        };
        let vtx = test_shading_vertex(Vec3::Z);
        let (alpha_x, alpha_y) = material.alpha_xy_at(&vtx);

        assert!(
            material
                .color_at(&vtx)
                .abs_diff_eq(Vec3::new(0.2, 0.4, 0.6), 1.0e-6)
        );
        assert!((alpha_x - 0.16).abs() < 1.0e-6);
        assert!((alpha_y - 0.16).abs() < 1.0e-6);
    }
}
