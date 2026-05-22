use std::sync::Arc;

use glam::Vec3;

use crate::{
    bsdf::{
        BsdfFlags, DielectricGgxDirectionalAlbedoLut, SheenDirectionalAlbedoLut,
        StandardSurfaceBsdf, StandardSurfaceBsdfParams, artist_friendly_complex_ior,
    },
    light_tree::{
        DiffuseLobePrecompute, LightTreePrecompute, btdf_importance, diffuse_importance,
        glossy_importance, make_glossy_lobe, merge_glossy_roughness,
    },
    math::{OrthonormalBasis, sg},
    sampler::{AuxRng, MaterialSampleRandoms},
};

use super::{
    GEOMETRIC_NORMAL_COS_EPSILON, MaterialSample, NormalMap, ScalarTexture, ShadingVertex, Texture,
};

const MIN_ALPHA: f32 = 1.0e-4;
const GLOSSY_DIFFUSE_CONE_SPREAD: f32 = 0.5;

#[derive(Debug, Clone, PartialEq)]
pub struct StandardSurfaceMaterial {
    pub base_color: Vec3,
    pub base: f32,
    pub diffuse_roughness: f32,
    pub metalness: f32,
    pub specular: f32,
    pub specular_color: Vec3,
    pub specular_roughness: f32,
    pub specular_ior: f32,
    pub specular_anisotropy: f32,
    pub specular_rotation: f32,
    pub transmission: f32,
    pub transmission_color: Vec3,
    pub transmission_depth: f32,
    pub transmission_extra_roughness: f32,
    pub transmission_dispersion: f32,
    pub transmission_scatter: Vec3,
    pub subsurface: f32,
    pub subsurface_color: Vec3,
    pub coat: f32,
    pub coat_color: Vec3,
    pub coat_roughness: f32,
    pub coat_anisotropy: f32,
    pub coat_rotation: f32,
    pub coat_ior: f32,
    pub coat_affect_color: f32,
    pub coat_affect_roughness: f32,
    pub sheen: f32,
    pub sheen_color: Vec3,
    pub sheen_roughness: f32,
    pub thin_film_thickness: f32,
    pub thin_film_ior: f32,
    pub thin_walled: bool,
    pub emission: f32,
    pub emission_color: Vec3,
    pub opacity: f32,

    pub base_color_texture: Option<Arc<Texture>>,
    pub specular_roughness_texture: Option<Arc<ScalarTexture>>,
    pub metalness_texture: Option<Arc<ScalarTexture>>,
    pub opacity_texture: Option<Arc<ScalarTexture>>,
    pub emission_color_texture: Option<Arc<Texture>>,
    pub normal_map: Option<NormalMap>,
    pub normal_strength: f32,
    pub coat_normal_map: Option<NormalMap>,
    pub coat_normal_strength: f32,

    spec_lut: Option<Arc<DielectricGgxDirectionalAlbedoLut>>,
    coat_lut: Option<Arc<DielectricGgxDirectionalAlbedoLut>>,
    sheen_lut: Option<Arc<SheenDirectionalAlbedoLut>>,
}

impl StandardSurfaceMaterial {
    pub fn new(base_color: Vec3) -> Self {
        Self {
            base_color,
            base: 0.8,
            diffuse_roughness: 0.0,
            metalness: 0.0,
            specular: 1.0,
            specular_color: Vec3::ONE,
            specular_roughness: 0.2,
            specular_ior: 1.5,
            specular_anisotropy: 0.0,
            specular_rotation: 0.0,
            transmission: 0.0,
            transmission_color: Vec3::ONE,
            transmission_depth: 0.0,
            transmission_extra_roughness: 0.0,
            transmission_dispersion: 0.0,
            transmission_scatter: Vec3::ZERO,
            subsurface: 0.0,
            subsurface_color: Vec3::ONE,
            coat: 0.0,
            coat_color: Vec3::ONE,
            coat_roughness: 0.1,
            coat_anisotropy: 0.0,
            coat_rotation: 0.0,
            coat_ior: 1.5,
            coat_affect_color: 0.0,
            coat_affect_roughness: 0.0,
            sheen: 0.0,
            sheen_color: Vec3::ONE,
            sheen_roughness: 0.3,
            thin_film_thickness: 0.0,
            thin_film_ior: 1.5,
            thin_walled: false,
            emission: 0.0,
            emission_color: Vec3::ONE,
            opacity: 1.0,
            base_color_texture: None,
            specular_roughness_texture: None,
            metalness_texture: None,
            opacity_texture: None,
            emission_color_texture: None,
            normal_map: None,
            normal_strength: 1.0,
            coat_normal_map: None,
            coat_normal_strength: 1.0,
            spec_lut: None,
            coat_lut: None,
            sheen_lut: None,
        }
    }

    pub fn with_base(mut self, v: f32) -> Self {
        self.base = v;
        self
    }
    pub fn with_diffuse_roughness(mut self, v: f32) -> Self {
        self.diffuse_roughness = v;
        self
    }
    pub fn with_metalness(mut self, v: f32) -> Self {
        self.metalness = v;
        self
    }
    pub fn with_specular(mut self, v: f32) -> Self {
        self.specular = v;
        self
    }
    pub fn with_specular_color(mut self, v: Vec3) -> Self {
        self.specular_color = v;
        self
    }
    pub fn with_specular_roughness(mut self, v: f32) -> Self {
        self.specular_roughness = v;
        self
    }
    pub fn with_specular_ior(mut self, v: f32) -> Self {
        self.specular_ior = v;
        self
    }
    pub fn with_specular_anisotropy(mut self, v: f32) -> Self {
        self.specular_anisotropy = v;
        self
    }
    pub fn with_specular_rotation(mut self, v: f32) -> Self {
        self.specular_rotation = v;
        self
    }
    pub fn with_transmission(mut self, v: f32) -> Self {
        self.transmission = v;
        self
    }
    pub fn with_transmission_color(mut self, v: Vec3) -> Self {
        self.transmission_color = v;
        self
    }
    pub fn with_transmission_depth(mut self, v: f32) -> Self {
        self.transmission_depth = v;
        self
    }
    pub fn with_transmission_extra_roughness(mut self, v: f32) -> Self {
        self.transmission_extra_roughness = v;
        self
    }
    pub fn with_transmission_dispersion(mut self, v: f32) -> Self {
        self.transmission_dispersion = v;
        self
    }
    pub fn with_transmission_scatter(mut self, v: Vec3) -> Self {
        self.transmission_scatter = v;
        self
    }
    pub fn with_subsurface(mut self, v: f32) -> Self {
        self.subsurface = v;
        self
    }
    pub fn with_subsurface_color(mut self, v: Vec3) -> Self {
        self.subsurface_color = v;
        self
    }
    pub fn with_coat(mut self, v: f32) -> Self {
        self.coat = v;
        self
    }
    pub fn with_coat_color(mut self, v: Vec3) -> Self {
        self.coat_color = v;
        self
    }
    pub fn with_coat_roughness(mut self, v: f32) -> Self {
        self.coat_roughness = v;
        self
    }
    pub fn with_coat_anisotropy(mut self, v: f32) -> Self {
        self.coat_anisotropy = v;
        self
    }
    pub fn with_coat_rotation(mut self, v: f32) -> Self {
        self.coat_rotation = v;
        self
    }
    pub fn with_coat_ior(mut self, v: f32) -> Self {
        self.coat_ior = v;
        self
    }
    pub fn with_coat_affect_color(mut self, v: f32) -> Self {
        self.coat_affect_color = v;
        self
    }
    pub fn with_coat_affect_roughness(mut self, v: f32) -> Self {
        self.coat_affect_roughness = v;
        self
    }
    pub fn with_sheen(mut self, v: f32) -> Self {
        self.sheen = v;
        self
    }
    pub fn with_sheen_color(mut self, v: Vec3) -> Self {
        self.sheen_color = v;
        self
    }
    pub fn with_sheen_roughness(mut self, v: f32) -> Self {
        self.sheen_roughness = v;
        self
    }
    pub fn with_thin_film_thickness(mut self, v: f32) -> Self {
        self.thin_film_thickness = v;
        self
    }
    pub fn with_thin_film_ior(mut self, v: f32) -> Self {
        self.thin_film_ior = v;
        self
    }
    pub fn with_thin_walled(mut self, v: bool) -> Self {
        self.thin_walled = v;
        self
    }
    pub fn with_emission(mut self, v: f32) -> Self {
        self.emission = v;
        self
    }
    pub fn with_emission_color(mut self, v: Vec3) -> Self {
        self.emission_color = v;
        self
    }
    pub fn with_opacity(mut self, v: f32) -> Self {
        self.opacity = v;
        self
    }

    pub fn with_base_color_texture(mut self, t: Arc<Texture>) -> Self {
        self.base_color_texture = Some(t);
        self
    }
    pub fn with_specular_roughness_texture(mut self, t: Arc<ScalarTexture>) -> Self {
        self.specular_roughness_texture = Some(t);
        self
    }
    pub fn with_metalness_texture(mut self, t: Arc<ScalarTexture>) -> Self {
        self.metalness_texture = Some(t);
        self
    }
    pub fn with_opacity_texture(mut self, t: Arc<ScalarTexture>) -> Self {
        self.opacity_texture = Some(t);
        self
    }
    pub fn with_emission_color_texture(mut self, t: Arc<Texture>) -> Self {
        self.emission_color_texture = Some(t);
        self
    }
    pub fn with_normal_map(mut self, m: NormalMap) -> Self {
        self.normal_map = Some(m);
        self
    }
    pub fn with_normal_strength(mut self, v: f32) -> Self {
        self.normal_strength = v;
        self
    }
    pub fn with_coat_normal_map(mut self, m: NormalMap) -> Self {
        self.coat_normal_map = Some(m);
        self
    }
    pub fn with_coat_normal_strength(mut self, v: f32) -> Self {
        self.coat_normal_strength = v;
        self
    }

    pub(crate) fn install_spec_lut(&mut self, lut: Arc<DielectricGgxDirectionalAlbedoLut>) {
        self.spec_lut = Some(lut);
    }
    pub(crate) fn install_coat_lut(&mut self, lut: Arc<DielectricGgxDirectionalAlbedoLut>) {
        self.coat_lut = Some(lut);
    }
    pub(crate) fn install_sheen_lut(&mut self, lut: Arc<SheenDirectionalAlbedoLut>) {
        self.sheen_lut = Some(lut);
    }

    pub(crate) fn requires_specular_eta(&self) -> f32 {
        self.specular_ior
    }
    pub(crate) fn requires_coat_eta(&self) -> f32 {
        self.coat_ior
    }

    pub fn validate_and_warn(&mut self) {
        if self.thin_walled
            && (self.specular_roughness > 0.0
                || self.specular_anisotropy != 0.0
                || self.transmission_extra_roughness > 0.0)
        {
            tracing::warn!(
                "[StandardSurface] thin_walled=true ignores specular_roughness/anisotropy/transmission_extra_roughness; treating specular as smooth."
            );
            self.specular_roughness = 0.0;
            self.specular_anisotropy = 0.0;
            self.transmission_extra_roughness = 0.0;
        }
        if !self.thin_walled && self.subsurface > 0.0 {
            tracing::warn!(
                "[StandardSurface] subsurface > 0 with thin_walled=false requires volumetric SSS, which is not supported; subsurface forced to 0."
            );
            self.subsurface = 0.0;
        }
        if self.transmission_depth > 0.0 {
            tracing::warn!(
                "[StandardSurface] transmission_depth > 0 (Beer's law absorption) is not supported; ignoring (treating as 0)."
            );
            self.transmission_depth = 0.0;
        }
        if self.transmission_scatter.length_squared() > 0.0 {
            tracing::warn!("[StandardSurface] transmission_scatter is not supported; ignoring.");
            self.transmission_scatter = Vec3::ZERO;
        }
    }

    pub fn opacity_at_uv(&self, shading_vertex: &ShadingVertex) -> f32 {
        let texture_factor = self
            .opacity_texture
            .as_ref()
            .map(|t| {
                t.sample_filtered(
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

    pub fn le(&self, shading_vertex: &ShadingVertex) -> Option<Vec3> {
        if self.emission <= 0.0 {
            return None;
        }
        let texture_factor = self
            .emission_color_texture
            .as_ref()
            .map(|t| {
                t.sample_filtered(
                    shading_vertex.uv,
                    shading_vertex.uv_dx(),
                    shading_vertex.uv_dy(),
                )
            })
            .unwrap_or(Vec3::ONE);
        Some(self.emission_color * self.emission * texture_factor)
    }

    pub fn may_emit(&self) -> bool {
        self.emission > 0.0
    }

    pub fn max_emission(&self) -> f32 {
        if self.emission <= 0.0 {
            return 0.0;
        }
        let texture_factor = self
            .emission_color_texture
            .as_ref()
            .map(|t| t.max_value())
            .unwrap_or(1.0);
        ((self.emission_color * self.emission).max_element() * texture_factor).max(0.0)
    }

    fn make_bsdf(&self, shading_vertex: &ShadingVertex) -> StandardSurfaceBsdf {
        let base_color = self.base_color_at(shading_vertex);
        let metalness = self.metalness_at(shading_vertex);
        let specular_roughness = self.specular_roughness_at(shading_vertex);

        let coat_factor = self.coat.clamp(0.0, 1.0);
        let coat_affect_color = self.coat_affect_color.clamp(0.0, 1.0);
        let coat_affect_roughness = self.coat_affect_roughness.clamp(0.0, 1.0);
        let coat_roughness_clamped = self.coat_roughness.clamp(0.0, 1.0);

        let color_pow = 1.0 + coat_factor * coat_affect_color;
        let modulated_base_color = vec3_pow(base_color, color_pow);
        let modulated_subsurface_color = vec3_pow(self.subsurface_color, color_pow);

        let roughness_modulator = coat_factor * coat_affect_roughness * coat_roughness_clamped;
        let modulated_spec_roughness = lerp(specular_roughness, 1.0, roughness_modulator);
        let modulated_btdf_extra =
            lerp(self.transmission_extra_roughness, 1.0, roughness_modulator);

        let (spec_alpha_x, spec_alpha_y) =
            alpha_xy_from_roughness(modulated_spec_roughness, self.specular_anisotropy);
        let btdf_roughness = (modulated_spec_roughness + modulated_btdf_extra).clamp(0.0, 1.0);
        let (btdf_alpha_x, btdf_alpha_y) =
            alpha_xy_from_roughness(btdf_roughness, self.specular_anisotropy);
        let (coat_alpha_x, coat_alpha_y) =
            alpha_xy_from_roughness(coat_roughness_clamped, self.coat_anisotropy);

        let (metal_n, metal_k) = artist_friendly_complex_ior(
            base_color.clamp(Vec3::ZERO, Vec3::ONE),
            self.specular_color.clamp(Vec3::ZERO, Vec3::ONE),
        );

        let coat_basis = self.coat_basis_in_base(shading_vertex);

        let params = StandardSurfaceBsdfParams {
            base_color: modulated_base_color,
            base: self.base,
            specular: self.specular,
            specular_color: self.specular_color,
            specular_alpha_x: spec_alpha_x,
            specular_alpha_y: spec_alpha_y,
            specular_eta: self.specular_ior,
            metalness,
            metal_n,
            metal_k,
            coat: coat_factor,
            coat_color: self.coat_color,
            coat_alpha_x,
            coat_alpha_y,
            coat_eta: self.coat_ior,
            sheen: self.sheen,
            sheen_color: self.sheen_color,
            sheen_roughness: self.sheen_roughness,
            transmission: self.transmission,
            transmission_color: self.transmission_color,
            transmission_alpha_x: btdf_alpha_x,
            transmission_alpha_y: btdf_alpha_y,
            transmission_dispersion_abbe: self.transmission_dispersion,
            diffuse_roughness: self.diffuse_roughness,
            subsurface: self.subsurface,
            subsurface_color: modulated_subsurface_color,
            thin_walled: self.thin_walled,
            thin_film_thickness: self.thin_film_thickness,
            thin_film_ior: self.thin_film_ior,
            front_face: shading_vertex.front_face,
            coat_basis_in_base: coat_basis,
            path_throughput: shading_vertex.path_throughput,
            wavelength_lock: shading_vertex.wavelength_lock,
        };

        StandardSurfaceBsdf::new(
            params,
            self.spec_lut
                .as_ref()
                .expect("StandardSurfaceMaterial requires spec_lut")
                .clone(),
            self.coat_lut
                .as_ref()
                .expect("StandardSurfaceMaterial requires coat_lut")
                .clone(),
            self.sheen_lut
                .as_ref()
                .expect("StandardSurfaceMaterial requires sheen_lut")
                .clone(),
        )
    }

    pub fn sample(
        &self,
        shading_vertex: &ShadingVertex,
        randoms: &MaterialSampleRandoms,
        _aux_rng: &mut AuxRng,
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
        let sample = bsdf.sample(wo_local, randoms)?;
        let wi = shading_vertex.frame.local_to_world(sample.wi);
        let cone_spread = if sample.flags.contains(BsdfFlags::GLOSSY) {
            2.0 * self.specular_roughness_at(shading_vertex).clamp(0.0, 1.0)
        } else {
            GLOSSY_DIFFUSE_CONE_SPREAD
        };
        if sample.flags.contains(BsdfFlags::TRANSMISSION) {
            if wi.dot(shading_vertex.ng) >= -GEOMETRIC_NORMAL_COS_EPSILON
                && !sample.flags.contains(BsdfFlags::DELTA)
            {
                return None;
            }
        } else if wi.dot(shading_vertex.ng) <= GEOMETRIC_NORMAL_COS_EPSILON {
            return None;
        }
        Some(MaterialSample {
            weight: sample.weight,
            wi,
            pdf: sample.pdf,
            flags: sample.flags,
            eta: sample.eta,
            cone_spread,
            wavelength_lock: sample.wavelength_lock,
        })
    }

    pub fn eval(&self, shading_vertex: &ShadingVertex, wi: Vec3, _aux_rng: &mut AuxRng) -> Vec3 {
        if shading_vertex.wo.dot(shading_vertex.ng) <= 0.0 {
            return Vec3::ZERO;
        }
        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let wi_local = shading_vertex.frame.world_to_local(wi).normalize_or_zero();
        if wo_local.z <= 0.0 {
            return Vec3::ZERO;
        }
        self.make_bsdf(shading_vertex).eval(wo_local, wi_local)
    }

    pub fn pdf(&self, shading_vertex: &ShadingVertex, wi: Vec3) -> f32 {
        if shading_vertex.wo.dot(shading_vertex.ng) <= 0.0 {
            return 0.0;
        }
        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let wi_local = shading_vertex.frame.world_to_local(wi).normalize_or_zero();
        if wo_local.z <= 0.0 {
            return 0.0;
        }
        self.make_bsdf(shading_vertex).pdf(wo_local, wi_local)
    }

    pub fn light_tree_precompute(
        &self,
        shading_vertex: &ShadingVertex,
    ) -> Option<LightTreePrecompute> {
        let base_color = self.base_color_at(shading_vertex);
        let metalness = self.metalness_at(shading_vertex);
        let spec_rough = self.specular_roughness_at(shading_vertex);

        let (spec_alpha_x, spec_alpha_y) =
            alpha_xy_from_roughness(spec_rough, self.specular_anisotropy);
        let (coat_alpha_x, coat_alpha_y) =
            alpha_xy_from_roughness(self.coat_roughness, self.coat_anisotropy);

        let coat_color_lin = self.coat_color;
        let under_coat = lerp_vec3(Vec3::ONE, coat_color_lin, self.coat * 0.5);

        let base_color_lin = base_color;
        let rho_diffuse = (1.0 - metalness)
            * (1.0 - self.subsurface * (if self.thin_walled { 1.0 } else { 0.0 }))
            * self.base
            * sg::luminance(base_color_lin * under_coat);
        let diffuse = if rho_diffuse > 0.0 {
            Some(DiffuseLobePrecompute { rho: rho_diffuse })
        } else {
            None
        };

        let rho_metal = metalness * sg::luminance(base_color_lin * under_coat);
        let spec_color_lin = self.specular_color;
        let f0_dielectric = ((self.specular_ior - 1.0) / (self.specular_ior + 1.0)).powi(2);
        let rho_spec = (1.0 - metalness)
            * self.specular
            * sg::luminance(spec_color_lin * under_coat)
            * f0_dielectric;
        let rho_coat = self.coat * 0.04;

        let mut merged: Option<(f32, (f32, f32))> = None;
        let mut push_lobe = |rho: f32, alpha: (f32, f32)| {
            if rho <= 0.0 {
                return;
            }
            merged = match merged {
                Some((cur_rho, cur_alpha)) => {
                    let merged_alpha = merge_glossy_roughness(cur_rho, cur_alpha, rho, alpha);
                    Some((cur_rho + rho, merged_alpha))
                }
                None => Some((rho, alpha)),
            };
        };
        push_lobe(rho_metal, (spec_alpha_x, spec_alpha_y));
        push_lobe(rho_spec, (spec_alpha_x, spec_alpha_y));
        push_lobe(rho_coat, (coat_alpha_x, coat_alpha_y));

        let glossy = match merged {
            Some((rho, (alpha_x, alpha_y))) => make_glossy_lobe(
                rho,
                shading_vertex.frame,
                shading_vertex.wo,
                alpha_x,
                alpha_y,
            ),
            None => None,
        };

        let btdf = if !self.thin_walled && self.transmission > 0.0 {
            let trans_color_lin = self.transmission_color;
            let rho_t =
                self.transmission * sg::luminance(trans_color_lin * under_coat) * (1.0 - metalness);
            let trans_rough = (spec_rough + self.transmission_extra_roughness).clamp(0.0, 1.0);
            let (trans_ax, trans_ay) =
                alpha_xy_from_roughness(trans_rough, self.specular_anisotropy);
            let eta_rel = if shading_vertex.front_face {
                1.0 / self.specular_ior
            } else {
                self.specular_ior
            };
            crate::light_tree::make_btdf_lobe(
                rho_t,
                shading_vertex.frame,
                shading_vertex.wo,
                trans_ax,
                trans_ay,
                eta_rel,
            )
        } else {
            None
        };

        if diffuse.is_none() && glossy.is_none() && btdf.is_none() {
            return None;
        }
        Some(LightTreePrecompute {
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
        if let Some(b) = precompute.btdf {
            imp += btdf_importance(b, precompute.frame, precompute.n, w, lobe);
        }
        imp.max(0.0)
    }

    fn coat_basis_in_base(&self, shading_vertex: &ShadingVertex) -> Option<OrthonormalBasis> {
        let normal_map = self.coat_normal_map.as_ref()?;
        let mapped_ns = normal_map.mapped_ns(shading_vertex, self.coat_normal_strength)?;
        let coat_normal_local = shading_vertex
            .frame
            .world_to_local(mapped_ns)
            .normalize_or_zero();
        if coat_normal_local.length_squared() == 0.0 || coat_normal_local.z <= 0.0 {
            return None;
        }
        Some(OrthonormalBasis::from_normal(coat_normal_local))
    }

    fn base_color_at(&self, shading_vertex: &ShadingVertex) -> Vec3 {
        self.base_color
            * self
                .base_color_texture
                .as_ref()
                .map(|t| {
                    t.sample_filtered(
                        shading_vertex.uv,
                        shading_vertex.uv_dx(),
                        shading_vertex.uv_dy(),
                    )
                })
                .unwrap_or(Vec3::ONE)
    }

    fn metalness_at(&self, shading_vertex: &ShadingVertex) -> f32 {
        self.metalness
            * self
                .metalness_texture
                .as_ref()
                .map(|t| {
                    t.sample_filtered(
                        shading_vertex.uv,
                        shading_vertex.uv_dx(),
                        shading_vertex.uv_dy(),
                    )
                })
                .unwrap_or(1.0)
    }

    fn specular_roughness_at(&self, shading_vertex: &ShadingVertex) -> f32 {
        self.specular_roughness
            * self
                .specular_roughness_texture
                .as_ref()
                .map(|t| {
                    t.sample_filtered(
                        shading_vertex.uv,
                        shading_vertex.uv_dx(),
                        shading_vertex.uv_dy(),
                    )
                })
                .unwrap_or(1.0)
    }
}

fn alpha_xy_from_roughness(roughness: f32, anisotropy: f32) -> (f32, f32) {
    let roughness = roughness.clamp(0.0, 1.0);
    let anisotropy = anisotropy.clamp(-1.0, 1.0);
    let alpha = roughness * roughness;
    let aspect = (1.0 - 0.9 * anisotropy.abs()).sqrt();
    let (alpha_x, alpha_y) = if anisotropy >= 0.0 {
        (alpha / aspect, alpha * aspect)
    } else {
        (alpha * aspect, alpha / aspect)
    };
    (alpha_x.clamp(MIN_ALPHA, 1.0), alpha_y.clamp(MIN_ALPHA, 1.0))
}

fn vec3_pow(v: Vec3, p: f32) -> Vec3 {
    Vec3::new(
        v.x.max(0.0).powf(p),
        v.y.max(0.0).powf(p),
        v.z.max(0.0).powf(p),
    )
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn lerp_vec3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use glam::{Vec2, Vec3};

    use crate::{
        bsdf::{DielectricGgxDirectionalAlbedoLut, SheenDirectionalAlbedoLut},
        material::ShadingVertex,
        math::OrthonormalBasis,
        scene::{InstanceIndex, TriangleRef},
    };

    use super::StandardSurfaceMaterial;

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

    fn material_with_luts() -> StandardSurfaceMaterial {
        let mut m = StandardSurfaceMaterial::new(Vec3::splat(0.8));
        m.install_spec_lut(Arc::new(
            DielectricGgxDirectionalAlbedoLut::constant_for_tests(1.5, 0.04),
        ));
        m.install_coat_lut(Arc::new(
            DielectricGgxDirectionalAlbedoLut::constant_for_tests(1.5, 0.04),
        ));
        m.install_sheen_lut(Arc::new(SheenDirectionalAlbedoLut::constant_for_tests(0.3)));
        m
    }

    #[test]
    fn default_does_not_emit() {
        let m = material_with_luts();
        assert!(!m.may_emit());
        assert_eq!(m.max_emission(), 0.0);
    }

    #[test]
    fn emission_makes_material_emissive() {
        let mut m = material_with_luts();
        m.emission = 4.0;
        m.emission_color = Vec3::ONE;
        let v = test_shading_vertex(Vec3::Z);
        assert!(m.may_emit());
        assert!(m.max_emission() > 0.0);
        let le = m.le(&v).unwrap();
        assert!(le.x > 0.0);
    }

    #[test]
    fn validate_clears_unsupported_combinations() {
        let mut m = material_with_luts();
        m.thin_walled = false;
        m.subsurface = 1.0;
        m.transmission_depth = 5.0;
        m.validate_and_warn();
        assert_eq!(m.subsurface, 0.0);
        assert_eq!(m.transmission_depth, 0.0);
    }

    #[test]
    fn evaluates_finite_for_default_setup() {
        let m = material_with_luts();
        let v = test_shading_vertex(Vec3::Z);
        let f = m.eval(
            &v,
            Vec3::new(0.2, 0.3, 0.9327379).normalize(),
            &mut crate::sampler::AuxRng::default(),
        );
        assert!(f.is_finite());
    }
}
