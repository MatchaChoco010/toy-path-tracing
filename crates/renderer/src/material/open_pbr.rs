use std::sync::Arc;

use glam::Vec3;
use rand::rngs::ThreadRng;

use crate::{
    bsdf::{
        BsdfFlags, ConductorGgxEnergyCompensationLut, DielectricGgxDirectionalAlbedoLut,
        DielectricGgxEnergyCompensationLut, OpenPbrBsdf, OpenPbrBsdfParams,
        artist_friendly_complex_ior,
    },
    math::{OrthonormalBasis, fresnel_dielectric, sg},
};

use super::{
    GEOMETRIC_NORMAL_COS_EPSILON, MaterialSample, NormalMap, ScalarTexture, ShadingVertex, Texture,
};

const MIN_ALPHA: f32 = 1.0e-4;
const GLOSSY_DIFFUSE_CONE_SPREAD: f32 = 0.5;

#[derive(Debug, Clone, PartialEq)]
pub struct OpenPbrMaterial {
    pub base_weight: f32,
    pub base_color: Vec3,
    pub base_diffuse_roughness: f32,
    pub base_metalness: f32,
    pub specular_weight: f32,
    pub specular_color: Vec3,
    pub specular_roughness: f32,
    pub specular_ior: f32,
    pub specular_roughness_anisotropy: f32,
    pub transmission_weight: f32,
    pub transmission_color: Vec3,
    pub transmission_depth: f32,
    pub transmission_scatter: Vec3,
    pub transmission_scatter_anisotropy: f32,
    pub transmission_dispersion_scale: f32,
    pub transmission_dispersion_abbe_number: f32,
    pub subsurface_weight: f32,
    pub subsurface_color: Vec3,
    pub subsurface_radius: f32,
    pub subsurface_radius_scale: Vec3,
    pub subsurface_scatter_anisotropy: f32,
    pub fuzz_weight: f32,
    pub fuzz_color: Vec3,
    pub fuzz_roughness: f32,
    pub coat_weight: f32,
    pub coat_color: Vec3,
    pub coat_roughness: f32,
    pub coat_roughness_anisotropy: f32,
    pub coat_ior: f32,
    pub coat_darkening: f32,
    pub thin_film_weight: f32,
    pub thin_film_thickness: f32,
    pub thin_film_ior: f32,
    pub thin_film_thickness_texture: Option<Arc<ScalarTexture>>,
    pub thin_film_thickness_min_nm: f32,
    pub thin_film_thickness_max_nm: f32,
    pub emission_luminance: f32,
    pub emission_color: Vec3,
    pub geometry_opacity: f32,
    pub geometry_thin_walled: bool,

    pub base_color_texture: Option<Arc<Texture>>,
    pub specular_roughness_texture: Option<Arc<ScalarTexture>>,
    pub fuzz_weight_texture: Option<Arc<ScalarTexture>>,
    pub fuzz_roughness_texture: Option<Arc<ScalarTexture>>,
    pub base_metalness_texture: Option<Arc<ScalarTexture>>,
    pub geometry_opacity_texture: Option<Arc<ScalarTexture>>,
    pub emission_color_texture: Option<Arc<Texture>>,
    pub normal_map: Option<NormalMap>,
    pub normal_strength: f32,
    pub coat_normal_map: Option<NormalMap>,
    pub coat_normal_strength: f32,

    spec_lut: Option<Arc<DielectricGgxDirectionalAlbedoLut>>,
    coat_lut: Option<Arc<DielectricGgxDirectionalAlbedoLut>>,
    conductor_ec_lut: Option<Arc<ConductorGgxEnergyCompensationLut>>,
    dielectric_ec_lut: Option<Arc<DielectricGgxEnergyCompensationLut>>,
}

impl OpenPbrMaterial {
    pub fn new(base_color: Vec3) -> Self {
        Self {
            base_weight: 1.0,
            base_color,
            base_diffuse_roughness: 0.0,
            base_metalness: 0.0,
            specular_weight: 1.0,
            specular_color: Vec3::ONE,
            specular_roughness: 0.3,
            specular_ior: 1.5,
            specular_roughness_anisotropy: 0.0,
            transmission_weight: 0.0,
            transmission_color: Vec3::ONE,
            transmission_depth: 0.0,
            transmission_scatter: Vec3::ZERO,
            transmission_scatter_anisotropy: 0.0,
            transmission_dispersion_scale: 0.0,
            transmission_dispersion_abbe_number: 20.0,
            subsurface_weight: 0.0,
            subsurface_color: Vec3::splat(0.8),
            subsurface_radius: 1.0,
            subsurface_radius_scale: Vec3::new(1.0, 0.5, 0.25),
            subsurface_scatter_anisotropy: 0.0,
            fuzz_weight: 0.0,
            fuzz_color: Vec3::ONE,
            fuzz_roughness: 0.5,
            coat_weight: 0.0,
            coat_color: Vec3::ONE,
            coat_roughness: 0.0,
            coat_roughness_anisotropy: 0.0,
            coat_ior: 1.6,
            coat_darkening: 1.0,
            thin_film_weight: 0.0,
            thin_film_thickness: 0.5,
            thin_film_ior: 1.4,
            thin_film_thickness_texture: None,
            thin_film_thickness_min_nm: 0.0,
            thin_film_thickness_max_nm: 1000.0,
            emission_luminance: 0.0,
            emission_color: Vec3::ONE,
            geometry_opacity: 1.0,
            geometry_thin_walled: false,
            base_color_texture: None,
            specular_roughness_texture: None,
            fuzz_weight_texture: None,
            fuzz_roughness_texture: None,
            base_metalness_texture: None,
            geometry_opacity_texture: None,
            emission_color_texture: None,
            normal_map: None,
            normal_strength: 1.0,
            coat_normal_map: None,
            coat_normal_strength: 1.0,
            spec_lut: None,
            coat_lut: None,
            conductor_ec_lut: None,
            dielectric_ec_lut: None,
        }
    }

    pub fn with_base_weight(mut self, v: f32) -> Self {
        self.base_weight = v;
        self
    }
    pub fn with_base_diffuse_roughness(mut self, v: f32) -> Self {
        self.base_diffuse_roughness = v;
        self
    }
    pub fn with_base_metalness(mut self, v: f32) -> Self {
        self.base_metalness = v;
        self
    }
    pub fn with_specular_weight(mut self, v: f32) -> Self {
        self.specular_weight = v;
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
    pub fn with_specular_roughness_anisotropy(mut self, v: f32) -> Self {
        self.specular_roughness_anisotropy = v;
        self
    }
    pub fn with_transmission_weight(mut self, v: f32) -> Self {
        self.transmission_weight = v;
        self
    }
    pub fn with_transmission_color(mut self, v: Vec3) -> Self {
        self.transmission_color = v;
        self
    }
    pub fn with_transmission_dispersion_scale(mut self, v: f32) -> Self {
        self.transmission_dispersion_scale = v;
        self
    }
    pub fn with_transmission_dispersion_abbe_number(mut self, v: f32) -> Self {
        self.transmission_dispersion_abbe_number = v;
        self
    }
    pub fn with_fuzz(mut self, weight: f32, color: Vec3, roughness: f32) -> Self {
        self.fuzz_weight = weight;
        self.fuzz_color = color;
        self.fuzz_roughness = roughness;
        self
    }
    pub fn with_fuzz_weight_texture(mut self, texture: Arc<ScalarTexture>) -> Self {
        self.fuzz_weight_texture = Some(texture);
        self
    }
    pub fn with_fuzz_roughness_texture(mut self, texture: Arc<ScalarTexture>) -> Self {
        self.fuzz_roughness_texture = Some(texture);
        self
    }
    pub fn with_coat_weight(mut self, v: f32) -> Self {
        self.coat_weight = v;
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
    pub fn with_coat_ior(mut self, v: f32) -> Self {
        self.coat_ior = v;
        self
    }
    pub fn with_coat_darkening(mut self, v: f32) -> Self {
        self.coat_darkening = v;
        self
    }
    pub fn with_thin_film_weight(mut self, v: f32) -> Self {
        self.thin_film_weight = v;
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
    pub fn with_thin_film_thickness_texture(
        mut self,
        texture: Arc<ScalarTexture>,
        min_nm: f32,
        max_nm: f32,
    ) -> Self {
        self.thin_film_thickness_texture = Some(texture);
        self.thin_film_thickness_min_nm = min_nm;
        self.thin_film_thickness_max_nm = max_nm;
        self
    }
    pub fn with_emission_luminance(mut self, v: f32) -> Self {
        self.emission_luminance = v;
        self
    }
    pub fn with_emission_color(mut self, v: Vec3) -> Self {
        self.emission_color = v;
        self
    }
    pub fn with_geometry_opacity(mut self, v: f32) -> Self {
        self.geometry_opacity = v;
        self
    }
    pub fn with_geometry_thin_walled(mut self, v: bool) -> Self {
        self.geometry_thin_walled = v;
        self
    }
    pub fn with_normal_map(mut self, m: NormalMap) -> Self {
        self.normal_map = Some(m);
        self
    }
    pub fn with_coat_normal_map(mut self, m: NormalMap) -> Self {
        self.coat_normal_map = Some(m);
        self
    }

    pub(crate) fn install_spec_lut(&mut self, lut: Arc<DielectricGgxDirectionalAlbedoLut>) {
        self.spec_lut = Some(lut);
    }
    pub(crate) fn install_coat_lut(&mut self, lut: Arc<DielectricGgxDirectionalAlbedoLut>) {
        self.coat_lut = Some(lut);
    }
    pub(crate) fn install_conductor_energy_compensation_lut(
        &mut self,
        lut: Arc<ConductorGgxEnergyCompensationLut>,
    ) {
        self.conductor_ec_lut = Some(lut);
    }
    pub(crate) fn install_dielectric_energy_compensation_lut(
        &mut self,
        lut: Arc<DielectricGgxEnergyCompensationLut>,
    ) {
        self.dielectric_ec_lut = Some(lut);
    }
    pub(crate) fn requires_specular_eta(&self) -> f32 {
        modulated_eta_from_specular_weight(
            self.specular_ior.max(1.0e-4),
            self.specular_weight.max(0.0),
        )
    }
    pub(crate) fn requires_coat_eta(&self) -> f32 {
        self.coat_ior
    }

    pub fn validate_and_warn(&mut self) {
        if self.transmission_depth > 0.0 || self.transmission_scatter.length_squared() > 0.0 {
            tracing::warn!(
                "[OpenPBR] interior medium absorption/scattering requires volume support, which is not supported; falling back to transparent interior."
            );
            self.transmission_depth = 0.0;
            self.transmission_scatter = Vec3::ZERO;
            self.transmission_scatter_anisotropy = 0.0;
            self.transmission_color = Vec3::ONE;
        }
        if self.subsurface_weight > 0.0 {
            tracing::warn!(
                "[OpenPBR] subsurface scattering requires volume/BSSRDF support, which is not supported; subsurface_weight forced to 0."
            );
            self.subsurface_weight = 0.0;
        }
    }

    pub fn opacity_at_uv(&self, shading_vertex: &ShadingVertex) -> f32 {
        let texture_factor = self
            .geometry_opacity_texture
            .as_ref()
            .map(|t| {
                t.sample_filtered(
                    shading_vertex.uv,
                    shading_vertex.uv_dx(),
                    shading_vertex.uv_dy(),
                )
            })
            .unwrap_or(1.0);
        (self.geometry_opacity * texture_factor).clamp(0.0, 1.0)
    }

    pub fn has_alpha_test(&self) -> bool {
        self.geometry_opacity < 1.0 || self.geometry_opacity_texture.is_some()
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
        if self.emission_luminance <= 0.0 {
            return None;
        }
        if !self.geometry_thin_walled && !shading_vertex.front_face {
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
        Some(
            self.emission_color
                * texture_factor
                * self.emission_luminance.max(0.0)
                * self.emission_exit_tint(shading_vertex),
        )
    }

    pub fn may_emit(&self) -> bool {
        self.emission_luminance > 0.0
    }

    pub fn max_emission(&self) -> f32 {
        if self.emission_luminance <= 0.0 {
            return 0.0;
        }
        let texture_factor = self
            .emission_color_texture
            .as_ref()
            .map(|t| t.max_value())
            .unwrap_or(1.0);
        (self.emission_color.max_element() * self.emission_luminance * texture_factor).max(0.0)
    }

    fn make_bsdf(&self, shading_vertex: &ShadingVertex) -> OpenPbrBsdf {
        let base_color = self.base_color_at(shading_vertex);
        let base_metalness = self.base_metalness_at(shading_vertex);
        let fuzz_weight = self.fuzz_weight_at(shading_vertex);
        let fuzz_roughness = self.fuzz_roughness_at(shading_vertex);
        let coat_roughness = self.effective_coat_roughness(fuzz_weight, fuzz_roughness);
        let specular_roughness = self.effective_specular_roughness(
            shading_vertex,
            coat_roughness,
            fuzz_weight,
            fuzz_roughness,
        );
        let (spec_alpha_x, spec_alpha_y) = open_pbr_alpha_xy_from_roughness(
            specular_roughness,
            self.specular_roughness_anisotropy,
        );
        let (coat_alpha_x, coat_alpha_y) =
            open_pbr_alpha_xy_from_roughness(coat_roughness, self.coat_roughness_anisotropy);
        let transmission_abbe = self.transmission_dispersion_abbe();

        let coat_weight = self.coat_weight.clamp(0.0, 1.0);
        let coat_color = self.coat_color.clamp(Vec3::ZERO, Vec3::ONE);
        let darkening = self.coat_darkening_factor(base_color, shading_vertex);
        let specular_color = self.specular_color.clamp(Vec3::ZERO, Vec3::ONE);
        let transmission_color = self.transmission_color.clamp(Vec3::ZERO, Vec3::ONE);
        let specular_eta = modulated_eta_from_specular_weight(
            self.specular_ior.max(1.0e-4),
            self.specular_weight.max(0.0),
        );
        let subsurface_color = self.subsurface_color.clamp(Vec3::ZERO, Vec3::ONE);
        let (metal_n, metal_k) = artist_friendly_complex_ior(
            base_color.clamp(Vec3::ZERO, Vec3::ONE),
            specular_color.clamp(Vec3::ZERO, Vec3::ONE),
        );

        let params = OpenPbrBsdfParams {
            base_color,
            base: self.base_weight.clamp(0.0, 1.0),
            specular: self.specular_weight.max(0.0),
            specular_color,
            specular_alpha_x: spec_alpha_x,
            specular_alpha_y: spec_alpha_y,
            specular_eta,
            transmission_eta: specular_eta,
            metalness: base_metalness,
            metal_n,
            metal_k,
            coat: coat_weight,
            coat_color,
            coat_darkening: darkening,
            coat_alpha_x,
            coat_alpha_y,
            coat_eta: self.coat_ior.max(1.0e-4),
            fuzz: fuzz_weight,
            fuzz_color: self.fuzz_color.clamp(Vec3::ZERO, Vec3::ONE),
            fuzz_roughness,
            transmission: self.transmission_weight.clamp(0.0, 1.0),
            transmission_color,
            transmission_alpha_x: spec_alpha_x,
            transmission_alpha_y: spec_alpha_y,
            transmission_dispersion_abbe: transmission_abbe,
            diffuse_roughness: self.base_diffuse_roughness.clamp(0.0, 1.0),
            subsurface: 0.0,
            subsurface_color,
            thin_walled: self.geometry_thin_walled,
            thin_film_weight: self.thin_film_weight.clamp(0.0, 1.0),
            thin_film_thickness: self.thin_film_thickness_nm_at(shading_vertex),
            thin_film_ior: self.thin_film_ior.max(1.0e-4),
            front_face: shading_vertex.front_face,
            coat_basis_in_base: self.coat_basis_in_base(shading_vertex),
            fuzz_basis_in_base: self.fuzz_basis_in_base(shading_vertex),
            path_throughput: shading_vertex.path_throughput,
            wavelength_lock: shading_vertex.wavelength_lock,
        };

        OpenPbrBsdf::new(
            params,
            self.spec_lut
                .as_ref()
                .expect("OpenPbrMaterial requires spec_lut")
                .clone(),
            self.coat_lut
                .as_ref()
                .expect("OpenPbrMaterial requires coat_lut")
                .clone(),
            self.conductor_ec_lut
                .as_ref()
                .expect("OpenPbrMaterial requires conductor energy compensation LUT")
                .clone(),
            self.dielectric_ec_lut
                .as_ref()
                .expect("OpenPbrMaterial requires dielectric energy compensation LUT")
                .clone(),
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
        let cone_spread = if sample.flags.contains(BsdfFlags::GLOSSY) {
            let fuzz_roughness = self.fuzz_roughness_at(shading_vertex);
            let fuzz_weight = self.fuzz_weight_at(shading_vertex);
            2.0 * self
                .effective_specular_roughness(
                    shading_vertex,
                    self.effective_coat_roughness(fuzz_weight, fuzz_roughness),
                    fuzz_weight,
                    fuzz_roughness,
                )
                .clamp(0.0, 1.0)
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

    pub fn eval(
        &self,
        shading_vertex: &ShadingVertex,
        wi: Vec3,
        _internal_rng: &mut ThreadRng,
    ) -> Vec3 {
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
        _shading_vertex: &ShadingVertex,
    ) -> Option<crate::light_tree::LightTreePrecompute> {
        None
    }

    pub fn light_tree_importance(
        &self,
        _precompute: &crate::light_tree::LightTreePrecompute,
        _w: f32,
        _lobe: &sg::SgLobe,
    ) -> f32 {
        0.0
    }

    fn emission_exit_tint(&self, shading_vertex: &ShadingVertex) -> Vec3 {
        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let cos_o = wo_local.z.abs().clamp(0.0, 1.0);
        if cos_o <= 0.0 {
            return Vec3::ZERO;
        }

        let mut tint = Vec3::ONE;
        let coat_weight = self.coat_weight.clamp(0.0, 1.0);
        if coat_weight > 0.0 {
            let eta_coat = self.coat_ior.max(1.0e-4);
            let mu_t = (1.0 - (1.0 - cos_o * cos_o) / (eta_coat * eta_coat))
                .max(0.0)
                .sqrt()
                .max(1.0e-4);
            let coat_color = self.coat_color.clamp(Vec3::ZERO, Vec3::ONE);
            let coat_transmittance = vec3_powf(coat_color, 1.0 / mu_t);
            tint *= lerp_vec3(Vec3::ONE, coat_transmittance, coat_weight);
        }

        let fuzz_weight = self.fuzz_weight_at(shading_vertex);
        if fuzz_weight > 0.0 {
            let e_fuzz = zeltner_dir_albedo(cos_o, self.fuzz_roughness_at(shading_vertex));
            tint *= Vec3::splat(lerp(1.0, (1.0 - e_fuzz).max(0.0), fuzz_weight));
        }

        tint
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

    fn fuzz_basis_in_base(&self, shading_vertex: &ShadingVertex) -> Option<OrthonormalBasis> {
        if self.fuzz_weight <= 0.0 || self.coat_weight <= 0.0 {
            return None;
        }
        let coat_basis = self.coat_basis_in_base(shading_vertex)?;
        let fuzz_normal = lerp_vec3(
            Vec3::Z,
            coat_basis.normal(),
            self.coat_weight.clamp(0.0, 1.0),
        )
        .normalize_or_zero();
        if fuzz_normal.length_squared() == 0.0 || fuzz_normal.z <= 0.0 {
            return None;
        }
        Some(OrthonormalBasis::from_normal(fuzz_normal))
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

    fn base_metalness_at(&self, shading_vertex: &ShadingVertex) -> f32 {
        (self.base_metalness
            * self
                .base_metalness_texture
                .as_ref()
                .map(|t| {
                    t.sample_filtered(
                        shading_vertex.uv,
                        shading_vertex.uv_dx(),
                        shading_vertex.uv_dy(),
                    )
                })
                .unwrap_or(1.0))
        .clamp(0.0, 1.0)
    }

    fn specular_roughness_at(&self, shading_vertex: &ShadingVertex) -> f32 {
        (self.specular_roughness
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
                .unwrap_or(1.0))
        .clamp(0.0, 1.0)
    }

    fn fuzz_weight_at(&self, shading_vertex: &ShadingVertex) -> f32 {
        (self.fuzz_weight
            * self
                .fuzz_weight_texture
                .as_ref()
                .map(|t| {
                    t.sample_filtered(
                        shading_vertex.uv,
                        shading_vertex.uv_dx(),
                        shading_vertex.uv_dy(),
                    )
                })
                .unwrap_or(1.0))
        .clamp(0.0, 1.0)
    }

    fn fuzz_roughness_at(&self, shading_vertex: &ShadingVertex) -> f32 {
        (self.fuzz_roughness
            * self
                .fuzz_roughness_texture
                .as_ref()
                .map(|t| {
                    t.sample_filtered(
                        shading_vertex.uv,
                        shading_vertex.uv_dx(),
                        shading_vertex.uv_dy(),
                    )
                })
                .unwrap_or(1.0))
        .clamp(0.0, 1.0)
    }

    fn effective_specular_roughness(
        &self,
        shading_vertex: &ShadingVertex,
        coat_roughness: f32,
        fuzz_weight: f32,
        fuzz_roughness: f32,
    ) -> f32 {
        let r_base = self.specular_roughness_at(shading_vertex);
        let coat_roughened = (r_base.powi(4) + 2.0 * coat_roughness.powi(4))
            .min(1.0)
            .powf(0.25);
        let with_coat = lerp(r_base, coat_roughened, self.coat_weight.clamp(0.0, 1.0));
        self.apply_fuzz_roughening(with_coat, fuzz_weight, fuzz_roughness)
    }

    fn effective_coat_roughness(&self, fuzz_weight: f32, fuzz_roughness: f32) -> f32 {
        self.apply_fuzz_roughening(
            self.coat_roughness.clamp(0.0, 1.0),
            fuzz_weight,
            fuzz_roughness,
        )
    }

    fn apply_fuzz_roughening(&self, roughness: f32, fuzz_weight: f32, fuzz_roughness: f32) -> f32 {
        if fuzz_weight <= 0.0 {
            return roughness;
        }
        let fuzz_color = self.fuzz_color.clamp(Vec3::ZERO, Vec3::ONE);
        let e_fuzz = zeltner_dir_albedo(1.0, fuzz_roughness);
        let r_f = sg::luminance(fuzz_color * e_fuzz).clamp(0.0, 1.0);
        let roughened = (roughness.powi(4) + 2.0 * r_f.powi(4)).min(1.0).powf(0.25);
        lerp(roughness, roughened, fuzz_weight)
    }

    fn transmission_dispersion_abbe(&self) -> f32 {
        let scale = self.transmission_dispersion_scale.max(0.0);
        if scale <= 0.0 {
            0.0
        } else {
            self.transmission_dispersion_abbe_number.max(1.0e-6) / scale
        }
    }

    fn thin_film_thickness_nm_at(&self, shading_vertex: &ShadingVertex) -> f32 {
        if self.thin_film_weight <= 0.0 {
            return 0.0;
        }
        self.thin_film_thickness_texture
            .as_ref()
            .map(|t| {
                let v = t
                    .sample_filtered(
                        shading_vertex.uv,
                        shading_vertex.uv_dx(),
                        shading_vertex.uv_dy(),
                    )
                    .clamp(0.0, 1.0);
                lerp(
                    self.thin_film_thickness_min_nm.max(0.0),
                    self.thin_film_thickness_max_nm.max(0.0),
                    v,
                )
            })
            .unwrap_or_else(|| self.thin_film_thickness.max(0.0) * 1000.0)
    }

    fn coat_darkening_factor(&self, base_color: Vec3, shading_vertex: &ShadingVertex) -> Vec3 {
        let coat_weight = self.coat_weight.clamp(0.0, 1.0);
        let delta = self.coat_darkening.clamp(0.0, 1.0);
        if coat_weight <= 0.0 || delta <= 0.0 {
            return Vec3::ONE;
        }
        let eta_coat = self.coat_ior.max(1.0);
        let ef = dielectric_fresnel_avg(eta_coat);
        let kr = 1.0 - (1.0 - ef) / (eta_coat * eta_coat);
        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let ks = fresnel_dielectric(wo_local.z.abs().clamp(0.0, 1.0), 1.0, eta_coat);
        let eta_s = lerp(
            self.specular_ior.max(1.0e-4),
            self.specular_ior.max(1.0e-4) / eta_coat,
            coat_weight,
        );
        let f0 = ((eta_s - 1.0) / (eta_s + 1.0)).powi(2);
        let fs = (self.specular_weight.max(0.0) * f0).clamp(0.0, 1.0);
        let r_spec = self.specular_roughness_at(shading_vertex);
        let rd = lerp(1.0, r_spec, fs);
        let rb = lerp(rd, r_spec, self.base_metalness_at(shading_vertex));
        let k = lerp(ks, kr, rb).clamp(0.0, 1.0);
        let base_weighted = base_color * self.base_weight.clamp(0.0, 1.0);
        let subsurface = self.subsurface_color.clamp(Vec3::ZERO, Vec3::ONE);
        let e_dielec = lerp_vec3(
            lerp_vec3(
                base_weighted,
                subsurface,
                self.subsurface_weight.clamp(0.0, 1.0),
            ),
            Vec3::splat(1.0 - f0),
            self.transmission_weight.clamp(0.0, 1.0),
        );
        let e_metal = (base_weighted * self.specular_weight.max(0.0)).clamp(Vec3::ZERO, Vec3::ONE);
        let e_base = lerp_vec3(e_dielec, e_metal, self.base_metalness_at(shading_vertex))
            .clamp(Vec3::ZERO, Vec3::ONE);
        let coat_color = self.coat_color.clamp(Vec3::ZERO, Vec3::ONE);
        let denom = (Vec3::ONE - e_base * (k * coat_color)).max(Vec3::splat(1.0e-4));
        let delta_factor = Vec3::splat((1.0 - k).max(0.0)) / denom;
        lerp_vec3(Vec3::ONE, delta_factor, coat_weight * delta).clamp(Vec3::ZERO, Vec3::ONE)
    }
}

fn open_pbr_alpha_xy_from_roughness(roughness: f32, anisotropy: f32) -> (f32, f32) {
    let r = roughness.clamp(0.0, 1.0);
    let a = anisotropy.clamp(0.0, 1.0);
    let alpha = r * r;
    let one_minus_a = 1.0 - a;
    let alpha_x = alpha * (2.0 / (1.0 + one_minus_a * one_minus_a)).sqrt();
    let alpha_y = one_minus_a * alpha_x;
    (alpha_x.clamp(MIN_ALPHA, 1.0), alpha_y.clamp(MIN_ALPHA, 1.0))
}

fn modulated_eta_from_specular_weight(eta: f32, specular_weight: f32) -> f32 {
    if (specular_weight - 1.0).abs() < 1.0e-6 {
        return eta;
    }
    let f0 = ((eta - 1.0) / (eta + 1.0)).powi(2);
    let eps = (specular_weight * f0).clamp(0.0, 0.999_99).sqrt() * (eta - 1.0).signum();
    ((1.0 + eps) / (1.0 - eps).max(1.0e-6)).max(1.0e-4)
}

fn dielectric_fresnel_avg(eta: f32) -> f32 {
    let eta = eta.clamp(1.0, 3.0);
    ((10893.0 * eta - 1438.2) / (-774.4 * eta * eta + 10212.0 * eta + 1.0))
        .max(1.0e-6)
        .ln()
        .clamp(0.0, 1.0)
}

fn zeltner_dir_albedo(ndot_v: f32, roughness: f32) -> f32 {
    let x = ndot_v.clamp(0.0, 1.0);
    let y = roughness.clamp(0.01, 1.0);
    let s = y * (0.020_660_7 + 1.584_91 * y) / (0.037_942_4 + y * (1.322_27 + y));
    let m = y * (-0.193_854 + y * (-1.148_85 + y * (1.793_2 - 0.959_43 * y * y))) / (0.046_391 + y);
    let o =
        y * (0.000_654_023 + (-0.020_781_8 + 0.119_681 * y) * y) / (1.262_64 + y * (-1.920_21 + y));
    (-0.5 * ((x - m) / s).powi(2)).exp() / (s * (2.0 * std::f32::consts::PI).sqrt()) + o
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn lerp_vec3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a + (b - a) * t
}

fn vec3_powf(v: Vec3, e: f32) -> Vec3 {
    Vec3::new(v.x.powf(e), v.y.powf(e), v.z.powf(e))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use glam::{Vec2, Vec3};

    use crate::{
        bsdf::{
            ConductorGgxEnergyCompensationLut, DielectricGgxDirectionalAlbedoLut,
            DielectricGgxEnergyCompensationLut,
        },
        material::ShadingVertex,
        math::OrthonormalBasis,
        scene::{InstanceIndex, TriangleRef},
    };

    use super::{OpenPbrMaterial, dielectric_fresnel_avg, open_pbr_alpha_xy_from_roughness};

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

    fn material_with_luts() -> OpenPbrMaterial {
        let mut m = OpenPbrMaterial::new(Vec3::splat(0.8));
        m.install_spec_lut(Arc::new(
            DielectricGgxDirectionalAlbedoLut::constant_for_tests(1.5, 0.04),
        ));
        m.install_coat_lut(Arc::new(
            DielectricGgxDirectionalAlbedoLut::constant_for_tests(1.6, 0.04),
        ));
        m.install_conductor_energy_compensation_lut(Arc::new(
            ConductorGgxEnergyCompensationLut::constant_for_tests(0.9, 0.9),
        ));
        m.install_dielectric_energy_compensation_lut(Arc::new(
            DielectricGgxEnergyCompensationLut::constant_for_tests(0.9, 0.9),
        ));
        m
    }

    #[test]
    fn anisotropy_mapping_preserves_average_roughness() {
        let (ax, ay) = open_pbr_alpha_xy_from_roughness(0.5, 0.8);
        let isotropic_alpha = 0.25_f32;
        assert!(((ax * ax + ay * ay) - 2.0 * isotropic_alpha * isotropic_alpha).abs() < 1.0e-5);
    }

    #[test]
    fn dispersion_scale_maps_to_abbe_number() {
        let mut m = material_with_luts();
        m.transmission_dispersion_abbe_number = 20.0;
        m.transmission_dispersion_scale = 0.25;
        assert!((m.transmission_dispersion_abbe() - 80.0).abs() < 1.0e-6);
        m.transmission_dispersion_scale = 0.0;
        assert_eq!(m.transmission_dispersion_abbe(), 0.0);
    }

    #[test]
    fn validation_falls_back_unsupported_volume_features() {
        let mut m = material_with_luts();
        m.transmission_depth = 1.0;
        m.transmission_color = Vec3::new(0.3, 0.4, 0.5);
        m.transmission_scatter = Vec3::splat(0.2);
        m.subsurface_weight = 1.0;
        m.validate_and_warn();
        assert_eq!(m.transmission_depth, 0.0);
        assert_eq!(m.transmission_scatter, Vec3::ZERO);
        assert_eq!(m.transmission_color, Vec3::ONE);
        assert_eq!(m.subsurface_weight, 0.0);
    }

    #[test]
    fn evaluates_finite_for_default_setup() {
        let m = material_with_luts();
        let v = test_shading_vertex(Vec3::Z);
        let mut rng = rand::rng();
        let f = m.eval(&v, Vec3::new(0.2, 0.3, 0.9327379).normalize(), &mut rng);
        assert!(f.is_finite());
    }

    #[test]
    fn coat_darkening_reduces_bright_base() {
        let mut m = material_with_luts();
        m.coat_weight = 1.0;
        m.coat_darkening = 1.0;
        m.base_color = Vec3::ONE;
        let v = test_shading_vertex(Vec3::Z);
        let factor = m.coat_darkening_factor(Vec3::ONE, &v);
        assert!(factor.max_element() <= 1.0);
        assert!(factor.min_element() > 0.0);
        assert!(dielectric_fresnel_avg(1.5) > 0.0);
    }

    #[test]
    fn emission_is_under_coat_and_fuzz() {
        let mut m = material_with_luts();
        m.emission_luminance = 10.0;
        m.emission_color = Vec3::ONE;
        let v = test_shading_vertex(Vec3::Z);
        let plain = m.le(&v).unwrap();

        m.coat_weight = 1.0;
        m.coat_color = Vec3::splat(0.25);
        let coated = m.le(&v).unwrap();
        assert!(coated.max_element() < plain.max_element());

        m.coat_weight = 0.0;
        m.fuzz_weight = 1.0;
        m.fuzz_roughness = 0.5;
        let fuzzed = m.le(&v).unwrap();
        assert!(fuzzed.max_element() < plain.max_element());
    }

    #[test]
    fn thin_film_weight_does_not_scale_thickness() {
        let mut m = material_with_luts();
        m.thin_film_weight = 0.25;
        m.thin_film_thickness = 0.5;
        let v = test_shading_vertex(Vec3::Z);
        assert_eq!(m.thin_film_thickness_nm_at(&v), 500.0);
    }
}
