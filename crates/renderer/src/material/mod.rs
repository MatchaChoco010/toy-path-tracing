mod conductor_ggx;
mod conductor_ggx_cui_2023;
mod dielectric_ggx;
mod disney_brdf;
mod emissive;
mod eon;
mod glass;
mod mirror;
pub mod mtlx;
mod mtlx_material;
mod normal_map;
mod normalized_lambert;
mod open_pbr;
mod oren_nayar;
pub mod pattern;
mod simple_pbr;
mod standard_surface;
pub mod texture;

use glam::{Vec2, Vec3};

use crate::{
    bsdf::{BsdfFlags, TransportMode},
    color::{self, OcioColorProcessor},
    light_tree::LightTreePrecompute,
    math::{OrthonormalBasis, sg::SgLobe},
    sampler::{AuxRng, MaterialSampleRandoms},
    scene::TriangleRef,
};

pub(super) const GEOMETRIC_NORMAL_COS_EPSILON: f32 = 1.0e-6;

pub(super) fn modified_bsdf_eval(
    shading_vertex: &ShadingVertex,
    wi: Vec3,
    f: Vec3,
    mode: TransportMode,
) -> Vec3 {
    if f.length_squared() == 0.0 {
        return Vec3::ZERO;
    }

    let correction_direction = match mode {
        TransportMode::Radiance => wi,
        TransportMode::Importance => shading_vertex.wo,
    };
    let cos_geom = correction_direction.dot(shading_vertex.ng).abs();
    if cos_geom <= GEOMETRIC_NORMAL_COS_EPSILON {
        return Vec3::ZERO;
    }

    f * (correction_direction.dot(shading_vertex.ns).abs() / cos_geom)
}

pub(super) fn modified_bsdf_sample_weight(
    shading_vertex: &ShadingVertex,
    wi: Vec3,
    weight: Vec3,
    flags: BsdfFlags,
    mode: TransportMode,
) -> Vec3 {
    if mode == TransportMode::Radiance
        || flags.contains(BsdfFlags::DELTA)
        || weight.length_squared() == 0.0
    {
        return weight;
    }

    let cos_wo_geom = shading_vertex.wo.dot(shading_vertex.ng).abs();
    let cos_wo_shading = shading_vertex.wo.dot(shading_vertex.ns).abs();
    let cos_wi_geom = wi.dot(shading_vertex.ng).abs();
    let cos_wi_shading = wi.dot(shading_vertex.ns).abs();
    if cos_wo_geom <= GEOMETRIC_NORMAL_COS_EPSILON || cos_wi_shading <= GEOMETRIC_NORMAL_COS_EPSILON
    {
        return Vec3::ZERO;
    }

    weight * (cos_wo_shading / cos_wo_geom) * (cos_wi_geom / cos_wi_shading)
}

pub use conductor_ggx::ConductorGgxMaterial;
pub use conductor_ggx_cui_2023::ConductorGgxCui2023Material;
pub use dielectric_ggx::DielectricGgxMaterial;
pub use disney_brdf::DisneyBrdfMaterial;
pub use emissive::EmissiveMaterial;
pub use eon::EonMaterial;
pub use glass::GlassMaterial;
pub use mirror::MirrorMaterial;
pub use mtlx::MtlxScratch;
pub use mtlx_material::MtlxMaterial;
pub use normal_map::NormalMap;
pub use normalized_lambert::NormalizedLambertMaterial;
pub use open_pbr::OpenPbrMaterial;
pub use oren_nayar::OrenNayarMaterial;
pub use simple_pbr::SimplePbrMaterial;
pub use standard_surface::StandardSurfaceMaterial;
pub use texture::{ScalarTexture, Texture, TextureColorSpace};

#[derive(Debug, Clone, PartialEq)]
pub enum Material {
    NormalizedLambert(NormalizedLambertMaterial),
    OrenNayar(OrenNayarMaterial),
    Eon(EonMaterial),
    Mirror(MirrorMaterial),
    ConductorGgx(ConductorGgxMaterial),
    ConductorGgxCui2023(ConductorGgxCui2023Material),
    DielectricGgx(DielectricGgxMaterial),
    Glass(GlassMaterial),
    SimplePBR(SimplePbrMaterial),
    DisneyBrdf(DisneyBrdfMaterial),
    StandardSurface(StandardSurfaceMaterial),
    OpenPbr(OpenPbrMaterial),
    Emissive(EmissiveMaterial),
    Mtlx(MtlxMaterial),
}

#[derive(Debug, Clone)]
pub struct ShadingVertex {
    pub triangle: TriangleRef,
    pub p: Vec3,
    pub uv: Vec2,
    pub dudx: f32,
    pub dvdx: f32,
    pub dudy: f32,
    pub dvdy: f32,
    pub ng: Vec3,
    pub ns: Vec3,
    pub wo: Vec3,
    pub dpdu: Vec3,
    pub dpdv: Vec3,
    pub dpdx: Vec3,
    pub dpdy: Vec3,
    pub dndu: Vec3,
    pub dndv: Vec3,
    pub frame: OrthonormalBasis,
    pub front_face: bool,
    pub path_throughput: Vec3,
    pub wavelength_lock: Option<f32>,
    pub object_to_world: glam::Mat4,
    pub world_to_object: glam::Mat4,
    pub object_normal_to_world: glam::Mat3,
    /// MaterialX bytecode の register file は `MtlxScratch` の `regs_pool` に
    /// 確保される。 ShadingVertex はそこへの handle のみ保持する (Vec を毎回
    /// alloc しない設計)。 `None` の間は precompute 未実行。
    pub mtlx_regs: Option<mtlx::RegsHandle>,
    pub mtlx_dalbedo: Option<mtlx::DalbedoHandle>,
    pub mtlx_precomputed_for: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaterialSample {
    pub weight: Vec3,
    pub wi: Vec3,
    pub pdf: f32,
    pub pdf_rev: f32,
    pub flags: BsdfFlags,
    pub eta: f32,
    pub cone_spread: f32,
    pub wavelength_lock: Option<f32>,
}

impl ShadingVertex {
    pub fn uv_dx(&self) -> Vec2 {
        Vec2::new(self.dudx, self.dvdx)
    }

    pub fn uv_dy(&self) -> Vec2 {
        Vec2::new(self.dudy, self.dvdy)
    }
}

impl Material {
    pub fn convert_constant_colors_to_rendering(
        &mut self,
        processor: &OcioColorProcessor,
    ) -> color::Result<()> {
        let convert = |rgb| processor.apply_rgb(rgb);
        match self {
            Self::NormalizedLambert(material) => material.rho = convert(material.rho)?,
            Self::OrenNayar(material) => material.rho = convert(material.rho)?,
            Self::Eon(material) => material.rho = convert(material.rho)?,
            Self::Mirror(material) => material.color = convert(material.color)?,
            Self::ConductorGgx(material) => material.base_color = convert(material.base_color)?,
            Self::ConductorGgxCui2023(material) => {
                material.base_color = convert(material.base_color)?;
            }
            Self::DielectricGgx(material) => material.color = convert(material.color)?,
            Self::Glass(material) => material.color = convert(material.color)?,
            Self::SimplePBR(material) => material.base_color = convert(material.base_color)?,
            Self::DisneyBrdf(material) => material.base_color = convert(material.base_color)?,
            Self::StandardSurface(material) => {
                material.base_color = convert(material.base_color)?;
                material.specular_color = convert(material.specular_color)?;
                material.transmission_color = convert(material.transmission_color)?;
                material.transmission_scatter = convert(material.transmission_scatter)?;
                material.subsurface_color = convert(material.subsurface_color)?;
                material.coat_color = convert(material.coat_color)?;
                material.sheen_color = convert(material.sheen_color)?;
                material.emission_color = convert(material.emission_color)?;
            }
            Self::OpenPbr(material) => {
                material.base_color = convert(material.base_color)?;
                material.specular_color = convert(material.specular_color)?;
                material.transmission_color = convert(material.transmission_color)?;
                material.transmission_scatter = convert(material.transmission_scatter)?;
                material.subsurface_color = convert(material.subsurface_color)?;
                material.fuzz_color = convert(material.fuzz_color)?;
                material.coat_color = convert(material.coat_color)?;
                material.emission_color = convert(material.emission_color)?;
            }
            Self::Emissive(material) => material.color = convert(material.color)?,
            Self::Mtlx(_) => {}
        }
        Ok(())
    }

    /// Materials such as `Mtlx` evaluate per-vertex bytecode and store the
    /// resulting locals on the vertex itself. The integrator calls this once
    /// per intersection before `sample` / `eval` / `pdf` / `le` so the BSDF
    /// queries can read those locals directly. For non-Mtlx materials this is
    /// a no-op.
    pub fn precompute_shading(
        &self,
        shading_vertex: &mut ShadingVertex,
        scratch: &mut MtlxScratch,
    ) {
        if let Self::Mtlx(material) = self {
            material.precompute_shading(shading_vertex, scratch);
        }
    }

    pub(crate) fn prepare_shading_vertex(&self, shading_vertex: &mut ShadingVertex) {
        match self {
            Self::NormalizedLambert(material) => material.prepare_shading_vertex(shading_vertex),
            Self::OrenNayar(material) => material.prepare_shading_vertex(shading_vertex),
            Self::Eon(material) => material.prepare_shading_vertex(shading_vertex),
            Self::Mirror(material) => material.prepare_shading_vertex(shading_vertex),
            Self::ConductorGgx(material) => material.prepare_shading_vertex(shading_vertex),
            Self::ConductorGgxCui2023(material) => material.prepare_shading_vertex(shading_vertex),
            Self::DielectricGgx(material) => material.prepare_shading_vertex(shading_vertex),
            Self::Glass(material) => material.prepare_shading_vertex(shading_vertex),
            Self::SimplePBR(material) => material.prepare_shading_vertex(shading_vertex),
            Self::DisneyBrdf(material) => material.prepare_shading_vertex(shading_vertex),
            Self::StandardSurface(material) => material.prepare_shading_vertex(shading_vertex),
            Self::OpenPbr(material) => material.prepare_shading_vertex(shading_vertex),
            Self::Emissive(_) => {}
            Self::Mtlx(material) => material.prepare_shading_vertex(shading_vertex),
        }
    }

    pub fn sample(
        &self,
        shading_vertex: &ShadingVertex,
        scratch: &MtlxScratch,
        randoms: &MaterialSampleRandoms,
        aux_rng: &mut AuxRng,
        mode: TransportMode,
    ) -> Option<MaterialSample> {
        match self {
            Self::NormalizedLambert(material) => {
                material.sample(shading_vertex, randoms, aux_rng, mode)
            }
            Self::OrenNayar(material) => material.sample(shading_vertex, randoms, aux_rng, mode),
            Self::Eon(material) => material.sample(shading_vertex, randoms, aux_rng, mode),
            Self::Mirror(material) => material.sample(shading_vertex, randoms, aux_rng, mode),
            Self::ConductorGgx(material) => material.sample(shading_vertex, randoms, aux_rng, mode),
            Self::ConductorGgxCui2023(material) => {
                material.sample(shading_vertex, randoms, aux_rng, mode)
            }
            Self::DielectricGgx(material) => {
                material.sample(shading_vertex, randoms, aux_rng, mode)
            }
            Self::Glass(material) => material.sample(shading_vertex, randoms, aux_rng, mode),
            Self::SimplePBR(material) => material.sample(shading_vertex, randoms, aux_rng, mode),
            Self::DisneyBrdf(material) => material.sample(shading_vertex, randoms, aux_rng, mode),
            Self::StandardSurface(material) => {
                material.sample(shading_vertex, randoms, aux_rng, mode)
            }
            Self::OpenPbr(material) => material.sample(shading_vertex, randoms, aux_rng, mode),
            Self::Emissive(material) => material.sample(shading_vertex, randoms, aux_rng, mode),
            Self::Mtlx(material) => {
                material.sample(shading_vertex, scratch, randoms, aux_rng, mode)
            }
        }
    }

    pub fn le(&self, shading_vertex: &ShadingVertex, scratch: &MtlxScratch) -> Option<Vec3> {
        match self {
            Self::NormalizedLambert(material) => material.le(shading_vertex),
            Self::OrenNayar(material) => material.le(shading_vertex),
            Self::Eon(material) => material.le(shading_vertex),
            Self::Mirror(material) => material.le(shading_vertex),
            Self::ConductorGgx(material) => material.le(shading_vertex),
            Self::ConductorGgxCui2023(material) => material.le(shading_vertex),
            Self::DielectricGgx(material) => material.le(shading_vertex),
            Self::Glass(material) => material.le(shading_vertex),
            Self::SimplePBR(material) => material.le(shading_vertex),
            Self::DisneyBrdf(material) => material.le(shading_vertex),
            Self::StandardSurface(material) => material.le(shading_vertex),
            Self::OpenPbr(material) => material.le(shading_vertex),
            Self::Emissive(material) => material.le(shading_vertex),
            Self::Mtlx(material) => material.le(shading_vertex, scratch),
        }
    }

    pub fn eval(
        &self,
        shading_vertex: &ShadingVertex,
        scratch: &MtlxScratch,
        wi: Vec3,
        aux_rng: &mut AuxRng,
        mode: TransportMode,
    ) -> Vec3 {
        match self {
            Self::NormalizedLambert(material) => material.eval(shading_vertex, wi, aux_rng, mode),
            Self::OrenNayar(material) => material.eval(shading_vertex, wi, aux_rng, mode),
            Self::Eon(material) => material.eval(shading_vertex, wi, aux_rng, mode),
            Self::Mirror(material) => material.eval(shading_vertex, wi, aux_rng, mode),
            Self::ConductorGgx(material) => material.eval(shading_vertex, wi, aux_rng, mode),
            Self::ConductorGgxCui2023(material) => material.eval(shading_vertex, wi, aux_rng, mode),
            Self::DielectricGgx(material) => material.eval(shading_vertex, wi, aux_rng, mode),
            Self::Glass(material) => material.eval(shading_vertex, wi, aux_rng, mode),
            Self::SimplePBR(material) => material.eval(shading_vertex, wi, aux_rng, mode),
            Self::DisneyBrdf(material) => material.eval(shading_vertex, wi, aux_rng, mode),
            Self::StandardSurface(material) => material.eval(shading_vertex, wi, aux_rng, mode),
            Self::OpenPbr(material) => material.eval(shading_vertex, wi, aux_rng, mode),
            Self::Emissive(material) => material.eval(shading_vertex, wi, aux_rng, mode),
            Self::Mtlx(material) => material.eval(shading_vertex, scratch, wi, aux_rng, mode),
        }
    }

    pub fn pdf(&self, shading_vertex: &ShadingVertex, scratch: &MtlxScratch, wi: Vec3) -> f32 {
        match self {
            Self::NormalizedLambert(material) => material.pdf(shading_vertex, wi),
            Self::OrenNayar(material) => material.pdf(shading_vertex, wi),
            Self::Eon(material) => material.pdf(shading_vertex, wi),
            Self::Mirror(material) => material.pdf(shading_vertex, wi),
            Self::ConductorGgx(material) => material.pdf(shading_vertex, wi),
            Self::ConductorGgxCui2023(material) => material.pdf(shading_vertex, wi),
            Self::DielectricGgx(material) => material.pdf(shading_vertex, wi),
            Self::Glass(material) => material.pdf(shading_vertex, wi),
            Self::SimplePBR(material) => material.pdf(shading_vertex, wi),
            Self::DisneyBrdf(material) => material.pdf(shading_vertex, wi),
            Self::StandardSurface(material) => material.pdf(shading_vertex, wi),
            Self::OpenPbr(material) => material.pdf(shading_vertex, wi),
            Self::Emissive(material) => material.pdf(shading_vertex, wi),
            Self::Mtlx(material) => material.pdf(shading_vertex, scratch, wi),
        }
    }

    pub fn eval_pdf(
        &self,
        shading_vertex: &ShadingVertex,
        scratch: &MtlxScratch,
        wi: Vec3,
        aux_rng: &mut AuxRng,
        mode: TransportMode,
    ) -> (Vec3, f32) {
        match self {
            Self::Mtlx(material) => material.eval_pdf(shading_vertex, scratch, wi, mode),
            _ => {
                let f = self.eval(shading_vertex, scratch, wi, aux_rng, mode);
                if f.length_squared() == 0.0 {
                    (f, 0.0)
                } else {
                    (f, self.pdf(shading_vertex, scratch, wi))
                }
            }
        }
    }

    pub fn may_emit(&self) -> bool {
        match self {
            Self::NormalizedLambert(material) => material.may_emit(),
            Self::OrenNayar(material) => material.may_emit(),
            Self::Eon(material) => material.may_emit(),
            Self::Mirror(material) => material.may_emit(),
            Self::ConductorGgx(material) => material.may_emit(),
            Self::ConductorGgxCui2023(material) => material.may_emit(),
            Self::DielectricGgx(material) => material.may_emit(),
            Self::Glass(material) => material.may_emit(),
            Self::SimplePBR(material) => material.may_emit(),
            Self::DisneyBrdf(material) => material.may_emit(),
            Self::StandardSurface(material) => material.may_emit(),
            Self::OpenPbr(material) => material.may_emit(),
            Self::Emissive(material) => material.may_emit(),
            Self::Mtlx(material) => material.may_emit(),
        }
    }

    pub fn is_pure_emitter(&self) -> bool {
        matches!(self, Self::Emissive(_))
    }

    pub fn max_emission(&self) -> f32 {
        match self {
            Self::NormalizedLambert(material) => material.max_emission(),
            Self::OrenNayar(material) => material.max_emission(),
            Self::Eon(material) => material.max_emission(),
            Self::Mirror(material) => material.max_emission(),
            Self::ConductorGgx(material) => material.max_emission(),
            Self::ConductorGgxCui2023(material) => material.max_emission(),
            Self::DielectricGgx(material) => material.max_emission(),
            Self::Glass(material) => material.max_emission(),
            Self::SimplePBR(material) => material.max_emission(),
            Self::DisneyBrdf(material) => material.max_emission(),
            Self::StandardSurface(material) => material.max_emission(),
            Self::OpenPbr(material) => material.max_emission(),
            Self::Emissive(material) => material.max_emission(),
            Self::Mtlx(material) => material.max_emission(),
        }
    }

    pub fn has_alpha_test(&self) -> bool {
        match self {
            Self::NormalizedLambert(material) => material.has_alpha_test(),
            Self::OrenNayar(material) => material.has_alpha_test(),
            Self::Eon(material) => material.has_alpha_test(),
            Self::Mirror(material) => material.has_alpha_test(),
            Self::ConductorGgx(material) => material.has_alpha_test(),
            Self::ConductorGgxCui2023(material) => material.has_alpha_test(),
            Self::DielectricGgx(material) => material.has_alpha_test(),
            Self::Glass(material) => material.has_alpha_test(),
            Self::SimplePBR(material) => material.has_alpha_test(),
            Self::DisneyBrdf(material) => material.has_alpha_test(),
            Self::StandardSurface(material) => material.has_alpha_test(),
            Self::OpenPbr(material) => material.has_alpha_test(),
            Self::Emissive(material) => material.has_alpha_test(),
            Self::Mtlx(material) => material.has_alpha_test(),
        }
    }

    pub fn any_hit(
        &self,
        shading_vertex: &mut ShadingVertex,
        scratch: &mut MtlxScratch,
        u: f32,
    ) -> bool {
        match self {
            Self::NormalizedLambert(material) => material.any_hit(shading_vertex, u),
            Self::OrenNayar(material) => material.any_hit(shading_vertex, u),
            Self::Eon(material) => material.any_hit(shading_vertex, u),
            Self::Mirror(material) => material.any_hit(shading_vertex, u),
            Self::ConductorGgx(material) => material.any_hit(shading_vertex, u),
            Self::ConductorGgxCui2023(material) => material.any_hit(shading_vertex, u),
            Self::DielectricGgx(material) => material.any_hit(shading_vertex, u),
            Self::Glass(material) => material.any_hit(shading_vertex, u),
            Self::SimplePBR(material) => material.any_hit(shading_vertex, u),
            Self::DisneyBrdf(material) => material.any_hit(shading_vertex, u),
            Self::StandardSurface(material) => material.any_hit(shading_vertex, u),
            Self::OpenPbr(material) => material.any_hit(shading_vertex, u),
            Self::Emissive(material) => material.any_hit(shading_vertex, u),
            Self::Mtlx(material) => material.any_hit(shading_vertex, scratch, u),
        }
    }

    pub fn light_tree_precompute(
        &self,
        shading_vertex: &ShadingVertex,
        scratch: &MtlxScratch,
    ) -> Option<LightTreePrecompute> {
        match self {
            Self::NormalizedLambert(material) => material.light_tree_precompute(shading_vertex),
            Self::OrenNayar(material) => material.light_tree_precompute(shading_vertex),
            Self::Eon(material) => material.light_tree_precompute(shading_vertex),
            Self::ConductorGgx(material) => material.light_tree_precompute(shading_vertex),
            Self::ConductorGgxCui2023(material) => material.light_tree_precompute(shading_vertex),
            Self::DielectricGgx(material) => material.light_tree_precompute(shading_vertex),
            Self::SimplePBR(material) => material.light_tree_precompute(shading_vertex),
            Self::DisneyBrdf(material) => material.light_tree_precompute(shading_vertex),
            Self::StandardSurface(material) => material.light_tree_precompute(shading_vertex),
            Self::OpenPbr(material) => material.light_tree_precompute(shading_vertex),
            Self::Mtlx(material) => material.light_tree_precompute(shading_vertex, scratch),
            Self::Mirror(_) | Self::Glass(_) | Self::Emissive(_) => None,
        }
    }

    pub fn light_tree_importance(
        &self,
        precompute: &LightTreePrecompute,
        w: f32,
        lobe: &SgLobe,
    ) -> f32 {
        match self {
            Self::NormalizedLambert(material) => {
                material.light_tree_importance(precompute, w, lobe)
            }
            Self::OrenNayar(material) => material.light_tree_importance(precompute, w, lobe),
            Self::Eon(material) => material.light_tree_importance(precompute, w, lobe),
            Self::ConductorGgx(material) => material.light_tree_importance(precompute, w, lobe),
            Self::ConductorGgxCui2023(material) => {
                material.light_tree_importance(precompute, w, lobe)
            }
            Self::DielectricGgx(material) => material.light_tree_importance(precompute, w, lobe),
            Self::SimplePBR(material) => material.light_tree_importance(precompute, w, lobe),
            Self::DisneyBrdf(material) => material.light_tree_importance(precompute, w, lobe),
            Self::StandardSurface(material) => material.light_tree_importance(precompute, w, lobe),
            Self::OpenPbr(material) => material.light_tree_importance(precompute, w, lobe),
            Self::Mtlx(material) => material.light_tree_importance(precompute, w, lobe),
            Self::Mirror(_) | Self::Glass(_) | Self::Emissive(_) => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use glam::Vec3;

    use crate::{
        bsdf::{BsdfFlags, TransportMode},
        math::OrthonormalBasis,
        scene::{InstanceIndex, TriangleRef},
    };

    use super::{
        ConductorGgxMaterial, DielectricGgxMaterial, EmissiveMaterial, GlassMaterial, Material,
        MirrorMaterial, MtlxScratch, NormalizedLambertMaterial, ShadingVertex,
    };

    fn test_shading_vertex(wo: Vec3) -> ShadingVertex {
        ShadingVertex {
            triangle: TriangleRef {
                instance_index: InstanceIndex(0),
                triangle_index: 0,
            },
            p: Vec3::ZERO,
            uv: glam::Vec2::ZERO,
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
    fn non_emissive_materials_report_zero_emission_capability() {
        let cases: Vec<(Material, &str)> = vec![
            (
                Material::NormalizedLambert(NormalizedLambertMaterial::new(Vec3::ONE)),
                "lambert",
            ),
            (Material::Mirror(MirrorMaterial::new(Vec3::ONE)), "mirror"),
            (
                Material::ConductorGgx(ConductorGgxMaterial::new(Vec3::ONE, 0.5, 0.0)),
                "conductor_ggx",
            ),
            (
                Material::Glass(GlassMaterial::new(1.5, Vec3::ONE, false)),
                "glass",
            ),
            (
                Material::DielectricGgx(DielectricGgxMaterial::new(
                    Vec3::ONE,
                    1.5,
                    0.3,
                    0.0,
                    false,
                )),
                "dielectric_ggx",
            ),
        ];

        for (material, name) in cases {
            assert!(!material.may_emit(), "{name} should not emit");
            assert_eq!(
                material.max_emission(),
                0.0,
                "{name} max_emission should be zero"
            );
        }
    }

    #[test]
    fn emissive_material_reports_emission_capability() {
        let material = Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 3.0));

        assert!(material.may_emit());
        assert!((material.max_emission() - 3.0).abs() < 1.0e-3);
    }

    #[test]
    fn emissive_material_eval_is_always_zero() {
        let material = Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 2.0));
        let scratch = MtlxScratch::default();
        let shading_vertex = test_shading_vertex(Vec3::Z);
        assert_eq!(
            material.eval(
                &shading_vertex,
                &scratch,
                Vec3::Z,
                &mut crate::sampler::AuxRng::default(),
                TransportMode::Radiance,
            ),
            Vec3::ZERO
        );
    }

    #[test]
    fn lambert_material_eval_delegates_to_bsdf() {
        let material = Material::NormalizedLambert(NormalizedLambertMaterial::new(Vec3::ONE));
        let scratch = MtlxScratch::default();
        let shading_vertex = test_shading_vertex(Vec3::Z);
        let f = material.eval(
            &shading_vertex,
            &scratch,
            Vec3::Z,
            &mut crate::sampler::AuxRng::default(),
            TransportMode::Radiance,
        );

        assert!(f.abs_diff_eq(Vec3::ONE / std::f32::consts::PI, 1.0e-3));
    }

    #[test]
    fn lambert_material_pdf_delegates_to_bsdf() {
        let material = Material::NormalizedLambert(NormalizedLambertMaterial::new(Vec3::ONE));
        let scratch = MtlxScratch::default();
        let shading_vertex = test_shading_vertex(Vec3::Z);
        let wi = Vec3::new(0.2, 0.3, 0.9327379).normalize();

        let pdf = material.pdf(&shading_vertex, &scratch, wi);

        assert!((pdf - wi.z / PI).abs() < 1.0e-6);
    }

    #[test]
    fn lambert_material_sample_returns_diffuse_flag() {
        let material = Material::NormalizedLambert(NormalizedLambertMaterial::new(Vec3::ONE));
        let scratch = MtlxScratch::default();
        let shading_vertex = test_shading_vertex(Vec3::Z);
        let mut rng = crate::sampler::AuxRng::from_seed(0);

        let sample = material
            .sample(
                &shading_vertex,
                &scratch,
                &crate::sampler::MaterialSampleRandoms::from_aux_rng(&mut rng),
                &mut crate::sampler::AuxRng::default(),
                TransportMode::Radiance,
            )
            .expect("expected a valid sample");

        assert_eq!(sample.flags, BsdfFlags::DIFFUSE | BsdfFlags::REFLECTION);
    }

    #[test]
    fn mirror_material_sample_returns_delta_flag() {
        let material = Material::Mirror(MirrorMaterial::new(Vec3::ONE));
        let wo = Vec3::new(0.3, -0.4, 0.8660254).normalize();
        let scratch = MtlxScratch::default();
        let shading_vertex = test_shading_vertex(wo);
        let mut rng = crate::sampler::AuxRng::from_seed(0);

        let sample = material
            .sample(
                &shading_vertex,
                &scratch,
                &crate::sampler::MaterialSampleRandoms::from_aux_rng(&mut rng),
                &mut crate::sampler::AuxRng::default(),
                TransportMode::Radiance,
            )
            .expect("expected a valid sample");

        let expected_wi = Vec3::new(-wo.x, -wo.y, wo.z).normalize();
        assert!(sample.wi.abs_diff_eq(expected_wi, 1.0e-6));
        assert!(sample.weight.abs_diff_eq(Vec3::ONE, 1.0e-3));
        assert_eq!(sample.pdf, 1.0);
        assert_eq!(sample.flags, BsdfFlags::DELTA | BsdfFlags::REFLECTION);
    }

    #[test]
    fn conductor_material_sample_returns_glossy_or_delta_reflection_flag() {
        let material = Material::ConductorGgx(ConductorGgxMaterial::new(
            Vec3::new(0.9, 0.7, 0.3),
            0.0,
            0.0,
        ));
        let scratch = MtlxScratch::default();
        let shading_vertex = test_shading_vertex(Vec3::new(0.3, -0.4, 0.8660254).normalize());
        let mut rng = crate::sampler::AuxRng::from_seed(0);

        let sample = material
            .sample(
                &shading_vertex,
                &scratch,
                &crate::sampler::MaterialSampleRandoms::from_aux_rng(&mut rng),
                &mut crate::sampler::AuxRng::default(),
                TransportMode::Radiance,
            )
            .expect("expected a valid sample");

        assert!(sample.flags.contains(BsdfFlags::REFLECTION));
        assert!(
            sample.flags.contains(BsdfFlags::GLOSSY) || sample.flags.contains(BsdfFlags::DELTA)
        );
    }

    #[test]
    fn mirror_material_eval_and_pdf_are_zero() {
        let material = Material::Mirror(MirrorMaterial::new(Vec3::ONE));
        let scratch = MtlxScratch::default();
        let shading_vertex = test_shading_vertex(Vec3::Z);
        assert_eq!(
            material.eval(
                &shading_vertex,
                &scratch,
                Vec3::Z,
                &mut crate::sampler::AuxRng::default(),
                TransportMode::Radiance,
            ),
            Vec3::ZERO
        );
        assert_eq!(material.pdf(&shading_vertex, &scratch, Vec3::Z), 0.0);
    }

    #[test]
    fn glass_material_sample_can_return_transmission_flag() {
        let material = Material::Glass(GlassMaterial::new(1.5, Vec3::ONE, false));
        let wo = Vec3::new(0.3, -0.4, 0.8660254).normalize();
        let scratch = MtlxScratch::default();
        let shading_vertex = test_shading_vertex(wo);
        let mut rng = crate::sampler::AuxRng::from_seed(0);

        let sample = (0..64)
            .find_map(|_| {
                let s = material.sample(
                    &shading_vertex,
                    &scratch,
                    &crate::sampler::MaterialSampleRandoms::from_aux_rng(&mut rng),
                    &mut crate::sampler::AuxRng::default(),
                    TransportMode::Radiance,
                )?;
                s.flags.contains(BsdfFlags::TRANSMISSION).then_some(s)
            })
            .expect("expected a transmission sample within retry budget");

        assert!(sample.wi.z < 0.0);
        assert!(sample.weight.abs_diff_eq(Vec3::ONE * 2.25, 1.0e-3));
        assert_eq!(sample.flags, BsdfFlags::DELTA | BsdfFlags::TRANSMISSION);
    }

    #[test]
    fn glass_material_eval_and_pdf_are_zero() {
        let material = Material::Glass(GlassMaterial::new(1.5, Vec3::ONE, false));
        let scratch = MtlxScratch::default();
        let shading_vertex = test_shading_vertex(Vec3::Z);
        assert_eq!(
            material.eval(
                &shading_vertex,
                &scratch,
                Vec3::Z,
                &mut crate::sampler::AuxRng::default(),
                TransportMode::Radiance,
            ),
            Vec3::ZERO
        );
        assert_eq!(material.pdf(&shading_vertex, &scratch, Vec3::Z), 0.0);
    }

    #[test]
    fn emissive_material_pdf_is_always_zero_for_mis() {
        let material = Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 2.0));
        let scratch = MtlxScratch::default();
        let shading_vertex = test_shading_vertex(Vec3::Z);

        assert_eq!(material.pdf(&shading_vertex, &scratch, Vec3::Z), 0.0);
        assert_eq!(material.pdf(&shading_vertex, &scratch, -Vec3::Z), 0.0);
    }
}
