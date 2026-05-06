mod conductor_ggx;
mod conductor_ggx_cui_2023;
mod dielectric_ggx;
mod disney_brdf;
mod emissive;
mod eon;
mod glass;
mod mirror;
mod normal_map;
mod normalized_lambert;
mod simple_pbr;
mod standard_surface;
mod texture;

use glam::{Vec2, Vec3};
use rand::rngs::ThreadRng;

use crate::{
    bsdf::BsdfFlags,
    light_tree::LightTreePrecompute,
    math::{OrthonormalBasis, sg::SgLobe},
    scene::TriangleRef,
};

pub(super) const GEOMETRIC_NORMAL_COS_EPSILON: f32 = 1.0e-6;

pub use conductor_ggx::ConductorGgxMaterial;
pub use conductor_ggx_cui_2023::ConductorGgxCui2023Material;
pub use dielectric_ggx::DielectricGgxMaterial;
pub use disney_brdf::DisneyBrdfMaterial;
pub use emissive::EmissiveMaterial;
pub use eon::EonMaterial;
pub use glass::GlassMaterial;
pub use mirror::MirrorMaterial;
pub use normal_map::NormalMap;
pub use normalized_lambert::NormalizedLambertMaterial;
pub use simple_pbr::SimplePbrMaterial;
pub use standard_surface::StandardSurfaceMaterial;
pub use texture::{ScalarTexture, Texture, TextureColorSpace};

#[derive(Debug, Clone, PartialEq)]
pub enum Material {
    NormalizedLambert(NormalizedLambertMaterial),
    Eon(EonMaterial),
    Mirror(MirrorMaterial),
    ConductorGgx(ConductorGgxMaterial),
    ConductorGgxCui2023(ConductorGgxCui2023Material),
    DielectricGgx(DielectricGgxMaterial),
    Glass(GlassMaterial),
    SimplePBR(SimplePbrMaterial),
    DisneyBrdf(DisneyBrdfMaterial),
    StandardSurface(StandardSurfaceMaterial),
    Emissive(EmissiveMaterial),
}

#[derive(Debug, Clone, Copy, PartialEq)]
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
    pub wavelength_lock: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaterialSample {
    pub weight: Vec3,
    pub wi: Vec3,
    pub pdf: f32,
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
    pub(crate) fn prepare_shading_vertex(&self, shading_vertex: &ShadingVertex) -> ShadingVertex {
        match self {
            Self::NormalizedLambert(material) => material.prepare_shading_vertex(shading_vertex),
            Self::Eon(material) => material.prepare_shading_vertex(shading_vertex),
            Self::Mirror(material) => material.prepare_shading_vertex(shading_vertex),
            Self::ConductorGgx(material) => material.prepare_shading_vertex(shading_vertex),
            Self::ConductorGgxCui2023(material) => material.prepare_shading_vertex(shading_vertex),
            Self::DielectricGgx(material) => material.prepare_shading_vertex(shading_vertex),
            Self::Glass(material) => material.prepare_shading_vertex(shading_vertex),
            Self::SimplePBR(material) => material.prepare_shading_vertex(shading_vertex),
            Self::DisneyBrdf(material) => material.prepare_shading_vertex(shading_vertex),
            Self::StandardSurface(material) => material.prepare_shading_vertex(shading_vertex),
            Self::Emissive(_) => *shading_vertex,
        }
    }

    pub fn sample(
        &self,
        shading_vertex: &ShadingVertex,
        rng: &mut ThreadRng,
    ) -> Option<MaterialSample> {
        match self {
            Self::NormalizedLambert(material) => material.sample(shading_vertex, rng),
            Self::Eon(material) => material.sample(shading_vertex, rng),
            Self::Mirror(material) => material.sample(shading_vertex, rng),
            Self::ConductorGgx(material) => material.sample(shading_vertex, rng),
            Self::ConductorGgxCui2023(material) => material.sample(shading_vertex, rng),
            Self::DielectricGgx(material) => material.sample(shading_vertex, rng),
            Self::Glass(material) => material.sample(shading_vertex, rng),
            Self::SimplePBR(material) => material.sample(shading_vertex, rng),
            Self::DisneyBrdf(material) => material.sample(shading_vertex, rng),
            Self::StandardSurface(material) => material.sample(shading_vertex, rng),
            Self::Emissive(material) => material.sample(shading_vertex, rng),
        }
    }

    pub fn le(&self, shading_vertex: &ShadingVertex) -> Option<Vec3> {
        match self {
            Self::NormalizedLambert(material) => material.le(shading_vertex),
            Self::Eon(material) => material.le(shading_vertex),
            Self::Mirror(material) => material.le(shading_vertex),
            Self::ConductorGgx(material) => material.le(shading_vertex),
            Self::ConductorGgxCui2023(material) => material.le(shading_vertex),
            Self::DielectricGgx(material) => material.le(shading_vertex),
            Self::Glass(material) => material.le(shading_vertex),
            Self::SimplePBR(material) => material.le(shading_vertex),
            Self::DisneyBrdf(material) => material.le(shading_vertex),
            Self::StandardSurface(material) => material.le(shading_vertex),
            Self::Emissive(material) => material.le(shading_vertex),
        }
    }

    pub fn eval(
        &self,
        shading_vertex: &ShadingVertex,
        wi: Vec3,
        internal_rng: &mut ThreadRng,
    ) -> Vec3 {
        match self {
            Self::NormalizedLambert(material) => material.eval(shading_vertex, wi, internal_rng),
            Self::Eon(material) => material.eval(shading_vertex, wi, internal_rng),
            Self::Mirror(material) => material.eval(shading_vertex, wi, internal_rng),
            Self::ConductorGgx(material) => material.eval(shading_vertex, wi, internal_rng),
            Self::ConductorGgxCui2023(material) => material.eval(shading_vertex, wi, internal_rng),
            Self::DielectricGgx(material) => material.eval(shading_vertex, wi, internal_rng),
            Self::Glass(material) => material.eval(shading_vertex, wi, internal_rng),
            Self::SimplePBR(material) => material.eval(shading_vertex, wi, internal_rng),
            Self::DisneyBrdf(material) => material.eval(shading_vertex, wi, internal_rng),
            Self::StandardSurface(material) => material.eval(shading_vertex, wi, internal_rng),
            Self::Emissive(material) => material.eval(shading_vertex, wi, internal_rng),
        }
    }

    pub fn pdf(&self, shading_vertex: &ShadingVertex, wi: Vec3) -> f32 {
        match self {
            Self::NormalizedLambert(material) => material.pdf(shading_vertex, wi),
            Self::Eon(material) => material.pdf(shading_vertex, wi),
            Self::Mirror(material) => material.pdf(shading_vertex, wi),
            Self::ConductorGgx(material) => material.pdf(shading_vertex, wi),
            Self::ConductorGgxCui2023(material) => material.pdf(shading_vertex, wi),
            Self::DielectricGgx(material) => material.pdf(shading_vertex, wi),
            Self::Glass(material) => material.pdf(shading_vertex, wi),
            Self::SimplePBR(material) => material.pdf(shading_vertex, wi),
            Self::DisneyBrdf(material) => material.pdf(shading_vertex, wi),
            Self::StandardSurface(material) => material.pdf(shading_vertex, wi),
            Self::Emissive(material) => material.pdf(shading_vertex, wi),
        }
    }

    pub fn may_emit(&self) -> bool {
        match self {
            Self::NormalizedLambert(material) => material.may_emit(),
            Self::Eon(material) => material.may_emit(),
            Self::Mirror(material) => material.may_emit(),
            Self::ConductorGgx(material) => material.may_emit(),
            Self::ConductorGgxCui2023(material) => material.may_emit(),
            Self::DielectricGgx(material) => material.may_emit(),
            Self::Glass(material) => material.may_emit(),
            Self::SimplePBR(material) => material.may_emit(),
            Self::DisneyBrdf(material) => material.may_emit(),
            Self::StandardSurface(material) => material.may_emit(),
            Self::Emissive(material) => material.may_emit(),
        }
    }

    pub fn max_emission(&self) -> f32 {
        match self {
            Self::NormalizedLambert(material) => material.max_emission(),
            Self::Eon(material) => material.max_emission(),
            Self::Mirror(material) => material.max_emission(),
            Self::ConductorGgx(material) => material.max_emission(),
            Self::ConductorGgxCui2023(material) => material.max_emission(),
            Self::DielectricGgx(material) => material.max_emission(),
            Self::Glass(material) => material.max_emission(),
            Self::SimplePBR(material) => material.max_emission(),
            Self::DisneyBrdf(material) => material.max_emission(),
            Self::StandardSurface(material) => material.max_emission(),
            Self::Emissive(material) => material.max_emission(),
        }
    }

    /// Returns true when this material can ever reject a hit through its
    /// `any_hit`. Renderers should short-circuit and skip the
    /// `ShadingVertex` construction whenever this returns false.
    pub fn has_alpha_test(&self) -> bool {
        match self {
            Self::NormalizedLambert(material) => material.has_alpha_test(),
            Self::Eon(material) => material.has_alpha_test(),
            Self::Mirror(material) => material.has_alpha_test(),
            Self::ConductorGgx(material) => material.has_alpha_test(),
            Self::ConductorGgxCui2023(material) => material.has_alpha_test(),
            Self::DielectricGgx(material) => material.has_alpha_test(),
            Self::Glass(material) => material.has_alpha_test(),
            Self::SimplePBR(material) => material.has_alpha_test(),
            Self::DisneyBrdf(material) => material.has_alpha_test(),
            Self::StandardSurface(material) => material.has_alpha_test(),
            Self::Emissive(material) => material.has_alpha_test(),
        }
    }

    /// any-hit shader equivalent. The renderer asks "is this surface hit
    /// accepted?" and the material answers with a single yes/no using the
    /// supplied uniform sample `u` (in `[0, 1)`). Materials are free to use
    /// `u` however they like (probabilistic transmission, hard cutoff,
    /// procedural opacity, ...). Materials that never reject simply return
    /// true.
    pub fn any_hit(&self, shading_vertex: &ShadingVertex, u: f32) -> bool {
        match self {
            Self::NormalizedLambert(material) => material.any_hit(shading_vertex, u),
            Self::Eon(material) => material.any_hit(shading_vertex, u),
            Self::Mirror(material) => material.any_hit(shading_vertex, u),
            Self::ConductorGgx(material) => material.any_hit(shading_vertex, u),
            Self::ConductorGgxCui2023(material) => material.any_hit(shading_vertex, u),
            Self::DielectricGgx(material) => material.any_hit(shading_vertex, u),
            Self::Glass(material) => material.any_hit(shading_vertex, u),
            Self::SimplePBR(material) => material.any_hit(shading_vertex, u),
            Self::DisneyBrdf(material) => material.any_hit(shading_vertex, u),
            Self::StandardSurface(material) => material.any_hit(shading_vertex, u),
            Self::Emissive(material) => material.any_hit(shading_vertex, u),
        }
    }

    /// Per-shading-point precompute for the hierarchical light tree. Each
    /// material returns a `LightTreePrecompute` that captures whichever
    /// lobes (diffuse / glossy / btdf) it needs the tree to favour. Returns
    /// `None` for delta lobes (mirror / glass) and emissive surfaces -- the
    /// integrator skips NEE on those anyway, so the tree query never fires.
    ///
    /// See `light_tree::lobe` for the shared SG/lobe helpers materials use
    /// to populate the precompute. See the "MULTI-LOBE NOTES" doc comment
    /// in that module before adding multi-glossy or multi-BTDF materials.
    pub fn light_tree_precompute(
        &self,
        shading_vertex: &ShadingVertex,
    ) -> Option<LightTreePrecompute> {
        match self {
            Self::NormalizedLambert(material) => material.light_tree_precompute(shading_vertex),
            Self::Eon(material) => material.light_tree_precompute(shading_vertex),
            Self::ConductorGgx(material) => material.light_tree_precompute(shading_vertex),
            Self::ConductorGgxCui2023(material) => material.light_tree_precompute(shading_vertex),
            Self::DielectricGgx(material) => material.light_tree_precompute(shading_vertex),
            Self::SimplePBR(material) => material.light_tree_precompute(shading_vertex),
            Self::DisneyBrdf(material) => material.light_tree_precompute(shading_vertex),
            Self::StandardSurface(material) => material.light_tree_precompute(shading_vertex),
            Self::Mirror(_) | Self::Glass(_) | Self::Emissive(_) => None,
        }
    }

    /// Convolve the SG light `W * g(o; xi, kappa)` with this material's
    /// lobes and return a non-negative importance. The `precompute` is the
    /// value returned by `light_tree_precompute` at the shading vertex.
    ///
    /// Implementations are expected to delegate to the helpers in
    /// `light_tree::lobe` (see also the multi-lobe guidance there).
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
            Self::Eon(material) => material.light_tree_importance(precompute, w, lobe),
            Self::ConductorGgx(material) => material.light_tree_importance(precompute, w, lobe),
            Self::ConductorGgxCui2023(material) => {
                material.light_tree_importance(precompute, w, lobe)
            }
            Self::DielectricGgx(material) => material.light_tree_importance(precompute, w, lobe),
            Self::SimplePBR(material) => material.light_tree_importance(precompute, w, lobe),
            Self::DisneyBrdf(material) => material.light_tree_importance(precompute, w, lobe),
            Self::StandardSurface(material) => material.light_tree_importance(precompute, w, lobe),
            Self::Mirror(_) | Self::Glass(_) | Self::Emissive(_) => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use glam::Vec3;

    use crate::{
        bsdf::BsdfFlags,
        math::OrthonormalBasis,
        scene::{InstanceIndex, TriangleRef},
    };

    use super::{
        ConductorGgxMaterial, DielectricGgxMaterial, EmissiveMaterial, GlassMaterial, Material,
        MirrorMaterial, NormalizedLambertMaterial, ShadingVertex,
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
            wavelength_lock: None,
        }
    }

    #[test]
    fn emissive_material_reports_emission_capability() {
        let material = Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 3.0));

        assert!(material.may_emit());
        assert_eq!(material.max_emission(), 3.0);
    }

    #[test]
    fn lambert_material_reports_no_emission_capability() {
        let material = Material::NormalizedLambert(NormalizedLambertMaterial::new(Vec3::ONE));

        assert!(!material.may_emit());
        assert_eq!(material.max_emission(), 0.0);
    }

    #[test]
    fn mirror_material_reports_no_emission_capability() {
        let material = Material::Mirror(MirrorMaterial::new(Vec3::ONE));

        assert!(!material.may_emit());
        assert_eq!(material.max_emission(), 0.0);
    }

    #[test]
    fn conductor_material_reports_no_emission_capability() {
        let material = Material::ConductorGgx(ConductorGgxMaterial::new(Vec3::ONE, 0.5, 0.0));

        assert!(!material.may_emit());
        assert_eq!(material.max_emission(), 0.0);
    }

    #[test]
    fn glass_material_reports_no_emission_capability() {
        let material = Material::Glass(GlassMaterial::new(1.5, Vec3::ONE, false));

        assert!(!material.may_emit());
        assert_eq!(material.max_emission(), 0.0);
    }

    #[test]
    fn dielectric_ggx_material_reports_no_emission_capability() {
        let material =
            Material::DielectricGgx(DielectricGgxMaterial::new(Vec3::ONE, 1.5, 0.3, 0.0, false));

        assert!(!material.may_emit());
        assert_eq!(material.max_emission(), 0.0);
    }

    #[test]
    fn emissive_material_eval_is_always_zero() {
        let material = Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 2.0));
        let shading_vertex = test_shading_vertex(Vec3::Z);
        let mut rng = rand::rng();

        assert_eq!(
            material.eval(&shading_vertex, Vec3::Z, &mut rng),
            Vec3::ZERO
        );
    }

    #[test]
    fn lambert_material_eval_delegates_to_bsdf() {
        let material = Material::NormalizedLambert(NormalizedLambertMaterial::new(Vec3::ONE));
        let shading_vertex = test_shading_vertex(Vec3::Z);
        let mut rng = rand::rng();
        let f = material.eval(&shading_vertex, Vec3::Z, &mut rng);

        assert!(f.abs_diff_eq(Vec3::ONE / std::f32::consts::PI, 1.0e-6));
    }

    #[test]
    fn lambert_material_pdf_delegates_to_bsdf() {
        let material = Material::NormalizedLambert(NormalizedLambertMaterial::new(Vec3::ONE));
        let shading_vertex = test_shading_vertex(Vec3::Z);
        let wi = Vec3::new(0.2, 0.3, 0.9327379).normalize();

        let pdf = material.pdf(&shading_vertex, wi);

        assert!((pdf - wi.z / PI).abs() < 1.0e-6);
    }

    #[test]
    fn lambert_material_sample_returns_diffuse_flag() {
        let material = Material::NormalizedLambert(NormalizedLambertMaterial::new(Vec3::ONE));
        let shading_vertex = test_shading_vertex(Vec3::Z);
        let mut rng = rand::rng();

        let sample = material
            .sample(&shading_vertex, &mut rng)
            .expect("expected a valid sample");

        assert_eq!(sample.flags, BsdfFlags::DIFFUSE | BsdfFlags::REFLECTION);
    }

    #[test]
    fn mirror_material_sample_returns_delta_flag() {
        let material = Material::Mirror(MirrorMaterial::new(Vec3::ONE));
        let wo = Vec3::new(0.3, -0.4, 0.8660254).normalize();
        let shading_vertex = test_shading_vertex(wo);
        let mut rng = rand::rng();

        let sample = material
            .sample(&shading_vertex, &mut rng)
            .expect("expected a valid sample");

        let expected_wi = Vec3::new(-wo.x, -wo.y, wo.z).normalize();
        assert!(sample.wi.abs_diff_eq(expected_wi, 1.0e-6));
        assert_eq!(sample.weight, Vec3::ONE);
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
        let shading_vertex = test_shading_vertex(Vec3::new(0.3, -0.4, 0.8660254).normalize());
        let mut rng = rand::rng();

        let sample = material
            .sample(&shading_vertex, &mut rng)
            .expect("expected a valid sample");

        assert!(sample.flags.contains(BsdfFlags::REFLECTION));
        assert!(
            sample.flags.contains(BsdfFlags::GLOSSY) || sample.flags.contains(BsdfFlags::DELTA)
        );
    }

    #[test]
    fn mirror_material_eval_and_pdf_are_zero() {
        let material = Material::Mirror(MirrorMaterial::new(Vec3::ONE));
        let shading_vertex = test_shading_vertex(Vec3::Z);
        let mut rng = rand::rng();

        assert_eq!(
            material.eval(&shading_vertex, Vec3::Z, &mut rng),
            Vec3::ZERO
        );
        assert_eq!(material.pdf(&shading_vertex, Vec3::Z), 0.0);
    }

    #[test]
    fn glass_material_sample_can_return_transmission_flag() {
        let material = Material::Glass(GlassMaterial::new(1.5, Vec3::ONE, false));
        let wo = Vec3::new(0.3, -0.4, 0.8660254).normalize();
        let shading_vertex = test_shading_vertex(wo);
        let mut rng = rand::rng();

        let sample = (0..64)
            .find_map(|_| {
                let s = material.sample(&shading_vertex, &mut rng)?;
                s.flags.contains(BsdfFlags::TRANSMISSION).then_some(s)
            })
            .expect("expected a transmission sample within retry budget");

        assert!(sample.wi.z < 0.0);
        assert!(sample.weight.abs_diff_eq(Vec3::ONE * 2.25, 1.0e-6));
        assert_eq!(sample.flags, BsdfFlags::DELTA | BsdfFlags::TRANSMISSION);
    }

    #[test]
    fn glass_material_eval_and_pdf_are_zero() {
        let material = Material::Glass(GlassMaterial::new(1.5, Vec3::ONE, false));
        let shading_vertex = test_shading_vertex(Vec3::Z);
        let mut rng = rand::rng();

        assert_eq!(
            material.eval(&shading_vertex, Vec3::Z, &mut rng),
            Vec3::ZERO
        );
        assert_eq!(material.pdf(&shading_vertex, Vec3::Z), 0.0);
    }

    #[test]
    fn emissive_material_pdf_is_always_zero_for_mis() {
        let material = Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 2.0));
        let shading_vertex = test_shading_vertex(Vec3::Z);

        assert_eq!(material.pdf(&shading_vertex, Vec3::Z), 0.0);
        assert_eq!(material.pdf(&shading_vertex, -Vec3::Z), 0.0);
    }
}
