use std::sync::Arc;

use glam::Vec3;

use crate::bsdf::{
    MtlxDielectricGgxDirectionalAlbedoLut, MtlxGeneralizedSchlickGgxDirectionalAlbedoLut,
    SheenDirectionalAlbedoLut,
};
use crate::light_tree::LightTreePrecompute;
use crate::math::sg::SgLobe;
use crate::sampler::{AuxRng, MaterialSampleRandoms};

use super::mtlx::{self, CompiledMaterial, MtlxScratch};
use super::{GEOMETRIC_NORMAL_COS_EPSILON, MaterialSample, ShadingVertex};

const DIFFUSE_CONE_SPREAD: f32 = 0.5;

#[derive(Debug, Clone)]
pub struct MtlxMaterial {
    pub compiled: Arc<CompiledMaterial>,
    pub back_compiled: Option<Arc<CompiledMaterial>>,
}

impl PartialEq for MtlxMaterial {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.compiled, &other.compiled)
            && match (&self.back_compiled, &other.back_compiled) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
    }
}

impl MtlxMaterial {
    pub fn new(compiled: Arc<CompiledMaterial>) -> Self {
        Self {
            compiled,
            back_compiled: None,
        }
    }

    pub fn with_back(
        compiled: Arc<CompiledMaterial>,
        back_compiled: Option<Arc<CompiledMaterial>>,
    ) -> Self {
        Self {
            compiled,
            back_compiled,
        }
    }

    pub(crate) fn install_sheen_lut(&mut self, lut: Arc<SheenDirectionalAlbedoLut>) {
        Arc::make_mut(&mut self.compiled).sheen_lut = Some(Arc::clone(&lut));
        if let Some(back) = &mut self.back_compiled {
            Arc::make_mut(back).sheen_lut = Some(lut);
        }
    }

    pub(crate) fn install_mtlx_dielectric_lut(
        &mut self,
        lut: Arc<MtlxDielectricGgxDirectionalAlbedoLut>,
    ) {
        Arc::make_mut(&mut self.compiled).mtlx_dielectric_lut = Some(Arc::clone(&lut));
        if let Some(back) = &mut self.back_compiled {
            Arc::make_mut(back).mtlx_dielectric_lut = Some(lut);
        }
    }

    pub(crate) fn install_mtlx_generalized_schlick_lut(
        &mut self,
        lut: Arc<MtlxGeneralizedSchlickGgxDirectionalAlbedoLut>,
    ) {
        Arc::make_mut(&mut self.compiled).mtlx_generalized_schlick_lut = Some(Arc::clone(&lut));
        if let Some(back) = &mut self.back_compiled {
            Arc::make_mut(back).mtlx_generalized_schlick_lut = Some(lut);
        }
    }

    fn active(&self, sv: &ShadingVertex) -> &Arc<CompiledMaterial> {
        if !sv.front_face
            && let Some(back) = &self.back_compiled
        {
            return back;
        }
        &self.compiled
    }

    pub(crate) fn prepare_shading_vertex(&self, _sv: &mut ShadingVertex) {}

    /// Runs the compiled bytecode for this shading vertex and stores the
    /// resulting locals on the vertex itself. The integrator must call this
    /// once per intersection before any `sample` / `eval` / `pdf` / `le` /
    /// `light_tree_precompute` query; subsequent invocations on the same
    /// `(material, sv)` pair short-circuit via `sv.mtlx_precomputed_for`.
    pub fn precompute_shading(&self, sv: &mut ShadingVertex, scratch: &mut MtlxScratch) {
        let active = self.active(sv);
        let key = Arc::as_ptr(active) as usize;
        if sv.mtlx_precomputed_for == Some(key) {
            return;
        }
        let compiled: &CompiledMaterial = active;
        let handle = scratch.alloc_regs(compiled.num_registers as usize);
        let dalbedo = scratch.alloc_dalbedo_cache(compiled.closure_nodes.len());
        mtlx::runtime::run_instructions(compiled, sv, scratch, handle);
        sv.mtlx_regs = Some(handle);
        sv.mtlx_dalbedo = Some(dalbedo);
        sv.mtlx_precomputed_for = Some(key);
    }

    #[inline(always)]
    fn dalbedo_cache<'a>(
        sv: &ShadingVertex,
        scratch: &'a MtlxScratch,
    ) -> &'a [std::cell::Cell<Option<Vec3>>] {
        scratch.dalbedo_slice(sv.mtlx_dalbedo.expect("precompute_shading not called"))
    }

    pub fn sample(
        &self,
        sv: &ShadingVertex,
        scratch: &MtlxScratch,
        randoms: &MaterialSampleRandoms,
        _aux_rng: &mut AuxRng,
    ) -> Option<MaterialSample> {
        let active = self.active(sv);
        let thin_walled = active.thin_walled;
        let regs = scratch.regs_slice(sv.mtlx_regs.expect("precompute_shading not called"));
        let dalbedo_cache = Self::dalbedo_cache(sv, scratch);
        let candidate =
            mtlx::runtime::sample_closure_cached(active, regs, sv, randoms, dalbedo_cache)?;
        let mut wi_local = candidate.wi_local;
        let mut eta = candidate.eta;
        let is_transmission = candidate
            .flags
            .contains(crate::bsdf::BsdfFlags::TRANSMISSION);
        if thin_walled && is_transmission {
            let wo_local = sv.frame.world_to_local(sv.wo).normalize_or_zero();
            wi_local = Vec3::new(-wo_local.x, -wo_local.y, -wo_local.z);
            eta = 1.0;
        }
        let wi = sv.frame.local_to_world(wi_local);
        let same_side = wi.dot(sv.ng) > GEOMETRIC_NORMAL_COS_EPSILON;
        let opposite_side = wi.dot(sv.ng) < -GEOMETRIC_NORMAL_COS_EPSILON;
        if !((is_transmission && opposite_side) || (!is_transmission && same_side)) {
            return None;
        }

        let wo_local = sv.frame.world_to_local(sv.wo).normalize_or_zero();
        let (f, pdf) = if thin_walled && is_transmission {
            (mtlx::runtime::thin_walled_transmittance(active, regs), 1.0)
        } else {
            let (f, pdf) = mtlx::runtime::eval_pdf_closure_cached(
                active,
                regs,
                sv,
                wo_local,
                wi_local,
                dalbedo_cache,
            );
            (f, pdf)
        };
        if pdf <= 1.0e-6 {
            return None;
        }
        let weight = if thin_walled && is_transmission {
            f
        } else {
            f * (wi_local.z.abs() / pdf)
        };

        let flags = if thin_walled && is_transmission {
            candidate.flags | crate::bsdf::BsdfFlags::DELTA
        } else {
            candidate.flags
        };

        Some(MaterialSample {
            weight,
            wi,
            pdf,
            flags,
            eta,
            cone_spread: DIFFUSE_CONE_SPREAD,
            wavelength_lock: None,
        })
    }

    pub fn eval(
        &self,
        sv: &ShadingVertex,
        scratch: &MtlxScratch,
        wi: Vec3,
        _aux_rng: &mut AuxRng,
    ) -> Vec3 {
        if sv.wo.dot(sv.ng) <= 0.0 {
            return Vec3::ZERO;
        }
        if wi.dot(sv.ng).abs() <= GEOMETRIC_NORMAL_COS_EPSILON {
            return Vec3::ZERO;
        }
        let active = self.active(sv);
        if active.thin_walled && wi.dot(sv.ng) < -GEOMETRIC_NORMAL_COS_EPSILON {
            return Vec3::ZERO;
        }
        let wo_local = sv.frame.world_to_local(sv.wo).normalize_or_zero();
        let wi_local = sv.frame.world_to_local(wi).normalize_or_zero();
        let regs = scratch.regs_slice(sv.mtlx_regs.expect("precompute_shading not called"));
        mtlx::runtime::eval_closure_cached(
            active,
            regs,
            sv,
            wo_local,
            wi_local,
            Self::dalbedo_cache(sv, scratch),
        )
    }

    pub fn pdf(&self, sv: &ShadingVertex, scratch: &MtlxScratch, wi: Vec3) -> f32 {
        if sv.wo.dot(sv.ng) <= 0.0 {
            return 0.0;
        }
        if wi.dot(sv.ng).abs() <= GEOMETRIC_NORMAL_COS_EPSILON {
            return 0.0;
        }
        let active = self.active(sv);
        if active.thin_walled && wi.dot(sv.ng) < -GEOMETRIC_NORMAL_COS_EPSILON {
            return 0.0;
        }
        let wo_local = sv.frame.world_to_local(sv.wo).normalize_or_zero();
        let wi_local = sv.frame.world_to_local(wi).normalize_or_zero();
        let regs = scratch.regs_slice(sv.mtlx_regs.expect("precompute_shading not called"));
        mtlx::runtime::pdf_closure_cached(
            active,
            regs,
            sv,
            wo_local,
            wi_local,
            Self::dalbedo_cache(sv, scratch),
        )
    }

    pub fn eval_pdf(&self, sv: &ShadingVertex, scratch: &MtlxScratch, wi: Vec3) -> (Vec3, f32) {
        if sv.wo.dot(sv.ng) <= 0.0 {
            return (Vec3::ZERO, 0.0);
        }
        if wi.dot(sv.ng).abs() <= GEOMETRIC_NORMAL_COS_EPSILON {
            return (Vec3::ZERO, 0.0);
        }
        let active = self.active(sv);
        if active.thin_walled && wi.dot(sv.ng) < -GEOMETRIC_NORMAL_COS_EPSILON {
            return (Vec3::ZERO, 0.0);
        }
        let wo_local = sv.frame.world_to_local(sv.wo).normalize_or_zero();
        let wi_local = sv.frame.world_to_local(wi).normalize_or_zero();
        let regs = scratch.regs_slice(sv.mtlx_regs.expect("precompute_shading not called"));
        mtlx::runtime::eval_pdf_closure_cached(
            active,
            regs,
            sv,
            wo_local,
            wi_local,
            Self::dalbedo_cache(sv, scratch),
        )
    }

    pub fn le(&self, sv: &ShadingVertex, scratch: &MtlxScratch) -> Option<Vec3> {
        let active = self.active(sv);
        if !active.may_emit {
            return None;
        }
        let regs = scratch.regs_slice(sv.mtlx_regs.expect("precompute_shading not called"));
        mtlx::runtime::evaluate_le(active, regs, sv)
    }

    pub fn may_emit(&self) -> bool {
        self.compiled.may_emit
            || self
                .back_compiled
                .as_ref()
                .is_some_and(|back| back.may_emit)
    }

    pub fn max_emission(&self) -> f32 {
        self.back_compiled
            .as_ref()
            .map_or(self.compiled.max_emission, |back| {
                self.compiled.max_emission.max(back.max_emission)
            })
    }

    pub fn has_alpha_test(&self) -> bool {
        self.compiled.passthrough
            || self.compiled.has_opacity_test
            || self
                .back_compiled
                .as_ref()
                .is_some_and(|back| back.passthrough || back.has_opacity_test)
    }

    pub fn any_hit(&self, sv: &mut ShadingVertex, scratch: &mut MtlxScratch, u: f32) -> bool {
        let active = self.active(sv);
        if active.passthrough {
            false
        } else if !active.has_opacity_test {
            true
        } else {
            let handle = scratch.alloc_regs(active.opacity_num_registers as usize);
            mtlx::runtime::run_opacity_instructions(active, sv, scratch, handle);
            let regs = scratch.regs_slice(handle);
            let opacity = mtlx::runtime::opacity_for_alpha_test(active, regs);
            u < opacity
        }
    }

    pub fn light_tree_precompute(
        &self,
        sv: &ShadingVertex,
        scratch: &MtlxScratch,
    ) -> Option<LightTreePrecompute> {
        let active = self.active(sv);
        let wo_local = sv.frame.world_to_local(sv.wo).normalize_or_zero();
        let regs = scratch.regs_slice(sv.mtlx_regs.expect("precompute_shading not called"));
        mtlx::runtime::light_tree_precompute_closure_cached(
            active,
            regs,
            sv,
            wo_local,
            Self::dalbedo_cache(sv, scratch),
        )
    }

    pub fn light_tree_importance(
        &self,
        precompute: &LightTreePrecompute,
        w: f32,
        lobe: &SgLobe,
    ) -> f32 {
        let mut imp = 0.0;
        if let Some(d) = precompute.diffuse {
            imp += crate::light_tree::diffuse_importance(d, precompute.n, w, lobe);
        }
        if let Some(g) = precompute.glossy {
            imp += crate::light_tree::glossy_importance(g, precompute.frame, precompute.n, w, lobe);
        }
        if let Some(t) = precompute.btdf {
            imp += crate::light_tree::btdf_importance(t, precompute.frame, precompute.n, w, lobe);
        }
        imp.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bsdf::mtlx::ScatterMode;
    use crate::material::Material;
    use crate::material::mtlx::compiled::{ClosureNode, CompiledMaterial, ParamRef};
    use crate::math::OrthonormalBasis;
    use crate::scene::{InstanceIndex, TriangleRef};
    use glam::Vec2;
    use std::path::Path;

    fn synthetic_sv(front_face: bool) -> ShadingVertex {
        let ng = Vec3::Z;
        let ns = Vec3::Z;
        let cos_in = 0.5_f32;
        let sin_in = (1.0 - cos_in * cos_in).sqrt();
        let wo = Vec3::new(sin_in, 0.0, cos_in);
        ShadingVertex {
            triangle: TriangleRef {
                instance_index: InstanceIndex(0),
                triangle_index: 0,
            },
            p: Vec3::ZERO,
            uv: Vec2::new(0.5, 0.5),
            dudx: 0.0,
            dvdx: 0.0,
            dudy: 0.0,
            dvdy: 0.0,
            ng,
            ns,
            wo,
            dpdu: Vec3::X,
            dpdv: Vec3::Y,
            dpdx: Vec3::ZERO,
            dpdy: Vec3::ZERO,
            dndu: Vec3::ZERO,
            dndv: Vec3::ZERO,
            frame: OrthonormalBasis::from_normal(ns),
            front_face,
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

    fn load_shader_ops_thin_walled() -> MtlxMaterial {
        use crate::bsdf::DirectionalAlbedoCache;
        use crate::scene::mtlx_loader::{load_mtlx_material, load_standard_library};
        let lib = load_standard_library(Path::new("lib/materialx/libraries"))
            .expect("standard library should load");
        let ocio = crate::color::OcioColorPipeline::new(
            crate::color::DEFAULT_OCIO_CONFIG,
            Some(crate::color::DEFAULT_RENDERING_SPACE.to_string()),
            crate::color::DEFAULT_TEXTURE_COLOR_SPACE,
        )
        .expect("default OCIO config");
        let mut material = load_mtlx_material(
            &lib,
            Path::new("assets/mtlx/shader_ops.mtlx"),
            "material_checker_opacity",
            &ocio,
        )
        .expect("shader_ops material should load");
        let mut cache = DirectionalAlbedoCache::default();
        material.install_sheen_lut(cache.get_or_build_sheen());
        material.install_mtlx_dielectric_lut(cache.get_or_build_mtlx_dielectric_ggx());
        material.install_mtlx_generalized_schlick_lut(
            cache.get_or_build_mtlx_generalized_schlick_ggx(),
        );
        material
    }

    fn dummy_compiled(
        passthrough: bool,
        has_opacity_test: bool,
        may_emit: bool,
        max_emission: f32,
    ) -> Arc<CompiledMaterial> {
        Arc::new(CompiledMaterial {
            instructions: vec![],
            operand_pool: vec![],
            value_pool: vec![],
            color_processors: Vec::new(),
            opacity_instructions: Vec::new(),
            opacity_operand_pool: Vec::new(),
            opacity_closure_nodes: Vec::new(),
            opacity_num_registers: 0,
            num_registers: 0,
            closure_nodes: vec![ClosureNode::Zero],
            root: 0,
            passthrough,
            max_emission,
            may_emit,
            has_opacity_test,
            thin_walled: false,
            sheen_lut: None,
            mtlx_dielectric_lut: None,
            mtlx_generalized_schlick_lut: None,
        })
    }

    fn closure_compiled(
        root: u32,
        nodes: Vec<ClosureNode>,
        thin_walled: bool,
    ) -> Arc<CompiledMaterial> {
        let mut cache = crate::bsdf::DirectionalAlbedoCache::default();
        Arc::new(CompiledMaterial {
            instructions: vec![],
            operand_pool: vec![],
            value_pool: vec![],
            color_processors: Vec::new(),
            opacity_instructions: Vec::new(),
            opacity_operand_pool: Vec::new(),
            opacity_closure_nodes: Vec::new(),
            opacity_num_registers: 0,
            num_registers: 0,
            closure_nodes: nodes,
            root,
            passthrough: false,
            max_emission: 0.0,
            may_emit: false,
            has_opacity_test: false,
            thin_walled,
            sheen_lut: Some(cache.get_or_build_sheen()),
            mtlx_dielectric_lut: Some(cache.get_or_build_mtlx_dielectric_ggx()),
            mtlx_generalized_schlick_lut: Some(cache.get_or_build_mtlx_generalized_schlick_ggx()),
        })
    }

    #[test]
    fn front_back_material_flags_are_combined() {
        let mat = MtlxMaterial::with_back(
            dummy_compiled(false, false, false, 0.25),
            Some(dummy_compiled(false, true, true, 2.0)),
        );

        assert!(mat.may_emit());
        assert!(mat.has_alpha_test());
        assert_eq!(mat.max_emission(), 2.0);
    }

    #[test]
    fn any_hit_uses_active_back_material_passthrough() {
        let mat = MtlxMaterial::with_back(
            dummy_compiled(false, false, false, 0.0),
            Some(dummy_compiled(true, false, false, 0.0)),
        );
        let mut sv = synthetic_sv(false);
        let mut scratch = MtlxScratch::default();

        assert!(!mat.any_hit(&mut sv, &mut scratch, 0.5));
    }

    #[test]
    fn light_tree_precompute_keeps_conductor_as_glossy_lobe() {
        let mat = MtlxMaterial::new(closure_compiled(
            0,
            vec![
                ClosureNode::Surface {
                    bsdf: 1,
                    edf: 2,
                    opacity: ParamRef::Float(1.0),
                    thin_walled: false,
                },
                ClosureNode::Conductor {
                    weight: ParamRef::Float(1.0),
                    ior: ParamRef::Color3(Vec3::splat(0.2)),
                    extinction: ParamRef::Color3(Vec3::splat(3.0)),
                    roughness: ParamRef::Vector2(Vec2::splat(0.2)),
                    thinfilm_thickness: ParamRef::Float(0.0),
                    thinfilm_ior: ParamRef::Float(1.0),
                    normal: None,
                    tangent: None,
                },
                ClosureNode::Zero,
            ],
            false,
        ));
        let mut sv = synthetic_sv(true);
        let mut scratch = MtlxScratch::default();

        mat.precompute_shading(&mut sv, &mut scratch);
        let pre = mat
            .light_tree_precompute(&sv, &scratch)
            .expect("conductor should produce light tree precompute");

        assert!(pre.diffuse.is_none());
        assert!(pre.glossy.is_some());
        assert!(pre.btdf.is_none());
    }

    #[test]
    fn layer_with_non_layerable_conductor_top_blocks_base() {
        let conductor = ClosureNode::Conductor {
            weight: ParamRef::Float(1.0),
            ior: ParamRef::Color3(Vec3::splat(0.2)),
            extinction: ParamRef::Color3(Vec3::splat(3.0)),
            roughness: ParamRef::Vector2(Vec2::splat(0.2)),
            thinfilm_thickness: ParamRef::Float(0.0),
            thinfilm_ior: ParamRef::Float(1.0),
            normal: None,
            tangent: None,
        };
        let base = ClosureNode::BurleyDiffuse {
            weight: ParamRef::Float(1.0),
            color: ParamRef::Color3(Vec3::new(1.0, 0.0, 0.0)),
            roughness: ParamRef::Float(0.0),
            normal: None,
        };
        let direct = MtlxMaterial::new(closure_compiled(
            0,
            vec![
                ClosureNode::Surface {
                    bsdf: 1,
                    edf: 2,
                    opacity: ParamRef::Float(1.0),
                    thin_walled: false,
                },
                conductor.clone(),
                ClosureNode::Zero,
            ],
            false,
        ));
        let layered = MtlxMaterial::new(closure_compiled(
            0,
            vec![
                ClosureNode::Surface {
                    bsdf: 1,
                    edf: 4,
                    opacity: ParamRef::Float(1.0),
                    thin_walled: false,
                },
                ClosureNode::Layer { top: 2, base: 3 },
                conductor,
                base,
                ClosureNode::Zero,
            ],
            false,
        ));
        let mut sv = synthetic_sv(true);
        let mut direct_scratch = MtlxScratch::default();
        let mut layered_scratch = MtlxScratch::default();
        direct.precompute_shading(&mut sv, &mut direct_scratch);
        let direct_f = direct.eval(
            &sv,
            &direct_scratch,
            Vec3::Z,
            &mut crate::sampler::AuxRng::default(),
        );
        sv.mtlx_regs = None;
        sv.mtlx_precomputed_for = None;
        layered.precompute_shading(&mut sv, &mut layered_scratch);
        let layered_f = layered.eval(
            &sv,
            &layered_scratch,
            Vec3::Z,
            &mut crate::sampler::AuxRng::default(),
        );

        assert!(layered_f.abs_diff_eq(direct_f, 1.0e-6));
    }

    #[test]
    fn light_tree_precompute_keeps_dielectric_transmission_as_btdf_lobe() {
        let mat = MtlxMaterial::new(closure_compiled(
            0,
            vec![
                ClosureNode::Surface {
                    bsdf: 1,
                    edf: 2,
                    opacity: ParamRef::Float(1.0),
                    thin_walled: false,
                },
                ClosureNode::Dielectric {
                    weight: ParamRef::Float(1.0),
                    tint: ParamRef::Color3(Vec3::ONE),
                    ior: ParamRef::Float(1.5),
                    roughness: ParamRef::Vector2(Vec2::splat(0.2)),
                    scatter_mode: ScatterMode::Transmission,
                    thinfilm_thickness: ParamRef::Float(0.0),
                    thinfilm_ior: ParamRef::Float(1.0),
                    normal: None,
                    tangent: None,
                },
                ClosureNode::Zero,
            ],
            false,
        ));
        let mut sv = synthetic_sv(true);
        let mut scratch = MtlxScratch::default();

        mat.precompute_shading(&mut sv, &mut scratch);
        let pre = mat
            .light_tree_precompute(&sv, &scratch)
            .expect("dielectric transmission should produce light tree precompute");

        assert!(pre.diffuse.is_none());
        assert!(pre.glossy.is_none());
        assert!(pre.btdf.is_some());
    }

    #[test]
    fn thin_walled_transmission_eval_and_pdf_are_delta_zero() {
        let mat = MtlxMaterial::new(closure_compiled(
            0,
            vec![
                ClosureNode::Surface {
                    bsdf: 1,
                    edf: 2,
                    opacity: ParamRef::Float(1.0),
                    thin_walled: true,
                },
                ClosureNode::Translucent {
                    weight: ParamRef::Float(1.0),
                    color: ParamRef::Color3(Vec3::ONE),
                    normal: None,
                },
                ClosureNode::Zero,
            ],
            true,
        ));
        let mut sv = synthetic_sv(true);
        let mut scratch = MtlxScratch::default();
        let mut rng = crate::sampler::AuxRng::from_seed(0);

        mat.precompute_shading(&mut sv, &mut scratch);
        assert_eq!(
            mat.eval(
                &sv,
                &scratch,
                -Vec3::Z,
                &mut crate::sampler::AuxRng::default()
            ),
            Vec3::ZERO
        );
        assert_eq!(mat.pdf(&sv, &scratch, -Vec3::Z), 0.0);

        let sample = mat
            .sample(
                &sv,
                &scratch,
                &crate::sampler::MaterialSampleRandoms::from_aux_rng(&mut rng),
                &mut crate::sampler::AuxRng::default(),
            )
            .expect("thin-walled transmission should sample");
        assert!(sample.flags.contains(crate::bsdf::BsdfFlags::DELTA));
        assert!(sample.flags.contains(crate::bsdf::BsdfFlags::TRANSMISSION));
        assert_eq!(sample.pdf, 1.0);
    }

    #[test]
    fn thin_walled_standard_surface_is_recognized() {
        let mat = load_shader_ops_thin_walled();
        assert!(
            mtlx::runtime::is_thin_walled(&mat.compiled),
            "shader_ops material_checker_opacity should be thin_walled"
        );
    }

    #[test]
    fn thin_walled_back_face_specular_selection_prob_matches_front() {
        use crate::bsdf::BsdfFlags;

        let mat = load_shader_ops_thin_walled();
        let material = Material::Mtlx(mat);

        let mut rng = crate::sampler::AuxRng::from_seed(0);
        let mut scratch = MtlxScratch::default();
        let n = 20_000u32;

        let glossy_fraction = |sv: &mut ShadingVertex,
                               rng: &mut crate::sampler::AuxRng,
                               scratch: &mut MtlxScratch|
         -> f32 {
            material.precompute_shading(sv, scratch);
            let mut glossy = 0u32;
            let mut total = 0u32;
            for _ in 0..n {
                let randoms = crate::sampler::MaterialSampleRandoms::from_aux_rng(rng);
                if let Some(s) = material.sample(
                    sv,
                    scratch,
                    &randoms,
                    &mut crate::sampler::AuxRng::default(),
                ) {
                    total += 1;
                    if s.flags.contains(BsdfFlags::GLOSSY)
                        && !s.flags.contains(BsdfFlags::TRANSMISSION)
                    {
                        glossy += 1;
                    }
                }
            }
            if total == 0 {
                0.0
            } else {
                glossy as f32 / total as f32
            }
        };

        let mut front = synthetic_sv(true);
        let mut back = synthetic_sv(false);
        let p_front = glossy_fraction(&mut front, &mut rng, &mut scratch);
        let p_back = glossy_fraction(&mut back, &mut rng, &mut scratch);

        assert!(
            p_front < 0.25,
            "front-face glossy selection probability should track Schlick F (~0.07), got {}",
            p_front
        );
        assert!(
            p_back < 0.25,
            "back-face glossy selection probability must NOT inflate from eta swap (pre-fix was ~1.0), got {}",
            p_back
        );
        let diff = (p_front - p_back).abs();
        assert!(
            diff < 0.05,
            "thin_walled requires symmetric front/back face Fresnel; observed front={}, back={}",
            p_front,
            p_back
        );
    }
}
