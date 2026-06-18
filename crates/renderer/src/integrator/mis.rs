use glam::{Vec2, Vec3};

use crate::{
    bsdf::BsdfFlags,
    light::{
        LightSampleContext, infinite_light_le, pdf_for_environment_hit, pdf_for_triangle_hit,
        sample_light_mis_compensated_lazy,
    },
    light_tree,
    material::{Material, MtlxScratch, ShadingVertex},
    math::ray::Ray,
    math::{balance_heuristic, russian_roulette_probability},
    sampler::{AuxRng, PathSampler},
    scene::Scene,
};

use super::{spawn_scattered_ray, unoccluded};

pub fn trace_radiance(
    scene: &Scene,
    initial_ray: Ray,
    sampler: &PathSampler,
    max_depth: u32,
    mtlx_scratch: &mut MtlxScratch,
) -> Vec3 {
    let mut radiance = Vec3::ZERO;
    let mut throughput = Vec3::ONE;

    let mut initial_aux_rng = AuxRng::from_seed(sampler.initial_aux_rng_seed());
    let initial_hit = scene
        .closest_hit(&initial_ray, &mut initial_aux_rng, mtlx_scratch)
        .expect("scene.build_qbvh() must be called before traversal");

    let Some(initial_hit) = initial_hit else {
        return infinite_light_le(scene, initial_ray.direction);
    };

    let mut ray = initial_ray;
    let mut hit_t = initial_hit.t;
    let mut vtx = scene.shading_vertex(initial_hit, &ray);
    let mut material = scene.instance_material(initial_hit.triangle.instance_index);
    let mut wavelength_lock: Option<f32> = None;
    material.precompute_shading(&mut vtx, mtlx_scratch);

    if let Some(le) = material.le(&vtx, mtlx_scratch) {
        radiance += le;
    }

    let rr_start_depth = 4;

    for depth in 0..max_depth {
        let randoms = sampler.path_vertex_randoms(depth);
        let mut aux_rng = AuxRng::from_seed(randoms.aux_rng_seed);
        vtx.path_throughput = throughput;
        vtx.wavelength_lock = wavelength_lock;
        if should_sample_direct_light(material) {
            let light_randoms = randoms.light;
            radiance += throughput
                * direct_light_mis_contribution(
                    scene,
                    material,
                    &vtx,
                    light_randoms.u_category,
                    light_randoms.u_tree,
                    light_randoms.u_light_aux,
                    light_randoms.u_surface,
                    &mut aux_rng,
                    mtlx_scratch,
                );
        }

        let Some(sample) = material.sample(&vtx, mtlx_scratch, &randoms.material, &mut aux_rng)
        else {
            break;
        };
        let is_delta_sample = sample.flags.contains(BsdfFlags::DELTA);

        throughput *= sample.weight;
        if let Some(lambda) = sample.wavelength_lock {
            wavelength_lock = Some(lambda);
        }

        if depth + 1 >= rr_start_depth {
            let survive_probability = russian_roulette_probability(throughput);
            if randoms.u_rr > survive_probability {
                break;
            }
            throughput /= survive_probability;
        }

        let next_ray = spawn_scattered_ray(&ray, hit_t, &vtx, &sample);
        let next_hit = scene
            .closest_hit(&next_ray, &mut aux_rng, mtlx_scratch)
            .expect("scene.build_qbvh() must be called before traversal");

        match next_hit {
            Some(next_hit) => {
                let mut next_vtx = scene.shading_vertex(next_hit, &next_ray);
                next_vtx.path_throughput = throughput;
                next_vtx.wavelength_lock = wavelength_lock;
                let next_material = scene.instance_material(next_hit.triangle.instance_index);
                radiance += emitted_radiance_from_bsdf_sample_area(
                    scene,
                    throughput,
                    sample.pdf,
                    is_delta_sample,
                    material,
                    &vtx,
                    &mut next_vtx,
                    next_material,
                    mtlx_scratch,
                );

                next_material.precompute_shading(&mut next_vtx, mtlx_scratch);
                vtx = next_vtx;
                material = next_material;
                ray = next_ray;
                hit_t = next_hit.t;
            }
            None => {
                radiance += emitted_radiance_from_bsdf_sample_infinite(
                    scene,
                    throughput,
                    sample.pdf,
                    is_delta_sample,
                    next_ray.direction,
                );
                break;
            }
        }
    }

    radiance
}

pub(super) fn should_sample_direct_light(material: &Material) -> bool {
    !material.is_pure_emitter()
}

pub(super) fn direct_light_mis_contribution(
    scene: &Scene,
    material: &Material,
    vtx: &ShadingVertex,
    u_root: f32,
    u_tree: f32,
    u_aux: f32,
    us: Vec2,
    aux_rng: &mut AuxRng,
    mtlx_scratch: &mut MtlxScratch,
) -> Vec3 {
    let ctx = LightSampleContext::from_vertex(vtx);

    let Some(sampled) = sample_light_mis_compensated_lazy(
        scene,
        &ctx,
        vtx,
        material,
        u_root,
        u_tree,
        u_aux,
        us,
        mtlx_scratch,
    ) else {
        return Vec3::ZERO;
    };
    let li = sampled.sample;

    if li.pdf <= 0.0 || sampled.selection_pmf <= 0.0 {
        return Vec3::ZERO;
    }

    let is_delta_light = li.light_type.is_delta();
    let (f, bsdf_pdf) = if is_delta_light {
        (material.eval(vtx, mtlx_scratch, li.wi, aux_rng), 0.0)
    } else {
        material.eval_pdf(vtx, mtlx_scratch, li.wi, aux_rng)
    };
    if f.length_squared() == 0.0 {
        return Vec3::ZERO;
    }

    let cos_surface = vtx.ns.dot(li.wi).abs();
    if cos_surface <= 0.0 {
        return Vec3::ZERO;
    }

    if !unoccluded(scene, vtx, &li, aux_rng, mtlx_scratch) {
        return Vec3::ZERO;
    }

    let pdf_total = sampled.selection_pmf * li.pdf;
    if pdf_total <= 0.0 {
        return Vec3::ZERO;
    }

    if is_delta_light {
        return li.radiance * f * (cos_surface / pdf_total);
    }

    let mis_weight = balance_heuristic(pdf_total, bsdf_pdf);
    li.radiance * f * (cos_surface * mis_weight / pdf_total)
}

fn emitted_radiance_from_bsdf_sample_area(
    scene: &Scene,
    throughput: Vec3,
    bsdf_pdf: f32,
    is_delta_sample: bool,
    shading_material: &Material,
    vtx: &ShadingVertex,
    lvtx: &mut ShadingVertex,
    light_material: &Material,
    mtlx_scratch: &mut MtlxScratch,
) -> Vec3 {
    light_material.precompute_shading(lvtx, mtlx_scratch);
    let Some(le) = light_material.le(lvtx, mtlx_scratch) else {
        return Vec3::ZERO;
    };

    if is_delta_sample {
        return throughput * le;
    }

    let tree_query = light_tree::build_query(vtx, shading_material, mtlx_scratch);
    let light_pdf = pdf_for_triangle_hit(scene, tree_query.as_ref(), vtx, lvtx);
    let mis_weight = balance_heuristic(bsdf_pdf, light_pdf);
    throughput * le * mis_weight
}

fn emitted_radiance_from_bsdf_sample_infinite(
    scene: &Scene,
    throughput: Vec3,
    bsdf_pdf: f32,
    is_delta_sample: bool,
    direction: Vec3,
) -> Vec3 {
    let le = infinite_light_le(scene, direction);
    if le == Vec3::ZERO {
        return Vec3::ZERO;
    }

    if is_delta_sample {
        return throughput * le;
    }

    let light_pdf = pdf_for_environment_hit(scene, direction, true);
    let mis_weight = balance_heuristic(bsdf_pdf, light_pdf);
    throughput * le * mis_weight
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use glam::{Vec2, Vec3};

    use crate::{
        light::EnvironmentLight,
        material::{
            EmissiveMaterial, Material, MirrorMaterial, MtlxScratch, NormalizedLambertMaterial,
        },
        math::ray::Ray,
        scene::Scene,
    };

    use super::super::test_helpers::{
        floor_with_area_light_scene, floor_with_directional_light_scene,
        floor_with_point_light_scene, floor_with_spot_light_scene, light_below_scene,
        mirror_to_light_scene, sample_area_light_vtx, sample_floor_vtx,
        standalone_shading_vertex_for_transmission, transmission_mtlx_material, triangle_ref,
        unit_mesh,
    };
    use super::{
        direct_light_mis_contribution, emitted_radiance_from_bsdf_sample_area,
        emitted_radiance_from_bsdf_sample_infinite, should_sample_direct_light, trace_radiance,
    };

    #[test]
    fn direct_light_sampling_skips_only_pure_emitters() {
        let mirror = Material::Mirror(MirrorMaterial::new(Vec3::ONE));
        let lambert = Material::NormalizedLambert(NormalizedLambertMaterial::new(Vec3::ONE));
        let light = Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 2.0));

        assert!(should_sample_direct_light(&mirror));
        assert!(should_sample_direct_light(&lambert));
        assert!(!should_sample_direct_light(&light));
    }

    #[test]
    fn direct_light_sampling_runs_for_mtlx_with_conservative_may_emit() {
        use crate::material::MtlxMaterial;
        use crate::material::mtlx::{ClosureNode, CompiledMaterial, ParamRef};
        use std::sync::Arc;

        let closure_nodes = vec![
            ClosureNode::Zero,
            ClosureNode::OrenNayarDiffuse {
                weight: ParamRef::Float(1.0),
                color: ParamRef::Color3(Vec3::splat(0.5)),
                roughness: ParamRef::Float(0.0),
                energy_compensation: false,
                normal: None,
            },
            ClosureNode::UniformEdf {
                color: ParamRef::Color3(Vec3::splat(0.5)),
            },
            ClosureNode::Surface {
                bsdf: 1,
                edf: 2,
                opacity: ParamRef::Float(1.0),
                thin_walled: false,
            },
        ];
        let compiled = CompiledMaterial {
            instructions: Vec::new(),
            operand_pool: Vec::new(),
            value_pool: Vec::new(),
            color_processors: Vec::new(),
            opacity_instructions: Vec::new(),
            opacity_operand_pool: Vec::new(),
            opacity_closure_nodes: Vec::new(),
            opacity_num_registers: 0,
            num_registers: 0,

            closure_nodes,
            root: 3,
            passthrough: false,
            max_emission: 0.5,
            may_emit: true,
            has_opacity_test: false,
            thin_walled: false,
            sheen_lut: None,
            mtlx_dielectric_lut: None,
            mtlx_generalized_schlick_lut: None,
        };
        let mtlx = Material::Mtlx(MtlxMaterial::new(Arc::new(compiled)));

        assert!(mtlx.may_emit());
        assert!(!mtlx.is_pure_emitter());
        assert!(should_sample_direct_light(&mtlx));
    }

    #[test]
    fn direct_light_mis_keeps_transmission_below_surface() {
        let scene = light_below_scene();
        let material = transmission_mtlx_material();
        let mut vtx = standalone_shading_vertex_for_transmission();
        let mut scratch = MtlxScratch::default();
        material.precompute_shading(&mut vtx, &mut scratch);

        let l = direct_light_mis_contribution(
            &scene,
            &material,
            &vtx,
            0.5,
            0.5,
            0.5,
            Vec2::new(0.25, 0.5),
            &mut crate::sampler::AuxRng::default(),
            &mut scratch,
        );

        assert!(
            l.length_squared() > 0.0,
            "transmission NEE/MIS must use abs cosine for lights below the surface"
        );
    }

    #[test]
    fn trace_radiance_counts_light_after_delta_bounce() {
        let (scene, ray, expected) = mirror_to_light_scene();
        let sampler =
            crate::sampler::PathSampler::new(glam::UVec2::ZERO, 0, 1, glam::UVec2::new(1, 1));

        let radiance = trace_radiance(&scene, ray, &sampler, 2, &mut MtlxScratch::default());

        assert!(radiance.abs_diff_eq(expected, 1.0e-3));
    }

    #[test]
    fn direct_radiance_applies_balance_heuristic_for_light_sampling() {
        let scene = floor_with_area_light_scene(0.8, 10.0);
        let vtx = sample_floor_vtx(&scene, Vec3::NEG_Z);
        let material = scene.instance_material(triangle_ref(0).instance_index);
        // Single area light -> light sampler pmf = 1.
        let radiance = direct_light_mis_contribution(
            &scene,
            material,
            &vtx,
            0.0,
            0.0,
            0.5,
            Vec2::new(0.25, 0.5),
            &mut crate::sampler::AuxRng::default(),
            &mut MtlxScratch::default(),
        );
        let expected = (4.0 / PI) * (2.0 / (2.0 + 1.0 / PI));

        assert!(radiance.abs_diff_eq(Vec3::splat(expected), 1.0e-3));
    }

    #[test]
    fn emitted_radiance_applies_balance_heuristic_for_bsdf_sampling() {
        let scene = floor_with_area_light_scene(0.8, 10.0);
        let vtx = sample_floor_vtx(&scene, Vec3::NEG_Z);
        let mut lvtx = sample_area_light_vtx(&scene);
        let material = scene.instance_material(triangle_ref(0).instance_index);
        let light_material = scene.instance_material(triangle_ref(1).instance_index);
        let scratch = MtlxScratch::default();
        let bsdf_pdf = material.pdf(&vtx, &scratch, Vec3::Z);
        let throughput = Vec3::splat(0.8);
        let radiance = emitted_radiance_from_bsdf_sample_area(
            &scene,
            throughput,
            bsdf_pdf,
            false,
            material,
            &vtx,
            &mut lvtx,
            light_material,
            &mut MtlxScratch::default(),
        );
        let expected = 8.0 * ((1.0 / PI) / (2.0 + 1.0 / PI));

        assert!(radiance.abs_diff_eq(Vec3::splat(expected), 1.0e-3));
    }

    #[test]
    fn emitted_radiance_from_delta_bsdf_sample_uses_full_weight() {
        let scene = floor_with_area_light_scene(0.8, 10.0);
        let vtx = sample_floor_vtx(&scene, Vec3::NEG_Z);
        let mut lvtx = sample_area_light_vtx(&scene);
        let shading_material = scene.instance_material(triangle_ref(0).instance_index);
        let light_material = scene.instance_material(triangle_ref(1).instance_index);
        let throughput = Vec3::splat(0.8);

        let radiance = emitted_radiance_from_bsdf_sample_area(
            &scene,
            throughput,
            1.0,
            true,
            shading_material,
            &vtx,
            &mut lvtx,
            light_material,
            &mut MtlxScratch::default(),
        );

        assert!(radiance.abs_diff_eq(Vec3::splat(8.0), 1.0e-3));
    }

    #[test]
    fn emitted_radiance_from_infinite_light_applies_mis_for_bsdf_sampling() {
        let mut scene = Scene::new();
        let env_radiance = Vec3::splat(2.0);
        let pixels = vec![env_radiance; 32 * 16];
        scene.set_environment_light(EnvironmentLight::from_pixels(32, 16, pixels, 1.0, 0.0));
        scene.build_qbvh();
        scene.build_light_tree();

        let throughput = Vec3::ONE;
        let direction = Vec3::Y;
        // bsdf_pdf = 0 (worst case): MIS weight degenerates to 0 because both
        // pdfs being zero means balance_heuristic returns 0 safely.
        let zero =
            emitted_radiance_from_bsdf_sample_infinite(&scene, throughput, 0.0, false, direction);
        assert_eq!(zero, Vec3::ZERO);

        // Delta bsdf sample: MIS is not applied; full env contribution is taken.
        let delta =
            emitted_radiance_from_bsdf_sample_infinite(&scene, throughput, 1.0, true, direction);
        assert!(delta.abs_diff_eq(env_radiance, 1.0e-5));

        // Non-delta bsdf sample with high bsdf_pdf dominates MIS weight.
        let high_bsdf = emitted_radiance_from_bsdf_sample_infinite(
            &scene, throughput, 1000.0, false, direction,
        );
        assert!(high_bsdf.x > env_radiance.x * 0.9);
    }

    #[test]
    fn trace_radiance_direct_hit_on_environment_reads_primary_radiance() {
        let mut scene = Scene::new();
        let env_radiance = Vec3::new(0.1, 0.4, 0.9);
        let pixels = vec![env_radiance; 16 * 8];
        scene.set_environment_light(EnvironmentLight::from_pixels(16, 8, pixels, 1.0, 0.0));
        scene.build_qbvh();
        scene.build_light_tree();

        let ray = Ray::new(Vec3::ZERO, Vec3::Y);
        let sampler =
            crate::sampler::PathSampler::new(glam::UVec2::ZERO, 0, 1, glam::UVec2::new(1, 1));
        let radiance = trace_radiance(&scene, ray, &sampler, 4, &mut MtlxScratch::default());

        assert!(radiance.abs_diff_eq(env_radiance, 1.0e-5));
    }

    #[test]
    fn direct_radiance_uses_full_weight_for_point_light() {
        let scene = floor_with_point_light_scene(0.8, 16.0 * PI);
        let vtx = sample_floor_vtx(&scene, Vec3::NEG_Z);
        let material = scene.instance_material(triangle_ref(0).instance_index);
        let radiance = direct_light_mis_contribution(
            &scene,
            material,
            &vtx,
            0.0,
            0.0,
            0.0,
            Vec2::ZERO,
            &mut crate::sampler::AuxRng::default(),
            &mut MtlxScratch::default(),
        );
        // Li=1 * lambert eval 0.8/PI * cos=1 / pmf=1 = 0.8/PI.
        let expected = 0.8 / PI;
        assert!(radiance.abs_diff_eq(Vec3::splat(expected), 1.0e-3));
    }

    #[test]
    fn direct_radiance_returns_zero_when_point_light_is_occluded() {
        let mut scene = floor_with_point_light_scene(0.8, 16.0 * PI);
        let blocker_mesh = scene.add_mesh(unit_mesh(1.0));
        let blocker_material = scene.add_material(Material::NormalizedLambert(
            NormalizedLambertMaterial::new(Vec3::splat(0.5)),
        ));
        scene.add_instance(blocker_mesh, blocker_material, glam::Mat4::IDENTITY);
        scene.build_qbvh();
        scene.build_light_tree();

        let vtx = sample_floor_vtx(&scene, Vec3::NEG_Z);
        let material = scene.instance_material(triangle_ref(0).instance_index);
        let radiance = direct_light_mis_contribution(
            &scene,
            material,
            &vtx,
            0.0,
            0.0,
            0.0,
            Vec2::ZERO,
            &mut crate::sampler::AuxRng::default(),
            &mut MtlxScratch::default(),
        );
        assert_eq!(radiance, Vec3::ZERO);
    }

    #[test]
    fn direct_radiance_uses_full_weight_for_directional_light() {
        let scene = floor_with_directional_light_scene(0.8, 2.0);
        let vtx = sample_floor_vtx(&scene, Vec3::NEG_Z);
        let material = scene.instance_material(triangle_ref(0).instance_index);
        let radiance = direct_light_mis_contribution(
            &scene,
            material,
            &vtx,
            0.0,
            0.0,
            0.0,
            Vec2::ZERO,
            &mut crate::sampler::AuxRng::default(),
            &mut MtlxScratch::default(),
        );
        // Li = color * irradiance = 2; lambert 0.8/PI * cos=1 / pmf=1.
        let expected = 2.0 * 0.8 / PI;
        assert!(radiance.abs_diff_eq(Vec3::splat(expected), 1.0e-3));
    }

    #[test]
    fn direct_radiance_uses_full_weight_for_spot_light_with_falloff() {
        let scene = floor_with_spot_light_scene(
            0.8,
            16.0 * PI,
            30f32.to_radians(),
            20f32.to_radians(),
            Vec3::NEG_Z,
        );
        let vtx = sample_floor_vtx(&scene, Vec3::NEG_Z);
        let material = scene.instance_material(triangle_ref(0).instance_index);
        let radiance = direct_light_mis_contribution(
            &scene,
            material,
            &vtx,
            0.0,
            0.0,
            0.0,
            Vec2::ZERO,
            &mut crate::sampler::AuxRng::default(),
            &mut MtlxScratch::default(),
        );
        let expected = 0.8 / PI;
        assert!(radiance.abs_diff_eq(Vec3::splat(expected), 1.0e-3));
    }

    #[test]
    fn direct_radiance_returns_zero_when_shading_point_is_outside_spot_cone() {
        let scene = floor_with_spot_light_scene(
            0.8,
            16.0 * PI,
            10f32.to_radians(),
            5f32.to_radians(),
            Vec3::X,
        );
        let vtx = sample_floor_vtx(&scene, Vec3::NEG_Z);
        let material = scene.instance_material(triangle_ref(0).instance_index);
        let radiance = direct_light_mis_contribution(
            &scene,
            material,
            &vtx,
            0.0,
            0.0,
            0.0,
            Vec2::ZERO,
            &mut crate::sampler::AuxRng::default(),
            &mut MtlxScratch::default(),
        );
        assert_eq!(radiance, Vec3::ZERO);
    }

    #[test]
    fn trace_radiance_mixes_area_light_and_environment_without_panics() {
        let mut scene = floor_with_area_light_scene(0.6, 2.0);

        let env_radiance = Vec3::splat(0.2);
        let pixels = vec![env_radiance; 16 * 8];
        scene.set_environment_light(EnvironmentLight::from_pixels(16, 8, pixels, 1.0, 0.0));
        scene.build_qbvh();
        scene.build_light_tree();

        let mut scratch = MtlxScratch::default();
        let mut accumulated = Vec3::ZERO;
        for sample_index in 0..32 {
            let ray = Ray::new(Vec3::new(0.25, 0.25, -2.0), Vec3::Z);
            let sampler = crate::sampler::PathSampler::new(
                glam::UVec2::ZERO,
                sample_index,
                32,
                glam::UVec2::new(1, 1),
            );
            accumulated += trace_radiance(&scene, ray, &sampler, 4, &mut scratch);
        }
        let mean = accumulated / 32.0;

        // Sanity: at least some radiance arrives (area + env both present).
        assert!(mean.x > 0.0);
        assert!(mean.y > 0.0);
        assert!(mean.z > 0.0);
    }
}
