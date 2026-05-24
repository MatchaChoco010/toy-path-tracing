use glam::{Vec2, Vec3};

use crate::{
    bsdf::BsdfFlags,
    light::{LightSampleContext, infinite_light_le, sample_light},
    light_tree,
    material::{Material, MtlxScratch, ShadingVertex},
    math::ray::Ray,
    math::russian_roulette_probability,
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
    let mut ray = initial_ray;
    let mut count_emission_at_hit = true;
    let mut wavelength_lock: Option<f32> = None;
    let rr_start_depth = 4;

    for depth in 0..max_depth {
        let randoms = sampler.path_vertex_randoms(depth);
        let mut aux_rng = AuxRng::from_seed(randoms.aux_rng_seed);
        let hit = scene
            .closest_hit(&ray, &mut aux_rng, mtlx_scratch)
            .expect("scene.build_qbvh() must be called before traversal");

        let Some(hit) = hit else {
            if count_emission_at_hit {
                radiance += throughput * infinite_light_le(scene, ray.direction);
            }
            break;
        };

        let mut vtx = scene.shading_vertex(hit, &ray);
        vtx.path_throughput = throughput;
        vtx.wavelength_lock = wavelength_lock;
        let material = scene.instance_material(hit.triangle.instance_index);
        material.precompute_shading(&mut vtx, mtlx_scratch);
        if count_emission_at_hit && let Some(le) = material.le(&vtx, mtlx_scratch) {
            radiance += throughput * le;
        }

        if should_sample_direct_light(material) {
            let light_randoms = randoms.light;
            radiance += throughput
                * direct_light_nee_contribution(
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

        ray = spawn_scattered_ray(&ray, hit.t, &vtx, &sample);
        count_emission_at_hit = is_delta_sample;
    }

    radiance
}

pub(super) fn should_sample_direct_light(material: &Material) -> bool {
    !material.is_pure_emitter()
}

pub(super) fn direct_light_nee_contribution(
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
    let tree_query = light_tree::build_query(vtx, material, mtlx_scratch);

    let Some(sampled) = sample_light(
        scene,
        &ctx,
        tree_query.as_ref(),
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

    let f = material.eval(vtx, mtlx_scratch, li.wi, aux_rng);
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

    li.radiance * f * (cos_surface / pdf_total)
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use glam::{Mat4, Vec2, Vec3};

    use super::super::test_helpers::mirror_to_light_scene;
    use super::{direct_light_nee_contribution, should_sample_direct_light, trace_radiance};
    use crate::{
        bsdf::mtlx::ScatterMode,
        light::{DirectionalLight, EnvironmentLight, PointLight, SpotLight},
        material::mtlx::{ClosureNode, CompiledMaterial, ParamRef},
        material::{
            EmissiveMaterial, Material, MirrorMaterial, MtlxMaterial, MtlxScratch,
            NormalizedLambertMaterial, ShadingVertex,
        },
        math::OrthonormalBasis,
        math::ray::Ray,
        scene::{InstanceIndex, Scene, TriangleRef},
        scene::{Mesh, Vertex},
    };
    use std::sync::Arc;

    fn unit_mesh(z: f32) -> Mesh {
        Mesh::new(
            vec![
                Vertex {
                    position: Vec3::new(0.0, 0.0, z),
                    normal: Vec3::Z,
                    uv: Vec2::ZERO,
                },
                Vertex {
                    position: Vec3::new(1.0, 0.0, z),
                    normal: Vec3::Z,
                    uv: Vec2::X,
                },
                Vertex {
                    position: Vec3::new(0.0, 1.0, z),
                    normal: Vec3::Z,
                    uv: Vec2::Y,
                },
            ],
            vec![0, 1, 2],
        )
    }

    fn triangle_ref(instance_index: usize) -> TriangleRef {
        TriangleRef {
            instance_index: InstanceIndex(instance_index),
            triangle_index: 0,
        }
    }

    fn transmission_mtlx_material() -> Material {
        Material::Mtlx(MtlxMaterial::new(Arc::new(CompiledMaterial {
            instructions: Vec::new(),
            operand_pool: Vec::new(),
            value_pool: Vec::new(),
            color_processors: Vec::new(),
            opacity_instructions: Vec::new(),
            opacity_operand_pool: Vec::new(),
            opacity_closure_nodes: Vec::new(),
            opacity_num_registers: 0,
            num_registers: 0,
            closure_nodes: vec![
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
                    retroreflective: false,
                    scatter_mode: ScatterMode::Transmission,
                    thinfilm_thickness: ParamRef::Float(0.0),
                    thinfilm_ior: ParamRef::Float(1.0),
                    normal: None,
                    tangent: None,
                },
                ClosureNode::Zero,
            ],
            root: 0,
            passthrough: false,
            max_emission: 0.0,
            may_emit: false,
            has_opacity_test: false,
            thin_walled: false,
            sheen_lut: None,
            mtlx_dielectric_lut: None,
            mtlx_generalized_schlick_lut: None,
        })))
    }

    fn standalone_shading_vertex_for_transmission() -> ShadingVertex {
        ShadingVertex {
            triangle: triangle_ref(0),
            p: Vec3::new(0.25, 0.25, 0.0),
            uv: Vec2::ZERO,
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
            object_to_world: Mat4::IDENTITY,
            world_to_object: Mat4::IDENTITY,
            object_normal_to_world: glam::Mat3::IDENTITY,
            mtlx_regs: None,
            mtlx_dalbedo: None,
            mtlx_precomputed_for: None,
        }
    }

    fn light_below_scene() -> Scene {
        let mut scene = Scene::new();
        let light_mesh = scene.add_mesh(unit_mesh(-1.0));
        let light_material =
            scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 4.0)));
        scene.add_instance(light_mesh, light_material, Mat4::IDENTITY);
        scene.build_qbvh();
        scene.build_light_tree();
        scene
    }

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
    fn direct_light_nee_keeps_transmission_below_surface() {
        let scene = light_below_scene();
        let material = transmission_mtlx_material();
        let mut vtx = standalone_shading_vertex_for_transmission();
        let mut scratch = MtlxScratch::default();
        material.precompute_shading(&mut vtx, &mut scratch);

        let l = direct_light_nee_contribution(
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
            "transmission NEE must use abs cosine for lights below the surface"
        );
    }

    #[test]
    fn trace_radiance_counts_light_after_delta_bounce() {
        let (scene, ray, expected) = mirror_to_light_scene();
        let sampler =
            crate::sampler::PathSampler::new(glam::UVec2::ZERO, 0, 1, glam::UVec2::new(1, 1));

        let mut scratch = MtlxScratch::default();
        let radiance = trace_radiance(&scene, ray, &sampler, 2, &mut scratch);

        assert!(radiance.abs_diff_eq(expected, 1.0e-3));
    }

    #[test]
    fn direct_radiance_matches_area_light_estimator_for_unoccluded_light() {
        let mut scene = Scene::new();
        let floor_mesh = scene.add_mesh(unit_mesh(0.0));
        let light_mesh = scene.add_mesh(unit_mesh(1.0));
        let floor_material = scene.add_material(Material::NormalizedLambert(
            NormalizedLambertMaterial::new(Vec3::splat(0.8)),
        ));
        let light_material =
            scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 10.0)));
        scene.add_instance(floor_mesh, floor_material, Mat4::IDENTITY);
        scene.add_instance(light_mesh, light_material, Mat4::IDENTITY);
        scene.build_qbvh();
        scene.build_light_tree();

        let vtx = scene.shading_vertex_from_triangle_sample(
            triangle_ref(0),
            Vec3::new(0.5, 0.25, 0.25),
            Vec3::NEG_Z,
        );
        let material = scene.instance_material(InstanceIndex(0));
        // With only a single area light the sampler always selects it (pmf=1),
        // so the estimator reduces to the classic direct-light formula.
        let radiance = direct_light_nee_contribution(
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

        assert!(radiance.abs_diff_eq(Vec3::splat(4.0 / PI), 1.0e-3));
    }

    #[test]
    fn direct_radiance_returns_zero_when_light_is_occluded() {
        let mut scene = Scene::new();
        let floor_mesh = scene.add_mesh(unit_mesh(0.0));
        let blocker_mesh = scene.add_mesh(unit_mesh(0.5));
        let light_mesh = scene.add_mesh(unit_mesh(1.0));
        let floor_material = scene.add_material(Material::NormalizedLambert(
            NormalizedLambertMaterial::new(Vec3::splat(0.8)),
        ));
        let blocker_material = scene.add_material(Material::NormalizedLambert(
            NormalizedLambertMaterial::new(Vec3::splat(0.5)),
        ));
        let light_material =
            scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 10.0)));
        scene.add_instance(floor_mesh, floor_material, Mat4::IDENTITY);
        scene.add_instance(blocker_mesh, blocker_material, Mat4::IDENTITY);
        scene.add_instance(light_mesh, light_material, Mat4::IDENTITY);
        scene.build_qbvh();
        scene.build_light_tree();

        let vtx = scene.shading_vertex_from_triangle_sample(
            triangle_ref(0),
            Vec3::new(0.5, 0.25, 0.25),
            Vec3::NEG_Z,
        );
        let material = scene.instance_material(InstanceIndex(0));
        let radiance = direct_light_nee_contribution(
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

        assert_eq!(radiance, Vec3::ZERO);
    }

    #[test]
    fn direct_radiance_includes_environment_light_contribution() {
        let mut scene = Scene::new();
        let floor_mesh = scene.add_mesh(unit_mesh(0.0));
        let floor_material = scene.add_material(Material::NormalizedLambert(
            NormalizedLambertMaterial::new(Vec3::splat(0.8)),
        ));
        scene.add_instance(floor_mesh, floor_material, Mat4::IDENTITY);
        let env_radiance = Vec3::splat(2.0);
        let pixels = vec![env_radiance; 16 * 8];
        scene.set_environment_light(EnvironmentLight::from_pixels(16, 8, pixels, 1.0, 0.0));
        scene.build_qbvh();
        scene.build_light_tree();

        let vtx = scene.shading_vertex_from_triangle_sample(
            triangle_ref(0),
            Vec3::new(0.5, 0.25, 0.25),
            Vec3::NEG_Z,
        );
        let material = scene.instance_material(InstanceIndex(0));
        // Infinite light is the only light, so sampler pmf is 1. Pick a u that
        // selects a direction roughly aligned with the surface normal so the
        // cos_theta factor is positive (env uses +Y up, surface normal is +Z
        // via spherical sampling the direction with u=0 and v=0.5 maps to +Z).
        let radiance = direct_light_nee_contribution(
            &scene,
            material,
            &vtx,
            0.5,
            0.0,
            0.0,
            Vec2::new(0.0, 0.5),
            &mut crate::sampler::AuxRng::default(),
            &mut MtlxScratch::default(),
        );

        assert!(radiance.x > 0.0);
        assert!(radiance.y > 0.0);
        assert!(radiance.z > 0.0);
    }

    #[test]
    fn direct_radiance_matches_point_light_estimator() {
        let mut scene = Scene::new();
        let floor_mesh = scene.add_mesh(unit_mesh(0.0));
        let floor_material = scene.add_material(Material::NormalizedLambert(
            NormalizedLambertMaterial::new(Vec3::splat(0.8)),
        ));
        scene.add_instance(floor_mesh, floor_material, Mat4::IDENTITY);
        scene.add_point_light(PointLight::new(
            Vec3::new(0.25, 0.25, 2.0),
            Vec3::ONE,
            16.0 * PI,
        ));
        scene.build_qbvh();
        scene.build_light_tree();

        let vtx = scene.shading_vertex_from_triangle_sample(
            triangle_ref(0),
            Vec3::new(0.5, 0.25, 0.25),
            Vec3::NEG_Z,
        );
        let material = scene.instance_material(InstanceIndex(0));
        let radiance = direct_light_nee_contribution(
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
    fn direct_radiance_matches_directional_light_estimator() {
        let mut scene = Scene::new();
        let floor_mesh = scene.add_mesh(unit_mesh(0.0));
        let floor_material = scene.add_material(Material::NormalizedLambert(
            NormalizedLambertMaterial::new(Vec3::splat(0.8)),
        ));
        scene.add_instance(floor_mesh, floor_material, Mat4::IDENTITY);
        scene.add_directional_light(DirectionalLight::new(
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::ONE,
            2.0,
        ));
        scene.build_qbvh();
        scene.build_light_tree();

        let vtx = scene.shading_vertex_from_triangle_sample(
            triangle_ref(0),
            Vec3::new(0.5, 0.25, 0.25),
            Vec3::NEG_Z,
        );
        let material = scene.instance_material(InstanceIndex(0));
        let radiance = direct_light_nee_contribution(
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
        let expected = 2.0 * 0.8 / PI;
        assert!(radiance.abs_diff_eq(Vec3::splat(expected), 1.0e-3));
    }

    #[test]
    fn direct_radiance_matches_spot_light_estimator_within_cone() {
        let mut scene = Scene::new();
        let floor_mesh = scene.add_mesh(unit_mesh(0.0));
        let floor_material = scene.add_material(Material::NormalizedLambert(
            NormalizedLambertMaterial::new(Vec3::splat(0.8)),
        ));
        scene.add_instance(floor_mesh, floor_material, Mat4::IDENTITY);
        // P=16π so Li at r=2 on axis equals 1.
        scene.add_spot_light(SpotLight::new(
            Vec3::new(0.25, 0.25, 2.0),
            Vec3::NEG_Z,
            Vec3::ONE,
            16.0 * PI,
            (30.0_f32).to_radians(),
            (20.0_f32).to_radians(),
        ));
        scene.build_qbvh();
        scene.build_light_tree();

        let vtx = scene.shading_vertex_from_triangle_sample(
            triangle_ref(0),
            Vec3::new(0.5, 0.25, 0.25),
            Vec3::NEG_Z,
        );
        let material = scene.instance_material(InstanceIndex(0));
        let radiance = direct_light_nee_contribution(
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
    fn direct_radiance_returns_zero_when_directional_light_is_blocked() {
        let mut scene = Scene::new();
        let floor_mesh = scene.add_mesh(unit_mesh(0.0));
        let blocker_mesh = scene.add_mesh(unit_mesh(1.0));
        let floor_material = scene.add_material(Material::NormalizedLambert(
            NormalizedLambertMaterial::new(Vec3::splat(0.8)),
        ));
        let blocker_material = scene.add_material(Material::NormalizedLambert(
            NormalizedLambertMaterial::new(Vec3::splat(0.3)),
        ));
        scene.add_instance(floor_mesh, floor_material, Mat4::IDENTITY);
        scene.add_instance(blocker_mesh, blocker_material, Mat4::IDENTITY);
        scene.add_directional_light(DirectionalLight::new(
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::ONE,
            2.0,
        ));
        scene.build_qbvh();
        scene.build_light_tree();

        let vtx = scene.shading_vertex_from_triangle_sample(
            triangle_ref(0),
            Vec3::new(0.5, 0.25, 0.25),
            Vec3::NEG_Z,
        );
        let material = scene.instance_material(InstanceIndex(0));
        let radiance = direct_light_nee_contribution(
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
    fn trace_radiance_sees_environment_on_direct_escape() {
        let mut scene = Scene::new();
        let env_radiance = Vec3::new(0.5, 0.25, 0.75);
        let pixels = vec![env_radiance; 16 * 8];
        scene.set_environment_light(EnvironmentLight::from_pixels(16, 8, pixels, 1.0, 0.0));
        scene.build_qbvh();
        scene.build_light_tree();

        let ray = Ray::new(Vec3::ZERO, Vec3::Y);
        let sampler =
            crate::sampler::PathSampler::new(glam::UVec2::ZERO, 0, 1, glam::UVec2::new(1, 1));
        let mut scratch = MtlxScratch::default();
        let radiance = trace_radiance(&scene, ray, &sampler, 2, &mut scratch);

        assert!(radiance.abs_diff_eq(env_radiance, 1.0e-5));
    }
}
