use glam::{Vec2, Vec3};
use rand::RngExt;

use crate::{
    bsdf::BsdfFlags,
    light::{LightSampleContext, infinite_light_le, sample_light},
    light_tree,
    material::{Material, ShadingVertex},
    math::russian_roulette_probability,
    ray::Ray,
    scene::Scene,
};

use super::{spawn_scattered_ray, unoccluded};

pub fn trace_radiance(
    scene: &Scene,
    initial_ray: Ray,
    rng: &mut rand::rngs::ThreadRng,
    max_depth: u32,
) -> Vec3 {
    let mut radiance = Vec3::ZERO;
    let mut throughput = Vec3::ONE;
    let mut ray = initial_ray;
    let mut count_emission_at_hit = true;
    let rr_start_depth = 4;

    for depth in 0..max_depth {
        let hit = scene
            .closest_hit(&ray, rng)
            .expect("scene.build_qbvh() must be called before traversal");

        let Some(hit) = hit else {
            if count_emission_at_hit {
                radiance += throughput * infinite_light_le(scene, ray.direction);
            }
            break;
        };

        let vtx = scene.shading_vertex(hit, &ray);
        let material = scene.instance_material(hit.triangle.instance_index);

        if count_emission_at_hit && let Some(le) = material.le(&vtx) {
            radiance += throughput * le;
        }

        let Some(sample) = material.sample(&vtx, rng) else {
            break;
        };
        let is_delta_sample = sample.flags.contains(BsdfFlags::DELTA);

        if should_sample_direct_light(material, sample.flags) {
            let u_root = rng.random::<f32>();
            let u_tree = rng.random::<f32>();
            let u_aux = rng.random::<f32>();
            let us = Vec2::new(rng.random::<f32>(), rng.random::<f32>());
            radiance += throughput
                * direct_light_nee_contribution(
                    scene, material, &vtx, u_root, u_tree, u_aux, us, rng,
                );
        }

        throughput *= sample.weight;

        if depth + 1 >= rr_start_depth {
            let survive_probability = russian_roulette_probability(throughput);
            if rng.random::<f32>() > survive_probability {
                break;
            }
            throughput /= survive_probability;
        }

        ray = spawn_scattered_ray(&ray, hit.t, &vtx, &sample);
        count_emission_at_hit = is_delta_sample;
    }

    radiance
}

pub(super) fn should_sample_direct_light(material: &Material, sample_flags: BsdfFlags) -> bool {
    !material.may_emit() && !sample_flags.contains(BsdfFlags::DELTA)
}

pub(super) fn direct_light_nee_contribution(
    scene: &Scene,
    material: &Material,
    vtx: &ShadingVertex,
    u_root: f32,
    u_tree: f32,
    u_aux: f32,
    us: Vec2,
    rng: &mut rand::rngs::ThreadRng,
) -> Vec3 {
    let ctx = LightSampleContext::from_vertex(vtx);
    let tree_query = light_tree::build_query(vtx, material);

    let Some(sampled) = sample_light(scene, &ctx, tree_query.as_ref(), u_root, u_tree, u_aux, us)
    else {
        return Vec3::ZERO;
    };
    let li = sampled.sample;

    if li.pdf <= 0.0 || sampled.selection_pmf <= 0.0 {
        return Vec3::ZERO;
    }

    let f = material.eval(vtx, li.wi);
    if f.length_squared() == 0.0 {
        return Vec3::ZERO;
    }

    let cos_surface = vtx.ns.dot(li.wi).max(0.0);
    if cos_surface <= 0.0 {
        return Vec3::ZERO;
    }

    if !unoccluded(scene, vtx, &li, rng) {
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
        bsdf::BsdfFlags,
        light::{DirectionalLight, EnvironmentLight, PointLight, SpotLight},
        material::{EmissiveMaterial, Material, MirrorMaterial, NormalizedLambertMaterial},
        mesh::{Mesh, Vertex},
        ray::Ray,
        scene::{InstanceIndex, Scene, TriangleRef},
    };

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

    #[test]
    fn direct_light_sampling_is_skipped_for_delta_samples() {
        let mirror = Material::Mirror(MirrorMaterial::new(Vec3::ONE));
        let lambert = Material::NormalizedLambert(NormalizedLambertMaterial::new(Vec3::ONE));
        let light = Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 2.0));

        assert!(!should_sample_direct_light(&mirror, BsdfFlags::DELTA));
        assert!(should_sample_direct_light(&lambert, BsdfFlags::DIFFUSE));
        assert!(!should_sample_direct_light(&light, BsdfFlags::DIFFUSE));
    }

    #[test]
    fn trace_radiance_counts_light_after_delta_bounce() {
        let (scene, ray, expected) = mirror_to_light_scene();
        let mut rng = rand::rng();

        let radiance = trace_radiance(&scene, ray, &mut rng, 2);

        assert!(radiance.abs_diff_eq(expected, 1.0e-5));
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
            &mut rand::rng(),
        );

        assert!(radiance.abs_diff_eq(Vec3::splat(4.0 / PI), 1.0e-5));
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
            &mut rand::rng(),
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
            &mut rand::rng(),
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
            &mut rand::rng(),
        );
        let expected = 0.8 / PI;
        assert!(radiance.abs_diff_eq(Vec3::splat(expected), 1.0e-5));
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
            &mut rand::rng(),
        );
        let expected = 2.0 * 0.8 / PI;
        assert!(radiance.abs_diff_eq(Vec3::splat(expected), 1.0e-5));
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
            &mut rand::rng(),
        );
        let expected = 0.8 / PI;
        assert!(radiance.abs_diff_eq(Vec3::splat(expected), 1.0e-5));
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
            &mut rand::rng(),
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

        let mut rng = rand::rng();
        let ray = Ray::new(Vec3::ZERO, Vec3::Y);
        let radiance = trace_radiance(&scene, ray, &mut rng, 2);

        assert!(radiance.abs_diff_eq(env_radiance, 1.0e-5));
    }
}
