use glam::{Vec2, Vec3};
use rand::RngExt;

use crate::{
    bsdf::BsdfFlags,
    light::{
        LightKind, LightSampleContext, area_light_pdf_li, infinite_light_le,
        infinite_light_pdf_li_mis_compensated, sample_light_li_mis_compensated,
    },
    material::{Material, ShadingVertex},
    math::{balance_heuristic, russian_roulette_probability},
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

    let initial_hit = scene
        .closest_hit(&initial_ray, rng)
        .expect("scene.build_qbvh() must be called before traversal");

    let Some(initial_hit) = initial_hit else {
        return infinite_light_le(scene, initial_ray.direction);
    };

    let mut ray = initial_ray;
    let mut hit_t = initial_hit.t;
    let mut vtx = scene.shading_vertex(initial_hit, &ray);
    let mut material = scene.instance_material(initial_hit.triangle.instance_index);

    if let Some(le) = material.le(&vtx) {
        radiance += le;
    }

    let rr_start_depth = 4;

    for depth in 0..max_depth {
        let Some(sample) = material.sample(&vtx, rng) else {
            break;
        };
        let is_delta_sample = sample.flags.contains(BsdfFlags::DELTA);

        if should_sample_direct_light(material, sample.flags) {
            let u_light_select = rng.random::<f32>();
            let u_aux = rng.random::<f32>();
            let us = Vec2::new(rng.random::<f32>(), rng.random::<f32>());
            radiance += throughput
                * direct_light_mis_contribution(
                    scene,
                    material,
                    &vtx,
                    u_light_select,
                    u_aux,
                    us,
                    rng,
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

        let next_ray = spawn_scattered_ray(&ray, hit_t, &vtx, &sample);
        let next_hit = scene
            .closest_hit(&next_ray, rng)
            .expect("scene.build_qbvh() must be called before traversal");

        match next_hit {
            Some(next_hit) => {
                let next_vtx = scene.shading_vertex(next_hit, &next_ray);
                let next_material = scene.instance_material(next_hit.triangle.instance_index);
                radiance += emitted_radiance_from_bsdf_sample_area(
                    scene,
                    throughput,
                    sample.pdf,
                    is_delta_sample,
                    &vtx,
                    &next_vtx,
                    next_material,
                );

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

pub(super) fn should_sample_direct_light(material: &Material, sample_flags: BsdfFlags) -> bool {
    !material.may_emit() && !sample_flags.contains(BsdfFlags::DELTA)
}

pub(super) fn direct_light_mis_contribution(
    scene: &Scene,
    material: &Material,
    vtx: &ShadingVertex,
    u_light_select: f32,
    u_aux: f32,
    us: Vec2,
    rng: &mut rand::rngs::ThreadRng,
) -> Vec3 {
    let Some(sampled_light) = scene.light_sampler.sample(u_light_select) else {
        return Vec3::ZERO;
    };

    let ctx = LightSampleContext::from_vertex(vtx);
    let Some(li) = sample_light_li_mis_compensated(scene, sampled_light.kind, &ctx, u_aux, us)
    else {
        return Vec3::ZERO;
    };

    if li.pdf <= 0.0 {
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

    let pdf_total = sampled_light.pmf * li.pdf;
    if pdf_total <= 0.0 {
        return Vec3::ZERO;
    }

    if li.light_type.is_delta() {
        return li.radiance * f * (cos_surface / pdf_total);
    }

    let bsdf_pdf = material.pdf(vtx, li.wi);
    let mis_weight = balance_heuristic(pdf_total, bsdf_pdf);
    li.radiance * f * (cos_surface * mis_weight / pdf_total)
}

fn emitted_radiance_from_bsdf_sample_area(
    scene: &Scene,
    throughput: Vec3,
    bsdf_pdf: f32,
    is_delta_sample: bool,
    vtx: &ShadingVertex,
    lvtx: &ShadingVertex,
    light_material: &Material,
) -> Vec3 {
    let Some(le) = light_material.le(lvtx) else {
        return Vec3::ZERO;
    };

    if is_delta_sample {
        return throughput * le;
    }

    let pmf = scene.light_sampler.pmf(LightKind::Area);
    if pmf <= 0.0 {
        return throughput * le;
    }

    let light_pdf = pmf * area_light_pdf_li(scene, vtx, lvtx);
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

    let pmf = scene.light_sampler.pmf(LightKind::Infinite);
    if pmf <= 0.0 {
        return throughput * le;
    }

    let light_pdf = pmf * infinite_light_pdf_li_mis_compensated(scene, direction);
    let mis_weight = balance_heuristic(bsdf_pdf, light_pdf);

    throughput * le * mis_weight
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use glam::{Mat4, Vec2, Vec3};

    use crate::{
        bsdf::BsdfFlags,
        light::{DirectionalLight, EnvironmentLight, PointLight, SpotLight},
        material::{EmissiveMaterial, Material, MirrorMaterial, NormalizedLambertMaterial},
        mesh::{Mesh, Vertex},
        ray::Ray,
        scene::{InstanceIndex, Scene, TriangleRef},
    };

    use super::super::test_helpers::mirror_to_light_scene;
    use super::{
        direct_light_mis_contribution, emitted_radiance_from_bsdf_sample_area,
        emitted_radiance_from_bsdf_sample_infinite, should_sample_direct_light, trace_radiance,
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
    fn direct_radiance_applies_balance_heuristic_for_light_sampling() {
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

        let vtx = scene.shading_vertex_from_triangle_sample(
            triangle_ref(0),
            Vec3::new(0.5, 0.25, 0.25),
            Vec3::NEG_Z,
        );
        let material = scene.instance_material(InstanceIndex(0));
        // Single area light -> light sampler pmf = 1.
        let radiance =
            direct_light_mis_contribution(&scene, material, &vtx, 0.0, 0.5, Vec2::new(0.25, 0.5), &mut rand::rng());
        let expected = (4.0 / PI) * (2.0 / (2.0 + 1.0 / PI));

        assert!(radiance.abs_diff_eq(Vec3::splat(expected), 1.0e-5));
    }

    #[test]
    fn emitted_radiance_applies_balance_heuristic_for_bsdf_sampling() {
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

        let vtx = scene.shading_vertex_from_triangle_sample(
            triangle_ref(0),
            Vec3::new(0.5, 0.25, 0.25),
            Vec3::NEG_Z,
        );
        let lvtx = scene.shading_vertex_from_triangle_sample(
            triangle_ref(1),
            Vec3::new(0.5, 0.25, 0.25),
            Vec3::Z,
        );
        let material = scene.instance_material(InstanceIndex(0));
        let light_material = scene.instance_material(InstanceIndex(1));
        let bsdf_pdf = material.pdf(&vtx, Vec3::Z);
        let throughput = Vec3::splat(0.8);
        let radiance = emitted_radiance_from_bsdf_sample_area(
            &scene,
            throughput,
            bsdf_pdf,
            false,
            &vtx,
            &lvtx,
            light_material,
        );
        let expected = 8.0 * ((1.0 / PI) / (2.0 + 1.0 / PI));

        assert!(radiance.abs_diff_eq(Vec3::splat(expected), 1.0e-5));
    }

    #[test]
    fn emitted_radiance_from_delta_bsdf_sample_uses_full_weight() {
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

        let vtx = scene.shading_vertex_from_triangle_sample(
            triangle_ref(0),
            Vec3::new(0.5, 0.25, 0.25),
            Vec3::NEG_Z,
        );
        let lvtx = scene.shading_vertex_from_triangle_sample(
            triangle_ref(1),
            Vec3::new(0.5, 0.25, 0.25),
            Vec3::Z,
        );
        let light_material = scene.instance_material(InstanceIndex(1));
        let throughput = Vec3::splat(0.8);

        let radiance = emitted_radiance_from_bsdf_sample_area(
            &scene,
            throughput,
            1.0,
            true,
            &vtx,
            &lvtx,
            light_material,
        );

        assert_eq!(radiance, Vec3::splat(8.0));
    }

    #[test]
    fn emitted_radiance_from_infinite_light_applies_mis_for_bsdf_sampling() {
        let mut scene = Scene::new();
        let env_radiance = Vec3::splat(2.0);
        let pixels = vec![env_radiance; 32 * 16];
        scene.set_environment_light(EnvironmentLight::from_pixels(32, 16, pixels, 1.0, 0.0));
        scene.build_qbvh();

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

        let mut rng = rand::rng();
        let ray = Ray::new(Vec3::ZERO, Vec3::Y);
        let radiance = trace_radiance(&scene, ray, &mut rng, 4);

        assert!(radiance.abs_diff_eq(env_radiance, 1.0e-5));
    }

    #[test]
    fn direct_radiance_uses_full_weight_for_point_light() {
        let mut scene = Scene::new();
        let floor_mesh = scene.add_mesh(unit_mesh(0.0));
        let floor_material = scene.add_material(Material::NormalizedLambert(
            NormalizedLambertMaterial::new(Vec3::splat(0.8)),
        ));
        scene.add_instance(floor_mesh, floor_material, Mat4::IDENTITY);
        // Power chosen so Li at r=2 equals 1: P / (4π·r²) = 1  =>  P = 16π.
        scene.add_point_light(PointLight::new(
            Vec3::new(0.25, 0.25, 2.0),
            Vec3::ONE,
            16.0 * PI,
        ));
        scene.build_qbvh();

        let vtx = scene.shading_vertex_from_triangle_sample(
            triangle_ref(0),
            Vec3::new(0.5, 0.25, 0.25),
            Vec3::NEG_Z,
        );
        let material = scene.instance_material(InstanceIndex(0));
        let radiance =
            direct_light_mis_contribution(&scene, material, &vtx, 0.0, 0.0, Vec2::ZERO, &mut rand::rng());
        // Li=1 * lambert eval 0.8/PI * cos=1 / pmf=1 = 0.8/PI.
        let expected = 0.8 / PI;
        assert!(radiance.abs_diff_eq(Vec3::splat(expected), 1.0e-5));
    }

    #[test]
    fn direct_radiance_returns_zero_when_point_light_is_occluded() {
        let mut scene = Scene::new();
        let floor_mesh = scene.add_mesh(unit_mesh(0.0));
        let blocker_mesh = scene.add_mesh(unit_mesh(1.0));
        let floor_material = scene.add_material(Material::NormalizedLambert(
            NormalizedLambertMaterial::new(Vec3::splat(0.8)),
        ));
        let blocker_material = scene.add_material(Material::NormalizedLambert(
            NormalizedLambertMaterial::new(Vec3::splat(0.5)),
        ));
        scene.add_instance(floor_mesh, floor_material, Mat4::IDENTITY);
        scene.add_instance(blocker_mesh, blocker_material, Mat4::IDENTITY);
        scene.add_point_light(PointLight::new(
            Vec3::new(0.25, 0.25, 2.0),
            Vec3::ONE,
            16.0 * PI,
        ));
        scene.build_qbvh();

        let vtx = scene.shading_vertex_from_triangle_sample(
            triangle_ref(0),
            Vec3::new(0.5, 0.25, 0.25),
            Vec3::NEG_Z,
        );
        let material = scene.instance_material(InstanceIndex(0));
        let radiance =
            direct_light_mis_contribution(&scene, material, &vtx, 0.0, 0.0, Vec2::ZERO, &mut rand::rng());
        assert_eq!(radiance, Vec3::ZERO);
    }

    #[test]
    fn direct_radiance_uses_full_weight_for_directional_light() {
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

        let vtx = scene.shading_vertex_from_triangle_sample(
            triangle_ref(0),
            Vec3::new(0.5, 0.25, 0.25),
            Vec3::NEG_Z,
        );
        let material = scene.instance_material(InstanceIndex(0));
        let radiance =
            direct_light_mis_contribution(&scene, material, &vtx, 0.0, 0.0, Vec2::ZERO, &mut rand::rng());
        // Li = color * irradiance = 2; lambert 0.8/PI * cos=1 / pmf=1.
        let expected = 2.0 * 0.8 / PI;
        assert!(radiance.abs_diff_eq(Vec3::splat(expected), 1.0e-5));
    }

    #[test]
    fn direct_radiance_uses_full_weight_for_spot_light_with_falloff() {
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

        let vtx = scene.shading_vertex_from_triangle_sample(
            triangle_ref(0),
            Vec3::new(0.5, 0.25, 0.25),
            Vec3::NEG_Z,
        );
        let material = scene.instance_material(InstanceIndex(0));
        let radiance =
            direct_light_mis_contribution(&scene, material, &vtx, 0.0, 0.0, Vec2::ZERO, &mut rand::rng());
        let expected = 0.8 / PI;
        assert!(radiance.abs_diff_eq(Vec3::splat(expected), 1.0e-5));
    }

    #[test]
    fn direct_radiance_returns_zero_when_shading_point_is_outside_spot_cone() {
        let mut scene = Scene::new();
        let floor_mesh = scene.add_mesh(unit_mesh(0.0));
        let floor_material = scene.add_material(Material::NormalizedLambert(
            NormalizedLambertMaterial::new(Vec3::splat(0.8)),
        ));
        scene.add_instance(floor_mesh, floor_material, Mat4::IDENTITY);
        // Spot pointing +X (away from the floor below it).
        scene.add_spot_light(SpotLight::new(
            Vec3::new(0.25, 0.25, 2.0),
            Vec3::X,
            Vec3::ONE,
            16.0 * PI,
            (10.0_f32).to_radians(),
            (5.0_f32).to_radians(),
        ));
        scene.build_qbvh();

        let vtx = scene.shading_vertex_from_triangle_sample(
            triangle_ref(0),
            Vec3::new(0.5, 0.25, 0.25),
            Vec3::NEG_Z,
        );
        let material = scene.instance_material(InstanceIndex(0));
        let radiance =
            direct_light_mis_contribution(&scene, material, &vtx, 0.0, 0.0, Vec2::ZERO, &mut rand::rng());
        assert_eq!(radiance, Vec3::ZERO);
    }

    #[test]
    fn trace_radiance_mixes_area_light_and_environment_without_panics() {
        let mut scene = Scene::new();
        let floor_mesh = scene.add_mesh(unit_mesh(-1.0));
        let light_mesh = scene.add_mesh(unit_mesh(2.0));
        let floor_material = scene.add_material(Material::NormalizedLambert(
            NormalizedLambertMaterial::new(Vec3::splat(0.6)),
        ));
        let light_material =
            scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 2.0)));
        scene.add_instance(floor_mesh, floor_material, Mat4::IDENTITY);
        scene.add_instance(light_mesh, light_material, Mat4::IDENTITY);

        let env_radiance = Vec3::splat(0.2);
        let pixels = vec![env_radiance; 16 * 8];
        scene.set_environment_light(EnvironmentLight::from_pixels(16, 8, pixels, 1.0, 0.0));
        scene.build_qbvh();

        let mut rng = rand::rng();
        let mut accumulated = Vec3::ZERO;
        for _ in 0..32 {
            let ray = Ray::new(Vec3::new(0.25, 0.25, -2.0), Vec3::Z);
            accumulated += trace_radiance(&scene, ray, &mut rng, 4);
        }
        let mean = accumulated / 32.0;

        // Sanity: at least some radiance arrives (area + env both present).
        assert!(mean.x > 0.0);
        assert!(mean.y > 0.0);
        assert!(mean.z > 0.0);
    }
}
