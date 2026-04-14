use glam::{Vec2, Vec3};
use rand::RngExt;

use crate::{
    bsdf::BsdfFlags,
    material::{Material, ShadingVertex},
    math::{balance_heuristic, russian_roulette_probability},
    ray::Ray,
    scene::Scene,
};

use super::spawn_ray;

pub fn trace_radiance(
    scene: &Scene,
    initial_ray: Ray,
    rng: &mut rand::rngs::ThreadRng,
    max_depth: u32,
) -> Vec3 {
    let Some(initial_hit) = scene
        .closest_hit(&initial_ray)
        .expect("scene.build_bvh() must be called before traversal")
    else {
        return Vec3::ZERO;
    };

    let mut radiance = Vec3::ZERO;
    let mut throughput = Vec3::ONE;
    let mut vtx = scene.shading_vertex(initial_hit, initial_ray.direction);
    let mut material = scene.instance_material(initial_hit.triangle.instance_index);
    let mut is_primary_hit = true;
    let rr_start_depth = 4;

    for depth in 0..max_depth {
        if is_primary_hit {
            if let Some(le) = material.le(&vtx) {
                radiance += throughput * le;
            }
        }

        let Some(sample) = material.sample(&vtx, rng) else {
            break;
        };

        if should_sample_direct_light(material, sample.flags) {
            let u_triangle = rng.random::<f32>();
            let us = Vec2::new(rng.random::<f32>(), rng.random::<f32>());
            radiance += throughput
                * sample_direct_area_light_radiance(scene, material, &vtx, u_triangle, us);
        }

        throughput *= sample.weight;

        if depth + 1 >= rr_start_depth {
            let survive_probability = russian_roulette_probability(throughput);
            if rng.random::<f32>() > survive_probability {
                break;
            }
            throughput /= survive_probability;
        }

        let next_ray = spawn_ray(vtx.p, vtx.ng, sample.wi);
        let Some(next_hit) = scene
            .closest_hit(&next_ray)
            .expect("scene.build_bvh() must be called before traversal")
        else {
            break;
        };

        let next_vtx = scene.shading_vertex(next_hit, next_ray.direction);
        let next_material = scene.instance_material(next_hit.triangle.instance_index);
        radiance += emitted_radiance_from_bsdf_sample(
            scene,
            throughput,
            sample.pdf,
            sample.flags,
            &vtx,
            &next_vtx,
            next_material,
        );

        vtx = next_vtx;
        material = next_material;
        is_primary_hit = false;
    }

    radiance
}

fn should_sample_direct_light(material: &Material, sample_flags: BsdfFlags) -> bool {
    !material.may_emit() && !sample_flags.contains(BsdfFlags::DELTA)
}

fn sample_direct_area_light_radiance(
    scene: &Scene,
    material: &Material,
    vtx: &ShadingVertex,
    u_triangle: f32,
    us: Vec2,
) -> Vec3 {
    let Some(light_sample) = scene.sample_area_light_point(u_triangle, us) else {
        return Vec3::ZERO;
    };

    let to_light = light_sample.p - vtx.p;
    let distance_squared = to_light.length_squared();
    if distance_squared <= 0.0 {
        return Vec3::ZERO;
    }

    let distance = distance_squared.sqrt();
    let wi = to_light / distance;
    let f = material.eval(vtx, wi);
    if f.length_squared() == 0.0 {
        return Vec3::ZERO;
    }

    let cos_surface = vtx.ns.dot(wi).max(0.0);
    if cos_surface <= 0.0 {
        return Vec3::ZERO;
    }

    let shadow_ray = spawn_ray(vtx.p, vtx.ng, wi);
    let Some(shadow_hit) = scene
        .closest_hit(&shadow_ray)
        .expect("scene.build_bvh() must be called before traversal")
    else {
        return Vec3::ZERO;
    };

    if shadow_hit.triangle != light_sample.triangle {
        return Vec3::ZERO;
    }

    let lvtx = scene.shading_vertex_from_triangle_sample(
        light_sample.triangle,
        light_sample.barycentric,
        shadow_ray.direction,
    );
    let light_material = scene.instance_material(light_sample.triangle.instance_index);
    let Some(le) = light_material.le(&lvtx) else {
        return Vec3::ZERO;
    };

    let light_pdf = scene.area_light_pdf_solid_angle(vtx, &lvtx).unwrap_or(0.0);
    if light_pdf <= 0.0 {
        return Vec3::ZERO;
    }

    let bsdf_pdf = material.pdf(vtx, wi);
    let mis_weight = balance_heuristic(light_pdf, bsdf_pdf);

    le * f * (cos_surface * mis_weight / light_pdf)
}

fn emitted_radiance_from_bsdf_sample(
    scene: &Scene,
    throughput: Vec3,
    bsdf_pdf: f32,
    bsdf_flags: BsdfFlags,
    vtx: &ShadingVertex,
    lvtx: &ShadingVertex,
    light_material: &Material,
) -> Vec3 {
    let Some(le) = light_material.le(lvtx) else {
        return Vec3::ZERO;
    };

    if bsdf_flags.contains(BsdfFlags::DELTA) {
        return throughput * le;
    }

    let light_pdf = scene.area_light_pdf_solid_angle(vtx, lvtx).unwrap_or(0.0);
    let mis_weight = balance_heuristic(bsdf_pdf, light_pdf);

    throughput * le * mis_weight
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use glam::{Mat4, Vec2, Vec3};

    use crate::{
        bsdf::BsdfFlags,
        material::{EmissiveMaterial, Material, MirrorMaterial, NormalizedLambertMaterial},
        mesh::{Mesh, Vertex},
        scene::{InstanceIndex, Scene, TriangleRef},
    };

    use super::super::test_helpers::mirror_to_light_scene;
    use super::{
        emitted_radiance_from_bsdf_sample, sample_direct_area_light_radiance,
        should_sample_direct_light, trace_radiance,
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
        scene.build_bvh();

        let vtx = scene.shading_vertex_from_triangle_sample(
            triangle_ref(0),
            Vec3::new(0.5, 0.25, 0.25),
            Vec3::NEG_Z,
        );
        let material = scene.instance_material(InstanceIndex(0));
        let radiance =
            sample_direct_area_light_radiance(&scene, material, &vtx, 0.5, Vec2::new(0.25, 0.5));
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
        scene.build_bvh();

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
        let radiance = emitted_radiance_from_bsdf_sample(
            &scene,
            throughput,
            bsdf_pdf,
            BsdfFlags::DIFFUSE,
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
        scene.build_bvh();

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

        let radiance = emitted_radiance_from_bsdf_sample(
            &scene,
            throughput,
            1.0,
            BsdfFlags::DELTA,
            &vtx,
            &lvtx,
            light_material,
        );

        assert_eq!(radiance, Vec3::splat(8.0));
    }
}
