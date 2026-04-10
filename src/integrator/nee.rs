use glam::{Vec2, Vec3};
use rand::RngExt;

use crate::{
    bsdf::BsdfFlags,
    material::{Material, ShadingVertex},
    math::russian_roulette_probability,
    ray::Ray,
    scene::Scene,
};

const RAY_EPSILON: f32 = 1.0e-4;

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
    let mut count_emission_at_hit = true;
    let rr_start_depth = 4;

    for depth in 0..max_depth {
        if count_emission_at_hit {
            if let Some(le) = material.le(&vtx) {
                radiance += throughput * le;
            }
        }

        let us = Vec2::new(rng.random::<f32>(), rng.random::<f32>());
        let Some(sample) = material.sample(&vtx, us) else {
            break;
        };
        let is_delta_sample = sample.flags.contains(BsdfFlags::DELTA);

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

        let next_ray = Ray::new(vtx.p + RAY_EPSILON * vtx.ng, sample.wi);
        let Some(light_hit) = scene
            .closest_hit(&next_ray)
            .expect("scene.build_bvh() must be called before traversal")
        else {
            break;
        };

        vtx = scene.shading_vertex(light_hit, next_ray.direction);
        material = scene.instance_material(light_hit.triangle.instance_index);
        count_emission_at_hit = is_delta_sample;
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

    if light_sample.pdf_area <= 0.0 {
        return Vec3::ZERO;
    }

    let to_light = light_sample.p - vtx.p;
    let distance_squared = to_light.length_squared();
    if distance_squared <= 0.0 {
        return Vec3::ZERO;
    }

    let distance = distance_squared.sqrt();
    // We need the actual distance for the geometry term anyway, so build the
    // unit direction from the same scalar instead of normalizing separately.
    let wi = to_light / distance;
    let f = material.eval(vtx, wi);
    if f.length_squared() == 0.0 {
        return Vec3::ZERO;
    }

    let cos_surface = vtx.ns.dot(wi).max(0.0);
    if cos_surface <= 0.0 {
        return Vec3::ZERO;
    }

    let shadow_ray = Ray::new(vtx.p + RAY_EPSILON * vtx.ng, wi);
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
    if !light_material.may_emit() {
        return Vec3::ZERO;
    }

    let Some(le) = light_material.le(&lvtx) else {
        return Vec3::ZERO;
    };

    let cos_light = lvtx.ng.dot(-wi).max(0.0);
    if cos_light <= 0.0 {
        return Vec3::ZERO;
    }

    let geometry = (cos_surface * cos_light) / distance_squared;
    le * f * (geometry / light_sample.pdf_area)
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use glam::{Mat4, Vec2, Vec3};

    use super::super::test_helpers::mirror_to_light_scene;
    use super::{sample_direct_area_light_radiance, should_sample_direct_light, trace_radiance};
    use crate::{
        bsdf::BsdfFlags,
        material::{EmissiveMaterial, Material, MirrorMaterial, NormalizedLambertMaterial},
        mesh::{Mesh, Vertex},
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
        scene.build_bvh();

        let vtx = scene.shading_vertex_from_triangle_sample(
            triangle_ref(0),
            Vec3::new(0.5, 0.25, 0.25),
            Vec3::NEG_Z,
        );
        let material = scene.instance_material(InstanceIndex(0));
        let radiance =
            sample_direct_area_light_radiance(&scene, material, &vtx, 0.5, Vec2::new(0.25, 0.5));

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
        scene.build_bvh();

        let vtx = scene.shading_vertex_from_triangle_sample(
            triangle_ref(0),
            Vec3::new(0.5, 0.25, 0.25),
            Vec3::NEG_Z,
        );
        let material = scene.instance_material(InstanceIndex(0));
        let radiance =
            sample_direct_area_light_radiance(&scene, material, &vtx, 0.5, Vec2::new(0.25, 0.5));

        assert_eq!(radiance, Vec3::ZERO);
    }
}
