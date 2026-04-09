use glam::{Vec2, Vec3};
use rand::RngExt;

use crate::{math::russian_roulette_probability, ray::Ray, scene::Scene};

pub fn trace_radiance(
    scene: &Scene,
    initial_ray: Ray,
    rng: &mut rand::rngs::ThreadRng,
    max_depth: u32,
) -> Vec3 {
    let mut radiance = Vec3::ZERO;
    let mut throughput = Vec3::ONE;
    let mut ray = initial_ray;
    let rr_start_depth = 4;

    for depth in 0..max_depth {
        let Some(hit) = scene
            .closest_hit(&ray)
            .expect("scene.build_bvh() must be called before traversal")
        else {
            break;
        };

        let shading_vertex = scene.shading_vertex(hit, ray.direction);
        let material = scene.instance_material(hit.triangle.instance_index);

        if let Some(le) = material.le(&shading_vertex) {
            radiance += throughput * le;
        }

        let us = Vec2::new(rng.random::<f32>(), rng.random::<f32>());
        let Some(sample) = material.sample(&shading_vertex, us) else {
            break;
        };

        throughput *= sample.weight;

        if depth + 1 >= rr_start_depth {
            let survive_probability = russian_roulette_probability(throughput);
            if rng.random::<f32>() > survive_probability {
                break;
            }
            throughput /= survive_probability;
        }

        ray = Ray::new(shading_vertex.p + 1.0e-4 * shading_vertex.ng, sample.wi);
    }

    radiance
}
