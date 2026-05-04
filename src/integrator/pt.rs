use glam::Vec3;
use rand::RngExt;

use crate::{light::infinite_light_le, math::russian_roulette_probability, ray::Ray, scene::Scene};

use super::spawn_scattered_ray;

pub fn trace_radiance(
    scene: &Scene,
    initial_ray: Ray,
    rng: &mut rand::rngs::ThreadRng,
    max_depth: u32,
) -> Vec3 {
    let mut radiance = Vec3::ZERO;
    let mut throughput = Vec3::ONE;
    let mut ray = initial_ray;
    let mut wavelength_lock: Option<f32> = None;
    let rr_start_depth = 4;

    for depth in 0..max_depth {
        let hit = scene
            .closest_hit(&ray, rng)
            .expect("scene.build_qbvh() must be called before traversal");

        let Some(hit) = hit else {
            radiance += throughput * infinite_light_le(scene, ray.direction);
            break;
        };

        let mut shading_vertex = scene.shading_vertex(hit, &ray);
        shading_vertex.wavelength_lock = wavelength_lock;
        let material = scene.instance_material(hit.triangle.instance_index);

        if let Some(le) = material.le(&shading_vertex) {
            radiance += throughput * le;
        }

        let Some(sample) = material.sample(&shading_vertex, rng) else {
            break;
        };

        throughput *= sample.weight;
        if let Some(lambda) = sample.wavelength_lock {
            wavelength_lock = Some(lambda);
        }

        if depth + 1 >= rr_start_depth {
            let survive_probability = russian_roulette_probability(throughput);
            if rng.random::<f32>() > survive_probability {
                break;
            }
            throughput /= survive_probability;
        }

        ray = spawn_scattered_ray(&ray, hit.t, &shading_vertex, &sample);
    }

    radiance
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::super::test_helpers::mirror_to_light_scene;
    use super::trace_radiance;
    use crate::{light::EnvironmentLight, ray::Ray, scene::Scene};

    #[test]
    fn trace_radiance_counts_light_after_delta_bounce() {
        let (scene, ray, expected) = mirror_to_light_scene();
        let mut rng = rand::rng();

        let radiance = trace_radiance(&scene, ray, &mut rng, 2);

        assert!(radiance.abs_diff_eq(expected, 1.0e-5));
    }

    #[test]
    fn trace_radiance_returns_environment_light_on_direct_escape() {
        let mut scene = Scene::new();
        let env_radiance = Vec3::new(0.2, 0.4, 0.8);
        let pixels = vec![env_radiance; 16 * 8];
        scene.set_environment_light(EnvironmentLight::from_pixels(16, 8, pixels, 1.0, 0.0));
        scene.build_qbvh();
        scene.build_light_tree();

        let mut rng = rand::rng();
        let ray = Ray::new(Vec3::ZERO, Vec3::Y);
        let radiance = trace_radiance(&scene, ray, &mut rng, 4);

        assert!(radiance.abs_diff_eq(env_radiance, 1.0e-5));
    }
}
