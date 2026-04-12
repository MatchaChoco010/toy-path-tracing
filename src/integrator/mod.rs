use clap::ValueEnum;
use glam::Vec3;
use rand::rngs::ThreadRng;

use crate::{ray::Ray, scene::Scene};

pub mod mis;
pub mod nee;
pub mod pt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum IntegratorKind {
    Mis,
    Pt,
    Nee,
}

impl IntegratorKind {
    pub fn trace_radiance(
        self,
        scene: &Scene,
        initial_ray: Ray,
        rng: &mut ThreadRng,
        max_depth: u32,
    ) -> Vec3 {
        match self {
            Self::Mis => mis::trace_radiance(scene, initial_ray, rng, max_depth),
            Self::Pt => pt::trace_radiance(scene, initial_ray, rng, max_depth),
            Self::Nee => nee::trace_radiance(scene, initial_ray, rng, max_depth),
        }
    }
}

const RAY_EPSILON: f32 = 1.0e-4;

pub(super) fn spawn_ray(origin: Vec3, geometric_normal: Vec3, direction: Vec3) -> Ray {
    let normal_offset = if direction.dot(geometric_normal) >= 0.0 {
        geometric_normal
    } else {
        -geometric_normal
    };

    Ray::new(origin + RAY_EPSILON * normal_offset, direction)
}

#[cfg(test)]
pub(super) mod test_helpers {
    use glam::{Vec2, Vec3};

    use crate::{
        material::{EmissiveMaterial, Material, MirrorMaterial},
        mesh::{Mesh, Vertex},
        ray::Ray,
        scene::Scene,
    };

    pub(super) fn mirror_to_light_scene() -> (Scene, Ray, Vec3) {
        let mut scene = Scene::new();
        let mirror_color = Vec3::new(0.25, 0.5, 0.75);
        let light_strength = 4.0;
        let mirror_material =
            scene.add_material(Material::Mirror(MirrorMaterial::new(mirror_color)));
        let light_material = scene.add_material(Material::Emissive(EmissiveMaterial::new(
            Vec3::ONE,
            light_strength,
        )));
        let mirror_mesh = scene.add_mesh(mirror_triangle_mesh());
        let light_mesh = scene.add_mesh(light_triangle_mesh());

        scene.add_instance(mirror_mesh, mirror_material, glam::Mat4::IDENTITY);
        scene.add_instance(light_mesh, light_material, glam::Mat4::IDENTITY);
        scene.build_bvh();

        let mirror_hit = Vec3::new(0.25, 0.20, 0.0);
        let light_hit = Vec3::new(0.65, 0.20, 1.0);
        let reflected_direction = (light_hit - mirror_hit).normalize();
        let incoming_direction = Vec3::new(
            reflected_direction.x,
            reflected_direction.y,
            -reflected_direction.z,
        )
        .normalize();
        let ray = Ray::new(mirror_hit - incoming_direction, incoming_direction);

        (scene, ray, mirror_color * light_strength)
    }

    fn mirror_triangle_mesh() -> Mesh {
        Mesh::new(
            vec![
                Vertex {
                    position: Vec3::new(-1.0, -1.0, 0.0),
                    normal: Vec3::Z,
                    uv: Vec2::ZERO,
                },
                Vertex {
                    position: Vec3::new(2.0, -1.0, 0.0),
                    normal: Vec3::Z,
                    uv: Vec2::X,
                },
                Vertex {
                    position: Vec3::new(-1.0, 2.0, 0.0),
                    normal: Vec3::Z,
                    uv: Vec2::Y,
                },
            ],
            vec![0, 1, 2],
        )
    }

    fn light_triangle_mesh() -> Mesh {
        Mesh::new(
            vec![
                Vertex {
                    position: Vec3::new(0.45, 0.0, 1.0),
                    normal: Vec3::Z,
                    uv: Vec2::ZERO,
                },
                Vertex {
                    position: Vec3::new(0.95, 0.0, 1.0),
                    normal: Vec3::Z,
                    uv: Vec2::X,
                },
                Vertex {
                    position: Vec3::new(0.45, 0.5, 1.0),
                    normal: Vec3::Z,
                    uv: Vec2::Y,
                },
            ],
            vec![0, 1, 2],
        )
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::spawn_ray;

    #[test]
    fn spawn_ray_offsets_along_the_sampled_hemisphere() {
        let reflection_ray = spawn_ray(Vec3::ZERO, Vec3::Z, Vec3::Z);
        let transmission_ray = spawn_ray(Vec3::ZERO, Vec3::Z, Vec3::NEG_Z);

        assert_eq!(reflection_ray.origin, Vec3::new(0.0, 0.0, 1.0e-4));
        assert_eq!(transmission_ray.origin, Vec3::new(0.0, 0.0, -1.0e-4));
    }
}
