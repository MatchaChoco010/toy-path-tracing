use glam::Vec3;
use std::{error::Error, path::Path};

use crate::{mesh::load_mesh, scene::Scene};

use super::{InstanceSpec, add_instances, create_camera, uniform_scale_for_height};

pub fn create_scene_1() -> Result<(Scene, crate::camera::PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();
    let bunny = load_mesh(Path::new("assets/bunny.glb"))?;
    let sphere = load_mesh(Path::new("assets/sphere.glb"))?;
    let bunny_mesh_index = scene.add_mesh(bunny);
    let sphere_mesh_index = scene.add_mesh(sphere);
    let bunny_base_scale = uniform_scale_for_height(&scene.meshes[bunny_mesh_index.0], 1.00);
    let sphere_base_scale = uniform_scale_for_height(&scene.meshes[sphere_mesh_index.0], 0.72);

    let bunny_instances = [
        InstanceSpec {
            translation: Vec3::new(-3.6, 0.0, -2.4),
            scale_multiplier: 0.92,
            rotation_degrees: Vec3::new(84.0, -32.0, -8.0),
        },
        InstanceSpec {
            translation: Vec3::new(-1.8, 0.0, -3.0),
            scale_multiplier: 1.06,
            rotation_degrees: Vec3::new(96.0, 24.0, 6.0),
        },
        InstanceSpec {
            translation: Vec3::new(0.1, 0.0, -2.0),
            scale_multiplier: 0.88,
            rotation_degrees: Vec3::new(90.0, -68.0, -4.0),
        },
        InstanceSpec {
            translation: Vec3::new(2.0, 0.0, -2.8),
            scale_multiplier: 1.14,
            rotation_degrees: Vec3::new(102.0, 58.0, 10.0),
        },
        InstanceSpec {
            translation: Vec3::new(3.7, 0.0, -1.6),
            scale_multiplier: 0.94,
            rotation_degrees: Vec3::new(88.0, 120.0, -6.0),
        },
        InstanceSpec {
            translation: Vec3::new(-3.1, 0.0, 0.4),
            scale_multiplier: 1.10,
            rotation_degrees: Vec3::new(93.0, 150.0, 12.0),
        },
        InstanceSpec {
            translation: Vec3::new(-1.0, 0.0, 1.2),
            scale_multiplier: 0.90,
            rotation_degrees: Vec3::new(86.0, -138.0, -9.0),
        },
        InstanceSpec {
            translation: Vec3::new(1.1, 0.0, 0.7),
            scale_multiplier: 1.18,
            rotation_degrees: Vec3::new(98.0, -96.0, 7.0),
        },
        InstanceSpec {
            translation: Vec3::new(2.9, 0.0, 1.8),
            scale_multiplier: 0.86,
            rotation_degrees: Vec3::new(92.0, 178.0, -11.0),
        },
        InstanceSpec {
            translation: Vec3::new(4.3, 0.0, 0.5),
            scale_multiplier: 1.04,
            rotation_degrees: Vec3::new(95.0, 74.0, 5.0),
        },
    ];

    let sphere_instances = [
        InstanceSpec {
            translation: Vec3::new(-4.4, 0.0, -1.8),
            scale_multiplier: 0.82,
            rotation_degrees: Vec3::ZERO,
        },
        InstanceSpec {
            translation: Vec3::new(-2.4, 0.0, -1.2),
            scale_multiplier: 1.18,
            rotation_degrees: Vec3::ZERO,
        },
        InstanceSpec {
            translation: Vec3::new(-0.5, 0.0, -3.3),
            scale_multiplier: 0.76,
            rotation_degrees: Vec3::ZERO,
        },
        InstanceSpec {
            translation: Vec3::new(1.7, 0.0, -1.1),
            scale_multiplier: 0.97,
            rotation_degrees: Vec3::ZERO,
        },
        InstanceSpec {
            translation: Vec3::new(4.0, 0.0, -3.0),
            scale_multiplier: 1.10,
            rotation_degrees: Vec3::ZERO,
        },
        InstanceSpec {
            translation: Vec3::new(-4.0, 0.0, 1.5),
            scale_multiplier: 0.90,
            rotation_degrees: Vec3::ZERO,
        },
        InstanceSpec {
            translation: Vec3::new(-1.7, 0.0, 2.7),
            scale_multiplier: 1.04,
            rotation_degrees: Vec3::ZERO,
        },
        InstanceSpec {
            translation: Vec3::new(0.6, 0.0, 2.2),
            scale_multiplier: 0.84,
            rotation_degrees: Vec3::ZERO,
        },
        InstanceSpec {
            translation: Vec3::new(2.8, 0.0, 3.0),
            scale_multiplier: 1.12,
            rotation_degrees: Vec3::ZERO,
        },
        InstanceSpec {
            translation: Vec3::new(4.8, 0.0, 1.9),
            scale_multiplier: 0.92,
            rotation_degrees: Vec3::ZERO,
        },
    ];

    add_instances(
        &mut scene,
        bunny_mesh_index,
        bunny_base_scale,
        &bunny_instances,
    );
    add_instances(
        &mut scene,
        sphere_mesh_index,
        sphere_base_scale,
        &sphere_instances,
    );

    let bounds = scene
        .bounds()
        .ok_or("scene must contain at least one instance")?;
    let camera = create_camera(
        bounds,
        Vec3::new(0.10, 0.72, 1.18),
        Vec3::new(0.02, 0.08, -0.02),
        46.0_f32.to_radians(),
    );

    Ok((scene, camera))
}
