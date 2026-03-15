use glam::{Quat, Vec3};
use std::{error::Error, path::Path};

use crate::{mesh::load_mesh, scene::Scene};

use super::{create_camera, game_rotation_degrees, uniform_scale_for_height};

pub fn create_scene_0() -> Result<(Scene, crate::camera::PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();
    let bunny = load_mesh(Path::new("assets/bunny.glb"))?;
    let sphere = load_mesh(Path::new("assets/sphere.glb"))?;
    let bunny_scale = uniform_scale_for_height(&bunny, 1.55);
    let sphere_scale = uniform_scale_for_height(&sphere, 1.10);
    let bunny_mesh_index = scene.add_mesh(bunny);
    let sphere_mesh_index = scene.add_mesh(sphere);

    scene.add_instance(
        bunny_mesh_index,
        Vec3::new(-0.95, 0.0, 0.05),
        game_rotation_degrees(90.0, 28.0, 0.0),
        Vec3::splat(bunny_scale),
    );
    scene.add_instance(
        sphere_mesh_index,
        Vec3::new(1.05, 0.0, -0.10),
        Quat::IDENTITY,
        Vec3::splat(sphere_scale),
    );

    let bounds = scene
        .bounds()
        .ok_or("scene must contain at least one instance")?;
    let camera = create_camera(
        bounds,
        Vec3::new(0.0, 0.22, 1.55),
        Vec3::new(0.0, 0.12, 0.0),
        33.0_f32.to_radians(),
    );

    Ok((scene, camera))
}
