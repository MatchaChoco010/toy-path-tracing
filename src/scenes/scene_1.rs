use glam::{Quat, Vec3};
use std::{error::Error, path::Path};

use crate::{camera::PinholeCamera, mesh::load_mesh, scene::Scene};

use super::{add_identity_instance, game_rotation_degrees, uniform_scale_for_height};

pub fn create_scene_1() -> Result<(Scene, crate::camera::PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();
    add_identity_instance(&mut scene, Path::new("assets/floor.glb"))?;
    add_identity_instance(&mut scene, Path::new("assets/ceiling.glb"))?;
    add_identity_instance(&mut scene, Path::new("assets/back-wall.glb"))?;
    add_identity_instance(&mut scene, Path::new("assets/left-wall.glb"))?;
    add_identity_instance(&mut scene, Path::new("assets/right-wall.glb"))?;
    add_identity_instance(&mut scene, Path::new("assets/light.glb"))?;

    let bunny_mesh_index = scene.add_mesh(load_mesh(Path::new("assets/bunny.glb"))?);
    let sphere_mesh_index = scene.add_mesh(load_mesh(Path::new("assets/sphere.glb"))?);
    let bunny_scale = uniform_scale_for_height(&scene.meshes[bunny_mesh_index.0], 1.38);
    let sphere_scale = uniform_scale_for_height(&scene.meshes[sphere_mesh_index.0], 0.90);

    scene.add_instance(
        bunny_mesh_index,
        Vec3::new(-0.78, 0.0, -0.30),
        game_rotation_degrees(0.0, 22.0, 0.0),
        Vec3::splat(bunny_scale),
    );
    scene.add_instance(
        sphere_mesh_index,
        Vec3::new(0.92, 0.0, -0.78),
        Quat::IDENTITY,
        Vec3::splat(sphere_scale * 1.10),
    );
    scene.add_instance(
        sphere_mesh_index,
        Vec3::new(0.35, 0.0, 0.98),
        Quat::IDENTITY,
        Vec3::splat(sphere_scale * 0.78),
    );

    let camera = PinholeCamera::new(
        Vec3::new(0.0, 2.10, 7.0),
        Vec3::new(0.0, 1.40, -0.05),
        Vec3::Y,
        38.0_f32.to_radians(),
    );

    Ok((scene, camera))
}
