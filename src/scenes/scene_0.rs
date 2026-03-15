use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{camera::PinholeCamera, mesh::load_mesh, scene::Scene};

use super::{add_identity_instance, game_rotation_degrees};

pub fn create_scene_0() -> Result<(Scene, crate::camera::PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();
    let gray = scene.add_material(crate::scene::Material::Diffuse {
        rho: Vec3::splat(0.75),
    });
    let red = scene.add_material(crate::scene::Material::Diffuse {
        rho: Vec3::new(0.63, 0.08, 0.05),
    });
    let green = scene.add_material(crate::scene::Material::Diffuse {
        rho: Vec3::new(0.14, 0.45, 0.091),
    });
    let light = scene.add_material(crate::scene::Material::Emissive {
        color: Vec3::ONE,
        strength: 20.0,
    });

    add_identity_instance(&mut scene, Path::new("assets/floor.glb"), gray)?;
    add_identity_instance(&mut scene, Path::new("assets/ceiling.glb"), gray)?;
    add_identity_instance(&mut scene, Path::new("assets/back-wall.glb"), gray)?;
    add_identity_instance(&mut scene, Path::new("assets/left-wall.glb"), red)?;
    add_identity_instance(&mut scene, Path::new("assets/right-wall.glb"), green)?;
    add_identity_instance(&mut scene, Path::new("assets/light.glb"), light)?;

    let bunny_mesh_index = scene.add_mesh(load_mesh(Path::new("assets/bunny.glb"))?);
    let box_mesh_index = scene.add_mesh(load_mesh(Path::new("assets/box.glb"))?);

    scene.add_instance(
        bunny_mesh_index,
        gray,
        Vec3::new(0.72, 0.0, 0.65),
        game_rotation_degrees(0.0, -28.0, 0.0),
        Vec3::splat(0.90),
    );
    scene.add_instance_raw(box_mesh_index, gray, Mat4::IDENTITY);

    let camera = PinholeCamera::new(
        Vec3::new(0.0, 2.15, 7.1),
        Vec3::new(0.0, 1.45, -0.05),
        Vec3::Y,
        38.0_f32.to_radians(),
    );

    Ok((scene, camera))
}
