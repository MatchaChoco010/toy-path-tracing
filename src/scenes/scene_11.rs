use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{
    camera::PinholeCamera,
    light::DirectionalLight,
    material::{Material, NormalizedLambertMaterial},
    mesh::load_gltf,
    scene::Scene,
    scenes::uniform_scale_for_height,
};

pub fn create_scene_11() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let floor_material = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(Vec3::splat(0.6)),
    ));
    let floor_mesh_index = scene.add_mesh(load_gltf(Path::new("assets/models/floor.glb"))?);
    let floor_transform = Mat4::from_scale(Vec3::new(8.0, 1.0, 8.0));
    scene.add_instance(floor_mesh_index, floor_material, floor_transform);

    let bunny_material = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(Vec3::new(0.55, 0.72, 0.92)),
    ));
    let bunny = load_gltf(Path::new("assets/models/bunny.glb"))?;
    let bunny_height = 2.0;
    let bunny_scale = uniform_scale_for_height(&bunny, bunny_height);
    let bunny_pivot = Vec3::new(
        bunny.bounds.center().x,
        bunny.bounds.min.y,
        bunny.bounds.center().z,
    );
    let bunny_mesh_index = scene.add_mesh(bunny);
    let bunny_transform = Mat4::from_translation(Vec3::ZERO)
        * Mat4::from_scale(Vec3::splat(bunny_scale))
        * Mat4::from_translation(-bunny_pivot);
    scene.add_instance(bunny_mesh_index, bunny_material, bunny_transform);

    scene.add_directional_light(DirectionalLight::new(
        Vec3::new(-0.4, -1.0, -0.35),
        Vec3::new(1.0, 0.96, 0.88),
        3.0,
    ));

    let camera = PinholeCamera::new(
        Vec3::new(0.0, 1.8, 6.0),
        Vec3::new(0.0, 0.9, 0.0),
        Vec3::Y,
        38.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}
