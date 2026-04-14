use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{
    camera::PinholeCamera,
    material::{DielectricGgxMaterial, EmissiveMaterial, Material, NormalizedLambertMaterial},
    mesh::load_mesh,
    scene::Scene,
};

use super::uniform_scale_for_height;

pub fn create_scene_7() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();
    let wall_gray = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(Vec3::splat(0.60)),
    ));
    let red = scene.add_material(Material::NormalizedLambert(NormalizedLambertMaterial::new(
        Vec3::new(0.63, 0.08, 0.05),
    )));
    let green = scene.add_material(Material::NormalizedLambert(NormalizedLambertMaterial::new(
        Vec3::new(0.14, 0.45, 0.091),
    )));
    let light = scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 20.0)));

    let floor_mesh_index = scene.add_mesh(load_mesh(Path::new("assets/floor.glb"))?);
    scene.add_instance(floor_mesh_index, wall_gray, Mat4::IDENTITY);
    let ceiling_mesh_index = scene.add_mesh(load_mesh(Path::new("assets/ceiling.glb"))?);
    scene.add_instance(ceiling_mesh_index, wall_gray, Mat4::IDENTITY);
    let back_wall_mesh_index = scene.add_mesh(load_mesh(Path::new("assets/back-wall.glb"))?);
    scene.add_instance(back_wall_mesh_index, wall_gray, Mat4::IDENTITY);
    let left_wall_mesh_index = scene.add_mesh(load_mesh(Path::new("assets/left-wall.glb"))?);
    scene.add_instance(left_wall_mesh_index, red, Mat4::IDENTITY);
    let right_wall_mesh_index = scene.add_mesh(load_mesh(Path::new("assets/right-wall.glb"))?);
    scene.add_instance(right_wall_mesh_index, green, Mat4::IDENTITY);
    let light_mesh_index = scene.add_mesh(load_mesh(Path::new("assets/light.glb"))?);
    scene.add_instance(light_mesh_index, light, Mat4::IDENTITY);

    let glass_color = Vec3::new(0.85, 0.95, 1.0);
    let eta = 1.5;
    let roughness = 0.3;
    let left_material = scene.add_material(Material::DielectricGgx(DielectricGgxMaterial::new(
        glass_color,
        eta,
        roughness,
        -1.0,
        false,
    )));
    let center_material = scene.add_material(Material::DielectricGgx(DielectricGgxMaterial::new(
        glass_color,
        eta,
        roughness,
        0.0,
        false,
    )));
    let right_material = scene.add_material(Material::DielectricGgx(DielectricGgxMaterial::new(
        glass_color,
        eta,
        roughness,
        1.0,
        false,
    )));

    let sphere = load_mesh(Path::new("assets/sphere.glb"))?;
    let sphere_scale = uniform_scale_for_height(&sphere, 1.05);
    let sphere_pivot = Vec3::new(
        sphere.bounds.center().x,
        sphere.bounds.min.y,
        sphere.bounds.center().z,
    );
    let sphere_mesh_index = scene.add_mesh(sphere);

    let left_transform = Mat4::from_translation(Vec3::new(-1.28, 1.0, -0.42))
        * Mat4::from_scale(Vec3::splat(sphere_scale))
        * Mat4::from_translation(-sphere_pivot);
    scene.add_instance(sphere_mesh_index, left_material, left_transform);

    let center_transform = Mat4::from_translation(Vec3::new(0.0, 1.0, -0.42))
        * Mat4::from_scale(Vec3::splat(sphere_scale))
        * Mat4::from_translation(-sphere_pivot);
    scene.add_instance(sphere_mesh_index, center_material, center_transform);

    let right_transform = Mat4::from_translation(Vec3::new(1.28, 1.0, -0.42))
        * Mat4::from_scale(Vec3::splat(sphere_scale))
        * Mat4::from_translation(-sphere_pivot);
    scene.add_instance(sphere_mesh_index, right_material, right_transform);

    let camera = PinholeCamera::new(
        Vec3::new(0.0, 2.08, 7.1),
        Vec3::new(0.0, 1.30, -0.35),
        Vec3::Y,
        38.0_f32.to_radians(),
    );

    Ok((scene, camera))
}
