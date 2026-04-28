use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{
    camera::PinholeCamera,
    material::{ConductorGgxMaterial, EmissiveMaterial, Material, NormalizedLambertMaterial},
    mesh::load_gltf,
    scene::Scene,
};

use super::{game_rotation_degrees, uniform_scale_for_height};

pub fn create_scene_18() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();
    let normal_strength = 0.2;
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

    let floor_mesh = scene.add_mesh(load_gltf(Path::new("assets/gltf/floor.glb"))?);
    scene.add_instance(floor_mesh, wall_gray, Mat4::IDENTITY);
    let ceiling_mesh = scene.add_mesh(load_gltf(Path::new("assets/gltf/ceiling.glb"))?);
    scene.add_instance(ceiling_mesh, wall_gray, Mat4::IDENTITY);
    let back_wall_mesh = scene.add_mesh(load_gltf(Path::new("assets/gltf/back-wall.glb"))?);
    scene.add_instance(back_wall_mesh, wall_gray, Mat4::IDENTITY);
    let left_wall_mesh = scene.add_mesh(load_gltf(Path::new("assets/gltf/left-wall.glb"))?);
    scene.add_instance(left_wall_mesh, red, Mat4::IDENTITY);
    let right_wall_mesh = scene.add_mesh(load_gltf(Path::new("assets/gltf/right-wall.glb"))?);
    scene.add_instance(right_wall_mesh, green, Mat4::IDENTITY);
    let light_mesh = scene.add_mesh(load_gltf(Path::new("assets/gltf/light.glb"))?);
    scene.add_instance(light_mesh, light, Mat4::IDENTITY);

    let normal_map_path = Path::new("assets/gltf/sphere-normal.png");
    let mut lambert_material = NormalizedLambertMaterial::try_new_with_texture_path(
        Vec3::new(0.72, 0.76, 0.82),
        None,
        Some(normal_map_path),
    )?;
    lambert_material.normal_strength = normal_strength;
    let lambert = scene.add_material(Material::NormalizedLambert(lambert_material));

    let mut metal_material = ConductorGgxMaterial::try_new_with_texture_paths(
        Vec3::new(0.95, 0.78, 0.42),
        0.4,
        0.0,
        None,
        None,
        Some(normal_map_path),
    )?;
    metal_material.normal_strength = normal_strength;
    let metal = scene.add_material(Material::ConductorGgx(metal_material));

    let sphere = load_gltf(Path::new("assets/gltf/sphere.glb"))?;
    let sphere_scale = uniform_scale_for_height(&sphere, 1.05);
    let sphere_pivot = Vec3::new(
        sphere.bounds.center().x,
        sphere.bounds.min.y,
        sphere.bounds.center().z,
    );
    let sphere_mesh = scene.add_mesh(sphere);

    let lambert_transform = Mat4::from_translation(Vec3::new(-0.72, 0.0, 0.35))
        * Mat4::from_quat(game_rotation_degrees(0.0, -25.0, 0.0))
        * Mat4::from_scale(Vec3::splat(sphere_scale))
        * Mat4::from_translation(-sphere_pivot);
    scene.add_instance(sphere_mesh, lambert, lambert_transform);

    let metal_transform = Mat4::from_translation(Vec3::new(0.72, 0.0, -0.85))
        * Mat4::from_quat(game_rotation_degrees(0.0, 25.0, 0.0))
        * Mat4::from_scale(Vec3::splat(sphere_scale))
        * Mat4::from_translation(-sphere_pivot);
    scene.add_instance(sphere_mesh, metal, metal_transform);

    let camera = PinholeCamera::new(
        Vec3::new(0.0, 2.05, 7.0),
        Vec3::new(0.0, 0.95, -0.25),
        Vec3::Y,
        38.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}
