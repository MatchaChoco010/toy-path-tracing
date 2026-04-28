use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{
    camera::PinholeCamera,
    material::{ConductorGgxMaterial, EmissiveMaterial, Material, NormalizedLambertMaterial},
    mesh::load_gltf,
    scene::Scene,
};

use super::uniform_scale_for_height;

pub fn create_scene_4() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
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

    let floor_mesh_index = scene.add_mesh(load_gltf(Path::new("assets/gltf/floor.glb"))?);
    scene.add_instance(floor_mesh_index, wall_gray, Mat4::IDENTITY);
    let ceiling_mesh_index = scene.add_mesh(load_gltf(Path::new("assets/gltf/ceiling.glb"))?);
    scene.add_instance(ceiling_mesh_index, wall_gray, Mat4::IDENTITY);
    let back_wall_mesh_index = scene.add_mesh(load_gltf(Path::new("assets/gltf/back-wall.glb"))?);
    scene.add_instance(back_wall_mesh_index, wall_gray, Mat4::IDENTITY);
    let left_wall_mesh_index = scene.add_mesh(load_gltf(Path::new("assets/gltf/left-wall.glb"))?);
    scene.add_instance(left_wall_mesh_index, red, Mat4::IDENTITY);
    let right_wall_mesh_index = scene.add_mesh(load_gltf(Path::new("assets/gltf/right-wall.glb"))?);
    scene.add_instance(right_wall_mesh_index, green, Mat4::IDENTITY);
    let light_mesh_index = scene.add_mesh(load_gltf(Path::new("assets/gltf/light.glb"))?);
    scene.add_instance(light_mesh_index, light, Mat4::IDENTITY);

    let gold_base_color = Vec3::new(1.00, 0.76, 0.34);
    let conductor_materials = [
        scene.add_material(Material::ConductorGgx(ConductorGgxMaterial::new(
            gold_base_color,
            0.0,
            0.0,
        ))),
        scene.add_material(Material::ConductorGgx(ConductorGgxMaterial::new(
            gold_base_color,
            0.25,
            0.0,
        ))),
        scene.add_material(Material::ConductorGgx(ConductorGgxMaterial::new(
            gold_base_color,
            0.5,
            0.0,
        ))),
        scene.add_material(Material::ConductorGgx(ConductorGgxMaterial::new(
            gold_base_color,
            0.75,
            0.0,
        ))),
        scene.add_material(Material::ConductorGgx(ConductorGgxMaterial::new(
            gold_base_color,
            1.0,
            0.0,
        ))),
    ];

    let sphere = load_gltf(Path::new("assets/gltf/sphere.glb"))?;
    let sphere_scale = uniform_scale_for_height(&sphere, 0.78);
    let sphere_pivot = Vec3::new(
        sphere.bounds.center().x,
        sphere.bounds.min.y,
        sphere.bounds.center().z,
    );
    let sphere_mesh_index = scene.add_mesh(sphere);

    let sphere_positions = [-1.55, -0.78, 0.0, 0.78, 1.55];
    for (material_index, x) in conductor_materials.into_iter().zip(sphere_positions) {
        let transform = Mat4::from_translation(Vec3::new(x, 0.0, -0.35))
            * Mat4::from_scale(Vec3::splat(sphere_scale))
            * Mat4::from_translation(-sphere_pivot);
        scene.add_instance(sphere_mesh_index, material_index, transform);
    }

    let camera = PinholeCamera::new(
        Vec3::new(0.0, 2.05, 7.35),
        Vec3::new(0.0, 1.15, -0.35),
        Vec3::Y,
        38.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}
