use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{
    camera::PinholeCamera,
    light::EnvironmentLight,
    material::{ConductorGgxMaterial, EmissiveMaterial, Material, NormalizedLambertMaterial},
    mesh::load_gltf,
    scene::Scene,
};

use super::{game_rotation_degrees, uniform_scale_for_height};

pub fn create_scene_9() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
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
    let light = scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 10.0)));

    let floor_mesh_index = scene.add_mesh(load_gltf(Path::new("assets/models/floor.glb"))?);
    scene.add_instance(floor_mesh_index, wall_gray, Mat4::IDENTITY);
    let ceiling_mesh_index = scene.add_mesh(load_gltf(Path::new("assets/models/ceiling.glb"))?);
    scene.add_instance(ceiling_mesh_index, wall_gray, Mat4::IDENTITY);
    let back_wall_mesh_index = scene.add_mesh(load_gltf(Path::new("assets/models/back-wall.glb"))?);
    scene.add_instance(back_wall_mesh_index, wall_gray, Mat4::IDENTITY);
    let left_wall_mesh_index = scene.add_mesh(load_gltf(Path::new("assets/models/left-wall.glb"))?);
    scene.add_instance(left_wall_mesh_index, red, Mat4::IDENTITY);
    let right_wall_mesh_index =
        scene.add_mesh(load_gltf(Path::new("assets/models/right-wall.glb"))?);
    scene.add_instance(right_wall_mesh_index, green, Mat4::IDENTITY);
    let light_mesh_index = scene.add_mesh(load_gltf(Path::new("assets/models/light.glb"))?);
    scene.add_instance(light_mesh_index, light, Mat4::IDENTITY);

    let bunny = load_gltf(Path::new("assets/models/bunny.glb"))?;
    let bunny_scale = uniform_scale_for_height(&bunny, 2.50);
    let bunny_pivot = Vec3::new(
        bunny.bounds.center().x,
        bunny.bounds.min.y,
        bunny.bounds.center().z,
    );
    let bunny_mesh_index = scene.add_mesh(bunny);

    let metal_material = scene.add_material(Material::ConductorGgx(ConductorGgxMaterial::new(
        Vec3::new(0.95, 0.78, 0.42),
        0.35,
        0.0,
    )));

    let bunny_transform = Mat4::from_translation(Vec3::new(0.0, 0.0, -0.2))
        * Mat4::from_quat(game_rotation_degrees(0.0, -25.0, 0.0))
        * Mat4::from_scale(Vec3::splat(bunny_scale))
        * Mat4::from_translation(-bunny_pivot);
    scene.add_instance(bunny_mesh_index, metal_material, bunny_transform);

    let env = EnvironmentLight::from_hdr_file("assets/sky/brown_photostudio_02_4k.hdr", 1.0, 0.0)?;
    scene.set_environment_light(env);

    let camera = PinholeCamera::new(
        Vec3::new(0.0, 2.10, 9.5),
        Vec3::new(0.0, 1.60, 0.0),
        Vec3::Y,
        42.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}
