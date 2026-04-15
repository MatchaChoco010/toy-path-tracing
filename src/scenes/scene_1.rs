use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{
    camera::PinholeCamera,
    material::{EmissiveMaterial, Material, NormalizedLambertMaterial},
    mesh::load_mesh,
    scene::Scene,
};

use super::{game_rotation_degrees, uniform_scale_for_height};

pub fn create_scene_1() -> Result<(Scene, crate::camera::PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();
    let wall_gray = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(Vec3::splat(0.60)),
    ));
    let object_gray = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(Vec3::splat(0.75)),
    ));
    let red = scene.add_material(Material::NormalizedLambert(NormalizedLambertMaterial::new(
        Vec3::new(0.63, 0.08, 0.05),
    )));
    let green = scene.add_material(Material::NormalizedLambert(NormalizedLambertMaterial::new(
        Vec3::new(0.14, 0.45, 0.091),
    )));
    let light = scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 20.0)));

    let floor_mesh_index = scene.add_mesh(load_mesh(Path::new("assets/gltf/floor.glb"))?);
    scene.add_instance(floor_mesh_index, wall_gray, Mat4::IDENTITY);
    let ceiling_mesh_index = scene.add_mesh(load_mesh(Path::new("assets/gltf/ceiling.glb"))?);
    scene.add_instance(ceiling_mesh_index, wall_gray, Mat4::IDENTITY);
    let back_wall_mesh_index = scene.add_mesh(load_mesh(Path::new("assets/gltf/back-wall.glb"))?);
    scene.add_instance(back_wall_mesh_index, wall_gray, Mat4::IDENTITY);
    let left_wall_mesh_index = scene.add_mesh(load_mesh(Path::new("assets/gltf/left-wall.glb"))?);
    scene.add_instance(left_wall_mesh_index, red, Mat4::IDENTITY);
    let right_wall_mesh_index = scene.add_mesh(load_mesh(Path::new("assets/gltf/right-wall.glb"))?);
    scene.add_instance(right_wall_mesh_index, green, Mat4::IDENTITY);
    let light_mesh_index = scene.add_mesh(load_mesh(Path::new("assets/gltf/light.glb"))?);
    scene.add_instance(light_mesh_index, light, Mat4::IDENTITY);

    let bunny = load_mesh(Path::new("assets/gltf/bunny.glb"))?;
    let bunny_scale = uniform_scale_for_height(&bunny, 1.38);
    let bunny_pivot = Vec3::new(
        bunny.bounds.center().x,
        bunny.bounds.min.y,
        bunny.bounds.center().z,
    );
    let bunny_mesh_index = scene.add_mesh(bunny);

    let sphere = load_mesh(Path::new("assets/gltf/sphere.glb"))?;
    let sphere_scale = uniform_scale_for_height(&sphere, 0.90);
    let sphere_pivot = Vec3::new(
        sphere.bounds.center().x,
        sphere.bounds.min.y,
        sphere.bounds.center().z,
    );
    let sphere_mesh_index = scene.add_mesh(sphere);

    let bunny_transform = Mat4::from_translation(Vec3::new(-0.78, 0.0, -0.30))
        * Mat4::from_quat(game_rotation_degrees(0.0, 22.0, 0.0))
        * Mat4::from_scale(Vec3::splat(bunny_scale))
        * Mat4::from_translation(-bunny_pivot);
    scene.add_instance(bunny_mesh_index, object_gray, bunny_transform);

    let sphere_0_transform = Mat4::from_translation(Vec3::new(0.92, 0.0, -0.78))
        * Mat4::from_scale(Vec3::splat(sphere_scale * 1.10))
        * Mat4::from_translation(-sphere_pivot);
    scene.add_instance(sphere_mesh_index, object_gray, sphere_0_transform);

    let sphere_1_transform = Mat4::from_translation(Vec3::new(0.35, 0.0, 0.98))
        * Mat4::from_scale(Vec3::splat(sphere_scale * 0.78))
        * Mat4::from_translation(-sphere_pivot);
    scene.add_instance(sphere_mesh_index, object_gray, sphere_1_transform);

    let camera = PinholeCamera::new(
        Vec3::new(0.0, 2.10, 7.0),
        Vec3::new(0.0, 1.40, -0.05),
        Vec3::Y,
        38.0_f32.to_radians(),
    );

    Ok((scene, camera))
}
