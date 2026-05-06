use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{
    camera::PinholeCamera,
    light::EnvironmentLight,
    material::{EonMaterial, Material, OrenNayarMaterial},
    mesh::load_gltf,
    scene::Scene,
};

use super::uniform_scale_for_height;

pub fn create_scene_42() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let sphere = load_gltf(Path::new("assets/models/sphere.glb"))?;
    let sphere_height = 0.6_f32;
    let sphere_scale = uniform_scale_for_height(&sphere, sphere_height);
    let sphere_pivot = Vec3::new(
        sphere.bounds.center().x,
        sphere.bounds.min.y,
        sphere.bounds.center().z,
    );
    let sphere_mesh = scene.add_mesh(sphere);

    let sphere_count = 9_usize;
    let spacing = 0.7_f32;
    let center_offset = (sphere_count as f32 - 1.0) * 0.5;
    let row_gap = 0.9_f32;
    let upper_row_y = row_gap * 0.5 - sphere_height * 0.5;
    let lower_row_y = -row_gap * 0.5 - sphere_height * 0.5;

    for i in 0..sphere_count {
        let roughness = i as f32 / (sphere_count as f32 - 1.0);
        let x = (i as f32 - center_offset) * spacing;

        let upper_material = scene.add_material(Material::OrenNayar(OrenNayarMaterial::new(
            Vec3::ONE,
            roughness,
        )));
        let upper_transform = Mat4::from_translation(Vec3::new(x, upper_row_y, 0.0))
            * Mat4::from_scale(Vec3::splat(sphere_scale))
            * Mat4::from_translation(-sphere_pivot);
        scene.add_instance(sphere_mesh, upper_material, upper_transform);

        let lower_material =
            scene.add_material(Material::Eon(EonMaterial::new(Vec3::ONE, roughness)));
        let lower_transform = Mat4::from_translation(Vec3::new(x, lower_row_y, 0.0))
            * Mat4::from_scale(Vec3::splat(sphere_scale))
            * Mat4::from_translation(-sphere_pivot);
        scene.add_instance(sphere_mesh, lower_material, lower_transform);
    }

    let env_width = 64;
    let env_height = 32;
    let env_pixels = vec![Vec3::ONE; env_width * env_height];
    let env = EnvironmentLight::from_pixels(env_width, env_height, env_pixels, 1.0, 0.0);
    scene.set_environment_light(env);

    let camera = PinholeCamera::new(
        Vec3::new(0.0, 0.0, 8.5),
        Vec3::ZERO,
        Vec3::Y,
        42.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}
