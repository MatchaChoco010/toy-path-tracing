use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{
    camera::PinholeCamera,
    light::EnvironmentLight,
    material::{ConductorGgxMaterial, DielectricGgxMaterial, Material},
    mesh::load_gltf,
    scene::Scene,
};

use super::uniform_scale_for_height;

pub fn create_scene_10() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let sphere = load_gltf(Path::new("assets/gltf/sphere.glb"))?;
    let sphere_height = 0.9;
    let sphere_scale = uniform_scale_for_height(&sphere, sphere_height);
    let sphere_pivot = Vec3::new(
        sphere.bounds.center().x,
        sphere.bounds.min.y,
        sphere.bounds.center().z,
    );
    let sphere_mesh_index = scene.add_mesh(sphere);

    let sphere_x = [-3.0_f32, -1.8, -0.6, 0.6, 1.8, 3.0];
    let roughness_values = [0.0_f32, 0.15, 0.30, 0.45, 0.60, 0.75];
    let silver_base_color = Vec3::splat(0.92);
    let glass_color = Vec3::ONE;
    let glass_eta = 1.5;

    let metal_y = -sphere_height * 0.5 - 0.15;
    let glass_y = sphere_height * 0.5 + 0.15;

    for (&x, &roughness) in sphere_x.iter().zip(roughness_values.iter()) {
        let metal_material = scene.add_material(Material::ConductorGgx(ConductorGgxMaterial::new(
            silver_base_color,
            roughness,
            0.0,
        )));
        let metal_transform = Mat4::from_translation(Vec3::new(x, metal_y, 0.0))
            * Mat4::from_scale(Vec3::splat(sphere_scale))
            * Mat4::from_translation(-sphere_pivot);
        scene.add_instance(sphere_mesh_index, metal_material, metal_transform);

        let glass_material = scene.add_material(Material::DielectricGgx(
            DielectricGgxMaterial::new(glass_color, glass_eta, roughness, 0.0, false),
        ));
        let glass_transform = Mat4::from_translation(Vec3::new(x, glass_y, 0.0))
            * Mat4::from_scale(Vec3::splat(sphere_scale))
            * Mat4::from_translation(-sphere_pivot);
        scene.add_instance(sphere_mesh_index, glass_material, glass_transform);
    }

    let env_width = 64;
    let env_height = 32;
    let env_pixels = vec![Vec3::ONE; env_width * env_height];
    let env = EnvironmentLight::from_pixels(env_width, env_height, env_pixels, 1.0, 0.0);
    scene.set_environment_light(env);

    let camera = PinholeCamera::new(
        Vec3::new(0.0, 0.0, 9.8),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::Y,
        42.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}
