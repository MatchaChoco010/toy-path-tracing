//! 一様な白い環境光のもと、SS Conductor / MS Conductor を 9 列 x 2 段で並べ、roughness をスイープする。

use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{
    light::EnvironmentLight,
    material::{ConductorGgxCui2023Material, ConductorGgxMaterial, Material},
    scene::PinholeCamera,
    scene::Scene,
    scene::load_gltf,
};

use super::uniform_scale_for_height;

pub fn create_scene_35(
    _ocio: &crate::color::OcioColorPipeline,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
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
    let row_gap = 2.4_f32;
    let single_row_y = row_gap * 0.5 - sphere_height * 0.5;
    let multi_row_y = -row_gap * 0.5 - sphere_height * 0.5;

    for i in 0..sphere_count {
        let roughness = i as f32 / (sphere_count as f32 - 1.0);
        let x = (i as f32 - center_offset) * spacing;

        let single_material = scene.add_material(Material::ConductorGgx(
            ConductorGgxMaterial::new(Vec3::ONE, roughness, 0.0),
        ));
        let single_transform = Mat4::from_translation(Vec3::new(x, single_row_y, 0.0))
            * Mat4::from_scale(Vec3::splat(sphere_scale))
            * Mat4::from_translation(-sphere_pivot);
        scene.add_instance(sphere_mesh, single_material, single_transform);

        let multi_material = scene.add_material(Material::ConductorGgxCui2023(
            ConductorGgxCui2023Material::new(Vec3::ONE, roughness, 0.0),
        ));
        let multi_transform = Mat4::from_translation(Vec3::new(x, multi_row_y, 0.0))
            * Mat4::from_scale(Vec3::splat(sphere_scale))
            * Mat4::from_translation(-sphere_pivot);
        scene.add_instance(sphere_mesh, multi_material, multi_transform);
    }

    let env_width = 64;
    let env_height = 32;
    let env_pixels = vec![Vec3::ONE; env_width * env_height];
    let env = EnvironmentLight::from_pixels(env_width, env_height, env_pixels, 1.0, 0.0);
    scene.set_environment_light(env);

    let camera = PinholeCamera::new(
        Vec3::new(0.0, 0.0, 7.5),
        Vec3::ZERO,
        Vec3::Y,
        42.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}
