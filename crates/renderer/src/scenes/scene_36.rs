//! HDRI のもと、compensation OFF の Conductor GGX と Dielectric GGX を 9 列 x 2 段で roughness スイープする。

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

pub fn create_scene_36() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    create_metal_glass_roughness_rows(false)
}

pub(super) fn create_metal_glass_roughness_rows(
    energy_compensation: bool,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let sphere = load_gltf(Path::new("assets/models/sphere.glb"))?;
    let sphere_height = 0.55_f32;
    let sphere_scale = uniform_scale_for_height(&sphere, sphere_height);
    let sphere_pivot = Vec3::new(
        sphere.bounds.center().x,
        sphere.bounds.center().y,
        sphere.bounds.center().z,
    );
    let sphere_mesh = scene.add_mesh(sphere);

    let sphere_count = 9_usize;
    let spacing = 0.65_f32;
    let row_gap = 0.7_f32;
    let center_offset = (sphere_count as f32 - 1.0) * 0.5;
    let metal_y = row_gap * 0.5;
    let glass_y = -row_gap * 0.5;
    let silver = Vec3::splat(0.92);
    let glass_color = Vec3::ONE;
    let glass_eta = 1.5;

    for i in 0..sphere_count {
        let roughness = i as f32 / (sphere_count as f32 - 1.0);
        let x = (i as f32 - center_offset) * spacing;

        let mut metal = ConductorGgxMaterial::new(silver, roughness, 0.0);
        if energy_compensation {
            metal = metal.with_energy_compensation();
        }
        let metal_index = scene.add_material(Material::ConductorGgx(metal));
        let metal_transform = Mat4::from_translation(Vec3::new(x, metal_y, 0.0))
            * Mat4::from_scale(Vec3::splat(sphere_scale))
            * Mat4::from_translation(-sphere_pivot);
        scene.add_instance(sphere_mesh, metal_index, metal_transform);

        let mut glass = DielectricGgxMaterial::new(glass_color, glass_eta, roughness, 0.0, false);
        if energy_compensation {
            glass = glass.with_energy_compensation();
        }
        let glass_index = scene.add_material(Material::DielectricGgx(glass));
        let glass_transform = Mat4::from_translation(Vec3::new(x, glass_y, 0.0))
            * Mat4::from_scale(Vec3::splat(sphere_scale))
            * Mat4::from_translation(-sphere_pivot);
        scene.add_instance(sphere_mesh, glass_index, glass_transform);
    }

    let env = EnvironmentLight::from_hdr_file(
        "assets/sky/brown_photostudio_02_4k.hdr",
        0.6,
        std::f32::consts::PI,
    )?;
    scene.set_environment_light(env);

    let camera_eye = Vec3::new(0.0, 0.0, 8.0);
    let camera_target = Vec3::ZERO;
    let camera = PinholeCamera::new(
        camera_eye,
        camera_target,
        Vec3::Y,
        32.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}
