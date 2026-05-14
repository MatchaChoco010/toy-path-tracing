//! HDRI のもと、ゴールドの Conductor GGX 球を上段 compensation OFF / 下段 ON で roughness スイープする。

use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{
    camera::PinholeCamera,
    light::EnvironmentLight,
    material::{ConductorGgxMaterial, Material},
    mesh::load_gltf,
    scene::Scene,
};

use super::uniform_scale_for_height;

pub fn create_scene_39() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
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
    let ss_y = row_gap * 0.5;
    let ms_y = -row_gap * 0.5;
    let gold = Vec3::new(1.00, 0.78, 0.34);

    for i in 0..sphere_count {
        let roughness = i as f32 / (sphere_count as f32 - 1.0);
        let x = (i as f32 - center_offset) * spacing;

        let ss_index = scene.add_material(Material::ConductorGgx(ConductorGgxMaterial::new(
            gold, roughness, 0.0,
        )));
        let ss_transform = Mat4::from_translation(Vec3::new(x, ss_y, 0.0))
            * Mat4::from_scale(Vec3::splat(sphere_scale))
            * Mat4::from_translation(-sphere_pivot);
        scene.add_instance(sphere_mesh, ss_index, ss_transform);

        let ms_index = scene.add_material(Material::ConductorGgx(
            ConductorGgxMaterial::new(gold, roughness, 0.0).with_energy_compensation(),
        ));
        let ms_transform = Mat4::from_translation(Vec3::new(x, ms_y, 0.0))
            * Mat4::from_scale(Vec3::splat(sphere_scale))
            * Mat4::from_translation(-sphere_pivot);
        scene.add_instance(sphere_mesh, ms_index, ms_transform);
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
