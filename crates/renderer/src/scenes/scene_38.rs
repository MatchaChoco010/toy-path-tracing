//! 一様な白い環境光のもと、SS/MS の Conductor と Dielectric を 9 列 x 4 段で roughness スイープする。

use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{
    light::EnvironmentLight,
    material::{ConductorGgxMaterial, DielectricGgxMaterial, Material},
    scene::PinholeCamera,
    scene::Scene,
    scene::load_gltf,
};

use super::uniform_scale_for_height;

pub fn create_scene_38(
    _ocio: &crate::color::OcioColorPipeline,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let sphere = load_gltf(Path::new("assets/models/sphere.glb"))?;
    let sphere_height = 0.5_f32;
    let sphere_scale = uniform_scale_for_height(&sphere, sphere_height);
    let sphere_pivot = Vec3::new(
        sphere.bounds.center().x,
        sphere.bounds.center().y,
        sphere.bounds.center().z,
    );
    let sphere_mesh = scene.add_mesh(sphere);

    let sphere_count = 9_usize;
    let spacing = 0.6_f32;
    let row_gap = 0.85_f32;
    let center_offset = (sphere_count as f32 - 1.0) * 0.5;

    let row_ys = [row_gap * 1.5, row_gap * 0.5, -row_gap * 0.5, -row_gap * 1.5];

    let glass_eta = 1.5_f32;

    for i in 0..sphere_count {
        let roughness = i as f32 / (sphere_count as f32 - 1.0);
        let x = (i as f32 - center_offset) * spacing;

        let materials: [Material; 4] = [
            Material::ConductorGgx(ConductorGgxMaterial::new(Vec3::ONE, roughness, 0.0)),
            Material::DielectricGgx(DielectricGgxMaterial::new(
                Vec3::ONE,
                glass_eta,
                roughness,
                0.0,
                false,
            )),
            Material::ConductorGgx(
                ConductorGgxMaterial::new(Vec3::ONE, roughness, 0.0).with_energy_compensation(),
            ),
            Material::DielectricGgx(
                DielectricGgxMaterial::new(Vec3::ONE, glass_eta, roughness, 0.0, false)
                    .with_energy_compensation(),
            ),
        ];

        for (row_index, material) in materials.into_iter().enumerate() {
            let material_index = scene.add_material(material);
            let transform = Mat4::from_translation(Vec3::new(x, row_ys[row_index], 0.0))
                * Mat4::from_scale(Vec3::splat(sphere_scale))
                * Mat4::from_translation(-sphere_pivot);
            scene.add_instance(sphere_mesh, material_index, transform);
        }
    }

    let env_width = 64;
    let env_height = 32;
    let env_pixels = vec![Vec3::ONE; env_width * env_height];
    let env = EnvironmentLight::from_pixels(env_width, env_height, env_pixels, 1.0, 0.0);
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
