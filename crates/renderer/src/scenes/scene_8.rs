//! Lambert 床と HDRI 環境光のもと、Conductor GGX 金属球と Dielectric GGX ガラス球を 2 段で roughness スイープする。

use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{
    camera::PinholeCamera,
    light::EnvironmentLight,
    material::{ConductorGgxMaterial, DielectricGgxMaterial, Material, NormalizedLambertMaterial},
    mesh::load_gltf,
    scene::Scene,
};

use super::uniform_scale_for_height;

pub fn create_scene_8() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let floor_material = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(Vec3::splat(0.55)),
    ));
    let floor_mesh_index = scene.add_mesh(load_gltf(Path::new("assets/models/floor.glb"))?);
    let floor_transform = Mat4::from_scale(Vec3::new(6.0, 1.0, 6.0));
    scene.add_instance(floor_mesh_index, floor_material, floor_transform);

    let sphere = load_gltf(Path::new("assets/models/sphere.glb"))?;
    let sphere_height = 0.6;
    let sphere_scale = uniform_scale_for_height(&sphere, sphere_height);
    let sphere_pivot = Vec3::new(
        sphere.bounds.center().x,
        sphere.bounds.min.y,
        sphere.bounds.center().z,
    );
    let sphere_mesh_index = scene.add_mesh(sphere);

    let sphere_x = [-2.0_f32, -1.2, -0.4, 0.4, 1.2, 2.0];
    let roughness_values = [0.0_f32, 0.15, 0.30, 0.45, 0.60, 0.75];
    let metal_base_color = Vec3::new(0.95, 0.82, 0.45);
    let glass_color = Vec3::ONE;
    let glass_eta = 1.5;

    let metal_lift = 0.0;
    let glass_lift = sphere_height + 0.3;

    for (&x, &roughness) in sphere_x.iter().zip(roughness_values.iter()) {
        let metal_material = scene.add_material(Material::ConductorGgx(ConductorGgxMaterial::new(
            metal_base_color,
            roughness,
            0.0,
        )));
        let metal_transform = Mat4::from_translation(Vec3::new(x, metal_lift, 0.0))
            * Mat4::from_scale(Vec3::splat(sphere_scale))
            * Mat4::from_translation(-sphere_pivot);
        scene.add_instance(sphere_mesh_index, metal_material, metal_transform);

        let glass_material = scene.add_material(Material::DielectricGgx(
            DielectricGgxMaterial::new(glass_color, glass_eta, roughness, 0.0, false),
        ));
        let glass_transform = Mat4::from_translation(Vec3::new(x, glass_lift, 0.0))
            * Mat4::from_scale(Vec3::splat(sphere_scale))
            * Mat4::from_translation(-sphere_pivot);
        scene.add_instance(sphere_mesh_index, glass_material, glass_transform);
    }

    let env = EnvironmentLight::from_hdr_file("assets/sky/brown_photostudio_02_4k.hdr", 1.0, 0.0)?;
    scene.set_environment_light(env);

    let camera = PinholeCamera::new(
        Vec3::new(0.0, 1.60, 7.0),
        Vec3::new(0.0, 0.60, 0.0),
        Vec3::Y,
        42.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}
