//! HDRI 環境下に sheen=1 の Disney BRDF 球を配置する。

use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{
    light::EnvironmentLight,
    material::{DisneyBrdfMaterial, Material},
    scene::PinholeCamera,
    scene::Scene,
    scene::load_gltf,
};

pub fn create_scene_28(
    _ocio: &crate::color::OcioColorPipeline,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let sphere = load_gltf(Path::new("assets/models/sphere.glb"))?;
    let sphere_extent = sphere.bounds.extent();
    let sphere_target_diameter = 1.6_f32;
    let sphere_scale = sphere_target_diameter / sphere_extent.y.max(1.0e-3);
    let sphere_center = sphere.bounds.center();
    let sphere_mesh = scene.add_mesh(sphere);

    let mut material = DisneyBrdfMaterial::new(Vec3::new(0.5, 0.15, 0.05))
        .with_roughness(0.7)
        .with_sheen(1.0)
        .with_sheen_tint(0.0);
    material.specular = 0.0;
    let material_id = scene.add_material(Material::DisneyBrdf(material));

    let transform =
        Mat4::from_scale(Vec3::splat(sphere_scale)) * Mat4::from_translation(-sphere_center);
    scene.add_instance(sphere_mesh, material_id, transform);

    let env = EnvironmentLight::from_hdr_file(
        "assets/sky/studio_small_08_4k.hdr",
        1.0,
        std::f32::consts::PI * 0.5,
    )?;
    scene.set_environment_light(env);

    let camera_eye = Vec3::new(0.0, 0.0, 3.6);
    let camera_target = Vec3::ZERO;
    let camera = PinholeCamera::new(
        camera_eye,
        camera_target,
        Vec3::Y,
        30.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}
