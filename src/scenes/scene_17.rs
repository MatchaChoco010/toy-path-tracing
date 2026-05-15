//! テクスチャ付き Lambert 床に完全鏡面の金属球とガラス球を置き、puresky HDRI で照らす。

use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{
    camera::PinholeCamera,
    light::EnvironmentLight,
    material::{GlassMaterial, Material, MirrorMaterial, NormalizedLambertMaterial},
    mesh::load_gltf,
    scene::Scene,
};

use super::uniform_scale_for_height;

pub fn create_scene_17() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let floor_material = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::try_new_with_texture_path(
            Vec3::ONE,
            Some(Path::new("assets/models/floor-brick.png")),
            None,
        )?,
    ));
    let floor_mesh = scene.add_mesh(load_gltf(Path::new("assets/models/floor.glb"))?);
    let floor_transform = Mat4::from_scale(Vec3::new(10.0, 1.0, 10.0));
    scene.add_instance(floor_mesh, floor_material, floor_transform);

    let mirror_metal = scene.add_material(Material::Mirror(MirrorMaterial::new(Vec3::splat(0.92))));
    let mirror_glass =
        scene.add_material(Material::Glass(GlassMaterial::new(1.5, Vec3::ONE, false)));

    let sphere = load_gltf(Path::new("assets/models/sphere.glb"))?;
    let sphere_scale = uniform_scale_for_height(&sphere, 1.2);
    let sphere_pivot = Vec3::new(
        sphere.bounds.center().x,
        sphere.bounds.min.y,
        sphere.bounds.center().z,
    );
    let sphere_mesh = scene.add_mesh(sphere);

    let metal_transform = Mat4::from_translation(Vec3::new(-0.85, 0.0, 0.15))
        * Mat4::from_scale(Vec3::splat(sphere_scale))
        * Mat4::from_translation(-sphere_pivot);
    scene.add_instance(sphere_mesh, mirror_metal, metal_transform);

    let glass_transform = Mat4::from_translation(Vec3::new(0.85, 0.0, -0.35))
        * Mat4::from_scale(Vec3::splat(sphere_scale))
        * Mat4::from_translation(-sphere_pivot);
    scene.add_instance(sphere_mesh, mirror_glass, glass_transform);

    let env = EnvironmentLight::from_hdr_file(
        "assets/sky/kloofendal_48d_partly_cloudy_puresky_4k.hdr",
        0.5,
        0.0,
    )?;
    scene.set_environment_light(env);

    let camera = PinholeCamera::new(
        Vec3::new(3.2, 1.35, -5.1),
        Vec3::new(0.05, 0.38, -0.10),
        Vec3::Y,
        40.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}
