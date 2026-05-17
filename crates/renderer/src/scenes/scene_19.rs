//! Lambert 床と puresky 環境光のもと、ノーマルマップ付き Lambert 球、Conductor GGX 球、Mirror 球を配置する。

use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{
    light::EnvironmentLight,
    material::{ConductorGgxMaterial, Material, MirrorMaterial, NormalizedLambertMaterial},
    scene::PinholeCamera,
    scene::Scene,
    scene::load_gltf,
};

use super::{game_rotation_degrees, uniform_scale_for_height};

pub fn create_scene_19(
    _ocio: &crate::color::OcioColorPipeline,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();
    let normal_strength = 0.2;

    let floor_material = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(Vec3::splat(0.62)),
    ));
    let floor_mesh = scene.add_mesh(load_gltf(Path::new("assets/models/floor.glb"))?);
    let floor_transform = Mat4::from_scale(Vec3::new(14.0, 1.0, 14.0));
    scene.add_instance(floor_mesh, floor_material, floor_transform);

    let normal_map_path = Path::new("assets/models/sphere-normal.png");
    let mut metal_material = ConductorGgxMaterial::try_new_with_texture_paths(
        Vec3::new(0.95, 0.78, 0.42),
        0.4,
        0.0,
        None,
        None,
        Some(normal_map_path),
        _ocio,
    )?;
    metal_material.normal_strength = normal_strength;
    let metal = scene.add_material(Material::ConductorGgx(metal_material));

    let mut lambert_material = NormalizedLambertMaterial::try_new_with_texture_path(
        Vec3::new(0.56, 0.72, 0.92),
        None,
        Some(normal_map_path),
        _ocio,
    )?;
    lambert_material.normal_strength = normal_strength;
    let lambert = scene.add_material(Material::NormalizedLambert(lambert_material));

    let mut mirror_material = MirrorMaterial::try_new_with_texture_path(
        Vec3::splat(0.92),
        None,
        Some(normal_map_path),
        _ocio,
    )?;
    mirror_material.normal_strength = normal_strength;
    let mirror = scene.add_material(Material::Mirror(mirror_material));

    let sphere = load_gltf(Path::new("assets/models/sphere.glb"))?;
    let sphere_scale = uniform_scale_for_height(&sphere, 1.15);
    let sphere_pivot = Vec3::new(
        sphere.bounds.center().x,
        sphere.bounds.min.y,
        sphere.bounds.center().z,
    );
    let sphere_mesh = scene.add_mesh(sphere);

    let metal_transform = Mat4::from_translation(Vec3::new(-1.45, 0.0, 0.35))
        * Mat4::from_quat(game_rotation_degrees(0.0, -35.0, 0.0))
        * Mat4::from_scale(Vec3::splat(sphere_scale))
        * Mat4::from_translation(-sphere_pivot);
    scene.add_instance(sphere_mesh, metal, metal_transform);

    let lambert_transform = Mat4::from_translation(Vec3::new(0.0, 0.0, -0.15))
        * Mat4::from_quat(game_rotation_degrees(0.0, 20.0, 0.0))
        * Mat4::from_scale(Vec3::splat(sphere_scale))
        * Mat4::from_translation(-sphere_pivot);
    scene.add_instance(sphere_mesh, lambert, lambert_transform);

    let mirror_transform = Mat4::from_translation(Vec3::new(1.45, 0.0, 0.35))
        * Mat4::from_quat(game_rotation_degrees(0.0, 35.0, 0.0))
        * Mat4::from_scale(Vec3::splat(sphere_scale))
        * Mat4::from_translation(-sphere_pivot);
    scene.add_instance(sphere_mesh, mirror, mirror_transform);

    let env = EnvironmentLight::from_hdr_file(
        "assets/sky/kloofendal_48d_partly_cloudy_puresky_4k.hdr",
        0.5,
        0.0,
    )?;
    scene.set_environment_light(env);

    let camera = PinholeCamera::new(
        Vec3::new(3.4, 1.55, -5.4),
        Vec3::new(0.0, 0.55, 0.10),
        Vec3::Y,
        40.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}
