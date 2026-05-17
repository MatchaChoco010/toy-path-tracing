//! mori-knob 風の床に NormalizedLambert と EON のオレンジ色の球を並べ、DirectionalLight で比較する。

use std::{error::Error, path::Path};

use glam::{Mat4, Vec3};

use crate::{
    light::DirectionalLight,
    material::{EonMaterial, Material, NormalizedLambertMaterial},
    scene::PinholeCamera,
    scene::Scene,
    scene::{load_gltf, load_obj},
};

use super::uniform_scale_for_height;

pub fn create_scene_44(
    _ocio: &crate::color::OcioColorPipeline,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let world_scale = 0.55_f32;
    let world_root = Mat4::from_scale(Vec3::splat(world_scale));

    let floor_mesh = scene.add_mesh(load_obj(Path::new("assets/mori-knob/floor.obj"))?);
    let floor_material = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(Vec3::splat(0.62)),
    ));
    scene.add_instance(floor_mesh, floor_material, world_root);

    let sphere = load_gltf(Path::new("assets/models/sphere.glb"))?;
    let sphere_height = 0.6_f32;
    let sphere_scale = uniform_scale_for_height(&sphere, sphere_height);
    let sphere_pivot = Vec3::new(
        sphere.bounds.center().x,
        sphere.bounds.min.y,
        sphere.bounds.center().z,
    );
    let sphere_mesh = scene.add_mesh(sphere);

    let orange = Vec3::new(0.95, 0.62, 0.45);
    let floor_top_y = -0.5 * world_scale;
    let float_offset = 0.12_f32;
    let sphere_center_y = floor_top_y + sphere_height * 0.5 + float_offset;
    let sphere_origin_y = floor_top_y + float_offset;
    let half_separation = 0.7_f32;

    // Camera looks toward +Z, so screen-left maps to world +X.
    let left_material = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(orange),
    ));
    let left_transform = Mat4::from_translation(Vec3::new(half_separation, sphere_origin_y, 0.0))
        * Mat4::from_scale(Vec3::splat(sphere_scale))
        * Mat4::from_translation(-sphere_pivot);
    scene.add_instance(sphere_mesh, left_material, left_transform);

    let right_material = scene.add_material(Material::Eon(EonMaterial::new(orange, 1.0)));
    let right_transform = Mat4::from_translation(Vec3::new(-half_separation, sphere_origin_y, 0.0))
        * Mat4::from_scale(Vec3::splat(sphere_scale))
        * Mat4::from_translation(-sphere_pivot);
    scene.add_instance(sphere_mesh, right_material, right_transform);

    let camera_distance = 5.0_f32;
    let camera_pitch = 15.0_f32.to_radians();
    let pitch_sin = camera_pitch.sin();
    let pitch_cos = camera_pitch.cos();
    let camera_target = Vec3::new(0.0, sphere_center_y, 0.0);
    let camera_eye = Vec3::new(
        0.0,
        sphere_center_y + camera_distance * pitch_sin,
        -camera_distance * pitch_cos,
    );

    scene.add_directional_light(DirectionalLight::new(
        Vec3::new(0.0, -pitch_sin, pitch_cos),
        Vec3::ONE,
        3.0,
    ));

    let camera = PinholeCamera::new(
        camera_eye,
        camera_target,
        Vec3::Y,
        20.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}
