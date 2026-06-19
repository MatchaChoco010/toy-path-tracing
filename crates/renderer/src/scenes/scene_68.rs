//! Cornell box に遮蔽壁と強い球ライトを置くBDPT検証用シーン。

use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{
    material::{EmissiveMaterial, Material, NormalizedLambertMaterial},
    scene::PinholeCamera,
    scene::Scene,
    scene::load_gltf,
};

pub fn create_scene_68(
    _ocio: &crate::color::OcioColorPipeline,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();
    let wall_gray = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(Vec3::splat(0.60)),
    ));
    let red = scene.add_material(Material::NormalizedLambert(NormalizedLambertMaterial::new(
        Vec3::new(0.63, 0.08, 0.05),
    )));
    let green = scene.add_material(Material::NormalizedLambert(NormalizedLambertMaterial::new(
        Vec3::new(0.14, 0.45, 0.091),
    )));
    let sphere_light = scene.add_material(Material::Emissive(EmissiveMaterial::new(
        Vec3::ONE,
        20_000.0,
    )));

    let floor_mesh = scene.add_mesh(load_gltf(Path::new("assets/models/floor.glb"))?);
    scene.add_instance(floor_mesh, wall_gray, Mat4::IDENTITY);
    let ceiling_mesh = scene.add_mesh(load_gltf(Path::new("assets/models/ceiling.glb"))?);
    scene.add_instance(ceiling_mesh, wall_gray, Mat4::IDENTITY);
    let back_wall_mesh = scene.add_mesh(load_gltf(Path::new("assets/models/back-wall.glb"))?);
    scene.add_instance(back_wall_mesh, wall_gray, Mat4::IDENTITY);
    let left_wall_mesh = scene.add_mesh(load_gltf(Path::new("assets/models/left-wall.glb"))?);
    scene.add_instance(left_wall_mesh, red, Mat4::IDENTITY);
    let right_wall_mesh = scene.add_mesh(load_gltf(Path::new("assets/models/right-wall.glb"))?);
    scene.add_instance(right_wall_mesh, green, Mat4::IDENTITY);
    let middle_wall_mesh = scene.add_mesh(load_gltf(Path::new("assets/models/middle-wall.glb"))?);
    scene.add_instance(middle_wall_mesh, wall_gray, Mat4::IDENTITY);
    let ico_sphere_mesh = scene.add_mesh(load_gltf(Path::new("assets/models/ico-sphere.glb"))?);
    scene.add_instance(ico_sphere_mesh, sphere_light, Mat4::IDENTITY);

    let camera = PinholeCamera::new(
        Vec3::new(0.0, 2.15, 7.1),
        Vec3::new(0.0, 1.45, -0.05),
        Vec3::Y,
        38.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}
