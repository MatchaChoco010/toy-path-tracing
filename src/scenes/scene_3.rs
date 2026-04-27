use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{
    camera::PinholeCamera,
    material::{EmissiveMaterial, GlassMaterial, Material, NormalizedLambertMaterial},
    mesh::load_gltf,
    scene::Scene,
};

use super::{game_rotation_degrees, uniform_scale_for_height};

pub fn create_scene_3() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
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
    let pale_blue = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(Vec3::new(0.28, 0.52, 0.88)),
    ));
    let clear_glass =
        scene.add_material(Material::Glass(GlassMaterial::new(1.5, Vec3::ONE, false)));
    let aqua_thin_glass = scene.add_material(Material::Glass(GlassMaterial::new(
        1.5,
        Vec3::new(0.80, 0.93, 1.00),
        true,
    )));
    let aqua_solid_glass = scene.add_material(Material::Glass(GlassMaterial::new(
        1.5,
        Vec3::new(0.80, 0.93, 1.00),
        false,
    )));
    let light = scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 20.0)));

    let floor_mesh_index = scene.add_mesh(load_gltf(Path::new("assets/gltf/floor.glb"))?);
    scene.add_instance(floor_mesh_index, wall_gray, Mat4::IDENTITY);
    let ceiling_mesh_index = scene.add_mesh(load_gltf(Path::new("assets/gltf/ceiling.glb"))?);
    scene.add_instance(ceiling_mesh_index, wall_gray, Mat4::IDENTITY);
    let back_wall_mesh_index = scene.add_mesh(load_gltf(Path::new("assets/gltf/back-wall.glb"))?);
    scene.add_instance(back_wall_mesh_index, wall_gray, Mat4::IDENTITY);
    let left_wall_mesh_index = scene.add_mesh(load_gltf(Path::new("assets/gltf/left-wall.glb"))?);
    scene.add_instance(left_wall_mesh_index, red, Mat4::IDENTITY);
    let right_wall_mesh_index = scene.add_mesh(load_gltf(Path::new("assets/gltf/right-wall.glb"))?);
    scene.add_instance(right_wall_mesh_index, green, Mat4::IDENTITY);
    let light_mesh_index = scene.add_mesh(load_gltf(Path::new("assets/gltf/light.glb"))?);
    scene.add_instance(light_mesh_index, light, Mat4::IDENTITY);

    let bunny = load_gltf(Path::new("assets/gltf/bunny.glb"))?;
    let bunny_scale = uniform_scale_for_height(&bunny, 1.18);
    let bunny_pivot = Vec3::new(
        bunny.bounds.center().x,
        bunny.bounds.min.y,
        bunny.bounds.center().z,
    );
    let bunny_mesh_index = scene.add_mesh(bunny);

    let sphere = load_gltf(Path::new("assets/gltf/sphere.glb"))?;
    let sphere_scale = uniform_scale_for_height(&sphere, 1.00);
    let sphere_pivot = Vec3::new(
        sphere.bounds.center().x,
        sphere.bounds.min.y,
        sphere.bounds.center().z,
    );
    let sphere_mesh_index = scene.add_mesh(sphere);

    let thin_bunny_transform = Mat4::from_translation(Vec3::new(-1.28, 0.18, -0.42))
        * Mat4::from_quat(game_rotation_degrees(0.0, 18.0, 0.0))
        * Mat4::from_scale(Vec3::splat(bunny_scale))
        * Mat4::from_translation(-bunny_pivot);
    scene.add_instance(bunny_mesh_index, aqua_thin_glass, thin_bunny_transform);

    let background_bunny_transform = Mat4::from_translation(Vec3::new(0.0, 0.0, -0.95))
        * Mat4::from_quat(game_rotation_degrees(0.0, 180.0, 0.0))
        * Mat4::from_scale(Vec3::splat(bunny_scale * 1.24))
        * Mat4::from_translation(-bunny_pivot);
    scene.add_instance(bunny_mesh_index, pale_blue, background_bunny_transform);

    let sphere_transform = Mat4::from_translation(Vec3::new(0.0, 0.28, 0.78))
        * Mat4::from_scale(Vec3::splat(sphere_scale * 0.92))
        * Mat4::from_translation(-sphere_pivot);
    scene.add_instance(sphere_mesh_index, clear_glass, sphere_transform);

    let solid_bunny_transform = Mat4::from_translation(Vec3::new(1.28, 0.18, -0.42))
        * Mat4::from_quat(game_rotation_degrees(0.0, -18.0, 0.0))
        * Mat4::from_scale(Vec3::splat(bunny_scale))
        * Mat4::from_translation(-bunny_pivot);
    scene.add_instance(bunny_mesh_index, aqua_solid_glass, solid_bunny_transform);

    let camera = PinholeCamera::new(
        Vec3::new(0.0, 2.08, 7.0),
        Vec3::new(0.0, 1.28, -0.35),
        Vec3::Y,
        38.0_f32.to_radians(),
    );

    Ok((scene, camera))
}
