use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{
    camera::PinholeCamera,
    light::{PointLight, SpotLight},
    material::{ConductorGgxMaterial, Material, NormalizedLambertMaterial},
    mesh::load_mesh,
    scene::Scene,
    scenes::{game_rotation_degrees, uniform_scale_for_height},
};

pub fn create_scene_12() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
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
    let bunny_blue = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(Vec3::new(0.55, 0.72, 0.92)),
    ));
    let gold = scene.add_material(Material::ConductorGgx(ConductorGgxMaterial::new(
        Vec3::new(0.95, 0.78, 0.35),
        0.35,
        0.0,
    )));
    let copper = scene.add_material(Material::ConductorGgx(ConductorGgxMaterial::new(
        Vec3::new(0.95, 0.64, 0.54),
        0.45,
        0.0,
    )));

    // Cornell box walls (no emissive ceiling light).
    let floor = scene.add_mesh(load_mesh(Path::new("assets/gltf/floor.glb"))?);
    scene.add_instance(floor, wall_gray, Mat4::IDENTITY);
    let ceiling = scene.add_mesh(load_mesh(Path::new("assets/gltf/ceiling.glb"))?);
    scene.add_instance(ceiling, wall_gray, Mat4::IDENTITY);
    let back_wall = scene.add_mesh(load_mesh(Path::new("assets/gltf/back-wall.glb"))?);
    scene.add_instance(back_wall, wall_gray, Mat4::IDENTITY);
    let left_wall = scene.add_mesh(load_mesh(Path::new("assets/gltf/left-wall.glb"))?);
    scene.add_instance(left_wall, red, Mat4::IDENTITY);
    let right_wall = scene.add_mesh(load_mesh(Path::new("assets/gltf/right-wall.glb"))?);
    scene.add_instance(right_wall, green, Mat4::IDENTITY);

    // Blue lambert bunny at the centre.
    let bunny = load_mesh(Path::new("assets/gltf/bunny.glb"))?;
    let bunny_scale = uniform_scale_for_height(&bunny, 1.7);
    let bunny_pivot = Vec3::new(
        bunny.bounds.center().x,
        bunny.bounds.min.y,
        bunny.bounds.center().z,
    );
    let bunny_mesh = scene.add_mesh(bunny);
    let bunny_transform = Mat4::from_translation(Vec3::new(0.0, 0.0, 0.6))
        * Mat4::from_quat(game_rotation_degrees(0.0, -15.0, 0.0))
        * Mat4::from_scale(Vec3::splat(bunny_scale))
        * Mat4::from_translation(-bunny_pivot);
    scene.add_instance(bunny_mesh, bunny_blue, bunny_transform);

    // Two rough metal spheres flanking the bunny.
    let sphere = load_mesh(Path::new("assets/gltf/sphere.glb"))?;
    let sphere_scale = uniform_scale_for_height(&sphere, 0.75);
    let sphere_pivot = Vec3::new(
        sphere.bounds.center().x,
        sphere.bounds.min.y,
        sphere.bounds.center().z,
    );
    let sphere_mesh = scene.add_mesh(sphere);
    let gold_transform = Mat4::from_translation(Vec3::new(-1.25, 0.0, 0.9))
        * Mat4::from_scale(Vec3::splat(sphere_scale))
        * Mat4::from_translation(-sphere_pivot);
    scene.add_instance(sphere_mesh, gold, gold_transform);
    let copper_transform = Mat4::from_translation(Vec3::new(1.3, 0.0, 0.3))
        * Mat4::from_scale(Vec3::splat(sphere_scale))
        * Mat4::from_translation(-sphere_pivot);
    scene.add_instance(sphere_mesh, copper, copper_transform);

    // Warm tungsten point light near the upper-left back corner.
    scene.add_point_light(PointLight::new(
        Vec3::new(-1.3, 2.6, -0.6),
        Vec3::new(1.0, 0.72, 0.45),
        280.0,
    ));
    // Cool point light near the upper-right back corner.
    scene.add_point_light(PointLight::new(
        Vec3::new(1.4, 2.6, -0.2),
        Vec3::new(0.65, 0.80, 1.0),
        220.0,
    ));

    // Magenta spot light aimed at the bunny from the front-left.
    let bunny_focus = Vec3::new(0.0, 0.8, 0.6);
    let spot_a_pos = Vec3::new(-1.6, 2.6, 2.6);
    scene.add_spot_light(SpotLight::new(
        spot_a_pos,
        (bunny_focus - spot_a_pos).normalize(),
        Vec3::new(1.0, 0.4, 0.85),
        600.0,
        (26.0_f32).to_radians(),
        (14.0_f32).to_radians(),
    ));
    // Teal spot light aimed at the bunny from the front-right.
    let spot_b_pos = Vec3::new(1.7, 2.6, 2.6);
    scene.add_spot_light(SpotLight::new(
        spot_b_pos,
        (bunny_focus - spot_b_pos).normalize(),
        Vec3::new(0.35, 0.95, 0.85),
        600.0,
        (22.0_f32).to_radians(),
        (12.0_f32).to_radians(),
    ));

    let camera = PinholeCamera::new(
        Vec3::new(0.0, 2.15, 7.1),
        Vec3::new(0.0, 1.45, -0.05),
        Vec3::Y,
        38.0_f32.to_radians(),
    );

    Ok((scene, camera))
}
