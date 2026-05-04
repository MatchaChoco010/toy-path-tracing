use std::{error::Error, path::Path};

use glam::{Mat4, Vec3};

use crate::{
    camera::PinholeCamera,
    light::EnvironmentLight,
    material::{Material, StandardSurfaceMaterial},
    mesh::load_obj,
    scene::Scene,
};

pub fn create_scene_31() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    create_paper_plane_scene(0.0)
}

pub(super) fn create_paper_plane_scene(
    subsurface: f32,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let plane = load_obj(Path::new("assets/models/paper-plane.obj"))?;
    let plane_extent = plane.bounds.extent();
    let plane_target_length = 1.4_f32;
    let plane_scale = plane_target_length / plane_extent.max_element().max(1.0e-3);
    let plane_center = plane.bounds.center();
    let plane_mesh = scene.add_mesh(plane);

    let material = StandardSurfaceMaterial::new(Vec3::new(0.95, 0.92, 0.86))
        .with_thin_walled(true)
        .with_subsurface(subsurface)
        .with_subsurface_color(Vec3::new(0.95, 0.92, 0.86))
        .with_specular(0.05)
        .with_specular_roughness(0.0)
        .with_diffuse_roughness(0.5);
    let material_id = scene.add_material(Material::StandardSurface(material));

    let plane_transform = Mat4::from_translation(Vec3::new(0.0, 0.6, 0.0))
        * Mat4::from_scale(Vec3::splat(plane_scale))
        * Mat4::from_translation(-plane_center);
    scene.add_instance(plane_mesh, material_id, plane_transform);

    let env = EnvironmentLight::from_hdr_file(
        "assets/sky/kloofendal_48d_partly_cloudy_puresky_4k.hdr",
        1.0,
        std::f32::consts::PI * 0.5,
    )?;
    scene.set_environment_light(env);

    let camera_eye = Vec3::new(0.9, -0.7, 0.7);
    let camera_target = Vec3::new(-0.1, 0.75, -0.1);
    let camera = PinholeCamera::new(
        camera_eye,
        camera_target,
        Vec3::Y,
        58.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}
