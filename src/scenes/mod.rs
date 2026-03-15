mod scene_0;
mod scene_1;

use glam::{EulerRot, Quat};
use std::error::Error;

use crate::{camera::PinholeCamera, mesh::Mesh, scene::Scene};

pub fn load_scene(scene_index: u32) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    match scene_index {
        1 => scene_1::create_scene_1(),
        _ => scene_0::create_scene_0(),
    }
}

pub(super) fn uniform_scale_for_height(mesh: &Mesh, target_height: f32) -> f32 {
    target_height / mesh.bounds.extent().y.max(1.0e-3)
}

pub(super) fn game_rotation_degrees(x_degrees: f32, y_degrees: f32, z_degrees: f32) -> Quat {
    Quat::from_euler(
        EulerRot::YXZ,
        y_degrees.to_radians(),
        x_degrees.to_radians(),
        z_degrees.to_radians(),
    )
}
