mod scene_0;
mod scene_1;
mod scene_10;
mod scene_11;
mod scene_12;
mod scene_2;
mod scene_3;
mod scene_4;
mod scene_5;
mod scene_6;
mod scene_7;
mod scene_8;
mod scene_9;

use glam::{EulerRot, Quat};
use std::error::Error;

use crate::{camera::PinholeCamera, mesh::Mesh, scene::Scene};

pub fn load_scene(scene_index: u32) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    match scene_index {
        1 => scene_1::create_scene_1(),
        2 => scene_2::create_scene_2(),
        3 => scene_3::create_scene_3(),
        4 => scene_4::create_scene_4(),
        5 => scene_5::create_scene_5(),
        6 => scene_6::create_scene_6(),
        7 => scene_7::create_scene_7(),
        8 => scene_8::create_scene_8(),
        9 => scene_9::create_scene_9(),
        10 => scene_10::create_scene_10(),
        11 => scene_11::create_scene_11(),
        12 => scene_12::create_scene_12(),
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
