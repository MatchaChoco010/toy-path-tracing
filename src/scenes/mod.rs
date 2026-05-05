mod scene_0;
mod scene_1;
mod scene_10;
mod scene_11;
mod scene_12;
mod scene_13;
mod scene_14;
mod scene_15;
mod scene_16;
mod scene_17;
mod scene_18;
mod scene_19;
mod scene_2;
mod scene_20;
mod scene_21;
mod scene_22;
mod scene_23;
mod scene_24;
mod scene_25;
mod scene_26;
mod scene_27;
mod scene_28;
mod scene_29;
mod scene_3;
mod scene_30;
mod scene_31;
mod scene_32;
mod scene_33;
mod scene_34;
mod scene_35;
mod scene_36;
mod scene_37;
mod scene_38;
mod scene_39;
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
        13 => scene_13::create_scene_13(),
        14 => scene_14::create_scene_14(),
        15 => scene_15::create_scene_15(),
        16 => scene_16::create_scene_16(),
        17 => scene_17::create_scene_17(),
        18 => scene_18::create_scene_18(),
        19 => scene_19::create_scene_19(),
        20 => scene_20::create_scene_20(),
        21 => scene_21::create_scene_21(),
        22 => scene_22::create_scene_22(),
        23 => scene_23::create_scene_23(),
        24 => scene_24::create_scene_24(),
        25 => scene_25::create_scene_25(),
        26 => scene_26::create_scene_26(),
        27 => scene_27::create_scene_27(),
        28 => scene_28::create_scene_28(),
        29 => scene_29::create_scene_29(),
        30 => scene_30::create_scene_30(),
        31 => scene_31::create_scene_31(),
        32 => scene_32::create_scene_32(),
        33 => scene_33::create_scene_33(),
        34 => scene_34::create_scene_34(),
        35 => scene_35::create_scene_35(),
        36 => scene_36::create_scene_36(),
        37 => scene_37::create_scene_37(),
        38 => scene_38::create_scene_38(),
        39 => scene_39::create_scene_39(),
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
