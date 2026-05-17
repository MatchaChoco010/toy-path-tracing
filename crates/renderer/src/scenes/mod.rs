mod helper;

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
mod scene_40;
mod scene_41;
mod scene_42;
mod scene_43;
mod scene_44;
mod scene_45;
mod scene_46;
mod scene_47;
mod scene_48;
mod scene_49;
mod scene_5;
mod scene_50;
mod scene_51;
mod scene_52;
mod scene_53;
mod scene_54;
mod scene_55;
mod scene_56;
mod scene_57;
mod scene_58;
mod scene_59;
mod scene_6;
mod scene_60;
mod scene_61;
mod scene_62;
mod scene_63;
mod scene_64;
mod scene_65;
mod scene_7;
mod scene_8;
mod scene_9;

use glam::{EulerRot, Quat};
use std::error::Error;

use crate::{
    color::OcioColorPipeline,
    scene::{Mesh, PinholeCamera, Scene},
};

pub fn load_scene(
    scene_index: u32,
    ocio: &OcioColorPipeline,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    match scene_index {
        1 => scene_1::create_scene_1(ocio),
        2 => scene_2::create_scene_2(ocio),
        3 => scene_3::create_scene_3(ocio),
        4 => scene_4::create_scene_4(ocio),
        5 => scene_5::create_scene_5(ocio),
        6 => scene_6::create_scene_6(ocio),
        7 => scene_7::create_scene_7(ocio),
        8 => scene_8::create_scene_8(ocio),
        9 => scene_9::create_scene_9(ocio),
        10 => scene_10::create_scene_10(ocio),
        11 => scene_11::create_scene_11(ocio),
        12 => scene_12::create_scene_12(ocio),
        13 => scene_13::create_scene_13(ocio),
        14 => scene_14::create_scene_14(ocio),
        15 => scene_15::create_scene_15(ocio),
        16 => scene_16::create_scene_16(ocio),
        17 => scene_17::create_scene_17(ocio),
        18 => scene_18::create_scene_18(ocio),
        19 => scene_19::create_scene_19(ocio),
        20 => scene_20::create_scene_20(ocio),
        21 => scene_21::create_scene_21(ocio),
        22 => scene_22::create_scene_22(ocio),
        23 => scene_23::create_scene_23(ocio),
        24 => scene_24::create_scene_24(ocio),
        25 => scene_25::create_scene_25(ocio),
        26 => scene_26::create_scene_26(ocio),
        27 => scene_27::create_scene_27(ocio),
        28 => scene_28::create_scene_28(ocio),
        29 => scene_29::create_scene_29(ocio),
        30 => scene_30::create_scene_30(ocio),
        31 => scene_31::create_scene_31(ocio),
        32 => scene_32::create_scene_32(ocio),
        33 => scene_33::create_scene_33(ocio),
        34 => scene_34::create_scene_34(ocio),
        35 => scene_35::create_scene_35(ocio),
        36 => scene_36::create_scene_36(ocio),
        37 => scene_37::create_scene_37(ocio),
        38 => scene_38::create_scene_38(ocio),
        39 => scene_39::create_scene_39(ocio),
        40 => scene_40::create_scene_40(ocio),
        41 => scene_41::create_scene_41(ocio),
        42 => scene_42::create_scene_42(ocio),
        43 => scene_43::create_scene_43(ocio),
        44 => scene_44::create_scene_44(ocio),
        45 => scene_45::create_scene_45(ocio),
        46 => scene_46::create_scene_46(ocio),
        47 => scene_47::create_scene_47(ocio),
        48 => scene_48::create_scene_48(ocio),
        49 => scene_49::create_scene_49(ocio),
        50 => scene_50::create_scene_50(ocio),
        51 => scene_51::create_scene_51(ocio),
        52 => scene_52::create_scene_52(ocio),
        53 => scene_53::create_scene_53(ocio),
        54 => scene_54::create_scene_54(ocio),
        55 => scene_55::create_scene_55(ocio),
        56 => scene_56::create_scene_56(ocio),
        57 => scene_57::create_scene_57(ocio),
        58 => scene_58::create_scene_58(ocio),
        59 => scene_59::create_scene_59(ocio),
        60 => scene_60::create_scene_60(ocio),
        61 => scene_61::create_scene_61(ocio),
        62 => scene_62::create_scene_62(ocio),
        63 => scene_63::create_scene_63(ocio),
        64 => scene_64::create_scene_64(ocio),
        65 => scene_65::create_scene_65(ocio),
        _ => scene_0::create_scene_0(ocio),
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
