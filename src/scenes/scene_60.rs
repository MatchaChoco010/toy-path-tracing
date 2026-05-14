//! OpenPBR の mori-knob で、metal に rough な青い clear coat と coat darkening を載せたマテリアルを表示する。

use std::error::Error;

use glam::Vec3;

use crate::{camera::PinholeCamera, material::OpenPbrMaterial, scene::Scene};

use super::openpbr_mori_knob::create_openpbr_mori_knob_scene;

pub fn create_scene_60() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    create_openpbr_mori_knob_scene(
        OpenPbrMaterial::new(Vec3::new(0.72, 0.7, 0.76))
            .with_base_metalness(1.0)
            .with_specular_color(Vec3::new(0.9, 0.88, 0.95))
            .with_specular_roughness(0.24)
            .with_coat_weight(1.0)
            .with_coat_color(Vec3::new(0.16, 0.34, 1.0))
            .with_coat_roughness(0.34)
            .with_coat_ior(1.85)
            .with_coat_darkening(0.85),
    )
}
