//! OpenPBR の単体バニーで、roughness 1/3 の fuzz を載せた metal を表示する。

use std::error::Error;

use glam::Vec3;

use crate::{camera::PinholeCamera, material::OpenPbrMaterial, scene::Scene};

use super::openpbr_single_bunny::create_single_openpbr_bunny_scene;

pub fn create_scene_52() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    create_single_openpbr_bunny_scene(
        OpenPbrMaterial::new(Vec3::new(0.72, 0.7, 0.76))
            .with_base_metalness(1.0)
            .with_specular_color(Vec3::new(0.9, 0.88, 0.95))
            .with_specular_roughness(0.22)
            .with_fuzz(0.75, Vec3::new(0.8, 0.72, 1.0), 1.0 / 3.0),
    )
}
