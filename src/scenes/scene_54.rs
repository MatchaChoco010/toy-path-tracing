//! OpenPBR の単体バニーで、暗めの rough diffuse マテリアルを表示する。

use std::error::Error;

use glam::Vec3;

use crate::{camera::PinholeCamera, material::OpenPbrMaterial, scene::Scene};

use super::openpbr_single_bunny::create_single_openpbr_bunny_scene;

pub fn create_scene_54() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    create_single_openpbr_bunny_scene(
        OpenPbrMaterial::new(Vec3::new(0.72, 0.52, 0.36))
            .with_specular_weight(0.12)
            .with_specular_roughness(0.82)
            .with_base_diffuse_roughness(0.92),
    )
}
