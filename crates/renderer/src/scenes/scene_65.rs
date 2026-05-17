//! OpenPBR の low-poly 単体バニーで、dispersion scale 1.5 の透明 glass を IBL とエリアライトで表示する。

use std::error::Error;

use glam::Vec3;

use crate::{camera::PinholeCamera, material::OpenPbrMaterial, scene::Scene};

use super::openpbr_single_bunny::create_single_openpbr_low_bunny_scene;

pub fn create_scene_65() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    create_single_openpbr_low_bunny_scene(
        OpenPbrMaterial::new(Vec3::ZERO)
            .with_base_weight(0.0)
            .with_specular_ior(1.4)
            .with_specular_roughness(0.0)
            .with_transmission_weight(1.0)
            .with_transmission_color(Vec3::ONE)
            .with_transmission_dispersion_scale(1.5),
    )
}
