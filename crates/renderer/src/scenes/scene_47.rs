//! OpenPBR の単体バニーで、roughness 0 の透明 dispersive glass を表示する。

use std::error::Error;

use glam::Vec3;

use crate::{material::OpenPbrMaterial, scene::PinholeCamera, scene::Scene};

use super::helper::create_single_openpbr_bunny_scene;

pub fn create_scene_47(
    _ocio: &crate::color::OcioColorPipeline,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    create_single_openpbr_bunny_scene(
        OpenPbrMaterial::new(Vec3::ZERO)
            .with_base_weight(0.0)
            .with_specular_ior(1.4)
            .with_specular_roughness(0.0)
            .with_transmission_weight(1.0)
            .with_transmission_color(Vec3::ONE)
            .with_transmission_dispersion_abbe_number(20.0)
            .with_transmission_dispersion_scale(1.0),
    )
}
