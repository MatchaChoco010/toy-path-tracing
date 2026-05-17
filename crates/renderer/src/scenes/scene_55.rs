//! OpenPBR の単体バニーで、暗めの rough diffuse に一様な fuzz を載せた velvet 風マテリアルを表示する。

use std::error::Error;

use glam::Vec3;

use crate::{material::OpenPbrMaterial, scene::PinholeCamera, scene::Scene};

use super::helper::create_single_openpbr_bunny_scene;

pub fn create_scene_55(
    _ocio: &crate::color::OcioColorPipeline,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    create_single_openpbr_bunny_scene(
        OpenPbrMaterial::new(Vec3::new(0.28, 0.04, 0.1))
            .with_specular_weight(0.05)
            .with_specular_roughness(0.88)
            .with_base_diffuse_roughness(0.95)
            .with_fuzz(1.0, Vec3::new(1.0, 0.32, 0.62), 0.5),
    )
}
