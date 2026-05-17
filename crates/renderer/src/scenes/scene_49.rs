//! OpenPBR の単体バニーで、銅色ベースに緑の F82 tint を持つ rough metal を表示する。

use std::error::Error;

use glam::Vec3;

use crate::{material::OpenPbrMaterial, scene::PinholeCamera, scene::Scene};

use super::helper::create_single_openpbr_bunny_scene;

pub fn create_scene_49(
    _ocio: &crate::color::OcioColorPipeline,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    create_single_openpbr_bunny_scene(
        OpenPbrMaterial::new(Vec3::new(0.95, 0.48, 0.22))
            .with_base_metalness(1.0)
            .with_specular_color(Vec3::new(0.0, 1.0, 0.0))
            .with_specular_roughness(0.24),
    )
}
