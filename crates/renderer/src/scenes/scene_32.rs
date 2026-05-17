//! puresky 環境光のもと、紙飛行機を subsurface=0.5 の thin-walled Standard Surface として配置する。

use std::error::Error;

use crate::{scene::PinholeCamera, scene::Scene};

use super::scene_31::create_paper_plane_scene;

pub fn create_scene_32(
    _ocio: &crate::color::OcioColorPipeline,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    create_paper_plane_scene(0.5)
}
