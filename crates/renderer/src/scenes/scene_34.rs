//! Scene 33 と同じ配置で、マテリアルを Cui 2023 multi-scattering Conductor GGX に差し替える。

use std::error::Error;

use crate::{
    material::{ConductorGgxCui2023Material, Material},
    scene::PinholeCamera,
    scene::Scene,
};

use super::scene_33::create_conductor_roughness_row;

pub fn create_scene_34(
    _ocio: &crate::color::OcioColorPipeline,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    create_conductor_roughness_row(|color, roughness| {
        Material::ConductorGgxCui2023(ConductorGgxCui2023Material::new(color, roughness, 0.0))
    })
}
