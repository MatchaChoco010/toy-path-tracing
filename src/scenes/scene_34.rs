use std::error::Error;

use crate::{
    camera::PinholeCamera,
    material::{ConductorGgxCui2023Material, Material},
    scene::Scene,
};

use super::scene_33::create_conductor_roughness_row;

pub fn create_scene_34() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    create_conductor_roughness_row(|color, roughness| {
        Material::ConductorGgxCui2023(ConductorGgxCui2023Material::new(color, roughness, 0.0))
    })
}
