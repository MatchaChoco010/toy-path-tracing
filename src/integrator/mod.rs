use clap::ValueEnum;
use glam::Vec3;
use rand::rngs::ThreadRng;

use crate::{ray::Ray, scene::Scene};

pub mod mis;
pub mod nee;
pub mod pt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum IntegratorKind {
    Mis,
    Pt,
    Nee,
}

impl IntegratorKind {
    pub fn trace_radiance(
        self,
        scene: &Scene,
        initial_ray: Ray,
        rng: &mut ThreadRng,
        max_depth: u32,
    ) -> Vec3 {
        match self {
            Self::Mis => mis::trace_radiance(scene, initial_ray, rng, max_depth),
            Self::Pt => pt::trace_radiance(scene, initial_ray, rng, max_depth),
            Self::Nee => nee::trace_radiance(scene, initial_ray, rng, max_depth),
        }
    }
}
