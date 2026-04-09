use clap::ValueEnum;
use glam::Vec3;
use rand::rngs::ThreadRng;

use crate::{ray::Ray, scene::Scene};

pub mod pt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum IntegratorKind {
    Pt,
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
            Self::Pt => pt::trace_radiance(scene, initial_ray, rng, max_depth),
        }
    }
}
