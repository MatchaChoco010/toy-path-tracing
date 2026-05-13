use glam::Vec3;

use crate::bsdf::BsdfFlags;

#[derive(Debug, Clone, Copy)]
pub struct MtlxLobeSample {
    pub weight: Vec3,
    pub wi_local: Vec3,
    pub pdf: f32,
    pub flags: BsdfFlags,
    pub eta: f32,
}
