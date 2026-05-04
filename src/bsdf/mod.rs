mod conductor_ggx;
mod dielectric_ggx;
mod directional_albedo;
mod disney_brdf;
pub(crate) mod dispersion;
mod glass;
mod gtr1;
mod mirror;
mod normalized_lambert;
mod oren_nayar;
mod smith_ggx;

use glam::Vec3;

pub use conductor_ggx::ConductorGgxBsdf;
pub use dielectric_ggx::{DielectricGgxAllowedPaths, DielectricGgxBsdf};
pub(crate) use directional_albedo::{
    DielectricGgxDirectionalAlbedoLut, DirectionalAlbedoCache, sanitize_dielectric_eta,
};
pub use disney_brdf::DisneyBrdfBsdf;
pub use glass::GlassBsdf;
pub use mirror::MirrorBsdf;
pub use normalized_lambert::NormalizedLambertBsdf;
pub use oren_nayar::OrenNayarBsdf;

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct BsdfFlags: u32 {
        const DIFFUSE = 1 << 0;
        const GLOSSY = 1 << 1;
        const DELTA = 1 << 2;
        const REFLECTION = 1 << 3;
        const TRANSMISSION = 1 << 4;
        const SMOOTH = Self::DIFFUSE.bits() | Self::GLOSSY.bits();
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BsdfSample {
    pub weight: Vec3,
    pub wi: Vec3,
    pub pdf: f32,
    pub flags: BsdfFlags,
    pub eta: f32,
    pub wavelength_lock: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::BsdfFlags;

    #[test]
    fn smooth_flag_is_diffuse_and_glossy_union() {
        assert_eq!(BsdfFlags::SMOOTH, BsdfFlags::DIFFUSE | BsdfFlags::GLOSSY);
    }

    #[test]
    fn reflection_and_transmission_flags_are_distinct() {
        assert!(!BsdfFlags::REFLECTION.intersects(BsdfFlags::TRANSMISSION));
    }
}
