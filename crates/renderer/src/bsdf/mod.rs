mod conductor_complex;
mod conductor_ggx;
mod conductor_ggx_cui_2023;
mod dielectric_ggx;
mod directional_albedo;
mod disney_brdf;
pub(crate) mod dispersion;
mod eon;
mod glass;
mod gtr1;
mod mirror;
pub mod mtlx;
mod normalized_lambert;
mod open_pbr;
mod oren_nayar;
mod sheen;
mod smith_ggx;
mod standard_surface;
pub(crate) mod thin_film;

use glam::Vec3;

pub fn integrate_hemisphere_vec3(f: impl Fn(Vec3) -> Vec3) -> Vec3 {
    const Z_SAMPLES: usize = 256;
    const PHI_SAMPLES: usize = 256;
    let dz = 1.0 / Z_SAMPLES as f32;
    let dphi = std::f32::consts::TAU / PHI_SAMPLES as f32;
    let domega = dz * dphi;
    let mut integral = Vec3::ZERO;

    for z_index in 0..Z_SAMPLES {
        let z = (z_index as f32 + 0.5) * dz;
        let r = (1.0 - z * z).max(0.0).sqrt();

        for phi_index in 0..PHI_SAMPLES {
            let phi = (phi_index as f32 + 0.5) * dphi;
            let w = Vec3::new(r * phi.cos(), r * phi.sin(), z);
            integral += f(w);
        }
    }

    integral * domega
}

pub fn integrate_upper_hemisphere_vec3(f: impl Fn(Vec3) -> Vec3) -> Vec3 {
    const Z_SAMPLES: usize = 128;
    const PHI_SAMPLES: usize = 128;
    let dz = 1.0 / Z_SAMPLES as f32;
    let dphi = std::f32::consts::TAU / PHI_SAMPLES as f32;
    let mut sum = Vec3::ZERO;

    for z_index in 0..Z_SAMPLES {
        let z = (z_index as f32 + 0.5) * dz;
        let r = (1.0 - z * z).max(0.0).sqrt();
        for phi_index in 0..PHI_SAMPLES {
            let phi = (phi_index as f32 + 0.5) * dphi;
            let wi = Vec3::new(r * phi.cos(), r * phi.sin(), z);
            sum += f(wi);
        }
    }

    sum * dz * dphi
}

pub use conductor_complex::{
    ConductorComplexGgxBsdf, artist_friendly_complex_ior, fresnel_complex,
};

pub use conductor_ggx::ConductorGgxBsdf;
pub use conductor_ggx_cui_2023::ConductorGgxCui2023Bsdf;
pub use dielectric_ggx::{DielectricGgxAllowedPaths, DielectricGgxBsdf};
pub(crate) use directional_albedo::{
    ConductorGgxEnergyCompensationLut, DielectricGgxDirectionalAlbedoLut,
    DielectricGgxEnergyCompensationLut, DirectionalAlbedoCache,
    MtlxDielectricGgxDirectionalAlbedoLut, MtlxGeneralizedSchlickGgxDirectionalAlbedoLut,
    SheenDirectionalAlbedoLut, sanitize_dielectric_eta,
};
pub use disney_brdf::DisneyBrdfBsdf;
pub use eon::EonBsdf;
pub use glass::GlassBsdf;
pub use mirror::MirrorBsdf;
pub use normalized_lambert::NormalizedLambertBsdf;
pub use open_pbr::{OpenPbrBsdf, OpenPbrBsdfParams};
pub use oren_nayar::OrenNayarBsdf;
pub use sheen::{SheenBsdf, sheen_directional_albedo_estimate};
pub use standard_surface::{StandardSurfaceBsdf, StandardSurfaceBsdfParams};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportMode {
    Radiance,
    Importance,
}

impl TransportMode {
    pub fn reverse(self) -> Self {
        match self {
            Self::Radiance => Self::Importance,
            Self::Importance => Self::Radiance,
        }
    }

    pub fn transmission_scale(self, eta_rel: f32) -> f32 {
        match self {
            Self::Radiance => 1.0 / (eta_rel * eta_rel),
            Self::Importance => 1.0,
        }
    }
}

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
    pub pdf_rev: f32,
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
