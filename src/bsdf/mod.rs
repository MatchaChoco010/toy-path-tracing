mod mirror;
mod normalized_lambert;

use glam::Vec3;

pub use mirror::MirrorBsdf;
pub use normalized_lambert::NormalizedLambertBsdf;

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
