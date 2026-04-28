use glam::Vec3;

pub fn srgb_to_linear_channel(channel: f32) -> f32 {
    if channel <= 0.04045 {
        channel / 12.92
    } else {
        ((channel + 0.055) / 1.055).powf(2.4)
    }
}

pub fn srgb_to_linear(rgb: Vec3) -> Vec3 {
    Vec3::new(
        srgb_to_linear_channel(rgb.x),
        srgb_to_linear_channel(rgb.y),
        srgb_to_linear_channel(rgb.z),
    )
}

pub fn linear_to_srgb_channel(channel: f32) -> f32 {
    if channel <= 0.0031308 {
        12.92 * channel
    } else {
        1.055 * channel.powf(1.0 / 2.4) - 0.055
    }
}

pub fn linear_to_srgb(rgb: Vec3) -> Vec3 {
    Vec3::new(
        linear_to_srgb_channel(rgb.x),
        linear_to_srgb_channel(rgb.y),
        linear_to_srgb_channel(rgb.z),
    )
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::{
        linear_to_srgb, linear_to_srgb_channel, srgb_to_linear, srgb_to_linear_channel,
    };

    #[test]
    fn srgb_to_linear_channel_at_zero_is_zero() {
        assert_eq!(srgb_to_linear_channel(0.0), 0.0);
    }

    #[test]
    fn srgb_to_linear_channel_at_one_is_one() {
        assert!((srgb_to_linear_channel(1.0) - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn srgb_to_linear_channel_uses_linear_segment_below_threshold() {
        let channel = 0.02;
        assert!((srgb_to_linear_channel(channel) - channel / 12.92).abs() < 1.0e-9);
    }

    #[test]
    fn srgb_to_linear_channel_at_half_matches_known_value() {
        assert!((srgb_to_linear_channel(0.5) - 0.21404114).abs() < 1.0e-6);
    }

    #[test]
    fn srgb_to_linear_decodes_each_component_independently() {
        let decoded = srgb_to_linear(Vec3::new(0.0, 0.5, 1.0));
        assert!(decoded.x.abs() < 1.0e-6);
        assert!((decoded.y - 0.21404114).abs() < 1.0e-6);
        assert!((decoded.z - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn linear_to_srgb_inverts_srgb_to_linear() {
        for c in [0.0_f32, 0.01, 0.04, 0.2, 0.5, 0.8, 1.0] {
            let round_trip = linear_to_srgb_channel(srgb_to_linear_channel(c));
            assert!((round_trip - c).abs() < 1.0e-5);
        }
    }

    #[test]
    fn linear_to_srgb_clamps_endpoints() {
        assert_eq!(linear_to_srgb(Vec3::ZERO), Vec3::ZERO);
        assert!(linear_to_srgb(Vec3::ONE).abs_diff_eq(Vec3::ONE, 1.0e-5));
    }
}
