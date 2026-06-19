use std::sync::OnceLock;

use glam::Vec3;

const LAMBDA_F_NM: f32 = 486.13;
const LAMBDA_D_NM: f32 = 587.56;
const LAMBDA_C_NM: f32 = 656.27;

pub(crate) const CAMERA_LAMBDA_MIN_NM: f32 = 400.0;
pub(crate) const CAMERA_LAMBDA_MAX_NM: f32 = 720.0;
pub(crate) const CAMERA_LAMBDA_STEP_NM: f32 = 10.0;
pub(crate) const CAMERA_SPECTRAL_SAMPLE_COUNT: usize = 33;
pub(crate) const CAMERA_SPECTRAL_INTERVAL_COUNT: usize = CAMERA_SPECTRAL_SAMPLE_COUNT - 1;

const CAMERA_LAMBDA_RANGE_NM: f32 = CAMERA_LAMBDA_MAX_NM - CAMERA_LAMBDA_MIN_NM;

// Canon 5D Mark II RGB spectral sensitivity, sampled from 400nm to 720nm in
// 10nm steps. Source: Jiang, Liu, Gu, and Süsstrunk, "What is the Space of
// Spectral Sensitivity Functions for Digital Color Cameras?", WACV 2013;
// Camera Spectral Sensitivity Database, https://zenodo.org/records/3245883,
// DOI 10.5281/zenodo.3245883. The dataset page lists CC BY-NC-SA 4.0.
#[allow(clippy::excessive_precision)]
const CANON_5D_MARK_II_R: [f32; CAMERA_SPECTRAL_SAMPLE_COUNT] = [
    0.0019, 0.0045, 0.0103, 0.0055, 0.0034, 0.0021, 0.0023, 0.0039, 0.0073, 0.0118, 0.0179, 0.0612,
    0.0874, 0.1534, 0.1686, 0.1724, 0.2003, 0.3158, 0.4514, 0.5258, 0.5989, 0.4728, 0.4084, 0.3562,
    0.292, 0.226, 0.1704, 0.1372, 0.0428, 0.0087, 0.0017, 0.0007, 0.0005,
];

#[allow(clippy::excessive_precision)]
const CANON_5D_MARK_II_G: [f32; CAMERA_SPECTRAL_SAMPLE_COUNT] = [
    0.0036, 0.0123, 0.0377, 0.0422, 0.0565, 0.0704, 0.097, 0.209, 0.43, 0.6381, 0.692, 1.0, 0.8735,
    0.9058, 0.8326, 0.8057, 0.712, 0.6467, 0.5426, 0.3935, 0.2958, 0.1287, 0.06, 0.0402, 0.0276,
    0.0182, 0.0138, 0.0143, 0.0061, 0.0017, 0.0008, 0.0006, 0.0005,
];

#[allow(clippy::excessive_precision)]
const CANON_5D_MARK_II_B: [f32; CAMERA_SPECTRAL_SAMPLE_COUNT] = [
    0.0127, 0.0971, 0.3516, 0.4765, 0.56, 0.6476, 0.7745, 0.6759, 0.6858, 0.5932, 0.3971, 0.3559,
    0.1617, 0.0883, 0.0551, 0.0424, 0.0269, 0.0205, 0.0159, 0.0122, 0.01, 0.0054, 0.0036, 0.0032,
    0.0029, 0.0032, 0.0032, 0.0034, 0.0013, 0.0005, 0.0004, 0.0004, 0.0004,
];

pub fn cauchy_ior(lambda_nm: f32, n_d: f32, abbe_v: f32) -> f32 {
    if abbe_v <= 0.0 {
        return n_d;
    }
    let inv_lf2 = 1.0 / (LAMBDA_F_NM * LAMBDA_F_NM);
    let inv_lc2 = 1.0 / (LAMBDA_C_NM * LAMBDA_C_NM);
    let inv_ld2 = 1.0 / (LAMBDA_D_NM * LAMBDA_D_NM);
    let b = (n_d - 1.0) / (abbe_v * (inv_lf2 - inv_lc2));
    let a = n_d - b * inv_ld2;
    a + b / (lambda_nm * lambda_nm)
}

#[cfg(test)]
pub fn sample_dispersion_wavelength(u: f32) -> (f32, Vec3) {
    sample_dispersion_wavelength_weighted(u, Vec3::ONE)
}

pub fn sample_dispersion_wavelength_weighted(u: f32, throughput: Vec3) -> (f32, Vec3) {
    let interval_weights = wavelength_interval_weights(throughput);
    let (index, local, pdf) = sample_interval(&interval_weights, u);
    let lambda = (CAMERA_LAMBDA_MIN_NM + CAMERA_LAMBDA_STEP_NM * (index as f32 + local))
        .clamp(CAMERA_LAMBDA_MIN_NM, CAMERA_LAMBDA_MAX_NM);
    (lambda, camera_rgb_basis_at(lambda) / pdf.max(1.0e-8))
}

pub(crate) fn sample_camera_channel_wavelength(channel: usize, u: f32) -> f32 {
    let interval_weights = camera_channel_interval_weights(channel);
    let (index, local, _) = sample_interval(&interval_weights, u);
    (CAMERA_LAMBDA_MIN_NM + CAMERA_LAMBDA_STEP_NM * (index as f32 + local))
        .clamp(CAMERA_LAMBDA_MIN_NM, CAMERA_LAMBDA_MAX_NM)
}

pub(crate) fn camera_rgb_basis_at(lambda_nm: f32) -> Vec3 {
    camera_rgb_sensitivity_at(lambda_nm) * camera_normalization_scale()
}

pub(crate) fn camera_wavelength_sampling_pdf(lambda_nm: f32, throughput: Vec3) -> f32 {
    let interval_weights = wavelength_interval_weights(throughput);
    let total = interval_weights.iter().sum::<f32>().max(1.0e-8);
    let interval = camera_interval_index(lambda_nm);
    interval_weights[interval] / (total * CAMERA_LAMBDA_STEP_NM)
}

pub(crate) fn camera_spectral_interval_midpoint_nm(index: usize) -> f32 {
    CAMERA_LAMBDA_MIN_NM + CAMERA_LAMBDA_STEP_NM * (index as f32 + 0.5)
}

fn wavelength_interval_weights(throughput: Vec3) -> [f32; CAMERA_SPECTRAL_INTERVAL_COUNT] {
    let throughput = throughput.max(Vec3::ZERO);
    let weights = if throughput.max_element() > 0.0 {
        throughput
    } else {
        Vec3::ONE
    };

    let mut interval_weights = [0.0; CAMERA_SPECTRAL_INTERVAL_COUNT];
    let mut sum = 0.0;
    for (i, weight) in interval_weights.iter_mut().enumerate() {
        let s0 = camera_rgb_basis_sample(i).dot(weights).max(0.0);
        let s1 = camera_rgb_basis_sample(i + 1).dot(weights).max(0.0);
        *weight = 0.5 * (s0 + s1) * CAMERA_LAMBDA_STEP_NM;
        sum += *weight;
    }

    if sum <= 0.0 {
        interval_weights.fill(CAMERA_LAMBDA_RANGE_NM / CAMERA_SPECTRAL_INTERVAL_COUNT as f32);
    }

    interval_weights
}

fn camera_channel_interval_weights(channel: usize) -> [f32; CAMERA_SPECTRAL_INTERVAL_COUNT] {
    let mut interval_weights = [0.0; CAMERA_SPECTRAL_INTERVAL_COUNT];
    let mut sum = 0.0;
    for (i, weight) in interval_weights.iter_mut().enumerate() {
        let s0 = rgb_component(camera_rgb_basis_sample(i), channel).max(0.0);
        let s1 = rgb_component(camera_rgb_basis_sample(i + 1), channel).max(0.0);
        *weight = 0.5 * (s0 + s1) * CAMERA_LAMBDA_STEP_NM;
        sum += *weight;
    }

    if sum <= 0.0 {
        interval_weights.fill(CAMERA_LAMBDA_RANGE_NM / CAMERA_SPECTRAL_INTERVAL_COUNT as f32);
    }

    interval_weights
}

fn sample_interval(
    interval_weights: &[f32; CAMERA_SPECTRAL_INTERVAL_COUNT],
    u: f32,
) -> (usize, f32, f32) {
    let total = interval_weights.iter().sum::<f32>().max(1.0e-8);
    let target = u.clamp(0.0, 1.0 - f32::EPSILON) * total;
    let mut sum = 0.0;
    let mut index = 0;
    for (i, weight) in interval_weights.iter().enumerate() {
        if target < sum + *weight {
            index = i;
            break;
        }
        sum += *weight;
        index = i;
    }

    let interval_weight = interval_weights[index].max(1.0e-8);
    let local = ((target - sum) / interval_weight).clamp(0.0, 1.0);
    let pdf = interval_weight / (total * CAMERA_LAMBDA_STEP_NM);
    (index, local, pdf)
}

fn camera_rgb_sensitivity_at(lambda_nm: f32) -> Vec3 {
    let position = (lambda_nm - CAMERA_LAMBDA_MIN_NM) / CAMERA_LAMBDA_STEP_NM;
    if position <= 0.0 {
        return camera_rgb_sensitivity_sample(0);
    }
    let max_index = CAMERA_SPECTRAL_SAMPLE_COUNT - 1;
    if position >= max_index as f32 {
        return camera_rgb_sensitivity_sample(max_index);
    }
    let lower = position.floor() as usize;
    let upper = lower + 1;
    let t = position - lower as f32;
    camera_rgb_sensitivity_sample(lower).lerp(camera_rgb_sensitivity_sample(upper), t)
}

fn camera_rgb_sensitivity_sample(index: usize) -> Vec3 {
    Vec3::new(
        CANON_5D_MARK_II_R[index],
        CANON_5D_MARK_II_G[index],
        CANON_5D_MARK_II_B[index],
    )
}

fn camera_rgb_basis_sample(index: usize) -> Vec3 {
    camera_rgb_sensitivity_sample(index) * camera_normalization_scale()
}

fn rgb_component(v: Vec3, channel: usize) -> f32 {
    match channel {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

fn camera_interval_index(lambda_nm: f32) -> usize {
    let position = ((lambda_nm - CAMERA_LAMBDA_MIN_NM) / CAMERA_LAMBDA_STEP_NM)
        .clamp(0.0, CAMERA_SPECTRAL_INTERVAL_COUNT as f32 - f32::EPSILON);
    position.floor() as usize
}

fn camera_normalization_scale() -> Vec3 {
    static CACHE: OnceLock<Vec3> = OnceLock::new();
    *CACHE.get_or_init(|| {
        let mut integral = Vec3::ZERO;
        for i in 0..CAMERA_SPECTRAL_INTERVAL_COUNT {
            integral += 0.5
                * (camera_rgb_sensitivity_sample(i) + camera_rgb_sensitivity_sample(i + 1))
                * CAMERA_LAMBDA_STEP_NM;
        }
        Vec3::new(
            1.0 / integral.x.max(1.0e-8),
            1.0 / integral.y.max(1.0e-8),
            1.0 / integral.z.max(1.0e-8),
        )
    })
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::{
        CAMERA_LAMBDA_MAX_NM, CAMERA_LAMBDA_MIN_NM, cauchy_ior, sample_camera_channel_wavelength,
        sample_dispersion_wavelength, sample_dispersion_wavelength_weighted,
    };

    #[test]
    fn cauchy_returns_n_d_at_d_line() {
        let n = cauchy_ior(587.56, 1.5, 50.0);
        assert!((n - 1.5).abs() < 1.0e-4);
    }

    #[test]
    fn cauchy_disabled_at_zero_abbe_returns_n_d() {
        assert!((cauchy_ior(450.0, 1.5, 0.0) - 1.5).abs() < 1.0e-6);
    }

    #[test]
    fn cauchy_red_lower_than_blue() {
        let n_red = cauchy_ior(656.0, 1.5, 30.0);
        let n_blue = cauchy_ior(486.0, 1.5, 30.0);
        assert!(n_red < n_blue);
    }

    #[test]
    fn dispersion_basis_integrates_to_white() {
        let n = 200_000;
        let mut acc = Vec3::ZERO;
        for i in 0..n {
            let u = (i as f32 + 0.5) / n as f32;
            let (_, basis) = sample_dispersion_wavelength(u);
            acc += basis;
        }
        let avg = acc / n as f32;
        assert!((avg.x - 1.0).abs() < 5.0e-3);
        assert!((avg.y - 1.0).abs() < 5.0e-3);
        assert!((avg.z - 1.0).abs() < 5.0e-3);
    }

    #[test]
    fn throughput_weighted_dispersion_basis_integrates_to_white() {
        let n = 200_000;
        let throughput = Vec3::new(0.8, 0.35, 0.1);
        let mut acc = Vec3::ZERO;
        for i in 0..n {
            let u = (i as f32 + 0.5) / n as f32;
            let (_, basis) = sample_dispersion_wavelength_weighted(u, throughput);
            acc += basis;
        }
        let avg = acc / n as f32;
        assert!((avg.x - 1.0).abs() < 5.0e-3);
        assert!((avg.y - 1.0).abs() < 5.0e-3);
        assert!((avg.z - 1.0).abs() < 5.0e-3);
    }

    #[test]
    fn dispersion_wavelength_stays_in_camera_table_range() {
        let n = 4096;
        for i in 0..n {
            let u = (i as f32 + 0.5) / n as f32;
            let (lambda, basis) = sample_dispersion_wavelength(u);
            assert!((CAMERA_LAMBDA_MIN_NM..=CAMERA_LAMBDA_MAX_NM).contains(&lambda));
            assert!(basis.x >= 0.0 && basis.y >= 0.0 && basis.z >= 0.0);
        }
    }

    #[test]
    fn camera_channel_wavelengths_stay_in_table_range() {
        let n = 4096;
        for channel in 0..3 {
            for i in 0..n {
                let u = (i as f32 + 0.5) / n as f32;
                let lambda = sample_camera_channel_wavelength(channel, u);
                assert!((CAMERA_LAMBDA_MIN_NM..=CAMERA_LAMBDA_MAX_NM).contains(&lambda));
            }
        }
    }
}
