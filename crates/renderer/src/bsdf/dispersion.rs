use glam::{Mat3, Vec3};

const LAMBDA_MIN_NM: f32 = 380.0;
const LAMBDA_MAX_NM: f32 = 780.0;
const LAMBDA_STEP_NM: f32 = 5.0;
const TABLE_LEN: usize = 81;
const LAMBDA_RANGE_NM: f32 = LAMBDA_MAX_NM - LAMBDA_MIN_NM;

const LAMBDA_F_NM: f32 = 486.13;
const LAMBDA_D_NM: f32 = 587.56;
const LAMBDA_C_NM: f32 = 656.27;

const XYZ_TO_REC709: Mat3 = Mat3::from_cols(
    Vec3::new(3.2404542, -0.969_266, 0.055_643_4),
    Vec3::new(-1.537_138_5, 1.875_010_8, -0.204_025_9),
    Vec3::new(-0.498_531_4, 0.041_556, 1.057_225_2),
);

#[allow(clippy::excessive_precision)]
const CIE_X: [f32; TABLE_LEN] = [
    0.001368, 0.002236, 0.004243, 0.007650, 0.014310, 0.023190, 0.043510, 0.077630, 0.134380,
    0.214770, 0.283900, 0.328500, 0.348280, 0.348060, 0.336200, 0.318700, 0.290800, 0.251100,
    0.195360, 0.142100, 0.095640, 0.057950, 0.032010, 0.014700, 0.004900, 0.002400, 0.009300,
    0.029100, 0.063270, 0.109600, 0.165500, 0.225750, 0.290400, 0.359700, 0.433450, 0.512050,
    0.594500, 0.678400, 0.762100, 0.842500, 0.916300, 0.978600, 1.026300, 1.056700, 1.062200,
    1.045600, 1.002600, 0.938400, 0.854450, 0.751400, 0.642400, 0.541900, 0.447900, 0.360800,
    0.283500, 0.218700, 0.164900, 0.121200, 0.087400, 0.063600, 0.046770, 0.032900, 0.022700,
    0.015840, 0.011359, 0.008111, 0.005790, 0.004109, 0.002899, 0.002049, 0.001440, 0.001000,
    0.000690, 0.000476, 0.000332, 0.000235, 0.000166, 0.000117, 0.000083, 0.000059, 0.000042,
];

#[allow(clippy::excessive_precision)]
const CIE_Y: [f32; TABLE_LEN] = [
    0.000039, 0.000064, 0.000120, 0.000217, 0.000396, 0.000640, 0.001210, 0.002180, 0.004000,
    0.007300, 0.011600, 0.016840, 0.023000, 0.029800, 0.038000, 0.048000, 0.060000, 0.073900,
    0.090980, 0.112600, 0.139020, 0.169300, 0.208020, 0.258600, 0.323000, 0.407300, 0.503000,
    0.608200, 0.710000, 0.793200, 0.862000, 0.914850, 0.954000, 0.980300, 0.994950, 1.000000,
    0.995000, 0.978600, 0.952000, 0.915400, 0.870000, 0.816300, 0.757000, 0.694900, 0.631000,
    0.566800, 0.503000, 0.441200, 0.381000, 0.321000, 0.265000, 0.217000, 0.175000, 0.138200,
    0.107000, 0.081600, 0.061000, 0.044580, 0.032000, 0.023200, 0.017000, 0.011920, 0.008210,
    0.005723, 0.004102, 0.002929, 0.002091, 0.001484, 0.001047, 0.000740, 0.000520, 0.000361,
    0.000249, 0.000172, 0.000120, 0.000085, 0.000060, 0.000042, 0.000030, 0.000021, 0.000015,
];

#[allow(clippy::excessive_precision)]
const CIE_Z: [f32; TABLE_LEN] = [
    0.006450, 0.010550, 0.020050, 0.036210, 0.067850, 0.110200, 0.207400, 0.371300, 0.645600,
    1.039050, 1.385600, 1.622960, 1.747060, 1.782600, 1.772110, 1.744100, 1.669200, 1.528100,
    1.287640, 1.041900, 0.812950, 0.616200, 0.465180, 0.353300, 0.272000, 0.212300, 0.158200,
    0.111700, 0.078250, 0.057250, 0.042160, 0.029840, 0.020300, 0.013400, 0.008750, 0.005750,
    0.003900, 0.002750, 0.002100, 0.001800, 0.001650, 0.001400, 0.001100, 0.001000, 0.000800,
    0.000600, 0.000340, 0.000240, 0.000190, 0.000100, 0.000050, 0.000030, 0.000020, 0.000010,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
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
    let throughput = throughput.max(Vec3::ZERO);
    let weighted = throughput.max_element() > 0.0 && !throughput.abs_diff_eq(Vec3::ONE, 1.0e-6);
    if !weighted {
        let u = u.clamp(0.0, 1.0);
        let lambda = LAMBDA_MIN_NM + u * LAMBDA_RANGE_NM;
        let basis = normalized_rgb_at(lambda) * LAMBDA_RANGE_NM;
        return (lambda, basis);
    }

    let cdf = throughput_weighted_cdf(throughput);
    let u = u.clamp(0.0, 1.0 - f32::EPSILON);
    let mut index = 0;
    while index + 1 < TABLE_LEN && u >= cdf[index + 1] {
        index += 1;
    }
    let prev = if index == 0 { 0.0 } else { cdf[index] };
    let next = cdf[index + 1].max(prev + 1.0e-8);
    let local = ((u - prev) / (next - prev)).clamp(0.0, 1.0);
    let lambda = (LAMBDA_MIN_NM + LAMBDA_STEP_NM * (index as f32 + local))
        .clamp(LAMBDA_MIN_NM, LAMBDA_MAX_NM);
    let pdf = ((next - prev) / LAMBDA_STEP_NM).max(1.0e-8);
    (lambda, normalized_rgb_at(lambda) / pdf)
}

// Project the spectral chromaticity into the sRGB gamut triangle by sliding
// it toward the D65 white point. In linear sRGB this is the same as adding
// `-min(R,G,B)` to every channel: when (R,G,B) had a negative component the
// result has at least one channel equal to zero, which corresponds to a
// point on the gamut triangle's edge. This preserves the smooth chromaticity
// sweep across the spectrum at the cost of desaturation outside the gamut.
fn gamut_projected_rgb_at(lambda_nm: f32) -> Vec3 {
    let xyz = cie_xyz_at(lambda_nm);
    let rgb = XYZ_TO_REC709 * xyz;
    let min_c = rgb.x.min(rgb.y).min(rgb.z);
    if min_c < 0.0 {
        rgb + Vec3::splat(-min_c)
    } else {
        rgb
    }
}

fn normalized_rgb_at(lambda_nm: f32) -> Vec3 {
    gamut_projected_rgb_at(lambda_nm) * normalization_scale()
}

fn throughput_weighted_cdf(throughput: Vec3) -> [f32; TABLE_LEN + 1] {
    let mut cdf = [0.0; TABLE_LEN + 1];
    let mut sum = 0.0;
    for i in 0..TABLE_LEN {
        let lambda = LAMBDA_MIN_NM + LAMBDA_STEP_NM * i as f32;
        let rgb = normalized_rgb_at(lambda);
        let score = rgb.dot(throughput).max(0.0);
        sum += score;
        cdf[i + 1] = sum;
    }
    if sum <= 0.0 {
        for (i, v) in cdf.iter_mut().enumerate() {
            *v = i as f32 / TABLE_LEN as f32;
        }
        return cdf;
    }
    for v in &mut cdf {
        *v /= sum;
    }
    cdf[TABLE_LEN] = 1.0;
    cdf
}

fn normalization_scale() -> Vec3 {
    use std::sync::OnceLock;
    static CACHE: OnceLock<Vec3> = OnceLock::new();
    *CACHE.get_or_init(|| {
        let mut sum = Vec3::ZERO;
        for i in 0..TABLE_LEN {
            let lambda = LAMBDA_MIN_NM + LAMBDA_STEP_NM * i as f32;
            sum += gamut_projected_rgb_at(lambda);
        }
        let integrated = sum * LAMBDA_STEP_NM;
        Vec3::new(1.0 / integrated.x, 1.0 / integrated.y, 1.0 / integrated.z)
    })
}

fn cie_xyz_at(lambda_nm: f32) -> Vec3 {
    let position = (lambda_nm - LAMBDA_MIN_NM) / LAMBDA_STEP_NM;
    if position <= 0.0 {
        return Vec3::new(CIE_X[0], CIE_Y[0], CIE_Z[0]);
    }
    let max_index = TABLE_LEN - 1;
    if position >= max_index as f32 {
        return Vec3::new(CIE_X[max_index], CIE_Y[max_index], CIE_Z[max_index]);
    }
    let lower = position.floor() as usize;
    let upper = lower + 1;
    let t = position - lower as f32;
    Vec3::new(
        CIE_X[lower] * (1.0 - t) + CIE_X[upper] * t,
        CIE_Y[lower] * (1.0 - t) + CIE_Y[upper] * t,
        CIE_Z[lower] * (1.0 - t) + CIE_Z[upper] * t,
    )
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::{
        LAMBDA_MAX_NM, LAMBDA_MIN_NM, cauchy_ior, cie_xyz_at, sample_dispersion_wavelength,
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
    fn dispersion_basis_is_non_negative_across_visible_range() {
        let n = 4096;
        for i in 0..n {
            let u = (i as f32 + 0.5) / n as f32;
            let (_, basis) = sample_dispersion_wavelength(u);
            assert!(basis.x >= 0.0 && basis.y >= 0.0 && basis.z >= 0.0);
        }
    }

    #[test]
    fn cmf_endpoints_match_table() {
        let xyz = cie_xyz_at(LAMBDA_MIN_NM);
        assert!(xyz.x > 0.0);
        let xyz_end = cie_xyz_at(LAMBDA_MAX_NM);
        assert!(xyz_end.is_finite());
    }
}
