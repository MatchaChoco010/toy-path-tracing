use glam::{Vec2, Vec3};

fn hextile_hash(p: Vec2) -> Vec2 {
    let mut p3 = Vec3::new(p.x, p.y, p.x) * Vec3::new(0.1031, 0.1030, 0.0973);
    p3 -= p3.floor();
    let dot_val = p3.dot(Vec3::new(p3.y, p3.z, p3.x) + Vec3::splat(33.33));
    p3 += Vec3::splat(dot_val);
    let result = Vec2::new(p3.x + p3.y, p3.x + p3.z) * Vec2::new(p3.z, p3.y);
    result - result.floor()
}

pub fn schlick_gain(x: f32, r: f32) -> f32 {
    let rr = r.clamp(0.001, 0.999);
    let a = (1.0 / rr - 2.0) * (1.0 - 2.0 * x);
    if x < 0.5 {
        x / (a + 1.0)
    } else {
        (a - x) / (a - 1.0)
    }
}

pub struct HextileData {
    pub coords: [Vec2; 3],
    pub weights: Vec3,
    pub rotations: Vec3,
}

#[inline]
fn lerp_f(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

pub fn hextile_coord(
    coord: Vec2,
    rotation: f32,
    rotation_range_deg: Vec2,
    scale: f32,
    scale_range: Vec2,
    offset: f32,
    offset_range: Vec2,
) -> HextileData {
    let sqrt3_2 = 3.0_f32.sqrt() * 2.0;
    let st = coord * sqrt3_2;
    let st_skewed = Vec2::new(st.x + st.y * -0.57735027, st.y * 1.154_700_5);

    let st_frac = Vec2::new(st_skewed.x.rem_euclid(1.0), st_skewed.y.rem_euclid(1.0));
    let temp = Vec3::new(st_frac.x, st_frac.y, 1.0 - st_frac.x - st_frac.y);

    let s = if -temp.z >= 0.0 { 1.0 } else { 0.0 };
    let s2 = 2.0 * s - 1.0;
    let w1 = -temp.z * s2;
    let w2 = s - temp.y * s2;
    let w3 = s - temp.x * s2;

    let base_id_x = st_skewed.x.floor() as i32;
    let base_id_y = st_skewed.y.floor() as i32;
    let si = s as i32;
    let ids = [
        (base_id_x + si, base_id_y + si),
        (base_id_x + si, base_id_y + 1 - si),
        (base_id_x + 1 - si, base_id_y + si),
    ];

    let seed_offset = Vec2::splat(0.12345);
    let rr = Vec2::new(
        rotation_range_deg.x.to_radians(),
        rotation_range_deg.y.to_radians(),
    );

    let mut ctr = [Vec2::ZERO; 3];
    let mut rotations_arr = [0.0_f32; 3];
    let mut scales = [1.0_f32; 3];
    let mut offsets = [Vec2::ZERO; 3];
    for i in 0..3 {
        let (idx, idy) = ids[i];
        ctr[i] = Vec2::new(idx as f32 + idy as f32 * 0.5, idy as f32 / 1.154_700_5) / sqrt3_2;
        let rand = hextile_hash(Vec2::new(idx as f32, idy as f32) + seed_offset);
        rotations_arr[i] = lerp_f(rr.x, rr.y, rand.x * rotation);
        scales[i] = lerp_f(1.0, lerp_f(scale_range.x, scale_range.y, rand.y), scale);
        offsets[i] = Vec2::new(
            lerp_f(offset_range.x, offset_range.y, rand.x * offset),
            lerp_f(offset_range.x, offset_range.y, rand.y * offset),
        );
    }

    let mut coords = [Vec2::ZERO; 3];
    for i in 0..3 {
        let d = coord - ctr[i];
        let c = rotations_arr[i].cos();
        let s = rotations_arr[i].sin();
        coords[i] = Vec2::new(
            (d.x * c - d.y * s) / scales[i] + ctr[i].x + offsets[i].x,
            (d.x * s + d.y * c) / scales[i] + ctr[i].y + offsets[i].y,
        );
    }

    HextileData {
        coords,
        weights: Vec3::new(w1, w2, w3),
        rotations: Vec3::new(rotations_arr[0], rotations_arr[1], rotations_arr[2]),
    }
}

pub fn compute_blend_weights(luminance_weights: Vec3, tile_weights: Vec3, falloff: f32) -> Vec3 {
    let tw7 = Vec3::new(
        tile_weights.x.powi(7),
        tile_weights.y.powi(7),
        tile_weights.z.powi(7),
    );
    let mut w = luminance_weights * tw7;
    let sum = w.x + w.y + w.z;
    w /= sum;
    if falloff != 0.5 {
        w = Vec3::new(
            schlick_gain(w.x, falloff),
            schlick_gain(w.y, falloff),
            schlick_gain(w.z, falloff),
        );
        let sum = w.x + w.y + w.z;
        w /= sum;
    }
    w
}

pub fn normals_to_gradient(n: Vec3, np: Vec3) -> Vec3 {
    let d = n.dot(np);
    (d * n - np) / d.abs().max(f32::MIN_POSITIVE)
}

pub fn gradient_blend_3_normals(
    n: Vec3,
    n1: Vec3,
    w1: f32,
    n2: Vec3,
    w2: f32,
    n3: Vec3,
    w3: f32,
) -> Vec3 {
    let w1 = w1.clamp(0.0, 1.0);
    let w2 = w2.clamp(0.0, 1.0);
    let w3 = w3.clamp(0.0, 1.0);
    let g1 = normals_to_gradient(n, n1);
    let g2 = normals_to_gradient(n, n2);
    let g3 = normals_to_gradient(n, n3);
    let gg = w1 * g1 + w2 * g2 + w3 * g3;
    (n - gg).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_to_one() {
        let h = hextile_coord(
            Vec2::new(0.3, 0.7),
            0.0,
            Vec2::new(0.0, 360.0),
            1.0,
            Vec2::new(1.0, 1.0),
            0.0,
            Vec2::new(0.0, 0.0),
        );
        let s = h.weights.x + h.weights.y + h.weights.z;
        assert!((s - 1.0).abs() < 1.0e-3, "weight sum = {}", s);
    }

    #[test]
    fn schlick_gain_endpoints() {
        assert!((schlick_gain(0.0, 0.5) - 0.0).abs() < 1.0e-5);
        assert!((schlick_gain(1.0, 0.5) - 1.0).abs() < 1.0e-5);
        assert!((schlick_gain(0.5, 0.5) - 0.5).abs() < 1.0e-3);
    }
}
