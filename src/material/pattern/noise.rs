use glam::{Vec2, Vec3};

#[inline]
fn rotl32(x: u32, k: u32) -> u32 {
    x.rotate_left(k)
}

#[inline]
fn bjmix(mut a: u32, mut b: u32, mut c: u32) -> (u32, u32, u32) {
    a = a.wrapping_sub(c);
    a ^= rotl32(c, 4);
    c = c.wrapping_add(b);
    b = b.wrapping_sub(a);
    b ^= rotl32(a, 6);
    a = a.wrapping_add(c);
    c = c.wrapping_sub(b);
    c ^= rotl32(b, 8);
    b = b.wrapping_add(a);
    a = a.wrapping_sub(c);
    a ^= rotl32(c, 16);
    c = c.wrapping_add(b);
    b = b.wrapping_sub(a);
    b ^= rotl32(a, 19);
    a = a.wrapping_add(c);
    c = c.wrapping_sub(b);
    c ^= rotl32(b, 4);
    b = b.wrapping_add(a);
    (a, b, c)
}

#[inline]
fn bjfinal(mut a: u32, mut b: u32, mut c: u32) -> u32 {
    c ^= b;
    c = c.wrapping_sub(rotl32(b, 14));
    a ^= c;
    a = a.wrapping_sub(rotl32(c, 11));
    b ^= a;
    b = b.wrapping_sub(rotl32(a, 25));
    c ^= b;
    c = c.wrapping_sub(rotl32(b, 16));
    a ^= c;
    a = a.wrapping_sub(rotl32(c, 4));
    b ^= a;
    b = b.wrapping_sub(rotl32(a, 14));
    c ^= b;
    c = c.wrapping_sub(rotl32(b, 24));
    c
}

#[inline]
fn seed_for_len(len: u32) -> u32 {
    0xdeadbeef_u32.wrapping_add(len << 2).wrapping_add(13)
}

fn hash_int2(x: i32, y: i32) -> u32 {
    let s = seed_for_len(2);
    let a = s.wrapping_add(x as u32);
    let b = s.wrapping_add(y as u32);
    bjfinal(a, b, s)
}

fn hash_int3(x: i32, y: i32, z: i32) -> u32 {
    let s = seed_for_len(3);
    let a = s.wrapping_add(x as u32);
    let b = s.wrapping_add(y as u32);
    let c = s.wrapping_add(z as u32);
    bjfinal(a, b, c)
}

fn hash_int4(x: i32, y: i32, z: i32, xx: i32) -> u32 {
    let s = seed_for_len(4);
    let (a, b, c) = bjmix(
        s.wrapping_add(x as u32),
        s.wrapping_add(y as u32),
        s.wrapping_add(z as u32),
    );
    let a = a.wrapping_add(xx as u32);
    bjfinal(a, b, c)
}

fn hash_int3_split2(x: i32, y: i32) -> (u32, u32, u32) {
    let h = hash_int2(x, y);
    (h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF)
}

fn hash_int3_split3(x: i32, y: i32, z: i32) -> (u32, u32, u32) {
    let h = hash_int3(x, y, z);
    (h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF)
}

#[inline]
fn bits_to_01(bits: u32) -> f32 {
    if bits <= i32::MAX as u32 {
        bits as f32 / 4_294_967_295.0
    } else {
        (bits >> 1) as f32 / 2_147_483_647.0
    }
}

#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn bilerp(v0: f32, v1: f32, v2: f32, v3: f32, s: f32, t: f32) -> f32 {
    let s1 = 1.0 - s;
    (1.0 - t) * (v0 * s1 + v1 * s) + t * (v2 * s1 + v3 * s)
}

#[inline]
fn trilerp(
    v0: f32,
    v1: f32,
    v2: f32,
    v3: f32,
    v4: f32,
    v5: f32,
    v6: f32,
    v7: f32,
    s: f32,
    t: f32,
    r: f32,
) -> f32 {
    let s1 = 1.0 - s;
    let t1 = 1.0 - t;
    let r1 = 1.0 - r;
    r1 * (t1 * (v0 * s1 + v1 * s) + t * (v2 * s1 + v3 * s))
        + r * (t1 * (v4 * s1 + v5 * s) + t * (v6 * s1 + v7 * s))
}

fn gradient_2d(h: u32, x: f32, y: f32) -> f32 {
    let h = h & 7;
    let u = if h < 4 { x } else { y };
    let v = 2.0 * if h < 4 { y } else { x };
    let signed_u = if h & 1 != 0 { -u } else { u };
    let signed_v = if h & 2 != 0 { -v } else { v };
    signed_u + signed_v
}

fn gradient_3d(h: u32, x: f32, y: f32, z: f32) -> f32 {
    let h = h & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        z
    };
    let signed_u = if h & 1 != 0 { -u } else { u };
    let signed_v = if h & 2 != 0 { -v } else { v };
    signed_u + signed_v
}

const GRADIENT_SCALE_2D: f32 = 0.6616;
const GRADIENT_SCALE_3D: f32 = 0.9820;

pub fn perlin2d(p: Vec2) -> f32 {
    let xi = p.x.floor();
    let yi = p.y.floor();
    let fx = p.x - xi;
    let fy = p.y - yi;
    let x = xi as i32;
    let y = yi as i32;
    let u = fade(fx);
    let v = fade(fy);
    let result = bilerp(
        gradient_2d(hash_int2(x, y), fx, fy),
        gradient_2d(hash_int2(x + 1, y), fx - 1.0, fy),
        gradient_2d(hash_int2(x, y + 1), fx, fy - 1.0),
        gradient_2d(hash_int2(x + 1, y + 1), fx - 1.0, fy - 1.0),
        u,
        v,
    );
    GRADIENT_SCALE_2D * result
}

pub fn perlin3d(p: Vec3) -> f32 {
    let xi = p.x.floor();
    let yi = p.y.floor();
    let zi = p.z.floor();
    let fx = p.x - xi;
    let fy = p.y - yi;
    let fz = p.z - zi;
    let x = xi as i32;
    let y = yi as i32;
    let z = zi as i32;
    let u = fade(fx);
    let v = fade(fy);
    let w = fade(fz);
    let result = trilerp(
        gradient_3d(hash_int3(x, y, z), fx, fy, fz),
        gradient_3d(hash_int3(x + 1, y, z), fx - 1.0, fy, fz),
        gradient_3d(hash_int3(x, y + 1, z), fx, fy - 1.0, fz),
        gradient_3d(hash_int3(x + 1, y + 1, z), fx - 1.0, fy - 1.0, fz),
        gradient_3d(hash_int3(x, y, z + 1), fx, fy, fz - 1.0),
        gradient_3d(hash_int3(x + 1, y, z + 1), fx - 1.0, fy, fz - 1.0),
        gradient_3d(hash_int3(x, y + 1, z + 1), fx, fy - 1.0, fz - 1.0),
        gradient_3d(hash_int3(x + 1, y + 1, z + 1), fx - 1.0, fy - 1.0, fz - 1.0),
        u,
        v,
        w,
    );
    GRADIENT_SCALE_3D * result
}

pub fn perlin2d_vec3(p: Vec2) -> Vec3 {
    let xi = p.x.floor();
    let yi = p.y.floor();
    let fx = p.x - xi;
    let fy = p.y - yi;
    let x = xi as i32;
    let y = yi as i32;
    let u = fade(fx);
    let v = fade(fy);
    let g = |xi: i32, yi: i32, fx: f32, fy: f32| -> Vec3 {
        let (h0, h1, h2) = hash_int3_split2(xi, yi);
        Vec3::new(
            gradient_2d(h0, fx, fy),
            gradient_2d(h1, fx, fy),
            gradient_2d(h2, fx, fy),
        )
    };
    let v0 = g(x, y, fx, fy);
    let v1 = g(x + 1, y, fx - 1.0, fy);
    let v2 = g(x, y + 1, fx, fy - 1.0);
    let v3 = g(x + 1, y + 1, fx - 1.0, fy - 1.0);
    let s1 = 1.0 - u;
    let r = (1.0 - v) * (v0 * s1 + v1 * u) + v * (v2 * s1 + v3 * u);
    GRADIENT_SCALE_2D * r
}

pub fn perlin3d_vec3(p: Vec3) -> Vec3 {
    let xi = p.x.floor();
    let yi = p.y.floor();
    let zi = p.z.floor();
    let fx = p.x - xi;
    let fy = p.y - yi;
    let fz = p.z - zi;
    let x = xi as i32;
    let y = yi as i32;
    let z = zi as i32;
    let u = fade(fx);
    let v = fade(fy);
    let w = fade(fz);
    let g = |xi: i32, yi: i32, zi: i32, fx: f32, fy: f32, fz: f32| -> Vec3 {
        let (h0, h1, h2) = hash_int3_split3(xi, yi, zi);
        Vec3::new(
            gradient_3d(h0, fx, fy, fz),
            gradient_3d(h1, fx, fy, fz),
            gradient_3d(h2, fx, fy, fz),
        )
    };
    let s1 = 1.0 - u;
    let t1 = 1.0 - v;
    let r1 = 1.0 - w;
    let v0 = g(x, y, z, fx, fy, fz);
    let v1 = g(x + 1, y, z, fx - 1.0, fy, fz);
    let v2 = g(x, y + 1, z, fx, fy - 1.0, fz);
    let v3 = g(x + 1, y + 1, z, fx - 1.0, fy - 1.0, fz);
    let v4 = g(x, y, z + 1, fx, fy, fz - 1.0);
    let v5 = g(x + 1, y, z + 1, fx - 1.0, fy, fz - 1.0);
    let v6 = g(x, y + 1, z + 1, fx, fy - 1.0, fz - 1.0);
    let v7 = g(x + 1, y + 1, z + 1, fx - 1.0, fy - 1.0, fz - 1.0);
    let r = r1 * (t1 * (v0 * s1 + v1 * u) + v * (v2 * s1 + v3 * u))
        + w * (t1 * (v4 * s1 + v5 * u) + v * (v6 * s1 + v7 * u));
    GRADIENT_SCALE_3D * r
}

pub fn cellnoise2d(p: Vec2) -> f32 {
    let ix = p.x.floor() as i32;
    let iy = p.y.floor() as i32;
    bits_to_01(hash_int2(ix, iy))
}

pub fn cellnoise3d(p: Vec3) -> f32 {
    let ix = p.x.floor() as i32;
    let iy = p.y.floor() as i32;
    let iz = p.z.floor() as i32;
    bits_to_01(hash_int3(ix, iy, iz))
}

pub fn cellnoise2d_vec3(p: Vec2) -> Vec3 {
    let ix = p.x.floor() as i32;
    let iy = p.y.floor() as i32;
    Vec3::new(
        bits_to_01(hash_int3(ix, iy, 0)),
        bits_to_01(hash_int3(ix, iy, 1)),
        bits_to_01(hash_int3(ix, iy, 2)),
    )
}

pub fn cellnoise3d_vec3(p: Vec3) -> Vec3 {
    let ix = p.x.floor() as i32;
    let iy = p.y.floor() as i32;
    let iz = p.z.floor() as i32;
    Vec3::new(
        bits_to_01(hash_int4(ix, iy, iz, 0)),
        bits_to_01(hash_int4(ix, iy, iz, 1)),
        bits_to_01(hash_int4(ix, iy, iz, 2)),
    )
}

pub fn fbm2d(p: Vec2, octaves: u32, lacunarity: f32, diminish: f32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut q = p;
    for _ in 0..octaves {
        sum += perlin2d(q) * amp;
        amp *= diminish;
        q *= lacunarity;
    }
    sum
}

pub fn fbm3d(p: Vec3, octaves: u32, lacunarity: f32, diminish: f32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut q = p;
    for _ in 0..octaves {
        sum += perlin3d(q) * amp;
        amp *= diminish;
        q *= lacunarity;
    }
    sum
}

pub fn fbm2d_vec3(p: Vec2, octaves: u32, lacunarity: f32, diminish: f32) -> Vec3 {
    let mut sum = Vec3::ZERO;
    let mut amp = 1.0;
    let mut q = p;
    for _ in 0..octaves {
        sum += perlin2d_vec3(q) * amp;
        amp *= diminish;
        q *= lacunarity;
    }
    sum
}

pub fn fbm3d_vec3(p: Vec3, octaves: u32, lacunarity: f32, diminish: f32) -> Vec3 {
    let mut sum = Vec3::ZERO;
    let mut amp = 1.0;
    let mut q = p;
    for _ in 0..octaves {
        sum += perlin3d_vec3(q) * amp;
        amp *= diminish;
        q *= lacunarity;
    }
    sum
}

pub fn worley2d_solid_vec3(p: Vec2, jitter: f32) -> Vec3 {
    let xi = p.x.floor();
    let yi = p.y.floor();
    let x = xi as i32;
    let y = yi as i32;
    let local = Vec2::new(p.x - xi, p.y - yi);
    let mut sqdist = 1.0e6_f32;
    let mut minpos = Vec2::ZERO;
    for dx in -1..=1 {
        for dy in -1..=1 {
            let cell = cellnoise2d_vec3(Vec2::new((x + dx) as f32, (y + dy) as f32));
            let mut off = Vec2::new(cell.x, cell.y);
            off -= Vec2::splat(0.5);
            off *= jitter;
            off += Vec2::splat(0.5);
            let cellpos = Vec2::new(dx as f32, dy as f32) + off;
            let diff = cellpos - local;
            let d = diff.dot(diff);
            if d < sqdist {
                sqdist = d;
                minpos = cellpos - local;
            }
        }
    }
    cellnoise2d_vec3(minpos + p)
}

pub fn worley3d_solid_vec3(p: Vec3, jitter: f32) -> Vec3 {
    let xi = p.x.floor();
    let yi = p.y.floor();
    let zi = p.z.floor();
    let x = xi as i32;
    let y = yi as i32;
    let z = zi as i32;
    let local = Vec3::new(p.x - xi, p.y - yi, p.z - zi);
    let mut sqdist = 1.0e6_f32;
    let mut minpos = Vec3::ZERO;
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                let cell =
                    cellnoise3d_vec3(Vec3::new((x + dx) as f32, (y + dy) as f32, (z + dz) as f32));
                let mut off = cell;
                off -= Vec3::splat(0.5);
                off *= jitter;
                off += Vec3::splat(0.5);
                let cellpos = Vec3::new(dx as f32, dy as f32, dz as f32) + off;
                let diff = cellpos - local;
                let d = diff.dot(diff);
                if d < sqdist {
                    sqdist = d;
                    minpos = cellpos - local;
                }
            }
        }
    }
    cellnoise3d_vec3(minpos + p)
}

fn worley_distance_2d(
    p: Vec2,
    x: i32,
    y: i32,
    xoff: i32,
    yoff: i32,
    jitter: f32,
    metric: i32,
) -> f32 {
    let cell = cellnoise2d_vec3(Vec2::new((x + xoff) as f32, (y + yoff) as f32));
    let mut off = Vec2::new(cell.x, cell.y);
    off -= Vec2::splat(0.5);
    off *= jitter;
    off += Vec2::splat(0.5);
    let cellpos = Vec2::new(x as f32, y as f32) + off;
    let diff = cellpos - p;
    match metric {
        2 => diff.x.abs() + diff.y.abs(),
        3 => diff.x.abs().max(diff.y.abs()),
        _ => diff.dot(diff),
    }
}

fn worley_distance_3d(
    p: Vec3,
    x: i32,
    y: i32,
    z: i32,
    xo: i32,
    yo: i32,
    zo: i32,
    jitter: f32,
    metric: i32,
) -> f32 {
    let cell = cellnoise3d_vec3(Vec3::new((x + xo) as f32, (y + yo) as f32, (z + zo) as f32));
    let mut off = cell;
    off -= Vec3::splat(0.5);
    off *= jitter;
    off += Vec3::splat(0.5);
    let cellpos = Vec3::new(x as f32, y as f32, z as f32) + off;
    let diff = cellpos - p;
    match metric {
        2 => diff.x.abs() + diff.y.abs() + diff.z.abs(),
        3 => diff.x.abs().max(diff.y.abs()).max(diff.z.abs()),
        _ => diff.dot(diff),
    }
}

pub fn worley2d(p: Vec2, jitter: f32) -> f32 {
    worley2d_metric(p, jitter, 0)
}

pub fn worley3d(p: Vec3, jitter: f32) -> f32 {
    worley3d_metric(p, jitter, 0)
}

pub fn worley2d_solid(p: Vec2, jitter: f32) -> f32 {
    let xi = p.x.floor();
    let yi = p.y.floor();
    let x = xi as i32;
    let y = yi as i32;
    let local = Vec2::new(p.x - xi, p.y - yi);
    let mut sqdist = 1.0e6_f32;
    let mut minpos = Vec2::ZERO;
    for dx in -1..=1 {
        for dy in -1..=1 {
            let cell = cellnoise2d_vec3(Vec2::new((x + dx) as f32, (y + dy) as f32));
            let mut off = Vec2::new(cell.x, cell.y);
            off -= Vec2::splat(0.5);
            off *= jitter;
            off += Vec2::splat(0.5);
            let cellpos = Vec2::new(dx as f32, dy as f32) + off;
            let diff = cellpos - local;
            let d = diff.dot(diff);
            if d < sqdist {
                sqdist = d;
                minpos = cellpos - local;
            }
        }
    }
    cellnoise2d(minpos + p)
}

pub fn worley3d_solid(p: Vec3, jitter: f32) -> f32 {
    let xi = p.x.floor();
    let yi = p.y.floor();
    let zi = p.z.floor();
    let x = xi as i32;
    let y = yi as i32;
    let z = zi as i32;
    let local = Vec3::new(p.x - xi, p.y - yi, p.z - zi);
    let mut sqdist = 1.0e6_f32;
    let mut minpos = Vec3::ZERO;
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                let cell =
                    cellnoise3d_vec3(Vec3::new((x + dx) as f32, (y + dy) as f32, (z + dz) as f32));
                let mut off = cell;
                off -= Vec3::splat(0.5);
                off *= jitter;
                off += Vec3::splat(0.5);
                let cellpos = Vec3::new(dx as f32, dy as f32, dz as f32) + off;
                let diff = cellpos - local;
                let d = diff.dot(diff);
                if d < sqdist {
                    sqdist = d;
                    minpos = cellpos - local;
                }
            }
        }
    }
    cellnoise3d(minpos + p)
}

pub fn worley2d_metric(p: Vec2, jitter: f32, metric: i32) -> f32 {
    let xi = p.x.floor();
    let yi = p.y.floor();
    let x = xi as i32;
    let y = yi as i32;
    let local = Vec2::new(p.x - xi, p.y - yi);
    let mut sqdist = 1.0e6_f32;
    for dx in -1..=1 {
        for dy in -1..=1 {
            let d = worley_distance_2d(local, dx, dy, x, y, jitter, metric);
            if d < sqdist {
                sqdist = d;
            }
        }
    }
    if metric == 0 { sqdist.sqrt() } else { sqdist }
}

pub fn worley2d_top2(p: Vec2, jitter: f32, metric: i32) -> Vec2 {
    let xi = p.x.floor();
    let yi = p.y.floor();
    let x = xi as i32;
    let y = yi as i32;
    let local = Vec2::new(p.x - xi, p.y - yi);
    let mut d = Vec2::splat(1.0e6_f32);
    for dx in -1..=1 {
        for dy in -1..=1 {
            let dist = worley_distance_2d(local, dx, dy, x, y, jitter, metric);
            if dist < d.x {
                d.y = d.x;
                d.x = dist;
            } else if dist < d.y {
                d.y = dist;
            }
        }
    }
    if metric == 0 {
        Vec2::new(d.x.sqrt(), d.y.sqrt())
    } else {
        d
    }
}

pub fn worley2d_top3(p: Vec2, jitter: f32, metric: i32) -> Vec3 {
    let xi = p.x.floor();
    let yi = p.y.floor();
    let x = xi as i32;
    let y = yi as i32;
    let local = Vec2::new(p.x - xi, p.y - yi);
    let mut d = Vec3::splat(1.0e6_f32);
    for dx in -1..=1 {
        for dy in -1..=1 {
            let dist = worley_distance_2d(local, dx, dy, x, y, jitter, metric);
            if dist < d.x {
                d.z = d.y;
                d.y = d.x;
                d.x = dist;
            } else if dist < d.y {
                d.z = d.y;
                d.y = dist;
            } else if dist < d.z {
                d.z = dist;
            }
        }
    }
    if metric == 0 {
        Vec3::new(d.x.sqrt(), d.y.sqrt(), d.z.sqrt())
    } else {
        d
    }
}

pub fn worley3d_metric(p: Vec3, jitter: f32, metric: i32) -> f32 {
    let xi = p.x.floor();
    let yi = p.y.floor();
    let zi = p.z.floor();
    let x = xi as i32;
    let y = yi as i32;
    let z = zi as i32;
    let local = Vec3::new(p.x - xi, p.y - yi, p.z - zi);
    let mut sqdist = 1.0e6_f32;
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                let d = worley_distance_3d(local, dx, dy, dz, x, y, z, jitter, metric);
                if d < sqdist {
                    sqdist = d;
                }
            }
        }
    }
    if metric == 0 { sqdist.sqrt() } else { sqdist }
}

pub fn worley3d_top2(p: Vec3, jitter: f32, metric: i32) -> Vec2 {
    let xi = p.x.floor();
    let yi = p.y.floor();
    let zi = p.z.floor();
    let x = xi as i32;
    let y = yi as i32;
    let z = zi as i32;
    let local = Vec3::new(p.x - xi, p.y - yi, p.z - zi);
    let mut d = Vec2::splat(1.0e6_f32);
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                let dist = worley_distance_3d(local, dx, dy, dz, x, y, z, jitter, metric);
                if dist < d.x {
                    d.y = d.x;
                    d.x = dist;
                } else if dist < d.y {
                    d.y = dist;
                }
            }
        }
    }
    if metric == 0 {
        Vec2::new(d.x.sqrt(), d.y.sqrt())
    } else {
        d
    }
}

pub fn worley3d_top3(p: Vec3, jitter: f32, metric: i32) -> Vec3 {
    let xi = p.x.floor();
    let yi = p.y.floor();
    let zi = p.z.floor();
    let x = xi as i32;
    let y = yi as i32;
    let z = zi as i32;
    let local = Vec3::new(p.x - xi, p.y - yi, p.z - zi);
    let mut d = Vec3::splat(1.0e6_f32);
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                let dist = worley_distance_3d(local, dx, dy, dz, x, y, z, jitter, metric);
                if dist < d.x {
                    d.z = d.y;
                    d.y = d.x;
                    d.x = dist;
                } else if dist < d.y {
                    d.z = d.y;
                    d.y = dist;
                } else if dist < d.z {
                    d.z = dist;
                }
            }
        }
    }
    if metric == 0 {
        Vec3::new(d.x.sqrt(), d.y.sqrt(), d.z.sqrt())
    } else {
        d
    }
}

fn random_float_from_cellnoise(x: f32, seed: i32, lo: f32, hi: f32) -> f32 {
    let v = cellnoise2d(Vec2::new(x, seed as f32));
    lo + (hi - lo) * v
}

pub fn random_float(input: f32, seed: i32, lo: f32, hi: f32, integer_input: bool) -> f32 {
    let x = if integer_input { input } else { input * 4096.0 };
    random_float_from_cellnoise(x, seed, lo, hi)
}

pub fn random_color(
    input: f32,
    seed: i32,
    hue_lo: f32,
    hue_hi: f32,
    sat_lo: f32,
    sat_hi: f32,
    val_lo: f32,
    val_hi: f32,
) -> Vec3 {
    let hue_seed = (seed as f32 + 413.3).ceil() as i32;
    let sat_seed = (seed as f32 + 1522.4).ceil() as i32;
    let val_seed = (seed as f32 + 1813.8).ceil() as i32;
    let hue = random_float(input, hue_seed, hue_lo, hue_hi, false);
    let sat = random_float(input, sat_seed, sat_lo, sat_hi, false);
    let val = random_float(input, val_seed, val_lo, val_hi, false);
    hsv_to_rgb(hue, sat, val)
}

pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Vec3 {
    let h = h.rem_euclid(1.0);
    let i = (h * 6.0).floor() as i32;
    let f = h * 6.0 - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match i.rem_euclid(6) {
        0 => Vec3::new(v, t, p),
        1 => Vec3::new(q, v, p),
        2 => Vec3::new(p, v, t),
        3 => Vec3::new(p, q, v),
        4 => Vec3::new(t, p, v),
        _ => Vec3::new(v, p, q),
    }
}

pub fn rgb_to_hsv(c: Vec3) -> Vec3 {
    let max = c.x.max(c.y).max(c.z);
    let min = c.x.min(c.y).min(c.z);
    let range = max - min;
    let inv_range = (1.0 / 6.0) / range;
    let s = if max > f32::MIN_POSITIVE {
        range / max
    } else {
        0.0
    };
    let h = if s != 0.0 {
        if max == c.x {
            (c.y - c.z) * inv_range
        } else if max == c.y {
            (2.0 / 6.0) + (c.z - c.x) * inv_range
        } else {
            (4.0 / 6.0) + (c.x - c.y) * inv_range
        }
    } else {
        0.0
    };
    Vec3::new(if h >= 0.0 { h } else { h + 1.0 }, s, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cellnoise_in_unit_range() {
        for i in 0..16 {
            let v = cellnoise2d(Vec2::new(i as f32, i as f32 * 0.5));
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn perlin2d_zero_at_lattice_points() {
        let v = perlin2d(Vec2::new(0.0, 0.0));
        assert!(v.abs() < 1.0e-6, "expected zero at lattice, got {}", v);
    }

    #[test]
    fn perlin3d_zero_at_lattice_points() {
        let v = perlin3d(Vec3::new(0.0, 0.0, 0.0));
        assert!(v.abs() < 1.0e-6);
    }

    #[test]
    fn worley2d_is_non_negative() {
        for x in 0..8 {
            for y in 0..8 {
                let v = worley2d(Vec2::new(x as f32 * 0.7, y as f32 * 0.7), 1.0);
                assert!(v >= 0.0);
            }
        }
    }

    #[test]
    fn fade_endpoints() {
        assert!((fade(0.0) - 0.0).abs() < 1.0e-6);
        assert!((fade(1.0) - 1.0).abs() < 1.0e-6);
        assert!((fade(0.5) - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn bits_to_01_matches_mdl_signed_int_mapping() {
        assert!(
            (bits_to_01(0x7fff_ffff) - (0x7fff_ffff_u32 as f32 / 4_294_967_295.0)).abs() < 1.0e-8
        );
        assert!(
            (bits_to_01(0x8000_0000) - (0x4000_0000_u32 as f32 / 2_147_483_647.0)).abs() < 1.0e-8
        );
        assert!(
            (bits_to_01(0xffff_ffff) - (0x7fff_ffff_u32 as f32 / 2_147_483_647.0)).abs() < 1.0e-8
        );
    }

    #[test]
    fn fbm_zero_octaves_matches_mdl_empty_loop() {
        assert_eq!(fbm2d(Vec2::new(0.37, 0.61), 0, 2.0, 0.5), 0.0);
        assert_eq!(fbm3d(Vec3::new(0.37, 0.61, 0.23), 0, 2.0, 0.5), 0.0);
        assert_eq!(fbm2d_vec3(Vec2::new(0.37, 0.61), 0, 2.0, 0.5), Vec3::ZERO);
        assert_eq!(
            fbm3d_vec3(Vec3::new(0.37, 0.61, 0.23), 0, 2.0, 0.5),
            Vec3::ZERO
        );
    }

    #[test]
    fn gradient_2d_matches_mdl_bit_pattern() {
        let v0 = gradient_2d(0, 0.3, 0.7);
        let expected = 0.3 + 2.0 * 0.7;
        assert!((v0 - expected).abs() < 1.0e-6, "h=0: u + 2v");
        let v3 = gradient_2d(3, 0.3, 0.7);
        let expected = -0.3 - 2.0 * 0.7;
        assert!(
            (v3 - expected).abs() < 1.0e-6,
            "h=3 (h&1=1, h&2=1): -u - 2v"
        );
        let v4 = gradient_2d(4, 0.3, 0.7);
        let expected = 0.7 + 2.0 * 0.3;
        assert!(
            (v4 - expected).abs() < 1.0e-6,
            "h=4 (h>=4): swap to (y, 2x)"
        );
    }

    #[test]
    fn worley_manhattan_3d_includes_z() {
        let p_xy_only = Vec3::new(0.7, 0.5, 0.5);
        let p_with_z = Vec3::new(0.7, 0.5, 0.8);
        let m_xy = worley3d_metric(p_xy_only, 0.0, 2);
        let m_z = worley3d_metric(p_with_z, 0.0, 2);
        assert!(m_z != m_xy);
    }

    #[test]
    fn mdl_noise_numeric_spots() {
        let p2 = Vec2::new(0.37, 0.61);
        let p3 = Vec3::new(0.37, 0.61, 0.23);
        assert!((perlin2d(p2) - 0.08116639).abs() < 1.0e-6);
        assert!(
            (perlin2d_vec3(p2) - Vec3::new(0.08116639, 0.6858292, -0.3733143))
                .abs()
                .max_element()
                < 1.0e-6
        );
        assert!((perlin3d(p3) - 0.17001536).abs() < 1.0e-6);
        assert!(
            (perlin3d_vec3(p3) - Vec3::new(0.17001536, -0.24784607, 0.34607157))
                .abs()
                .max_element()
                < 1.0e-6
        );
        assert!((cellnoise2d(p2) - 0.86031276).abs() < 1.0e-6);
        assert!((cellnoise3d(p3) - 0.61106765).abs() < 1.0e-6);
        assert!(
            (worley2d_top3(p2, 1.0, 0) - Vec3::new(0.48789248, 0.6037186, 1.094374))
                .abs()
                .max_element()
                < 1.0e-6
        );
        assert!(
            (worley3d_top3(p3, 1.0, 0) - Vec3::new(0.40005127, 0.8472812, 0.94316894))
                .abs()
                .max_element()
                < 1.0e-6
        );
    }
}
