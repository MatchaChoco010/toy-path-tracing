use clap::Parser;
use glam::{UVec2, Vec2, Vec3};
use image::RgbImage;
use rand::RngExt;
use rayon::prelude::*;
use std::{
    f32::consts::{PI, TAU},
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use toy_path_tracing::{
    ray::Ray,
    scene::{Material, Scene},
    scenes::load_scene,
};

#[derive(Debug, Parser)]
struct Args {
    #[arg(short = 'o', long = "output", default_value = "result/output.png")]
    output: PathBuf,

    #[arg(long = "scene", default_value_t = 0)]
    scene: u32,

    #[arg(long = "spp", default_value_t = 32, value_parser = clap::value_parser!(u32).range(1..))]
    spp: u32,

    #[arg(long = "depth", default_value_t = 16, value_parser = clap::value_parser!(u32).range(1..))]
    depth: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let resolution = UVec2::new(512, 512);
    let (mut scene, camera) = load_scene(args.scene)?;
    let build_bvh_start = Instant::now();
    scene.build_bvh();
    println!("build_bvh: {}", format_duration(build_bvh_start.elapsed()));

    let mut pixels = vec![0_u8; (resolution.x * resolution.y * 3) as usize];
    let intersect_start = Instant::now();
    pixels
        .par_chunks_mut(3)
        .enumerate()
        .for_each_init(rand::rng, |rng, (index, pixel)| {
            let x = (index as u32) % resolution.x;
            let y = (index as u32) / resolution.x;
            let mut color = Vec3::ZERO;

            for sample_index in 0..args.spp {
                let us = Vec2::new(rng.random::<f32>(), rng.random::<f32>());
                let ray = camera.generate_ray(resolution, UVec2::new(x, y), us);
                let sample = trace_radiance(&scene, ray, rng, args.depth);
                let sample_count = (sample_index + 1) as f32;
                color += (sample - color) / sample_count;
            }

            let mapped = reinhard(color);
            pixel[0] = float_to_u8(mapped.x);
            pixel[1] = float_to_u8(mapped.y);
            pixel[2] = float_to_u8(mapped.z);
        });
    println!("intersect: {}", format_duration(intersect_start.elapsed()));

    let image = RgbImage::from_raw(resolution.x, resolution.y, pixels)
        .expect("pixel buffer size must match the image resolution");
    create_output_directory(&args.output)?;
    image.save(&args.output)?;

    Ok(())
}

fn float_to_u8(value: f32) -> u8 {
    (255.0 * value.clamp(0.0, 1.0)) as u8
}

fn trace_radiance(
    scene: &Scene,
    initial_ray: Ray,
    rng: &mut rand::rngs::ThreadRng,
    max_depth: u32,
) -> Vec3 {
    let mut radiance = Vec3::ZERO;
    let mut throughput = Vec3::ONE;
    let mut ray = initial_ray;

    for _ in 0..max_depth {
        let Some(hit) = scene
            .closest_hit(&ray)
            .expect("scene.build_bvh() must be called before traversal")
        else {
            break;
        };

        let [p0, p1, p2] = scene.triangle_positions(hit.triangle);
        let [n0, n1, n2] = scene.triangle_normals(hit.triangle);
        let hit_position = hit.barycentric.x * p0 + hit.barycentric.y * p1 + hit.barycentric.z * p2;
        let geometric_normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        let shading_normal =
            (hit.barycentric.x * n0 + hit.barycentric.y * n1 + hit.barycentric.z * n2)
                .normalize_or_zero();
        let mut normal = if shading_normal.length_squared() > 0.0 {
            shading_normal
        } else {
            geometric_normal
        };

        if normal.dot(-ray.direction) < 0.0 {
            normal = -normal;
        }

        match scene.instance_material(hit.triangle.instance_index) {
            Material::Emissive { color, strength } => {
                radiance += throughput * (color * strength);
                break;
            }
            Material::Diffuse { rho } => {
                let us = Vec2::new(rng.random::<f32>(), rng.random::<f32>());
                let local_direction = sample_uniform_hemisphere(us);
                let next_direction = local_to_world(local_direction, normal);
                let cos_theta = next_direction.dot(normal).max(0.0);
                if cos_theta <= 0.0 {
                    break;
                }

                let pdf = 1.0 / (2.0 * PI);
                let bsdf = rho / PI;
                throughput *= bsdf * (cos_theta / pdf);
                ray = Ray::new(hit_position + 1.0e-4 * normal, next_direction);
            }
        }
    }

    radiance
}

fn sample_uniform_hemisphere(us: Vec2) -> Vec3 {
    let z = us.x;
    let r = (1.0 - z * z).sqrt();
    let phi = TAU * us.y;

    Vec3::new(r * phi.cos(), r * phi.sin(), z)
}

fn local_to_world(local_direction: Vec3, normal: Vec3) -> Vec3 {
    let tangent = build_tangent(normal);
    let bitangent = normal.cross(tangent);

    (local_direction.x * tangent + local_direction.y * bitangent + local_direction.z * normal)
        .normalize()
}

fn build_tangent(normal: Vec3) -> Vec3 {
    let axis = if normal.z.abs() < 0.999 {
        Vec3::Z
    } else {
        Vec3::X
    };
    normal.cross(axis).normalize()
}

fn reinhard(color: Vec3) -> Vec3 {
    color / (Vec3::ONE + color)
}

fn create_output_directory(output_path: &Path) -> std::io::Result<()> {
    if let Some(parent) = output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }

    Ok(())
}

fn format_duration(duration: Duration) -> String {
    let total_millis = duration.as_millis();
    let minutes = total_millis / 60_000;
    let seconds = (total_millis % 60_000) / 1_000;
    let millis = total_millis % 1_000;

    format!("{minutes:02}m:{seconds:02}s:{millis:03}ms")
}
