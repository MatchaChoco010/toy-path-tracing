use clap::Parser;
use glam::{UVec2, Vec2, Vec3};
use image::RgbImage;
use rand::RngExt;
use rayon::prelude::*;
use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use toy_path_tracing::scenes::load_scene;

#[derive(Debug, Parser)]
struct Args {
    #[arg(short = 'o', long = "output", default_value = "result/output.png")]
    output: PathBuf,

    #[arg(long = "scene", default_value_t = 0)]
    scene: u32,

    #[arg(long = "spp", default_value_t = 32, value_parser = clap::value_parser!(u32).range(1..))]
    spp: u32,
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
                let sample = scene
                    .closest_hit(&ray)
                    .expect("scene.build_bvh() must be called before traversal")
                    .map(|hit| {
                        let [n0, n1, n2] = scene.triangle_normals(hit.triangle);
                        let normal = (hit.barycentric.x * n0
                            + hit.barycentric.y * n1
                            + hit.barycentric.z * n2)
                            .normalize_or_zero();
                        0.5 * (normal + Vec3::ONE)
                    })
                    .unwrap_or(Vec3::ZERO);
                let sample_count = (sample_index + 1) as f32;
                color += (sample - color) / sample_count;
            }

            pixel[0] = float_to_u8(color.x);
            pixel[1] = float_to_u8(color.y);
            pixel[2] = float_to_u8(color.z);
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
