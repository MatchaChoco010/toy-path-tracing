use clap::Parser;
use glam::{UVec2, Vec3};
use image::RgbImage;
use rayon::prelude::*;
use std::{fs, path::Path};
use toy_path_tracing::{camera::PinholeCamera, ray::intersect_triangle_unbounded};

#[derive(Debug, Parser)]
struct Args {
    #[arg(short = 'o', long = "output", default_value = "result/output.png")]
    output: std::path::PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let resolution = UVec2::new(512, 512);
    let triangle = [
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(-1.0, -1.0, 0.0),
        Vec3::new(1.0, -1.0, 0.0),
    ];
    let vertex_colors = [Vec3::X, Vec3::Y, Vec3::Z];
    let camera = PinholeCamera::new(
        Vec3::new(0.0, 0.0, 5.0),
        Vec3::ZERO,
        Vec3::Y,
        45.0_f32.to_radians(),
    );

    let mut pixels = vec![0_u8; (resolution.x * resolution.y * 3) as usize];
    pixels
        .par_chunks_mut(3)
        .enumerate()
        .for_each(|(index, pixel)| {
            let x = (index as u32) % resolution.x;
            let y = (index as u32) / resolution.x;
            let ray = camera.generate_ray(resolution, UVec2::new(x, y));

            let color = intersect_triangle_unbounded(&ray, triangle[0], triangle[1], triangle[2])
                .map(|hit| hit.interpolate(vertex_colors[0], vertex_colors[1], vertex_colors[2]))
                .unwrap_or(Vec3::ZERO);

            pixel[0] = float_to_u8(color.x);
            pixel[1] = float_to_u8(color.y);
            pixel[2] = float_to_u8(color.z);
        });

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
