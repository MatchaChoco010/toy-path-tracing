use clap::Parser;
use glam::{Mat3, Mat4, UVec2, Vec3};
use image::RgbImage;
use rayon::prelude::*;
use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use toy_path_tracing::{
    camera::PinholeCamera,
    mesh::{Mesh, load_mesh},
    scene::{Bounds, Scene},
};

#[derive(Debug, Parser)]
struct Args {
    #[arg(short = 'o', long = "output", default_value = "result/output.png")]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let resolution = UVec2::new(512, 512);
    let bunny = place_mesh(
        load_mesh(Path::new("assets/bunny.glb"))?,
        1.55,
        Vec3::new(-0.95, 0.0, 0.05),
        28.0_f32.to_radians(),
    );
    let sphere = place_mesh(
        load_mesh(Path::new("assets/sphere.glb"))?,
        1.10,
        Vec3::new(1.05, 0.0, -0.10),
        0.0,
    );

    let mut scene = Scene::new();
    scene.add_mesh(bunny);
    scene.add_mesh(sphere);

    let scene_bounds = scene
        .bounds()
        .ok_or("scene must contain at least one vertex")?;
    let camera = create_camera(scene_bounds);

    let mut pixels = vec![0_u8; (resolution.x * resolution.y * 3) as usize];
    let intersect_start = Instant::now();
    pixels
        .par_chunks_mut(3)
        .enumerate()
        .for_each(|(index, pixel)| {
            let x = (index as u32) % resolution.x;
            let y = (index as u32) / resolution.x;
            let ray = camera.generate_ray(resolution, UVec2::new(x, y));

            let color = scene
                .closest_hit(&ray)
                .map(|hit| {
                    let [n0, n1, n2] = scene.triangle_normals(hit.triangle);
                    let normal =
                        (hit.barycentric.x * n0 + hit.barycentric.y * n1 + hit.barycentric.z * n2)
                            .normalize_or_zero();
                    0.5 * (normal + Vec3::ONE)
                })
                .unwrap_or(Vec3::ZERO);

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

fn create_camera(bounds: Bounds) -> PinholeCamera {
    let center = bounds.center();
    let extent = bounds.extent();
    let radius = 0.5 * extent.length().max(1.0e-3);
    let fov_y = 33.0_f32.to_radians();
    let distance = radius / (0.5 * fov_y).tan();

    let eye = center + Vec3::new(0.0, 0.22 * extent.y.max(radius), 1.55 * distance);
    let look_at = center + Vec3::new(0.0, 0.12 * extent.y, 0.0);

    PinholeCamera::new(eye, look_at, Vec3::Y, fov_y)
}

fn place_mesh(mesh: Mesh, target_height: f32, translation: Vec3, yaw_radians: f32) -> Mesh {
    let bounds = mesh_bounds(&mesh);
    let height = bounds.extent().y.max(1.0e-3);
    let scale = target_height / height;
    let pivot = Vec3::new(bounds.center().x, bounds.min.y, bounds.center().z);
    let transform = Mat4::from_translation(translation)
        * Mat4::from_rotation_y(yaw_radians)
        * Mat4::from_scale(Vec3::splat(scale))
        * Mat4::from_translation(-pivot);

    transform_mesh(mesh, transform)
}

fn transform_mesh(mut mesh: Mesh, transform: Mat4) -> Mesh {
    let normal_transform = Mat3::from_mat4(transform).inverse().transpose();

    for vertex in &mut mesh.vertices {
        vertex.position = transform.transform_point3(vertex.position);
        vertex.normal = normal_transform.mul_vec3(vertex.normal).normalize_or_zero();
    }

    mesh
}

fn mesh_bounds(mesh: &Mesh) -> Bounds {
    let mut vertices = mesh.vertices.iter();
    let first = vertices
        .next()
        .expect("mesh must contain at least one vertex");
    let mut min = first.position;
    let mut max = first.position;

    for vertex in vertices {
        min = min.min(vertex.position);
        max = max.max(vertex.position);
    }

    Bounds { min, max }
}

fn format_duration(duration: Duration) -> String {
    let total_millis = duration.as_millis();
    let minutes = total_millis / 60_000;
    let seconds = (total_millis % 60_000) / 1_000;
    let millis = total_millis % 1_000;

    format!("{minutes:02}m:{seconds:02}s:{millis:03}ms")
}
