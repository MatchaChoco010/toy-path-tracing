use clap::Parser;
use glam::{UVec2, Vec2, Vec3};
use rand::RngExt;
use rayon::prelude::*;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use toy_path_tracing::{
    color::management::{
        DEFAULT_OCIO_CONFIG, DEFAULT_OUTPUT_DISPLAY, DEFAULT_OUTPUT_VIEW,
        DEFAULT_TEXTURE_COLOR_SPACE, OcioRenderContext, set_current,
    },
    integrator::IntegratorKind,
    output_image::{OutputTransform, save_output},
    scenes::load_scene,
};
use tracing_subscriber::{EnvFilter, fmt};

#[derive(Debug, Parser)]
struct Args {
    #[arg(short = 'o', long = "output", default_value = "result/output.png")]
    output: PathBuf,

    #[arg(long = "width", default_value_t = 512, value_parser = clap::value_parser!(u32).range(1..))]
    width: u32,

    #[arg(long = "height", default_value_t = 512, value_parser = clap::value_parser!(u32).range(1..))]
    height: u32,

    #[arg(long = "scene", default_value_t = 0)]
    scene: u32,

    #[arg(long = "spp", default_value_t = 32, value_parser = clap::value_parser!(u32).range(1..))]
    spp: u32,

    #[arg(long = "depth", default_value_t = 16, value_parser = clap::value_parser!(u32).range(1..))]
    depth: u32,

    #[arg(short = 'i', long = "integrator", value_enum, default_value_t = IntegratorKind::Mis)]
    integrator: IntegratorKind,

    #[arg(long = "ocio-config", default_value = DEFAULT_OCIO_CONFIG)]
    ocio_config: String,

    #[arg(long = "ocio-rendering-space")]
    ocio_rendering_space: Option<String>,

    #[arg(long = "texture-color-space", default_value = DEFAULT_TEXTURE_COLOR_SPACE)]
    texture_color_space: String,

    #[arg(long = "output-display")]
    output_display: Option<String>,

    #[arg(long = "output-view")]
    output_view: Option<String>,

    #[arg(long = "log-filter")]
    log_filter: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    init_tracing(args.log_filter.as_deref())?;
    let ocio = Arc::new(OcioRenderContext::new(
        &args.ocio_config,
        args.ocio_rendering_space.clone(),
        &args.texture_color_space,
    )?);
    tracing::info!(
        ocio_config = ocio.config_source(),
        rendering_space = ocio.rendering_space(),
        texture_color_space = ocio.texture_color_space(),
        "initialized OCIO"
    );
    set_current(Arc::clone(&ocio))?;
    let output_transform = output_transform(&args, &ocio)?;
    let resolution = UVec2::new(args.width, args.height);
    let (mut scene, camera) = load_scene(args.scene)?;
    let build_bvh_start = Instant::now();
    scene.build_qbvh();
    tracing::info!("build_bvh: {}", format_duration(build_bvh_start.elapsed()));

    let build_light_tree_start = Instant::now();
    scene.build_light_tree();
    tracing::info!(
        "build_light_tree: {}",
        format_duration(build_light_tree_start.elapsed())
    );

    let exposure = camera.exposure;
    let mut pixels = vec![0.0_f32; (resolution.x * resolution.y * 3) as usize];
    let intersect_start = Instant::now();
    pixels.par_chunks_mut(3).enumerate().for_each_init(
        || (rand::rng(), scene.make_mtlx_scratch()),
        |(rng, mtlx_scratch), (index, pixel)| {
            let x = (index as u32) % resolution.x;
            let y = (index as u32) / resolution.x;
            let mut color = Vec3::ZERO;

            for sample_index in 0..args.spp {
                let us = Vec2::new(rng.random::<f32>(), rng.random::<f32>());
                let ray =
                    camera.generate_ray_differential(resolution, UVec2::new(x, y), us, args.spp);
                let sample =
                    args.integrator
                        .trace_radiance(&scene, ray, rng, args.depth, mtlx_scratch);
                let sample_count = (sample_index + 1) as f32;
                color += (sample - color) / sample_count;
            }

            let exposed = color * exposure;
            pixel[0] = exposed.x;
            pixel[1] = exposed.y;
            pixel[2] = exposed.z;
        },
    );
    tracing::info!("render: {}", format_duration(intersect_start.elapsed()));

    create_output_directory(&args.output)?;
    save_output(&args.output, resolution, &pixels, &ocio, &output_transform)?;

    Ok(())
}

fn output_transform(
    args: &Args,
    ocio: &OcioRenderContext,
) -> Result<OutputTransform, Box<dyn std::error::Error>> {
    if output_extension(&args.output).as_deref() == Some("exr")
        && args.output_display.is_none()
        && args.output_view.is_none()
    {
        return Ok(OutputTransform::Rendering);
    }

    let output_display = args
        .output_display
        .clone()
        .unwrap_or_else(|| DEFAULT_OUTPUT_DISPLAY.to_string());
    let view = args
        .output_view
        .clone()
        .unwrap_or_else(|| DEFAULT_OUTPUT_VIEW.to_string());
    ocio.validate_display_view(&output_display, &view)?;
    tracing::info!(%output_display, output_view = %view, "selected OCIO output view");
    Ok(OutputTransform::DisplayView {
        display: output_display,
        view,
    })
}

fn init_tracing(log_filter: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let filter = if let Some(log_filter) = log_filter {
        EnvFilter::try_new(log_filter)?
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
    };

    fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stdout)
        .init();

    Ok(())
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

fn output_extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase)
}

fn format_duration(duration: Duration) -> String {
    let total_millis = duration.as_millis();
    let minutes = total_millis / 60_000;
    let seconds = (total_millis % 60_000) / 1_000;
    let millis = total_millis % 1_000;

    format!("{minutes:02}m:{seconds:02}s:{millis:03}ms")
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::Args;
    use toy_path_tracing::integrator::IntegratorKind;

    #[test]
    fn integrator_defaults_to_mis() {
        let args = Args::try_parse_from(["toy-path-tracing"]).expect("expected valid defaults");

        assert_eq!(args.integrator, IntegratorKind::Mis);
    }

    #[test]
    fn integrator_accepts_mis_from_cli() {
        let args = Args::try_parse_from(["toy-path-tracing", "-i", "mis"])
            .expect("expected valid mis integrator");

        assert_eq!(args.integrator, IntegratorKind::Mis);
    }

    #[test]
    fn log_filter_is_optional() {
        let args = Args::try_parse_from(["toy-path-tracing"]).expect("expected valid defaults");

        assert_eq!(args.log_filter, None);
    }

    #[test]
    fn output_display_view_are_optional() {
        let args = Args::try_parse_from(["toy-path-tracing"]).expect("expected valid defaults");

        assert_eq!(args.output_display, None);
        assert_eq!(args.output_view, None);
    }
}
