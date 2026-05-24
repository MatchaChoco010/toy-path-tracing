use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
};

const MATERIALX_FILES: &[&str] = &[
    "stdlib/stdlib_defs.mtlx",
    "stdlib/stdlib_ng.mtlx",
    "pbrlib/pbrlib_defs.mtlx",
    "pbrlib/pbrlib_ng.mtlx",
    "bxdf/standard_surface.mtlx",
    "bxdf/disney_principled.mtlx",
    "bxdf/open_pbr_surface.mtlx",
    "bxdf/usd_preview_surface.mtlx",
    "bxdf/gltf_pbr.mtlx",
    "nprlib/nprlib_defs.mtlx",
    "nprlib/nprlib_ng.mtlx",
];

fn main() -> ExitCode {
    if let Err(error) = run() {
        eprintln!("update-materialx-libs failed: {error}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run() -> Result<(), Box<dyn Error>> {
    let root = workspace_root()?;
    let source_root = root.join("third_party/MaterialX");
    let destination_root = root.join("lib/materialx");

    if !source_root.join("libraries").is_dir() {
        return Err(format!(
            "MaterialX submodule libraries not found at `{}`",
            source_root.display()
        )
        .into());
    }

    fs::create_dir_all(destination_root.join("libraries"))?;
    fs::copy(
        source_root.join("LICENSE"),
        destination_root.join("LICENSE"),
    )?;

    for rel in MATERIALX_FILES {
        let source = source_root.join("libraries").join(rel);
        let destination = destination_root.join("libraries").join(rel);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&source, &destination)
            .map_err(|error| format!("copy `{}` failed: {error}", source.display()))?;
    }
    apply_local_patches(&destination_root)?;

    println!(
        "copied {} MaterialX library files from {}",
        MATERIALX_FILES.len(),
        source_root.display()
    );
    Ok(())
}

fn workspace_root() -> Result<PathBuf, Box<dyn Error>> {
    let exe_dir = std::env::current_dir()?;
    for candidate in exe_dir.ancestors() {
        if candidate.join(".git").exists() && candidate.join("Cargo.toml").exists() {
            return Ok(candidate.to_path_buf());
        }
    }
    Err(format!(
        "could not find workspace root from `{}`",
        Path::new(".").canonicalize()?.display()
    )
    .into())
}

fn apply_local_patches(destination_root: &Path) -> Result<(), Box<dyn Error>> {
    let stdlib_ng = destination_root.join("libraries/stdlib/stdlib_ng.mtlx");
    replace_once(
        &stdlib_ng,
        r#"<ifgreater name="ifgreatereq1" type="color4" nodedef="ND_ifgreatereq_color4I">
      <input name="value1" type="integer" interfacename="interval_num" />
      <input name="value2" type="integer" interfacename="num_intervals" />
      <input name="in1" type="color4" interfacename="prev_color" />
      <input name="in2" type="color4" nodename="ifgreater3" />
    </ifgreater>"#,
        r#"<ifgreatereq name="ifgreatereq1" type="color4" nodedef="ND_ifgreatereq_color4I">
      <input name="value1" type="integer" interfacename="interval_num" />
      <input name="value2" type="integer" interfacename="num_intervals" />
      <input name="in1" type="color4" interfacename="prev_color" />
      <input name="in2" type="color4" nodename="ifgreater3" />
    </ifgreatereq>"#,
    )?;

    let standard_surface = destination_root.join("libraries/bxdf/standard_surface.mtlx");
    replace_once(
        &standard_surface,
        r#"      <input name="opacity" type="float" nodename="opacity_luminance_float" />
    </surface>"#,
        r#"      <input name="opacity" type="float" nodename="opacity_luminance_float" />
      <input name="thin_walled" type="boolean" interfacename="thin_walled" />
    </surface>"#,
    )?;

    Ok(())
}

fn replace_once(path: &Path, from: &str, to: &str) -> Result<(), Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let count = text.matches(from).count();
    if count != 1 {
        return Err(format!(
            "expected one patch target in `{}`, found {count}",
            path.display()
        )
        .into());
    }
    fs::write(path, text.replace(from, to))?;
    Ok(())
}
