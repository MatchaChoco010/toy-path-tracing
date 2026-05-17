use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::flatten::{FlatGraph, FlatInput, FlatNodeKind};
use super::types::{MtlxType, MtlxValue};
use super::{MtlxLibrary, flatten_material, parse_document};
use crate::color::management;
use crate::material::mtlx::compiled::{UdimTile, UdimTiles};
use crate::material::{MtlxMaterial, ScalarTexture, Texture, TextureColorSpace};

#[derive(Debug)]
pub enum LoadError {
    Parse(String),
    Flatten(String),
    Material(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(s) => write!(f, "mtlx parse error: {}", s),
            Self::Flatten(s) => write!(f, "mtlx flatten error: {}", s),
            Self::Material(s) => write!(f, "mtlx material error: {}", s),
        }
    }
}

impl std::error::Error for LoadError {}

pub fn load_mtlx_material(
    library: &MtlxLibrary,
    mtlx_path: &Path,
    material_name: &str,
) -> Result<MtlxMaterial, LoadError> {
    let document = parse_document(mtlx_path).map_err(|e| LoadError::Parse(format!("{}", e)))?;
    // The document may declare custom nodedefs (e.g. shader_ops.mtlx
    // defines ND_checker_float). Make those visible to the flattener
    // by merging them into a per-load copy of the library.
    let mut merged = MtlxLibrary::new();
    for nd in &library.nodedefs {
        merged.add_document(super::types::RawMtlxDocument {
            nodedefs: vec![nd.def.clone()],
            ..Default::default()
        });
    }
    for ng in &library.nodegraphs {
        merged.add_document(super::types::RawMtlxDocument {
            nodegraphs: vec![ng.graph.clone()],
            ..Default::default()
        });
    }
    for im in &library.implementations {
        merged.add_document(super::types::RawMtlxDocument {
            implementations: vec![im.clone()],
            ..Default::default()
        });
    }
    merged.add_document(document.clone());
    let graph = flatten_material(&merged, &document, material_name)
        .map_err(|e| LoadError::Flatten(format!("{}", e)))?;
    let (textures, alpha_textures, mut udim_textures) = collect_color_textures(&graph, mtlx_path);
    let (scalar_textures, scalar_udim_textures) = collect_scalar_textures(&graph, mtlx_path);
    merge_udim_tile_sets(&mut udim_textures, scalar_udim_textures);
    let arc_color: HashMap<Arc<str>, Arc<Texture>> = textures
        .into_iter()
        .map(|(k, v)| (Arc::from(k.as_str()), v))
        .collect();
    let arc_alpha: HashMap<Arc<str>, Arc<ScalarTexture>> = alpha_textures
        .into_iter()
        .map(|(k, v)| (Arc::from(k.as_str()), v))
        .collect();
    let arc_udim: HashMap<Arc<str>, Arc<UdimTiles>> = udim_textures
        .into_iter()
        .map(|(k, v)| (Arc::from(k.as_str()), v))
        .collect();
    let arc_scalar: HashMap<Arc<str>, Arc<ScalarTexture>> = scalar_textures
        .into_iter()
        .map(|(k, v)| (Arc::from(k.as_str()), v))
        .collect();
    let compiled = crate::material::mtlx::compile(
        &graph,
        arc_color.clone(),
        arc_alpha.clone(),
        arc_udim.clone(),
        arc_scalar.clone(),
    )
    .map_err(|e| LoadError::Material(format!("{}", e)))?;

    let back_compiled = if let Some(back_root) = graph.back_root {
        let mut back_graph = graph.clone();
        back_graph.root = back_root;
        back_graph.back_root = None;
        Some(
            crate::material::mtlx::compile(&back_graph, arc_color, arc_alpha, arc_udim, arc_scalar)
                .map_err(|e| LoadError::Material(format!("{}", e)))?,
        )
    } else {
        None
    };

    Ok(MtlxMaterial::with_back(
        Arc::new(compiled),
        back_compiled.map(Arc::new),
    ))
}

fn extract_image_filename(node: &super::flatten::FlatNode) -> Option<(&String, Option<&str>)> {
    let input = node.inputs.iter().find(|i| i.name == "file")?;
    let filename = match &input.binding {
        FlatInput::Value(MtlxValue::Filename(s)) | FlatInput::Value(MtlxValue::String(s)) => s,
        FlatInput::String(s) => s,
        _ => return None,
    };
    if filename.is_empty() {
        return None;
    }
    Some((filename, input.colorspace.as_deref()))
}

fn resolve_path(parent: &Path, filename: &str) -> PathBuf {
    let direct = if Path::new(filename).is_absolute() {
        PathBuf::from(filename)
    } else {
        parent.join(filename)
    };
    if direct.exists() {
        return direct;
    }
    let dir = direct.parent().unwrap_or(Path::new("."));
    let target_name = direct.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str()
                && name.eq_ignore_ascii_case(target_name)
            {
                return entry.path();
            }
        }
    }
    direct
}

fn pick_color_space(output_type: &MtlxType, colorspace: Option<&str>) -> TextureColorSpace {
    match output_type {
        MtlxType::Color3 | MtlxType::Color4 => match colorspace {
            None => TextureColorSpace::DefaultColor,
            Some("none") | Some("linear") | Some("scene_linear") => TextureColorSpace::Linear,
            Some(other) => {
                TextureColorSpace::ocio(management::map_materialx_color_space(other).to_string())
            }
        },
        _ => TextureColorSpace::Linear,
    }
}

type ColorTextureCollection = (
    HashMap<String, Arc<Texture>>,
    HashMap<String, Arc<ScalarTexture>>,
    HashMap<String, Arc<UdimTiles>>,
);

fn collect_color_textures(graph: &FlatGraph, mtlx_path: &Path) -> ColorTextureCollection {
    let parent = mtlx_path.parent().unwrap_or_else(|| Path::new("."));
    let mut out: HashMap<String, Arc<Texture>> = HashMap::new();
    let mut alpha_out: HashMap<String, Arc<ScalarTexture>> = HashMap::new();
    let mut udim_out: HashMap<String, Arc<UdimTiles>> = HashMap::new();
    for node in &graph.nodes {
        let category = match &node.kind {
            FlatNodeKind::Pattern { category } => category.as_str(),
            _ => continue,
        };
        if !matches!(
            category,
            "image" | "tiledimage" | "latlongimage" | "hextiledimage" | "hextilednormalmap"
        ) {
            continue;
        }
        if !is_color_image(node) {
            continue;
        }
        let Some((filename, cs)) = extract_image_filename(node) else {
            continue;
        };
        let space = pick_color_space(&node.output_type, cs);
        let wants_alpha = is_color4_image(node);

        if filename.contains("<UDIM>") || filename.contains("<UVTILE>") {
            if udim_out.contains_key(filename) {
                if wants_alpha {
                    match load_udim_tiles(parent, filename, space.clone(), true) {
                        Ok(tiles) if !tiles.tiles.is_empty() => {
                            merge_udim_tile_sets(
                                &mut udim_out,
                                HashMap::from([(filename.clone(), Arc::new(tiles))]),
                            );
                        }
                        Ok(_) => {}
                        Err(e) => {
                            tracing::warn!(
                                "[mtlx] warning: failed scanning alpha UDIM tiles for `{}`: {}",
                                filename,
                                e
                            );
                        }
                    }
                }
                continue;
            }
            match load_udim_tiles(parent, filename, space.clone(), wants_alpha) {
                Ok(tiles) if !tiles.tiles.is_empty() => {
                    udim_out.insert(filename.clone(), Arc::new(tiles));
                }
                Ok(_) => {
                    tracing::warn!(
                        "[mtlx] warning: filename `{}` declares <UDIM>/<UVTILE> but no matching tiles were found",
                        filename
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "[mtlx] warning: failed scanning UDIM tiles for `{}`: {}",
                        filename,
                        e
                    );
                }
            }
            continue;
        }

        if out.contains_key(filename) {
            if wants_alpha && !alpha_out.contains_key(filename) {
                let resolved = resolve_path(parent, filename);
                match Texture::from_file_with_alpha(&resolved, space.clone()) {
                    Ok((tex, alpha)) => {
                        out.insert(filename.clone(), Arc::new(tex));
                        if let Some(a) = alpha {
                            alpha_out.insert(filename.clone(), Arc::new(a));
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "[mtlx] warning: could not load color4 image `{}` (resolved to {}): {}",
                            filename,
                            resolved.display(),
                            e
                        );
                    }
                }
            }
            continue;
        }

        let resolved = resolve_path(parent, filename);
        if wants_alpha {
            match Texture::from_file_with_alpha(&resolved, space.clone()) {
                Ok((tex, alpha)) => {
                    out.insert(filename.clone(), Arc::new(tex));
                    if let Some(a) = alpha {
                        alpha_out.insert(filename.clone(), Arc::new(a));
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "[mtlx] warning: could not load color image `{}` (resolved to {}): {}",
                        filename,
                        resolved.display(),
                        e
                    );
                }
            }
        } else {
            match Texture::from_file_with_color_space(&resolved, space.clone()) {
                Ok(tex) => {
                    out.insert(filename.clone(), Arc::new(tex));
                }
                Err(e) => {
                    tracing::warn!(
                        "[mtlx] warning: could not load color image `{}` (resolved to {}): {}",
                        filename,
                        resolved.display(),
                        e
                    );
                }
            }
        }
    }
    (out, alpha_out, udim_out)
}

/// Scan the filesystem for files matching `pattern` after substituting
/// `<UDIM>` with each 4-digit id and/or `<UVTILE>` with `u<U>_v<V>` per
/// spec §Filename Substitutions, and build a [`UdimTiles`] from what is
/// actually present on disk. Each found tile is loaded with its alpha
/// pyramid when `wants_alpha` is true (image_color4 case).
fn load_udim_tiles(
    parent: &Path,
    pattern: &str,
    space: TextureColorSpace,
    wants_alpha: bool,
) -> std::io::Result<UdimTiles> {
    let mut tiles: HashMap<u32, UdimTile> = HashMap::new();
    // Iterate the entries of the directory the file lives in once and try
    // to recover the UDIM index from each name that matches the pattern,
    // so we do not stat thousands of non-existent paths.
    let probe_path = resolve_path(
        parent,
        &pattern
            .replace("<UDIM>", "1001")
            .replace("<UVTILE>", "u1_v1"),
    );
    let dir = probe_path.parent().unwrap_or(parent).to_path_buf();
    let pattern_filename = Path::new(pattern)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(pattern);
    let entries = std::fs::read_dir(&dir)?;
    for entry in entries.flatten() {
        let Some(name) = entry.file_name().to_str().map(|s| s.to_string()) else {
            continue;
        };
        let Some(udim_id) = match_udim_pattern(pattern_filename, &name) else {
            continue;
        };
        let path = entry.path();
        if wants_alpha {
            match Texture::from_file_with_alpha(&path, space.clone()) {
                Ok((tex, alpha)) => {
                    tiles.insert(
                        udim_id,
                        UdimTile {
                            rgb: Arc::new(tex),
                            alpha: alpha.map(Arc::new),
                            scalar: None,
                        },
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "[mtlx] warning: failed loading UDIM tile {} ({}): {}",
                        udim_id,
                        path.display(),
                        e
                    );
                }
            }
        } else {
            match Texture::from_file_with_color_space(&path, space.clone()) {
                Ok(tex) => {
                    tiles.insert(
                        udim_id,
                        UdimTile {
                            rgb: Arc::new(tex),
                            alpha: None,
                            scalar: None,
                        },
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "[mtlx] warning: failed loading UDIM tile {} ({}): {}",
                        udim_id,
                        path.display(),
                        e
                    );
                }
            }
        }
    }
    Ok(UdimTiles { tiles })
}

/// Returns the UDIM id encoded in `name` if it matches `pattern` (which is
/// expected to contain `<UDIM>` and/or `<UVTILE>`). Both spellings of the
/// id are accepted: Mari-style 4-digit `1001..` for `<UDIM>` and
/// Mudbox-style `u<U+1>_v<V+1>` for `<UVTILE>`.
fn match_udim_pattern(pattern: &str, name: &str) -> Option<u32> {
    if let Some(idx) = pattern.find("<UDIM>") {
        let before = &pattern[..idx];
        let after = &pattern[idx + "<UDIM>".len()..];
        if !name.starts_with(before) || !name.ends_with(after) {
            return None;
        }
        let mid = &name[before.len()..name.len() - after.len()];
        if mid.len() != 4 {
            return None;
        }
        return mid
            .parse::<u32>()
            .ok()
            .filter(|&n| (1001..10000).contains(&n));
    }
    if let Some(idx) = pattern.find("<UVTILE>") {
        let before = &pattern[..idx];
        let after = &pattern[idx + "<UVTILE>".len()..];
        if !name.starts_with(before) || !name.ends_with(after) {
            return None;
        }
        let mid = &name[before.len()..name.len() - after.len()];
        // Expected form: u<U+1>_v<V+1>; both U+1 and V+1 are >= 1.
        let u_part = mid.strip_prefix('u')?;
        let (u_str, rest) = u_part.split_once('_')?;
        let v_str = rest.strip_prefix('v')?;
        let u_plus1: u32 = u_str.parse().ok()?;
        let v_plus1: u32 = v_str.parse().ok()?;
        if u_plus1 == 0 || v_plus1 == 0 {
            return None;
        }
        let u = u_plus1 - 1;
        let v = v_plus1 - 1;
        if u >= 10 {
            return None;
        }
        return Some(1001 + u + v * 10);
    }
    None
}

fn is_color4_image(node: &super::flatten::FlatNode) -> bool {
    matches!(
        node.output_type,
        super::types::MtlxType::Color4 | super::types::MtlxType::Vector4
    )
}

type ScalarTextureCollection = (
    HashMap<String, Arc<ScalarTexture>>,
    HashMap<String, Arc<UdimTiles>>,
);

fn collect_scalar_textures(graph: &FlatGraph, mtlx_path: &Path) -> ScalarTextureCollection {
    let parent = mtlx_path.parent().unwrap_or_else(|| Path::new("."));
    let mut out: HashMap<String, Arc<ScalarTexture>> = HashMap::new();
    let mut udim_out: HashMap<String, Arc<UdimTiles>> = HashMap::new();
    for node in &graph.nodes {
        let category = match &node.kind {
            FlatNodeKind::Pattern { category } => category.as_str(),
            _ => continue,
        };
        if !matches!(category, "image" | "tiledimage" | "hextiledimage") {
            continue;
        }
        if !is_scalar_image(node) {
            continue;
        }
        let Some((filename, _cs)) = extract_image_filename(node) else {
            continue;
        };
        if filename.contains("<UDIM>") || filename.contains("<UVTILE>") {
            if udim_out.contains_key(filename) {
                continue;
            }
            match load_scalar_udim_tiles(parent, filename) {
                Ok(tiles) if !tiles.tiles.is_empty() => {
                    udim_out.insert(filename.clone(), Arc::new(tiles));
                }
                Ok(_) => {
                    tracing::warn!(
                        "[mtlx] warning: filename `{}` declares <UDIM>/<UVTILE> but no matching scalar tiles were found",
                        filename
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "[mtlx] warning: failed scanning scalar UDIM tiles for `{}`: {}",
                        filename,
                        e
                    );
                }
            }
            continue;
        }
        if out.contains_key(filename) {
            continue;
        }
        let resolved = resolve_path(parent, filename);
        match ScalarTexture::from_file(&resolved) {
            Ok(tex) => {
                out.insert(filename.clone(), Arc::new(tex));
            }
            Err(e) => {
                tracing::warn!(
                    "[mtlx] warning: could not load scalar image `{}` (resolved to {}): {}",
                    filename,
                    resolved.display(),
                    e
                );
            }
        }
    }
    (out, udim_out)
}

fn load_scalar_udim_tiles(parent: &Path, pattern: &str) -> std::io::Result<UdimTiles> {
    let mut tiles: HashMap<u32, UdimTile> = HashMap::new();
    let probe_path = resolve_path(
        parent,
        &pattern
            .replace("<UDIM>", "1001")
            .replace("<UVTILE>", "u1_v1"),
    );
    let dir = probe_path.parent().unwrap_or(parent).to_path_buf();
    let pattern_filename = Path::new(pattern)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(pattern);
    let entries = std::fs::read_dir(&dir)?;
    for entry in entries.flatten() {
        let Some(name) = entry.file_name().to_str().map(|s| s.to_string()) else {
            continue;
        };
        let Some(udim_id) = match_udim_pattern(pattern_filename, &name) else {
            continue;
        };
        let path = entry.path();
        match (
            Texture::from_file_with_color_space(&path, TextureColorSpace::Linear),
            ScalarTexture::from_file(&path),
        ) {
            (Ok(rgb), Ok(scalar)) => {
                tiles.insert(
                    udim_id,
                    UdimTile {
                        rgb: Arc::new(rgb),
                        alpha: None,
                        scalar: Some(Arc::new(scalar)),
                    },
                );
            }
            (Err(e), _) | (_, Err(e)) => {
                tracing::warn!(
                    "[mtlx] warning: failed loading scalar UDIM tile {} ({}): {}",
                    udim_id,
                    path.display(),
                    e
                );
            }
        }
    }
    Ok(UdimTiles { tiles })
}

fn merge_udim_tile_sets(
    dst: &mut HashMap<String, Arc<UdimTiles>>,
    src: HashMap<String, Arc<UdimTiles>>,
) {
    for (name, incoming) in src {
        if let Some(existing) = dst.get_mut(&name) {
            let merged = Arc::make_mut(existing);
            for (id, tile) in &incoming.tiles {
                if let Some(existing_tile) = merged.tiles.get_mut(id) {
                    if tile.alpha.is_some() {
                        existing_tile.alpha = tile.alpha.clone();
                    }
                    if tile.scalar.is_some() {
                        existing_tile.scalar = tile.scalar.clone();
                    }
                } else {
                    merged.tiles.insert(*id, tile.clone());
                }
            }
        } else {
            dst.insert(name, incoming);
        }
    }
}

fn is_color_image(node: &super::flatten::FlatNode) -> bool {
    matches!(
        node.output_type,
        super::types::MtlxType::Color3
            | super::types::MtlxType::Color4
            | super::types::MtlxType::Vector3
            | super::types::MtlxType::Vector4
    )
}

fn is_scalar_image(node: &super::flatten::FlatNode) -> bool {
    matches!(
        node.output_type,
        super::types::MtlxType::Float | super::types::MtlxType::Integer
    )
}

#[cfg(test)]
mod udim_tests {
    use super::{collect_color_textures, match_udim_pattern, pick_color_space};
    use crate::scene_loader::mtlx_loader::flatten::{
        FlatGraph, FlatInput, FlatNode, FlatNodeInput, FlatNodeKind,
    };
    use crate::scene_loader::mtlx_loader::types::{MtlxType, MtlxValue};

    #[test]
    fn matches_udim_4digit_mari_style() {
        assert_eq!(
            match_udim_pattern("brick.<UDIM>.tif", "brick.1001.tif"),
            Some(1001)
        );
        assert_eq!(
            match_udim_pattern("brick.<UDIM>.tif", "brick.1011.tif"),
            Some(1011)
        );
        // Outside the Mari-defined range or wrong digit count must not match.
        assert_eq!(
            match_udim_pattern("brick.<UDIM>.tif", "brick.0999.tif"),
            None
        );
        assert_eq!(
            match_udim_pattern("brick.<UDIM>.tif", "brick.101.tif"),
            None
        );
        assert_eq!(
            match_udim_pattern("brick.<UDIM>.tif", "brick.10011.tif"),
            None
        );
        // Pattern prefix/suffix mismatch.
        assert_eq!(
            match_udim_pattern("brick.<UDIM>.tif", "other.1001.tif"),
            None
        );
    }

    #[test]
    fn matches_uvtile_mudbox_style() {
        // u1_v1 → (U=0, V=0) → UDIM 1001
        assert_eq!(
            match_udim_pattern("brick.<UVTILE>.tif", "brick.u1_v1.tif"),
            Some(1001)
        );
        // u2_v1 → (U=1, V=0) → UDIM 1002
        assert_eq!(
            match_udim_pattern("brick.<UVTILE>.tif", "brick.u2_v1.tif"),
            Some(1002)
        );
        // u1_v2 → (U=0, V=1) → UDIM 1011
        assert_eq!(
            match_udim_pattern("brick.<UVTILE>.tif", "brick.u1_v2.tif"),
            Some(1011)
        );
        assert_eq!(
            match_udim_pattern("brick.<UVTILE>.tif", "brick.u0_v1.tif"),
            None
        );
        assert_eq!(match_udim_pattern("brick.<UVTILE>.tif", "u1_v1.tif"), None);
    }

    #[test]
    fn color4_use_adds_alpha_when_color3_loaded_same_file_first() {
        let dir = std::env::temp_dir().join(format!(
            "toy_path_tracing_mtlx_alpha_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp image dir");
        let image_path = dir.join("shared.png");
        let img = image::RgbaImage::from_pixel(1, 1, image::Rgba([64, 128, 192, 128]));
        img.save(&image_path).expect("save temp image");
        let graph = FlatGraph {
            nodes: vec![image_node(MtlxType::Color3), image_node(MtlxType::Color4)],
            root: 0,
            back_root: None,
            material_name: "test".to_string(),
        };
        let mtlx_path = dir.join("mat.mtlx");
        let (_rgb, alpha, _udim) = collect_color_textures(&graph, &mtlx_path);
        assert!(alpha.contains_key("shared.png"));
    }

    #[test]
    fn image_colorspace_is_kept_as_ocio_space() {
        assert_eq!(
            pick_color_space(&MtlxType::Color3, Some("lin_rec709")),
            crate::material::TextureColorSpace::ocio("lin_rec709".to_string())
        );
        assert_eq!(
            pick_color_space(&MtlxType::Color3, None),
            crate::material::TextureColorSpace::DefaultColor
        );
    }

    #[test]
    fn vector_images_are_loaded_as_data_without_color_conversion() {
        assert_eq!(
            pick_color_space(&MtlxType::Vector3, None),
            crate::material::TextureColorSpace::Linear
        );
        assert_eq!(
            pick_color_space(&MtlxType::Vector3, Some("srgb_texture")),
            crate::material::TextureColorSpace::Linear
        );
        assert_eq!(
            pick_color_space(&MtlxType::Vector4, Some("lin_rec709")),
            crate::material::TextureColorSpace::Linear
        );
    }

    fn image_node(output_type: MtlxType) -> FlatNode {
        FlatNode {
            kind: FlatNodeKind::Pattern {
                category: "image".to_string(),
            },
            output_type,
            inputs: vec![FlatNodeInput {
                name: "file".to_string(),
                ty: MtlxType::Filename,
                colorspace: None,
                unit: None,
                unittype: None,
                binding: FlatInput::Value(MtlxValue::Filename("shared.png".to_string())),
            }],
        }
    }
}
