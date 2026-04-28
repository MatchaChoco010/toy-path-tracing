use std::{
    cell::RefCell,
    collections::HashMap,
    error::Error,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    rc::Rc,
};

use glam::{Mat4, Quat, Vec3, Vec4};
use russimp::{
    Matrix4x4,
    material::{Material, PropertyTypeInfo, Texture as RussimpTexture, TextureType},
    mesh::Mesh,
    node::Node,
    scene::{PostProcess, Scene},
};
use serde_json::{Value, json};

const POSITION_COMPONENT_TYPE: u32 = 5126;
const INDEX_COMPONENT_TYPE_U32: u32 = 5125;
const ARRAY_BUFFER_TARGET: u32 = 34962;
const ELEMENT_ARRAY_BUFFER_TARGET: u32 = 34963;

fn main() -> ExitCode {
    let mut args = std::env::args();
    let prog = args.next().unwrap_or_else(|| "convert-bistro".to_string());
    let collected: Vec<String> = args.collect();

    if collected.len() != 3 {
        eprintln!(
            "usage: {prog} <input.fbx> <output_dir> <name>\n\n\
             Converts a Bistro FBX file (NVIDIA ORCA distribution) to a glTF 2.0\n\
             text document next to a single binary buffer file. DDS textures\n\
             referenced by the FBX are transcoded to PNG and emitted into\n\
             <output_dir>/textures/."
        );
        return ExitCode::from(64);
    }

    let input = PathBuf::from(&collected[0]);
    let output_dir = PathBuf::from(&collected[1]);
    let name = collected[2].clone();

    if let Err(error) = run(&input, &output_dir, &name) {
        eprintln!("convert-bistro failed: {error}");
        let mut current = error.source();
        while let Some(cause) = current {
            eprintln!("  caused by: {cause}");
            current = cause.source();
        }
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

fn run(input: &Path, output_dir: &Path, name: &str) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(output_dir)?;
    let texture_dir = output_dir.join("textures");
    fs::create_dir_all(&texture_dir)?;

    let bistro_root = input
        .parent()
        .ok_or("input FBX path has no parent directory")?
        .to_path_buf();
    let scene = Scene::from_file(
        input
            .to_str()
            .ok_or("input FBX path is not valid UTF-8")?,
        vec![
            PostProcess::Triangulate,
            PostProcess::JoinIdenticalVertices,
            PostProcess::GenerateSmoothNormals,
            PostProcess::ImproveCacheLocality,
            PostProcess::SortByPrimitiveType,
        ],
    )
    .map_err(|error| format!("assimp failed to load FBX: {error}"))?;

    let mut document = GltfDocument::new();
    let mut texture_resolver = TextureResolver::new(&bistro_root, &texture_dir);

    document.add_meshes(&scene.meshes)?;
    document.add_materials(&scene.materials, &mut texture_resolver)?;

    if let Some(root) = &scene.root {
        let root_index = document.add_node_recursive(root);
        document.set_scene_root(root_index);
    }

    let buffer_path = output_dir.join(format!("{name}.bin"));
    let mut buffer_writer = BufWriter::new(File::create(&buffer_path)?);
    buffer_writer.write_all(&document.binary_buffer)?;
    buffer_writer.flush()?;

    let buffer_uri = format!("{name}.bin");
    let mesh_count = document.meshes_json.len();
    let material_count = document.materials_json.len();
    let image_count = document.images_json.len();
    let json = document.into_json(buffer_uri, name);

    let gltf_path = output_dir.join(format!("{name}.gltf"));
    fs::write(&gltf_path, serde_json::to_string_pretty(&json)?)?;

    println!(
        "{}: wrote {} meshes, {} materials, {} textures",
        gltf_path.display(),
        mesh_count,
        material_count,
        image_count,
    );

    Ok(())
}

struct GltfDocument {
    binary_buffer: Vec<u8>,
    accessors_json: Vec<Value>,
    buffer_views_json: Vec<Value>,
    meshes_json: Vec<Value>,
    materials_json: Vec<Value>,
    nodes_json: Vec<Value>,
    images_json: Vec<Value>,
    textures_json: Vec<Value>,
    samplers_json: Vec<Value>,
    scene_root_node: Option<usize>,
}

impl GltfDocument {
    fn new() -> Self {
        Self {
            binary_buffer: Vec::new(),
            accessors_json: Vec::new(),
            buffer_views_json: Vec::new(),
            meshes_json: Vec::new(),
            materials_json: Vec::new(),
            nodes_json: Vec::new(),
            images_json: Vec::new(),
            textures_json: Vec::new(),
            samplers_json: vec![json!({
                "magFilter": 9729,
                "minFilter": 9987,
                "wrapS": 10497,
                "wrapT": 10497,
            })],
            scene_root_node: None,
        }
    }

    fn add_meshes(&mut self, meshes: &[Mesh]) -> Result<(), Box<dyn Error>> {
        for mesh in meshes {
            let primitive = self.encode_primitive(mesh)?;
            self.meshes_json.push(json!({
                "name": mesh.name,
                "primitives": [primitive],
            }));
        }
        Ok(())
    }

    fn encode_primitive(&mut self, mesh: &Mesh) -> Result<Value, Box<dyn Error>> {
        if mesh.vertices.is_empty() || mesh.faces.is_empty() {
            return Err(format!("mesh '{}' has no geometry", mesh.name).into());
        }

        let vertex_count = mesh.vertices.len();
        let position_accessor = self.encode_vec3_accessor(
            mesh.vertices.iter().map(|v| Vec3::new(v.x, v.y, v.z)),
            vertex_count,
            true,
            ARRAY_BUFFER_TARGET,
        );
        let normal_accessor = if mesh.normals.len() == vertex_count {
            Some(self.encode_vec3_accessor(
                mesh.normals.iter().map(|n| Vec3::new(n.x, n.y, n.z)),
                vertex_count,
                false,
                ARRAY_BUFFER_TARGET,
            ))
        } else {
            None
        };
        let uv_accessor = mesh
            .texture_coords
            .first()
            .and_then(|c| c.as_ref())
            .filter(|c| c.len() == vertex_count)
            .map(|coords| {
                self.encode_vec2_accessor(
                    coords.iter().map(|c| (c.x, 1.0 - c.y)),
                    vertex_count,
                    ARRAY_BUFFER_TARGET,
                )
            });

        let mut indices: Vec<u32> = Vec::with_capacity(mesh.faces.len() * 3);
        for face in &mesh.faces {
            if face.0.len() != 3 {
                continue;
            }
            indices.extend_from_slice(&face.0);
        }
        if indices.is_empty() {
            return Err(format!("mesh '{}' had no triangle faces", mesh.name).into());
        }
        let index_accessor = self.encode_index_accessor(&indices);

        let mut attributes = serde_json::Map::new();
        attributes.insert("POSITION".into(), Value::from(position_accessor));
        if let Some(index) = normal_accessor {
            attributes.insert("NORMAL".into(), Value::from(index));
        }
        if let Some(index) = uv_accessor {
            attributes.insert("TEXCOORD_0".into(), Value::from(index));
        }

        Ok(json!({
            "attributes": attributes,
            "indices": index_accessor,
            "material": mesh.material_index,
            "mode": 4,
        }))
    }

    fn encode_vec3_accessor(
        &mut self,
        values: impl IntoIterator<Item = Vec3>,
        count: usize,
        record_min_max: bool,
        target: u32,
    ) -> usize {
        let offset = align_buffer(&mut self.binary_buffer, 4);
        let start = self.binary_buffer.len();
        let mut bb_min = [f32::INFINITY; 3];
        let mut bb_max = [f32::NEG_INFINITY; 3];
        for v in values {
            self.binary_buffer.extend_from_slice(&v.x.to_le_bytes());
            self.binary_buffer.extend_from_slice(&v.y.to_le_bytes());
            self.binary_buffer.extend_from_slice(&v.z.to_le_bytes());
            if record_min_max {
                bb_min[0] = bb_min[0].min(v.x);
                bb_min[1] = bb_min[1].min(v.y);
                bb_min[2] = bb_min[2].min(v.z);
                bb_max[0] = bb_max[0].max(v.x);
                bb_max[1] = bb_max[1].max(v.y);
                bb_max[2] = bb_max[2].max(v.z);
            }
        }
        let _ = offset;
        let length = self.binary_buffer.len() - start;
        let buffer_view = self.add_buffer_view(start, length, Some(12), Some(target));
        let mut accessor = json!({
            "bufferView": buffer_view,
            "componentType": POSITION_COMPONENT_TYPE,
            "count": count,
            "type": "VEC3",
        });
        if record_min_max {
            accessor["min"] = json!([bb_min[0], bb_min[1], bb_min[2]]);
            accessor["max"] = json!([bb_max[0], bb_max[1], bb_max[2]]);
        }
        let index = self.accessors_json.len();
        self.accessors_json.push(accessor);
        index
    }

    fn encode_vec2_accessor(
        &mut self,
        values: impl IntoIterator<Item = (f32, f32)>,
        count: usize,
        target: u32,
    ) -> usize {
        align_buffer(&mut self.binary_buffer, 4);
        let start = self.binary_buffer.len();
        for (u, v) in values {
            self.binary_buffer.extend_from_slice(&u.to_le_bytes());
            self.binary_buffer.extend_from_slice(&v.to_le_bytes());
        }
        let length = self.binary_buffer.len() - start;
        let buffer_view = self.add_buffer_view(start, length, Some(8), Some(target));
        let accessor = json!({
            "bufferView": buffer_view,
            "componentType": POSITION_COMPONENT_TYPE,
            "count": count,
            "type": "VEC2",
        });
        let index = self.accessors_json.len();
        self.accessors_json.push(accessor);
        index
    }

    fn encode_index_accessor(&mut self, indices: &[u32]) -> usize {
        align_buffer(&mut self.binary_buffer, 4);
        let start = self.binary_buffer.len();
        for &i in indices {
            self.binary_buffer.extend_from_slice(&i.to_le_bytes());
        }
        let length = self.binary_buffer.len() - start;
        let buffer_view = self.add_buffer_view(start, length, None, Some(ELEMENT_ARRAY_BUFFER_TARGET));
        let accessor = json!({
            "bufferView": buffer_view,
            "componentType": INDEX_COMPONENT_TYPE_U32,
            "count": indices.len(),
            "type": "SCALAR",
        });
        let index = self.accessors_json.len();
        self.accessors_json.push(accessor);
        index
    }

    fn add_buffer_view(
        &mut self,
        offset: usize,
        length: usize,
        stride: Option<usize>,
        target: Option<u32>,
    ) -> usize {
        let mut view = json!({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": length,
        });
        if let Some(stride) = stride {
            view["byteStride"] = json!(stride);
        }
        if let Some(target) = target {
            view["target"] = json!(target);
        }
        let index = self.buffer_views_json.len();
        self.buffer_views_json.push(view);
        index
    }

    fn add_materials(
        &mut self,
        materials: &[Material],
        textures: &mut TextureResolver,
    ) -> Result<(), Box<dyn Error>> {
        for (index, material) in materials.iter().enumerate() {
            let info = MaterialInfo::from_russimp(material);
            let mut texture_indices = MaterialTextureIndices::default();

            if let Some(filename) = info.texture(TextureType::BaseColor)
                .or_else(|| info.texture(TextureType::Diffuse))
            {
                texture_indices.base_color =
                    self.register_texture_from_filename(filename, textures)?;
            }
            if let Some(filename) = info.texture(TextureType::EmissionColor)
                .or_else(|| info.texture(TextureType::Emissive))
            {
                texture_indices.emissive =
                    self.register_texture_from_filename(filename, textures)?;
            }
            if let Some(filename) = info.texture(TextureType::Normals) {
                texture_indices.normal =
                    self.register_texture_from_filename(filename, textures)?;
            }

            let mut pbr = serde_json::Map::new();
            let base_color = info.diffuse.unwrap_or(Vec4::ONE);
            pbr.insert(
                "baseColorFactor".into(),
                json!([base_color.x, base_color.y, base_color.z, base_color.w]),
            );
            if let Some(index) = texture_indices.base_color {
                pbr.insert("baseColorTexture".into(), json!({ "index": index }));
            }
            pbr.insert("metallicFactor".into(), json!(info.metallic_factor()));
            pbr.insert("roughnessFactor".into(), json!(info.roughness_factor()));

            let mut material_json = serde_json::Map::new();
            let material_name = info
                .name
                .clone()
                .unwrap_or_else(|| format!("material_{index}"));
            material_json.insert("name".into(), Value::String(material_name));
            material_json.insert("pbrMetallicRoughness".into(), Value::Object(pbr));
            let emissive = info.emissive.unwrap_or(Vec3::ZERO);
            material_json.insert(
                "emissiveFactor".into(),
                json!([emissive.x, emissive.y, emissive.z]),
            );
            if let Some(index) = texture_indices.emissive {
                material_json.insert("emissiveTexture".into(), json!({ "index": index }));
            }
            if let Some(index) = texture_indices.normal {
                material_json.insert("normalTexture".into(), json!({ "index": index }));
            }
            material_json.insert(
                "alphaMode".into(),
                Value::String(info.alpha_mode().to_string()),
            );
            if matches!(info.alpha_mode_value(), AlphaModeValue::Mask) {
                material_json.insert("alphaCutoff".into(), json!(info.alpha_cutoff));
            }
            material_json.insert("doubleSided".into(), Value::Bool(info.double_sided));

            self.materials_json.push(Value::Object(material_json));
        }
        Ok(())
    }

    fn register_texture_from_filename(
        &mut self,
        filename: &str,
        textures: &mut TextureResolver,
    ) -> Result<Option<usize>, Box<dyn Error>> {
        let Some(image_index) = textures.resolve(filename, &mut self.images_json)? else {
            return Ok(None);
        };
        if let Some(existing) = textures.texture_index(image_index) {
            return Ok(Some(existing));
        }
        let index = self.textures_json.len();
        self.textures_json.push(json!({
            "sampler": 0,
            "source": image_index,
        }));
        textures.register_texture_index(image_index, index);
        Ok(Some(index))
    }

    fn add_node_recursive(&mut self, node: &Rc<Node>) -> usize {
        let mut children_indices = Vec::new();
        for child in node.children.borrow().iter() {
            children_indices.push(self.add_node_recursive(child));
        }

        let mut json = serde_json::Map::new();
        json.insert("name".into(), Value::String(node.name.clone()));

        let (translation, rotation, scale) = decompose_matrix(&node.transformation);
        if !nearly_zero_translation(translation) {
            json.insert(
                "translation".into(),
                json!([translation.x, translation.y, translation.z]),
            );
        }
        if !nearly_identity_rotation(rotation) {
            json.insert(
                "rotation".into(),
                json!([rotation.x, rotation.y, rotation.z, rotation.w]),
            );
        }
        if !nearly_unit_scale(scale) {
            json.insert("scale".into(), json!([scale.x, scale.y, scale.z]));
        }

        if !node.meshes.is_empty() {
            if node.meshes.len() == 1 {
                json.insert("mesh".into(), json!(node.meshes[0]));
            } else {
                let merged_mesh = self.merge_mesh_refs(&node.meshes);
                json.insert("mesh".into(), json!(merged_mesh));
            }
        }
        if !children_indices.is_empty() {
            json.insert("children".into(), Value::Array(
                children_indices.into_iter().map(Value::from).collect(),
            ));
        }

        let index = self.nodes_json.len();
        self.nodes_json.push(Value::Object(json));
        index
    }

    fn merge_mesh_refs(&mut self, mesh_refs: &[u32]) -> usize {
        let mut primitives: Vec<Value> = Vec::with_capacity(mesh_refs.len());
        for &mesh_index in mesh_refs {
            let mesh = &self.meshes_json[mesh_index as usize];
            if let Some(prims) = mesh.get("primitives").and_then(Value::as_array) {
                for prim in prims {
                    primitives.push(prim.clone());
                }
            }
        }
        let merged_index = self.meshes_json.len();
        self.meshes_json.push(json!({
            "primitives": primitives,
        }));
        merged_index
    }

    fn set_scene_root(&mut self, root_index: usize) {
        self.scene_root_node = Some(root_index);
    }

    fn into_json(self, buffer_uri: String, name: &str) -> Value {
        let scene_nodes = self
            .scene_root_node
            .map(|index| vec![index])
            .unwrap_or_default();

        let asset = json!({
            "version": "2.0",
            "generator": format!("convert-bistro/{}", env!("CARGO_PKG_VERSION")),
        });

        let mut root = serde_json::Map::new();
        root.insert("asset".into(), asset);
        root.insert(
            "scene".into(),
            Value::from(if scene_nodes.is_empty() { 0 } else { 0 }),
        );
        root.insert(
            "scenes".into(),
            json!([{
                "name": name,
                "nodes": scene_nodes,
            }]),
        );
        root.insert("nodes".into(), Value::Array(self.nodes_json));
        root.insert("meshes".into(), Value::Array(self.meshes_json));
        root.insert("materials".into(), Value::Array(self.materials_json));
        if !self.textures_json.is_empty() {
            root.insert("textures".into(), Value::Array(self.textures_json));
        }
        if !self.images_json.is_empty() {
            root.insert("images".into(), Value::Array(self.images_json));
        }
        root.insert("samplers".into(), Value::Array(self.samplers_json));
        root.insert("accessors".into(), Value::Array(self.accessors_json));
        root.insert("bufferViews".into(), Value::Array(self.buffer_views_json));
        root.insert(
            "buffers".into(),
            json!([{
                "uri": buffer_uri,
                "byteLength": self.binary_buffer.len(),
            }]),
        );

        Value::Object(root)
    }
}

struct MaterialInfo {
    name: Option<String>,
    diffuse: Option<Vec4>,
    emissive: Option<Vec3>,
    shininess: Option<f32>,
    alpha_cutoff: f32,
    double_sided: bool,
    textures: HashMap<TextureType, String>,
    metallic_property: Option<f32>,
    roughness_property: Option<f32>,
}

#[derive(Default)]
struct MaterialTextureIndices {
    base_color: Option<usize>,
    emissive: Option<usize>,
    normal: Option<usize>,
}

#[derive(Clone, Copy)]
enum AlphaModeValue {
    Opaque,
    Mask,
    Blend,
}

impl AlphaModeValue {
    fn to_string(self) -> &'static str {
        match self {
            Self::Opaque => "OPAQUE",
            Self::Mask => "MASK",
            Self::Blend => "BLEND",
        }
    }
}

impl MaterialInfo {
    fn from_russimp(material: &Material) -> Self {
        let mut name = None;
        let mut diffuse = None;
        let mut emissive = None;
        let mut shininess = None;
        let mut opacity = 1.0_f32;
        let mut metallic_property = None;
        let mut roughness_property = None;

        for prop in &material.properties {
            match prop.key.as_str() {
                "?mat.name" => {
                    if let PropertyTypeInfo::String(value) = &prop.data {
                        name = Some(value.clone());
                    }
                }
                "$clr.diffuse" => {
                    if let PropertyTypeInfo::FloatArray(v) = &prop.data {
                        diffuse = float_array_to_vec4(v, 1.0);
                    }
                }
                "$clr.base" => {
                    if let PropertyTypeInfo::FloatArray(v) = &prop.data {
                        diffuse = float_array_to_vec4(v, 1.0);
                    }
                }
                "$clr.emissive" => {
                    if let PropertyTypeInfo::FloatArray(v) = &prop.data {
                        emissive = float_array_to_vec3(v);
                    }
                }
                "$mat.shininess" => {
                    if let PropertyTypeInfo::FloatArray(v) = &prop.data {
                        shininess = v.first().copied();
                    }
                }
                "$mat.opacity" => {
                    if let PropertyTypeInfo::FloatArray(v) = &prop.data {
                        opacity = v.first().copied().unwrap_or(1.0);
                    }
                }
                "$mat.metallicFactor" => {
                    if let PropertyTypeInfo::FloatArray(v) = &prop.data {
                        metallic_property = v.first().copied();
                    }
                }
                "$mat.roughnessFactor" => {
                    if let PropertyTypeInfo::FloatArray(v) = &prop.data {
                        roughness_property = v.first().copied();
                    }
                }
                _ => {}
            }
        }

        if let Some(d) = diffuse.as_mut() {
            d.w = (d.w * opacity).clamp(0.0, 1.0);
        }

        let mut textures = HashMap::new();
        for (kind, texture) in &material.textures {
            let texture_ref: Rc<RefCell<RussimpTexture>> = Rc::clone(texture);
            let filename = texture_ref.borrow().filename.clone();
            if !filename.is_empty() {
                textures.insert(*kind, filename);
            }
        }

        for prop in &material.properties {
            if prop.key.as_str() == "$tex.file" {
                if let PropertyTypeInfo::String(filename) = &prop.data {
                    textures.entry(prop.semantic).or_insert_with(|| filename.clone());
                }
            }
        }

        let _ = opacity;

        Self {
            name,
            diffuse,
            emissive,
            shininess,
            alpha_cutoff: 0.5,
            double_sided: false,
            textures,
            metallic_property,
            roughness_property,
        }
    }

    fn texture(&self, kind: TextureType) -> Option<&str> {
        self.textures.get(&kind).map(String::as_str)
    }

    fn metallic_factor(&self) -> f32 {
        self.metallic_property.unwrap_or(0.0)
    }

    fn roughness_factor(&self) -> f32 {
        if let Some(roughness) = self.roughness_property {
            return roughness.clamp(0.0, 1.0);
        }
        let shininess = self.shininess.unwrap_or(16.0).max(1.0);
        (2.0 / (shininess + 2.0)).sqrt().clamp(0.05, 1.0)
    }

    fn alpha_mode_value(&self) -> AlphaModeValue {
        let alpha = self.diffuse.map(|d| d.w).unwrap_or(1.0);
        if alpha < 0.999 {
            AlphaModeValue::Blend
        } else if self
            .textures
            .contains_key(&TextureType::Opacity)
        {
            AlphaModeValue::Mask
        } else {
            AlphaModeValue::Opaque
        }
    }

    fn alpha_mode(&self) -> &'static str {
        self.alpha_mode_value().to_string()
    }
}

fn float_array_to_vec4(values: &[f32], default_alpha: f32) -> Option<Vec4> {
    match values.len() {
        3 => Some(Vec4::new(values[0], values[1], values[2], default_alpha)),
        4 => Some(Vec4::new(values[0], values[1], values[2], values[3])),
        _ => None,
    }
}

fn float_array_to_vec3(values: &[f32]) -> Option<Vec3> {
    match values.len() {
        3 | 4 => Some(Vec3::new(values[0], values[1], values[2])),
        _ => None,
    }
}

struct TextureResolver {
    bistro_root: PathBuf,
    output_textures_dir: PathBuf,
    images_by_normalized_filename: HashMap<String, usize>,
    image_to_texture: HashMap<usize, usize>,
}

impl TextureResolver {
    fn new(bistro_root: &Path, output_textures_dir: &Path) -> Self {
        Self {
            bistro_root: bistro_root.to_path_buf(),
            output_textures_dir: output_textures_dir.to_path_buf(),
            images_by_normalized_filename: HashMap::new(),
            image_to_texture: HashMap::new(),
        }
    }

    fn resolve(
        &mut self,
        filename: &str,
        images_json: &mut Vec<Value>,
    ) -> Result<Option<usize>, Box<dyn Error>> {
        let normalized = normalize_filename(filename);
        if let Some(&index) = self.images_by_normalized_filename.get(&normalized) {
            return Ok(Some(index));
        }

        let basename = Path::new(&normalized)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&normalized)
            .to_string();
        let source = locate_source_texture(&self.bistro_root, &basename);
        let source = match source {
            Some(path) => path,
            None => {
                eprintln!(
                    "warning: convert-bistro could not locate texture '{filename}' under {}",
                    self.bistro_root.display()
                );
                return Ok(None);
            }
        };

        let stem = Path::new(&basename)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&basename);
        let png_filename = format!("{stem}.png");
        let png_relative = format!("textures/{png_filename}");
        let png_path = self.output_textures_dir.join(&png_filename);

        if is_dds_extension(&basename) {
            if !png_path.exists() {
                decode_dds_to_png(&source, &png_path)?;
            }
        } else if !png_path.exists() {
            fs::copy(&source, &png_path)?;
        }

        let image_index = images_json.len();
        images_json.push(json!({
            "name": basename,
            "uri": png_relative,
            "mimeType": "image/png",
        }));
        self.images_by_normalized_filename
            .insert(normalized, image_index);
        Ok(Some(image_index))
    }

    fn texture_index(&self, image_index: usize) -> Option<usize> {
        self.image_to_texture.get(&image_index).copied()
    }

    fn register_texture_index(&mut self, image_index: usize, texture_index: usize) {
        self.image_to_texture.insert(image_index, texture_index);
    }
}

fn normalize_filename(filename: &str) -> String {
    filename
        .replace('\\', "/")
        .trim_start_matches('/')
        .to_string()
}

fn is_dds_extension(filename: &str) -> bool {
    Path::new(filename)
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.eq_ignore_ascii_case("dds"))
        .unwrap_or(false)
}

fn locate_source_texture(bistro_root: &Path, basename: &str) -> Option<PathBuf> {
    let direct = bistro_root.join("Textures").join(basename);
    if direct.exists() {
        return Some(direct);
    }
    let case_insensitive = case_insensitive_lookup(&bistro_root.join("Textures"), basename);
    if case_insensitive.is_some() {
        return case_insensitive;
    }
    walk_for_basename(bistro_root, basename)
}

fn case_insensitive_lookup(dir: &Path, basename: &str) -> Option<PathBuf> {
    let target = basename.to_lowercase();
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let entry_name = entry.file_name();
        if let Some(name) = entry_name.to_str() {
            if name.to_lowercase() == target {
                return Some(entry.path());
            }
        }
    }
    None
}

fn walk_for_basename(root: &Path, basename: &str) -> Option<PathBuf> {
    let target = basename.to_lowercase();
    fn descend(dir: &Path, target: &str) -> Option<PathBuf> {
        for entry in fs::read_dir(dir).ok()?.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(found) = descend(&path, target) {
                    return Some(found);
                }
            } else if path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.to_lowercase() == target)
                .unwrap_or(false)
            {
                return Some(path);
            }
        }
        None
    }
    descend(root, &target)
}

fn decode_dds_to_png(source: &Path, destination: &Path) -> Result<(), Box<dyn Error>> {
    let mut reader = File::open(source)?;
    let dds = ddsfile::Dds::read(&mut reader)
        .map_err(|error| format!("failed to parse {}: {error}", source.display()))?;
    let image = image_dds::image_from_dds(&dds, 0)
        .map_err(|error| format!("failed to decode {}: {error}", source.display()))?;
    image
        .save(destination)
        .map_err(|error| format!("failed to write {}: {error}", destination.display()))?;
    Ok(())
}

fn align_buffer(buffer: &mut Vec<u8>, alignment: usize) -> usize {
    let remainder = buffer.len() % alignment;
    if remainder != 0 {
        buffer.resize(buffer.len() + alignment - remainder, 0);
    }
    buffer.len()
}

fn decompose_matrix(matrix: &Matrix4x4) -> (Vec3, Quat, Vec3) {
    let mat = Mat4::from_cols(
        Vec4::new(matrix.a1, matrix.b1, matrix.c1, matrix.d1),
        Vec4::new(matrix.a2, matrix.b2, matrix.c2, matrix.d2),
        Vec4::new(matrix.a3, matrix.b3, matrix.c3, matrix.d3),
        Vec4::new(matrix.a4, matrix.b4, matrix.c4, matrix.d4),
    );
    let (scale, rotation, translation) = mat.to_scale_rotation_translation();
    (translation, rotation, scale)
}

fn nearly_zero_translation(translation: Vec3) -> bool {
    translation.length_squared() < 1.0e-12
}

fn nearly_identity_rotation(rotation: Quat) -> bool {
    let xyz = Vec3::new(rotation.x, rotation.y, rotation.z);
    xyz.length_squared() < 1.0e-12 && (rotation.w - 1.0).abs() < 1.0e-6
}

fn nearly_unit_scale(scale: Vec3) -> bool {
    (scale - Vec3::ONE).length_squared() < 1.0e-12
}

