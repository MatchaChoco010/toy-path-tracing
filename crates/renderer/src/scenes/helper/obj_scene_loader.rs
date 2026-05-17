use std::{
    collections::HashMap,
    fmt, fs,
    path::{Path, PathBuf},
};

use glam::{Vec2, Vec3};

use crate::scene::{
    LoadMeshError, Mesh, ObjVertexKey, Vertex, append_obj_vertex, generate_vertex_normals,
    obj_error, parse_obj_face_corner, parse_obj_float,
};

#[derive(Debug, Clone, PartialEq)]
pub struct ObjMaterial {
    pub name: String,
    pub diffuse: Vec3,
    pub specular_exponent: f32,
    pub dissolve: f32,
    pub illum: u32,
    pub transmission_filter: Vec3,
    pub emission: Vec3,
    pub diffuse_texture_path: Option<PathBuf>,
    pub emission_texture_path: Option<PathBuf>,
}

impl Default for ObjMaterial {
    fn default() -> Self {
        Self {
            name: String::new(),
            diffuse: Vec3::ONE,
            specular_exponent: 16.0,
            dissolve: 1.0,
            illum: 2,
            transmission_filter: Vec3::ZERO,
            emission: Vec3::ZERO,
            diffuse_texture_path: None,
            emission_texture_path: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObjMaterialMesh {
    pub material_name: Option<String>,
    pub mesh: Mesh,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObjScene {
    pub material_meshes: Vec<ObjMaterialMesh>,
    pub materials: Vec<ObjMaterial>,
    pub mtl_dir: PathBuf,
}

impl ObjScene {
    pub fn material(&self, name: &str) -> Option<&ObjMaterial> {
        self.materials.iter().find(|material| material.name == name)
    }
}

#[derive(Debug)]
pub enum LoadObjSceneError {
    Io(std::io::Error),
    Mesh(LoadMeshError),
    Mtl {
        path: PathBuf,
        line: usize,
        message: String,
    },
}

impl fmt::Display for LoadObjSceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Mesh(error) => write!(f, "{error}"),
            Self::Mtl {
                path,
                line,
                message,
            } => {
                write!(f, "MTL parse error in {path:?} on line {line}: {message}")
            }
        }
    }
}

impl std::error::Error for LoadObjSceneError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Mesh(error) => Some(error),
            Self::Mtl { .. } => None,
        }
    }
}

impl From<std::io::Error> for LoadObjSceneError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<LoadMeshError> for LoadObjSceneError {
    fn from(error: LoadMeshError) -> Self {
        Self::Mesh(error)
    }
}

pub fn load_obj_scene(obj_path: &Path) -> Result<ObjScene, LoadObjSceneError> {
    let obj_path = crate::utils::workspace_path(obj_path);
    let source = fs::read_to_string(&obj_path)?;
    let dir = obj_path.parent().map(Path::to_path_buf).unwrap_or_default();
    let scene = parse_obj_scene_source(&source, &dir)?;

    Ok(scene)
}

fn parse_obj_scene_source(source: &str, dir: &Path) -> Result<ObjScene, LoadObjSceneError> {
    let mut positions = Vec::new();
    let mut uvs = Vec::new();
    let mut normals = Vec::new();
    let mut mtl_libs: Vec<PathBuf> = Vec::new();

    let mut builders: HashMap<String, MaterialMeshBuilder> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    let mut current_material = String::new();

    for (line_index, raw_line) in source.lines().enumerate() {
        let line_number = line_index + 1;
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }

        let mut fields = line.split_whitespace();
        let directive = fields.next().expect("non-empty line must have a directive");

        match directive {
            "v" => {
                positions.push(Vec3::new(
                    parse_obj_float(&mut fields, line_number, "x position")?,
                    parse_obj_float(&mut fields, line_number, "y position")?,
                    parse_obj_float(&mut fields, line_number, "z position")?,
                ));
            }
            "vt" => {
                let u = parse_obj_float(&mut fields, line_number, "u texture coordinate")?;
                let v = parse_obj_float(&mut fields, line_number, "v texture coordinate")?;
                uvs.push(Vec2::new(u, 1.0 - v));
            }
            "vn" => {
                normals.push(
                    Vec3::new(
                        parse_obj_float(&mut fields, line_number, "x normal")?,
                        parse_obj_float(&mut fields, line_number, "y normal")?,
                        parse_obj_float(&mut fields, line_number, "z normal")?,
                    )
                    .normalize_or_zero(),
                );
            }
            "mtllib" => {
                for token in fields {
                    mtl_libs.push(dir.join(normalize_obj_path(token)));
                }
            }
            "usemtl" => {
                current_material = fields.next().unwrap_or("").to_string();
            }
            "f" => {
                let mut corners = Vec::new();
                for token in fields {
                    corners.push(parse_obj_face_corner(
                        token,
                        line_number,
                        positions.len(),
                        uvs.len(),
                        normals.len(),
                    )?);
                }
                if corners.len() < 3 {
                    return Err(LoadObjSceneError::Mesh(obj_error(
                        line_number,
                        "face must reference at least three vertices",
                    )));
                }

                let key = current_material.clone();
                let builder = match builders.get_mut(&key) {
                    Some(builder) => builder,
                    None => {
                        order.push(key.clone());
                        builders.insert(key.clone(), MaterialMeshBuilder::default());
                        builders.get_mut(&key).expect("builder just inserted")
                    }
                };

                for corner_index in 1..corners.len() - 1 {
                    for corner in [corners[0], corners[corner_index], corners[corner_index + 1]] {
                        let index = append_obj_vertex(
                            corner,
                            &positions,
                            &uvs,
                            &normals,
                            &mut builder.vertices,
                            &mut builder.vertex_map,
                        )?;
                        builder.indices.push(index);
                    }
                }
            }
            _ => {}
        }
    }

    let mut material_meshes = Vec::with_capacity(order.len());
    for name in order {
        let builder = builders.remove(&name).expect("builder must exist");
        let MaterialMeshBuilder {
            mut vertices,
            indices,
            ..
        } = builder;
        if vertices.is_empty() || indices.is_empty() {
            continue;
        }

        let positions: Vec<Vec3> = vertices.iter().map(|vertex| vertex.position).collect();
        let generated = generate_vertex_normals(&positions, &indices);
        for (vertex, generated_normal) in vertices.iter_mut().zip(generated) {
            if vertex.normal.length_squared() == 0.0 {
                vertex.normal = generated_normal;
            }
        }

        let mesh = Mesh::new(vertices, indices);
        material_meshes.push(ObjMaterialMesh {
            material_name: if name.is_empty() { None } else { Some(name) },
            mesh,
        });
    }

    let mut materials = Vec::new();
    for mtl_path in &mtl_libs {
        let mtl_materials = parse_mtl_file(mtl_path)?;
        materials.extend(mtl_materials);
    }

    Ok(ObjScene {
        material_meshes,
        materials,
        mtl_dir: dir.to_path_buf(),
    })
}

#[derive(Default)]
struct MaterialMeshBuilder {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    vertex_map: HashMap<ObjVertexKey, u32>,
}

fn parse_mtl_file(path: &Path) -> Result<Vec<ObjMaterial>, LoadObjSceneError> {
    let source = fs::read_to_string(path)?;
    parse_mtl_source(&source, path)
}

fn parse_mtl_source(source: &str, path: &Path) -> Result<Vec<ObjMaterial>, LoadObjSceneError> {
    let mut materials = Vec::new();
    let mut current: Option<ObjMaterial> = None;

    for (line_index, raw_line) in source.lines().enumerate() {
        let line_number = line_index + 1;
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }

        let mut fields = line.split_whitespace();
        let directive = fields.next().expect("non-empty line must have a directive");

        match directive {
            "newmtl" => {
                if let Some(material) = current.take() {
                    materials.push(material);
                }
                let name = fields
                    .next()
                    .ok_or_else(|| mtl_error(path, line_number, "newmtl missing name"))?
                    .to_string();
                current = Some(ObjMaterial {
                    name,
                    ..ObjMaterial::default()
                });
            }
            "Kd" => {
                let material = current.as_mut().ok_or_else(|| {
                    mtl_error(path, line_number, "Kd before any newmtl directive")
                })?;
                material.diffuse = parse_mtl_color(&mut fields, path, line_number, "Kd")?;
            }
            "Ke" => {
                let material = current.as_mut().ok_or_else(|| {
                    mtl_error(path, line_number, "Ke before any newmtl directive")
                })?;
                material.emission = parse_mtl_color(&mut fields, path, line_number, "Ke")?;
            }
            "Tf" => {
                let material = current.as_mut().ok_or_else(|| {
                    mtl_error(path, line_number, "Tf before any newmtl directive")
                })?;
                material.transmission_filter =
                    parse_mtl_color(&mut fields, path, line_number, "Tf")?;
            }
            "Ns" => {
                let material = current.as_mut().ok_or_else(|| {
                    mtl_error(path, line_number, "Ns before any newmtl directive")
                })?;
                material.specular_exponent =
                    parse_mtl_scalar(&mut fields, path, line_number, "Ns")?;
            }
            "d" => {
                let material = current
                    .as_mut()
                    .ok_or_else(|| mtl_error(path, line_number, "d before any newmtl directive"))?;
                material.dissolve = parse_mtl_scalar(&mut fields, path, line_number, "d")?;
            }
            "Tr" => {
                let material = current.as_mut().ok_or_else(|| {
                    mtl_error(path, line_number, "Tr before any newmtl directive")
                })?;
                let tr = parse_mtl_scalar(&mut fields, path, line_number, "Tr")?;
                material.dissolve = (1.0 - tr).clamp(0.0, 1.0);
            }
            "illum" => {
                let material = current.as_mut().ok_or_else(|| {
                    mtl_error(path, line_number, "illum before any newmtl directive")
                })?;
                let value = fields
                    .next()
                    .ok_or_else(|| mtl_error(path, line_number, "illum missing value"))?;
                material.illum = value.parse::<u32>().map_err(|_| {
                    mtl_error(path, line_number, format!("invalid illum value '{value}'"))
                })?;
            }
            "map_Kd" => {
                let material = current.as_mut().ok_or_else(|| {
                    mtl_error(path, line_number, "map_Kd before any newmtl directive")
                })?;
                let path_text = line
                    .split_once(char::is_whitespace)
                    .map(|(_, rest)| rest.trim())
                    .unwrap_or("");
                if path_text.is_empty() {
                    return Err(mtl_error(path, line_number, "map_Kd missing texture path"));
                }
                material.diffuse_texture_path = Some(normalize_obj_path(path_text));
            }
            "map_Ke" => {
                let material = current.as_mut().ok_or_else(|| {
                    mtl_error(path, line_number, "map_Ke before any newmtl directive")
                })?;
                let path_text = line
                    .split_once(char::is_whitespace)
                    .map(|(_, rest)| rest.trim())
                    .unwrap_or("");
                if path_text.is_empty() {
                    return Err(mtl_error(path, line_number, "map_Ke missing texture path"));
                }
                material.emission_texture_path = Some(normalize_obj_path(path_text));
            }
            _ => {}
        }
    }

    if let Some(material) = current.take() {
        materials.push(material);
    }

    Ok(materials)
}

fn parse_mtl_color<'a>(
    fields: &mut impl Iterator<Item = &'a str>,
    path: &Path,
    line_number: usize,
    label: &str,
) -> Result<Vec3, LoadObjSceneError> {
    let r = parse_mtl_scalar(fields, path, line_number, label)?;
    let g = parse_mtl_scalar(fields, path, line_number, label).unwrap_or(r);
    let b = parse_mtl_scalar(fields, path, line_number, label).unwrap_or(r);
    Ok(Vec3::new(r, g, b))
}

fn parse_mtl_scalar<'a>(
    fields: &mut impl Iterator<Item = &'a str>,
    path: &Path,
    line_number: usize,
    label: &str,
) -> Result<f32, LoadObjSceneError> {
    let token = fields
        .next()
        .ok_or_else(|| mtl_error(path, line_number, format!("missing {label}")))?;
    if let Ok(value) = token.parse::<f32>() {
        return Ok(value);
    }
    if let Some(value) = parse_leading_f32(token) {
        return Ok(value);
    }
    Err(mtl_error(
        path,
        line_number,
        format!("invalid {label} '{token}'"),
    ))
}

fn parse_leading_f32(token: &str) -> Option<f32> {
    let bytes = token.as_bytes();
    let mut end = 0usize;
    let mut best: Option<f32> = None;
    while end < bytes.len() {
        end += 1;
        if let Ok(value) = token[..end].parse::<f32>() {
            best = Some(value);
        }
    }
    best
}

fn mtl_error(path: &Path, line_number: usize, message: impl Into<String>) -> LoadObjSceneError {
    LoadObjSceneError::Mtl {
        path: path.to_path_buf(),
        line: line_number,
        message: message.into(),
    }
}

fn normalize_obj_path(token: &str) -> PathBuf {
    PathBuf::from(token.replace('\\', "/"))
}

#[cfg(test)]
mod tests {
    use super::{parse_mtl_source, parse_obj_scene_source};
    use std::path::Path;

    #[test]
    fn obj_scene_splits_meshes_per_material() {
        let source = "
v 0 0 0
v 1 0 0
v 0 1 0
v 1 1 0
v 2 0 0
v 2 1 0
usemtl red
f 1 2 3
f 2 4 3
usemtl blue
f 2 5 4
f 5 6 4
";
        let scene = parse_obj_scene_source(source, Path::new(".")).expect("parsed");
        assert_eq!(scene.material_meshes.len(), 2);
        let names: Vec<_> = scene
            .material_meshes
            .iter()
            .map(|m| m.material_name.as_deref().unwrap())
            .collect();
        assert_eq!(names, vec!["red", "blue"]);
        for slot in &scene.material_meshes {
            assert_eq!(slot.mesh.triangle_count(), 2);
        }
    }

    #[test]
    fn obj_scene_handles_unassigned_faces() {
        let source = "
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
";
        let scene = parse_obj_scene_source(source, Path::new(".")).expect("parsed");
        assert_eq!(scene.material_meshes.len(), 1);
        assert!(scene.material_meshes[0].material_name.is_none());
        assert_eq!(scene.material_meshes[0].mesh.triangle_count(), 1);
    }

    #[test]
    fn mtl_parses_scalar_with_trailing_garbage_via_leading_prefix() {
        let source = "
newmtl wall
Ns 100.000Textures\\
Kd 0.4 0.5 0.6
illum 2
";
        let materials = parse_mtl_source(source, Path::new("scene.mtl")).expect("parsed");
        assert_eq!(materials.len(), 1);
        assert_eq!(materials[0].specular_exponent, 100.0);
    }

    #[test]
    fn mtl_map_kd_supports_paths_with_spaces() {
        let source = "
newmtl sign
Kd 1 1 1
illum 2
map_Kd ../PropTextures/Paris_ShopSign_ties shop_diff.png
";
        let materials = parse_mtl_source(source, Path::new("scene.mtl")).expect("parsed");
        assert_eq!(materials.len(), 1);
        assert_eq!(
            materials[0].diffuse_texture_path.as_deref(),
            Some(Path::new(
                "../PropTextures/Paris_ShopSign_ties shop_diff.png"
            )),
        );
    }

    #[test]
    fn mtl_map_ke_records_emission_texture_path() {
        let source = "
newmtl lantern
Kd 1 1 1
Ns 32
illum 2
map_Kd ../PropTextures/Paris_Lantern_01A_diff.png
map_Ke ..\\PropTextures\\Paris_Lantern_01A_emi.png
";
        let materials = parse_mtl_source(source, Path::new("scene.mtl")).expect("parsed");
        assert_eq!(materials.len(), 1);
        assert_eq!(
            materials[0].emission_texture_path.as_deref(),
            Some(Path::new("../PropTextures/Paris_Lantern_01A_emi.png"))
        );
    }

    #[test]
    fn mtl_parses_kd_dissolve_illum_tf_ke_and_map_kd() {
        let source = "
newmtl wall
Kd 0.4 0.5 0.6
Ns 32
d 1
illum 2
map_Kd textures\\wall_color.png

newmtl glass
Kd 1 1 1
Ns 1024
d 1
illum 4
Tf 0.1 0.2 0.3
Ke 0.5 0.5 0.5
";
        let materials = parse_mtl_source(source, Path::new("scene.mtl")).expect("parsed");
        assert_eq!(materials.len(), 2);
        assert_eq!(materials[0].name, "wall");
        assert!(
            materials[0]
                .diffuse
                .abs_diff_eq(glam::Vec3::new(0.4, 0.5, 0.6), 1.0e-6)
        );
        assert_eq!(materials[0].specular_exponent, 32.0);
        assert_eq!(materials[0].dissolve, 1.0);
        assert_eq!(materials[0].illum, 2);
        assert_eq!(
            materials[0].diffuse_texture_path.as_deref(),
            Some(Path::new("textures/wall_color.png"))
        );

        assert_eq!(materials[1].name, "glass");
        assert_eq!(materials[1].illum, 4);
        assert!(
            materials[1]
                .transmission_filter
                .abs_diff_eq(glam::Vec3::new(0.1, 0.2, 0.3), 1.0e-6)
        );
        assert!(
            materials[1]
                .emission
                .abs_diff_eq(glam::Vec3::splat(0.5), 1.0e-6)
        );
        assert!(materials[1].diffuse_texture_path.is_none());
    }
}
