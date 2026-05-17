use glam::{Mat3, Mat4, Vec2, Vec3};
use std::{
    collections::HashMap,
    fmt, fs,
    io::{self, Read, Seek},
    path::Path,
};

use crate::qbvh::{Qbvh, build_qbvh};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub min: Vec3,
    pub max: Vec3,
}

impl Bounds {
    pub const EMPTY: Self = Self {
        min: Vec3::splat(f32::INFINITY),
        max: Vec3::splat(f32::NEG_INFINITY),
    };

    pub fn center(&self) -> Vec3 {
        0.5 * (self.min + self.max)
    }

    pub fn extent(&self) -> Vec3 {
        self.max - self.min
    }

    pub fn surface_area(&self) -> f32 {
        let extent = self.extent().max(Vec3::ZERO);
        2.0 * (extent.x * extent.y + extent.x * extent.z + extent.y * extent.z)
    }

    pub fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub bounds: Bounds,
    pub qbvh: Option<Qbvh>,
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        let bounds = compute_bounds(&vertices).expect("mesh must contain at least one vertex");

        Self {
            vertices,
            indices,
            bounds,
            qbvh: None,
        }
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn triangle_positions(&self, triangle_index: usize) -> [Vec3; 3] {
        let [i0, i1, i2] = self.triangle_indices(triangle_index);

        [
            self.vertices[i0 as usize].position,
            self.vertices[i1 as usize].position,
            self.vertices[i2 as usize].position,
        ]
    }

    pub fn triangle_normals(&self, triangle_index: usize) -> [Vec3; 3] {
        let [i0, i1, i2] = self.triangle_indices(triangle_index);

        [
            self.vertices[i0 as usize].normal,
            self.vertices[i1 as usize].normal,
            self.vertices[i2 as usize].normal,
        ]
    }

    pub fn triangle_uvs(&self, triangle_index: usize) -> [Vec2; 3] {
        let [i0, i1, i2] = self.triangle_indices(triangle_index);

        [
            self.vertices[i0 as usize].uv,
            self.vertices[i1 as usize].uv,
            self.vertices[i2 as usize].uv,
        ]
    }

    pub fn triangle_bounds(&self, triangle_index: usize) -> Bounds {
        let [v0, v1, v2] = self.triangle_positions(triangle_index);

        Bounds {
            min: v0.min(v1).min(v2),
            max: v0.max(v1).max(v2),
        }
    }

    pub fn build_qbvh(&mut self) {
        let triangle_bounds = (0..self.triangle_count())
            .map(|triangle_index| self.triangle_bounds(triangle_index))
            .collect::<Vec<_>>();
        self.qbvh = build_qbvh(&triangle_bounds);
    }

    fn triangle_indices(&self, triangle_index: usize) -> [u32; 3] {
        let base = 3 * triangle_index;
        [
            self.indices[base],
            self.indices[base + 1],
            self.indices[base + 2],
        ]
    }
}

#[derive(Debug)]
pub enum LoadMeshError {
    Gltf(gltf::Error),
    Io(std::io::Error),
    Obj {
        line: usize,
        message: String,
    },
    Stl {
        message: String,
    },
    EmptyMesh,
    MissingPositions {
        mesh_index: usize,
        primitive_index: usize,
    },
    VertexCountOverflow,
    InvalidTriangleIndexCount {
        mesh_index: usize,
        primitive_index: usize,
    },
    UnsupportedPrimitiveMode {
        mesh_index: usize,
        primitive_index: usize,
        mode: gltf::mesh::Mode,
    },
}

impl fmt::Display for LoadMeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gltf(error) => write!(f, "{error}"),
            Self::Io(error) => write!(f, "{error}"),
            Self::Obj { line, message } => write!(f, "OBJ parse error on line {line}: {message}"),
            Self::Stl { message } => write!(f, "STL parse error: {message}"),
            Self::EmptyMesh => write!(f, "the mesh asset did not contain any triangle mesh data"),
            Self::MissingPositions {
                mesh_index,
                primitive_index,
            } => write!(
                f,
                "mesh {mesh_index} primitive {primitive_index} did not have POSITION data"
            ),
            Self::VertexCountOverflow => write!(f, "the mesh had more than u32::MAX vertices"),
            Self::InvalidTriangleIndexCount {
                mesh_index,
                primitive_index,
            } => write!(
                f,
                "mesh {mesh_index} primitive {primitive_index} did not contain indices in groups of three"
            ),
            Self::UnsupportedPrimitiveMode {
                mesh_index,
                primitive_index,
                mode,
            } => write!(
                f,
                "mesh {mesh_index} primitive {primitive_index} used unsupported mode {mode:?}"
            ),
        }
    }
}

impl std::error::Error for LoadMeshError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gltf(error) => Some(error),
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<gltf::Error> for LoadMeshError {
    fn from(error: gltf::Error) -> Self {
        Self::Gltf(error)
    }
}

impl From<std::io::Error> for LoadMeshError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

pub fn load_gltf(path: &Path) -> Result<Mesh, LoadMeshError> {
    let path = crate::utils::workspace_path(path);
    let (document, buffers, _) = gltf::import(&path)?;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut appended_mesh = false;

    if let Some(scene) = document
        .default_scene()
        .or_else(|| document.scenes().next())
    {
        for node in scene.nodes() {
            appended_mesh |=
                append_gltf_node(&buffers, node, Mat4::IDENTITY, &mut vertices, &mut indices)?;
        }
    } else {
        for mesh in document.meshes() {
            append_gltf_mesh(&buffers, mesh, Mat4::IDENTITY, &mut vertices, &mut indices)?;
            appended_mesh = true;
        }
    }

    if !appended_mesh || vertices.is_empty() || indices.is_empty() {
        return Err(LoadMeshError::EmptyMesh);
    }

    Ok(Mesh::new(vertices, indices))
}

pub fn load_obj(path: &Path) -> Result<Mesh, LoadMeshError> {
    let path = crate::utils::workspace_path(path);
    let source = fs::read_to_string(path)?;
    parse_obj(&source)
}

pub fn load_stl(path: &Path) -> Result<Mesh, LoadMeshError> {
    let path = crate::utils::workspace_path(path);
    let mut reader = fs::File::open(path)?;
    read_stl_into_mesh(&mut reader)
}

fn append_gltf_node(
    buffers: &[gltf::buffer::Data],
    node: gltf::Node<'_>,
    parent_transform: Mat4,
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
) -> Result<bool, LoadMeshError> {
    let local_transform = Mat4::from_cols_array_2d(&node.transform().matrix());
    let node_transform = parent_transform * local_transform;
    let mut appended_mesh = false;

    if let Some(mesh) = node.mesh() {
        append_gltf_mesh(buffers, mesh, node_transform, vertices, indices)?;
        appended_mesh = true;
    }

    for child in node.children() {
        appended_mesh |= append_gltf_node(buffers, child, node_transform, vertices, indices)?;
    }

    Ok(appended_mesh)
}

fn append_gltf_mesh(
    buffers: &[gltf::buffer::Data],
    mesh: gltf::Mesh<'_>,
    transform: Mat4,
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
) -> Result<(), LoadMeshError> {
    let normal_transform = Mat3::from_mat4(transform.inverse().transpose());

    for primitive in mesh.primitives() {
        if primitive.mode() != gltf::mesh::Mode::Triangles {
            return Err(LoadMeshError::UnsupportedPrimitiveMode {
                mesh_index: mesh.index(),
                primitive_index: primitive.index(),
                mode: primitive.mode(),
            });
        }

        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
        let positions = reader
            .read_positions()
            .ok_or_else(|| LoadMeshError::MissingPositions {
                mesh_index: mesh.index(),
                primitive_index: primitive.index(),
            })?
            .map(Vec3::from_array)
            .map(|position| transform.transform_point3(position))
            .collect::<Vec<_>>();

        let local_indices = reader
            .read_indices()
            .map(|indices| indices.into_u32().collect::<Vec<_>>())
            .unwrap_or_else(|| (0..positions.len() as u32).collect());
        let uvs = reader
            .read_tex_coords(0)
            .map(|uvs| uvs.into_f32().map(Vec2::from_array).collect::<Vec<_>>())
            .unwrap_or_else(|| vec![Vec2::ZERO; positions.len()]);

        if local_indices.len() % 3 != 0 {
            return Err(LoadMeshError::InvalidTriangleIndexCount {
                mesh_index: mesh.index(),
                primitive_index: primitive.index(),
            });
        }

        let generated_normals = generate_vertex_normals(&positions, &local_indices);
        let normals = reader
            .read_normals()
            .map(|normals| {
                normals
                    .map(Vec3::from_array)
                    .map(|normal| normal_transform.mul_vec3(normal).normalize_or_zero())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| generated_normals.clone());

        let base_vertex =
            u32::try_from(vertices.len()).map_err(|_| LoadMeshError::VertexCountOverflow)?;

        for (index, position) in positions.into_iter().enumerate() {
            let normal = normals[index];
            vertices.push(Vertex {
                position,
                normal: if normal.length_squared() > 0.0 {
                    normal
                } else {
                    generated_normals[index]
                },
                uv: uvs[index],
            });
        }

        indices.extend(local_indices.into_iter().map(|index| base_vertex + index));
    }

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ObjVertexKey {
    pub position_index: usize,
    pub uv_index: Option<usize>,
    pub normal_index: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ObjFaceCorner {
    pub position_index: usize,
    pub uv_index: Option<usize>,
    pub normal_index: Option<usize>,
}

fn parse_obj(source: &str) -> Result<Mesh, LoadMeshError> {
    let mut positions = Vec::new();
    let mut uvs = Vec::new();
    let mut normals = Vec::new();
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut vertex_map = HashMap::new();

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
                    return Err(obj_error(
                        line_number,
                        "face must reference at least three vertices",
                    ));
                }

                for corner_index in 1..corners.len() - 1 {
                    for corner in [corners[0], corners[corner_index], corners[corner_index + 1]] {
                        indices.push(append_obj_vertex(
                            corner,
                            &positions,
                            &uvs,
                            &normals,
                            &mut vertices,
                            &mut vertex_map,
                        )?);
                    }
                }
            }
            _ => {}
        }
    }

    if vertices.is_empty() || indices.is_empty() {
        return Err(LoadMeshError::EmptyMesh);
    }

    let positions = vertices
        .iter()
        .map(|vertex| vertex.position)
        .collect::<Vec<_>>();
    let generated_normals = generate_vertex_normals(&positions, &indices);
    for (vertex, generated_normal) in vertices.iter_mut().zip(generated_normals) {
        if vertex.normal.length_squared() == 0.0 {
            vertex.normal = generated_normal;
        }
    }

    Ok(Mesh::new(vertices, indices))
}

pub(crate) fn append_obj_vertex(
    corner: ObjFaceCorner,
    positions: &[Vec3],
    uvs: &[Vec2],
    normals: &[Vec3],
    vertices: &mut Vec<Vertex>,
    vertex_map: &mut HashMap<ObjVertexKey, u32>,
) -> Result<u32, LoadMeshError> {
    let key = ObjVertexKey {
        position_index: corner.position_index,
        uv_index: corner.uv_index,
        normal_index: corner.normal_index,
    };
    if let Some(index) = vertex_map.get(&key) {
        return Ok(*index);
    }

    let index = u32::try_from(vertices.len()).map_err(|_| LoadMeshError::VertexCountOverflow)?;
    vertices.push(Vertex {
        position: positions[corner.position_index],
        normal: corner
            .normal_index
            .map(|normal_index| normals[normal_index])
            .unwrap_or(Vec3::ZERO),
        uv: corner
            .uv_index
            .map(|uv_index| uvs[uv_index])
            .unwrap_or(Vec2::ZERO),
    });
    vertex_map.insert(key, index);

    Ok(index)
}

pub(crate) fn parse_obj_face_corner(
    token: &str,
    line_number: usize,
    position_count: usize,
    uv_count: usize,
    normal_count: usize,
) -> Result<ObjFaceCorner, LoadMeshError> {
    let mut parts = token.split('/');
    let position = parts.next().unwrap_or("");
    let uv = parts.next();
    let normal = parts.next();
    if parts.next().is_some() {
        return Err(obj_error(
            line_number,
            format!("invalid face vertex '{token}'"),
        ));
    }
    if position.is_empty() {
        return Err(obj_error(
            line_number,
            format!("face vertex '{token}' did not include a position index"),
        ));
    }

    Ok(ObjFaceCorner {
        position_index: resolve_obj_index(position, position_count, "position", line_number)?,
        uv_index: resolve_optional_obj_index(uv, uv_count, "texture coordinate", line_number)?,
        normal_index: resolve_optional_obj_index(normal, normal_count, "normal", line_number)?,
    })
}

fn resolve_optional_obj_index(
    raw_index: Option<&str>,
    value_count: usize,
    label: &str,
    line_number: usize,
) -> Result<Option<usize>, LoadMeshError> {
    match raw_index {
        Some(raw_index) if !raw_index.is_empty() => {
            resolve_obj_index(raw_index, value_count, label, line_number).map(Some)
        }
        _ => Ok(None),
    }
}

fn resolve_obj_index(
    raw_index: &str,
    value_count: usize,
    label: &str,
    line_number: usize,
) -> Result<usize, LoadMeshError> {
    let parsed_index = raw_index
        .parse::<isize>()
        .map_err(|_| obj_error(line_number, format!("invalid {label} index '{raw_index}'")))?;
    if parsed_index == 0 {
        return Err(obj_error(
            line_number,
            format!("{label} indices are 1-based and cannot be 0"),
        ));
    }

    let resolved_index = if parsed_index > 0 {
        parsed_index - 1
    } else {
        value_count as isize + parsed_index
    };
    if resolved_index < 0 || resolved_index >= value_count as isize {
        return Err(obj_error(
            line_number,
            format!("{label} index '{raw_index}' was out of bounds"),
        ));
    }

    Ok(resolved_index as usize)
}

pub(crate) fn parse_obj_float<'a>(
    fields: &mut impl Iterator<Item = &'a str>,
    line_number: usize,
    label: &str,
) -> Result<f32, LoadMeshError> {
    let token = fields
        .next()
        .ok_or_else(|| obj_error(line_number, format!("missing {label}")))?;
    token
        .parse::<f32>()
        .map_err(|_| obj_error(line_number, format!("invalid {label} '{token}'")))
}

pub(crate) fn obj_error(line: usize, message: impl Into<String>) -> LoadMeshError {
    LoadMeshError::Obj {
        line,
        message: message.into(),
    }
}

fn read_stl_into_mesh<R: Read + Seek>(reader: &mut R) -> Result<Mesh, LoadMeshError> {
    let indexed = stl_io::read_stl(reader).map_err(stl_io_error)?;
    build_stl_mesh(indexed)
}

fn build_stl_mesh(indexed: stl_io::IndexedMesh) -> Result<Mesh, LoadMeshError> {
    if indexed.faces.is_empty() || indexed.vertices.is_empty() {
        return Err(LoadMeshError::EmptyMesh);
    }

    let positions: Vec<Vec3> = indexed
        .vertices
        .iter()
        .map(|vertex| Vec3::new(vertex[0], vertex[1], vertex[2]))
        .collect();

    let mut indices = Vec::with_capacity(indexed.faces.len() * 3);
    for face in &indexed.faces {
        for vertex_index in face.vertices {
            indices
                .push(u32::try_from(vertex_index).map_err(|_| LoadMeshError::VertexCountOverflow)?);
        }
    }

    let normals = generate_vertex_normals(&positions, &indices);
    let vertices = positions
        .into_iter()
        .zip(normals)
        .map(|(position, normal)| Vertex {
            position,
            normal,
            uv: Vec2::ZERO,
        })
        .collect();

    Ok(Mesh::new(vertices, indices))
}

fn stl_io_error(error: io::Error) -> LoadMeshError {
    LoadMeshError::Stl {
        message: format!("{error}"),
    }
}

pub(crate) fn generate_vertex_normals(positions: &[Vec3], indices: &[u32]) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; positions.len()];

    for triangle in indices.chunks_exact(3) {
        let i0 = triangle[0] as usize;
        let i1 = triangle[1] as usize;
        let i2 = triangle[2] as usize;

        let p0 = positions[i0];
        let p1 = positions[i1];
        let p2 = positions[i2];
        let face_normal = (p1 - p0).cross(p2 - p0);

        if face_normal.length_squared() == 0.0 {
            continue;
        }

        normals[i0] += face_normal;
        normals[i1] += face_normal;
        normals[i2] += face_normal;
    }

    normals
        .into_iter()
        .map(|normal| normal.normalize_or_zero())
        .collect()
}

fn compute_bounds(vertices: &[Vertex]) -> Option<Bounds> {
    let mut vertices = vertices.iter();
    let first = vertices.next()?;
    let mut min = first.position;
    let mut max = first.position;

    for vertex in vertices {
        min = min.min(vertex.position);
        max = max.max(vertex.position);
    }

    Some(Bounds { min, max })
}

#[cfg(test)]
mod tests {
    use super::{Bounds, LoadMeshError, Mesh, Vertex, parse_obj, read_stl_into_mesh};
    use glam::{Vec2, Vec3};
    use std::io::Cursor;

    #[test]
    fn triangle_accessors_follow_index_buffer() {
        let mesh = Mesh::new(
            vec![
                Vertex {
                    position: Vec3::new(0.0, 0.0, 0.0),
                    normal: Vec3::X,
                    uv: Vec2::ZERO,
                },
                Vertex {
                    position: Vec3::new(1.0, 0.0, 0.0),
                    normal: Vec3::Y,
                    uv: Vec2::X,
                },
                Vertex {
                    position: Vec3::new(0.0, 1.0, 0.0),
                    normal: Vec3::Z,
                    uv: Vec2::Y,
                },
            ],
            vec![0, 1, 2],
        );

        assert_eq!(
            mesh.triangle_positions(0),
            [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ]
        );
        assert_eq!(mesh.triangle_normals(0), [Vec3::X, Vec3::Y, Vec3::Z]);
        assert_eq!(mesh.triangle_uvs(0), [Vec2::ZERO, Vec2::X, Vec2::Y]);
    }

    #[test]
    fn mesh_computes_bounds() {
        let mesh = Mesh::new(
            vec![
                Vertex {
                    position: Vec3::new(-2.0, 1.0, 3.0),
                    normal: Vec3::X,
                    uv: Vec2::ZERO,
                },
                Vertex {
                    position: Vec3::new(4.0, -1.0, -5.0),
                    normal: Vec3::Y,
                    uv: Vec2::ONE,
                },
            ],
            vec![0, 1, 1],
        );

        assert_eq!(
            mesh.bounds,
            Bounds {
                min: Vec3::new(-2.0, -1.0, -5.0),
                max: Vec3::new(4.0, 1.0, 3.0),
            }
        );
    }

    #[test]
    fn load_obj_triangulates_faces_and_generates_normals() {
        let mesh = parse_obj(
            "
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
f 1/1 2/2 3/3 4/4
",
        )
        .expect("OBJ should load");

        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.indices, vec![0, 1, 2, 0, 2, 3]);
        assert_eq!(mesh.triangle_count(), 2);
        assert!(mesh.vertices.iter().all(|vertex| vertex.normal == Vec3::Z));
        assert_eq!(mesh.vertices[2].uv, Vec2::X);
    }

    #[test]
    fn load_obj_resolves_negative_indices() {
        let mesh = parse_obj(
            "
v 0 0 0
v 1 0 0
v 0 1 0
f -3 -2 -1
",
        )
        .expect("OBJ should load");

        assert_eq!(mesh.vertices.len(), 3);
        assert_eq!(mesh.indices, vec![0, 1, 2]);
        assert_eq!(mesh.triangle_normals(0), [Vec3::Z; 3]);
    }

    #[test]
    fn load_stl_ascii_dedups_shared_vertices_and_generates_normals() {
        let source = b"solid quad
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 1 1 0
  endloop
endfacet
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 1 0
    vertex 0 1 0
  endloop
endfacet
endsolid quad
";
        let mesh =
            read_stl_into_mesh(&mut Cursor::new(&source[..])).expect("ASCII STL should load");

        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.triangle_count(), 2);
        assert!(
            mesh.vertices
                .iter()
                .all(|vertex| vertex.normal.abs_diff_eq(Vec3::Z, 1.0e-5))
        );
    }

    #[test]
    fn load_stl_binary_reads_one_triangle() {
        const HEADER_BYTES: usize = 80;
        let mut bytes = vec![0u8; HEADER_BYTES];
        bytes.extend_from_slice(&1u32.to_le_bytes());
        let triangle: [Vec3; 3] = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        for component in [0.0f32, 0.0, 1.0] {
            bytes.extend_from_slice(&component.to_le_bytes());
        }
        for vertex in &triangle {
            for component in [vertex.x, vertex.y, vertex.z] {
                bytes.extend_from_slice(&component.to_le_bytes());
            }
        }
        bytes.extend_from_slice(&0u16.to_le_bytes());

        let mesh = read_stl_into_mesh(&mut Cursor::new(bytes)).expect("binary STL should load");

        assert_eq!(mesh.vertices.len(), 3);
        assert_eq!(mesh.triangle_count(), 1);
        assert_eq!(mesh.triangle_normals(0), [Vec3::Z; 3]);
    }

    #[test]
    fn load_stl_ascii_rejects_unterminated_facet() {
        let source = b"solid bad
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
";
        match read_stl_into_mesh(&mut Cursor::new(&source[..])) {
            Err(LoadMeshError::Stl { .. }) | Err(LoadMeshError::EmptyMesh) => {}
            Err(other) => panic!("expected Stl/EmptyMesh error, got {other:?}"),
            Ok(_) => panic!("unterminated facet should not parse"),
        }
    }
}
