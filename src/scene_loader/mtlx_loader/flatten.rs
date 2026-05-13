use std::collections::HashMap;

use crate::color::srgb_to_linear;

use super::library::MtlxLibrary;
use super::types::{
    InputBinding, MtlxType, MtlxValue, RawInput, RawNodeGraph, RawNodeUse, parse_literal,
};

pub type FlatNodeId = u32;

#[derive(Debug, Clone)]
pub enum FlatInput {
    Value(MtlxValue),
    String(String),
    Node {
        node: FlatNodeId,
        output: Option<String>,
    },
    GeomProp(String),
    Empty,
}

#[derive(Debug, Clone)]
pub struct FlatNodeInput {
    pub name: String,
    pub ty: MtlxType,
    pub colorspace: Option<String>,
    pub unit: Option<String>,
    pub unittype: Option<String>,
    pub binding: FlatInput,
}

#[derive(Debug, Clone)]
pub enum FlatNodeKind {
    Pattern { category: String },
    Shading { category: String },
    SurfaceMaterial,
    Surface,
    SurfaceUnlit,
    Displacement,
    Light,
    Combinator { category: String },
    Geometric { kind: GeometricKind, index: i32 },
    Constant { value: MtlxValue },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeometricKind {
    Position,
    Normal,
    Tangent,
    Bitangent,
    Texcoord,
    Geomcolor,
    ViewDirection,
    Geompropvalue(String),
}

#[derive(Debug, Clone)]
pub struct FlatNode {
    pub kind: FlatNodeKind,
    pub output_type: MtlxType,
    pub inputs: Vec<FlatNodeInput>,
}

// Nodes are stored in topological order: every input reference has a
// smaller FlatNodeId than the node containing it.
#[derive(Debug, Clone)]
pub struct FlatGraph {
    pub nodes: Vec<FlatNode>,
    pub root: FlatNodeId,
    pub back_root: Option<FlatNodeId>,
    pub material_name: String,
}

pub const PRIMITIVE_CATEGORIES: &[&str] = &[
    "constant",
    "image",
    "tiledimage",
    "latlongimage",
    "hextiledimage",
    "texcoord",
    "position",
    "normal",
    "tangent",
    "bitangent",
    "geomcolor",
    "geompropvalue",
    "geompropvalueuniform",
    "bump",
    "frame",
    "time",
    "add",
    "subtract",
    "multiply",
    "divide",
    "modulo",
    "fract",
    "invert",
    "absval",
    "sign",
    "floor",
    "ceil",
    "round",
    "power",
    "safepower",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan2",
    "sqrt",
    "ln",
    "exp",
    "clamp",
    "trianglewave",
    "min",
    "max",
    "normalize",
    "magnitude",
    "distance",
    "dotproduct",
    "crossproduct",
    "mix",
    "smoothstep",
    "range",
    "remap",
    "contrast",
    "hsvadjust",
    "saturate",
    "colorcorrect",
    "luminance",
    "rgbtohsv",
    "hsvtorgb",
    "extract",
    "convert",
    "combine2",
    "combine3",
    "combine4",
    "and",
    "or",
    "xor",
    "not",
    "ifgreater",
    "ifgreatereq",
    "ifequal",
    "switch",
    "noise2d",
    "noise3d",
    "fractal2d",
    "fractal3d",
    "cellnoise2d",
    "cellnoise3d",
    "worleynoise2d",
    "worleynoise3d",
    "unifiednoise2d",
    "unifiednoise3d",
    "randomfloat",
    "randomcolor",
    "ramplr",
    "ramptb",
    "ramp4",
    "splitlr",
    "splittb",
    "checkerboard",
    "transformpoint",
    "transformvector",
    "transformnormal",
    "transformmatrix",
    "rotate2d",
    "rotate3d",
    "place2d",
    "reflect",
    "refract",
    "dot",
    "creatematrix",
    "transpose",
    "determinant",
    "invertmatrix",
    "normalmap",
    "heighttonormal",
    "hextilednormalmap",
    "premult",
    "unpremult",
    "plus",
    "minus",
    "difference",
    "burn",
    "dodge",
    "screen",
    "overlay",
    "disjointover",
    "in",
    "mask",
    "matte",
    "out",
    "over",
    "inside",
    "outside",
    "blackbody",
    "artistic_ior",
    "roughness_anisotropy",
    "roughness_dual",
    "glossiness_anisotropy",
    "viewdirection",
    "facingratio",
    "oren_nayar_diffuse_bsdf",
    "burley_diffuse_bsdf",
    "translucent_bsdf",
    "dielectric_bsdf",
    "conductor_bsdf",
    "generalized_schlick_bsdf",
    "sheen_bsdf",
    "subsurface_bsdf",
    "thin_film_bsdf",
    "chiang_hair_bsdf",
    "uniform_edf",
    "conical_edf",
    "measured_edf",
    "generalized_schlick_edf",
    "absorption_vdf",
    "anisotropic_vdf",
    "surface",
    "surface_unlit",
    "displacement",
    "light",
    "volume",
    "mix_bsdf",
    "layer",
    "add_bsdf",
    "multiply_bsdf",
];

#[derive(Debug)]
pub enum FlattenError {
    Missing { what: String },
    Cycle,
    Unsupported { what: String },
    Resolve(super::resolver::ResolveError),
}

impl std::fmt::Display for FlattenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing { what } => write!(f, "missing reference: {}", what),
            Self::Cycle => write!(f, "cycle detected during flatten"),
            Self::Unsupported { what } => write!(f, "unsupported feature: {}", what),
            Self::Resolve(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for FlattenError {}

impl From<super::resolver::ResolveError> for FlattenError {
    fn from(e: super::resolver::ResolveError) -> Self {
        Self::Resolve(e)
    }
}

pub fn flatten_material(
    library: &MtlxLibrary,
    document: &super::types::RawMtlxDocument,
    material_name: &str,
) -> Result<FlatGraph, FlattenError> {
    let mut nodegraphs: HashMap<String, &RawNodeGraph> = HashMap::new();
    for ng in &document.nodegraphs {
        nodegraphs.insert(ng.name.clone(), ng);
    }
    for ng in &library.nodegraphs {
        nodegraphs.entry(ng.graph.name.clone()).or_insert(&ng.graph);
    }

    let mut document_root_nodes: HashMap<String, &RawNodeUse> = HashMap::new();
    for n in &document.root_nodes {
        document_root_nodes.insert(n.name.clone(), n);
    }

    let mut document_nodedefs: HashMap<String, &super::types::RawNodeDef> = HashMap::new();
    let mut document_nodedefs_by_category: HashMap<String, Vec<&super::types::RawNodeDef>> =
        HashMap::new();
    for nd in &document.nodedefs {
        document_nodedefs.insert(nd.name.clone(), nd);
        document_nodedefs_by_category
            .entry(nd.node.clone())
            .or_default()
            .push(nd);
    }

    let material = document
        .materials
        .iter()
        .find(|m| m.name == material_name)
        .ok_or_else(|| FlattenError::Missing {
            what: format!("material `{}`", material_name),
        })?;

    if material.category == "volumematerial" {
        eprintln!(
            "warning: material `{}` is `volumematerial`; volume rendering is not implemented, treating it as passthrough",
            material_name
        );
        return Ok(empty_surface_graph(material_name));
    }

    if material.category == "lightmaterial" {
        return Err(FlattenError::Missing {
            what: format!(
                "material `{}` is `{}`, which is not supported by this renderer (only surfacematerial)",
                material_name, material.category
            ),
        });
    }

    let mut builder = Builder {
        library,
        nodegraphs: &nodegraphs,
        document_root_nodes: &document_root_nodes,
        document_nodedefs: &document_nodedefs,
        document_nodedefs_by_category: &document_nodedefs_by_category,
        nodes: Vec::new(),
        cache: HashMap::new(),
        stack: Vec::new(),
    };

    let mut material_scope = Scope::root();
    if let Some(parent_ng_name) = &material.parent_nodegraph
        && let Some(parent_ng) = nodegraphs.get(parent_ng_name).copied()
    {
        material_scope.id = ScopeId::NodeGraph(parent_ng_name.clone(), 0);
        for n in &parent_ng.nodes {
            material_scope.nodes.insert(n.name.clone(), n);
        }
        for o in &parent_ng.outputs {
            material_scope.outputs.insert(o.name.clone(), o);
        }
        apply_token_declarations(&mut material_scope.tokens, &parent_ng.tokens)?;
    }

    let surfaceshader = material.inputs.iter().find(|i| i.name == "surfaceshader");
    let surface_root = if let Some(surfaceshader) = surfaceshader {
        builder.materialize_input(
            &material_scope,
            surfaceshader,
            &MtlxType::Surfaceshader,
            None,
        )?
    } else {
        FlatInput::Empty
    };

    if let Some(input) = material
        .inputs
        .iter()
        .find(|i| i.name == "displacementshader")
        && !is_empty_shader_binding(&input.binding)
    {
        eprintln!(
            "warning: surfacematerial.displacementshader is not implemented; geometry displacement will be ignored"
        );
    }

    let backsurfaceshader = material
        .inputs
        .iter()
        .find(|i| i.name == "backsurfaceshader");
    let back_surface_root = if let Some(input) = backsurfaceshader {
        let has_binding = !is_empty_shader_binding(&input.binding);
        if has_binding {
            Some(builder.materialize_input(
                &material_scope,
                input,
                &MtlxType::Surfaceshader,
                None,
            )?)
        } else {
            None
        }
    } else {
        None
    };

    let mut nodes = builder.nodes;
    let root_idx = nodes.len() as FlatNodeId;
    let mut inputs = vec![FlatNodeInput {
        name: "surfaceshader".into(),
        ty: MtlxType::Surfaceshader,
        colorspace: None,
        unit: None,
        unittype: None,
        binding: surface_root,
    }];
    let mut back_root_idx = None;
    if let Some(back_binding) = back_surface_root {
        inputs.push(FlatNodeInput {
            name: "backsurfaceshader".into(),
            ty: MtlxType::Surfaceshader,
            colorspace: None,
            unit: None,
            unittype: None,
            binding: back_binding.clone(),
        });
        if let FlatInput::Node { node, .. } = back_binding {
            back_root_idx = Some(node);
        }
    }
    nodes.push(FlatNode {
        kind: FlatNodeKind::SurfaceMaterial,
        output_type: MtlxType::Material,
        inputs,
    });

    apply_unit_conversions(&mut nodes)?;

    Ok(FlatGraph {
        nodes,
        root: root_idx,
        back_root: back_root_idx,
        material_name: material_name.to_string(),
    })
}

fn empty_surface_graph(material_name: &str) -> FlatGraph {
    FlatGraph {
        nodes: vec![FlatNode {
            kind: FlatNodeKind::SurfaceMaterial,
            output_type: MtlxType::Material,
            inputs: vec![FlatNodeInput {
                name: "surfaceshader".into(),
                ty: MtlxType::Surfaceshader,
                colorspace: None,
                unit: None,
                unittype: None,
                binding: FlatInput::Empty,
            }],
        }],
        root: 0,
        back_root: None,
        material_name: material_name.to_string(),
    }
}

fn is_empty_shader_binding(binding: &super::types::InputBinding) -> bool {
    match binding {
        super::types::InputBinding::Empty => true,
        super::types::InputBinding::Value(text) => text.trim().is_empty(),
        _ => false,
    }
}

fn unit_scale_to_base(unittype: &str, unit: &str) -> Option<f32> {
    match unittype {
        "distance" => match unit {
            "nanometer" => Some(1e-9),
            "micron" => Some(1e-6),
            "millimeter" => Some(1e-3),
            "centimeter" => Some(1e-2),
            "inch" => Some(0.0254),
            "foot" => Some(0.3048),
            "yard" => Some(0.9144),
            "meter" => Some(1.0),
            "kilometer" => Some(1000.0),
            "mile" => Some(1609.34),
            _ => None,
        },
        "angle" => match unit {
            "degree" => Some(1.0),
            "radian" => Some(57.295_78),
            _ => None,
        },
        _ => None,
    }
}

fn scale_mtlx_value(value: &MtlxValue, scale: f32) -> Option<MtlxValue> {
    use glam::{Vec2, Vec3, Vec4};
    Some(match value {
        MtlxValue::Float(v) => MtlxValue::Float(v * scale),
        MtlxValue::Vector2(v) => MtlxValue::Vector2(Vec2::new(v.x * scale, v.y * scale)),
        MtlxValue::Vector3(v) => {
            MtlxValue::Vector3(Vec3::new(v.x * scale, v.y * scale, v.z * scale))
        }
        MtlxValue::Vector4(v) => MtlxValue::Vector4(Vec4::new(
            v.x * scale,
            v.y * scale,
            v.z * scale,
            v.w * scale,
        )),
        MtlxValue::FloatArray(arr) => {
            MtlxValue::FloatArray(arr.iter().map(|v| v * scale).collect())
        }
        MtlxValue::Vector2Array(arr) => MtlxValue::Vector2Array(
            arr.iter()
                .map(|v| Vec2::new(v.x * scale, v.y * scale))
                .collect(),
        ),
        MtlxValue::Vector3Array(arr) => MtlxValue::Vector3Array(
            arr.iter()
                .map(|v| Vec3::new(v.x * scale, v.y * scale, v.z * scale))
                .collect(),
        ),
        MtlxValue::Vector4Array(arr) => MtlxValue::Vector4Array(
            arr.iter()
                .map(|v| Vec4::new(v.x * scale, v.y * scale, v.z * scale, v.w * scale))
                .collect(),
        ),
        _ => return None,
    })
}

fn apply_unit_conversions(nodes: &mut [FlatNode]) -> Result<(), FlattenError> {
    for node in nodes.iter_mut() {
        for input in node.inputs.iter_mut() {
            let (Some(unit), Some(unittype)) = (input.unit.as_deref(), input.unittype.as_deref())
            else {
                continue;
            };
            let scale =
                unit_scale_to_base(unittype, unit).ok_or_else(|| FlattenError::Unsupported {
                    what: format!("unknown unit `{}` for unittype `{}`", unit, unittype),
                })?;
            if (scale - 1.0).abs() < f32::EPSILON {
                continue;
            }
            if let FlatInput::Value(v) = &input.binding {
                let scaled =
                    scale_mtlx_value(v, scale).ok_or_else(|| FlattenError::Unsupported {
                        what: format!(
                            "unit `{}` (unittype `{}`) cannot be applied to value type {:?}",
                            unit, unittype, v
                        ),
                    })?;
                input.binding = FlatInput::Value(scaled);
            }
        }
    }
    Ok(())
}

struct Builder<'a> {
    library: &'a MtlxLibrary,
    nodegraphs: &'a HashMap<String, &'a RawNodeGraph>,
    document_root_nodes: &'a HashMap<String, &'a RawNodeUse>,
    document_nodedefs: &'a HashMap<String, &'a super::types::RawNodeDef>,
    document_nodedefs_by_category: &'a HashMap<String, Vec<&'a super::types::RawNodeDef>>,
    nodes: Vec<FlatNode>,
    cache: HashMap<NodeKey, FlatNodeId>,
    stack: Vec<NodeKey>,
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
struct NodeKey {
    scope_id: ScopeId,
    node_name: String,
    output: Option<String>,
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
enum ScopeId {
    Root,
    NodeGraph(String, u32),
}

#[derive(Clone)]
struct Scope<'a> {
    id: ScopeId,
    interface: HashMap<String, FlatInput>,
    nodes: HashMap<String, &'a RawNodeUse>,
    outputs: HashMap<String, &'a super::types::RawOutput>,
    /// Tokens visible in this scope (from <token> elements in the surrounding
    /// nodegraph/nodedef). Used for `[token_name]` filename substitution.
    tokens: HashMap<String, String>,
}

impl<'a> Scope<'a> {
    fn root() -> Self {
        Self {
            id: ScopeId::Root,
            interface: HashMap::new(),
            nodes: HashMap::new(),
            outputs: HashMap::new(),
            tokens: HashMap::new(),
        }
    }
}

fn substitute_filename_tokens(value: &str, tokens: &HashMap<String, String>) -> String {
    if !value.contains('[') && !value.contains('{') && !value.contains('<') {
        return value.to_string();
    }
    let mut out = String::with_capacity(value.len());
    let mut i = 0;
    let bytes = value.as_bytes();
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'[' {
            if let Some(end) = value[i..].find(']') {
                let name = &value[i + 1..i + end];
                if let Some(v) = tokens.get(name) {
                    out.push_str(v);
                    i += end + 1;
                    continue;
                }
            }
        } else if b == b'{'
            && let Some(end) = value[i..].find('}')
        {
            let token = &value[i + 1..i + end];
            if token == "frame" {
                eprintln!(
                    "[mtlx] warning: animated image token `{{frame}}` is not supported yet; using frame 0"
                );
                out.push('0');
                i += end + 1;
                continue;
            }
            if let Some(stripped) = token.strip_suffix("frame")
                && let Some(width_str) = stripped.strip_prefix('0')
                && let Ok(width) = width_str.parse::<usize>()
            {
                eprintln!(
                    "[mtlx] warning: animated image token `{{{}frame}}` is not supported yet; using frame 0",
                    stripped
                );
                out.push_str(&format!("{:0width$}", 0, width = width));
                i += end + 1;
                continue;
            }
        } else if b == b'<'
            && let Some(end) = value[i..].find('>')
        {
            let token = &value[i + 1..i + end];
            if token != "UDIM" && token != "UVTILE" {
                eprintln!(
                    "[mtlx] warning: geometry filename token `<{}>` is not supported yet; leaving token unresolved",
                    token
                );
            }
        }
        out.push(b as char);
        i += 1;
    }
    out
}

fn apply_token_declarations(
    tokens: &mut HashMap<String, String>,
    declarations: &[super::types::RawToken],
) -> Result<(), FlattenError> {
    for token in declarations {
        let value = token_to_string(token, tokens)?;
        tokens.insert(token.name.clone(), value);
    }
    Ok(())
}

fn apply_nodedef_tokens(
    tokens: &mut HashMap<String, String>,
    declarations: &[super::types::RawToken],
    overrides: &[super::types::RawToken],
) -> Result<(), FlattenError> {
    for override_token in overrides {
        if !declarations.iter().any(|t| t.name == override_token.name) {
            return Err(FlattenError::Missing {
                what: format!("token `{}` on nodedef interface", override_token.name),
            });
        }
    }
    for token in declarations {
        let chosen = overrides
            .iter()
            .find(|override_token| override_token.name == token.name)
            .unwrap_or(token);
        let value = token_to_string(chosen, tokens)?;
        tokens.insert(token.name.clone(), value);
    }
    Ok(())
}

fn token_to_string(
    token: &super::types::RawToken,
    tokens: &HashMap<String, String>,
) -> Result<String, FlattenError> {
    if let Some(interface) = &token.interface {
        return tokens
            .get(interface)
            .cloned()
            .ok_or_else(|| FlattenError::Missing {
                what: format!("interface token `{}` for token `{}`", interface, token.name),
            });
    }
    let Some(value) = &token.value else {
        return Err(FlattenError::Missing {
            what: format!("required token `{}` of type {:?}", token.name, token.ty),
        });
    };
    if matches!(
        token.ty,
        MtlxType::String | MtlxType::Filename | MtlxType::Geomname
    ) || parse_literal(&token.ty, value).is_some()
    {
        Ok(value.clone())
    } else {
        Err(FlattenError::Unsupported {
            what: format!(
                "invalid token value `{}` for `{}` of type {:?}",
                value, token.name, token.ty
            ),
        })
    }
}

impl<'a> Builder<'a> {
    fn materialize_input(
        &mut self,
        scope: &Scope<'a>,
        input: &RawInput,
        expected_type: &MtlxType,
        default: Option<&MtlxValue>,
    ) -> Result<FlatInput, FlattenError> {
        match &input.binding {
            InputBinding::Empty => {
                if let Some(v) = default {
                    Ok(FlatInput::Value(v.clone()))
                } else if expected_type.is_shader_like() {
                    Ok(FlatInput::Empty)
                } else if let Some(v) = zero_value(expected_type) {
                    Ok(FlatInput::Value(v))
                } else {
                    Err(FlattenError::Unsupported {
                        what: format!(
                            "missing value for unsupported input type {:?}",
                            expected_type
                        ),
                    })
                }
            }
            InputBinding::Value(text) => {
                if expected_type.is_shader_like() && text.trim().is_empty() {
                    return Ok(FlatInput::Empty);
                }
                // Filename / string-typed inputs get token substitution so
                // `[interface_token]` and `{frame}` resolve before the texture
                // loader sees the path. (<UDIM>/<UVTILE> would need per-pixel
                // resolution and are left intact for downstream warnings.)
                let needs_subst = matches!(
                    expected_type,
                    MtlxType::Filename | MtlxType::String | MtlxType::Geomname
                );
                let owned;
                let text_slice: &str = if needs_subst {
                    owned = substitute_filename_tokens(text, &scope.tokens);
                    // `<UDIM>`/`<UVTILE>` markers are preserved here on purpose:
                    // the texture loader expands them to a tile set, and the
                    // runtime resolves the tile per shading point from the UV.
                    owned.as_str()
                } else {
                    text.as_str()
                };
                let parsed = parse_literal(expected_type, text_slice);
                if let Some(v) = parsed {
                    Ok(FlatInput::Value(apply_input_color_space(
                        v,
                        expected_type,
                        input.colorspace.as_deref(),
                        &input.name,
                    )))
                } else if matches!(
                    expected_type,
                    MtlxType::Filename | MtlxType::String | MtlxType::Geomname
                ) {
                    Ok(FlatInput::String(text_slice.to_string()))
                } else {
                    Err(FlattenError::Unsupported {
                        what: format!(
                            "invalid literal `{}` for input `{}` of type {:?}",
                            text_slice, input.name, expected_type
                        ),
                    })
                }
            }
            InputBinding::NodeRef { nodename, output } => {
                let node = scope
                    .nodes
                    .get(nodename.as_str())
                    .copied()
                    .or_else(|| self.document_root_nodes.get(nodename.as_str()).copied())
                    .ok_or_else(|| FlattenError::Missing {
                        what: format!("node `{}` referenced in scope", nodename),
                    })?;
                let normalized_output =
                    self.normalize_node_output(node, output.as_deref(), expected_type)?;
                let is_primitive = PRIMITIVE_CATEGORIES.contains(&node.category.as_str());
                let id =
                    self.materialize_node_for_output(scope, node, normalized_output.as_deref())?;
                Ok(FlatInput::Node {
                    node: id,
                    output: is_primitive.then_some(normalized_output).flatten(),
                })
            }
            InputBinding::NodeGraphRef { nodegraph, output } => {
                let ng = self
                    .nodegraphs
                    .get(nodegraph.as_str())
                    .copied()
                    .ok_or_else(|| FlattenError::Missing {
                        what: format!("nodegraph `{}`", nodegraph),
                    })?;
                let out_name =
                    select_nodegraph_ref_output(ng, output.as_deref(), expected_type)?.to_string();
                self.materialize_nodegraph_output(ng, &out_name, scope, expected_type)
            }
            InputBinding::InterfaceName(name) => {
                if let Some(binding) = scope.interface.get(name) {
                    Ok(binding.clone())
                } else if let Some(default) = default {
                    Ok(FlatInput::Value(default.clone()))
                } else {
                    Err(FlattenError::Missing {
                        what: format!("interface `{}` for input `{}`", name, input.name),
                    })
                }
            }
            InputBinding::DefaultGeomProp(prop) => {
                let kind = geometric_kind(prop);
                let inputs = match implied_geom_space(prop) {
                    Some(space) => vec![FlatNodeInput {
                        name: "space".to_string(),
                        ty: MtlxType::String,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::String(space.to_string())),
                    }],
                    None => vec![],
                };
                let id = self.add_node(FlatNode {
                    kind: FlatNodeKind::Geometric {
                        kind: kind.clone(),
                        index: 0,
                    },
                    output_type: geometric_output_type(&kind),
                    inputs,
                });
                Ok(FlatInput::Node {
                    node: id,
                    output: None,
                })
            }
        }
    }

    fn normalize_node_output(
        &self,
        node: &RawNodeUse,
        requested: Option<&str>,
        expected_type: &MtlxType,
    ) -> Result<Option<String>, FlattenError> {
        let Some(outputs) = self.nodedef_outputs_for_node(node)? else {
            if requested.is_some() {
                return Err(FlattenError::Missing {
                    what: format!("output declaration for node `{}`", node.name),
                });
            }
            return Ok(None);
        };
        if outputs.is_empty() {
            return Err(FlattenError::Missing {
                what: format!("output declaration for node `{}`", node.name),
            });
        }
        if outputs.len() == 1 {
            let out = &outputs[0];
            if !output_type_compatible(&out.ty, expected_type) {
                return Err(FlattenError::Unsupported {
                    what: format!(
                        "output `{}` on node `{}` has type {:?}, expected {:?}",
                        out.name, node.name, out.ty, expected_type
                    ),
                });
            }
            return Ok(None);
        }
        let Some(name) = requested else {
            return Err(FlattenError::Missing {
                what: format!("output name for multi-output node `{}`", node.name),
            });
        };
        let out = outputs
            .iter()
            .find(|o| o.name == name)
            .ok_or_else(|| FlattenError::Missing {
                what: format!("output `{}` on node `{}`", name, node.name),
            })?;
        if !output_type_compatible(&out.ty, expected_type) {
            return Err(FlattenError::Unsupported {
                what: format!(
                    "output `{}` on node `{}` has type {:?}, expected {:?}",
                    name, node.name, out.ty, expected_type
                ),
            });
        }
        Ok(Some(name.to_string()))
    }

    fn nodedef_outputs_for_node(
        &self,
        node: &RawNodeUse,
    ) -> Result<Option<&'a [super::types::RawOutput]>, FlattenError> {
        if let Some(local) = self.find_local_nodedef(node) {
            return Ok(Some(&local.outputs));
        }
        let collected = node
            .inputs
            .iter()
            .filter(|i| i.name != "disable")
            .map(|i| (i.name.clone(), i.ty.clone()))
            .collect::<Vec<_>>();
        let mut resolve_node = node.clone();
        resolve_node.ty = node_resolve_output_type(node);
        match super::resolver::resolve_node_use(self.library, &resolve_node, &collected) {
            Ok(def) => Ok(Some(&def.def.outputs)),
            Err(e) => {
                if PRIMITIVE_CATEGORIES.contains(&node.category.as_str())
                    || !self
                        .library
                        .nodedefs_for_category(&node.category)
                        .is_empty()
                {
                    Err(FlattenError::Resolve(e))
                } else {
                    Ok(None)
                }
            }
        }
    }

    fn materialize_node_for_output(
        &mut self,
        scope: &Scope<'a>,
        node: &'a RawNodeUse,
        output: Option<&str>,
    ) -> Result<FlatNodeId, FlattenError> {
        let key = NodeKey {
            scope_id: scope.id.clone(),
            node_name: node.name.clone(),
            output: output.map(str::to_string),
        };
        if let Some(id) = self.cache.get(&key) {
            return Ok(*id);
        }
        if self.stack.contains(&key) {
            return Err(FlattenError::Cycle);
        }
        self.stack.push(key.clone());

        // Specification §"disable" input: a node with `disable="true"` must
        // short-circuit to its `defaultinput` named input (or `default` value
        // on the nodedef output) instead of running the actual implementation.
        if Self::is_node_disabled(node)
            && let Some(id) = self.materialize_disabled_passthrough(scope, node, output)?
        {
            self.stack.pop();
            self.cache.insert(key, id);
            return Ok(id);
        }

        let is_primitive = PRIMITIVE_CATEGORIES.contains(&node.category.as_str());

        let result = if is_primitive {
            self.materialize_primitive(scope, node, output)
        } else {
            self.materialize_via_nodegraph(scope, node, output)
        };

        self.stack.pop();
        let id = result?;
        self.cache.insert(key, id);
        Ok(id)
    }

    fn is_node_disabled(node: &RawNodeUse) -> bool {
        node.inputs.iter().any(|i| {
            i.name == "disable"
                && match &i.binding {
                    InputBinding::Value(s) => {
                        let s = s.trim();
                        s.eq_ignore_ascii_case("true") || s == "1"
                    }
                    _ => false,
                }
        })
    }

    fn materialize_disabled_passthrough(
        &mut self,
        scope: &Scope<'a>,
        node: &'a RawNodeUse,
        output: Option<&str>,
    ) -> Result<Option<FlatNodeId>, FlattenError> {
        // Locate the nodedef so we can read its output's `defaultinput`/`default`.
        let nodedef = if let Some(nd) = self.find_local_nodedef(node) {
            Some((&nd.inputs[..], &nd.outputs[..]))
        } else {
            let collected: Vec<_> = node
                .inputs
                .iter()
                .filter(|i| i.name != "disable")
                .map(|i| (i.name.clone(), i.ty.clone()))
                .collect();
            let mut resolve_node = node.clone();
            resolve_node.ty = node_resolve_output_type(node);
            let d = super::resolver::resolve_node_use(self.library, &resolve_node, &collected)?;
            Some((&d.def.inputs[..], &d.def.outputs[..]))
        };
        let Some((def_inputs, def_outputs)) = nodedef else {
            return Ok(None);
        };
        let out = select_nodedef_output(node, def_outputs, output)?;
        if let Some(out) = out {
            let out_ty = out.ty.clone();
            if let Some(default_input_name) = &out.default_input {
                // Use the use_node's binding for the named input if present;
                // otherwise materialize the nodedef default literal.
                if let Some(raw) = node.inputs.iter().find(|i| &i.name == default_input_name) {
                    let decl = def_inputs
                        .iter()
                        .find(|i| &i.name == default_input_name)
                        .ok_or_else(|| FlattenError::Missing {
                            what: format!(
                                "defaultinput `{}` on nodedef for node `{}`",
                                default_input_name, node.name
                            ),
                        })?;
                    let expected = decl.ty.clone();
                    let bound = self.materialize_input(scope, raw, &expected, None)?;
                    return Ok(Some(self.flat_input_to_node_id(bound, &out_ty)?));
                }
                if let Some(decl) = def_inputs.iter().find(|i| &i.name == default_input_name)
                    && let InputBinding::Value(text) = &decl.binding
                {
                    let v =
                        parse_literal(&decl.ty, text).ok_or_else(|| FlattenError::Unsupported {
                            what: format!(
                                "invalid defaultinput literal `{}` for `{}` of type {:?}",
                                text, decl.name, decl.ty
                            ),
                        })?;
                    return Ok(Some(self.flat_input_to_node_id(
                        FlatInput::Value(apply_input_color_space(
                            v,
                            &decl.ty,
                            decl.colorspace.as_deref(),
                            &decl.name,
                        )),
                        &out_ty,
                    )?));
                }
            }
            if let Some(default_text) = &out.default {
                let v = parse_literal(&out.ty, default_text).ok_or_else(|| {
                    FlattenError::Unsupported {
                        what: format!(
                            "invalid output default `{}` for `{}` of type {:?}",
                            default_text, out.name, out.ty
                        ),
                    }
                })?;
                return Ok(Some(
                    self.flat_input_to_node_id(FlatInput::Value(v), &out_ty)?,
                ));
            }
        }
        Ok(None)
    }

    fn materialize_primitive(
        &mut self,
        scope: &Scope<'a>,
        node: &'a RawNodeUse,
        output: Option<&str>,
    ) -> Result<FlatNodeId, FlattenError> {
        let output_type = match self.nodedef_outputs_for_node(node)? {
            Some(outputs) => select_nodedef_output(node, outputs, output)?
                .map(|out| out.ty.clone())
                .unwrap_or_else(|| node.ty.clone()),
            None => node.ty.clone(),
        };
        let mut inputs = Vec::with_capacity(node.inputs.len());
        for raw in &node.inputs {
            let bound = self.materialize_input(scope, raw, &raw.ty, None)?;
            inputs.push(FlatNodeInput {
                name: raw.name.clone(),
                ty: raw.ty.clone(),
                colorspace: raw.colorspace.clone(),
                unit: raw.unit.clone(),
                unittype: raw.unittype.clone(),
                binding: bound,
            });
        }
        let kind = if is_combinator(&node.category) {
            FlatNodeKind::Combinator {
                category: node.category.clone(),
            }
        } else if is_shading_category(&node.category) {
            FlatNodeKind::Shading {
                category: node.category.clone(),
            }
        } else if is_geometric_category(&node.category) {
            let kind = geometric_kind(&node.category);
            let index = node
                .inputs
                .iter()
                .find(|i| i.name == "index")
                .map(|i| match &i.binding {
                    InputBinding::Value(v) => {
                        v.parse::<i32>().map_err(|_| FlattenError::Unsupported {
                            what: format!(
                                "invalid geometric index literal `{}` on node `{}`",
                                v, node.name
                            ),
                        })
                    }
                    _ => Err(FlattenError::Unsupported {
                        what: format!("geometric index on node `{}` must be a literal", node.name),
                    }),
                })
                .transpose()?
                .unwrap_or(0);
            FlatNodeKind::Geometric { kind, index }
        } else if node.category == "surface" {
            FlatNodeKind::Surface
        } else if node.category == "surface_unlit" {
            FlatNodeKind::SurfaceUnlit
        } else if node.category == "displacement" {
            FlatNodeKind::Displacement
        } else if node.category == "light" {
            FlatNodeKind::Light
        } else {
            FlatNodeKind::Pattern {
                category: node.category.clone(),
            }
        };
        Ok(self.add_node(FlatNode {
            kind,
            output_type,
            inputs,
        }))
    }

    fn find_local_nodedef(&self, node: &RawNodeUse) -> Option<&'a super::types::RawNodeDef> {
        if let Some(name) = &node.nodedef {
            return self.document_nodedefs.get(name).copied().filter(|nd| {
                nd.node == node.category
                    && nd.target.is_none()
                    && local_nodedef_matches(
                        nd,
                        &node
                            .inputs
                            .iter()
                            .filter(|i| i.name != "disable")
                            .map(|i| (i.name.clone(), i.ty.clone()))
                            .collect::<Vec<_>>(),
                        &node_resolve_output_type(node),
                    )
            });
        }
        let candidates = self.document_nodedefs_by_category.get(&node.category)?;
        let inputs = node
            .inputs
            .iter()
            .filter(|i| i.name != "disable")
            .map(|i| (i.name.clone(), i.ty.clone()))
            .collect::<Vec<_>>();
        let mut filtered = candidates
            .iter()
            .copied()
            .filter(|nd| nd.target.is_none())
            .filter(|nd| local_nodedef_matches(nd, &inputs, &node_resolve_output_type(node)))
            .collect::<Vec<_>>();
        if filtered.is_empty() {
            return None;
        }
        if let Some(version) = node.version.as_deref() {
            return filtered
                .into_iter()
                .find(|nd| version_matches(nd.version.as_deref(), version));
        }
        if let Some(nd) = filtered.iter().find(|nd| nd.is_default_version) {
            return Some(*nd);
        }
        filtered.sort_by(|a, b| a.name.cmp(&b.name));
        Some(filtered[0])
    }

    fn find_local_nodegraph_for_nodedef(&self, nd_name: &str) -> Option<&'a RawNodeGraph> {
        for ng in self.nodegraphs.values() {
            if ng.nodedef.as_deref() == Some(nd_name) && ng.target.is_none() {
                return Some(*ng);
            }
        }
        None
    }

    fn materialize_via_nodegraph(
        &mut self,
        scope: &Scope<'a>,
        node: &'a RawNodeUse,
        output: Option<&str>,
    ) -> Result<FlatNodeId, FlattenError> {
        let collected = node
            .inputs
            .iter()
            .filter(|i| i.name != "disable")
            .map(|i| (i.name.clone(), i.ty.clone()))
            .collect::<Vec<_>>();
        let local_def = self.find_local_nodedef(node);
        let def_inputs: &[super::types::RawInput];
        let def_tokens: &[super::types::RawToken];
        let def_name: String;
        let nd_outputs: &[super::types::RawOutput];
        let ng_ref: &RawNodeGraph;
        match local_def {
            Some(local) => {
                def_inputs = &local.inputs;
                def_tokens = &local.tokens;
                def_name = local.name.clone();
                nd_outputs = &local.outputs;
                let ng = self
                    .find_local_nodegraph_for_nodedef(&local.name)
                    .ok_or_else(|| FlattenError::Missing {
                        what: format!(
                            "nodegraph implementing local nodedef `{}` (category `{}`)",
                            local.name, node.category
                        ),
                    })?;
                ng_ref = ng;
            }
            None => {
                if let Some(name) = &node.nodedef
                    && let Some(local) = self.document_nodedefs.get(name)
                {
                    return Err(FlattenError::Unsupported {
                        what: format!(
                            "local nodedef `{}` declares node `{}` but does not match node `{}` category `{}` or signature",
                            name, local.node, node.name, node.category
                        ),
                    });
                }
                let mut resolve_node = node.clone();
                resolve_node.ty = node_resolve_output_type(node);
                let def_result =
                    super::resolver::resolve_node_use(self.library, &resolve_node, &collected);
                let def = match def_result {
                    Ok(def) => def,
                    Err(e) => return Err(FlattenError::Resolve(e)),
                };
                def_inputs = &def.def.inputs;
                def_tokens = &def.def.tokens;
                def_name = def.def.name.clone();
                nd_outputs = &def.def.outputs;
                let ng = match self.library.nodegraph_for_nodedef(&def.def.name) {
                    Some(ng) => &ng.graph,
                    None => {
                        return Err(FlattenError::Missing {
                            what: format!(
                                "nodegraph implementation for nodedef `{}` (category `{}`)",
                                def.def.name, node.category
                            ),
                        });
                    }
                };
                ng_ref = ng;
            }
        }
        let out = select_nodegraph_output(node, nd_outputs, ng_ref, output)?;
        if !ng_ref.inputs.is_empty() {
            return Err(FlattenError::Unsupported {
                what: format!(
                    "functional nodegraph `{}` implementing node `{}` declares child inputs",
                    ng_ref.name, node.name
                ),
            });
        }
        let mut interface = HashMap::new();
        for raw in &node.inputs {
            if raw.name == "disable" {
                continue;
            }
            let decl = def_inputs
                .iter()
                .find(|d| d.name == raw.name)
                .ok_or_else(|| FlattenError::Missing {
                    what: format!(
                        "input `{}` on nodedef `{}` for node `{}`",
                        raw.name, def_name, node.name
                    ),
                })?;
            let expected = decl.ty.clone();
            let bound = self.materialize_input(scope, raw, &expected, None)?;
            interface.insert(raw.name.clone(), bound);
        }
        for nd_input in def_inputs {
            if !interface.contains_key(&nd_input.name) {
                let default = match &nd_input.binding {
                    InputBinding::Empty => {
                        return Err(FlattenError::Missing {
                            what: format!(
                                "required nodedef input `{}` of type {:?}",
                                nd_input.name, nd_input.ty
                            ),
                        });
                    }
                    InputBinding::Value(text) => {
                        if nd_input.ty.is_shader_like() && text.trim().is_empty() {
                            FlatInput::Empty
                        } else {
                            let value = parse_literal(&nd_input.ty, text).ok_or_else(|| {
                                FlattenError::Unsupported {
                                    what: format!(
                                        "invalid nodedef default literal `{}` for `{}` of type {:?}",
                                        text, nd_input.name, nd_input.ty
                                    ),
                                }
                            })?;
                            FlatInput::Value(apply_input_color_space(
                                value,
                                &nd_input.ty,
                                nd_input.colorspace.as_deref(),
                                &nd_input.name,
                            ))
                        }
                    }
                    InputBinding::DefaultGeomProp(prop) => {
                        let kind = geometric_kind(prop);
                        let inputs = match implied_geom_space(prop) {
                            Some(space) => vec![FlatNodeInput {
                                name: "space".to_string(),
                                ty: MtlxType::String,
                                colorspace: None,
                                unit: None,
                                unittype: None,
                                binding: FlatInput::Value(MtlxValue::String(space.to_string())),
                            }],
                            None => vec![],
                        };
                        let id = self.add_node(FlatNode {
                            kind: FlatNodeKind::Geometric {
                                kind: kind.clone(),
                                index: 0,
                            },
                            output_type: geometric_output_type(&kind),
                            inputs,
                        });
                        FlatInput::Node {
                            node: id,
                            output: None,
                        }
                    }
                    _ => {
                        return Err(FlattenError::Unsupported {
                            what: format!(
                                "unsupported default binding for nodedef input `{}`",
                                nd_input.name
                            ),
                        });
                    }
                };
                interface.insert(nd_input.name.clone(), default);
            }
        }

        let scope_id = ScopeId::NodeGraph(ng_ref.name.clone(), self.nodes.len() as u32);
        let mut nodes = HashMap::new();
        for n in &ng_ref.nodes {
            nodes.insert(n.name.clone(), n);
        }
        let mut outputs = HashMap::new();
        for o in &ng_ref.outputs {
            outputs.insert(o.name.clone(), o);
        }
        let mut tokens = scope.tokens.clone();
        apply_nodedef_tokens(&mut tokens, def_tokens, &node.tokens)?;
        apply_token_declarations(&mut tokens, &ng_ref.tokens)?;
        let inner_scope = Scope {
            id: scope_id,
            interface,
            nodes,
            outputs,
            tokens,
        };
        let flat_input = self.materialize_output(&inner_scope, out)?;
        self.flat_input_to_node_id(flat_input, &out.ty)
    }

    fn flat_input_to_node_id(
        &mut self,
        input: FlatInput,
        ty: &MtlxType,
    ) -> Result<FlatNodeId, FlattenError> {
        Ok(match input {
            FlatInput::Node { node, .. } => node,
            FlatInput::Value(v) => self.add_node(FlatNode {
                kind: FlatNodeKind::Constant { value: v },
                output_type: ty.clone(),
                inputs: vec![],
            }),
            FlatInput::String(s) => self.add_node(FlatNode {
                kind: FlatNodeKind::Constant {
                    value: MtlxValue::String(s),
                },
                output_type: ty.clone(),
                inputs: vec![],
            }),
            FlatInput::GeomProp(prop) => {
                let kind = geometric_kind(&prop);
                let out_ty = geometric_output_type(&kind);
                self.add_node(FlatNode {
                    kind: FlatNodeKind::Geometric { kind, index: 0 },
                    output_type: out_ty,
                    inputs: vec![],
                })
            }
            FlatInput::Empty => {
                return Err(FlattenError::Missing {
                    what: format!("resolved value for output of type {:?}", ty),
                });
            }
        })
    }

    fn materialize_nodegraph_output(
        &mut self,
        ng: &'a RawNodeGraph,
        out_name: &str,
        parent_scope: &Scope<'a>,
        _expected: &MtlxType,
    ) -> Result<FlatInput, FlattenError> {
        let out = ng
            .outputs
            .iter()
            .find(|o| o.name == out_name)
            .ok_or_else(|| FlattenError::Missing {
                what: format!("output `{}` on nodegraph `{}`", out_name, ng.name),
            })?;
        let mut interface = HashMap::new();
        for input in &ng.inputs {
            if let Some(binding) = parent_scope.interface.get(&input.name) {
                interface.insert(input.name.clone(), binding.clone());
            } else {
                let default = match &input.binding {
                    InputBinding::Empty => {
                        return Err(FlattenError::Missing {
                            what: format!(
                                "required nodegraph input `{}` of type {:?}",
                                input.name, input.ty
                            ),
                        });
                    }
                    InputBinding::Value(text) => {
                        if input.ty.is_shader_like() && text.trim().is_empty() {
                            FlatInput::Empty
                        } else {
                            let value = parse_literal(&input.ty, text).ok_or_else(|| {
                                FlattenError::Unsupported {
                                    what: format!(
                                        "invalid nodegraph input default `{}` for `{}` of type {:?}",
                                        text, input.name, input.ty
                                    ),
                                }
                            })?;
                            FlatInput::Value(apply_input_color_space(
                                value,
                                &input.ty,
                                input.colorspace.as_deref(),
                                &input.name,
                            ))
                        }
                    }
                    InputBinding::DefaultGeomProp(prop) => {
                        let kind = geometric_kind(prop);
                        let id = self.add_node(FlatNode {
                            kind: FlatNodeKind::Geometric {
                                kind: kind.clone(),
                                index: 0,
                            },
                            output_type: geometric_output_type(&kind),
                            inputs: vec![],
                        });
                        FlatInput::Node {
                            node: id,
                            output: None,
                        }
                    }
                    _ => {
                        return Err(FlattenError::Unsupported {
                            what: format!(
                                "unsupported default binding for nodegraph input `{}`",
                                input.name
                            ),
                        });
                    }
                };
                interface.insert(input.name.clone(), default);
            }
        }
        let mut nodes = HashMap::new();
        for n in &ng.nodes {
            nodes.insert(n.name.clone(), n);
        }
        let mut outputs = HashMap::new();
        for o in &ng.outputs {
            outputs.insert(o.name.clone(), o);
        }
        let mut tokens = parent_scope.tokens.clone();
        apply_token_declarations(&mut tokens, &ng.tokens)?;
        let scope = Scope {
            id: ScopeId::NodeGraph(ng.name.clone(), self.nodes.len() as u32),
            interface,
            nodes,
            outputs,
            tokens,
        };
        self.materialize_output(&scope, out)
    }

    fn materialize_output(
        &mut self,
        scope: &Scope<'a>,
        out: &super::types::RawOutput,
    ) -> Result<FlatInput, FlattenError> {
        match &out.binding {
            InputBinding::NodeRef { nodename, output } => {
                let node = scope.nodes.get(nodename.as_str()).copied().ok_or_else(|| {
                    FlattenError::Missing {
                        what: format!("output references missing node `{}`", nodename),
                    }
                })?;
                let normalized_output =
                    self.normalize_node_output(node, output.as_deref(), &out.ty)?;
                let is_primitive = PRIMITIVE_CATEGORIES.contains(&node.category.as_str());
                let id =
                    self.materialize_node_for_output(scope, node, normalized_output.as_deref())?;
                Ok(FlatInput::Node {
                    node: id,
                    output: is_primitive.then_some(normalized_output).flatten(),
                })
            }
            InputBinding::InterfaceName(name) => {
                scope
                    .interface
                    .get(name)
                    .cloned()
                    .ok_or_else(|| FlattenError::Missing {
                        what: format!("interface `{}` for output `{}`", name, out.name),
                    })
            }
            InputBinding::Value(text) => {
                let parsed =
                    parse_literal(&out.ty, text).ok_or_else(|| FlattenError::Unsupported {
                        what: format!(
                            "invalid output literal `{}` for `{}` of type {:?}",
                            text, out.name, out.ty
                        ),
                    })?;
                Ok(FlatInput::Value(parsed))
            }
            InputBinding::Empty => Err(FlattenError::Missing {
                what: format!("nodename on nodegraph output `{}`", out.name),
            }),
            InputBinding::NodeGraphRef { nodegraph, output } => {
                let ng = self
                    .nodegraphs
                    .get(nodegraph.as_str())
                    .copied()
                    .ok_or_else(|| FlattenError::Missing {
                        what: format!("nested nodegraph `{}`", nodegraph),
                    })?;
                let out_name =
                    select_nodegraph_ref_output(ng, output.as_deref(), &out.ty)?.to_string();
                self.materialize_nodegraph_output(ng, &out_name, scope, &out.ty)
            }
            InputBinding::DefaultGeomProp(prop) => {
                let kind = geometric_kind(prop);
                let id = self.add_node(FlatNode {
                    kind: FlatNodeKind::Geometric {
                        kind: kind.clone(),
                        index: 0,
                    },
                    output_type: geometric_output_type(&kind),
                    inputs: vec![],
                });
                Ok(FlatInput::Node {
                    node: id,
                    output: None,
                })
            }
        }
    }

    fn add_node(&mut self, n: FlatNode) -> FlatNodeId {
        let id = self.nodes.len() as FlatNodeId;
        self.nodes.push(n);
        id
    }
}

fn is_combinator(category: &str) -> bool {
    matches!(category, "mix" | "layer" | "add" | "multiply")
        // The disambiguated forms used inside a flat graph.
        || matches!(category, "mix_bsdf" | "add_bsdf" | "multiply_bsdf")
}

fn is_shading_category(category: &str) -> bool {
    matches!(
        category,
        "oren_nayar_diffuse_bsdf"
            | "burley_diffuse_bsdf"
            | "translucent_bsdf"
            | "dielectric_bsdf"
            | "conductor_bsdf"
            | "generalized_schlick_bsdf"
            | "sheen_bsdf"
            | "subsurface_bsdf"
            | "thin_film_bsdf"
            | "chiang_hair_bsdf"
            | "uniform_edf"
            | "conical_edf"
            | "measured_edf"
            | "generalized_schlick_edf"
            | "absorption_vdf"
            | "anisotropic_vdf"
    )
}

fn is_geometric_category(category: &str) -> bool {
    matches!(
        category,
        "position" | "normal" | "tangent" | "bitangent" | "texcoord" | "geomcolor"
    )
}

fn local_nodedef_matches(
    def: &super::types::RawNodeDef,
    inputs: &[(String, MtlxType)],
    output: &MtlxType,
) -> bool {
    if def.outputs.len() == 1 && !output_type_compatible(&def.outputs[0].ty, output) {
        return false;
    }
    if def.outputs.len() > 1
        && *output != MtlxType::None
        && !def
            .outputs
            .iter()
            .any(|o| output_type_compatible(&o.ty, output))
    {
        return false;
    }
    for (name, ty) in inputs {
        let Some(decl) = def.inputs.iter().find(|i| &i.name == name) else {
            return false;
        };
        if decl.ty != *ty && *ty != MtlxType::None {
            return false;
        }
    }
    true
}

fn node_resolve_output_type(node: &RawNodeUse) -> MtlxType {
    if node.ty.as_str() == "multioutput" {
        MtlxType::None
    } else {
        node.ty.clone()
    }
}

fn output_type_compatible(output: &MtlxType, expected: &MtlxType) -> bool {
    output == expected
        || *output == MtlxType::None
        || *expected == MtlxType::None
        || (*output == MtlxType::String && *expected == MtlxType::Filename)
}

fn version_matches(candidate: Option<&str>, requested: &str) -> bool {
    let Some(candidate) = candidate else {
        return false;
    };
    match (parse_version(candidate), parse_version(requested)) {
        (Some(a), Some(b)) => a == b,
        _ => candidate == requested,
    }
}

fn parse_version(s: &str) -> Option<(u32, u32)> {
    let mut parts = s.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().map(|p| p.parse().ok()).unwrap_or(Some(0))?;
    if parts.next().is_some() {
        return None;
    }
    Some((major, minor))
}

fn apply_input_color_space(
    value: MtlxValue,
    ty: &MtlxType,
    colorspace: Option<&str>,
    input_name: &str,
) -> MtlxValue {
    let Some(colorspace) = colorspace else {
        return value;
    };
    match (value, ty) {
        (MtlxValue::Color3(c), MtlxType::Color3) => {
            MtlxValue::Color3(convert_color3(c, colorspace, input_name))
        }
        (MtlxValue::Color4(c), MtlxType::Color4) => {
            let rgb = convert_color3(glam::Vec3::new(c.x, c.y, c.z), colorspace, input_name);
            MtlxValue::Color4(glam::Vec4::new(rgb.x, rgb.y, rgb.z, c.w))
        }
        (v, _) => v,
    }
}

fn convert_color3(c: glam::Vec3, colorspace: &str, input_name: &str) -> glam::Vec3 {
    match colorspace {
        "srgb_texture" | "g22_rec709" | "g22_ap1" | "srgb_displayp3" => srgb_to_linear(c),
        "linear" | "lin_rec709" | "scene_linear" | "none" => c,
        other => {
            eprintln!(
                "[mtlx] warning: colorspace `{}` on value input `{}` is not supported yet; treating value as linear",
                other, input_name
            );
            c
        }
    }
}

fn select_nodedef_output<'a>(
    node: &RawNodeUse,
    outputs: &'a [super::types::RawOutput],
    requested: Option<&str>,
) -> Result<Option<&'a super::types::RawOutput>, FlattenError> {
    if outputs.is_empty() {
        return Ok(None);
    }
    if outputs.len() == 1 {
        return Ok(Some(&outputs[0]));
    }
    let Some(name) = requested else {
        return Err(FlattenError::Missing {
            what: format!("output name for multi-output node `{}`", node.name),
        });
    };
    let out = outputs
        .iter()
        .find(|o| o.name == name)
        .ok_or_else(|| FlattenError::Missing {
            what: format!("output `{}` on nodedef for node `{}`", name, node.name),
        })?;
    Ok(Some(out))
}

fn select_nodegraph_output<'a>(
    node: &RawNodeUse,
    nd_outputs: &[super::types::RawOutput],
    ng: &'a RawNodeGraph,
    requested: Option<&str>,
) -> Result<&'a super::types::RawOutput, FlattenError> {
    if nd_outputs.is_empty() {
        return Err(FlattenError::Missing {
            what: format!("output declaration on nodedef for node `{}`", node.name),
        });
    }
    if nd_outputs.len() != ng.outputs.len() {
        return Err(FlattenError::Unsupported {
            what: format!(
                "nodegraph `{}` output count {} does not match nodedef output count {} for node `{}`",
                ng.name,
                ng.outputs.len(),
                nd_outputs.len(),
                node.name
            ),
        });
    }
    let nd_out = if nd_outputs.len() == 1 {
        &nd_outputs[0]
    } else {
        let Some(name) = requested else {
            return Err(FlattenError::Missing {
                what: format!("output name for multi-output node `{}`", node.name),
            });
        };
        nd_outputs
            .iter()
            .find(|o| o.name == name)
            .ok_or_else(|| FlattenError::Missing {
                what: format!("output `{}` on nodedef for node `{}`", name, node.name),
            })?
    };
    if nd_outputs.len() > 1 {
        for nd in nd_outputs {
            let ng_out = ng
                .outputs
                .iter()
                .find(|o| o.name == nd.name)
                .ok_or_else(|| FlattenError::Missing {
                    what: format!(
                        "output `{}` on nodegraph `{}` implementing node `{}`",
                        nd.name, ng.name, node.name
                    ),
                })?;
            if !output_type_compatible(&ng_out.ty, &nd.ty) {
                return Err(FlattenError::Unsupported {
                    what: format!(
                        "nodegraph `{}` output `{}` has type {:?}, nodedef expects {:?}",
                        ng.name, ng_out.name, ng_out.ty, nd.ty
                    ),
                });
            }
        }
    }
    let ng_out = if nd_outputs.len() == 1 {
        &ng.outputs[0]
    } else {
        ng.outputs
            .iter()
            .find(|o| o.name == nd_out.name)
            .ok_or_else(|| FlattenError::Missing {
                what: format!(
                    "output `{}` on nodegraph `{}` implementing node `{}`",
                    nd_out.name, ng.name, node.name
                ),
            })?
    };
    if !output_type_compatible(&ng_out.ty, &nd_out.ty) {
        return Err(FlattenError::Unsupported {
            what: format!(
                "nodegraph `{}` output `{}` has type {:?}, nodedef expects {:?}",
                ng.name, ng_out.name, ng_out.ty, nd_out.ty
            ),
        });
    }
    Ok(ng_out)
}

fn select_nodegraph_ref_output<'a>(
    ng: &'a RawNodeGraph,
    requested: Option<&str>,
    expected_type: &MtlxType,
) -> Result<&'a str, FlattenError> {
    if ng.outputs.is_empty() {
        return Err(FlattenError::Missing {
            what: format!("output on nodegraph `{}`", ng.name),
        });
    }
    let out = if ng.outputs.len() == 1 {
        &ng.outputs[0]
    } else {
        let Some(name) = requested else {
            return Err(FlattenError::Missing {
                what: format!("output name for multi-output nodegraph `{}`", ng.name),
            });
        };
        ng.outputs
            .iter()
            .find(|o| o.name == name)
            .ok_or_else(|| FlattenError::Missing {
                what: format!("output `{}` on nodegraph `{}`", name, ng.name),
            })?
    };
    if !output_type_compatible(&out.ty, expected_type) {
        return Err(FlattenError::Unsupported {
            what: format!(
                "output `{}` on nodegraph `{}` has type {:?}, expected {:?}",
                out.name, ng.name, out.ty, expected_type
            ),
        });
    }
    Ok(out.name.as_str())
}

fn geometric_kind(name: &str) -> GeometricKind {
    match name {
        "position" | "Pworld" | "Pobject" => GeometricKind::Position,
        "normal" | "Nworld" | "Nobject" => GeometricKind::Normal,
        "tangent" | "Tworld" | "Tobject" => GeometricKind::Tangent,
        "bitangent" | "Bworld" | "Bobject" => GeometricKind::Bitangent,
        "texcoord" | "UV0" => GeometricKind::Texcoord,
        "geomcolor" => GeometricKind::Geomcolor,
        "viewdirection" | "Vworld" => GeometricKind::ViewDirection,
        other => GeometricKind::Geompropvalue(other.to_string()),
    }
}

fn implied_geom_space(prop: &str) -> Option<&'static str> {
    match prop {
        "Pworld" | "Nworld" | "Tworld" | "Bworld" | "Vworld" => Some("world"),
        "Pobject" | "Nobject" | "Tobject" | "Bobject" => Some("object"),
        _ => None,
    }
}

fn geometric_output_type(kind: &GeometricKind) -> MtlxType {
    match kind {
        GeometricKind::Position
        | GeometricKind::Normal
        | GeometricKind::Tangent
        | GeometricKind::Bitangent
        | GeometricKind::ViewDirection => MtlxType::Vector3,
        GeometricKind::Texcoord => MtlxType::Vector2,
        GeometricKind::Geomcolor => MtlxType::Color3,
        GeometricKind::Geompropvalue(_) => MtlxType::Float,
    }
}

fn zero_value(ty: &MtlxType) -> Option<MtlxValue> {
    use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};
    Some(match ty {
        MtlxType::Boolean => MtlxValue::Boolean(false),
        MtlxType::Integer => MtlxValue::Integer(0),
        MtlxType::Float => MtlxValue::Float(0.0),
        MtlxType::Color3 => MtlxValue::Color3(Vec3::ZERO),
        MtlxType::Color4 => MtlxValue::Color4(Vec4::ZERO),
        MtlxType::Vector2 => MtlxValue::Vector2(Vec2::ZERO),
        MtlxType::Vector3 => MtlxValue::Vector3(Vec3::ZERO),
        MtlxType::Vector4 => MtlxValue::Vector4(Vec4::ZERO),
        MtlxType::Matrix33 => MtlxValue::Matrix33(Mat3::IDENTITY),
        MtlxType::Matrix44 => MtlxValue::Matrix44(Mat4::IDENTITY),
        MtlxType::String => MtlxValue::String(String::new()),
        MtlxType::Filename => MtlxValue::Filename(String::new()),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::mtlx::compile;
    use crate::scene_loader::mtlx_loader::library::load_standard_library;
    use crate::scene_loader::mtlx_loader::parser::parse_str;
    use std::path::{Path, PathBuf};

    fn lib_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("lib/materialx/libraries")
    }

    const SAMPLE_LAMBERT: &str = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodegraph name="NG_my">
    <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
      <input name="weight" type="float" value="1.0"/>
      <input name="color" type="color3" value="0.8, 0.4, 0.2"/>
      <input name="roughness" type="float" value="0.0"/>
    </oren_nayar_diffuse_bsdf>
    <surface name="srf" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="diffuse"/>
    </surface>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_my"/>
  </surfacematerial>
</materialx>"#;

    #[test]
    fn flatten_simple_lambert() {
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(SAMPLE_LAMBERT, Path::new("inline.mtlx")).expect("parse");
        let graph = flatten_material(&lib, &doc, "MyMat").expect("flatten");
        assert!(graph.nodes.len() >= 3, "got {} nodes", graph.nodes.len());
        match &graph.nodes[graph.root as usize].kind {
            FlatNodeKind::SurfaceMaterial => {}
            other => panic!("root not surfacematerial: {:?}", other),
        }
    }

    #[test]
    fn inherited_srgb_color_value_is_linearized() {
        let src = r#"<?xml version="1.0"?>
<materialx version="1.39" colorspace="srgb_texture">
  <nodegraph name="NG_my">
    <constant name="c" type="color3">
      <input name="value" type="color3" value="0.5,0.5,0.5"/>
    </constant>
    <surface_unlit name="srf" type="surfaceshader">
      <input name="emission_color" type="color3" nodename="c"/>
    </surface_unlit>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_my"/>
  </surfacematerial>
</materialx>"#;
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(src, Path::new("inline.mtlx")).expect("parse");
        let graph = flatten_material(&lib, &doc, "MyMat").expect("flatten");
        let constant = graph
            .nodes
            .iter()
            .find(|node| {
                matches!(
                    &node.kind,
                    FlatNodeKind::Pattern { category } if category == "constant"
                )
            })
            .expect("constant node");
        let value = constant
            .inputs
            .iter()
            .find(|input| input.name == "value")
            .expect("constant value input");
        let expected = crate::color::srgb_to_linear(glam::Vec3::splat(0.5));
        match &value.binding {
            FlatInput::Value(MtlxValue::Color3(c)) => {
                assert!(c.abs_diff_eq(expected, 1.0e-6));
            }
            other => panic!("expected converted color value, got {:?}", other),
        }
    }

    #[test]
    fn local_nodedef_version_treats_missing_minor_as_zero() {
        let src = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodedef name="ND_my_const" node="my_const" version="1.0" isdefaultversion="true">
    <output name="out" type="color3"/>
  </nodedef>
  <nodegraph name="NG_my_const" nodedef="ND_my_const">
    <constant name="c" type="color3">
      <input name="value" type="color3" value="0.25,0.5,0.75"/>
    </constant>
    <output name="out" type="color3" nodename="c"/>
  </nodegraph>
  <nodegraph name="NG_mat">
    <my_const name="c" type="color3" version="1"/>
    <surface_unlit name="srf" type="surfaceshader">
      <input name="emission_color" type="color3" nodename="c"/>
    </surface_unlit>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_mat"/>
  </surfacematerial>
</materialx>"#;
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(src, Path::new("inline.mtlx")).expect("parse");
        flatten_material(&lib, &doc, "MyMat").expect("local version 1 should match 1.0");
    }

    #[test]
    fn triplanarprojection_nodegraphs_flatten_and_compile() {
        let lib = load_standard_library(&lib_root()).expect("library");
        let cases = ["float", "color3", "color4", "vector2", "vector3", "vector4"];
        for ty in cases {
            let (bridge, input_name, input_ty, input_node) = if ty == "color3" {
                ("".to_string(), "emission_color", "color3", "tri")
            } else if ty == "float" {
                ("".to_string(), "emission", "float", "tri")
            } else {
                (
                    format!(
                        r#"
    <extract name="drive" type="float">
      <input name="in" type="{ty}" nodename="tri"/>
      <input name="index" type="integer" value="0"/>
    </extract>"#
                    ),
                    "emission",
                    "float",
                    "drive",
                )
            };
            let doc_src = format!(
                r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodegraph name="NG_my">
    <triplanarprojection name="tri" type="{ty}">
      <input name="filtertype" type="string" value="linear"/>
    </triplanarprojection>{bridge}
    <surface_unlit name="srf" type="surfaceshader">
      <input name="{input_name}" type="{input_ty}" nodename="{input_node}"/>
    </surface_unlit>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_my"/>
  </surfacematerial>
</materialx>"#
            );
            let doc = parse_str(&doc_src, Path::new("inline.mtlx")).expect("parse");
            let graph = flatten_material(&lib, &doc, "MyMat").expect("flatten");
            compile(
                &graph,
                Default::default(),
                Default::default(),
                Default::default(),
                Default::default(),
            )
            .expect("compile");
        }
    }

    #[test]
    fn standard_surface_nodegraph_flattens_and_compiles() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let lib = load_standard_library(&lib_root()).expect("library");
                let src = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodegraph name="NG_my">
    <standard_surface name="srf" type="surfaceshader">
      <input name="subsurface" type="float" value="0.5"/>
      <input name="emission" type="float" value="0.25"/>
      <input name="opacity" type="color3" value="0.8,0.8,0.8"/>
    </standard_surface>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_my"/>
  </surfacematerial>
</materialx>"#;
                let doc = parse_str(src, Path::new("inline.mtlx")).expect("parse");
                let graph = flatten_material(&lib, &doc, "MyMat").expect("flatten");
                compile(
                    &graph,
                    Default::default(),
                    Default::default(),
                    Default::default(),
                    Default::default(),
                )
                .expect("compile");
            })
            .expect("spawn")
            .join()
            .expect("standard_surface compile test panicked");
    }

    #[test]
    fn bxdf_surface_nodegraphs_flatten_and_compile() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let lib = load_standard_library(&lib_root()).expect("library");
                for category in [
                    "gltf_pbr",
                    "UsdPreviewSurface",
                    "open_pbr_surface",
                    "disney_principled",
                ] {
                    let extra_inputs = if category == "gltf_pbr" {
                        r#"
      <input name="attenuation_distance" type="float" value="1.0"/>"#
                    } else {
                        ""
                    };
                    let doc_src = format!(
                        r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodegraph name="NG_my">
    <{category} name="srf" type="surfaceshader">{extra_inputs}
    </{category}>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_my"/>
  </surfacematerial>
</materialx>"#
                    );
                    let doc = parse_str(&doc_src, Path::new("inline.mtlx")).expect("parse");
                    let graph = flatten_material(&lib, &doc, "MyMat").expect("flatten");
                    compile(
                        &graph,
                        Default::default(),
                        Default::default(),
                        Default::default(),
                        Default::default(),
                    )
                    .unwrap_or_else(|e| panic!("{} failed to compile: {}", category, e));
                }
            })
            .expect("spawn")
            .join()
            .expect("bxdf surface compile test panicked");
    }

    #[test]
    fn procedural_shape_nodegraphs_flatten_and_compile() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let lib = load_standard_library(&lib_root()).expect("library");
                let cases = [
                    ("line", "float", "emission"),
                    ("circle", "float", "emission"),
                    ("cloverleaf", "float", "emission"),
                    ("hexagon", "float", "emission"),
                    ("grid", "color3", "emission_color"),
                    ("crosshatch", "color3", "emission_color"),
                    ("tiledcircles", "color3", "emission_color"),
                    ("tiledcloverleafs", "color3", "emission_color"),
                    ("tiledhexagons", "color3", "emission_color"),
                ];
                for (category, ty, input_name) in cases {
                    let doc_src = format!(
                        r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodegraph name="NG_proc">
    <{category} name="pattern" type="{ty}"/>
    <surface_unlit name="srf" type="surfaceshader">
      <input name="{input_name}" type="{ty}" nodename="pattern"/>
    </surface_unlit>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_proc"/>
  </surfacematerial>
</materialx>"#
                    );
                    let doc = parse_str(&doc_src, Path::new("inline.mtlx")).expect("parse");
                    let graph = flatten_material(&lib, &doc, "MyMat").expect(category);
                    compile(
                        &graph,
                        HashMap::new(),
                        HashMap::new(),
                        HashMap::new(),
                        HashMap::new(),
                    )
                    .expect(category);
                }
            })
            .expect("spawn procedural shape nodegraph test")
            .join()
            .expect("procedural shape nodegraph test");
    }

    #[test]
    fn gooch_shade_uses_official_color3_nodegraph() {
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(
            r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodegraph name="NG_gooch">
    <gooch_shade name="gooch" type="color3"/>
    <surface_unlit name="srf" type="surfaceshader">
      <input name="emission_color" type="color3" nodename="gooch"/>
    </surface_unlit>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_gooch"/>
  </surfacematerial>
</materialx>"#,
            Path::new("inline.mtlx"),
        )
        .expect("parse");
        let graph = flatten_material(&lib, &doc, "MyMat").expect("flatten");
        assert!(
            graph
                .nodes
                .iter()
                .all(|node| !matches!(&node.kind, FlatNodeKind::Shading { category } if category == "gooch_shade"))
        );
        compile(
            &graph,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .expect("compile");
    }

    #[test]
    fn geompropvalue_flattens_to_pattern_node() {
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(
            r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodegraph name="NG_geomprop">
    <geompropvalue name="flag" type="boolean">
      <input name="default" type="boolean" value="true"/>
    </geompropvalue>
    <ifequal name="emit" type="float">
      <input name="value1" type="boolean" nodename="flag"/>
      <input name="value2" type="boolean" value="true"/>
      <input name="in1" type="float" value="1"/>
      <input name="in2" type="float" value="0"/>
    </ifequal>
    <surface_unlit name="srf" type="surfaceshader">
      <input name="emission" type="float" nodename="emit"/>
    </surface_unlit>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_geomprop"/>
  </surfacematerial>
</materialx>"#,
            Path::new("inline.mtlx"),
        )
        .expect("parse");
        let graph = flatten_material(&lib, &doc, "MyMat").expect("flatten");
        assert!(graph.nodes.iter().any(|node| {
            matches!(&node.kind, FlatNodeKind::Pattern { category } if category == "geompropvalue")
        }));
        compile(
            &graph,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .expect("compile");
    }

    #[test]
    fn custom_defaultgeomprop_errors_instead_of_texcoord_fallback() {
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(
            r#"<?xml version="1.0"?>
<materialx version="1.39">
  <geompropdef name="uv_custom" type="vector2" geomprop="texcoord" index="1"/>
  <nodegraph name="NG_custom_geomprop">
    <image name="img" type="color3">
      <input name="file" type="filename" value=""/>
      <input name="texcoord" type="vector2" defaultgeomprop="uv_custom"/>
      <input name="default" type="color3" value="1, 0, 0"/>
    </image>
    <surface_unlit name="srf" type="surfaceshader">
      <input name="emission_color" type="color3" nodename="img"/>
    </surface_unlit>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_custom_geomprop"/>
  </surfacematerial>
</materialx>"#,
            Path::new("inline.mtlx"),
        )
        .expect("parse");
        let graph = flatten_material(&lib, &doc, "MyMat").expect("flatten");
        let err = compile(
            &graph,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .expect_err("custom defaultgeomprop must not silently use UV0");
        assert!(
            err.to_string()
                .contains("custom defaultgeomprop `uv_custom`")
        );
    }

    const SAMPLE_DISABLED_ADD: &str = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodegraph name="NG_my">
    <add name="adder" type="color3">
      <input name="in1" type="color3" value="0.25, 0.5, 0.75"/>
      <input name="in2" type="color3" value="0.4, 0.4, 0.4"/>
      <input name="disable" type="boolean" value="true"/>
    </add>
    <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
      <input name="weight" type="float" value="1.0"/>
      <input name="color" type="color3" nodename="adder"/>
      <input name="roughness" type="float" value="0.0"/>
    </oren_nayar_diffuse_bsdf>
    <surface name="srf" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="diffuse"/>
    </surface>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_my"/>
  </surfacematerial>
</materialx>"#;

    const SAMPLE_NAMESPACED: &str = r#"<?xml version="1.0"?>
<materialx version="1.39" namespace="mylib">
  <nodegraph name="NG_my">
    <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
      <input name="weight" type="float" value="1.0"/>
      <input name="color" type="color3" value="0.7, 0.1, 0.1"/>
      <input name="roughness" type="float" value="0.0"/>
    </oren_nayar_diffuse_bsdf>
    <surface name="srf" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="diffuse"/>
    </surface>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_my"/>
  </surfacematerial>
</materialx>"#;

    #[test]
    fn flatten_namespaced_material_resolves_internal_references() {
        // A document with `namespace="mylib"` must resolve `MyMat` as
        // `mylib:MyMat`, and the internal `nodegraph="NG_my"` reference must
        // resolve to `mylib:NG_my` as well.
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(SAMPLE_NAMESPACED, Path::new("inline.mtlx")).expect("parse");
        // The material name should now be qualified.
        assert!(doc.materials.iter().any(|m| m.name == "mylib:MyMat"));
        let graph = flatten_material(&lib, &doc, "mylib:MyMat").expect("flatten");
        match &graph.nodes[graph.root as usize].kind {
            FlatNodeKind::SurfaceMaterial => {}
            other => panic!("root not surfacematerial: {:?}", other),
        }
    }

    #[test]
    fn flatten_disable_passes_through_default_input() {
        // Spec: a node with `disable="true"` outputs its defaultinput-named
        // input (or `default` value). add's nodedef declares
        // `<output ... defaultinput="in1"/>`, so a disabled add should expose
        // in1 (0.25,0.5,0.75) to the downstream diffuse BSDF, not the sum.
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(SAMPLE_DISABLED_ADD, Path::new("inline.mtlx")).expect("parse");
        let graph = flatten_material(&lib, &doc, "MyMat").expect("flatten");
        let mut diffuse: Option<&FlatNode> = None;
        for node in &graph.nodes {
            if let FlatNodeKind::Shading { category } = &node.kind
                && category == "oren_nayar_diffuse_bsdf"
            {
                diffuse = Some(node);
            }
        }
        let diffuse = diffuse.expect("oren_nayar_diffuse_bsdf node");
        let color_input = diffuse
            .inputs
            .iter()
            .find(|i| i.name == "color")
            .expect("color input");
        match &color_input.binding {
            FlatInput::Value(MtlxValue::Color3(v)) => {
                assert!(
                    (v.x - 0.25).abs() < 1e-6
                        && (v.y - 0.5).abs() < 1e-6
                        && (v.z - 0.75).abs() < 1e-6,
                    "disabled add should pass in1 through; got {:?}",
                    v
                );
            }
            FlatInput::Node { node: id, .. } => {
                // Cache may have materialised the value as a Constant.
                match &graph.nodes[*id as usize].kind {
                    FlatNodeKind::Constant {
                        value: MtlxValue::Color3(v),
                    } => {
                        assert!(
                            (v.x - 0.25).abs() < 1e-6
                                && (v.y - 0.5).abs() < 1e-6
                                && (v.z - 0.75).abs() < 1e-6,
                            "disabled add should pass in1 through; got {:?}",
                            v
                        );
                    }
                    other => panic!(
                        "color input must resolve to in1 literal, got node kind {:?}",
                        other
                    ),
                }
            }
            other => panic!("color input must be a literal or constant, got {:?}", other),
        }
    }

    const SAMPLE_UNIT_DISTANCE: &str = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodegraph name="NG_my">
    <constant name="size" type="float">
      <input name="value" type="float" value="100.0" unittype="distance" unit="centimeter"/>
    </constant>
    <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
      <input name="weight" type="float" nodename="size"/>
      <input name="color" type="color3" value="0.5, 0.5, 0.5"/>
      <input name="roughness" type="float" value="0.0"/>
    </oren_nayar_diffuse_bsdf>
    <surface name="srf" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="diffuse"/>
    </surface>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_my"/>
  </surfacematerial>
</materialx>"#;

    #[test]
    fn flatten_distance_unit_centimeter_scales_to_meter_base() {
        // Spec stdlib unitdef: centimeter has scale=0.01. A literal 100 cm
        // input must be converted to 1.0 m before reaching the compile stage.
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(SAMPLE_UNIT_DISTANCE, Path::new("inline.mtlx")).expect("parse");
        let graph = flatten_material(&lib, &doc, "MyMat").expect("flatten");
        let mut found = false;
        for node in &graph.nodes {
            if let FlatNodeKind::Pattern { category } = &node.kind
                && category == "constant"
            {
                let value_input = node
                    .inputs
                    .iter()
                    .find(|i| i.name == "value")
                    .expect("constant.value input");
                if let FlatInput::Value(MtlxValue::Float(v)) = &value_input.binding
                    && (*v - 1.0).abs() < 1e-5
                {
                    found = true;
                }
            }
        }
        assert!(
            found,
            "expected constant.value=1.0 after centimeter→meter unit conversion; graph: {:?}",
            graph
                .nodes
                .iter()
                .map(|n| format!("{:?}", n.kind))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn flatten_unknown_unit_errors() {
        // Spec compliance: unknown unit/unittype must be rejected rather than
        // silently passed through.
        let bad = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodegraph name="NG_my">
    <constant name="size" type="float">
      <input name="value" type="float" value="1.0" unittype="distance" unit="parsec"/>
    </constant>
    <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
      <input name="weight" type="float" nodename="size"/>
      <input name="color" type="color3" value="0.5, 0.5, 0.5"/>
      <input name="roughness" type="float" value="0.0"/>
    </oren_nayar_diffuse_bsdf>
    <surface name="srf" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="diffuse"/>
    </surface>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_my"/>
  </surfacematerial>
</materialx>"#;
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(bad, Path::new("inline.mtlx")).expect("parse");
        let res = flatten_material(&lib, &doc, "MyMat");
        assert!(
            matches!(res, Err(FlattenError::Unsupported { .. })),
            "expected Unsupported for unknown unit; got {:?}",
            res
        );
    }

    #[test]
    fn invalid_numeric_literal_errors_instead_of_becoming_string() {
        let bad = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodegraph name="NG_bad">
    <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
      <input name="weight" type="float" value="not-a-float"/>
      <input name="color" type="color3" value="0.8, 0.4, 0.2"/>
      <input name="roughness" type="float" value="0.0"/>
    </oren_nayar_diffuse_bsdf>
    <surface name="srf" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="diffuse"/>
    </surface>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_bad"/>
  </surfacematerial>
</materialx>"#;
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(bad, Path::new("inline.mtlx")).expect("parse");
        let err = flatten_material(&lib, &doc, "MyMat").expect_err("expected literal error");
        assert!(err.to_string().contains("invalid literal"));
    }

    #[test]
    fn missing_output_interface_errors_instead_of_empty_value() {
        let bad = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodegraph name="NG_bad">
    <output name="out" type="surfaceshader" interfacename="missing"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_bad"/>
  </surfacematerial>
</materialx>"#;
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(bad, Path::new("inline.mtlx")).expect("parse");
        let err = flatten_material(&lib, &doc, "MyMat").expect_err("expected interface error");
        assert!(err.to_string().contains("interface `missing`"));
    }

    #[test]
    fn missing_input_interface_errors_instead_of_empty_value() {
        let bad = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodegraph name="NG_bad">
    <constant name="c" type="color3">
      <input name="value" type="color3" interfacename="missing"/>
    </constant>
    <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
      <input name="weight" type="float" value="1.0"/>
      <input name="color" type="color3" nodename="c"/>
      <input name="roughness" type="float" value="0.0"/>
    </oren_nayar_diffuse_bsdf>
    <surface name="srf" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="diffuse"/>
    </surface>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_bad"/>
  </surfacematerial>
</materialx>"#;
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(bad, Path::new("inline.mtlx")).expect("parse");
        let err = flatten_material(&lib, &doc, "MyMat").expect_err("expected interface error");
        assert!(err.to_string().contains("interface `missing`"));
    }

    #[test]
    fn empty_shader_value_flattens_to_empty_connection() {
        let src = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" value=""/>
    <input name="backsurfaceshader" type="surfaceshader" value=""/>
    <input name="displacementshader" type="displacementshader" value=""/>
  </surfacematerial>
</materialx>"#;
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(src, Path::new("inline.mtlx")).expect("parse");
        let graph = flatten_material(&lib, &doc, "MyMat").expect("flatten");
        let root = &graph.nodes[graph.root as usize];
        let surface = root
            .inputs
            .iter()
            .find(|i| i.name == "surfaceshader")
            .expect("surfaceshader input");
        assert!(matches!(surface.binding, FlatInput::Empty));
        assert!(graph.back_root.is_none());
    }

    #[test]
    fn volumematerial_flattens_to_empty_passthrough_surface() {
        let src = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <volumematerial name="VolMat" type="material">
    <input name="volumeshader" type="volumeshader" value=""/>
  </volumematerial>
</materialx>"#;
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(src, Path::new("inline.mtlx")).expect("parse");
        let graph = flatten_material(&lib, &doc, "VolMat").expect("flatten");
        assert_eq!(graph.nodes.len(), 1);
        assert!(matches!(
            graph.nodes[graph.root as usize].kind,
            FlatNodeKind::SurfaceMaterial
        ));
        assert!(matches!(
            graph.nodes[graph.root as usize].inputs[0].binding,
            FlatInput::Empty
        ));
    }

    #[test]
    fn missing_required_nodedef_input_errors_instead_of_zero_default() {
        let src = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodedef name="ND_custom_color3" node="custom_color" nodegroup="test">
    <input name="required" type="float"/>
    <output name="out" type="color3"/>
  </nodedef>
  <nodegraph name="NG_custom_color3" nodedef="ND_custom_color3">
    <constant name="c" type="color3">
      <input name="value" type="color3" value="0.1,0.2,0.3"/>
    </constant>
    <output name="out" type="color3" nodename="c"/>
  </nodegraph>
  <nodegraph name="NG_mat">
    <custom_color name="c" type="color3"/>
    <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
      <input name="color" type="color3" nodename="c"/>
    </oren_nayar_diffuse_bsdf>
    <surface name="srf" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="diffuse"/>
    </surface>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_mat"/>
  </surfacematerial>
</materialx>"#;
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(src, Path::new("inline.mtlx")).expect("parse");
        let err = flatten_material(&lib, &doc, "MyMat").expect_err("expected required input error");
        assert!(
            err.to_string()
                .contains("required nodedef input `required`")
        );
    }

    #[test]
    fn missing_required_nodegraph_input_errors_instead_of_zero_default() {
        let src = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodegraph name="NG_mat">
    <input name="required" type="float"/>
    <surface name="srf" type="surfaceshader"/>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_mat"/>
  </surfacematerial>
</materialx>"#;
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(src, Path::new("inline.mtlx")).expect("parse");
        let err = flatten_material(&lib, &doc, "MyMat").expect_err("expected required input error");
        assert!(
            err.to_string()
                .contains("required nodegraph input `required`")
        );
    }

    #[test]
    fn nodegraph_output_without_nodename_errors_instead_of_zero_default() {
        let src = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodegraph name="NG_bad">
    <output name="out" type="surfaceshader"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_bad"/>
  </surfacematerial>
</materialx>"#;
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(src, Path::new("inline.mtlx")).expect("parse");
        let err = flatten_material(&lib, &doc, "MyMat").expect_err("expected missing output error");
        assert!(err.to_string().contains("nodename on nodegraph output"));
    }

    #[test]
    fn local_nodedef_overload_matches_input_types() {
        let src = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodedef name="ND_custom_float" node="custom_overload">
    <input name="in" type="float"/>
    <output name="out" type="color3"/>
  </nodedef>
  <nodegraph name="NG_custom_float" nodedef="ND_custom_float">
    <constant name="c" type="color3">
      <input name="value" type="color3" value="1,0,0"/>
    </constant>
    <output name="out" type="color3" nodename="c"/>
  </nodegraph>
  <nodedef name="ND_custom_color3" node="custom_overload">
    <input name="in" type="color3"/>
    <output name="out" type="color3"/>
  </nodedef>
  <nodegraph name="NG_custom_color3" nodedef="ND_custom_color3">
    <constant name="c" type="color3">
      <input name="value" type="color3" interfacename="in"/>
    </constant>
    <output name="out" type="color3" nodename="c"/>
  </nodegraph>
  <nodegraph name="NG_mat">
    <custom_overload name="c" type="color3">
      <input name="in" type="color3" value="0.2,0.3,0.4"/>
    </custom_overload>
    <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
      <input name="color" type="color3" nodename="c"/>
    </oren_nayar_diffuse_bsdf>
    <surface name="srf" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="diffuse"/>
    </surface>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_mat"/>
  </surfacematerial>
</materialx>"#;
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(src, Path::new("inline.mtlx")).expect("parse");
        let graph = flatten_material(&lib, &doc, "MyMat").expect("flatten");
        let custom = graph
            .nodes
            .iter()
            .find(|n| {
                matches!(
                    &n.kind,
                    FlatNodeKind::Pattern { category } if category == "constant"
                )
            })
            .expect("custom overload output constant");
        let value = custom
            .inputs
            .iter()
            .find(|i| i.name == "value")
            .expect("constant value input");
        let FlatInput::Value(MtlxValue::Color3(v)) = &value.binding else {
            panic!("expected color3 value, got {:?}", value.binding);
        };
        assert!((v.x - 0.2).abs() < 1e-6 && (v.y - 0.3).abs() < 1e-6);
    }

    #[test]
    fn target_specific_nodegraph_is_not_used_as_universal_implementation() {
        let src = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodedef name="ND_custom_color3" node="custom_targeted">
    <output name="out" type="color3"/>
  </nodedef>
  <nodegraph name="NG_custom_color3_genmdl" nodedef="ND_custom_color3" target="genmdl">
    <constant name="c" type="color3">
      <input name="value" type="color3" value="1,0,0"/>
    </constant>
    <output name="out" type="color3" nodename="c"/>
  </nodegraph>
  <nodegraph name="NG_mat">
    <custom_targeted name="c" type="color3"/>
    <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
      <input name="color" type="color3" nodename="c"/>
    </oren_nayar_diffuse_bsdf>
    <surface name="srf" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="diffuse"/>
    </surface>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_mat"/>
  </surfacematerial>
</materialx>"#;
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(src, Path::new("inline.mtlx")).expect("parse");
        let err = flatten_material(&lib, &doc, "MyMat").expect_err("expected target mismatch");
        assert!(
            err.to_string()
                .contains("nodegraph implementing local nodedef")
        );
    }

    #[test]
    fn target_specific_nodedef_is_not_used_as_universal_definition() {
        let src = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodedef name="ND_custom_color3" node="custom_targeted" target="genmdl">
    <output name="out" type="color3"/>
  </nodedef>
  <nodegraph name="NG_custom_color3" nodedef="ND_custom_color3">
    <constant name="c" type="color3">
      <input name="value" type="color3" value="1,0,0"/>
    </constant>
    <output name="out" type="color3" nodename="c"/>
  </nodegraph>
  <nodegraph name="NG_mat">
    <custom_targeted name="c" type="color3"/>
    <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
      <input name="color" type="color3" nodename="c"/>
    </oren_nayar_diffuse_bsdf>
    <surface name="srf" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="diffuse"/>
    </surface>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_mat"/>
  </surfacematerial>
</materialx>"#;
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(src, Path::new("inline.mtlx")).expect("parse");
        let err = flatten_material(&lib, &doc, "MyMat").expect_err("expected target mismatch");
        assert!(err.to_string().contains("no nodedef found"));
    }

    #[test]
    fn multi_output_node_requires_valid_output_name() {
        let src = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodegraph name="NG_mat">
    <artistic_ior name="ior" type="multioutput"/>
    <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
      <input name="color" type="color3" nodename="ior"/>
    </oren_nayar_diffuse_bsdf>
    <surface name="srf" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="diffuse"/>
    </surface>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_mat"/>
  </surfacematerial>
</materialx>"#;
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(src, Path::new("inline.mtlx")).expect("parse");
        let err = flatten_material(&lib, &doc, "MyMat").expect_err("expected missing output");
        assert!(
            err.to_string()
                .contains("output name for multi-output node")
        );

        let bad = src.replace("nodename=\"ior\"", "nodename=\"ior\" output=\"bad\"");
        let doc = parse_str(&bad, Path::new("inline.mtlx")).expect("parse");
        let err = flatten_material(&lib, &doc, "MyMat").expect_err("expected unknown output");
        assert!(err.to_string().contains("output `bad`"));
    }

    #[test]
    fn custom_multi_output_nodegraph_uses_requested_output() {
        let src = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodedef name="ND_double_color" node="double_color">
    <input name="in" type="color3" value="0,0,0"/>
    <output name="c1" type="color3"/>
    <output name="c2" type="color3"/>
  </nodedef>
  <nodegraph name="NG_double_color" nodedef="ND_double_color">
    <constant name="red" type="color3">
      <input name="value" type="color3" value="1,0,0"/>
    </constant>
    <constant name="passed" type="color3">
      <input name="value" type="color3" interfacename="in"/>
    </constant>
    <output name="c1" type="color3" nodename="red"/>
    <output name="c2" type="color3" nodename="passed"/>
  </nodegraph>
  <nodegraph name="NG_mat">
    <double_color name="dbl" type="color3">
      <input name="in" type="color3" value="0.2,0.3,0.4"/>
    </double_color>
    <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
      <input name="color" type="color3" nodename="dbl" output="c2"/>
    </oren_nayar_diffuse_bsdf>
    <surface name="srf" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="diffuse"/>
    </surface>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_mat"/>
  </surfacematerial>
</materialx>"#;
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(src, Path::new("inline.mtlx")).expect("parse");
        let graph = flatten_material(&lib, &doc, "MyMat").expect("flatten");
        let passed = graph
            .nodes
            .iter()
            .find(|n| {
                matches!(
                    &n.kind,
                    FlatNodeKind::Pattern { category } if category == "constant"
                ) && n.inputs.iter().any(|i| {
                    matches!(
                        &i.binding,
                        FlatInput::Value(MtlxValue::Color3(v))
                            if (v.x - 0.2).abs() < 1e-6 && (v.y - 0.3).abs() < 1e-6
                    )
                })
            })
            .expect("requested c2 constant");
        assert_eq!(passed.output_type, MtlxType::Color3);
    }

    #[test]
    fn multi_output_nodegraph_reference_requires_valid_output_name() {
        let src = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodegraph name="NG_multi">
    <surface name="srf1" type="surfaceshader"/>
    <surface name="srf2" type="surfaceshader"/>
    <output name="a" type="surfaceshader" nodename="srf1"/>
    <output name="b" type="surfaceshader" nodename="srf2"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_multi"/>
  </surfacematerial>
</materialx>"#;
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(src, Path::new("inline.mtlx")).expect("parse");
        let err = flatten_material(&lib, &doc, "MyMat").expect_err("expected missing output");
        assert!(
            err.to_string()
                .contains("output name for multi-output nodegraph")
        );

        let bad = src.replace(
            "nodegraph=\"NG_multi\"",
            "nodegraph=\"NG_multi\" output=\"bad\"",
        );
        let doc = parse_str(&bad, Path::new("inline.mtlx")).expect("parse");
        let err = flatten_material(&lib, &doc, "MyMat").expect_err("expected unknown output");
        assert!(err.to_string().contains("output `bad`"));
    }

    #[test]
    fn disabled_multi_output_node_uses_requested_output_default() {
        let src = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodedef name="ND_double_color" node="double_color">
    <output name="c1" type="color3" default="1,0,0"/>
    <output name="c2" type="color3" default="0,1,0"/>
  </nodedef>
  <nodegraph name="NG_double_color" nodedef="ND_double_color">
    <constant name="red" type="color3">
      <input name="value" type="color3" value="1,0,0"/>
    </constant>
    <output name="c1" type="color3" nodename="red"/>
    <output name="c2" type="color3" nodename="red"/>
  </nodegraph>
  <nodegraph name="NG_mat">
    <double_color name="dbl" type="color3">
      <input name="disable" type="boolean" value="true"/>
    </double_color>
    <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
      <input name="color" type="color3" nodename="dbl" output="c2"/>
    </oren_nayar_diffuse_bsdf>
    <surface name="srf" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="diffuse"/>
    </surface>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_mat"/>
  </surfacematerial>
</materialx>"#;
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(src, Path::new("inline.mtlx")).expect("parse");
        let graph = flatten_material(&lib, &doc, "MyMat").expect("flatten");
        let green = graph.nodes.iter().any(|n| {
            matches!(
                &n.kind,
                FlatNodeKind::Constant {
                    value: MtlxValue::Color3(v)
                } if (v.x - 0.0).abs() < 1e-6 && (v.y - 1.0).abs() < 1e-6
            )
        });
        assert!(green);
    }

    #[test]
    fn functional_nodegraph_outputs_must_match_nodedef_outputs() {
        let src = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodedef name="ND_custom_color" node="custom_color">
    <output name="out" type="color3"/>
  </nodedef>
  <nodegraph name="NG_custom_color" nodedef="ND_custom_color">
    <constant name="a" type="color3">
      <input name="value" type="color3" value="1,0,0"/>
    </constant>
    <constant name="b" type="color3">
      <input name="value" type="color3" value="0,1,0"/>
    </constant>
    <output name="out" type="color3" nodename="a"/>
    <output name="extra" type="color3" nodename="b"/>
  </nodegraph>
  <nodegraph name="NG_mat">
    <custom_color name="c" type="color3"/>
    <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
      <input name="color" type="color3" nodename="c"/>
    </oren_nayar_diffuse_bsdf>
    <surface name="srf" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="diffuse"/>
    </surface>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_mat"/>
  </surfacematerial>
</materialx>"#;
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(src, Path::new("inline.mtlx")).expect("parse");
        let err = flatten_material(&lib, &doc, "MyMat").expect_err("expected output mismatch");
        assert!(err.to_string().contains("output count"));
    }

    #[test]
    fn functional_nodegraph_child_inputs_are_rejected() {
        let src = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodedef name="ND_custom_color" node="custom_color">
    <input name="in" type="color3" value="0,0,0"/>
    <output name="out" type="color3"/>
  </nodedef>
  <nodegraph name="NG_custom_color" nodedef="ND_custom_color">
    <input name="bad" type="color3" value="1,0,0"/>
    <constant name="c" type="color3">
      <input name="value" type="color3" interfacename="in"/>
    </constant>
    <output name="out" type="color3" nodename="c"/>
  </nodegraph>
  <nodegraph name="NG_mat">
    <custom_color name="c" type="color3"/>
    <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
      <input name="color" type="color3" nodename="c"/>
    </oren_nayar_diffuse_bsdf>
    <surface name="srf" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="diffuse"/>
    </surface>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_mat"/>
  </surfacematerial>
</materialx>"#;
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(src, Path::new("inline.mtlx")).expect("parse");
        let err = flatten_material(&lib, &doc, "MyMat").expect_err("expected child input error");
        assert!(err.to_string().contains("declares child inputs"));
    }

    #[test]
    fn missing_unsupported_input_type_errors() {
        let src = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodedef name="ND_custom_color" node="custom_color">
    <input name="unknown" type="studio_token"/>
    <output name="out" type="color3"/>
  </nodedef>
  <nodegraph name="NG_custom_color" nodedef="ND_custom_color">
    <constant name="c" type="color3">
      <input name="value" type="color3" value="1,0,0"/>
    </constant>
    <output name="out" type="color3" nodename="c"/>
  </nodegraph>
  <nodegraph name="NG_mat">
    <custom_color name="c" type="color3"/>
    <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
      <input name="color" type="color3" nodename="c"/>
    </oren_nayar_diffuse_bsdf>
    <surface name="srf" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="diffuse"/>
    </surface>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_mat"/>
  </surfacematerial>
</materialx>"#;
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(src, Path::new("inline.mtlx")).expect("parse");
        let err = flatten_material(&lib, &doc, "MyMat").expect_err("expected missing custom input");
        assert!(
            err.to_string().contains("required nodedef input `unknown`"),
            "{}",
            err
        );
    }

    #[test]
    fn nodedef_token_override_substitutes_filename_token() {
        let src = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodedef name="ND_token_image" node="token_image">
    <token name="tex" type="string" value="default"/>
    <output name="out" type="color3"/>
  </nodedef>
  <nodegraph name="NG_token_image" nodedef="ND_token_image">
    <image name="img" type="color3">
      <input name="file" type="filename" value="[tex].png"/>
    </image>
    <output name="out" type="color3" nodename="img"/>
  </nodegraph>
  <nodegraph name="NG_mat">
    <token_image name="img" type="color3">
      <token name="tex" type="string" value="custom"/>
    </token_image>
    <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
      <input name="color" type="color3" nodename="img"/>
    </oren_nayar_diffuse_bsdf>
    <surface name="srf" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="diffuse"/>
    </surface>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_mat"/>
  </surfacematerial>
</materialx>"#;
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(src, Path::new("inline.mtlx")).expect("parse");
        let graph = flatten_material(&lib, &doc, "MyMat").expect("flatten");
        let image = graph
            .nodes
            .iter()
            .find(|n| matches!(&n.kind, FlatNodeKind::Pattern { category } if category == "image"))
            .expect("image node");
        let file = image
            .inputs
            .iter()
            .find(|i| i.name == "file")
            .expect("image file");
        assert!(
            matches!(&file.binding, FlatInput::Value(MtlxValue::Filename(s)) if s == "custom.png")
        );
    }

    #[test]
    fn unsupported_geometry_filename_token_is_preserved() {
        let mut tokens = HashMap::new();
        tokens.insert("tex".to_string(), "albedo".to_string());
        let filename = substitute_filename_tokens("[tex].<asset>.<UDIM>.png", &tokens);
        assert_eq!(filename, "albedo.<asset>.<UDIM>.png");
    }

    #[test]
    fn missing_required_nodedef_token_errors() {
        let src = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodedef name="ND_token_image" node="token_image">
    <token name="tex" type="string"/>
    <output name="out" type="color3"/>
  </nodedef>
  <nodegraph name="NG_token_image" nodedef="ND_token_image">
    <image name="img" type="color3">
      <input name="file" type="filename" value="[tex].png"/>
    </image>
    <output name="out" type="color3" nodename="img"/>
  </nodegraph>
  <nodegraph name="NG_mat">
    <token_image name="img" type="color3"/>
    <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
      <input name="color" type="color3" nodename="img"/>
    </oren_nayar_diffuse_bsdf>
    <surface name="srf" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="diffuse"/>
    </surface>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_mat"/>
  </surfacematerial>
</materialx>"#;
        let lib = load_standard_library(&lib_root()).expect("library");
        let doc = parse_str(src, Path::new("inline.mtlx")).expect("parse");
        let err = flatten_material(&lib, &doc, "MyMat").expect_err("expected token error");
        assert!(err.to_string().contains("required token `tex`"));
    }
}
