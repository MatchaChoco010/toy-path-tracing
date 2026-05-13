use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use roxmltree::{Document, Node};

use super::types::{
    InputBinding, MtlxType, RawGeomPropDef, RawImplementation, RawInput, RawMaterial,
    RawMtlxDocument, RawNodeDef, RawNodeGraph, RawNodeUse, RawOutput, RawToken, RawTypeDef,
};

#[derive(Debug)]
pub enum ParseError {
    Io(std::io::Error, PathBuf),
    Xml(roxmltree::Error, PathBuf),
    Structure { message: String, path: PathBuf },
    Cycle { path: PathBuf },
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e, p) => write!(f, "I/O error reading {}: {}", p.display(), e),
            Self::Xml(e, p) => write!(f, "XML parse error in {}: {}", p.display(), e),
            Self::Structure { message, path } => {
                write!(f, "Malformed mtlx in {}: {}", path.display(), message)
            }
            Self::Cycle { path } => {
                write!(f, "XInclude cycle detected at {}", path.display())
            }
        }
    }
}

impl std::error::Error for ParseError {}

pub fn parse_document(path: &Path) -> Result<RawMtlxDocument, ParseError> {
    let mut visited = HashSet::new();
    let canonical = canonicalize(path);
    visited.insert(canonical.clone());
    parse_recursive(&canonical, &mut visited, None)
}

pub fn parse_str(content: &str, source_path: &Path) -> Result<RawMtlxDocument, ParseError> {
    parse_str_with_inherited_fileprefix(content, source_path, None)
}

fn parse_str_with_inherited_fileprefix(
    content: &str,
    source_path: &Path,
    inherited_fileprefix: Option<&str>,
) -> Result<RawMtlxDocument, ParseError> {
    let canonical = canonicalize(source_path);
    let doc = Document::parse(content).map_err(|e| ParseError::Xml(e, canonical.clone()))?;
    let root = doc.root_element();
    expect_materialx(&root, &canonical)?;

    let mut out = RawMtlxDocument {
        source_path: canonical.clone(),
        ..Default::default()
    };
    out.version = parse_version(required_attr(&root, &canonical, "version")?, &canonical)?;
    out.colorspace = root.attribute("colorspace").map(str::to_owned);
    out.namespace = root.attribute("namespace").map(str::to_owned);
    let doc_prefix = root
        .attribute("fileprefix")
        .or(inherited_fileprefix)
        .unwrap_or("");
    let doc_ns = out.namespace.clone();
    let doc_colorspace = out.colorspace.as_deref();

    for child in root.children().filter(Node::is_element) {
        match child.tag_name().name() {
            "include" => {}
            "typedef" => {
                out.typedefs.push(parse_typedef(&child, &canonical)?);
            }
            "geompropdef" => {
                out.geompropdefs
                    .push(parse_geompropdef(&child, &canonical)?);
            }
            "nodedef" => {
                out.nodedefs.push(parse_nodedef(
                    &child,
                    doc_prefix,
                    doc_colorspace,
                    &canonical,
                )?);
            }
            "nodegraph" => {
                let ng_prefix = child.attribute("fileprefix").unwrap_or(doc_prefix);
                let ng_colorspace = child.attribute("colorspace").or(doc_colorspace);
                let ng_name = required_attr(&child, &canonical, "name")?.to_string();
                let mut nested = Vec::new();
                collect_nested_materials(
                    child,
                    ng_prefix,
                    ng_colorspace,
                    &ng_name,
                    &mut nested,
                    &canonical,
                )?;
                if !nested.is_empty() {
                    for sib in child.children().filter(Node::is_element) {
                        let tag = sib.tag_name().name();
                        if matches!(
                            tag,
                            "input"
                                | "output"
                                | "token"
                                | "surfacematerial"
                                | "volumematerial"
                                | "material"
                        ) {
                            continue;
                        }
                        if sib.attribute("type").is_some() {
                            out.root_nodes.push(parse_node_use(
                                &sib,
                                ng_prefix,
                                ng_colorspace,
                                &canonical,
                            )?);
                        }
                    }
                }
                out.materials.extend(nested);
                out.nodegraphs.push(parse_nodegraph(
                    &child,
                    doc_prefix,
                    doc_colorspace,
                    &canonical,
                )?);
            }
            "implementation" => {
                out.implementations
                    .push(parse_implementation(&child, &canonical)?);
            }
            "surfacematerial" | "volumematerial" => {
                let category = child.tag_name().name().to_string();
                let name = required_attr(&child, &canonical, "name")?.to_string();
                let material_prefix = child.attribute("fileprefix").unwrap_or(doc_prefix);
                let inputs = parse_inputs(&child, material_prefix, doc_colorspace, &canonical)?;
                out.materials.push(RawMaterial {
                    name,
                    category,
                    inputs,
                    parent_nodegraph: None,
                });
            }
            "material" => {
                let name = required_attr(&child, &canonical, "name")?.to_string();
                let material_prefix = child.attribute("fileprefix").unwrap_or(doc_prefix);
                out.materials.push(RawMaterial {
                    name,
                    category: "material".into(),
                    inputs: parse_inputs(&child, material_prefix, doc_colorspace, &canonical)?,
                    parent_nodegraph: None,
                });
            }
            "look" | "lookgroup" | "propertyset" | "unittypedef" | "unitdef" | "targetdef"
            | "variantset" => {}
            _ => {
                if child.attribute("type").is_some() {
                    let category = child.tag_name().name().to_string();
                    let name = required_attr(&child, &canonical, "name")?.to_string();
                    let ty = MtlxType::parse(required_attr(&child, &canonical, "type")?);
                    out.root_nodes.push(super::types::RawNodeUse {
                        name,
                        category,
                        ty,
                        inputs: parse_inputs(&child, doc_prefix, doc_colorspace, &canonical)?,
                        tokens: parse_tokens(&child, &canonical)?,
                        outputs: parse_outputs(&child, &canonical)?,
                        version: child.attribute("version").map(str::to_owned),
                        nodedef: child.attribute("nodedef").map(str::to_owned),
                    });
                }
            }
        }
    }

    if let Some(ns) = &doc_ns {
        apply_namespace(&mut out, ns);
    }

    Ok(out)
}

/// Prefix all top-level element names and unqualified internal references
/// with the document namespace, so a `<materialx namespace="X">` document
/// behaves as if every element is in namespace `X`. References that are
/// already qualified (`other_ns:name`) are left untouched.
fn apply_namespace(doc: &mut RawMtlxDocument, ns: &str) {
    let qualify = |name: &str| -> String {
        if name.is_empty() || name.contains(':') {
            name.to_string()
        } else {
            format!("{}:{}", ns, name)
        }
    };
    for nd in &mut doc.nodedefs {
        nd.name = qualify(&nd.name);
        nd.node = qualify(&nd.node);
        if let Some(inh) = &nd.inherit {
            nd.inherit = Some(qualify(inh));
        }
    }
    for ng in &mut doc.nodegraphs {
        ng.name = qualify(&ng.name);
        if let Some(nd) = &ng.nodedef {
            ng.nodedef = Some(qualify(nd));
        }
        for n in &mut ng.nodes {
            qualify_node_use(n, &qualify);
        }
        for o in &mut ng.outputs {
            qualify_input_binding(&mut o.binding, &qualify);
        }
        for input in &mut ng.inputs {
            qualify_input_binding(&mut input.binding, &qualify);
        }
    }
    for m in &mut doc.materials {
        m.name = qualify(&m.name);
        if let Some(pn) = &m.parent_nodegraph {
            m.parent_nodegraph = Some(qualify(pn));
        }
        for input in &mut m.inputs {
            qualify_input_binding(&mut input.binding, &qualify);
        }
    }
    for n in &mut doc.root_nodes {
        qualify_node_use(n, &qualify);
    }
    for im in &mut doc.implementations {
        im.name = qualify(&im.name);
        im.nodedef = qualify(&im.nodedef);
        if let Some(g) = &im.nodegraph {
            im.nodegraph = Some(qualify(g));
        }
    }
}

fn qualify_node_use(n: &mut super::types::RawNodeUse, qualify: &impl Fn(&str) -> String) {
    n.name = qualify(&n.name);
    if let Some(nd) = &n.nodedef {
        n.nodedef = Some(qualify(nd));
    }
    for input in &mut n.inputs {
        qualify_input_binding(&mut input.binding, qualify);
    }
    for o in &mut n.outputs {
        qualify_input_binding(&mut o.binding, qualify);
    }
}

fn qualify_input_binding(binding: &mut InputBinding, qualify: &impl Fn(&str) -> String) {
    match binding {
        InputBinding::NodeRef { nodename, .. } => {
            *nodename = qualify(nodename);
        }
        InputBinding::NodeGraphRef { nodegraph, .. } => {
            *nodegraph = qualify(nodegraph);
        }
        _ => {}
    }
}

fn parse_recursive(
    path: &Path,
    visited: &mut HashSet<PathBuf>,
    inherited_fileprefix: Option<&str>,
) -> Result<RawMtlxDocument, ParseError> {
    let content = fs::read_to_string(path).map_err(|e| ParseError::Io(e, path.to_path_buf()))?;
    let doc = Document::parse(&content).map_err(|e| ParseError::Xml(e, path.to_path_buf()))?;
    let root = doc.root_element();
    expect_materialx(&root, path)?;

    let mut out = RawMtlxDocument {
        source_path: path.to_path_buf(),
        ..Default::default()
    };
    out.version = parse_version(required_attr(&root, path, "version")?, path)?;
    out.colorspace = root.attribute("colorspace").map(str::to_owned);
    out.namespace = root.attribute("namespace").map(str::to_owned);
    let child_inherited_fileprefix = root.attribute("fileprefix").or(inherited_fileprefix);

    for child in root.children().filter(Node::is_element) {
        let tag = child.tag_name();
        let is_xi_include = (tag.namespace() == Some("http://www.w3.org/2001/XInclude")
            && tag.name() == "include")
            || tag.name() == "xi:include";
        if is_xi_include {
            let href = required_attr(&child, path, "href")?;
            let included_path = match path.parent() {
                Some(parent) => parent.join(href),
                None => PathBuf::from(href),
            };
            let canonical = canonicalize(&included_path);
            if visited.contains(&canonical) {
                return Err(ParseError::Cycle { path: canonical });
            }
            visited.insert(canonical.clone());
            let mut included = parse_recursive(&canonical, visited, child_inherited_fileprefix)?;
            if included.colorspace.is_none()
                && let Some(parent_colorspace) = root.attribute("colorspace")
            {
                apply_inherited_colorspace(&mut included, parent_colorspace);
            }
            if included.namespace.is_none()
                && let Some(parent_namespace) = root.attribute("namespace")
            {
                apply_namespace(&mut included, parent_namespace);
                included.namespace = Some(parent_namespace.to_string());
            }
            merge(&mut out, included);
        }
    }

    let current = parse_str_with_inherited_fileprefix(&content, path, inherited_fileprefix)?;
    merge(&mut out, current);

    Ok(out)
}

fn merge(dst: &mut RawMtlxDocument, src: RawMtlxDocument) {
    dst.nodedefs.extend(src.nodedefs);
    dst.nodegraphs.extend(src.nodegraphs);
    dst.implementations.extend(src.implementations);
    dst.typedefs.extend(src.typedefs);
    dst.geompropdefs.extend(src.geompropdefs);
    dst.materials.extend(src.materials);
    dst.root_nodes.extend(src.root_nodes);
}

fn canonicalize(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn qualify_with_namespace(name: &str, namespace: Option<&str>) -> String {
    if let Some(ns) = namespace
        && !name.is_empty()
        && !name.contains(':')
    {
        return format!("{}:{}", ns, name);
    }
    name.to_string()
}

fn apply_inherited_colorspace(doc: &mut RawMtlxDocument, colorspace: &str) {
    doc.colorspace = Some(colorspace.to_string());
    let apply_inputs = |inputs: &mut Vec<RawInput>| {
        for input in inputs {
            if input.colorspace.is_none() {
                input.colorspace = Some(colorspace.to_string());
            }
        }
    };
    for nd in &mut doc.nodedefs {
        apply_inputs(&mut nd.inputs);
    }
    for ng in &mut doc.nodegraphs {
        apply_inputs(&mut ng.inputs);
        for node in &mut ng.nodes {
            apply_inputs(&mut node.inputs);
        }
    }
    for material in &mut doc.materials {
        apply_inputs(&mut material.inputs);
    }
    for node in &mut doc.root_nodes {
        apply_inputs(&mut node.inputs);
    }
}

fn expect_materialx(root: &Node, path: &Path) -> Result<(), ParseError> {
    if root.tag_name().name() == "materialx" {
        Ok(())
    } else {
        Err(ParseError::Structure {
            message: format!(
                "expected <materialx> root, found <{}>",
                root.tag_name().name()
            ),
            path: path.to_path_buf(),
        })
    }
}

fn required_attr<'a, 'input>(
    node: &Node<'a, 'input>,
    path: &Path,
    attr: &str,
) -> Result<&'a str, ParseError> {
    node.attribute(attr).ok_or_else(|| ParseError::Structure {
        message: format!(
            "<{}> is missing required `{}` attribute",
            node.tag_name().name(),
            attr
        ),
        path: path.to_path_buf(),
    })
}

fn optional_bool_attr(node: &Node, path: &Path, attr: &str) -> Result<bool, ParseError> {
    let Some(value) = node.attribute(attr) else {
        return Ok(false);
    };
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(ParseError::Structure {
            message: format!(
                "<{}> attribute `{}` must be boolean, got `{}`",
                node.tag_name().name(),
                attr,
                value
            ),
            path: path.to_path_buf(),
        }),
    }
}

fn parse_version(attr: &str, path: &Path) -> Result<(u32, u32), ParseError> {
    let mut parts = attr.split('.');
    let major = parts
        .next()
        .ok_or_else(|| ParseError::Structure {
            message: "materialx version is empty".to_string(),
            path: path.to_path_buf(),
        })?
        .parse()
        .map_err(|_| ParseError::Structure {
            message: format!("invalid materialx version `{}`", attr),
            path: path.to_path_buf(),
        })?;
    let minor = parts
        .next()
        .ok_or_else(|| ParseError::Structure {
            message: format!("invalid materialx version `{}`", attr),
            path: path.to_path_buf(),
        })?
        .parse()
        .map_err(|_| ParseError::Structure {
            message: format!("invalid materialx version `{}`", attr),
            path: path.to_path_buf(),
        })?;
    if parts.next().is_some() {
        return Err(ParseError::Structure {
            message: format!("invalid materialx version `{}`", attr),
            path: path.to_path_buf(),
        });
    }
    Ok((major, minor))
}

fn parse_typedef(node: &Node, path: &Path) -> Result<RawTypeDef, ParseError> {
    let name = required_attr(node, path, "name")?.to_string();
    Ok(RawTypeDef {
        name,
        semantic: node.attribute("semantic").map(str::to_owned),
        context: node.attribute("context").map(str::to_owned),
    })
}

fn parse_geompropdef(node: &Node, path: &Path) -> Result<RawGeomPropDef, ParseError> {
    let name = qualify_with_namespace(
        required_attr(node, path, "name")?,
        node.attribute("namespace"),
    );
    let ty_str = required_attr(node, path, "type")?;
    let ty = MtlxType::parse(ty_str);
    let uniform = optional_bool_attr(node, path, "uniform")?;
    let geomprop = node.attribute("geomprop").map(str::to_owned);
    let space = node.attribute("space").map(str::to_owned);
    let index = node
        .attribute("index")
        .map(|s| {
            s.parse::<i32>().map_err(|_| ParseError::Structure {
                message: format!("geompropdef `{}` has invalid index `{}`", name, s),
                path: path.to_path_buf(),
            })
        })
        .transpose()?;
    validate_geompropdef(
        &name,
        &ty,
        uniform,
        geomprop.as_deref(),
        space.as_deref(),
        index,
        path,
    )?;
    Ok(RawGeomPropDef {
        name,
        ty,
        uniform,
        geomprop,
        space,
        index,
        unittype: node.attribute("unittype").map(str::to_owned),
        unit: node.attribute("unit").map(str::to_owned),
    })
}

fn validate_geompropdef(
    name: &str,
    ty: &MtlxType,
    uniform: bool,
    geomprop: Option<&str>,
    space: Option<&str>,
    index: Option<i32>,
    path: &Path,
) -> Result<(), ParseError> {
    if is_array_type(ty) {
        return Err(ParseError::Structure {
            message: format!(
                "geompropdef `{}` type `{}` must be non-array",
                name,
                ty.as_str()
            ),
            path: path.to_path_buf(),
        });
    }
    if matches!(ty, MtlxType::String | MtlxType::Filename) && !uniform {
        return Err(ParseError::Structure {
            message: format!(
                "geompropdef `{}` type `{}` must declare uniform=\"true\"",
                name,
                ty.as_str()
            ),
            path: path.to_path_buf(),
        });
    }
    if uniform && (geomprop.is_some() || space.is_some() || index.is_some()) {
        return Err(ParseError::Structure {
            message: format!(
                "geompropdef `{}` cannot specify geomprop, space, or index when uniform=\"true\"",
                name
            ),
            path: path.to_path_buf(),
        });
    }
    if geomprop.is_none() {
        if space.is_some() || index.is_some() {
            return Err(ParseError::Structure {
                message: format!(
                    "geompropdef `{}` cannot specify space or index without geomprop",
                    name
                ),
                path: path.to_path_buf(),
            });
        }
        return Ok(());
    }
    let geomprop = geomprop.unwrap();
    if let Some(i) = index
        && i < 0
    {
        return Err(ParseError::Structure {
            message: format!("geompropdef `{}` index must be non-negative", name),
            path: path.to_path_buf(),
        });
    }
    if let Some(space) = space
        && !matches!(space, "object" | "world" | "model")
    {
        return Err(ParseError::Structure {
            message: format!("geompropdef `{}` has invalid space `{}`", name, space),
            path: path.to_path_buf(),
        });
    }
    match geomprop {
        "position" | "normal" | "viewdirection" => {
            validate_geomprop_type(name, ty, &[MtlxType::Vector3], path)?;
            if index.is_some() {
                return Err(ParseError::Structure {
                    message: format!(
                        "geompropdef `{}` geomprop `{}` cannot specify index",
                        name, geomprop
                    ),
                    path: path.to_path_buf(),
                });
            }
        }
        "tangent" | "bitangent" => {
            validate_geomprop_type(name, ty, &[MtlxType::Vector3], path)?;
        }
        "texcoord" => {
            validate_geomprop_type(name, ty, &[MtlxType::Vector2], path)?;
            if space.is_some() {
                return Err(ParseError::Structure {
                    message: format!(
                        "geompropdef `{}` geomprop `texcoord` cannot specify space",
                        name
                    ),
                    path: path.to_path_buf(),
                });
            }
        }
        "geomcolor" => {
            validate_geomprop_type(
                name,
                ty,
                &[MtlxType::Float, MtlxType::Color3, MtlxType::Color4],
                path,
            )?;
            if space.is_some() {
                return Err(ParseError::Structure {
                    message: format!(
                        "geompropdef `{}` geomprop `geomcolor` cannot specify space",
                        name
                    ),
                    path: path.to_path_buf(),
                });
            }
        }
        other => {
            return Err(ParseError::Structure {
                message: format!("geompropdef `{}` has invalid geomprop `{}`", name, other),
                path: path.to_path_buf(),
            });
        }
    }
    Ok(())
}

fn validate_geomprop_type(
    name: &str,
    ty: &MtlxType,
    allowed: &[MtlxType],
    path: &Path,
) -> Result<(), ParseError> {
    if allowed.iter().any(|candidate| candidate == ty) {
        Ok(())
    } else {
        let allowed = allowed
            .iter()
            .map(MtlxType::as_str)
            .collect::<Vec<_>>()
            .join("/");
        Err(ParseError::Structure {
            message: format!(
                "geompropdef `{}` type `{}` does not match required type `{}`",
                name,
                ty.as_str(),
                allowed
            ),
            path: path.to_path_buf(),
        })
    }
}

fn is_array_type(ty: &MtlxType) -> bool {
    matches!(
        ty,
        MtlxType::IntegerArray
            | MtlxType::FloatArray
            | MtlxType::Color3Array
            | MtlxType::Color4Array
            | MtlxType::Vector2Array
            | MtlxType::Vector3Array
            | MtlxType::Vector4Array
            | MtlxType::StringArray
            | MtlxType::GeomnameArray
    )
}

fn parse_nodedef(
    node: &Node,
    parent_prefix: &str,
    parent_colorspace: Option<&str>,
    path: &Path,
) -> Result<RawNodeDef, ParseError> {
    let namespace = node.attribute("namespace");
    let name = qualify_with_namespace(required_attr(node, path, "name")?, namespace);
    let nodecat = qualify_with_namespace(required_attr(node, path, "node")?, namespace);
    let local_prefix = node.attribute("fileprefix").unwrap_or(parent_prefix);
    let colorspace = node.attribute("colorspace").or(parent_colorspace);
    Ok(RawNodeDef {
        name,
        node: nodecat,
        inputs: parse_inputs(node, local_prefix, colorspace, path)?,
        tokens: parse_tokens(node, path)?,
        outputs: parse_outputs(node, path)?,
        version: node.attribute("version").map(str::to_owned),
        is_default_version: optional_bool_attr(node, path, "isdefaultversion")?,
        inherit: node.attribute("inherit").map(str::to_owned),
        target: node.attribute("target").map(str::to_owned),
        nodegroup: node.attribute("nodegroup").map(str::to_owned),
        doc: node.attribute("doc").map(str::to_owned),
    })
}

fn parse_implementation(node: &Node, path: &Path) -> Result<RawImplementation, ParseError> {
    let name = required_attr(node, path, "name")?.to_string();
    let nodedef = required_attr(node, path, "nodedef")?.to_string();
    Ok(RawImplementation {
        name,
        nodedef,
        nodegraph: node.attribute("nodegraph").map(str::to_owned),
        function: node.attribute("function").map(str::to_owned),
        file: node.attribute("file").map(str::to_owned),
        target: node.attribute("target").map(str::to_owned),
        format: node.attribute("format").map(str::to_owned),
    })
}

fn parse_nodegraph(
    node: &Node,
    parent_prefix: &str,
    parent_colorspace: Option<&str>,
    path: &Path,
) -> Result<RawNodeGraph, ParseError> {
    let name = qualify_with_namespace(
        required_attr(node, path, "name")?,
        node.attribute("namespace"),
    );
    let nodedef = node.attribute("nodedef").map(str::to_owned);
    let local_prefix = node.attribute("fileprefix").unwrap_or(parent_prefix);
    let colorspace = node.attribute("colorspace").or(parent_colorspace);
    let inputs = parse_inputs(node, local_prefix, colorspace, path)?;
    let outputs = parse_outputs(node, path)?;

    let mut nodes = Vec::new();
    let mut tokens = Vec::new();
    for child in node.children().filter(Node::is_element) {
        let tag = child.tag_name().name();
        if matches!(tag, "input" | "output") {
            continue;
        }
        if tag == "token"
            && let Some(t) = parse_token(&child, path)?
        {
            tokens.push(t);
            continue;
        }
        if matches!(tag, "surfacematerial" | "volumematerial" | "material") {
            continue;
        }
        nodes.push(parse_node_use(&child, local_prefix, colorspace, path)?);
    }

    Ok(RawNodeGraph {
        name,
        nodedef,
        target: node.attribute("target").map(str::to_owned),
        nodes,
        inputs,
        outputs,
        tokens,
    })
}

fn parse_token(node: &Node, path: &Path) -> Result<Option<RawToken>, ParseError> {
    let name = required_attr(node, path, "name")?.to_string();
    let ty = MtlxType::parse(required_attr(node, path, "type")?);
    if node.attribute("value").is_some() && node.attribute("interfacename").is_some() {
        return Err(ParseError::Structure {
            message: format!(
                "token `{}` cannot specify both value and interfacename",
                name
            ),
            path: path.to_path_buf(),
        });
    }
    let value = node.attribute("value").map(str::to_owned);
    let interface = node.attribute("interfacename").map(str::to_owned);
    Ok(Some(RawToken {
        name,
        ty,
        value,
        interface,
    }))
}

fn collect_nested_materials<'a>(
    node: Node<'a, '_>,
    parent_prefix: &str,
    parent_colorspace: Option<&str>,
    parent_nodegraph: &str,
    out: &mut Vec<RawMaterial>,
    path: &Path,
) -> Result<(), ParseError> {
    let local_prefix = node.attribute("fileprefix").unwrap_or(parent_prefix);
    let colorspace = node.attribute("colorspace").or(parent_colorspace);
    for child in node.children().filter(Node::is_element) {
        let tag = child.tag_name().name();
        match tag {
            "surfacematerial" | "volumematerial" | "material" => {
                let category = tag.to_string();
                let name = required_attr(&child, path, "name")?.to_string();
                let material_prefix = child.attribute("fileprefix").unwrap_or(local_prefix);
                let parent = if parent_nodegraph.is_empty() {
                    None
                } else {
                    Some(parent_nodegraph.to_string())
                };
                out.push(RawMaterial {
                    name,
                    category,
                    inputs: parse_inputs(&child, material_prefix, colorspace, path)?,
                    parent_nodegraph: parent,
                });
            }
            "nodegraph" => {
                let inner_name = required_attr(&child, path, "name")?;
                collect_nested_materials(child, local_prefix, colorspace, inner_name, out, path)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn parse_node_use(
    node: &Node,
    parent_prefix: &str,
    parent_colorspace: Option<&str>,
    path: &Path,
) -> Result<RawNodeUse, ParseError> {
    let category = node.tag_name().name().to_string();
    let name = required_attr(node, path, "name")?.to_string();
    let ty = MtlxType::parse(required_attr(node, path, "type")?);
    let local_prefix = node.attribute("fileprefix").unwrap_or(parent_prefix);
    let colorspace = node.attribute("colorspace").or(parent_colorspace);
    Ok(RawNodeUse {
        name,
        category,
        ty,
        inputs: parse_inputs(node, local_prefix, colorspace, path)?,
        tokens: parse_tokens(node, path)?,
        outputs: parse_outputs(node, path)?,
        version: node.attribute("version").map(str::to_owned),
        nodedef: node.attribute("nodedef").map(str::to_owned),
    })
}

fn parse_inputs(
    parent: &Node,
    parent_prefix: &str,
    parent_colorspace: Option<&str>,
    path: &Path,
) -> Result<Vec<RawInput>, ParseError> {
    let mut inputs = Vec::new();
    for child in parent
        .children()
        .filter(|c| c.is_element() && c.tag_name().name() == "input")
    {
        inputs.push(parse_input(&child, parent_prefix, parent_colorspace, path)?);
    }
    Ok(inputs)
}

fn parse_outputs(parent: &Node, path: &Path) -> Result<Vec<RawOutput>, ParseError> {
    let mut outputs = Vec::new();
    for child in parent
        .children()
        .filter(|c| c.is_element() && c.tag_name().name() == "output")
    {
        outputs.push(parse_output(&child, path)?);
    }
    Ok(outputs)
}

fn parse_tokens(parent: &Node, path: &Path) -> Result<Vec<RawToken>, ParseError> {
    let mut tokens = Vec::new();
    for child in parent
        .children()
        .filter(|c| c.is_element() && c.tag_name().name() == "token")
    {
        if let Some(token) = parse_token(&child, path)? {
            tokens.push(token);
        }
    }
    Ok(tokens)
}

fn parse_input(
    node: &Node,
    parent_prefix: &str,
    parent_colorspace: Option<&str>,
    path: &Path,
) -> Result<RawInput, ParseError> {
    let name = required_attr(node, path, "name")?.to_string();
    let ty = MtlxType::parse(required_attr(node, path, "type")?);
    let local_prefix = node.attribute("fileprefix").unwrap_or(parent_prefix);
    let uniform = optional_bool_attr(node, path, "uniform")?;
    validate_input_binding_attrs(node, path, &name, &ty, uniform)?;
    let binding = if let Some(g) = node.attribute("nodegraph") {
        InputBinding::NodeGraphRef {
            nodegraph: g.to_string(),
            output: node.attribute("output").map(str::to_owned),
        }
    } else if let Some(n) = node.attribute("nodename") {
        InputBinding::NodeRef {
            nodename: n.to_string(),
            output: node.attribute("output").map(str::to_owned),
        }
    } else if let Some(i) = node.attribute("interfacename") {
        InputBinding::InterfaceName(i.to_string())
    } else if let Some(g) = node.attribute("defaultgeomprop") {
        InputBinding::DefaultGeomProp(g.to_string())
    } else if let Some(v) = node.attribute("value") {
        let value = if matches!(ty, MtlxType::Filename)
            && !local_prefix.is_empty()
            && !is_absolute_filename(v)
        {
            format!("{}{}", local_prefix, v)
        } else {
            v.to_string()
        };
        InputBinding::Value(value)
    } else {
        InputBinding::Empty
    };

    Ok(RawInput {
        name,
        ty,
        binding,
        colorspace: node
            .attribute("colorspace")
            .or(parent_colorspace)
            .map(str::to_owned),
        unit: node.attribute("unit").map(str::to_owned),
        unittype: node.attribute("unittype").map(str::to_owned),
        uniform,
    })
}

fn is_absolute_filename(s: &str) -> bool {
    s.starts_with('/') || s.starts_with('\\') || (s.len() >= 2 && s.chars().nth(1) == Some(':'))
}

fn validate_input_binding_attrs(
    node: &Node,
    path: &Path,
    name: &str,
    ty: &MtlxType,
    uniform: bool,
) -> Result<(), ParseError> {
    let binding_count = [
        "nodegraph",
        "nodename",
        "interfacename",
        "defaultgeomprop",
        "value",
    ]
    .into_iter()
    .filter(|attr| node.attribute(*attr).is_some())
    .count();
    if binding_count > 1 {
        return Err(ParseError::Structure {
            message: format!(
                "input `{}` cannot specify multiple value or connection attributes",
                name
            ),
            path: path.to_path_buf(),
        });
    }
    if node.attribute("output").is_some()
        && node.attribute("nodegraph").is_none()
        && node.attribute("nodename").is_none()
    {
        return Err(ParseError::Structure {
            message: format!(
                "input `{}` output attribute requires nodename or nodegraph",
                name
            ),
            path: path.to_path_buf(),
        });
    }
    if node.attribute("defaultgeomprop").is_some() {
        if uniform {
            return Err(ParseError::Structure {
                message: format!(
                    "input `{}` cannot specify defaultgeomprop when uniform=\"true\"",
                    name
                ),
                path: path.to_path_buf(),
            });
        }
        if !matches!(ty, MtlxType::Vector2 | MtlxType::Vector3) {
            return Err(ParseError::Structure {
                message: format!(
                    "input `{}` defaultgeomprop requires vector2 or vector3 type",
                    name
                ),
                path: path.to_path_buf(),
            });
        }
    }
    Ok(())
}

fn parse_output(node: &Node, path: &Path) -> Result<RawOutput, ParseError> {
    let name = required_attr(node, path, "name")?.to_string();
    let ty = MtlxType::parse(required_attr(node, path, "type")?);
    validate_output_binding_attrs(node, path, &name)?;
    let binding = if let Some(g) = node.attribute("nodegraph") {
        InputBinding::NodeGraphRef {
            nodegraph: g.to_string(),
            output: node.attribute("output").map(str::to_owned),
        }
    } else if let Some(n) = node.attribute("nodename") {
        InputBinding::NodeRef {
            nodename: n.to_string(),
            output: node.attribute("output").map(str::to_owned),
        }
    } else if let Some(i) = node.attribute("interfacename") {
        InputBinding::InterfaceName(i.to_string())
    } else {
        InputBinding::Empty
    };
    Ok(RawOutput {
        name,
        ty,
        binding,
        default: node.attribute("default").map(str::to_owned),
        default_input: node.attribute("defaultinput").map(str::to_owned),
    })
}

fn validate_output_binding_attrs(node: &Node, path: &Path, name: &str) -> Result<(), ParseError> {
    let binding_count = ["nodegraph", "nodename", "interfacename"]
        .into_iter()
        .filter(|attr| node.attribute(*attr).is_some())
        .count();
    if binding_count > 1 {
        return Err(ParseError::Structure {
            message: format!(
                "output `{}` cannot specify multiple connection attributes",
                name
            ),
            path: path.to_path_buf(),
        });
    }
    if node.attribute("output").is_some()
        && node.attribute("nodegraph").is_none()
        && node.attribute("nodename").is_none()
    {
        return Err(ParseError::Structure {
            message: format!(
                "output `{}` output attribute requires nodename or nodegraph",
                name
            ),
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"<?xml version="1.0"?>
<materialx version="1.39" colorspace="lin_rec709">
  <typedef name="boolean"/>
  <typedef name="color3" semantic="color"/>
  <geompropdef name="UV0" type="vector2" geomprop="texcoord" index="0"/>
  <nodedef name="ND_constant_color3" node="constant" version="1.0" isdefaultversion="true">
    <input name="value" type="color3" value="0.5, 0.5, 0.5"/>
    <output name="out" type="color3"/>
  </nodedef>
  <surfacematerial name="MyMat">
    <input name="surfaceshader" type="surfaceshader" nodename="srf"/>
  </surfacematerial>
  <nodegraph name="NG_test">
    <input name="base" type="color3" value="1.0, 0.5, 0.25"/>
    <constant name="srf" type="color3">
      <input name="value" type="color3" interfacename="base"/>
    </constant>
    <output name="out" type="color3" nodename="srf"/>
  </nodegraph>
</materialx>"#;

    #[test]
    fn parser_extracts_top_level_pieces() {
        let doc = parse_str(SAMPLE, Path::new("inline.mtlx")).unwrap();
        assert_eq!(doc.version, (1, 39));
        assert_eq!(doc.colorspace.as_deref(), Some("lin_rec709"));
        assert_eq!(doc.typedefs.len(), 2);
        assert_eq!(doc.geompropdefs.len(), 1);
        assert_eq!(doc.nodedefs.len(), 1);
        assert_eq!(doc.materials.len(), 1);
        assert_eq!(doc.nodegraphs.len(), 1);
    }

    #[test]
    fn input_bindings_distinguish_kinds() {
        let doc = parse_str(SAMPLE, Path::new("inline.mtlx")).unwrap();
        let ng = &doc.nodegraphs[0];
        assert_eq!(ng.nodes.len(), 1);
        let constant = &ng.nodes[0];
        assert_eq!(constant.category, "constant");
        match &constant.inputs[0].binding {
            InputBinding::InterfaceName(name) => assert_eq!(name, "base"),
            _ => panic!("expected interface binding"),
        }
        match &doc.materials[0].inputs[0].binding {
            InputBinding::NodeRef { nodename, .. } => assert_eq!(nodename, "srf"),
            _ => panic!("expected node ref binding"),
        }
    }

    #[test]
    fn colorspace_inherits_from_document_and_node_scope() {
        let src = r#"
<materialx version="1.39" colorspace="srgb_texture">
  <nodegraph name="NG_scope" colorspace="lin_rec709">
    <constant name="c" type="color3">
      <input name="value" type="color3" value="0.5,0.5,0.5"/>
    </constant>
    <constant name="d" type="color3" colorspace="none">
      <input name="value" type="color3" value="0.5,0.5,0.5"/>
    </constant>
  </nodegraph>
  <constant name="root_c" type="color3">
    <input name="value" type="color3" value="0.5,0.5,0.5"/>
  </constant>
</materialx>"#;
        let doc = parse_str(src, Path::new("inline.mtlx")).unwrap();
        let ng = &doc.nodegraphs[0];
        assert_eq!(
            ng.nodes[0].inputs[0].colorspace.as_deref(),
            Some("lin_rec709")
        );
        assert_eq!(ng.nodes[1].inputs[0].colorspace.as_deref(), Some("none"));
        assert_eq!(
            doc.root_nodes[0].inputs[0].colorspace.as_deref(),
            Some("srgb_texture")
        );
    }

    #[test]
    fn element_namespace_qualifies_nodedef_and_nodegraph_names() {
        let src = r#"
<materialx version="1.39">
  <nodedef name="ND_myshader" node="myshader" namespace="myns">
    <output name="out" type="surfaceshader"/>
  </nodedef>
  <nodegraph name="NG_myshader" nodedef="myns:ND_myshader" namespace="myns">
    <output name="out" type="surfaceshader"/>
  </nodegraph>
</materialx>"#;
        let doc = parse_str(src, Path::new("inline.mtlx")).unwrap();
        assert_eq!(doc.nodedefs[0].name, "myns:ND_myshader");
        assert_eq!(doc.nodedefs[0].node, "myns:myshader");
        assert_eq!(doc.nodegraphs[0].name, "myns:NG_myshader");
        assert_eq!(
            doc.nodegraphs[0].nodedef.as_deref(),
            Some("myns:ND_myshader")
        );
    }

    #[test]
    fn xinclude_children_inherit_root_namespace_and_colorspace() {
        let dir = std::env::temp_dir().join(format!(
            "toy_path_tracing_mtlx_xinclude_inherit_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp mtlx dir");
        let child_path = dir.join("child.mtlx");
        std::fs::write(
            &child_path,
            r#"<materialx version="1.39">
  <nodedef name="ND_child" node="child">
    <input name="value" type="color3" value="0.5,0.5,0.5"/>
    <output name="out" type="color3"/>
  </nodedef>
  <nodegraph name="NG_child">
    <constant name="c" type="color3">
      <input name="value" type="color3" value="0.1,0.2,0.3"/>
    </constant>
    <output name="out" type="color3" nodename="c"/>
  </nodegraph>
</materialx>"#,
        )
        .expect("write child mtlx");
        let parent_path = dir.join("parent.mtlx");
        std::fs::write(
            &parent_path,
            r#"<materialx xmlns:xi="http://www.w3.org/2001/XInclude" version="1.39" namespace="parentns" colorspace="lin_rec709">
  <xi:include href="child.mtlx"/>
</materialx>"#,
        )
        .expect("write parent mtlx");

        let doc = parse_document(&parent_path).expect("parse parent with include");
        assert_eq!(doc.nodedefs[0].name, "parentns:ND_child");
        assert_eq!(doc.nodedefs[0].node, "parentns:child");
        assert_eq!(
            doc.nodedefs[0].inputs[0].colorspace.as_deref(),
            Some("lin_rec709")
        );
        assert_eq!(doc.nodegraphs[0].name, "parentns:NG_child");
        assert_eq!(
            doc.nodegraphs[0].nodes[0].inputs[0].colorspace.as_deref(),
            Some("lin_rec709")
        );
        match &doc.nodegraphs[0].outputs[0].binding {
            InputBinding::NodeRef { nodename, .. } => assert_eq!(nodename, "parentns:c"),
            other => panic!("expected node ref, got {:?}", other),
        }
    }

    #[test]
    fn xinclude_children_precede_parent_content_and_inherit_fileprefix() {
        let dir = std::env::temp_dir().join(format!(
            "toy_path_tracing_mtlx_xinclude_order_prefix_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp mtlx dir");
        let child_path = dir.join("child.mtlx");
        std::fs::write(
            &child_path,
            r#"<materialx version="1.39">
  <nodedef name="ND_dup" node="childcat">
    <output name="out" type="color3"/>
  </nodedef>
  <nodegraph name="NG_child">
    <image name="img" type="color3">
      <input name="file" type="filename" value="albedo.png"/>
    </image>
    <image name="local_img" type="color3" fileprefix="local/">
      <input name="file" type="filename" value="normal.png"/>
    </image>
  </nodegraph>
</materialx>"#,
        )
        .expect("write child mtlx");
        let parent_path = dir.join("parent.mtlx");
        std::fs::write(
            &parent_path,
            r#"<materialx xmlns:xi="http://www.w3.org/2001/XInclude" version="1.39" fileprefix="textures/">
  <xi:include href="child.mtlx"/>
  <nodedef name="ND_dup" node="parentcat">
    <output name="out" type="color3"/>
  </nodedef>
</materialx>"#,
        )
        .expect("write parent mtlx");

        let doc = parse_document(&parent_path).expect("parse parent with include");
        assert_eq!(doc.nodedefs[0].node, "childcat");
        assert_eq!(doc.nodedefs[1].node, "parentcat");
        match &doc.nodegraphs[0].nodes[0].inputs[0].binding {
            InputBinding::Value(value) => assert_eq!(value, "textures/albedo.png"),
            other => panic!("expected file value, got {:?}", other),
        }
        match &doc.nodegraphs[0].nodes[1].inputs[0].binding {
            InputBinding::Value(value) => assert_eq!(value, "local/normal.png"),
            other => panic!("expected file value, got {:?}", other),
        }
    }

    #[test]
    fn malformed_version_is_an_error() {
        let src = r#"<materialx version="not-a-version"/>"#;
        let err = parse_str(src, Path::new("bad.mtlx")).expect_err("version must be rejected");
        assert!(err.to_string().contains("invalid materialx version"));
    }

    #[test]
    fn missing_required_input_attribute_is_an_error() {
        let src = r#"
<materialx version="1.39">
  <nodegraph name="NG_bad">
    <constant name="c" type="float">
      <input name="value" value="1.0"/>
    </constant>
  </nodegraph>
</materialx>"#;
        let err = parse_str(src, Path::new("bad.mtlx")).expect_err("input type must be rejected");
        assert!(err.to_string().contains("missing required `type`"));
    }

    #[test]
    fn token_type_is_required() {
        let src = r#"
<materialx version="1.39">
  <nodedef name="ND_bad" node="bad">
    <token name="tex" value="a"/>
    <output name="out" type="color3"/>
  </nodedef>
</materialx>"#;
        let err = parse_str(src, Path::new("bad.mtlx")).expect_err("token type must be rejected");
        assert!(err.to_string().contains("missing required `type`"));
    }

    #[test]
    fn token_cannot_bind_value_and_interface() {
        let src = r#"
<materialx version="1.39">
  <nodegraph name="NG_bad">
    <token name="tex" type="string" value="a" interfacename="b"/>
  </nodegraph>
</materialx>"#;
        let err =
            parse_str(src, Path::new("bad.mtlx")).expect_err("ambiguous token must be rejected");
        assert!(err.to_string().contains("cannot specify both"));
    }

    #[test]
    fn malformed_geompropdef_index_is_an_error() {
        let src = r#"
<materialx version="1.39">
  <geompropdef name="UV0" type="vector2" index="not-an-index"/>
</materialx>"#;
        let err = parse_str(src, Path::new("bad.mtlx")).expect_err("index must be rejected");
        assert!(err.to_string().contains("invalid index"));
    }

    #[test]
    fn invalid_geompropdef_semantics_are_errors() {
        let cases = [
            (
                r#"<geompropdef name="bad" type="vector2" geomprop="position"/>"#,
                "does not match required type",
            ),
            (
                r#"<geompropdef name="bad" type="vector2" geomprop="texcoord" space="world"/>"#,
                "cannot specify space",
            ),
            (
                r#"<geompropdef name="bad" type="vector3" geomprop="unknown"/>"#,
                "invalid geomprop",
            ),
            (
                r#"<geompropdef name="bad" type="string"/>"#,
                "must declare uniform",
            ),
            (
                r#"<geompropdef name="bad" type="floatarray"/>"#,
                "must be non-array",
            ),
            (
                r#"<geompropdef name="bad" type="float" uniform="true" geomprop="position"/>"#,
                "cannot specify geomprop",
            ),
        ];
        for (geompropdef, message) in cases {
            let src = format!(
                r#"<materialx version="1.39">
  {geompropdef}
</materialx>"#
            );
            let err = parse_str(&src, Path::new("bad.mtlx")).expect_err(message);
            assert!(err.to_string().contains(message), "{}", err);
        }
    }

    #[test]
    fn ambiguous_input_binding_attributes_are_errors() {
        let cases = [
            (
                r#"<input name="in" type="float" value="1" nodename="x"/>"#,
                "multiple value or connection",
            ),
            (
                r#"<input name="in" type="vector2" value="0,0" defaultgeomprop="UV0"/>"#,
                "multiple value or connection",
            ),
            (
                r#"<input name="in" type="float" value="1" output="out"/>"#,
                "output attribute requires",
            ),
            (
                r#"<input name="in" type="vector2" defaultgeomprop="UV0" uniform="true"/>"#,
                "cannot specify defaultgeomprop",
            ),
            (
                r#"<input name="in" type="float" defaultgeomprop="UV0"/>"#,
                "requires vector2 or vector3",
            ),
        ];
        for (input, message) in cases {
            let src = format!(
                r#"<materialx version="1.39">
  <nodegraph name="NG_bad">
    <constant name="c" type="float">
      {input}
    </constant>
  </nodegraph>
</materialx>"#
            );
            let err = parse_str(&src, Path::new("bad.mtlx")).expect_err(message);
            assert!(err.to_string().contains(message), "{}", err);
        }
    }

    #[test]
    fn ambiguous_output_binding_attributes_are_errors() {
        let cases = [
            (
                r#"<output name="out" type="float" nodename="a" nodegraph="NG"/>"#,
                "multiple connection",
            ),
            (
                r#"<output name="out" type="float" output="x"/>"#,
                "output attribute requires",
            ),
        ];
        for (output, message) in cases {
            let src = format!(
                r#"<materialx version="1.39">
  <nodegraph name="NG_bad">
    {output}
  </nodegraph>
</materialx>"#
            );
            let err = parse_str(&src, Path::new("bad.mtlx")).expect_err(message);
            assert!(err.to_string().contains(message), "{}", err);
        }
    }

    #[test]
    fn malformed_boolean_attribute_is_an_error() {
        let src = r#"
<materialx version="1.39">
  <nodedef name="ND_bad" node="bad" isdefaultversion="maybe">
    <output name="out" type="float"/>
  </nodedef>
</materialx>"#;
        let err = parse_str(src, Path::new("bad.mtlx")).expect_err("boolean must be rejected");
        assert!(err.to_string().contains("must be boolean"));
    }

    #[test]
    fn xinclude_missing_href_is_an_error() {
        let dir = std::env::temp_dir().join(format!(
            "toy_path_tracing_mtlx_xinclude_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp mtlx dir");
        let path = dir.join("bad.mtlx");
        std::fs::write(
            &path,
            r#"<materialx xmlns:xi="http://www.w3.org/2001/XInclude" version="1.39">
  <xi:include/>
</materialx>"#,
        )
        .expect("write temp mtlx");

        let err = parse_document(&path).expect_err("XInclude href must be rejected");
        assert!(err.to_string().contains("missing required `href`"));
    }
}
