//! Writing a [`UsdLayer`] as a `.usdc` crate file.
//!
//! The reverse of [`super::crate_read`], and the tables come out in the same
//! six pieces they go in as. Three things make writing the harder direction:
//!
//! - **Values live in the file, not in the tables.** A field holds a
//!   [`ValueRep`](super::crate_read) — 64 bits that either carry the value or
//!   say where it is — so the value area has to be laid down first and the
//!   offsets folded back into the fields.
//! - **The path tree becomes a walk.** Each entry's `jump` is the size of its
//!   own subtree, which is not known until the subtree has been written, so
//!   the entry is emitted blank and patched afterwards.
//! - **Everything is interned.** Names, values and whole field sets are shared,
//!   and a writer that does not share them produces a file that reads correctly
//!   and is several times larger than it should be.
//!
//! What comes out is checked three ways: this crate reads it back, the result
//! is compared against the layer that went in, and — where OpenUSD is
//! installed — `usdcat` is asked to convert it.

use std::collections::HashMap;

use super::ints::{encode_u32, encode_u64, encodable_u64, write_u32_array};
use super::lz4;
use super::parse::{Specifier, UsdLayer, UsdPrim, UsdProperty, UsdVariantSet};
use super::value::UsdValue;

/// The version this writes. The layout has been stable across 0.8.x, and
/// writing the version this crate was verified against is more honest than
/// claiming the newest.
const VERSION: (u8, u8, u8) = (0, 8, 0);
const BOOTSTRAP: usize = 88;

/// Spec types, confirmed against files `usdcat` wrote.
mod spec {
    pub const ATTRIBUTE: u32 = 1;
    pub const PRIM: u32 = 6;
    pub const PSEUDO_ROOT: u32 = 7;
    pub const RELATIONSHIP: u32 = 8;
    pub const VARIANT: u32 = 10;
    pub const VARIANT_SET: u32 = 11;
}

/// The value types, matching [`super::crate_read`]'s reader.
mod kind {
    pub const BOOL: u8 = 1;
    pub const UCHAR: u8 = 2;
    pub const INT: u8 = 3;
    pub const UINT: u8 = 4;
    pub const INT64: u8 = 5;
    pub const UINT64: u8 = 6;
    pub const HALF: u8 = 7;
    pub const FLOAT: u8 = 8;
    pub const DOUBLE: u8 = 9;
    pub const STRING: u8 = 10;
    pub const TOKEN: u8 = 11;
    pub const ASSET_PATH: u8 = 12;
    pub const MATRIX2D: u8 = 13;
    pub const MATRIX3D: u8 = 14;
    pub const MATRIX4D: u8 = 15;
    pub const QUATD: u8 = 16;
    pub const QUATF: u8 = 17;
    pub const QUATH: u8 = 18;
    pub const VEC2D: u8 = 19;
    pub const VEC2F: u8 = 20;
    pub const VEC2H: u8 = 21;
    pub const VEC2I: u8 = 22;
    pub const VEC3D: u8 = 23;
    pub const VEC3F: u8 = 24;
    pub const VEC3H: u8 = 25;
    pub const VEC3I: u8 = 26;
    pub const VEC4D: u8 = 27;
    pub const VEC4F: u8 = 28;
    pub const VEC4H: u8 = 29;
    pub const VEC4I: u8 = 30;
    /// A nested map of named values — `clips`, `customData`.
    pub const DICTIONARY: u8 = 31;
    /// `apiSchemas` — a list operation, not a plain vector.
    pub const TOKEN_LIST_OP: u8 = 32;
    /// `foo = None`, an explicit block.
    pub const VALUE_BLOCK: u8 = 51;
    /// A frame number. Writing one is what makes USD call a crate 0.9.0.
    pub const TIME_CODE: u8 = 56;
    /// A collection's membership expression, indexed like a string but always
    /// written out of line. A crate holding one is version 0.10.0.
    pub const PATH_EXPRESSION: u8 = 57;
    /// `variantSets`.
    pub const STRING_LIST_OP: u8 = 33;
    /// A relationship's targets, and `inherits` and `specializes`.
    pub const PATH_LIST_OP: u8 = 34;
    pub const REFERENCE_LIST_OP: u8 = 35;
    pub const TOKEN_VECTOR: u8 = 41;
    pub const SPECIFIER: u8 = 42;
    pub const VARIABILITY: u8 = 44;
    /// Which variant of each set is selected.
    pub const VARIANT_SELECTION_MAP: u8 = 45;
    pub const TIME_SAMPLES: u8 = 46;
    /// `subLayers`.
    pub const LAYER_OFFSET_VECTOR: u8 = 49;
    pub const STRING_VECTOR: u8 = 50;
    pub const PAYLOAD_LIST_OP: u8 = 55;
}

const IS_ARRAY: u64 = 1 << 63;
const IS_INLINED: u64 = 1 << 62;

/// A USD type name as the crate type it is stored as, and whether it is an
/// array.
///
/// USD's *role* types all share an underlying layout — `point3f`, `normal3f`,
/// `color3f` and `vector3f` are each three floats — and the role survives on
/// its own as the `typeName` field, so nothing is lost by collapsing them here.
fn kind_of(type_name: &str) -> Option<(u8, bool)> {
    let (base, array) = match type_name.strip_suffix("[]") {
        Some(base) => (base, true),
        None => (type_name, false),
    };
    let kind = match base {
        "bool" => kind::BOOL,
        "uchar" => kind::UCHAR,
        "int" => kind::INT,
        "uint" => kind::UINT,
        "int64" => kind::INT64,
        "uint64" => kind::UINT64,
        "half" => kind::HALF,
        "float" => kind::FLOAT,
        "double" => kind::DOUBLE,
        "timecode" => kind::TIME_CODE,
        "string" => kind::STRING,
        "token" => kind::TOKEN,
        "asset" => kind::ASSET_PATH,
        "pathExpression" => kind::PATH_EXPRESSION,
        "matrix2d" => kind::MATRIX2D,
        "matrix3d" => kind::MATRIX3D,
        "matrix4d" | "frame4d" => kind::MATRIX4D,
        "quatd" => kind::QUATD,
        "quatf" => kind::QUATF,
        "quath" => kind::QUATH,
        "double2" | "point2d" | "normal2d" | "vector2d" | "texCoord2d" => kind::VEC2D,
        "float2" | "point2f" | "normal2f" | "vector2f" | "texCoord2f" => kind::VEC2F,
        "half2" | "texCoord2h" => kind::VEC2H,
        "int2" => kind::VEC2I,
        "double3" | "point3d" | "normal3d" | "vector3d" | "color3d" | "texCoord3d" => kind::VEC3D,
        "float3" | "point3f" | "normal3f" | "vector3f" | "color3f" | "texCoord3f" => kind::VEC3F,
        "half3" | "point3h" | "normal3h" | "vector3h" | "color3h" | "texCoord3h" => kind::VEC3H,
        "int3" => kind::VEC3I,
        "double4" | "color4d" => kind::VEC4D,
        "float4" | "color4f" => kind::VEC4F,
        "half4" | "color4h" => kind::VEC4H,
        "int4" => kind::VEC4I,
        _ => return None,
    };
    Some((kind, array))
}

/// How many numbers one element of a type has, and how wide each is.
fn shape(kind: u8) -> Option<(usize, usize)> {
    Some(match kind {
        kind::BOOL | kind::UCHAR => (1, 1),
        kind::INT | kind::UINT | kind::FLOAT => (1, 4),
        kind::INT64 | kind::UINT64 | kind::DOUBLE | kind::TIME_CODE => (1, 8),
        kind::HALF => (1, 2),
        kind::VEC2H => (2, 2),
        kind::VEC3H => (3, 2),
        kind::VEC4H | kind::QUATH => (4, 2),
        kind::VEC2F => (2, 4),
        kind::VEC3F => (3, 4),
        kind::VEC4F | kind::QUATF => (4, 4),
        kind::VEC2I => (2, 4),
        kind::VEC3I => (3, 4),
        kind::VEC4I => (4, 4),
        kind::VEC2D => (2, 8),
        kind::VEC3D => (3, 8),
        kind::VEC4D | kind::QUATD => (4, 8),
        kind::MATRIX2D => (4, 8),
        kind::MATRIX3D => (9, 8),
        kind::MATRIX4D => (16, 8),
        _ => return None,
    })
}

fn is_float_kind(kind: u8) -> bool {
    matches!(
        kind,
        kind::HALF
            | kind::FLOAT
            | kind::DOUBLE
            | kind::VEC2H
            | kind::VEC3H
            | kind::VEC4H
            | kind::QUATH
            | kind::VEC2F
            | kind::VEC3F
            | kind::VEC4F
            | kind::QUATF
            | kind::VEC2D
            | kind::VEC3D
            | kind::VEC4D
            | kind::QUATD
            | kind::MATRIX2D
            | kind::MATRIX3D
            | kind::MATRIX4D
            | kind::TIME_CODE
    )
}

fn is_quat(kind: u8) -> bool {
    matches!(kind, kind::QUATD | kind::QUATF | kind::QUATH)
}

/// f32 as half-precision, for the `half` types.
fn f32_to_half(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;

    if exponent == 0xFF {
        // Infinity, or a NaN that must stay one.
        let payload = if mantissa != 0 { 0x200 } else { 0 };
        return sign | 0x7C00 | payload;
    }
    let unbiased = exponent - 127 + 15;
    if unbiased >= 0x1F {
        return sign | 0x7C00; // Overflows to infinity.
    }
    if unbiased <= 0 {
        // Subnormal, or too small to represent at all.
        if unbiased < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x80_0000;
        let shift = (14 - unbiased) as u32;
        let kept = (mantissa >> shift) as u16;
        return sign | (kept + round_up(mantissa, shift, kept) as u16);
    }
    let kept = ((unbiased as u16) << 10) | (mantissa >> 13) as u16;
    // Carrying out of the mantissa into the exponent is correct, and so is
    // carrying all the way to infinity from the largest finite half.
    sign | (kept + round_up(mantissa, 13, kept) as u16)
}

/// Whether dropping the low `shift` bits of `mantissa` should round up.
///
/// Round half to even, which is what IEEE requires and what every other
/// implementation does — truncating instead is a bias toward zero on every
/// single value, and `half` is what USD stores normals and colours as.
fn round_up(mantissa: u32, shift: u32, kept: u16) -> bool {
    let dropped = mantissa & ((1u32 << shift) - 1);
    let halfway = 1u32 << (shift - 1);
    dropped > halfway || (dropped == halfway && kept & 1 == 1)
}

/// The tables of a crate file, under construction.
#[derive(Default)]
struct Writer {
    tokens: Vec<String>,
    token_index: HashMap<String, u32>,
    strings: Vec<u32>,
    string_index: HashMap<u32, u32>,
    /// `(token index of the field name, the value's rep word)`.
    fields: Vec<(u32, u64)>,
    field_index: HashMap<(u32, u64), u32>,
    /// Runs of field indices, each ended by `-1`.
    field_sets: Vec<i32>,
    set_index: HashMap<Vec<i32>, u32>,
    /// The path table's walk.
    path_elements: Vec<i32>,
    path_jumps: Vec<i32>,
    /// `(path index, field set index, spec type)`.
    specs: Vec<(u32, u32, u32)>,
    /// The value area, which begins right after the bootstrap.
    data: Vec<u8>,
    /// Where each absolute path landed in the walk.
    slots: HashMap<String, u32>,
    /// The slot standing for the empty path — the one an arc with no prim
    /// named points at.
    ///
    /// Slot zero, which the walk never writes: USD reads an unwritten slot as
    /// the default-constructed path, and that is exactly empty. It has to be
    /// *inside* the numbering rather than one past the end, because USD checks
    /// that `numPaths` is no larger than the highest index actually used.
    empty_path_slot: u32,
    /// The lowest crate version that can hold what has been written. Most of
    /// the format is 0.8.0; a few value types are newer, and USD refuses a
    /// file whose version is older than the types inside it.
    needs_version: (u8, u8, u8),
    /// Values already written, by their bytes. Two attributes holding the same
    /// thing point at one copy — which for a scene of near-identical prims is
    /// the difference between a few kilobytes and a few dozen.
    interned: HashMap<Vec<u8>, u64>,
}

impl Writer {
    fn token(&mut self, text: &str) -> u32 {
        if let Some(found) = self.token_index.get(text) {
            return *found;
        }
        let index = self.tokens.len() as u32;
        self.tokens.push(text.to_string());
        self.token_index.insert(text.to_string(), index);
        index
    }

    /// A quoted string, which is indexed one level deeper than a token.
    fn string(&mut self, text: &str) -> u32 {
        let token = self.token(text);
        if let Some(found) = self.string_index.get(&token) {
            return *found;
        }
        let index = self.strings.len() as u32;
        self.strings.push(token);
        self.string_index.insert(token, index);
        index
    }

    fn field(&mut self, name: &str, rep: u64) -> u32 {
        let name = self.token(name);
        if let Some(found) = self.field_index.get(&(name, rep)) {
            return *found;
        }
        let index = self.fields.len() as u32;
        self.fields.push((name, rep));
        self.field_index.insert((name, rep), index);
        index
    }

    /// Intern a whole set of fields. Two specs with identical fields — which
    /// is most of a large asset — share one run.
    fn field_set(&mut self, mut members: Vec<i32>) -> u32 {
        members.push(-1);
        if let Some(found) = self.set_index.get(&members) {
            return *found;
        }
        let index = self.field_sets.len() as u32;
        self.field_sets.extend_from_slice(&members);
        self.set_index.insert(members, index);
        index
    }

    /// The path table slot for an absolute path.
    ///
    /// An arc with no prim path points at the *empty* path, which is a real
    /// slot in the table that the walk never writes — USD reads an unwritten
    /// slot as the default-constructed path, which is exactly empty.
    fn path_slot(&mut self, path: &str) -> u32 {
        if path.is_empty() {
            return self.empty_path_slot;
        }
        self.slots.get(path).copied().unwrap_or(self.empty_path_slot)
    }

    /// Note that the file needs at least this version to be read.
    fn needs(&mut self, major: u8, minor: u8, patch: u8) {
        self.needs_version = self.needs_version.max((major, minor, patch));
    }

    /// Append a value, or point at an identical one already written.
    fn intern(&mut self, bytes: Vec<u8>) -> u64 {
        if let Some(found) = self.interned.get(&bytes) {
            return *found;
        }
        while !self.data.len().is_multiple_of(8) {
            self.data.push(0);
        }
        let at = (BOOTSTRAP + self.data.len()) as u64;
        self.data.extend_from_slice(&bytes);
        self.interned.insert(bytes, at);
        at
    }
}

/// A node of the path tree, before it becomes a walk.
///
/// The tree has to exist in full before a single entry is written, because an
/// entry's jump is the size of its own subtree and because a relationship may
/// name a path that appears later — or that has no prim at all, which USD
/// permits and preserves.
#[derive(Default)]
struct Node {
    name: String,
    /// Whether this is a property of the prim above it rather than a child.
    property: bool,
    children: Vec<Node>,
}

impl Node {
    fn child_mut(&mut self, name: &str, property: bool) -> &mut Node {
        if let Some(i) = self
            .children
            .iter()
            .position(|c| c.name == name && c.property == property)
        {
            return &mut self.children[i];
        }
        self.children.push(Node {
            name: name.to_string(),
            property,
            children: Vec::new(),
        });
        self.children.last_mut().unwrap()
    }

    /// Add an absolute path, creating whatever is missing along the way.
    fn insert_path(&mut self, path: &str) {
        let Some(rest) = path.strip_prefix('/') else {
            return;
        };
        // A property is the last segment after a dot.
        let (prims, property) = match rest.split_once('.') {
            Some((prims, property)) => (prims, Some(property)),
            None => (rest, None),
        };
        let mut node = self;
        for part in prims.split('/').filter(|p| !p.is_empty()) {
            node = node.child_mut(part, false);
        }
        if let Some(property) = property {
            node.child_mut(property, true);
        }
    }
}

/// Build the tree from the layer's prims, in the order they were authored.
fn collect(node: &mut Node, prims: &[UsdPrim]) {
    for prim in prims {
        let child = node.child_mut(&prim.name, false);
        collect_one(child, prim);
    }
}

/// One prim's properties, variant bodies and children.
///
/// A variant does not live *inside* the prim in a crate file; it hangs off it
/// under a path element of its own — `{look=}` for the set and `{look=oak}`
/// for each choice — which is how one prim can hold several alternative bodies
/// without any of them being a child.
fn collect_one(node: &mut Node, prim: &UsdPrim) {
    for property in &prim.properties {
        node.child_mut(&property.name, true);
    }
    for set in &prim.variant_sets {
        // The set and each of its choices are both children of the prim, not
        // of each other: the path of a variant is `/Chair{look=oak}`, which
        // hangs off `/Chair` and not off `/Chair{look=}`.
        node.child_mut(&format!("{{{}=}}", set.name), false);
        for (choice, body) in &set.variants {
            let variant = node.child_mut(&format!("{{{}={}}}", set.name, choice), false);
            collect_one(variant, body);
        }
    }
    collect(node, &prim.children);
}

/// Every path anything points at: a relationship target, a connection, or the
/// prim half of a composition arc.
///
/// All of them are stored as indices into the path table, so a path that
/// nothing defines still has to be *in* that table or the arc has nowhere to
/// point.
fn referenced_paths(prims: &[UsdPrim], out: &mut Vec<String>) {
    for prim in prims {
        for property in &prim.properties {
            match &property.value {
                UsdValue::Path(p) => out.push(p.clone()),
                UsdValue::Array(items) if property.relationship => {
                    out.extend(items.iter().filter_map(|v| v.as_str()).map(str::to_string));
                }
                _ => {}
            }
        }
        for (key, value) in &prim.metadata {
            // Every qualifier, not just two of them: an `add references` names
            // a prim path that has to be in the table like any other, and
            // leaving it out makes the arc point at nothing.
            if !matches!(
                split_qualifier(key).1,
                "references" | "payload" | "inherits" | "specializes"
            ) {
                continue;
            }
            for arc in value.references() {
                if !arc.prim_path.is_empty() {
                    out.push(arc.prim_path.clone());
                }
            }
        }
        for set in &prim.variant_sets {
            for (_, body) in &set.variants {
                referenced_paths(std::slice::from_ref(body), out);
            }
        }
        referenced_paths(&prim.children, out);
    }
}

/// Write a layer as a crate file.
pub fn write(layer: &UsdLayer) -> Vec<u8> {
    let mut w = Writer::default();
    // The empty token first, so the pseudo-root has something to name and so
    // index zero is never a real name.
    w.token("");

    // --- The path tree, in full, before anything is written.
    let mut tree = Node::default();
    collect(&mut tree, &layer.prims);
    let mut referenced = Vec::new();
    referenced_paths(&layer.prims, &mut referenced);
    for path in &referenced {
        tree.insert_path(path);
    }

    // --- The walk, which is what gives every path its index.
    let root_at = begin_entry(&mut w, 0);
    w.slots.insert("/".to_string(), (root_at + 1) as u32);
    let mut total = 1usize;
    for (i, child) in tree.children.iter().enumerate() {
        total += emit(&mut w, child, "", i + 1 < tree.children.len());
    }
    w.path_jumps[root_at] = jump_for(!tree.children.is_empty(), false, total);

    // --- The specs, which refer to those indices.
    write_root_spec(&mut w, layer);
    for prim in &layer.prims {
        write_prim_spec(&mut w, prim, "");
    }

    assemble(w)
}

/// Emit one node and its subtree, returning how many slots it took — which is
/// exactly the jump its sibling needs.
fn emit(w: &mut Writer, node: &Node, parent: &str, has_sibling: bool) -> usize {
    let token = w.token(&node.name) as i32;
    // A negative element marks a property rather than a child prim.
    let at = begin_entry(w, if node.property { -token } else { token });
    let path = if node.property {
        format!("{parent}.{}", node.name)
    } else if node.name.starts_with('{') {
        // A variant selection qualifies the prim rather than sitting under it.
        format!("{parent}{}", node.name)
    } else {
        format!("{parent}/{}", node.name)
    };
    w.slots.insert(path.clone(), (at + 1) as u32);

    let mut total = 1usize;
    for (i, child) in node.children.iter().enumerate() {
        total += emit(w, child, &path, i + 1 < node.children.len());
    }
    w.path_jumps[at] = jump_for(!node.children.is_empty(), has_sibling, total);
    total
}

/// Reserve a slot in the path walk, to be patched once its subtree is known.
fn begin_entry(w: &mut Writer, element: i32) -> usize {
    let at = w.path_elements.len();
    w.path_elements.push(element);
    w.path_jumps.push(-2);
    at
}

/// The layer's own metadata, which hangs off the pseudo-root.
fn write_root_spec(w: &mut Writer, layer: &UsdLayer) {
    let mut members = Vec::new();
    for (name, rep) in arc_fields(w, &layer.metadata) {
        members.push(w.field(name, rep) as i32);
    }
    for (key, value) in &layer.metadata {
        if key.is_empty() || is_arc_name(split_qualifier(key).1) {
            continue;
        }
        if let Some(rep) = value_rep(w, key, value, "") {
            members.push(w.field(key, rep) as i32);
        }
    }
    if !layer.prims.is_empty() {
        let names: Vec<UsdValue> = layer
            .prims
            .iter()
            .map(|p| UsdValue::Token(p.name.clone()))
            .collect();
        let rep = token_vector(w, &names);
        members.push(w.field("primChildren", rep) as i32);
    }
    let set = w.field_set(members);
    let at = w.slots.get("/").copied().unwrap_or(0);
    w.specs.push((at, set, spec::PSEUDO_ROOT));
}

fn write_prim_spec(w: &mut Writer, prim: &UsdPrim, parent: &str) {
    let path = format!("{parent}/{}", prim.name);

    let mut members = Vec::new();
    let specifier = match prim.specifier {
        Specifier::Def => 0u64,
        Specifier::Over => 1,
        Specifier::Class => 2,
    };
    let rep = inlined(kind::SPECIFIER, specifier);
    members.push(w.field("specifier", rep) as i32);
    if !prim.type_name.is_empty() {
        let token = w.token(&prim.type_name);
        let rep = inlined(kind::TOKEN, token as u64);
        members.push(w.field("typeName", rep) as i32);
    }
    for (name, rep) in arc_fields(w, &prim.metadata) {
        members.push(w.field(name, rep) as i32);
    }
    for (key, value) in &prim.metadata {
        if key.is_empty() || key == "specifier" || key == "typeName" {
            continue;
        }
        // The composition fields were grouped and written above.
        if is_arc_name(split_qualifier(key).1) {
            continue;
        }
        if let Some(rep) = value_rep(w, key, value, "") {
            members.push(w.field(key, rep) as i32);
        }
    }
    if !prim.properties.is_empty() {
        let names: Vec<UsdValue> = prim
            .properties
            .iter()
            .map(|p| UsdValue::Token(p.name.clone()))
            .collect();
        let rep = token_vector(w, &names);
        members.push(w.field("properties", rep) as i32);
    }
    if !prim.children.is_empty() {
        let names: Vec<UsdValue> = prim
            .children
            .iter()
            .map(|c| UsdValue::Token(c.name.clone()))
            .collect();
        let rep = token_vector(w, &names);
        members.push(w.field("primChildren", rep) as i32);
    }

    if !prim.variant_sets.is_empty() {
        let names: Vec<UsdValue> = prim
            .variant_sets
            .iter()
            .map(|s| UsdValue::Token(s.name.clone()))
            .collect();
        let rep = token_vector(w, &names);
        members.push(w.field("variantSetChildren", rep) as i32);
    }

    let set = w.field_set(members);
    if let Some(at) = w.slots.get(path.as_str()).copied() {
        w.specs.push((at, set, spec::PRIM));
    }

    for property in &prim.properties {
        write_property_spec(w, property, &path);
    }
    for variant_set in &prim.variant_sets {
        write_variant_set_spec(w, variant_set, &path);
    }
    for child in &prim.children {
        write_prim_spec(w, child, &path);
    }
}

/// The set itself, and each of its alternatives.
fn write_variant_set_spec(w: &mut Writer, set: &UsdVariantSet, prim_path: &str) {
    let set_path = format!("{prim_path}{{{}=}}", set.name);
    let names: Vec<UsdValue> = set
        .variants
        .iter()
        .map(|(n, _)| UsdValue::Token(n.clone()))
        .collect();
    let rep = token_vector(w, &names);
    let members = vec![w.field("variantChildren", rep) as i32];
    let field_set = w.field_set(members);
    if let Some(at) = w.slots.get(set_path.as_str()).copied() {
        w.specs.push((at, field_set, spec::VARIANT_SET));
    }

    for (choice, body) in &set.variants {
        let variant_path = format!("{prim_path}{{{}={choice}}}", set.name);
        write_variant_spec(w, body, &variant_path);
    }
}

/// One variant's body, which is a prim body under a path of its own.
fn write_variant_spec(w: &mut Writer, body: &UsdPrim, path: &str) {
    let mut members = Vec::new();
    if !body.type_name.is_empty() {
        let token = w.token(&body.type_name);
        let rep = inlined(kind::TOKEN, token as u64);
        members.push(w.field("typeName", rep) as i32);
    }
    for (name, rep) in arc_fields(w, &body.metadata) {
        members.push(w.field(name, rep) as i32);
    }
    for (key, value) in &body.metadata {
        if key.is_empty() || is_arc_name(split_qualifier(key).1) {
            continue;
        }
        if let Some(rep) = value_rep(w, key, value, "") {
            members.push(w.field(key, rep) as i32);
        }
    }
    if !body.properties.is_empty() {
        let names: Vec<UsdValue> = body
            .properties
            .iter()
            .map(|p| UsdValue::Token(p.name.clone()))
            .collect();
        let rep = token_vector(w, &names);
        members.push(w.field("properties", rep) as i32);
    }
    if !body.children.is_empty() {
        let names: Vec<UsdValue> = body
            .children
            .iter()
            .map(|c| UsdValue::Token(c.name.clone()))
            .collect();
        let rep = token_vector(w, &names);
        members.push(w.field("primChildren", rep) as i32);
    }

    let field_set = w.field_set(members);
    if let Some(at) = w.slots.get(path).copied() {
        w.specs.push((at, field_set, spec::VARIANT));
    }
    for property in &body.properties {
        write_property_spec(w, property, path);
    }
    for child in &body.children {
        write_prim_spec(w, child, path);
    }
}

fn write_property_spec(w: &mut Writer, property: &UsdProperty, prim_path: &str) {
    let path = format!("{prim_path}.{}", property.name);

    let mut members = Vec::new();
    if !property.type_name.is_empty() {
        let token = w.token(&property.type_name);
        let rep = inlined(kind::TOKEN, token as u64);
        members.push(w.field("typeName", rep) as i32);
    }
    // Varying is the fallback for an attribute and uniform is the fallback for
    // a relationship, so a relationship has to say `uniform` explicitly or USD
    // reads back a `varying rel`.
    if property.uniform || property.relationship {
        let rep = inlined(kind::VARIABILITY, 1);
        members.push(w.field("variability", rep) as i32);
    }

    if property.relationship {
        if let Some(rep) = path_list_op_with(w, &property.value, list_op_header(&property.qualifier))
        {
            members.push(w.field("targetPaths", rep) as i32);
        }
    } else if property.value.samples().is_some() {
        if let Some(rep) = time_samples(w, &property.value, &property.type_name) {
            members.push(w.field("timeSamples", rep) as i32);
        }
    } else if matches!(property.value, UsdValue::Path(_)) {
        // A connected attribute: the value is where it is wired from.
        if let Some(rep) = path_list_op(w, &property.value) {
            members.push(w.field("connectionPaths", rep) as i32);
        }
    } else if !matches!(property.value, UsdValue::None) {
        if let Some(rep) = value_rep(w, "default", &property.value, &property.type_name) {
            members.push(w.field("default", rep) as i32);
        }
    }
    for (key, value) in &property.metadata {
        if key.is_empty() {
            continue;
        }
        if let Some(rep) = value_rep(w, key, value, "") {
            members.push(w.field(key, rep) as i32);
        }
    }

    let set = w.field_set(members);
    if let Some(at) = w.slots.get(path.as_str()).copied() {
        w.specs.push((
            at,
            set,
            if property.relationship {
                spec::RELATIONSHIP
            } else {
                spec::ATTRIBUTE
            },
        ));
    }
}

/// The jump value for an entry: what the reader needs to find what comes next.
fn jump_for(has_child: bool, has_sibling: bool, subtree: usize) -> i32 {
    match (has_child, has_sibling) {
        (true, true) => subtree as i32,
        (true, false) => -1,
        (false, true) => 0,
        (false, false) => -2,
    }
}

fn inlined(kind: u8, payload: u64) -> u64 {
    IS_INLINED | ((kind as u64) << 48) | (payload & ((1 << 48) - 1))
}

fn referenced(kind: u8, at: u64, array: bool) -> u64 {
    let mut rep = ((kind as u64) << 48) | (at & ((1 << 48) - 1));
    if array {
        rep |= IS_ARRAY;
    }
    rep
}

/// A `u64` count followed by token indices — what `primChildren` and
/// `properties` are.
fn token_vector(w: &mut Writer, names: &[UsdValue]) -> u64 {
    let indices: Vec<u32> = names
        .iter()
        .map(|v| w.token(v.as_str().unwrap_or_default()))
        .collect();
    let mut bytes = (indices.len() as u64).to_le_bytes().to_vec();
    for index in indices {
        bytes.extend_from_slice(&index.to_le_bytes());
    }
    referenced(kind::TOKEN_VECTOR, w.intern(bytes), false)
}

/// A composition field, under the name a crate gives it.
///
/// Three of them are stored under different names than a document writes:
/// `inherits` is `inheritPaths`, `variants` is `variantSelection`, and
/// `variantSets` is `variantSetNames`. Writing them under the document's
/// spelling produces a file that reads back here and composes to nothing in
/// USD, because nothing over there is looking for a field called `inherits`.
/// Every composition field a prim carries, one entry per field.
///
/// A field may be stated several times over with different qualifiers —
/// `add references` and `delete references` are one field with two sub-lists —
/// and a spec can only hold one field of a given name. Writing them separately
/// loses whichever comes second.
fn arc_fields(w: &mut Writer, metadata: &[(String, UsdValue)]) -> Vec<(&'static str, u64)> {
    // Group by the field underneath the qualifier, keeping the order met.
    let mut groups: Vec<(String, Vec<(u8, UsdValue)>)> = Vec::new();
    for (key, value) in metadata {
        let (qualifier, bare) = split_qualifier(key);
        if !is_arc_name(bare) {
            continue;
        }
        let bit = list_op_bit(qualifier);
        match groups.iter_mut().find(|(name, _)| name == bare) {
            Some((_, parts)) => parts.push((bit, value.clone())),
            None => groups.push((bare.to_string(), vec![(bit, value.clone())])),
        }
    }

    let mut out = Vec::new();
    for (bare, parts) in groups {
        if let Some(field) = combined_field(w, &bare, &parts) {
            out.extend(field);
        }
    }
    out
}

fn split_qualifier(key: &str) -> (&str, &str) {
    match key.split_once(' ') {
        Some((q, rest)) if matches!(q, "prepend" | "append" | "delete" | "add" | "reorder") => {
            (q, rest)
        }
        _ => ("", key),
    }
}

fn is_arc_name(name: &str) -> bool {
    matches!(
        name,
        "references"
            | "payload"
            | "inherits"
            | "specializes"
            | "variants"
            | "variantSets"
            | "subLayers"
            | "apiSchemas"
            | "nameChildren"
            | "properties"
    )
}

/// Which sub-list of a list operation a qualifier selects.
fn list_op_bit(qualifier: &str) -> u8 {
    match qualifier {
        "add" => 2,
        "delete" => 3,
        "reorder" => 4,
        "prepend" => 5,
        "append" => 6,
        _ => 1,
    }
}

/// One field built from every qualified statement of it.
fn combined_field(
    w: &mut Writer,
    bare: &str,
    parts: &[(u8, UsdValue)],
) -> Option<Vec<(&'static str, u64)>> {
    // The fields that are not list operations take the one value they have.
    let single = || parts.first().map(|(_, v)| v.clone());
    match bare {
        "references" => Some(vec![(
            "references",
            sublists(w, kind::REFERENCE_LIST_OP, parts),
        )]),
        "payload" => Some(vec![("payload", sublists(w, kind::PAYLOAD_LIST_OP, parts))]),
        "inherits" => Some(vec![("inheritPaths", sublists(w, kind::PATH_LIST_OP, parts))]),
        "specializes" => Some(vec![("specializes", sublists(w, kind::PATH_LIST_OP, parts))]),
        "apiSchemas" => Some(vec![("apiSchemas", sublists(w, kind::TOKEN_LIST_OP, parts))]),
        "variantSets" => Some(vec![(
            "variantSetNames",
            sublists(w, kind::STRING_LIST_OP, parts),
        )]),
        "variants" => {
            let value = single()?;
            Some(vec![("variantSelection", variant_selection(w, &value)?)])
        }
        "nameChildren" => {
            let value = single()?;
            Some(vec![("primOrder", token_vector(w, &as_items(&value)))])
        }
        "properties" => {
            let value = single()?;
            Some(vec![("propertyOrder", token_vector(w, &as_items(&value)))])
        }
        "subLayers" => {
            let value = single()?;
            let (layers, _) = string_vector(w, &value);
            let offsets: Vec<(f64, f64)> = value
                .references()
                .into_iter()
                .map(|arc| (arc.offset, arc.scale))
                .collect();
            Some(vec![
                ("subLayers", layers),
                ("subLayerOffsets", layer_offsets(w, &offsets)),
            ])
        }
        _ => None,
    }
}

/// A list operation with every sub-list that was authored.
///
/// The sub-lists are laid down in bit order, which is the order a reader walks
/// them in — putting them in the order they were written would have the reader
/// take one for another.
fn sublists(w: &mut Writer, kind: u8, parts: &[(u8, UsdValue)]) -> u64 {
    let mut header = 0u8;
    for (bit, _) in parts {
        header |= 1 << bit;
        // An explicit list says so twice: the bit and the flag.
        if *bit == 1 {
            header |= 1;
        }
    }

    let mut bytes = vec![header];
    for bit in 1..7u8 {
        let Some((_, value)) = parts.iter().find(|(b, _)| *b == bit) else {
            continue;
        };
        match kind {
            kind::REFERENCE_LIST_OP | kind::PAYLOAD_LIST_OP => {
                let arcs = value.references();
                bytes.extend_from_slice(&(arcs.len() as u64).to_le_bytes());
                for arc in &arcs {
                    let asset = w.string(&arc.asset);
                    let path = w.path_slot(&arc.prim_path);
                    bytes.extend_from_slice(&asset.to_le_bytes());
                    bytes.extend_from_slice(&path.to_le_bytes());
                    bytes.extend_from_slice(&arc.offset.to_le_bytes());
                    bytes.extend_from_slice(&arc.scale.to_le_bytes());
                    bytes.extend_from_slice(&0u64.to_le_bytes());
                }
            }
            kind::PATH_LIST_OP => {
                let arcs = value.references();
                bytes.extend_from_slice(&(arcs.len() as u64).to_le_bytes());
                for arc in &arcs {
                    bytes.extend_from_slice(&w.path_slot(&arc.prim_path).to_le_bytes());
                }
            }
            kind::STRING_LIST_OP => {
                let names = as_items(value);
                bytes.extend_from_slice(&(names.len() as u64).to_le_bytes());
                for name in &names {
                    let index = w.string(name.as_str().unwrap_or_default());
                    bytes.extend_from_slice(&index.to_le_bytes());
                }
            }
            _ => {
                let names = value.flat_tokens();
                bytes.extend_from_slice(&(names.len() as u64).to_le_bytes());
                for name in names {
                    let index = w.token(name);
                    bytes.extend_from_slice(&index.to_le_bytes());
                }
            }
        }
    }
    referenced(kind, w.intern(bytes), false)
}


/// The `(offset, scale)` that accompanies each sublayer, unshifted.
fn layer_offsets(w: &mut Writer, offsets: &[(f64, f64)]) -> u64 {
    let mut bytes = (offsets.len() as u64).to_le_bytes().to_vec();
    for (offset, scale) in offsets {
        bytes.extend_from_slice(&offset.to_le_bytes());
        bytes.extend_from_slice(&scale.to_le_bytes());
    }
    referenced(kind::LAYER_OFFSET_VECTOR, w.intern(bytes), false)
}



/// A dictionary, as a count and one blob per entry.
///
/// Each blob ends with the rep naming its value. The value itself is written
/// through the usual interning path and the rep points at it, which is allowed
/// because that payload is an absolute offset — a blob need only *end* with
/// the rep, not contain what it names.
fn dictionary(w: &mut Writer, entries: &[(String, UsdValue)]) -> u64 {
    // Every value first, so their offsets exist before the table refers to
    // them.
    let reps: Vec<(u32, u64)> = entries
        .iter()
        .map(|(key, value)| {
            let rep = match value {
                UsdValue::Dict(inner) => Some(dictionary(w, inner)),
                other => value_rep_in(w, key, other, "", true),
            };
            (w.string(key), rep)
        })
        // A value with no type this crate can name is left out rather than
        // written with a type of zero, which USD refuses to unpack at all —
        // losing one entry beats losing the file.
        .filter_map(|(key, rep)| rep.map(|rep| (key, rep)))
        .collect();

    let mut bytes = (reps.len() as u64).to_le_bytes().to_vec();
    for (key, rep) in reps {
        bytes.extend_from_slice(&key.to_le_bytes());
        // The blob is the rep and nothing else.
        bytes.extend_from_slice(&8u64.to_le_bytes());
        bytes.extend_from_slice(&rep.to_le_bytes());
    }
    referenced(kind::DICTIONARY, w.intern(bytes), false)
}

/// A value's items, so a single name and a list of them are the same shape.
fn as_items(value: &UsdValue) -> Vec<UsdValue> {
    match value {
        UsdValue::Array(items) => items.clone(),
        other => vec![other.clone()],
    }
}



/// `variants`: which choice of each set is selected.
fn variant_selection(w: &mut Writer, value: &UsdValue) -> Option<u64> {
    let UsdValue::Dict(entries) = value else {
        return None;
    };
    let mut bytes = (entries.len() as u64).to_le_bytes().to_vec();
    for (set, choice) in entries {
        bytes.extend_from_slice(&w.string(set).to_le_bytes());
        bytes.extend_from_slice(&w.string(choice.as_str().unwrap_or_default()).to_le_bytes());
    }
    Some(referenced(kind::VARIANT_SELECTION_MAP, w.intern(bytes), false))
}

/// `subLayers`: a plain list of layer names, and how many there were.
fn string_vector(w: &mut Writer, value: &UsdValue) -> (u64, usize) {
    // A sublayer with a time offset parses as an arc rather than a bare
    // asset, so both shapes have to be read for the name.
    let names: Vec<String> = match value {
        UsdValue::Array(items) => items.iter().filter_map(name_of).collect(),
        other => name_of(other).into_iter().collect(),
    };
    let mut bytes = (names.len() as u64).to_le_bytes().to_vec();
    for name in &names {
        bytes.extend_from_slice(&w.string(name).to_le_bytes());
    }
    let count = names.len();
    (referenced(kind::STRING_VECTOR, w.intern(bytes), false), count)
}

fn name_of(value: &UsdValue) -> Option<String> {
    match value {
        UsdValue::Reference(arc) => Some(arc.asset.clone()),
        other => other.as_str().map(str::to_string),
    }
}

/// A relationship's targets or an attribute's connections.
///
/// Written as an explicit list, which is what a single-layer document means by
/// naming a target at all.
/// The header bit for a list operation's qualifier.
fn list_op_header(qualifier: &str) -> u8 {
    match qualifier {
        "add" => 1 << 2,
        "delete" => 1 << 3,
        "reorder" => 1 << 4,
        "prepend" => 1 << 5,
        "append" => 1 << 6,
        // Explicit, with explicit items.
        _ => 0b11,
    }
}

fn path_list_op(w: &mut Writer, value: &UsdValue) -> Option<u64> {
    path_list_op_with(w, value, 0b11)
}

fn path_list_op_with(w: &mut Writer, value: &UsdValue, header: u8) -> Option<u64> {
    let targets: Vec<&str> = match value {
        UsdValue::Path(p) => vec![p.as_str()],
        UsdValue::Array(items) => items.iter().filter_map(|v| v.as_str()).collect(),
        _ => return None,
    };
    if targets.is_empty() {
        return None;
    }
    // Paths are referred to by index, and the walk put every referenced one
    // in the table before any of this ran.
    let indices: Vec<u32> = targets
        .iter()
        .filter_map(|p| w.slots.get(*p).copied())
        .collect();
    if indices.is_empty() {
        return None;
    }

    let mut bytes = vec![header];
    bytes.extend_from_slice(&(indices.len() as u64).to_le_bytes());
    for index in indices {
        bytes.extend_from_slice(&index.to_le_bytes());
    }
    Some(referenced(kind::PATH_LIST_OP, w.intern(bytes), false))
}

/// A time-sampled value.
fn time_samples(w: &mut Writer, value: &UsdValue, type_name: &str) -> Option<u64> {
    let samples = value.samples()?;
    let times: Vec<f64> = samples.iter().map(|(t, _)| *t).collect();

    // Each sample's value, written before the section that names them.
    let reps: Vec<u64> = samples
        .iter()
        .map(|(_, v)| value_rep(w, "default", v, type_name).unwrap_or(0))
        .collect();

    // The times, then a rep pointing back at them: the layout that lets two
    // attributes keyed on the same frames share one copy.
    let mut time_bytes = (times.len() as u64).to_le_bytes().to_vec();
    for time in &times {
        time_bytes.extend_from_slice(&time.to_le_bytes());
    }
    // Two attributes keyed on the same frames share one copy of them, which is
    // what the rep-after-the-array layout exists for. 48 is `DoubleVector`.
    let times_rep = referenced(48, w.intern(time_bytes), false);

    let section = 8 + times.len() * 8 + 8;
    let mut bytes = (section as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&(times.len() as u64).to_le_bytes());
    for time in &times {
        bytes.extend_from_slice(&time.to_le_bytes());
    }
    bytes.extend_from_slice(&times_rep.to_le_bytes());
    // The width of a value rep, then the values.
    bytes.extend_from_slice(&8u64.to_le_bytes());
    bytes.extend_from_slice(&(reps.len() as u64).to_le_bytes());
    for rep in reps {
        bytes.extend_from_slice(&rep.to_le_bytes());
    }
    Some(referenced(kind::TIME_SAMPLES, w.intern(bytes), false))
}

/// One value, written into the data area if it does not fit in its own word.
///
/// `type_name` is the attribute's declared type where there is one; layer and
/// prim metadata have none, and their type is taken from the value's shape.
fn value_rep(w: &mut Writer, key: &str, value: &UsdValue, type_name: &str) -> Option<u64> {
    value_rep_in(w, key, value, type_name, false)
}

/// `in_dictionary` says the key is a dictionary key rather than a schema field
/// name. It matters: `active` is a boolean on a prim and a table of clip
/// switches inside a `clips` dictionary, and the schema's answer is the wrong
/// one there.
fn value_rep_in(
    w: &mut Writer,
    key: &str,
    value: &UsdValue,
    type_name: &str,
    in_dictionary: bool,
) -> Option<u64> {
    // The fields this crate synthesises have types the document never states.
    match key {
        "primChildren" | "properties" => {
            if let UsdValue::Array(items) = value {
                return Some(token_vector(w, items));
            }
        }
        _ => {}
    }

    // A dictionary carries its own shape and has no type name to look up, so
    // it is answered before the type resolution that would give up on it.
    if let UsdValue::Dict(entries) = value {
        return Some(dictionary(w, entries));
    }
    // A block has no value at all — it is the absence of one, stated.
    if matches!(value, UsdValue::Block) {
        return Some(inlined(kind::VALUE_BLOCK, 0));
    }

    let (kind, array) = match kind_of(type_name) {
        Some(found) => found,
        // A property states its type; a metadata field's comes from the schema,
        // and only failing both is it guessed from the value.
        None if in_dictionary => infer(value)?,
        None => match known_field_type(key).and_then(kind_of) {
            Some(found) => found,
            None => infer(value)?,
        },
    };

    if array {
        return Some(write_array(w, kind, value));
    }
    // Text that USD wants as a name goes in as a token whether or not the
    // document quoted it, and text it wants as a string goes in as a string.
    // Which one it is comes from the type, not from the punctuation.
    if let Some(text) = value.as_str() {
        return Some(match kind {
            kind::STRING => {
                let index = w.string(text);
                inlined(kind::STRING, index as u64)
            }
            // A membership expression is indexed exactly like a string but is
            // written out of line regardless, and a crate holding one is
            // version 0.10.0. USD will not open it as 0.8.0.
            kind::PATH_EXPRESSION => {
                let index = w.string(text);
                w.needs(0, 10, 0);
                let at = w.intern(index.to_le_bytes().to_vec());
                at | ((kind::PATH_EXPRESSION as u64) << 48)
            }
            kind::ASSET_PATH => {
                let token = w.token(text);
                inlined(kind::ASSET_PATH, token as u64)
            }
            _ => {
                let token = w.token(text);
                inlined(kind::TOKEN, token as u64)
            }
        });
    }
    if let UsdValue::Path(_) = value {
        return path_list_op(w, value);
    }

    // A scalar that fits the 48-bit payload is carried in the rep word itself.
    // This is not only a saving: USD reads the small arithmetic types straight
    // out of the payload, so a `float` written out of line comes back as its
    // own file offset reinterpreted as bits.
    if let Some(payload) = inline_payload(kind, value) {
        return Some(inlined(kind, payload));
    }

    let (lanes, width) = shape(kind)?;
    let mut bytes = Vec::new();
    write_numbers(&mut bytes, kind, value, lanes, width);
    Some(referenced(kind, w.intern(bytes), false))
}

/// A scalar's bits, if the type is one USD keeps in the rep word.
///
/// Vectors and matrices are excluded: USD inlines those only when every
/// component fits in a signed byte, and writing them out of line is both
/// simpler and read correctly.
fn inline_payload(kind: u8, value: &UsdValue) -> Option<u64> {
    Some(match kind {
        kind::BOOL => match value {
            UsdValue::Bool(v) => *v as u64,
            other => (other.as_f64()? != 0.0) as u64,
        },
        kind::UCHAR => as_i128(value)? as u8 as u64,
        kind::INT | kind::UINT => as_i128(value)? as i64 as i32 as u32 as u64,
        kind::INT64 | kind::UINT64 => {
            let v = as_i128(value)?;
            // Only what survives the narrowing; anything wider goes to the
            // data area where all 64 bits fit.
            i32::try_from(v).ok()? as u32 as u64
        }
        kind::HALF => f32_to_half(value.as_f64()? as f32) as u64,
        kind::FLOAT => (value.as_f64()? as f32).to_bits() as u64,
        // A time code is *not* inlined, even though it would fit: USD writes it
        // out of line as a full double, and reads an inlined one as zero.
        kind::DOUBLE => {
            let v = value.as_f64()?;
            // A double is inlined as f32 bits, so only one that survives the
            // round trip may be.
            let narrow = v as f32;
            if narrow as f64 != v {
                return None;
            }
            narrow.to_bits() as u64
        }
        _ => return None,
    })
}

/// A value as an integer, without the detour through `f64` that costs
/// precision above 2^53.
///
/// Held at `i128` because `uint64`'s upper half does not fit an `i64`; the
/// narrowing to whatever width is being written happens at the point of
/// writing, where the type is known.
fn as_i128(value: &UsdValue) -> Option<i128> {
    match value {
        UsdValue::Int(v) => Some(*v),
        UsdValue::Bool(v) => Some(*v as i128),
        other => other.as_f64().map(|v| v as i128),
    }
}

/// The type USD's own schema gives a metadata field.
///
/// A document does not declare the type of `metersPerUnit` — it just writes a
/// number — and inferring one from the value is not good enough: `24` parses as
/// an integer, and a `timeCodesPerSecond` stored as an integer is not the field
/// USD is looking for. Everything named here is a field whose type is fixed by
/// the schema rather than by what was typed.
fn known_field_type(key: &str) -> Option<&'static str> {
    Some(match key {
        "defaultPrim" | "upAxis" | "kind" | "interpolation" | "visibility" => "token",
        "metersPerUnit" | "timeCodesPerSecond" | "framesPerSecond" | "startTimeCode"
        | "endTimeCode" => "double",
        "documentation" | "comment" => "string",
        "elementSize" => "int",
        "active" | "hidden" | "instanceable" => "bool",
        _ => return None,
    })
}

/// Guess a type from a value, for the metadata that never declares one.
fn infer(value: &UsdValue) -> Option<(u8, bool)> {
    Some(match value {
        UsdValue::Bool(_) => (kind::BOOL, false),
        UsdValue::Int(_) => (kind::INT, false),
        UsdValue::Float(_) => (kind::DOUBLE, false),
        UsdValue::String(_) => (kind::STRING, false),
        UsdValue::Token(_) => (kind::TOKEN, false),
        UsdValue::Asset(_) => (kind::ASSET_PATH, false),
        UsdValue::Path(_) => (kind::PATH_LIST_OP, false),
        UsdValue::Tuple(items) => match items.len() {
            2 => (kind::VEC2D, false),
            3 => (kind::VEC3D, false),
            4 => (kind::VEC4D, false),
            9 => (kind::MATRIX3D, false),
            16 => (kind::MATRIX4D, false),
            _ => return None,
        },
        // A dictionary's values carry no declared type — the document states
        // one, but a dictionary entry is just a value — so the shape has to
        // answer for it.
        UsdValue::Array(items) => match items.first() {
            None => (kind::TOKEN_VECTOR, false),
            Some(UsdValue::Asset(_)) => (kind::ASSET_PATH, true),
            Some(UsdValue::String(_)) => (kind::STRING, true),
            Some(UsdValue::Token(_)) => (kind::TOKEN_VECTOR, false),
            Some(UsdValue::Int(_)) => (kind::INT, true),
            Some(UsdValue::Bool(_)) => (kind::BOOL, true),
            Some(UsdValue::Float(_)) => (kind::DOUBLE, true),
            // Double precision, because a dictionary is where a layer keeps
            // the numbers it means exactly: frame ranges, clip tables.
            Some(UsdValue::Tuple(lanes)) => (
                match lanes.len() {
                    2 => kind::VEC2D,
                    3 => kind::VEC3D,
                    4 => kind::VEC4D,
                    9 => kind::MATRIX3D,
                    16 => kind::MATRIX4D,
                    _ => return None,
                },
                true,
            ),
            _ => return None,
        },
        _ => return None,
    })
}

/// An array: a count, then the elements.
fn write_array(w: &mut Writer, kind: u8, value: &UsdValue) -> u64 {
    let empty = Vec::new();
    let items = match value {
        UsdValue::Array(items) => items,
        // A single value where an array is declared is an array of one.
        UsdValue::None => &empty,
        other => std::slice::from_ref(other),
    };

    if matches!(kind, kind::TOKEN | kind::STRING | kind::ASSET_PATH) {
        let indices: Vec<u32> = items
            .iter()
            .map(|v| {
                let text = v.as_str().unwrap_or_default();
                // An array of assets goes through the string table; a single
                // asset goes through the tokens. USD reads them that way.
                if matches!(kind, kind::STRING | kind::ASSET_PATH) {
                    w.string(text)
                } else {
                    w.token(text)
                }
            })
            .collect();
        let mut bytes = (indices.len() as u64).to_le_bytes().to_vec();
        for index in indices {
            bytes.extend_from_slice(&index.to_le_bytes());
        }
        return referenced(kind, w.intern(bytes), true);
    }

    let Some((lanes, width)) = shape(kind) else {
        return referenced(kind, w.intern(0u64.to_le_bytes().to_vec()), true);
    };

    let mut bytes = (items.len() as u64).to_le_bytes().to_vec();
    for item in items {
        write_numbers(&mut bytes, kind, item, lanes, width);
    }
    referenced(kind, w.intern(bytes), true)
}

/// Every number inside a value, however deeply nested.
///
/// A matrix is a tuple of rows rather than a flat run, so flattening one level
/// would find no numbers at all and write a matrix of zeros. Kept at `f64`
/// because the double-precision types are exactly the ones written this way.
fn flat_f64(value: &UsdValue) -> Vec<f64> {
    match value {
        UsdValue::Tuple(items) | UsdValue::Array(items) => {
            items.iter().flat_map(flat_f64).collect()
        }
        other => other.as_f64().into_iter().collect(),
    }
}

/// Write one element as `lanes` numbers of `width` bytes.
fn write_numbers(out: &mut Vec<u8>, kind: u8, value: &UsdValue, lanes: usize, width: usize) {
    if is_float_kind(kind) {
        let mut components = flat_f64(value);
        // A quaternion is written imaginary part first, which is the reverse of
        // how it is spelled in a document.
        if is_quat(kind) && components.len() == 4 {
            components.rotate_left(1);
        }
        components.resize(lanes, 0.0);
        for value in components {
            match width {
                2 => out.extend_from_slice(&f32_to_half(value as f32).to_le_bytes()),
                4 => out.extend_from_slice(&(value as f32).to_le_bytes()),
                _ => out.extend_from_slice(&value.to_le_bytes()),
            }
        }
        return;
    }
    // Integers keep away from `f64` entirely: everything past 2^53 would be
    // rounded on the way through it, and `int64[]` exists precisely to hold
    // numbers that big.
    let mut components = flat_i128(value);
    components.resize(lanes, 0);
    for value in components {
        match width {
            1 => out.push(value as u8),
            2 => out.extend_from_slice(&(value as i16).to_le_bytes()),
            4 => out.extend_from_slice(&(value as i32).to_le_bytes()),
            // Truncating to 64 bits writes the same bytes whether the type is
            // signed or unsigned, which is exactly what is wanted.
            _ => out.extend_from_slice(&(value as i64).to_le_bytes()),
        }
    }
}

/// Every integer inside a value, however deeply nested.
fn flat_i128(value: &UsdValue) -> Vec<i128> {
    match value {
        UsdValue::Tuple(items) | UsdValue::Array(items) => {
            items.iter().flat_map(flat_i128).collect()
        }
        other => as_i128(other).into_iter().collect(),
    }
}

/// Lay the tables down as a file.
fn assemble(w: Writer) -> Vec<u8> {
    let Writer {
        tokens,
        strings,
        fields,
        field_sets,
        path_elements,
        path_jumps,
        specs,
        data,
        needs_version,
        ..
    } = w;

    let version = VERSION.max(needs_version);
    let mut out = vec![0u8; BOOTSTRAP];
    out[..8].copy_from_slice(b"PXR-USDC");
    out[8] = version.0;
    out[9] = version.1;
    out[10] = version.2;
    out.extend_from_slice(&data);

    let mut sections: Vec<(&str, u64, u64)> = Vec::new();
    let mut section = |out: &mut Vec<u8>, name: &'static str, body: Vec<u8>| {
        while !out.len().is_multiple_of(8) {
            out.push(0);
        }
        let at = out.len() as u64;
        out.extend_from_slice(&body);
        sections.push((name, at, body.len() as u64));
    };

    // TOKENS: every name, null separated, compressed as one run.
    let mut text = Vec::new();
    for token in &tokens {
        text.extend_from_slice(token.as_bytes());
        text.push(0);
    }
    let packed = lz4::compress(&text);
    let mut body = Vec::new();
    body.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    body.extend_from_slice(&(text.len() as u64).to_le_bytes());
    body.extend_from_slice(&(packed.len() as u64).to_le_bytes());
    body.extend_from_slice(&packed);
    section(&mut out, "TOKENS", body);

    // STRINGS: a count and the token index of each.
    let mut body = (strings.len() as u64).to_le_bytes().to_vec();
    for index in &strings {
        body.extend_from_slice(&index.to_le_bytes());
    }
    section(&mut out, "STRINGS", body);

    // FIELDS: the names packed, then the reps compressed.
    let mut body = (fields.len() as u64).to_le_bytes().to_vec();
    write_u32_array(
        &mut body,
        &fields.iter().map(|(name, _)| *name as i32).collect::<Vec<_>>(),
    );
    let mut reps = Vec::with_capacity(fields.len() * 8);
    for (_, rep) in &fields {
        reps.extend_from_slice(&rep.to_le_bytes());
    }
    let packed = lz4::compress(&reps);
    body.extend_from_slice(&(packed.len() as u64).to_le_bytes());
    body.extend_from_slice(&packed);
    section(&mut out, "FIELDS", body);

    // FIELDSETS.
    let mut body = (field_sets.len() as u64).to_le_bytes().to_vec();
    write_u32_array(&mut body, &field_sets);
    section(&mut out, "FIELDSETS", body);

    // PATHS: the walk, as three packed arrays.
    // The walk writes slots 1..=N, leaving slot zero for the empty path. USD
    // requires `numPaths` to be exactly one past the highest index used, so
    // the spare slot has to sit inside the numbering rather than after it.
    let indexes: Vec<i32> = (1..=path_elements.len() as i32).collect();
    let mut body = (path_elements.len() as u64 + 1).to_le_bytes().to_vec();
    body.extend_from_slice(&(path_elements.len() as u64).to_le_bytes());
    write_u32_array(&mut body, &indexes);
    write_u32_array(&mut body, &path_elements);
    write_u32_array(&mut body, &path_jumps);
    section(&mut out, "PATHS", body);

    // SPECS.
    let mut body = (specs.len() as u64).to_le_bytes().to_vec();
    write_u32_array(&mut body, &specs.iter().map(|(p, _, _)| *p as i32).collect::<Vec<_>>());
    write_u32_array(&mut body, &specs.iter().map(|(_, s, _)| *s as i32).collect::<Vec<_>>());
    write_u32_array(&mut body, &specs.iter().map(|(_, _, t)| *t as i32).collect::<Vec<_>>());
    section(&mut out, "SPECS", body);

    // The table of contents, and the pointer to it.
    while !out.len().is_multiple_of(8) {
        out.push(0);
    }
    let toc_at = out.len() as u64;
    out.extend_from_slice(&(sections.len() as u64).to_le_bytes());
    for (name, at, size) in &sections {
        let mut padded = [0u8; 16];
        padded[..name.len()].copy_from_slice(name.as_bytes());
        out.extend_from_slice(&padded);
        out.extend_from_slice(&at.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes());
    }
    out[16..24].copy_from_slice(&toc_at.to_le_bytes());

    let _ = (encode_u32, encode_u64, encodable_u64);
    out
}

#[cfg(test)]
mod half_tests {
    use super::*;
    use super::super::crate_read;

    /// Half conversion has to round, not truncate. Truncating biases every
    /// value toward zero, and the error is visible in the first two decimals:
    /// `0.333` becomes `0.332764` instead of `0.333008`.
    #[test]
    fn half_rounds_to_nearest() {
        for (value, expected) in [
            (0.333f32, 0.333008f32),
            (0.1, 0.099976),
            (-0.333, -0.333008),
            (1.0 / 3.0, 0.333252),
        ] {
            let back = crate_read::half_to_f32_for_test(f32_to_half(value));
            assert!(
                (back - expected).abs() < 1e-6,
                "{value} became {back}, expected {expected}"
            );
        }
    }

    /// Whatever a half can hold, it holds exactly.
    #[test]
    fn exact_values_are_unchanged() {
        for value in [0.0f32, 1.0, -1.0, 2.0, 0.5, -0.5, 65504.0, -65504.0] {
            let back = crate_read::half_to_f32_for_test(f32_to_half(value));
            assert_eq!(back, value, "{value} did not survive");
        }
    }

    /// And the round trip is never worse than half's own precision, over a
    /// sweep rather than a handful of chosen numbers.
    #[test]
    fn the_error_is_never_worse_than_half_precision() {
        let mut worst = 0.0f32;
        for i in -2000..2000 {
            let value = i as f32 * 0.01;
            let back = crate_read::half_to_f32_for_test(f32_to_half(value));
            // Half has about 11 bits of mantissa: a relative error of 2^-11.
            let tolerance = value.abs() * 0.00049 + 1e-7;
            let error = (back - value).abs();
            assert!(error <= tolerance, "{value} became {back}");
            worst = worst.max(error / (value.abs() + 1e-7));
        }
        // Correct rounding keeps the error within half an ulp, which at the
        // bottom of a binade is 2^-11. Truncation would allow twice that, so
        // this bound is what tells the two apart.
        assert!(worst < 0.000489, "worst relative error {worst}");
        assert!(worst > 0.0001, "suspiciously good — is the sweep hitting anything?");
    }

    /// Rounding up out of the top of the mantissa carries into the exponent.
    #[test]
    fn a_carry_out_of_the_mantissa_is_not_lost() {
        // Just under 1, close enough that rounding lands exactly on it.
        let back = crate_read::half_to_f32_for_test(f32_to_half(0.99999));
        assert_eq!(back, 1.0);
        // And past the largest half, which becomes infinity rather than wrapping.
        assert!(crate_read::half_to_f32_for_test(f32_to_half(1e30)).is_infinite());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{crate_read, parse::parse, write::layer_to_usda};
    use super::super::value::UsdValue;

    /// Write a layer as a crate, read it back, and require it to mean the
    /// same thing.
    ///
    /// Not the same *text*: two differences are inherent to the binary form
    /// and shared with USD's own writer. A document remembers that `3` was
    /// written without a decimal point and the crate does not, so integers
    /// come back as floats; and `color3f` is single precision, so `0.8`
    /// returns as `0.800000011920929`. Both are correct. What must not change
    /// is the structure, the names, and the numbers to within the precision
    /// the declared type actually has.
    fn round_trip(source: &str) -> UsdLayer {
        let original = parse(source).expect("the source parses");
        let bytes = write(&original);
        let back = crate_read::read(&bytes).expect("what we wrote, we can read");
        compare_prims(&original.prims, &back.prims, "");
        for (key, value) in &original.metadata {
            let mine = back
                .meta(key)
                .unwrap_or_else(|| panic!("layer metadata {key} was lost"));
            compare_values(value, mine, key);
        }

        // And it is stable: what comes back writes to the same bytes again, so
        // nothing drifts on a second pass through.
        let again = crate_read::read(&write(&back)).expect("stable");
        assert_eq!(
            layer_to_usda(&again),
            layer_to_usda(&back),
            "a second round trip changed the layer"
        );
        back
    }

    fn compare_prims(want: &[UsdPrim], got: &[UsdPrim], at: &str) {
        assert_eq!(
            want.iter().map(|p| &p.name).collect::<Vec<_>>(),
            got.iter().map(|p| &p.name).collect::<Vec<_>>(),
            "children of {at:?}"
        );
        for (want, got) in want.iter().zip(got) {
            let path = format!("{at}/{}", want.name);
            assert_eq!(want.type_name, got.type_name, "type of {path}");
            assert_eq!(want.specifier, got.specifier, "specifier of {path}");
            assert_eq!(
                want.properties.iter().map(|p| &p.name).collect::<Vec<_>>(),
                got.properties.iter().map(|p| &p.name).collect::<Vec<_>>(),
                "properties of {path}"
            );
            assert_eq!(
                want.variant_sets.iter().map(|s| &s.name).collect::<Vec<_>>(),
                got.variant_sets.iter().map(|s| &s.name).collect::<Vec<_>>(),
                "variant sets of {path}"
            );
            for (want, got) in want.properties.iter().zip(&got.properties) {
                let where_ = format!("{path}.{}", want.name);
                assert_eq!(want.type_name, got.type_name, "type of {where_}");
                assert_eq!(want.uniform, got.uniform, "uniform of {where_}");
                assert_eq!(want.relationship, got.relationship, "rel of {where_}");
                compare_values_of_type(&want.value, &got.value, &where_, &want.type_name);
            }
            compare_prims(&want.children, &got.children, &path);
        }
    }

    fn compare_values(want: &UsdValue, got: &UsdValue, at: &str) {
        compare_values_of_type(want, got, at, "")
    }

    /// The tolerance a value is entitled to depends on the type it is declared
    /// as: `half` keeps about eleven bits, so `0.333` comes back as
    /// `0.3330078` and that is the type working, not the writer failing.
    fn compare_values_of_type(want: &UsdValue, got: &UsdValue, at: &str, type_name: &str) {
        match (want.samples(), got.samples()) {
            (Some(want), Some(got)) => {
                assert_eq!(want.len(), got.len(), "sample count of {at}");
                for (a, b) in want.iter().zip(got) {
                    assert_eq!(a.0, b.0, "sample time of {at}");
                    compare_values_of_type(&a.1, &b.1, at, type_name);
                }
                return;
            }
            (None, None) => {}
            _ => panic!("{at} changed whether it is animated"),
        }
        if let Some(text) = want.as_str() {
            assert_eq!(got.as_str(), Some(text), "{at}");
            return;
        }
        let (want, got) = (want.flat_f32(), got.flat_f32());
        assert_eq!(want.len(), got.len(), "number count of {at}");
        // Half keeps about eleven bits of mantissa; everything else here is at
        // least single precision.
        // `quath[]` is half precision too, so the array brackets come off
        // before the type is judged.
        let base = type_name.strip_suffix("[]").unwrap_or(type_name);
        let relative = if base.starts_with("half") || base.ends_with('h') {
            5e-4
        } else {
            1e-6
        };
        for (a, b) in want.iter().zip(&got) {
            // Exactly equal covers the infinities that a `double` too large
            // for `f32` narrows to, where subtracting gives a NaN and every
            // comparison after it is false.
            if a == b || (a.is_nan() && b.is_nan()) {
                continue;
            }
            assert!(
                (a - b).abs() <= a.abs() * relative + 1e-6,
                "{at}: {a} became {b} (tolerance {relative})"
            );
        }
    }

    #[test]
    fn a_triangle_survives() {
        round_trip(
            r#"#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 0.01
    upAxis = "Y"
)

def Xform "Root"
{
    def Mesh "Tri"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        uniform token subdivisionScheme = "none"
    }
}
"#,
        );
    }

    #[test]
    fn the_documents_this_crate_already_reads_survive() {
        round_trip(include_str!("testdata/rich.usda"));
        round_trip(include_str!("testdata/anim.usda"));
        round_trip(include_str!("testdata/wide.usda"));
        round_trip(include_str!("testdata/roundtrip.usda"));
    }

    /// Composition arcs survive the binary form, in every spelling.
    #[test]
    fn arcs_round_trip() {
        let layer = round_trip(include_str!("testdata/comp/arcs.usda"));
        let a = layer.prim_at("/A").expect("A");

        let refs = a.meta("references").expect("references").references();
        assert_eq!(refs.len(), 3);
        assert_eq!((refs[0].asset.as_str(), refs[0].prim_path.as_str()), ("asset.usda", "/Chair"));
        assert_eq!((refs[1].asset.as_str(), refs[1].prim_path.as_str()), ("other.usda", ""));
        assert_eq!((refs[2].asset.as_str(), refs[2].prim_path.as_str()), ("", "/A/Local"));

        let payload = a.meta("payload").expect("payload").references();
        assert_eq!((payload[0].asset.as_str(), payload[0].prim_path.as_str()), ("heavy.usda", "/Deep"));
        assert_eq!(a.meta("specializes").unwrap().references()[0].prim_path, "/A/Base");

        let b = layer.prim_at("/B").unwrap().meta("references").unwrap().references();
        assert_eq!((b[0].offset, b[0].scale), (10.0, 2.0), "the layer offset");
    }

    /// Variant sets, their bodies, and the selection of one.
    #[test]
    fn variants_round_trip() {
        let layer = round_trip(include_str!("testdata/comp/asset.usda"));
        let chair = layer.prim_at("/Chair").expect("Chair");

        let set = chair
            .variant_sets
            .iter()
            .find(|s| s.name == "look")
            .expect("the look set");
        let mut names: Vec<&str> = set.variants.iter().map(|(n, _)| n.as_str()).collect();
        names.sort();
        assert_eq!(names, vec!["oak", "steel"]);
        assert_eq!(
            set.get("steel").unwrap().value("primvars:displayColor").unwrap().flat_f32(),
            vec![0.7, 0.7, 0.75]
        );
        assert_eq!(chair.meta("inherits").unwrap().references()[0].prim_path, "/Furniture");
    }

    /// And the whole pipeline still composes after a trip through the binary
    /// form — which is the only thing any of this is for.
    #[test]
    fn a_pipeline_still_composes_after_being_written_as_crates() {
        use super::super::compose::{compose, ComposeOptions, MemoryResolver};

        let mut binary = MemoryResolver::new();
        for (name, text) in [
            ("asset.usda", include_str!("testdata/comp/asset.usda")),
            ("base.usda", include_str!("testdata/comp/base.usda")),
        ] {
            binary.insert(name, write(&parse(text).unwrap()));
        }
        let shot = write(&parse(include_str!("testdata/comp/shot.usda")).unwrap());
        let root = crate_read::read(&shot).unwrap();
        let composed = compose(&root, "shot.usda", &binary, &ComposeOptions::default()).unwrap();

        let chair = composed.prim_at("/Room/Chair1").expect("the chair composed");
        assert_eq!(chair.type_name, "Xform");
        assert_eq!(chair.value("xformOp:translate").unwrap().flat_f32(), vec![9.0, 1.0, 0.0]);
        assert_eq!(chair.value("purpose").unwrap().as_str(), Some("render"));
        assert_eq!(
            chair.value("primvars:displayColor").unwrap().flat_f32(),
            vec![0.7, 0.7, 0.75],
            "the shot's variant selection still wins"
        );
        assert!(composed.prim_at("/Room/Chair1/Seat").is_some());
    }

    /// The awkward cases, all at once: empty arrays, integers at the limits of
    /// their types, half precision, a matrix, a quaternion, a `faceVarying`
    /// primvar, a relationship with several targets, two variant sets on one
    /// prim, negative time codes and an escaped string.
    ///
    /// Two bugs came out of this fixture, both silent: halves were truncated
    /// rather than rounded, biasing every one of them toward zero; and a
    /// `uint64` above `i64::MAX` saturated to a different number entirely.
    #[test]
    fn the_awkward_cases_survive() {
        let layer = round_trip(include_str!("testdata/stress.usda"));

        let types = layer.prim_at("/Root/Types").expect("Types");
        // The halves this crate writes are the ones USD would have written:
        // the nearest half to 0.333 is 0.3330078, and truncating gives the one
        // below it, 0.3327637.
        let h = types.value("h").unwrap().flat_f32()[0];
        assert!((h - 0.3330078).abs() < 1e-7, "0.333 became {h}");
        // And the top of the unsigned range is not the top of the signed one.
        assert_eq!(
            types.value("huger").unwrap(),
            &UsdValue::Int(18446744073709551615)
        );
        assert_eq!(
            types.value("huge").unwrap(),
            &UsdValue::Int(-9223372036854775808)
        );
        assert_eq!(types.value("neg").unwrap(), &UsdValue::Int(-2147483648));
        // A double too large for an f32 keeps its full range.
        assert_eq!(
            types.value("d").unwrap().as_f64(),
            Some(1.7976931348623157e308)
        );
        assert_eq!(types.value("big").unwrap(), &UsdValue::Int(4294967295));

        // An empty array is an empty array, not a missing one.
        let empty = layer.prim_at("/Root/Empty").unwrap();
        assert_eq!(empty.value("points").unwrap().flat_f32(), Vec::<f32>::new());
        assert!(empty.property("faceVertexCounts").is_some());

        // Two variant sets on one prim, each with its own choices.
        let multi = layer.prim_at("/Root/Multi").unwrap();
        let mut sets: Vec<&str> = multi.variant_sets.iter().map(|s| s.name.as_str()).collect();
        sets.sort();
        assert_eq!(sets, vec!["lod", "look"]);

        // A relationship with more than one target keeps all of them.
        let many = types.value("many").unwrap().flat_tokens();
        assert_eq!(many.len(), 3, "{many:?}");

        // Time codes may be negative, and samples need not be whole numbers.
        let keyed = layer.prim_at("/Root/Keyed").unwrap();
        let samples = keyed.value("xformOp:translate").unwrap().samples().unwrap();
        assert_eq!(samples[0].0, -12.0);
        assert_eq!(samples[2].0, 0.5);
    }

    /// And the same document read from the crate OpenUSD wrote, rather than
    /// from the one this crate wrote.
    #[test]
    fn the_awkward_cases_read_the_same_from_openusds_own_crate() {
        let theirs = crate_read::read(include_bytes!("testdata/stress.usdc")).expect("reads");
        let ours = parse(include_str!("testdata/stress.usda")).unwrap();
        compare_prims(&ours.prims, &theirs.prims, "");
    }

    /// A mesh big enough to exercise the paths that only large data reaches:
    /// long literal runs in the compressor, integer arrays wide enough to need
    /// four-byte deltas, and a value area past the point where offsets stop
    /// being small.
    #[test]
    fn a_large_mesh_survives() {
        const N: usize = 40_000;
        let points: Vec<UsdValue> = (0..N)
            .map(|i| {
                let f = i as f64;
                UsdValue::Tuple(vec![
                    UsdValue::Float(f * 0.001),
                    UsdValue::Float((f * 0.7).sin()),
                    UsdValue::Float(-f * 0.002),
                ])
            })
            .collect();
        // Indices that jump around, so the deltas cannot all be the common one.
        let indices: Vec<UsdValue> = (0..N)
            .map(|i| UsdValue::Int(((i * 7919) % N) as i128))
            .collect();

        let mut prim = UsdPrim {
            specifier: Specifier::Def,
            type_name: "Mesh".into(),
            name: "Big".into(),
            metadata: Vec::new(),
            properties: Vec::new(),
            children: Vec::new(),
            variant_sets: Vec::new(),
        };
        for (name, type_name, value) in [
            ("points", "point3f[]", UsdValue::Array(points.clone())),
            ("faceVertexIndices", "int[]", UsdValue::Array(indices.clone())),
        ] {
            prim.properties.push(UsdProperty {
                qualifier: String::new(),
                name: name.into(),
                type_name: type_name.into(),
                uniform: false,
                relationship: false,
                value,
                metadata: Vec::new(),
            });
        }
        let layer = UsdLayer {
            metadata: Vec::new(),
            prims: vec![prim],
        };

        let bytes = write(&layer);
        let back = crate_read::read(&bytes).expect("a large crate reads");
        let big = back.prim_at("/Big").expect("Big");

        let got = big.value("points").unwrap().flat_f32();
        assert_eq!(got.len(), N * 3);
        for (i, expected) in points.iter().enumerate() {
            let expected = expected.flat_f32();
            for lane in 0..3 {
                let (a, b) = (expected[lane], got[i * 3 + lane]);
                assert!((a - b).abs() < 1e-6, "point {i} lane {lane}: {a} vs {b}");
            }
        }
        let got = big.value("faceVertexIndices").unwrap().flat_u32();
        let want: Vec<u32> = (0..N).map(|i| ((i * 7919) % N) as u32).collect();
        assert_eq!(got, want);
    }

    /// `apiSchemas` is a token *list operation*, not a token vector.
    ///
    /// Written as a vector it comes back from OpenUSD as an uninterpretable
    /// blob and every schema on the prim is silently lost — anchoring, physics
    /// and material binding alike. Nothing in this crate noticed, because both
    /// halves of it agreed; `usdchecker` did.
    #[test]
    fn api_schemas_survive_as_a_list_operation() {
        let layer = round_trip(include_str!("testdata/apple.usda"));

        let root = layer.prim_at("/Root").expect("Root");
        assert_eq!(
            root.meta("apiSchemas").map(UsdValue::flat_tokens),
            Some(vec!["Preliminary_AnchoringAPI"]),
            "metadata: {:?}",
            root.metadata
        );
        // The qualifier is part of what was said, and it is kept.
        assert!(
            root.meta_exact("prepend apiSchemas").is_some(),
            "the `prepend` was dropped: {:?}",
            root.metadata
        );

        assert_eq!(
            layer
                .prim_at("/Root/Body")
                .unwrap()
                .meta("apiSchemas")
                .map(UsdValue::flat_tokens),
            Some(vec!["Preliminary_PhysicsColliderAPI"])
        );
    }

    /// Apple's AR extensions are ordinary prims with typed attributes, and
    /// they survive because nothing here throws away what it does not
    /// recognise.
    #[test]
    fn apples_ar_schemas_survive() {
        let layer = round_trip(include_str!("testdata/apple.usda"));

        assert_eq!(
            layer.prim_at("/Root").unwrap().value("preliminary:anchoring:type").unwrap().as_str(),
            Some("plane")
        );
        assert_eq!(
            layer.prim_at("/Root/Gravity").unwrap().type_name,
            "Preliminary_PhysicsGravitationalForce"
        );
        assert_eq!(
            layer
                .prim_at("/Root/Gravity")
                .unwrap()
                .value("physics:gravitationalForce:acceleration")
                .unwrap()
                .flat_f32(),
            vec![0.0, -9.8, 0.0]
        );

        // A behaviour, its trigger and its action, with the relationships
        // between them intact.
        let tap = layer.prim_at("/Root/Tap").expect("the behaviour");
        assert_eq!(
            tap.value("preliminary:behavior:triggers").unwrap().as_str(),
            Some("/Root/Tap/Trigger")
        );
        assert_eq!(
            layer.prim_at("/Root/Tap/Trigger").unwrap().value("info:id").unwrap().as_str(),
            Some("TapGesture")
        );
        assert_eq!(
            layer.prim_at("/Root/Label").unwrap().value("content").unwrap().as_str(),
            Some("hello")
        );
    }

    /// And the same asset read from the crate OpenUSD wrote.
    #[test]
    fn apples_ar_schemas_read_from_openusds_own_crate() {
        let theirs = crate_read::read(include_bytes!("testdata/apple.usdc")).expect("reads");
        assert_eq!(
            theirs.prim_at("/Root").unwrap().meta("apiSchemas").map(UsdValue::flat_tokens),
            Some(vec!["Preliminary_AnchoringAPI"]),
            "apiSchemas was lost reading OpenUSD's own file"
        );
        assert_eq!(
            theirs.prim_at("/Root/Label").unwrap().value("height").unwrap().flat_f32(),
            vec![0.1]
        );
    }

    /// A dictionary — `clips`, `customData` — survives the binary form.
    ///
    /// It was being dropped entirely: a dictionary carries no USD type name,
    /// and the writer gave up on any value whose type it could not resolve. So
    /// value clips, which live in one, did not survive `.usdc` at all.
    #[test]
    fn dictionaries_survive() {
        let layer = round_trip(include_str!("testdata/clips/stage.usda"));
        let UsdValue::Dict(outer) = layer.prim_at("/Shot").unwrap().meta("clips").expect("clips")
        else {
            panic!("expected a dictionary");
        };
        assert_eq!(outer.len(), 1);
        assert_eq!(outer[0].0, "default");

        let UsdValue::Dict(inner) = &outer[0].1 else {
            panic!("nested dictionary");
        };
        let field = |name: &str| inner.iter().find(|(k, _)| k == name).map(|(_, v)| v);
        assert_eq!(field("primPath").and_then(|v| v.as_str()), Some("/Cache"));
        assert_eq!(
            field("active").map(UsdValue::flat_f32),
            Some(vec![0.0, 0.0, 2.0, 1.0])
        );
        // An *array* of assets indexes the string table where a single asset
        // indexes the tokens; reading it the other way returns whatever names
        // happen to sit at those indices.
        assert_eq!(
            field("assetPaths").map(|v| v.flat_tokens()),
            Some(vec!["clip_0.usda", "clip_1.usda"])
        );
        assert_eq!(
            field("manifestAssetPath").and_then(|v| v.as_str()),
            Some("manifest.usda")
        );
    }

    /// A relationship's list operation is part of what was said: `prepend rel`
    /// and `rel` compose differently.
    #[test]
    fn a_relationships_qualifier_survives() {
        let layer = round_trip(include_str!("testdata/instancer.usda"));
        let prototypes = layer
            .prim_at("/World/Scatter")
            .unwrap()
            .property("prototypes")
            .expect("the relationship");
        assert!(prototypes.relationship);
        assert_eq!(prototypes.qualifier, "prepend");
        assert_eq!(prototypes.value.flat_tokens().len(), 2);
    }

    /// A document using every part of the grammar: doc comments and
    /// documentation as the separate fields they are, `custom`, `reorder`,
    /// every list-operation qualifier, nested dictionaries with declared
    /// types, sublayer offsets, and a variant with its own body and children.
    #[test]
    fn the_whole_grammar_survives() {
        let layer = round_trip(include_str!("testdata/grammar.usda"));

        // A bare string is `comment`; `documentation` is a different field,
        // and a layer may carry both.
        assert!(layer.meta("comment").unwrap().as_str().unwrap().contains("doc comment"));
        assert!(layer.meta("documentation").unwrap().as_str().unwrap().contains("triple-quoted"));

        // A sublayer's time offset is part of the sublayer.
        let sub = layer.meta("subLayers").unwrap().references();
        assert_eq!((sub[0].offset, sub[0].scale), (5.0, 2.0));

        let root = layer.prim_at("/Root").expect("Root");
        // Every qualifier of one field is one field with several sub-lists.
        assert_eq!(
            root.meta_exact("add references").unwrap().references()[0].prim_path,
            "/Thing"
        );
        assert!(root.meta_exact("delete references").is_some(), "{:?}", root.metadata);
        assert_eq!(root.meta_exact("reorder nameChildren").unwrap().flat_tokens(), vec!["B", "A"]);

        // `custom` is kept, and so are the typed values inside a dictionary.
        let attr = root.property("customAttr").expect("customAttr");
        assert!(attr.metadata.iter().any(|(k, _)| k == "custom"));
        let UsdValue::Dict(custom_data) = root.meta("customData").unwrap() else {
            panic!("a dictionary");
        };
        let nested = custom_data.iter().find(|(k, _)| k == "nested").unwrap();
        let UsdValue::Dict(inner) = &nested.1 else { panic!("nested") };
        assert!(
            matches!(inner[0].1, UsdValue::Token(_)),
            "a `token` declaration makes a token, not a string: {:?}",
            inner[0].1
        );

        // The relationship qualifiers, each on its own property.
        for (name, qualifier) in [
            ("listRel", "prepend"),
            ("listRel2", "append"),
            ("listRel3", "delete"),
            ("listRel4", "reorder"),
        ] {
            assert_eq!(
                root.property(name).unwrap_or_else(|| panic!("{name}")).qualifier,
                qualifier
            );
        }

        // And the variant, with its body and its child.
        let set = root.variant_sets.iter().find(|s| s.name == "style").expect("style");
        assert_eq!(set.get("plain").unwrap().value("inVariant").unwrap().flat_f32(), vec![1.0]);
        assert_eq!(set.get("plain").unwrap().children[0].name, "OnlyInPlain");
    }

    /// `relocates` survives a crate round trip through this crate, though not
    /// under the type USD would give it.
    ///
    /// A relocation map is `SdfRelocatesMap`, which needs crate version
    /// 0.11.0; this writes 0.8.0 and stores it as a dictionary of paths, so
    /// the data comes back intact but USD labels it a dictionary. There is no
    /// ground truth to match against here: the OpenUSD shipped with macOS
    /// cannot write a relocation map into a crate at all — at layer level it
    /// produces a file it then fails to read back, and at prim level it
    /// refuses outright with "Attempted to pack unsupported type
    /// `map<SdfPath, SdfPath>`". A dictionary that survives is more than the
    /// reference implementation manages, and guessing at a layout that cannot
    /// be checked is how a reader comes to mis-read real files.
    #[test]
    fn relocations_survive_a_crate_round_trip() {
        let layer = round_trip(include_str!("testdata/gaps/reloc2.usda"));
        let UsdValue::Dict(entries) = layer.meta("relocates").expect("relocates") else {
            panic!("expected a map");
        };
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, "/Holder/OldName");
        assert_eq!(entries[0].1.as_str(), Some("/Holder/NewName"));
    }

    /// A collection's membership expression survives the crate format, and
    /// carries the file's version up with it.
    ///
    /// `pathExpression` is a 0.10.0 type. Writing one into a file stamped
    /// 0.8.0 produces something USD will not open at all, so the version is
    /// raised for the file that needs it and left alone for every file that
    /// does not — a stage of plain geometry is still a 0.8.0 crate.
    #[test]
    fn a_membership_expression_survives_and_lifts_the_version() {
        let source = include_str!("testdata/gaps/pexpr.usda");
        let layer = round_trip(source);
        let prim = layer.prim_at("/World").expect("World");
        assert_eq!(
            prim.value("collection:a:membershipExpression")
                .and_then(|v| v.as_str().map(str::to_owned)),
            Some("/Alpha//*".to_string())
        );
        assert_eq!(
            prim.value("collection:b:membershipExpression")
                .and_then(|v| v.as_str().map(str::to_owned)),
            Some("/Beta//*".to_string())
        );

        let bytes = super::write(&parse(source).unwrap());
        assert_eq!((bytes[8], bytes[9], bytes[10]), (0, 10, 0));

        // And a file with no such value keeps the older version.
        let plain = super::write(&parse("#usda 1.0\n\ndef Scope \"A\" {}\n").unwrap());
        assert_eq!((plain[8], plain[9], plain[10]), (0, 8, 0));
    }

    /// A time code is written out of line, not inlined. USD reads an inlined
    /// one as zero — a value silently replaced by a different value.
    #[test]
    fn a_time_code_is_not_inlined() {
        let layer = round_trip(include_str!("testdata/block/strong.usda"));
        assert_eq!(
            layer.prim_at("/Root").unwrap().value("when").unwrap().flat_f32(),
            vec![24.0]
        );
    }

    #[test]
    fn a_relationship_keeps_its_target() {
        let layer = round_trip(
            r#"#usda 1.0
def Mesh "M"
{
    rel material:binding = </Mat>
}

def Material "Mat"
{
}
"#,
        );
        let binding = layer.prim_at("/M").unwrap().property("material:binding").unwrap();
        assert!(binding.relationship);
        assert_eq!(binding.value.as_str(), Some("/Mat"));
    }

    /// USD keeps a target that points at nothing, and so must this.
    #[test]
    fn a_target_with_no_prim_behind_it_is_still_written() {
        let layer = round_trip(
            r#"#usda 1.0
def Mesh "M"
{
    rel material:binding = </Nowhere/At/All>
}
"#,
        );
        assert_eq!(
            layer.prim_at("/M").unwrap().value("material:binding").unwrap().as_str(),
            Some("/Nowhere/At/All")
        );
    }

    #[test]
    fn animation_survives() {
        let layer = round_trip(include_str!("testdata/anim.usda"));
        let samples = layer
            .prim_at("/Spin")
            .unwrap()
            .value("xformOp:translate")
            .unwrap()
            .samples()
            .expect("still animated");
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[2].0, 48.0);
        assert_eq!(samples[2].1.flat_f32(), vec![10.0, 10.0, 0.0]);
    }

    #[test]
    fn an_empty_layer_is_a_valid_file() {
        let bytes = write(&UsdLayer::default());
        let info = super::super::usdc::info(&bytes).expect("a readable bootstrap");
        assert_eq!(info.version, (0, 8, 0));
        assert_eq!(info.sections.len(), 6);
        assert!(crate_read::read(&bytes).unwrap().prims.is_empty());
    }

    /// Names, values and whole field sets are shared. Without that a file is
    /// correct and several times larger than it should be, so the test asserts
    /// the size rather than trusting the intent.
    #[test]
    fn repeated_structure_is_interned() {
        let mut source = String::from("#usda 1.0\n");
        for i in 0..200 {
            source.push_str(&format!(
                "def Mesh \"M{i}\"\n{{\n    int[] faceVertexCounts = [3]\n    \
                 uniform token subdivisionScheme = \"none\"\n}}\n"
            ));
        }
        let layer = parse(&source).unwrap();
        let bytes = write(&layer);
        // 200 prims that differ only in name: the tokens, the field sets and
        // the values are all shared, so this is a few bytes each.
        assert!(
            bytes.len() < 6000,
            "200 near-identical prims took {} bytes",
            bytes.len()
        );
        assert_eq!(crate_read::read(&bytes).unwrap().prims.len(), 200);
    }
}

#[cfg(test)]
mod external {
    use super::*;
    use super::super::parse::parse;

    /// Write crate files to disk so OpenUSD's own tools can be pointed at
    /// them. Ignored by default: the assertion that matters is made by
    /// `usdcat`, not by this process.
    #[test]
    #[ignore = "writes files for external validation"]
    fn write_crates_for_usdcat() {
        let dir = std::env::var("USD_OUT").unwrap_or_else(|_| "/tmp".into());
        // `USD_CONVERT=a.usda,b.usda` converts named layers instead, which is
        // how a whole pipeline is turned into crates for `usdcat --flatten` to
        // compose. The layers have to keep their names for the arcs between
        // them to resolve, so each is written beside its source.
        if let Ok(list) = std::env::var("USD_CONVERT") {
            for path in list.split(',').filter(|p| !p.is_empty()) {
                let text = std::fs::read_to_string(path).expect("readable");
                let layer = parse(&text).expect("parses");
                let out = path.replace(".usda", ".usdc");
                std::fs::write(&out, write(&layer)).expect("writable");
                println!("wrote {out}");
            }
            return;
        }
        for (name, source) in [
            ("arcs", include_str!("testdata/comp/arcs.usda")),
            ("variants", include_str!("testdata/comp/asset.usda")),
            ("shot", include_str!("testdata/comp/shot.usda")),
            ("base", include_str!("testdata/comp/base.usda")),
            ("rich", include_str!("testdata/rich.usda")),
            ("anim", include_str!("testdata/anim.usda")),
            ("wide", include_str!("testdata/wide.usda")),
            ("roundtrip", include_str!("testdata/roundtrip.usda")),
        ] {
            let layer = parse(source).expect("parses");
            let bytes = write(&layer);
            let path = format!("{dir}/ours_{name}.usdc");
            std::fs::write(&path, &bytes).unwrap();
            println!("wrote {path} ({} bytes)", bytes.len());
        }
    }
}
