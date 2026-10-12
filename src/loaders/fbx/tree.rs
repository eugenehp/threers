//! An FBX document as the node tree it is, whichever form it arrived in.
//!
//! FBX is two serialisations of one structure: a tree of *nodes*, each with a
//! name, a list of properties, and children. The binary form writes that tree
//! with byte offsets and zlib-compressed arrays; the text form writes the same
//! tree with braces. Everything above this — geometry, materials, the scene
//! graph — reads the tree and does not care which form it came from.
//!
//! The loader used to skip this step, searching the bytes for the first
//! `Vertices` it could find. That reads one mesh out of a file that may hold
//! fifty, and cannot see a normal, a UV or a parent, because knowing what those
//! belong to means knowing where you are in the tree.

use std::collections::HashMap;

/// One property of a node. FBX distinguishes scalar types it could have shared,
/// so a reader has to accept whichever the writer happened to choose.
#[derive(Debug, Clone, PartialEq)]
pub enum Property {
    Bool(bool),
    Short(i16),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    String(String),
    Raw(Vec<u8>),
    Floats(Vec<f32>),
    Doubles(Vec<f64>),
    Ints(Vec<i32>),
    Longs(Vec<i64>),
    Bools(Vec<bool>),
}

impl Property {
    /// The property as an `f64`, whichever numeric type it is stored as.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Property::Float(v) => Some(*v as f64),
            Property::Double(v) => Some(*v),
            Property::Int(v) => Some(*v as f64),
            Property::Long(v) => Some(*v as f64),
            Property::Short(v) => Some(*v as f64),
            Property::Bool(v) => Some(*v as u8 as f64),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Property::Int(v) => Some(*v as i64),
            Property::Long(v) => Some(*v),
            Property::Short(v) => Some(*v as i64),
            Property::Double(v) => Some(*v as i64),
            Property::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Property::String(s) => Some(s),
            _ => None,
        }
    }

    /// A numeric array as `f32`s, whatever width it was written at.
    ///
    /// FBX writes vertex positions as doubles and normals sometimes as floats,
    /// in the same file, so asking for one width and taking nothing otherwise
    /// silently loses attributes.
    pub fn as_f32_array(&self) -> Option<Vec<f32>> {
        match self {
            Property::Floats(v) => Some(v.clone()),
            Property::Doubles(v) => Some(v.iter().map(|d| *d as f32).collect()),
            Property::Ints(v) => Some(v.iter().map(|i| *i as f32).collect()),
            Property::Longs(v) => Some(v.iter().map(|i| *i as f32).collect()),
            _ => None,
        }
    }

    /// An integer array as `i32`s. Indices are `i32` in a 7.x file and `i64` in
    /// some 7.5 writers.
    pub fn as_i32_array(&self) -> Option<Vec<i32>> {
        match self {
            Property::Ints(v) => Some(v.clone()),
            Property::Longs(v) => Some(v.iter().map(|i| *i as i32).collect()),
            _ => None,
        }
    }
}

/// A node: its name, its properties, and whatever is nested inside it.
#[derive(Debug, Clone, Default)]
pub struct Node {
    pub name: String,
    pub properties: Vec<Property>,
    pub children: Vec<Node>,
}

impl Node {
    /// The first child with this name.
    pub fn child(&self, name: &str) -> Option<&Node> {
        self.children.iter().find(|c| c.name == name)
    }

    /// Every child with this name. `Objects` holds many `Geometry` nodes and
    /// many `Model` nodes as siblings, so this is the common case.
    pub fn children_named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Node> + 'a {
        self.children.iter().filter(move |c| c.name == name)
    }

    /// The first property of the named child — the shape most scalars take.
    pub fn value(&self, name: &str) -> Option<&Property> {
        self.child(name)?.properties.first()
    }

    /// A `Properties70` entry by name, which is where FBX keeps everything it
    /// decided was not structural: transforms, colours, visibility.
    ///
    /// Each is a `P` node whose first property is the name and whose *fifth*
    /// onward are the values — the three between are the type, a label and a
    /// flags string, and they are not always the same, so the values are taken
    /// by position from the end of the fixed prefix rather than by matching.
    pub fn property70(&self, name: &str) -> Option<&[Property]> {
        let bag = self.child("Properties70").or_else(|| self.child("Properties60"))?;
        for p in bag.children_named("P") {
            if p.properties.first().and_then(Property::as_str) == Some(name) {
                return p.properties.get(4..);
            }
        }
        None
    }

    /// Three numbers out of a `Properties70` entry, for the transform fields.
    pub fn property70_vec3(&self, name: &str) -> Option<[f32; 3]> {
        let values = self.property70(name)?;
        if values.len() < 3 {
            return None;
        }
        Some([
            values[0].as_f64()? as f32,
            values[1].as_f64()? as f32,
            values[2].as_f64()? as f32,
        ])
    }

    /// Every node in the document keyed by its FBX id, for resolving
    /// `Connections`.
    ///
    /// An object's id is its first property. Ids are unique across the file,
    /// which is what lets connections be a flat list of number pairs.
    pub fn by_id(&self) -> HashMap<i64, &Node> {
        let mut out = HashMap::new();
        if let Some(objects) = self.child("Objects") {
            for node in &objects.children {
                if let Some(id) = node.properties.first().and_then(Property::as_i64) {
                    out.insert(id, node);
                }
            }
        }
        out
    }
}
