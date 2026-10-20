//! Named parameters: how a pass's three.js properties and uniforms are read
//! and written from outside it.
//!
//! three.js passes are configured by assigning properties
//! (`bloomPass.strength = 2`) and uniform values
//! (`filmPass.uniforms.intensity.value = 0.8`). Every pass here describes the
//! same names in [`Pass::params`](super::Pass::params) and answers
//! [`get`](super::Pass::get) / [`set`](super::Pass::set) for them, so a binding
//! — the browser shim's pass classes — can forward each property to the Rust
//! field behind it rather than keeping a copy of its own.

use std::sync::Arc;

use crate::core::ObjectId;
use crate::textures::{CubeTexture, Texture};

/// The shape of a parameter, which decides how a binding presents it: a
/// number, a `Vector2`, a `Color`, a texture, a list of objects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamKind {
    Float,
    Int,
    Bool,
    Vec2,
    Vec3,
    Vec4,
    /// An RGB colour, linear, as three.js `Color` holds it.
    Color,
    /// A list of numbers (`bloomTintColors` flattened, a kernel).
    Floats,
    Texture,
    CubeTexture,
    /// Scene objects, by id (`OutlinePass.selectedObjects`).
    Objects,
}

/// One parameter: its three.js name, its shape, and whether three.js has it
/// under `pass.uniforms[name].value` rather than as `pass[name]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamInfo {
    pub name: &'static str,
    pub kind: ParamKind,
    pub uniform: bool,
}

impl ParamInfo {
    pub const fn prop(name: &'static str, kind: ParamKind) -> Self {
        Self {
            name,
            kind,
            uniform: false,
        }
    }

    pub const fn uniform(name: &'static str, kind: ParamKind) -> Self {
        Self {
            name,
            kind,
            uniform: true,
        }
    }
}

/// A parameter's value.
#[derive(Clone, Debug)]
pub enum ParamValue {
    Float(f32),
    Int(i32),
    Bool(bool),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Color([f32; 3]),
    Floats(Vec<f32>),
    Texture(Option<Arc<Texture>>),
    CubeTexture(Option<Arc<CubeTexture>>),
    Objects(Vec<ObjectId>),
}

impl ParamValue {
    /// As a number: a float, an int, or a bool as 0 / 1 — three.js uniforms
    /// are loose about which (`useTexture` is an int set from a bool).
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Int(v) => Some(*v as f32),
            Self::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
            Self::Floats(v) if v.len() == 1 => Some(v[0]),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<i32> {
        self.as_f32().map(|v| v as i32)
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            other => other.as_f32().map(|v| v != 0.0),
        }
    }

    /// The first `N` numbers of a vector, colour or list.
    pub fn as_array<const N: usize>(&self) -> Option<[f32; N]> {
        let s: &[f32] = match self {
            Self::Vec2(v) => v,
            Self::Vec3(v) | Self::Color(v) => v,
            Self::Vec4(v) => v,
            Self::Floats(v) => v,
            _ => return None,
        };
        if s.len() < N {
            return None;
        }
        let mut out = [0.0; N];
        out.copy_from_slice(&s[..N]);
        Some(out)
    }

    pub fn as_floats(&self) -> Option<Vec<f32>> {
        match self {
            Self::Floats(v) => Some(v.clone()),
            Self::Vec2(v) => Some(v.to_vec()),
            Self::Vec3(v) | Self::Color(v) => Some(v.to_vec()),
            Self::Vec4(v) => Some(v.to_vec()),
            other => other.as_f32().map(|v| vec![v]),
        }
    }
}

/// Assign `value` to a field if it has the field's shape; `false` if not,
/// which [`Pass::set`](super::Pass::set) reports as the name not taking it.
pub(crate) fn assign_f32(field: &mut f32, value: &ParamValue) -> bool {
    match value.as_f32() {
        Some(v) => {
            *field = v;
            true
        }
        None => false,
    }
}

pub(crate) fn assign_bool(field: &mut bool, value: &ParamValue) -> bool {
    match value.as_bool() {
        Some(v) => {
            *field = v;
            true
        }
        None => false,
    }
}

pub(crate) fn assign_array<const N: usize>(field: &mut [f32; N], value: &ParamValue) -> bool {
    match value.as_array::<N>() {
        Some(v) => {
            *field = v;
            true
        }
        None => false,
    }
}
