//! The native [`AnimationHost`]: a mixer animating an
//! [`ObjectArena`](crate::core::ObjectArena) (a [`Scene`](crate::Scene)'s
//! graph).
//!
//! Nodes are [`ObjectId`]s; a node's three.js `uuid` is `@` followed by its
//! id as a number ([`SceneHost::uuid_of`]), so tracks can name nodes that
//! have no name. Bound properties write the scene directly:
//!
//! - `position`, `quaternion`, `scale`, `rotation` (Euler, in the node's
//!   order) and `visible` on any node, bones included (`.bones[name].…`);
//! - `morphTargetInfluences` — all of them, or one by index or by name — on
//!   meshes and skinned meshes;
//! - `material.color` / `emissive` / `opacity` / `roughness` / `metalness`
//!   / `emissiveIntensity` (and `material[i].…` for a material array): a
//!   material shared by several nodes changes for all of them, as a shared
//!   three.js material does;
//! - a light's `color` and `intensity`.
//!
//! Anything else binds to a named scalar slot in a [`PropertySlots`] when the
//! host has one — three.js lets a track animate any property of any object,
//! and a slot is where a native caller reads such a value back.

use std::collections::HashMap;
use std::sync::Arc;

use super::binding::{find_node, AnimationHost, HostKind, NodeTree, ParsedPath};
use super::track::ValueType;
use crate::core::{ObjectArena, ObjectId, ObjectKind};
use crate::materials::Material;
use crate::math::{Color, Euler, Quaternion, Vector3};

/// The native host's node and target types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneKind;

impl HostKind for SceneKind {
    type Node = ObjectId;
    type Target = SceneTarget;
}

/// A material property the native host animates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialProperty {
    Color,
    Emissive,
    EmissiveIntensity,
    Opacity,
    Roughness,
    Metalness,
}

/// A bound property of a native scene.
#[derive(Debug, Clone, PartialEq)]
pub enum SceneTarget {
    Position(ObjectId),
    Quaternion(ObjectId),
    Scale(ObjectId),
    /// `rotation`: Euler angles in the node's rotation order.
    Rotation(ObjectId),
    Visible(ObjectId),
    /// `morphTargetInfluences` as a whole.
    MorphInfluences(ObjectId),
    /// `morphTargetInfluences[i]`.
    MorphInfluence(ObjectId, usize),
    /// A property of the node's material (`index`: in its material array).
    Material {
        node: ObjectId,
        index: Option<usize>,
        property: MaterialProperty,
    },
    LightColor(ObjectId),
    LightIntensity(ObjectId),
    /// A named slot in the host's [`PropertySlots`], `width` values wide.
    Slot(ObjectId, String, usize),
}

/// Values for properties the scene has no field for, by node and path
/// (the track name without its node part, e.g. `userData[speed].x`).
#[derive(Debug, Clone, Default)]
pub struct PropertySlots {
    values: HashMap<(ObjectId, String), Vec<f64>>,
}

impl PropertySlots {
    pub fn new() -> Self {
        Self::default()
    }
    /// The values last written to a slot.
    pub fn get(&self, node: ObjectId, path: &str) -> Option<&[f64]> {
        self.values
            .get(&(node, path.to_string()))
            .map(|v| v.as_slice())
    }
    /// Set a slot (its value before an animation starts).
    pub fn set(&mut self, node: ObjectId, path: &str, values: &[f64]) {
        self.values
            .insert((node, path.to_string()), values.to_vec());
    }
}

/// The native host, borrowing a scene graph (and optionally slots) for one
/// mixer call.
pub struct SceneHost<'a> {
    pub arena: &'a mut ObjectArena,
    pub slots: Option<&'a mut PropertySlots>,
}

impl<'a> SceneHost<'a> {
    pub fn new(arena: &'a mut ObjectArena) -> Self {
        Self { arena, slots: None }
    }
    pub fn with_slots(arena: &'a mut ObjectArena, slots: &'a mut PropertySlots) -> Self {
        Self {
            arena,
            slots: Some(slots),
        }
    }
    /// The uuid a native node answers to: `@` and its id as a number.
    pub fn uuid_of(id: ObjectId) -> String {
        use slotmap::Key;
        format!("@{}", id.data().as_ffi())
    }
}

struct Tree<'b>(&'b ObjectArena);

impl NodeTree for Tree<'_> {
    type Id = ObjectId;
    fn name_is(&self, node: &ObjectId, name: &str) -> bool {
        self.0
            .nodes
            .get(*node)
            .map(|o| o.name == name)
            .unwrap_or(false)
    }
    fn uuid_is(&self, node: &ObjectId, uuid: &str) -> bool {
        uuid.starts_with('@') && SceneHost::uuid_of(*node) == uuid
    }
    fn skeleton_bones(&self, node: &ObjectId) -> Option<Vec<ObjectId>> {
        match &self.0.nodes.get(*node)?.kind {
            ObjectKind::SkinnedMesh(sm) => Some(sm.skeleton.bones.iter().map(|b| b.node).collect()),
            _ => None,
        }
    }
    fn children(&self, node: &ObjectId) -> Vec<ObjectId> {
        self.0
            .nodes
            .get(*node)
            .map(|o| o.children.clone())
            .unwrap_or_default()
    }
}

fn material_property(name: &str) -> Option<MaterialProperty> {
    Some(match name {
        "color" => MaterialProperty::Color,
        "emissive" => MaterialProperty::Emissive,
        "emissiveIntensity" => MaterialProperty::EmissiveIntensity,
        "opacity" => MaterialProperty::Opacity,
        "roughness" => MaterialProperty::Roughness,
        "metalness" => MaterialProperty::Metalness,
        _ => return None,
    })
}

fn material_get(m: &Material, p: MaterialProperty) -> Option<Vec<f64>> {
    let c3 = |c: Color| Some(vec![c.r as f64, c.g as f64, c.b as f64]);
    match p {
        MaterialProperty::Color => c3(m.color()),
        MaterialProperty::Opacity => Some(vec![m.opacity() as f64]),
        MaterialProperty::Emissive => match m {
            Material::Lambert(x) => c3(x.emissive),
            Material::Phong(x) => c3(x.emissive),
            Material::Standard(x) => c3(x.emissive),
            Material::Physical(x) => c3(x.emissive),
            Material::Toon(x) => c3(x.emissive),
            _ => None,
        },
        MaterialProperty::EmissiveIntensity => match m {
            Material::Standard(x) => Some(vec![x.emissive_intensity as f64]),
            Material::Physical(x) => Some(vec![x.emissive_intensity as f64]),
            _ => None,
        },
        MaterialProperty::Roughness => match m {
            Material::Standard(_) | Material::Physical(_) => Some(vec![m.roughness() as f64]),
            _ => None,
        },
        MaterialProperty::Metalness => match m {
            Material::Standard(_) | Material::Physical(_) => Some(vec![m.metalness() as f64]),
            _ => None,
        },
    }
}

fn color_of(v: &[f64]) -> Color {
    let g = |i: usize| v.get(i).copied().unwrap_or(f64::NAN) as f32;
    Color::new(g(0), g(1), g(2))
}

/// Write `p` into `m`; false when the material has no such property.
fn material_set(m: &mut Material, p: MaterialProperty, v: &[f64]) -> bool {
    let s = v.first().copied().unwrap_or(f64::NAN) as f32;
    match p {
        MaterialProperty::Color => {
            let c = color_of(v);
            match m {
                Material::Basic(x) => x.color = c,
                Material::Lambert(x) => x.color = c,
                Material::Phong(x) => x.color = c,
                Material::Standard(x) => x.color = c,
                Material::Physical(x) => x.color = c,
                Material::Toon(x) => x.color = c,
                Material::Matcap(x) => x.color = c,
                Material::Line(x) => x.color = c,
                Material::Points(x) => x.color = c,
                Material::Sprite(x) => x.color = c,
                Material::Mirror(x) => x.color = c,
                Material::Atmosphere(x) => x.color = c,
                _ => return false,
            }
        }
        MaterialProperty::Emissive => {
            let c = color_of(v);
            match m {
                Material::Lambert(x) => x.emissive = c,
                Material::Phong(x) => x.emissive = c,
                Material::Standard(x) => x.emissive = c,
                Material::Physical(x) => x.emissive = c,
                Material::Toon(x) => x.emissive = c,
                _ => return false,
            }
        }
        MaterialProperty::EmissiveIntensity => match m {
            Material::Standard(x) => x.emissive_intensity = s,
            Material::Physical(x) => x.emissive_intensity = s,
            _ => return false,
        },
        MaterialProperty::Opacity => match m {
            Material::Basic(x) => x.opacity = s,
            Material::Lambert(x) => x.opacity = s,
            Material::Phong(x) => x.opacity = s,
            Material::Standard(x) => x.opacity = s,
            Material::Physical(x) => x.opacity = s,
            Material::Normal(x) => x.opacity = s,
            Material::Depth(x) => x.opacity = s,
            Material::Toon(x) => x.opacity = s,
            Material::Matcap(x) => x.opacity = s,
            Material::Line(x) => x.opacity = s,
            Material::Points(x) => x.opacity = s,
            Material::Sprite(x) => x.opacity = s,
            Material::Atmosphere(x) => x.opacity = s,
            Material::Shader(x) => x.opacity = s,
            _ => return false,
        },
        MaterialProperty::Roughness => match m {
            Material::Standard(x) => x.roughness = s,
            Material::Physical(x) => x.roughness = s,
            _ => return false,
        },
        MaterialProperty::Metalness => match m {
            Material::Standard(x) => x.metalness = s,
            Material::Physical(x) => x.metalness = s,
            _ => return false,
        },
    }
    true
}

/// The material a node draws with (`index`: of its material array).
fn node_material(
    arena: &ObjectArena,
    node: ObjectId,
    index: Option<usize>,
) -> Option<&Arc<Material>> {
    let (single, list): (&Arc<Material>, &[Arc<Material>]) = match &arena.nodes.get(node)?.kind {
        ObjectKind::Mesh(m) => (&m.material, &m.materials),
        ObjectKind::SkinnedMesh(m) => (&m.material, &m.materials),
        ObjectKind::InstancedMesh(m) => (&m.material, &m.materials),
        ObjectKind::LineSegments(m) => (&m.material, &[]),
        ObjectKind::Points(m) => (&m.material, &[]),
        ObjectKind::Sprite(m) => (&m.material, &[]),
        _ => return None,
    };
    match index {
        None => Some(single),
        Some(i) => list.get(i),
    }
}

fn morph_influences(arena: &mut ObjectArena, node: ObjectId) -> Option<&mut Vec<f32>> {
    match &mut arena.nodes.get_mut(node)?.kind {
        ObjectKind::Mesh(m) => Some(&mut m.morph_influences),
        ObjectKind::SkinnedMesh(m) => Some(&mut m.morph_influences),
        _ => None,
    }
}

impl AnimationHost<SceneKind> for SceneHost<'_> {
    fn node_uuid(&self, node: &ObjectId) -> String {
        Self::uuid_of(*node)
    }

    fn find_node(&mut self, root: &ObjectId, node_name: Option<&str>) -> Option<ObjectId> {
        // `@<id>` is a node's uuid: found directly, when it is the root or
        // under it (where three.js's search would find it).
        if let Some(id) = node_name
            .and_then(|n| n.strip_prefix('@'))
            .and_then(|n| n.parse::<u64>().ok())
        {
            let id: ObjectId = slotmap::KeyData::from_ffi(id).into();
            let mut at = Some(id).filter(|i| self.arena.nodes.contains_key(*i));
            while let Some(n) = at {
                if n == *root {
                    return Some(id);
                }
                at = self.arena.nodes.get(n).and_then(|o| o.parent);
            }
            return None;
        }
        find_node(&Tree(self.arena), root, node_name)
    }

    fn bind(
        &mut self,
        node: &ObjectId,
        parsed: &ParsedPath,
        path: &str,
        _value_type: ValueType,
        value_size: usize,
    ) -> Option<SceneTarget> {
        let mut node = *node;
        let obj = self.arena.nodes.get(node)?;
        let prop = parsed.property_name.as_str();
        let slot_key = || -> String {
            // the path past the node name
            let mut k = String::new();
            if let Some(o) = &parsed.object_name {
                k.push_str(o);
                if let Some(i) = &parsed.object_index {
                    k.push_str(&format!("[{i}]"));
                }
                k.push('.');
            }
            k.push_str(&parsed.property_name);
            if let Some(i) = &parsed.property_index {
                k.push_str(&format!("[{i}]"));
            }
            k
        };
        let slot = |host: &Self, node: ObjectId| -> Option<SceneTarget> {
            if host.slots.is_some() {
                Some(SceneTarget::Slot(node, slot_key(), value_size.max(1)))
            } else {
                log::error!(
                    "THREE.PropertyBinding: Trying to update property for track: {path} but it wasn't found."
                );
                None
            }
        };
        let mut material_index: Option<Option<usize>> = None;
        match parsed.object_name.as_deref() {
            None => {}
            Some("bones") => {
                let ObjectKind::SkinnedMesh(sm) = &obj.kind else {
                    log::error!("THREE.PropertyBinding: Can not bind to bones as node does not have a skeleton.");
                    return None;
                };
                let bones: Vec<ObjectId> = sm.skeleton.bones.iter().map(|b| b.node).collect();
                let idx = parsed.object_index.as_deref().unwrap_or("");
                let found = bones
                    .iter()
                    .copied()
                    .find(|b| {
                        self.arena
                            .nodes
                            .get(*b)
                            .map(|o| o.name == idx)
                            .unwrap_or(false)
                    })
                    .or_else(|| {
                        idx.parse::<usize>()
                            .ok()
                            .and_then(|i| bones.get(i).copied())
                    });
                let Some(bone) = found else {
                    log::error!(
                        "THREE.PropertyBinding: Trying to bind to objectIndex of objectName, but is undefined."
                    );
                    return None;
                };
                node = bone;
            }
            Some("material") => {
                let index = match parsed.object_index.as_deref() {
                    None => None,
                    Some(i) => match i.parse::<usize>() {
                        Ok(i) => Some(i),
                        Err(_) => return slot(self, node),
                    },
                };
                if node_material(self.arena, node, index).is_none() {
                    log::error!(
                        "THREE.PropertyBinding: Can not bind to material as node does not have a material."
                    );
                    return None;
                }
                material_index = Some(index);
            }
            // `material.materials` (three's old MultiMaterial) and `map`
            // (a texture's transform) have nothing to bind to here.
            Some(_) => return slot(self, node),
        }
        let obj = self.arena.nodes.get(node)?;
        if let Some(index) = material_index {
            let mat = node_material(self.arena, node, index)?;
            return match material_property(prop) {
                Some(p) if material_get(mat, p).is_some() => Some(SceneTarget::Material {
                    node,
                    index,
                    property: p,
                }),
                _ => slot(self, node),
            };
        }
        let target = match (prop, parsed.property_index.as_deref()) {
            ("position", None) => SceneTarget::Position(node),
            ("quaternion", None) => SceneTarget::Quaternion(node),
            ("scale", None) => SceneTarget::Scale(node),
            ("rotation", None) => SceneTarget::Rotation(node),
            ("visible", None) => SceneTarget::Visible(node),
            ("morphTargetInfluences", index) => {
                let (dict, geom_names) = match &obj.kind {
                    ObjectKind::Mesh(m) => {
                        (&m.morph_target_dictionary, &m.geometry.morph_target_names)
                    }
                    ObjectKind::SkinnedMesh(m) => {
                        (&m.morph_target_dictionary, &m.geometry.morph_target_names)
                    }
                    _ => return slot(self, node),
                };
                match index {
                    None => SceneTarget::MorphInfluences(node),
                    Some(i) => {
                        let resolved = dict
                            .get(i)
                            .copied()
                            .or_else(|| geom_names.iter().position(|n| n == i))
                            .or_else(|| i.parse::<usize>().ok());
                        match resolved {
                            Some(i) => SceneTarget::MorphInfluence(node, i),
                            None => {
                                log::error!(
                                    "THREE.PropertyBinding: no morph target named {i} for track {path}"
                                );
                                return None;
                            }
                        }
                    }
                }
            }
            ("color", None) if matches!(obj.kind, ObjectKind::Light(_)) => {
                SceneTarget::LightColor(node)
            }
            ("intensity", None) if matches!(obj.kind, ObjectKind::Light(_)) => {
                SceneTarget::LightIntensity(node)
            }
            _ => return slot(self, node),
        };
        Some(target)
    }

    fn get_value(&mut self, target: &SceneTarget, out: &mut [f64]) {
        let mut put = |vals: &[f64]| {
            for (o, v) in out.iter_mut().zip(vals) {
                *o = *v;
            }
        };
        let v3 = |v: Vector3| [v.x as f64, v.y as f64, v.z as f64];
        match target {
            SceneTarget::Position(n) => {
                if let Some(o) = self.arena.nodes.get(*n) {
                    put(&v3(o.position));
                }
            }
            SceneTarget::Scale(n) => {
                if let Some(o) = self.arena.nodes.get(*n) {
                    put(&v3(o.scale));
                }
            }
            SceneTarget::Quaternion(n) => {
                if let Some(o) = self.arena.nodes.get(*n) {
                    let q = o.quaternion;
                    put(&[q.x as f64, q.y as f64, q.z as f64, q.w as f64]);
                }
            }
            SceneTarget::Rotation(n) => {
                if let Some(o) = self.arena.nodes.get(*n) {
                    let e = o.rotation();
                    put(&[e.x as f64, e.y as f64, e.z as f64]);
                }
            }
            SceneTarget::Visible(n) => {
                if let Some(o) = self.arena.nodes.get(*n) {
                    put(&[o.visible as u8 as f64]);
                }
            }
            SceneTarget::MorphInfluences(n) => {
                if let Some(inf) = morph_influences(self.arena, *n) {
                    let vals: Vec<f64> = inf.iter().map(|&v| v as f64).collect();
                    put(&vals);
                }
            }
            SceneTarget::MorphInfluence(n, i) => {
                if let Some(inf) = morph_influences(self.arena, *n) {
                    put(&[inf.get(*i).map(|&v| v as f64).unwrap_or(f64::NAN)]);
                }
            }
            SceneTarget::Material {
                node,
                index,
                property,
            } => {
                if let Some(vals) = node_material(self.arena, *node, *index)
                    .and_then(|m| material_get(m, *property))
                {
                    put(&vals);
                }
            }
            SceneTarget::LightColor(n) => {
                if let Some(ObjectKind::Light(l)) = self.arena.nodes.get(*n).map(|o| &o.kind) {
                    let c = l.color();
                    put(&[c.r as f64, c.g as f64, c.b as f64]);
                }
            }
            SceneTarget::LightIntensity(n) => {
                if let Some(ObjectKind::Light(l)) = self.arena.nodes.get(*n).map(|o| &o.kind) {
                    put(&[l.intensity() as f64]);
                }
            }
            SceneTarget::Slot(n, key, _) => {
                if let Some(vals) = self.slots.as_ref().and_then(|s| s.get(*n, key)) {
                    put(vals);
                }
            }
        }
    }

    fn set_value(&mut self, target: &SceneTarget, values: &[f64]) {
        let g = |i: usize| values.get(i).copied().unwrap_or(f64::NAN) as f32;
        match target {
            SceneTarget::Position(n) => {
                if let Some(o) = self.arena.nodes.get_mut(*n) {
                    o.position = Vector3::new(g(0), g(1), g(2));
                }
            }
            SceneTarget::Scale(n) => {
                if let Some(o) = self.arena.nodes.get_mut(*n) {
                    o.scale = Vector3::new(g(0), g(1), g(2));
                }
            }
            SceneTarget::Quaternion(n) => {
                if let Some(o) = self.arena.nodes.get_mut(*n) {
                    o.quaternion = Quaternion::new(g(0), g(1), g(2), g(3));
                }
            }
            SceneTarget::Rotation(n) => {
                if let Some(o) = self.arena.nodes.get_mut(*n) {
                    let order = o.rotation_order;
                    o.set_rotation(Euler::with_order(g(0), g(1), g(2), order));
                }
            }
            SceneTarget::Visible(n) => {
                if let Some(o) = self.arena.nodes.get_mut(*n) {
                    let v = values.first().copied().unwrap_or(0.0);
                    o.visible = v != 0.0 && !v.is_nan();
                }
            }
            SceneTarget::MorphInfluences(n) => {
                if let Some(inf) = morph_influences(self.arena, *n) {
                    for (i, d) in inf.iter_mut().enumerate() {
                        *d = values.get(i).copied().unwrap_or(f64::NAN) as f32;
                    }
                }
            }
            SceneTarget::MorphInfluence(n, i) => {
                if let Some(inf) = morph_influences(self.arena, *n) {
                    if inf.len() <= *i {
                        inf.resize(*i + 1, 0.0);
                    }
                    inf[*i] = g(0);
                }
            }
            SceneTarget::Material {
                node,
                index,
                property,
            } => {
                let Some(old) = node_material(self.arena, *node, *index).cloned() else {
                    return;
                };
                let mut m = (*old).clone();
                if !material_set(&mut m, *property, values) {
                    return;
                }
                let new = Arc::new(m);
                // Every node drawing with the same material sees the change.
                for (_, o) in self.arena.nodes.iter_mut() {
                    let slots: Vec<&mut Arc<Material>> = match &mut o.kind {
                        ObjectKind::Mesh(x) => std::iter::once(&mut x.material)
                            .chain(x.materials.iter_mut())
                            .collect(),
                        ObjectKind::SkinnedMesh(x) => std::iter::once(&mut x.material)
                            .chain(x.materials.iter_mut())
                            .collect(),
                        ObjectKind::InstancedMesh(x) => std::iter::once(&mut x.material)
                            .chain(x.materials.iter_mut())
                            .collect(),
                        ObjectKind::LineSegments(x) => vec![&mut x.material],
                        ObjectKind::Points(x) => vec![&mut x.material],
                        ObjectKind::Sprite(x) => vec![&mut x.material],
                        _ => Vec::new(),
                    };
                    for s in slots {
                        if Arc::ptr_eq(s, &old) {
                            *s = new.clone();
                        }
                    }
                }
            }
            SceneTarget::LightColor(n) => {
                if let Some(o) = self.arena.nodes.get_mut(*n) {
                    if let ObjectKind::Light(l) = &mut o.kind {
                        l.set_color(color_of(values));
                    }
                }
            }
            SceneTarget::LightIntensity(n) => {
                if let Some(o) = self.arena.nodes.get_mut(*n) {
                    if let ObjectKind::Light(l) = &mut o.kind {
                        l.set_intensity(g(0));
                    }
                }
            }
            SceneTarget::Slot(n, key, width) => {
                if let Some(slots) = self.slots.as_mut() {
                    slots.set(*n, key, &values[..(*width).min(values.len())]);
                }
            }
        }
    }
}
