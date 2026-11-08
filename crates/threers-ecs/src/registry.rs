//! Components by name: saves, prefabs and the inspector all read this.
//!
//! Rust keeps no field names at run time, so a type says what it has with
//! [`reflect!`](crate::reflect!), and [`World::register`] adds it to the list. Everything else
//! works from that list.
//!
//! ```
//! use threers_ecs::prelude::*;
//! use threers_ecs::reflect;
//!
//! #[derive(Default)]
//! struct Health { hp: f32, max: f32 }
//! impl Component for Health {}
//! reflect!(Health { hp, max });
//!
//! let mut w = World::new();
//! w.register::<Health>();
//! ```
//!
//! # What a save holds
//!
//! The transforms and registered components of every entity with one, and the
//! registered resources. Not meshes or materials: those come from building the
//! level, so build it first and then call [`World::load`]. Saved entities find
//! their nodes again by name, so give a node a name if it has a look worth keeping.

use crate::world::{Bundle, Component, Entity, World};
use std::collections::HashMap;
use std::rc::Rc;
use threers::core::{Object3D, ObjectKind};
use threers::loaders::json::{self, Value};
use threers::math::{Quaternion, Vector3};

/// One field's value, in a form that can be saved and shown.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldValue {
    /// A `bool`.
    Bool(bool),
    /// Any number, as `f64`.
    Number(f64),
    /// A `String`.
    Text(String),
    /// A [`Vector3`], as `x, y, z`.
    Vector([f32; 3]),
    /// A reference to another entity, or none.
    Entity(Option<Entity>),
}

impl std::fmt::Display for FieldValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FieldValue::Bool(v) => write!(f, "{v}"),
            FieldValue::Number(v) => write!(f, "{v:.3}"),
            FieldValue::Text(v) => write!(f, "{v:?}"),
            FieldValue::Vector([x, y, z]) => write!(f, "({x:.2}, {y:.2}, {z:.2})"),
            FieldValue::Entity(Some(id)) => write!(f, "{id:?}"),
            FieldValue::Entity(None) => write!(f, "none"),
        }
    }
}

/// A type that can be one field of a reflected component.
pub trait Field: Sized {
    /// This value, for saving and showing.
    fn to_field(&self) -> FieldValue;
    /// Back from a field; `None` if it is the wrong kind.
    fn from_field(value: &FieldValue) -> Option<Self>;
}

macro_rules! number_field {
    ($($ty:ty),+) => {
        $(impl Field for $ty {
            fn to_field(&self) -> FieldValue {
                FieldValue::Number(*self as f64)
            }
            fn from_field(value: &FieldValue) -> Option<Self> {
                match value {
                    FieldValue::Number(v) => Some(*v as $ty),
                    _ => None,
                }
            }
        })+
    };
}

number_field!(f32, f64, i32, u32, i64, u64, usize);

impl Field for bool {
    fn to_field(&self) -> FieldValue {
        FieldValue::Bool(*self)
    }
    fn from_field(value: &FieldValue) -> Option<Self> {
        match value {
            FieldValue::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

impl Field for String {
    fn to_field(&self) -> FieldValue {
        FieldValue::Text(self.clone())
    }
    fn from_field(value: &FieldValue) -> Option<Self> {
        match value {
            FieldValue::Text(v) => Some(v.clone()),
            _ => None,
        }
    }
}

impl Field for Vector3 {
    fn to_field(&self) -> FieldValue {
        FieldValue::Vector([self.x, self.y, self.z])
    }
    fn from_field(value: &FieldValue) -> Option<Self> {
        match value {
            FieldValue::Vector([x, y, z]) => Some(Vector3::new(*x, *y, *z)),
            _ => None,
        }
    }
}

impl Field for Option<Entity> {
    fn to_field(&self) -> FieldValue {
        FieldValue::Entity(*self)
    }
    fn from_field(value: &FieldValue) -> Option<Self> {
        match value {
            FieldValue::Entity(v) => Some(*v),
            _ => None,
        }
    }
}

/// Field names and values, in declaration order.
pub type Fields = Vec<(&'static str, FieldValue)>;

/// What [`reflect!`](crate::reflect!) implements.
#[diagnostic::on_unimplemented(
    message = "`{Self}` does not list its fields",
    note = "add `reflect!({Self} {{ field_a, field_b }})`, and derive `Default`"
)]
pub trait Reflect: Default + 'static {
    /// The name saves use, so renaming a type invalidates old saves.
    fn type_name() -> &'static str;
    /// The fields, by name, in declaration order.
    fn fields(&self) -> Fields;
    /// Returns false if there is no such field or the value is the wrong kind.
    fn set_field(&mut self, name: &str, value: &FieldValue) -> bool;
}

/// Declare a type's fields for saves, prefabs and the inspector. Tuple fields go
/// by number, and a type with no data lists nothing.
///
/// ```
/// # use threers_ecs::reflect;
/// #[derive(Default)]
/// struct Speed(f32);
/// reflect!(Speed { 0 });
///
/// #[derive(Default)]
/// struct Enemy;
/// reflect!(Enemy {});
/// ```
#[macro_export]
macro_rules! reflect {
    ($ty:ident { $($field:tt),* $(,)? }) => {
        impl $crate::Reflect for $ty {
            fn type_name() -> &'static str {
                stringify!($ty)
            }
            fn fields(&self) -> $crate::Fields {
                vec![$((stringify!($field), $crate::Field::to_field(&self.$field)),)*]
            }
            #[allow(unused_variables)]
            fn set_field(&mut self, name: &str, value: &$crate::FieldValue) -> bool {
                $(if name == stringify!($field) {
                    return match $crate::Field::from_field(value) {
                        Some(v) => {
                            self.$field = v;
                            true
                        }
                        None => false,
                    };
                })*
                false
            }
        }
    };
}

#[derive(Clone, Copy)]
struct ComponentEntry {
    name: &'static str,
    ids: fn(&World) -> Vec<Entity>,
    read: fn(&World, Entity) -> Option<Fields>,
    write: fn(&mut World, Entity, &[(String, FieldValue)]),
    remove: fn(&mut World, Entity),
    set: fn(&mut World, Entity, &str, &FieldValue) -> bool,
}

#[derive(Clone, Copy)]
struct ResourceEntry {
    name: &'static str,
    read: fn(&World) -> Option<Fields>,
    write: fn(&mut World, &[(String, FieldValue)]),
}

/// Everything registered with a world.
#[derive(Default)]
pub(crate) struct Registry {
    components: Vec<ComponentEntry>,
    resources: Vec<ResourceEntry>,
    after_load: Vec<fn(&mut World)>,
}

fn apply<T: Reflect>(target: &mut T, fields: &[(String, FieldValue)]) {
    for (name, value) in fields {
        // A field that has gone or changed kind keeps its default: an old save
        // should still load after the game adds to a component.
        target.set_field(name, value);
    }
}

impl World {
    /// List a component for saves, prefabs and the inspector.
    pub fn register<C: Component + Reflect>(&mut self) -> &mut Self {
        if self
            .registry
            .components
            .iter()
            .any(|e| e.name == C::type_name())
        {
            return self;
        }
        self.registry.components.push(ComponentEntry {
            name: C::type_name(),
            ids: |w| w.entities_ref::<C>(),
            read: |w, id| w.get::<C>(id).map(Reflect::fields),
            write: |w, id, fields| {
                let mut value = C::default();
                apply(&mut value, fields);
                w.insert(id, value);
            },
            remove: |w, id| {
                w.remove::<C>(id);
            },
            set: |w, id, field, value| {
                w.get_mut::<C>(id)
                    .is_some_and(|c| c.set_field(field, value))
            },
        });
        self
    }

    /// List a resource for saves.
    pub fn register_resource<R: Reflect>(&mut self) -> &mut Self {
        if self
            .registry
            .resources
            .iter()
            .any(|e| e.name == R::type_name())
        {
            return self;
        }
        self.registry.resources.push(ResourceEntry {
            name: R::type_name(),
            read: |w| w.get_resource::<R>().map(Reflect::fields),
            write: |w, fields| {
                let mut value = R::default();
                apply(&mut value, fields);
                w.insert_resource(value);
            },
        });
        self
    }

    /// Attach a registered component by name, returning whether there is one.
    pub fn insert_registered(
        &mut self,
        id: Entity,
        name: &str,
        fields: &[(String, FieldValue)],
    ) -> bool {
        match self.registry.components.iter().find(|e| e.name == name) {
            Some(entry) => {
                let write = entry.write;
                write(self, id, fields);
                true
            }
            None => false,
        }
    }

    /// The registered components' names.
    pub fn registered_names(&self) -> Vec<&'static str> {
        self.registry
            .components
            .iter()
            .map(|entry| entry.name)
            .collect()
    }

    /// Set one field of a registered component, by name, as the inspector does.
    /// Returns false if there is no such component on `id`, no such field, or
    /// the value is the wrong kind.
    pub fn set_field(
        &mut self,
        id: Entity,
        component: &str,
        field: &str,
        value: &FieldValue,
    ) -> bool {
        match self
            .registry
            .components
            .iter()
            .find(|e| e.name == component)
        {
            Some(entry) => (entry.set)(self, id, field, value),
            None => false,
        }
    }

    /// Run `hook` at the end of every [`load`](Self::load), for state kept outside
    /// the saved data that has to follow it.
    pub fn after_load(&mut self, hook: fn(&mut World)) -> &mut Self {
        self.registry.after_load.push(hook);
        self
    }

    /// The registered components on an entity, by name.
    pub fn inspect(&self, id: Entity) -> Vec<(&'static str, Fields)> {
        self.registry
            .components
            .iter()
            .filter_map(|entry| Some((entry.name, (entry.read)(self, id)?)))
            .collect()
    }

    /// Every entity holding a registered component, oldest slot first.
    pub fn inspectable(&self) -> Vec<Entity> {
        let mut ids: Vec<Entity> = self
            .registry
            .components
            .iter()
            .flat_map(|entry| (entry.ids)(self))
            .collect();
        ids.sort();
        ids.dedup();
        ids
    }

    /// The world's saved state as text. See the [module docs](crate::registry).
    pub fn save(&self) -> String {
        let ids = self.inspectable();
        let index: HashMap<Entity, usize> =
            ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let entities = ids
            .iter()
            .filter_map(|&id| {
                let node = self.scene.get(id)?;
                let components = self
                    .inspect(id)
                    .into_iter()
                    .map(|(name, fields)| (name.to_owned(), fields_to_json(&fields, &index)))
                    .collect();
                Some(Value::object([
                    ("name", Value::string(node.name.clone())),
                    ("position", vector(node.position)),
                    (
                        "quaternion",
                        numbers(&[
                            node.quaternion.x,
                            node.quaternion.y,
                            node.quaternion.z,
                            node.quaternion.w,
                        ]),
                    ),
                    ("scale", vector(node.scale)),
                    ("components", Value::Object(components)),
                ]))
            })
            .collect();
        let resources = self
            .registry
            .resources
            .iter()
            .filter_map(|entry| {
                let fields = (entry.read)(self)?;
                Some((entry.name.to_owned(), fields_to_json(&fields, &index)))
            })
            .collect();
        Value::object([
            ("version", Value::number(1)),
            ("entities", Value::Array(entities)),
            ("resources", Value::Object(resources)),
        ])
        .to_json()
    }

    /// Put the world back as [`save`](Self::save) found it: saved entities get
    /// their transforms and components back, and anything with a registered
    /// component that the save does not have is despawned.
    ///
    /// # Errors
    ///
    /// If the text is not a save, or names a type that is not registered. The
    /// world is left untouched.
    pub fn load(&mut self, save: &str) -> Result<(), String> {
        let saved = parse_save(save, &self.registry)?;

        let mut named: HashMap<String, Entity> = HashMap::new();
        for (id, node) in self.scene.arena.nodes.iter() {
            if !node.name.is_empty() {
                named.entry(node.name.clone()).or_insert(id);
            }
        }

        let ids: Vec<Entity> = saved
            .entities
            .iter()
            .map(|entity| match named.get(&entity.name) {
                Some(&id) => id,
                None => {
                    let id = self.spawn(Object3D::group());
                    if let Some(node) = self.scene.get_mut(id) {
                        node.name = entity.name.clone();
                    }
                    id
                }
            })
            .collect();

        let stale: Vec<Entity> = self
            .inspectable()
            .into_iter()
            .filter(|id| !ids.contains(id))
            .collect();
        self.despawn_many(&stale);
        let entries = self.registry.components.clone();
        for &id in &ids {
            for entry in &entries {
                (entry.remove)(self, id);
            }
        }

        let resolve = |index: usize| ids.get(index).copied();
        for (entity, &id) in saved.entities.iter().zip(&ids) {
            if let Some(node) = self.scene.get_mut(id) {
                node.position = entity.position;
                node.quaternion = entity.quaternion;
                node.scale = entity.scale;
            }
            for (name, fields) in &entity.components {
                let fields = resolve_entities(fields, &resolve);
                if let Some(entry) = entries.iter().find(|e| e.name == name) {
                    (entry.write)(self, id, &fields);
                }
            }
        }
        let resources = self.registry.resources.clone();
        for (name, fields) in &saved.resources {
            let fields = resolve_entities(fields, &resolve);
            if let Some(entry) = resources.iter().find(|e| e.name == name) {
                (entry.write)(self, &fields);
            }
        }
        for hook in self.registry.after_load.clone() {
            hook(self);
        }
        Ok(())
    }

    /// Copy an entity and everything under it into a [`Prefab`].
    pub fn capture_prefab(&self, root: Entity) -> Option<Prefab> {
        self.scene.get(root)?;
        let mut order = vec![root];
        let mut parents = vec![None];
        let mut next = 0;
        while next < order.len() {
            if let Some(node) = self.scene.get(order[next]) {
                for &child in &node.children {
                    order.push(child);
                    parents.push(Some(next));
                }
            }
            next += 1;
        }
        let index: HashMap<Entity, usize> =
            order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let nodes = order
            .iter()
            .zip(parents)
            .filter_map(|(&id, parent)| {
                let mut object = self.scene.get(id)?.clone();
                object.parent = None;
                object.children.clear();
                let components = self
                    .inspect(id)
                    .into_iter()
                    .map(|(name, fields)| {
                        let fields = fields
                            .into_iter()
                            .map(|(field, value)| (field.to_owned(), value))
                            .collect();
                        (name, fields)
                    })
                    .collect();
                Some(PrefabNode {
                    object,
                    parent,
                    components,
                })
            })
            .collect();
        Some(Prefab {
            nodes,

            index,
            setup: None,
        })
    }

    /// Make a copy of a [`Prefab`] under the scene root, returning its top entity.
    pub fn spawn_prefab(&mut self, prefab: &Prefab) -> Entity {
        let root = self.scene.root;
        self.spawn_prefab_under(root, prefab)
    }

    /// Make a copy of a [`Prefab`] under `parent`.
    pub fn spawn_prefab_under(&mut self, parent: Entity, prefab: &Prefab) -> Entity {
        let ids: Vec<Entity> = prefab
            .nodes
            .iter()
            .map(|node| self.scene.arena.insert(node.object.clone()))
            .collect();
        for (node, &id) in prefab.nodes.iter().zip(&ids) {
            let to = node.parent.map_or(parent, |index| ids[index]);
            self.scene.arena.add_child(to, id);
        }
        // Bones and entity fields point inside the source; point them here.
        let map = |old: Entity| prefab.index.get(&old).map(|&index| ids[index]);
        for &id in &ids {
            if let Some(ObjectKind::SkinnedMesh(mesh)) = self.scene.get_mut(id).map(|n| &mut n.kind)
            {
                let bones = mesh
                    .skeleton
                    .bones
                    .iter()
                    .map(|bone| threers::core::Bone {
                        node: map(bone.node).unwrap_or(bone.node),
                        inverse_bind: bone.inverse_bind,
                    })
                    .collect();
                mesh.skeleton = std::sync::Arc::new(threers::core::Skeleton::new(bones));
            }
        }
        let entries = self.registry.components.clone();
        for (node, &id) in prefab.nodes.iter().zip(&ids) {
            for (name, fields) in &node.components {
                let fields: Vec<(String, FieldValue)> = fields
                    .iter()
                    .map(|(field, value)| {
                        let value = match value {
                            FieldValue::Entity(Some(old)) => {
                                FieldValue::Entity(map(*old).or(Some(*old)))
                            }
                            other => other.clone(),
                        };
                        (field.clone(), value)
                    })
                    .collect();
                if let Some(entry) = entries.iter().find(|e| e.name == *name) {
                    (entry.write)(self, id, &fields);
                }
            }
        }
        if let Some(setup) = &prefab.setup {
            setup(self, ids[0]);
        }
        ids[0]
    }

    /// Spawn a bare entity carrying `bundle`, for setup code that has no scene
    /// node to hang it on.
    pub fn spawn_with(&mut self, object: Object3D, bundle: impl Bundle) -> Entity {
        let id = self.spawn(object);
        self.insert(id, bundle);
        id
    }
}

type NamedFields = Vec<(String, FieldValue)>;

struct PrefabNode {
    object: Object3D,
    parent: Option<usize>,
    components: Vec<(&'static str, NamedFields)>,
}

/// A copy of a subtree, to stamp out again: nodes with their meshes, and their
/// registered components.
pub struct Prefab {
    nodes: Vec<PrefabNode>,

    index: HashMap<Entity, usize>,
    setup: Option<Rc<Setup>>,
}

type Setup = dyn Fn(&mut World, Entity);

impl Prefab {
    /// Run `setup` on each new copy, for what a prefab cannot hold, such as
    /// attaching a rigid body.
    pub fn with_setup(mut self, setup: impl Fn(&mut World, Entity) + 'static) -> Self {
        self.setup = Some(Rc::new(setup));
        self
    }
}

fn vector(v: Vector3) -> Value {
    numbers(&[v.x, v.y, v.z])
}

fn numbers(values: &[f32]) -> Value {
    Value::Array(values.iter().map(|&v| Value::number(v as f64)).collect())
}

fn fields_to_json(fields: &Fields, index: &HashMap<Entity, usize>) -> Value {
    Value::Object(
        fields
            .iter()
            .map(|(name, value)| ((*name).to_owned(), field_to_json(value, index)))
            .collect(),
    )
}

fn field_to_json(value: &FieldValue, index: &HashMap<Entity, usize>) -> Value {
    match value {
        FieldValue::Bool(v) => Value::Bool(*v),
        FieldValue::Number(v) => Value::Number(*v),
        FieldValue::Text(v) => Value::String(v.clone()),
        FieldValue::Vector([x, y, z]) => numbers(&[*x, *y, *z]),
        FieldValue::Entity(id) => Value::object([(
            "entity",
            id.and_then(|id| index.get(&id))
                .map_or(Value::Null, |&i| Value::number(i as f64)),
        )]),
    }
}

/// A field as saved. Entities are positions in the save until it is applied.
enum SavedField {
    Plain(FieldValue),
    Entity(Option<usize>),
}

type SavedFields = Vec<(String, SavedField)>;

struct SavedEntity {
    name: String,
    position: Vector3,
    quaternion: Quaternion,
    scale: Vector3,
    components: Vec<(String, SavedFields)>,
}

struct Save {
    entities: Vec<SavedEntity>,
    resources: Vec<(String, SavedFields)>,
}

fn resolve_entities(
    fields: &SavedFields,
    resolve: &impl Fn(usize) -> Option<Entity>,
) -> Vec<(String, FieldValue)> {
    fields
        .iter()
        .map(|(name, field)| {
            let value = match field {
                SavedField::Plain(value) => value.clone(),
                SavedField::Entity(index) => FieldValue::Entity(index.and_then(resolve)),
            };
            (name.clone(), value)
        })
        .collect()
}

fn field_from_json(value: &Value) -> Option<SavedField> {
    Some(match value {
        Value::Bool(v) => SavedField::Plain(FieldValue::Bool(*v)),
        Value::Number(v) => SavedField::Plain(FieldValue::Number(*v)),
        Value::String(v) => SavedField::Plain(FieldValue::Text(v.clone())),
        Value::Array(_) => {
            let [x, y, z] = floats::<3>(value)?;
            SavedField::Plain(FieldValue::Vector([x, y, z]))
        }
        Value::Object(_) => {
            SavedField::Entity(value.get("entity")?.as_number().map(|i| i as usize))
        }
        Value::Null => return None,
    })
}

fn floats<const N: usize>(value: &Value) -> Option<[f32; N]> {
    let items = value.as_array()?;
    let mut out = [0.0; N];
    if items.len() != N {
        return None;
    }
    for (slot, item) in out.iter_mut().zip(items) {
        *slot = item.as_number()? as f32;
    }
    Some(out)
}

fn saved_fields(value: &Value) -> Result<SavedFields, String> {
    let object = value.as_object().ok_or("fields are not an object")?;
    let mut fields = Vec::with_capacity(object.len());
    for (name, value) in object {
        let field = field_from_json(value).ok_or_else(|| format!("bad field {name:?}"))?;
        fields.push((name.clone(), field));
    }
    Ok(fields)
}

fn parse_save(text: &str, registry: &Registry) -> Result<Save, String> {
    let root = json::parse(text).map_err(|e| format!("not a save: {e}"))?;
    if root.get("version").and_then(Value::as_number) != Some(1.0) {
        return Err("not a save this version can read".to_owned());
    }
    let mut entities = Vec::new();
    for entity in root
        .get("entities")
        .and_then(Value::as_array)
        .ok_or("save has no entities")?
    {
        let [px, py, pz] = entity
            .get("position")
            .and_then(floats::<3>)
            .ok_or("bad position")?;
        let [qx, qy, qz, qw] = entity
            .get("quaternion")
            .and_then(floats::<4>)
            .ok_or("bad quaternion")?;
        let [sx, sy, sz] = entity
            .get("scale")
            .and_then(floats::<3>)
            .ok_or("bad scale")?;
        let mut components = Vec::new();
        for (name, fields) in entity
            .get("components")
            .and_then(Value::as_object)
            .ok_or("entity has no components")?
        {
            if !registry.components.iter().any(|e| e.name == name) {
                return Err(format!("{name} is not registered"));
            }
            components.push((name.clone(), saved_fields(fields)?));
        }
        entities.push(SavedEntity {
            name: entity
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned(),
            position: Vector3::new(px, py, pz),
            quaternion: Quaternion::new(qx, qy, qz, qw),
            scale: Vector3::new(sx, sy, sz),
            components,
        });
    }
    let mut resources = Vec::new();
    for (name, fields) in root
        .get("resources")
        .and_then(Value::as_object)
        .ok_or("save has no resources")?
    {
        if !registry.resources.iter().any(|e| e.name == name) {
            return Err(format!("{name} is not registered"));
        }
        resources.push((name.clone(), saved_fields(fields)?));
    }
    Ok(Save {
        entities,
        resources,
    })
}
