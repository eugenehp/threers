//! Many still meshes as few draws.
//!
//! Mark scenery [`Static`], or spawn it with [`Spawned::static_mesh`](crate::Spawned::static_mesh),
//! and call [`World::bake_static`] once it is all placed. Meshes with the same
//! material and shadow settings become one mesh, and one draw. The originals stay,
//! hidden, so their bodies and components go on working.

use crate::query::{NodeRef, With};
use crate::world::{Component, Entity, World};
use std::collections::HashMap;
use std::fmt::Write;
use std::sync::Arc;
use threers::core::{Mesh, Object3D, ObjectKind};
use threers::geometries::MeshBuilder;
use threers::materials::Material;
use threers::math::Matrix4;

/// A mesh that will not move, for [`World::bake_static`] to merge.
#[derive(Debug, Clone, Copy, Default)]
pub struct Static;
impl Component for Static {}

/// What [`World::bake_static`] made: the merged meshes it spawned.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Baked {
    /// How many meshes went into the merged ones.
    pub merged: usize,
    /// The merged meshes, one per material and shadow setting.
    pub meshes: Vec<Entity>,
}

/// The same material, as far as a draw is concerned. Two materials printing
/// the same are the same; one printing past a few kilobytes (it holds a texture's
/// pixels) is known only by its own `Arc`, so it is never merged with a copy.
fn material_key(material: &Arc<Material>) -> String {
    struct Capped(String);
    impl Write for Capped {
        fn write_str(&mut self, s: &str) -> std::fmt::Result {
            if self.0.len() + s.len() > 4096 {
                return Err(std::fmt::Error);
            }
            self.0.push_str(s);
            Ok(())
        }
    }
    let mut out = Capped(String::new());
    match write!(out, "{material:?}") {
        Ok(()) => out.0,
        Err(_) => format!("arc:{:p}", Arc::as_ptr(material)),
    }
}

/// A node's place in the world as a matrix, scale included.
fn world_matrix(w: &World, id: Entity) -> Matrix4 {
    let mut chain = Vec::new();
    let mut at = Some(id);
    while let Some(node) = at.and_then(|i| w.scene.get(i)) {
        chain.push(Matrix4::compose(node.position, node.quaternion, node.scale));
        at = node.parent;
    }
    chain
        .into_iter()
        .rev()
        .fold(Matrix4::identity(), |m, local| m.multiply(&local))
}

impl World {
    /// Merge every visible [`Static`] mesh into one mesh per material and shadow
    /// setting. Run it once the level is built; a static mesh added later stays
    /// on its own until it is run again.
    ///
    /// ```
    /// # use threers_ecs::prelude::*;
    /// let mut w = World::new();
    /// for i in 0..100 {
    ///     w.mesh(BoxGeometry::new(1.0, 1.0, 1.0), StandardMaterial::new(Color::from_hex(0x888888)))
    ///         .at(Vector3::new(i as f32 * 2.0, 0.0, 0.0))
    ///         .static_mesh();
    /// }
    /// let baked = w.bake_static();
    /// assert_eq!((baked.merged, baked.meshes.len()), (100, 1));
    /// ```
    pub fn bake_static(&mut self) -> Baked {
        // By material, shadow casting and receiving.
        type Group = (Arc<Material>, Vec<Entity>);
        let mut groups: HashMap<(String, bool, bool), Group> = HashMap::new();
        for (id, node, _) in self.iter::<(Entity, NodeRef, With<Static>)>() {
            let ObjectKind::Mesh(mesh) = &node.kind else {
                continue;
            };
            if !node.visible || !mesh.materials.is_empty() {
                continue;
            }
            let key = (
                material_key(&mesh.material),
                node.cast_shadow,
                node.receive_shadow,
            );
            groups
                .entry(key)
                .or_insert_with(|| (mesh.material.clone(), Vec::new()))
                .1
                .push(id);
        }
        let mut baked = Baked::default();
        for ((_, cast, receive), (material, ids)) in groups {
            if ids.len() < 2 {
                continue;
            }
            let mut builder = MeshBuilder::new();
            for &id in &ids {
                let frame = world_matrix(self, id);
                if let Some(ObjectKind::Mesh(mesh)) = self.scene.get(id).map(|n| &n.kind) {
                    builder.append(&mesh.geometry, &frame);
                }
            }
            for &id in &ids {
                if let Some(node) = self.scene.get_mut(id) {
                    node.visible = false;
                }
            }
            let mut mesh = Mesh::new(builder.build(), (*material).clone());
            // The same material, not a copy: one pipeline, one bind group.
            mesh.material = material;
            let mut node = Object3D::mesh(mesh);
            node.name = "baked".to_owned();
            node.cast_shadow = cast;
            node.receive_shadow = receive;
            baked.merged += ids.len();
            baked.meshes.push(self.spawn(node));
        }
        baked
    }
}
