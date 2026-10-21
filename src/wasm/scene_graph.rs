//! Bindings split out of `wasm.rs` by area (see the `mod` list there): the
//! scene-graph mirror. `web/threejs-shim.js` keeps one wasm node per three.js
//! `Object3D` — groups, meshes, lines, points, sprites, lights, bones,
//! cameras, helpers — with the same parent, the same children in the same
//! order, and the same local state (transform or matrix, visibility, layers,
//! geometry, materials, morph influences, instance data, skeleton and bind
//! matrices). Nodes are created detached and placed with `setChildren`, so the
//! shim can build, move and re-parent subtrees as three.js does; the
//! renderer, the raycaster and the controls then read one graph.
//!
//! Also here: the camera pose from a `matrixWorld`, the raycaster's three.js
//! entry points (`setFromCamera` from three.js's own matrices, per-object
//! casts, invisible objects), light parameters, and `DragControls`.
#![allow(unused_imports)]

use super::*;
use crate::core::{BufferGeometry, ObjectKind};
use crate::math::{Matrix4, Quaternion, Vector2, Vector3};

fn matrix(elements: &[f32]) -> Option<Matrix4> {
    <[f32; 16]>::try_from(elements)
        .ok()
        .map(|e| Matrix4 { elements: e })
}

fn ids(list: &[f64]) -> Vec<ObjectId> {
    list.iter()
        .filter_map(|&id| object_id_from_f64(id))
        .collect()
}

fn handle(id: ObjectId) -> WebObjectHandle {
    WebObjectHandle { id }
}

fn light_of(light: &WebLight) -> crate::lights::Light {
    match &light.inner {
        LightInner::Ambient(l) => (*l).into(),
        LightInner::Directional(l) => (*l).into(),
        LightInner::Point(l) => (*l).into(),
        LightInner::Spot(l) => (*l).into(),
        LightInner::Hemisphere(l) => (*l).into(),
        LightInner::RectArea(l) => (*l).into(),
    }
}

/// three.js `Line` modes for [`WebScene::create_line`]: `0` `LineSegments`,
/// `1` `Line` (strip), `2` `LineLoop`.
fn line_of(
    geometry: Arc<BufferGeometry>,
    material: Arc<crate::Material>,
    mode: u32,
) -> crate::core::LineSegments {
    match mode {
        1 => crate::core::LineSegments::strip(geometry, material, false),
        2 => crate::core::LineSegments::strip(geometry, material, true),
        _ => crate::core::LineSegments::from_arc(geometry, material),
    }
}

impl WebScene {
    fn insert_node(&mut self, obj: crate::core::Object3D) -> WebObjectHandle {
        handle(self.inner.arena.insert(obj))
    }

    /// A helper built as a Rust object, without links to nodes of its own.
    fn insert_helper(&mut self, mut obj: crate::core::Object3D) -> WebObjectHandle {
        obj.parent = None;
        obj.children.clear();
        self.insert_node(obj)
    }
}

#[wasm_bindgen]
impl WebScene {
    /// The scene root (three.js's `Scene` itself), for `setChildren`.
    #[wasm_bindgen(js_name = rootHandle)]
    pub fn root_handle(&self) -> WebObjectHandle {
        handle(self.inner.root)
    }

    /// A new detached node with no geometry: three.js `Object3D`, `Bone`,
    /// a camera's place in the graph — or, with `is_group`, a `Group`.
    #[wasm_bindgen(js_name = createGroup)]
    pub fn create_group(&mut self, is_group: bool) -> WebObjectHandle {
        let mut o = crate::core::Object3D::group();
        o.is_group = is_group;
        self.insert_node(o)
    }

    /// A new detached `Mesh`.
    #[wasm_bindgen(js_name = createMesh)]
    pub fn create_mesh(&mut self, geom: &WebGeometry, mat: &WebMaterial) -> WebObjectHandle {
        let mesh = crate::core::Mesh::from_arc(geom.inner.clone(), mat.inner.clone());
        self.insert_node(crate::core::Object3D::mesh(mesh))
    }

    /// A new detached `SkinnedMesh`, bound to no bones yet (`setSkeleton`).
    /// The geometry's `joint` / `weight` attributes are three.js's
    /// `skinIndex` / `skinWeight`.
    #[wasm_bindgen(js_name = createSkinnedMesh)]
    pub fn create_skinned_mesh(
        &mut self,
        geom: &WebGeometry,
        mat: &WebMaterial,
    ) -> WebObjectHandle {
        let sm = crate::core::SkinnedMesh::from_arc(
            geom.inner.clone(),
            mat.inner.clone(),
            Arc::new(crate::core::Skeleton::default()),
        );
        self.insert_node(crate::core::Object3D::skinned_mesh(sm))
    }

    /// A new detached `InstancedMesh` of `count` identity instances.
    #[wasm_bindgen(js_name = createInstancedMesh)]
    pub fn create_instanced_mesh(
        &mut self,
        geom: &WebGeometry,
        mat: &WebMaterial,
        count: usize,
    ) -> WebObjectHandle {
        let im = crate::core::InstancedMesh::from_arc(geom.inner.clone(), mat.inner.clone(), count);
        self.insert_node(crate::core::Object3D::instanced_mesh(im))
    }

    /// A new detached shell-fur object (`threers::fur`) grown on `geom`,
    /// animated by `stepFur`.
    #[wasm_bindgen(js_name = createFur)]
    pub fn create_fur(&mut self, geom: &WebGeometry, fur: &WebFurMaterial) -> WebObjectHandle {
        let h = self.insert_node(crate::fur::fur_object((*geom.inner).clone(), &fur.inner));
        self.fur_animator.track(h.id, &fur.inner);
        h
    }

    /// A new detached strand-hair object (`threers::fur::strands`) grown on
    /// `geom`, animated by `stepFur`.
    /// `shadowStrands` as for `addHair`.
    #[wasm_bindgen(js_name = createHair)]
    pub fn create_hair(
        &mut self,
        geom: &WebGeometry,
        fur: &WebFurMaterial,
        count: u32,
        segments: u32,
        width: f32,
        shadow_strands: Option<f32>,
    ) -> WebObjectHandle {
        let options = crate::fur::strands::StrandOptions {
            count: count as usize,
            segments: segments as usize,
            width,
            shadow_strands: shadow_strands
                .unwrap_or(crate::fur::strands::StrandOptions::default().shadow_strands),
        };
        let h = self.insert_node(crate::fur::strands::hair_object(
            &geom.inner,
            &fur.inner,
            &options,
        ));
        self.fur_animator.track(h.id, &fur.inner);
        h
    }

    /// A new detached strand-hair object grown on the skinned mesh
    /// `skinned` and carried through its animation by `stepFur` — each strand
    /// on its triangle, trailing its root on a spring. Link it as the skinned
    /// mesh's child, with no transform of its own. A group if `skinned` is
    /// not a skinned mesh.
    /// `shadowStrands` as for `addHair`.
    #[wasm_bindgen(js_name = createSkinnedHair)]
    pub fn create_skinned_hair(
        &mut self,
        skinned: &WebObjectHandle,
        fur: &WebFurMaterial,
        count: u32,
        segments: u32,
        width: f32,
        shadow_strands: Option<f32>,
    ) -> WebObjectHandle {
        let options = crate::fur::strands::StrandOptions {
            count: count as usize,
            segments: segments as usize,
            width,
            shadow_strands: shadow_strands
                .unwrap_or(crate::fur::strands::StrandOptions::default().shadow_strands),
        };
        let grown = self.fur_animator.hair_on_skinned(
            &mut self.inner,
            skinned.id,
            &fur.inner,
            &options,
            |scene, hair| scene.arena.insert(hair),
        );
        match grown {
            Some(id) => handle(id),
            None => self.create_group(false),
        }
    }

    /// A new detached line: `mode` 0 `LineSegments`, 1 `Line`, 2 `LineLoop`.
    #[wasm_bindgen(js_name = createLine)]
    pub fn create_line(
        &mut self,
        geom: &WebGeometry,
        mat: &WebMaterial,
        mode: u32,
    ) -> WebObjectHandle {
        let ls = line_of(geom.inner.clone(), mat.inner.clone(), mode);
        self.insert_node(crate::core::Object3D::line_segments(ls))
    }

    /// A new detached `Points`.
    #[wasm_bindgen(js_name = createPoints)]
    pub fn create_points(&mut self, geom: &WebGeometry, mat: &WebMaterial) -> WebObjectHandle {
        let p = crate::core::Points {
            geometry: geom.inner.clone(),
            material: mat.inner.clone(),
        };
        self.insert_node(crate::core::Object3D::points(p))
    }

    /// A new detached `Sprite`.
    #[wasm_bindgen(js_name = createSprite)]
    pub fn create_sprite(&mut self, mat: &WebMaterial) -> WebObjectHandle {
        let s = crate::core::Sprite {
            material: mat.inner.clone(),
            center: Vector2::new(0.5, 0.5),
        };
        self.insert_node(crate::core::Object3D::sprite(s))
    }

    /// A new detached light, a copy of `light` (`updateLight` re-copies).
    #[wasm_bindgen(js_name = createLight)]
    pub fn create_light(&mut self, light: &WebLight) -> WebObjectHandle {
        self.insert_node(crate::core::Object3D::light(light_of(light)))
    }

    /// A new detached `AxesHelper` node, a copy of the helper's lines.
    #[wasm_bindgen(js_name = createAxesHelper)]
    pub fn create_axes_helper(&mut self, helper: &WebAxesHelper) -> WebObjectHandle {
        self.insert_helper(helper.obj.clone())
    }

    /// A new detached `GridHelper` node.
    #[wasm_bindgen(js_name = createGridHelper)]
    pub fn create_grid_helper(&mut self, helper: &WebGridHelper) -> WebObjectHandle {
        self.insert_helper(helper.obj.clone())
    }

    /// A new detached `PolarGridHelper` node.
    #[wasm_bindgen(js_name = createPolarGridHelper)]
    pub fn create_polar_grid_helper(&mut self, helper: &WebPolarGridHelper) -> WebObjectHandle {
        self.insert_helper(helper.obj.clone())
    }

    /// three.js `parent.children = list`: exactly these nodes, in this
    /// order, under `parent` (see `ObjectArena::set_children`).
    #[wasm_bindgen(js_name = setChildren)]
    pub fn set_children(&mut self, parent: &WebObjectHandle, children: Vec<f64>) {
        self.inner.arena.set_children(parent.id, &ids(&children));
    }

    /// Detach a node from its parent (it stays in the scene's arena, undrawn,
    /// until re-parented or deleted).
    #[wasm_bindgen(js_name = detachNode)]
    pub fn detach_node(&mut self, node: &WebObjectHandle) {
        self.inner.arena.remove_child(node.id);
    }

    /// Drop a node for good; its children are left parentless.
    #[wasm_bindgen(js_name = deleteNode)]
    pub fn delete_node(&mut self, node: &WebObjectHandle) {
        if node.id != self.inner.root {
            self.inner.arena.delete(node.id);
            self.node_uuids.remove(&node.id);
        }
    }

    /// Whether `node` is (still) in this scene's arena.
    #[wasm_bindgen(js_name = hasNode)]
    pub fn has_node(&self, node: &WebObjectHandle) -> bool {
        self.inner.arena.get(node.id).is_some()
    }

    /// three.js `position`, `quaternion` and `scale`, with `matrixAutoUpdate`
    /// on: the local matrix is composed from them.
    #[wasm_bindgen(js_name = setTRS)]
    pub fn set_trs(
        &mut self,
        node: &WebObjectHandle,
        px: f32,
        py: f32,
        pz: f32,
        qx: f32,
        qy: f32,
        qz: f32,
        qw: f32,
        sx: f32,
        sy: f32,
        sz: f32,
    ) {
        if let Some(o) = self.inner.get_mut(node.id) {
            o.position = Vector3::new(px, py, pz);
            o.quaternion = Quaternion::new(qx, qy, qz, qw);
            o.scale = Vector3::new(sx, sy, sz);
            o.matrix_auto_update = true;
        }
    }

    /// `[px, py, pz, qx, qy, qz, qw, sx, sy, sz]` of a node — what a control
    /// that moved it wrote.
    #[wasm_bindgen(js_name = getTRS)]
    pub fn get_trs(&self, node: &WebObjectHandle) -> Vec<f32> {
        self.inner
            .get(node.id)
            .map(|o| {
                let (p, q, s) = (o.position, o.quaternion, o.scale);
                vec![p.x, p.y, p.z, q.x, q.y, q.z, q.w, s.x, s.y, s.z]
            })
            .unwrap_or_default()
    }

    /// The node's world matrix (column-major), after a world update.
    #[wasm_bindgen(js_name = getWorldMatrix)]
    pub fn get_world_matrix(&mut self, node: &WebObjectHandle) -> Vec<f32> {
        self.inner.update_world();
        self.inner
            .get(node.id)
            .map(|o| o.matrix_world.elements.to_vec())
            .unwrap_or_default()
    }

    /// three.js `object.layers.mask`.
    #[wasm_bindgen(js_name = setLayersMask)]
    pub fn set_layers_mask(&mut self, node: &WebObjectHandle, mask: u32) {
        if let Some(o) = self.inner.get_mut(node.id) {
            o.layers.mask = mask;
        }
    }

    /// The node's geometry, for every kind that has one. For a line, `mode`
    /// as in `createLine` (the strip is re-expanded).
    #[wasm_bindgen(js_name = setNodeGeometry)]
    pub fn set_node_geometry(
        &mut self,
        node: &WebObjectHandle,
        geom: &WebGeometry,
        mode: Option<u32>,
    ) {
        let Some(o) = self.inner.get_mut(node.id) else {
            return;
        };
        let g = geom.inner.clone();
        match &mut o.kind {
            ObjectKind::Mesh(m) => m.geometry = g,
            ObjectKind::SkinnedMesh(m) => m.geometry = g,
            ObjectKind::InstancedMesh(m) => m.geometry = g,
            ObjectKind::Points(p) => p.geometry = g,
            ObjectKind::LineSegments(ls) => {
                let mode = mode.unwrap_or(match &ls.strip {
                    Some(s) if s.closed => 2,
                    Some(_) => 1,
                    None => 0,
                });
                *ls = line_of(g, ls.material.clone(), mode);
            }
            ObjectKind::Group | ObjectKind::Light(_) | ObjectKind::Sprite(_) => {}
        }
    }

    /// The node's (single) material, for every kind that has one. A mesh's
    /// material array is set with `clearMeshMaterials` / `pushMeshMaterial`.
    #[wasm_bindgen(js_name = setNodeMaterial)]
    pub fn set_node_material(&mut self, node: &WebObjectHandle, mat: &WebMaterial) {
        let Some(o) = self.inner.get_mut(node.id) else {
            return;
        };
        let m = mat.inner.clone();
        match &mut o.kind {
            ObjectKind::Mesh(x) => x.material = m,
            ObjectKind::SkinnedMesh(x) => x.material = m,
            ObjectKind::InstancedMesh(x) => x.material = m,
            ObjectKind::Points(x) => x.material = m,
            ObjectKind::LineSegments(x) => x.material = m,
            ObjectKind::Sprite(x) => x.material = m,
            ObjectKind::Group | ObjectKind::Light(_) => {}
        }
    }

    /// A mesh's material array (three.js `mesh.material = [a, b, …]`) cleared
    /// — follow with `pushNodeMaterial` per entry. Mesh, SkinnedMesh and
    /// InstancedMesh.
    #[wasm_bindgen(js_name = clearNodeMaterials)]
    pub fn clear_node_materials(&mut self, node: &WebObjectHandle) {
        match self.inner.get_mut(node.id).map(|o| &mut o.kind) {
            Some(ObjectKind::Mesh(m)) => m.materials.clear(),
            Some(ObjectKind::SkinnedMesh(m)) => m.materials.clear(),
            Some(ObjectKind::InstancedMesh(m)) => m.materials.clear(),
            _ => {}
        }
    }

    /// Append to a mesh's material array (see `clearNodeMaterials`); the
    /// geometry's groups pick entries by `materialIndex`.
    #[wasm_bindgen(js_name = pushNodeMaterial)]
    pub fn push_node_material(&mut self, node: &WebObjectHandle, mat: &WebMaterial) {
        let m = mat.inner.clone();
        match self.inner.get_mut(node.id).map(|o| &mut o.kind) {
            Some(ObjectKind::Mesh(x)) => x.materials.push(m),
            Some(ObjectKind::SkinnedMesh(x)) => x.materials.push(m),
            Some(ObjectKind::InstancedMesh(x)) => x.materials.push(m),
            _ => {}
        }
    }

    /// three.js `instanceMatrix` (column-major, 16 per instance) after
    /// `needsUpdate`, cut to `count`: the instance count becomes
    /// `matrices.length / 16`.
    #[wasm_bindgen(js_name = setInstanceMatrices)]
    pub fn set_instance_matrices(&mut self, node: &WebObjectHandle, matrices: Vec<f32>) {
        if let Some(ObjectKind::InstancedMesh(im)) =
            self.inner.get_mut(node.id).map(|o| &mut o.kind)
        {
            im.transforms = matrices
                .chunks_exact(16)
                .map(|c| matrix(c).unwrap_or_else(Matrix4::identity))
                .collect();
        }
    }

    /// three.js `instanceColor` (rgb per instance, linear); empty removes it.
    #[wasm_bindgen(js_name = setInstanceColors)]
    pub fn set_instance_colors(&mut self, node: &WebObjectHandle, rgb: Vec<f32>) {
        if let Some(ObjectKind::InstancedMesh(im)) =
            self.inner.get_mut(node.id).map(|o| &mut o.kind)
        {
            im.colors = rgb
                .chunks_exact(3)
                .map(|c| crate::math::Color::new(c[0], c[1], c[2]))
                .collect();
        }
    }

    /// Per-instance custom column `slot` (0 or 1) for a `ShaderMaterial`:
    /// `components` (1–4) floats an instance, read in WGSL as
    /// `in.icustom0` / `in.icustom1` (see `InstancedMesh::set_custom`).
    /// Anything else is ignored.
    #[wasm_bindgen(js_name = setInstanceCustom)]
    pub fn set_instance_custom(
        &mut self,
        node: &WebObjectHandle,
        slot: u32,
        components: u32,
        values: Vec<f32>,
    ) {
        if slot > 1
            || !(1..=4).contains(&components)
            || !values.len().is_multiple_of(components as usize)
        {
            return;
        }
        if let Some(ObjectKind::InstancedMesh(im)) =
            self.inner.get_mut(node.id).map(|o| &mut o.kind)
        {
            im.set_custom(slot as usize, components, &values);
        }
    }

    /// three.js `skeleton`: the bones (node ids, in skeleton order) and their
    /// `boneInverses` (16 per bone). The palette is computed from the bones'
    /// world matrices at every world update.
    #[wasm_bindgen(js_name = setSkeleton)]
    pub fn set_skeleton(&mut self, node: &WebObjectHandle, bones: Vec<f64>, inverses: Vec<f32>) {
        let bones: Vec<crate::core::Bone> = bones
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| {
                Some(crate::core::Bone {
                    node: object_id_from_f64(b)?,
                    inverse_bind: inverses
                        .get(i * 16..i * 16 + 16)
                        .and_then(matrix)
                        .unwrap_or_else(Matrix4::identity),
                })
            })
            .collect();
        if let Some(ObjectKind::SkinnedMesh(sm)) = self.inner.get_mut(node.id).map(|o| &mut o.kind)
        {
            sm.skeleton = Arc::new(crate::core::Skeleton::new(bones));
        }
    }

    /// three.js `bindMatrix` and `bindMode` (`detached` false is
    /// `AttachedBindMode`).
    #[wasm_bindgen(js_name = setBindMatrix)]
    pub fn set_bind_matrix(&mut self, node: &WebObjectHandle, elements: Vec<f32>, detached: bool) {
        if let Some(ObjectKind::SkinnedMesh(sm)) = self.inner.get_mut(node.id).map(|o| &mut o.kind)
        {
            sm.bind_matrix = matrix(&elements).unwrap_or_else(Matrix4::identity);
            sm.bind_mode = if detached {
                crate::core::BindMode::Detached
            } else {
                crate::core::BindMode::Attached
            };
        }
    }

    /// The skinning palette the renderers draw with, 16 floats per bone,
    /// after a world update.
    #[wasm_bindgen(js_name = getBoneMatrices)]
    pub fn get_bone_matrices(&mut self, node: &WebObjectHandle) -> Vec<f32> {
        self.inner.update_world();
        match self.inner.get(node.id).map(|o| &o.kind) {
            Some(ObjectKind::SkinnedMesh(sm)) => sm
                .skeleton
                .bone_matrices
                .iter()
                .flat_map(|m| m.elements)
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Diagnostics: a node as the scene graph has it — `kind`, `parent` (id,
    /// or undefined), `children` (ids, in order), `visible`, `layers`,
    /// `renderOrder`, `castShadow`, `receiveShadow`, `matrixAutoUpdate`, and
    /// per kind `materials` (array length), `instances`, `instanceColors`,
    /// `bones`, `morphInfluences`, `drawRange` (`[start, count]`), `groups`,
    /// `lineMode`. For tests that compare the mirror with the JS graph.
    #[wasm_bindgen(js_name = nodeInfo)]
    pub fn node_info(&self, node: &WebObjectHandle) -> JsValue {
        let Some(o) = self.inner.get(node.id) else {
            return JsValue::UNDEFINED;
        };
        let out = js_sys::Object::new();
        let set = |k: &str, v: JsValue| js_set(&out, k, &v);
        let kind = match &o.kind {
            ObjectKind::Group => "group",
            ObjectKind::Mesh(_) => "mesh",
            ObjectKind::Light(_) => "light",
            ObjectKind::LineSegments(_) => "line",
            ObjectKind::Points(_) => "points",
            ObjectKind::Sprite(_) => "sprite",
            ObjectKind::InstancedMesh(_) => "instanced",
            ObjectKind::SkinnedMesh(_) => "skinned",
        };
        set("kind", JsValue::from_str(kind));
        if let Some(p) = o.parent {
            set("parent", object_id_to_f64(p).into());
        }
        let children = js_sys::Array::new();
        for c in &o.children {
            children.push(&object_id_to_f64(*c).into());
        }
        set("children", children.into());
        set("visible", o.visible.into());
        set("layers", o.layers.mask.into());
        set("renderOrder", o.render_order.into());
        set("castShadow", o.cast_shadow.into());
        set("receiveShadow", o.receive_shadow.into());
        set("matrixAutoUpdate", o.matrix_auto_update.into());
        set("isGroup", o.is_group.into());
        let geometry_info = |g: &BufferGeometry| {
            let dr = js_sys::Array::new();
            dr.push(&(g.draw_range.start as f64).into());
            let count = if g.draw_range.count == u32::MAX {
                f64::INFINITY
            } else {
                g.draw_range.count as f64
            };
            dr.push(&count.into());
            set("drawRange", dr.into());
            set("groups", (g.groups.len() as f64).into());
            set("morphTargets", (g.morph_target_count() as f64).into());
        };
        let influences = |w: &[f32]| {
            let a = js_sys::Array::new();
            for v in w {
                a.push(&(*v as f64).into());
            }
            a.into()
        };
        match &o.kind {
            ObjectKind::Mesh(m) => {
                geometry_info(&m.geometry);
                set("materials", (m.materials.len() as f64).into());
                set("morphInfluences", influences(&m.morph_influences));
            }
            ObjectKind::SkinnedMesh(m) => {
                geometry_info(&m.geometry);
                set("materials", (m.materials.len() as f64).into());
                set("morphInfluences", influences(&m.morph_influences));
                set("bones", (m.skeleton.bones.len() as f64).into());
            }
            ObjectKind::InstancedMesh(m) => {
                geometry_info(&m.geometry);
                set("materials", (m.materials.len() as f64).into());
                set("instances", (m.transforms.len() as f64).into());
                set("instanceColors", (m.colors.len() as f64).into());
            }
            ObjectKind::LineSegments(l) => {
                geometry_info(
                    l.strip
                        .as_ref()
                        .map(|s| &*s.geometry)
                        .unwrap_or(&l.geometry),
                );
                set(
                    "lineMode",
                    match &l.strip {
                        None => 0.0,
                        Some(s) if s.closed => 2.0,
                        Some(_) => 1.0,
                    }
                    .into(),
                );
            }
            ObjectKind::Points(p) => geometry_info(&p.geometry),
            _ => {}
        }
        out.into()
    }

    /// Diagnostics: an InstancedMesh node's instance matrices, 16 per
    /// instance, as the renderer draws them.
    #[wasm_bindgen(js_name = getInstanceMatrices)]
    pub fn get_instance_matrices(&self, node: &WebObjectHandle) -> Vec<f32> {
        match self.inner.get(node.id).map(|o| &o.kind) {
            Some(ObjectKind::InstancedMesh(im)) => {
                im.transforms.iter().flat_map(|m| m.elements).collect()
            }
            _ => Vec::new(),
        }
    }

    /// three.js `sprite.center`.
    #[wasm_bindgen(js_name = setSpriteCenter)]
    pub fn set_sprite_center(&mut self, node: &WebObjectHandle, x: f32, y: f32) {
        if let Some(ObjectKind::Sprite(s)) = self.inner.get_mut(node.id).map(|o| &mut o.kind) {
            s.center = Vector2::new(x, y);
        }
    }

    /// The light a node carries, replaced by a copy of `light` (after a
    /// change to its colour, intensity or shape).
    #[wasm_bindgen(js_name = setNodeLight)]
    pub fn set_node_light(&mut self, node: &WebObjectHandle, light: &WebLight) {
        if let Some(ObjectKind::Light(slot)) = self.inner.get_mut(node.id).map(|o| &mut o.kind) {
            *slot = light_of(light);
        }
    }
}

#[wasm_bindgen]
impl WebLight {
    /// three.js `light.color` (the sky colour of a hemisphere light), linear.
    #[wasm_bindgen(js_name = setColor)]
    pub fn set_color(&mut self, color: &WebColor) {
        let c = color.inner;
        match &mut self.inner {
            LightInner::Ambient(l) => l.color = c,
            LightInner::Directional(l) => l.color = c,
            LightInner::Point(l) => l.color = c,
            LightInner::Spot(l) => l.color = c,
            LightInner::Hemisphere(l) => l.sky_color = c,
            LightInner::RectArea(l) => l.color = c,
        }
    }

    /// three.js `HemisphereLight.groundColor`.
    #[wasm_bindgen(js_name = setGroundColor)]
    pub fn set_ground_color(&mut self, color: &WebColor) {
        if let LightInner::Hemisphere(l) = &mut self.inner {
            l.ground_color = color.inner;
        }
    }

    /// three.js `distance` and `decay` (point and spot lights).
    #[wasm_bindgen(js_name = setRange)]
    pub fn set_range(&mut self, distance: f32, decay: f32) {
        match &mut self.inner {
            LightInner::Point(l) => {
                l.distance = distance;
                l.decay = decay;
            }
            LightInner::Spot(l) => {
                l.distance = distance;
                l.decay = decay;
            }
            _ => {}
        }
    }

    /// three.js `SpotLight.angle` and `penumbra`.
    #[wasm_bindgen(js_name = setCone)]
    pub fn set_cone(&mut self, angle: f32, penumbra: f32) {
        if let LightInner::Spot(l) = &mut self.inner {
            l.angle = angle;
            l.penumbra = penumbra;
        }
    }

    /// three.js `RectAreaLight.width` and `height`.
    #[wasm_bindgen(js_name = setRectSize)]
    pub fn set_rect_size(&mut self, width: f32, height: f32) {
        if let LightInner::RectArea(l) = &mut self.inner {
            l.width = width;
            l.height = height;
        }
    }
}

#[wasm_bindgen]
impl WebMaterial {
    /// three.js `SpriteMaterial.rotation` (radians).
    #[wasm_bindgen(js_name = setSpriteRotation)]
    pub fn set_sprite_rotation(&mut self, rotation: f32) {
        if let crate::Material::Sprite(m) = Arc::make_mut(&mut self.inner) {
            m.rotation = rotation;
        }
    }

    /// three.js `sizeAttenuation` on `SpriteMaterial` and `PointsMaterial`.
    #[wasm_bindgen(js_name = setSizeAttenuation)]
    pub fn set_size_attenuation(&mut self, on: bool) {
        match Arc::make_mut(&mut self.inner) {
            crate::Material::Sprite(m) => m.size_attenuation = on,
            crate::Material::Points(m) => m.size_attenuation = on,
            _ => {}
        }
    }
}

#[wasm_bindgen]
impl WebGeometry {
    /// three.js `BoxGeometry(width, height, depth, widthSegments,
    /// heightSegments, depthSegments)`.
    #[wasm_bindgen(js_name = boxSegments)]
    pub fn box_segments(
        width: f32,
        height: f32,
        depth: f32,
        ws: u32,
        hs: u32,
        ds: u32,
    ) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::geometries::BoxGeometry::with_segments(
                width, height, depth, ws, hs, ds,
            )),
        }
    }

    /// A handle on the same geometry as `g` (shared, not copied), for a node;
    /// `g` stays usable — unlike `fromBufferGeometry`, which takes it.
    #[wasm_bindgen(js_name = shareBufferGeometry)]
    pub fn share_buffer_geometry(g: &WebBufferGeometry) -> WebGeometry {
        WebGeometry {
            inner: g.inner.clone(),
        }
    }

    /// three.js `geometry.setDrawRange(start, count)`; `count` may be
    /// `Infinity`.
    #[wasm_bindgen(js_name = setDrawRange)]
    pub fn set_draw_range(&mut self, start: f64, count: f64) {
        let count = if !count.is_finite() || count >= u32::MAX as f64 {
            u32::MAX
        } else {
            count.max(0.0) as u32
        };
        let start = start.max(0.0) as u32;
        let g = Arc::make_mut(&mut self.inner);
        g.set_draw_range(start, count);
    }

    /// three.js `geometry.groups`, `[start, count, materialIndex, …]`.
    #[wasm_bindgen(js_name = setGeometryGroups)]
    pub fn set_geometry_groups(&mut self, flat: Vec<u32>) {
        let g = Arc::make_mut(&mut self.inner);
        g.clear_groups();
        for c in flat.chunks_exact(3) {
            g.add_group(c[0], c[1], c[2]);
        }
    }

    /// three.js `geometry.morphAttributes[name]` (`position`, `normal`,
    /// `color`): `arrays` holds the targets back to back, each `item_size`
    /// per vertex; an empty array removes the name.
    #[wasm_bindgen(js_name = setMorphAttributeList)]
    pub fn set_morph_attribute_list(
        &mut self,
        name: &str,
        arrays: Vec<f32>,
        targets: usize,
        item_size: usize,
    ) {
        let g = Arc::make_mut(&mut self.inner);
        if targets == 0 || arrays.is_empty() {
            g.morph_attributes.remove(name);
            g.bump_version();
            return;
        }
        let per = arrays.len() / targets;
        let list = arrays
            .chunks_exact(per.max(1))
            .take(targets)
            .map(|c| crate::core::BufferAttribute::new(c.to_vec(), item_size.max(1)))
            .collect();
        g.set_morph_attribute(name.to_string(), list);
    }

    /// three.js `geometry.morphTargetsRelative`.
    #[wasm_bindgen(js_name = setMorphTargetsRelative)]
    pub fn set_morph_targets_relative(&mut self, relative: bool) {
        Arc::make_mut(&mut self.inner).morph_targets_relative = relative;
    }

    /// three-mesh-bvh's `geometry.boundsTree`: the raycaster walks it for
    /// meshes drawn with this geometry (one BVH built from the same
    /// triangles, by `WebMeshBvh`).
    #[cfg(feature = "mesh-bvh")]
    #[wasm_bindgen(js_name = setBoundsTree)]
    pub fn set_bounds_tree(&mut self, bvh: &WebMeshBvh) {
        Arc::make_mut(&mut self.inner).bounds_tree = Some(bvh.inner.clone());
    }

    /// `geometry.boundsTree = null` / `disposeBoundsTree()`.
    #[cfg(feature = "mesh-bvh")]
    #[wasm_bindgen(js_name = clearBoundsTree)]
    pub fn clear_bounds_tree(&mut self) {
        Arc::make_mut(&mut self.inner).bounds_tree = None;
    }
}

#[wasm_bindgen]
impl WebCamera {
    /// Place the camera by three.js's `matrixWorld` (column-major): its
    /// position, the direction it looks (-Z) and its up (+Y). The target is
    /// kept at the distance it was, so orbit-style readers of it keep their
    /// pivot distance.
    #[wasm_bindgen(js_name = setMatrixWorld)]
    pub fn set_matrix_world(&mut self, elements: Vec<f32>) {
        let Some(m) = matrix(&elements) else {
            return;
        };
        let e = &m.elements;
        let position = Vector3::new(e[12], e[13], e[14]);
        let forward = Vector3::new(-e[8], -e[9], -e[10]).normalize();
        let up = Vector3::new(e[4], e[5], e[6]).normalize();
        let place = |p: &mut Vector3, t: &mut Vector3, u: &mut Vector3| {
            let d = (*t - *p).length();
            let d = if d.is_finite() && d > 1e-6 { d } else { 1.0 };
            *p = position;
            *t = position + forward * d;
            *u = up;
        };
        // A world matrix that is not rotation + translation (a scaled or
        // sheared parent): three.js views through its inverse as is.
        let x = Vector3::new(e[0], e[1], e[2]);
        let y = Vector3::new(e[4], e[5], e[6]);
        let z = Vector3::new(e[8], e[9], e[10]);
        const EPS: f32 = 1e-5;
        let rigid = (x.length() - 1.0).abs() < EPS
            && (y.length() - 1.0).abs() < EPS
            && (z.length() - 1.0).abs() < EPS
            && x.dot(y).abs() < EPS
            && y.dot(z).abs() < EPS
            && z.dot(x).abs() < EPS;
        let view = |position, target, up| {
            (!rigid).then(|| crate::cameras::ViewOverride {
                view: m.invert(),
                position,
                target,
                up,
            })
        };
        match &mut self.inner {
            CameraInner::Perspective(c) => {
                place(&mut c.position, &mut c.target, &mut c.up);
                c.view_override = view(c.position, c.target, c.up);
            }
            CameraInner::Orthographic(c) => {
                place(&mut c.position, &mut c.target, &mut c.up);
                c.view_override = view(c.position, c.target, c.up);
            }
        }
    }

    /// The camera's up vector (controls that roll the camera move it).
    #[wasm_bindgen(js_name = readUp)]
    pub fn read_up(&self) -> WebVector3 {
        let up = match &self.inner {
            CameraInner::Perspective(c) => c.up,
            CameraInner::Orthographic(c) => c.up,
        };
        WebVector3 { inner: up }
    }

    /// three.js `camera.layers.mask`: the layers drawn.
    #[wasm_bindgen(js_name = setLayersMask)]
    pub fn set_layers_mask(&mut self, mask: u32) {
        match &mut self.inner {
            CameraInner::Perspective(c) => c.layers.mask = mask,
            CameraInner::Orthographic(c) => c.layers.mask = mask,
        }
    }

    /// three.js `OrthographicCamera` `left`, `right`, `top`, `bottom`,
    /// `near`, `far` (after `updateProjectionMatrix`; `zoom` divides the
    /// frustum, as there). Perspective: `near`, `far` only.
    #[wasm_bindgen(js_name = setFrustum)]
    pub fn set_frustum(
        &mut self,
        left: f32,
        right: f32,
        top: f32,
        bottom: f32,
        near: f32,
        far: f32,
        zoom: f32,
    ) {
        match &mut self.inner {
            CameraInner::Orthographic(c) => {
                let z = if zoom > 0.0 { zoom } else { 1.0 };
                let (dx, dy) = ((right - left) / (2.0 * z), (top - bottom) / (2.0 * z));
                let (cx, cy) = ((right + left) / 2.0, (top + bottom) / 2.0);
                c.left = cx - dx;
                c.right = cx + dx;
                c.top = cy + dy;
                c.bottom = cy - dy;
                c.near = near;
                c.far = far;
            }
            CameraInner::Perspective(c) => {
                c.near = near;
                c.far = far;
            }
        }
    }
}

#[wasm_bindgen]
impl WebRaycaster {
    /// three.js r165 `setFromCamera(coords, camera)` from three.js's own
    /// camera matrices: `matrixWorld` and `projectionMatrixInverse` (both
    /// column-major, OpenGL depth). `ortho` with the camera's `near` / `far`
    /// for an `OrthographicCamera`. Also remembers the camera for sprites.
    #[wasm_bindgen(js_name = setFromCameraMatrices)]
    pub fn set_from_camera_matrices(
        &mut self,
        x: f32,
        y: f32,
        matrix_world: Vec<f32>,
        projection_inverse: Vec<f32>,
        ortho: bool,
        near: f32,
        far: f32,
    ) {
        let (Some(w), Some(p)) = (matrix(&matrix_world), matrix(&projection_inverse)) else {
            return;
        };
        let ortho = ortho.then_some((near, far));
        self.inner
            .set_from_matrices(Vector2::new(x, y), &w, &p, ortho);
    }

    /// three.js `raycaster.camera` set by hand (sprites face it): its
    /// `matrixWorld`, and whether it is a perspective camera. `undefined`
    /// clears it.
    #[wasm_bindgen(js_name = setCameraMatrices)]
    pub fn set_camera_matrices(&mut self, matrix_world: Option<Vec<f32>>, perspective: bool) {
        self.inner.camera =
            matrix_world
                .as_deref()
                .and_then(matrix)
                .map(|w| crate::core::RaycastCamera {
                    matrix_world: w,
                    matrix_world_inverse: w.invert(),
                    perspective,
                });
    }

    /// Test invisible objects too — three.js's rule (`Raycaster.intersect`
    /// tests only the layers). Off by default.
    #[wasm_bindgen(getter, js_name = includeInvisible)]
    pub fn include_invisible(&self) -> bool {
        self.inner.include_invisible
    }
    #[wasm_bindgen(setter, js_name = includeInvisible)]
    pub fn set_include_invisible(&mut self, on: bool) {
        self.inner.include_invisible = on;
    }

    /// Each object in `object_ids` on its own (no descent, no layer test —
    /// three.js's `intersect` walked the graph and called each one's
    /// `raycast`), `first_only[i]` non-zero keeping only that object's
    /// nearest hit (three-mesh-bvh's `firstHitOnly` for a mesh with a
    /// `boundsTree`). Hits as `intersectObject` reports them, object by
    /// object, each object's nearest first.
    ///
    /// `world_matrices`, 16 numbers per object, are the objects' three.js
    /// `matrixWorld` as they stand: three.js's raycaster uses those and does
    /// not update them, so an object moved since its last
    /// `updateMatrixWorld` (or render) is hit where it was. Empty casts with
    /// the graph's own world matrices.
    #[wasm_bindgen(js_name = intersectEach)]
    pub fn intersect_each(
        &self,
        scene: &mut WebScene,
        object_ids: Vec<f64>,
        first_only: Vec<u8>,
        world_matrices: Vec<f32>,
    ) -> js_sys::Array {
        scene.inner.update_world();
        if world_matrices.len() == object_ids.len() * 16 {
            for (i, &id) in object_ids.iter().enumerate() {
                if let Some(obj) =
                    object_id_from_f64(id).and_then(|id| scene.inner.arena.get_mut(id))
                {
                    obj.matrix_world = crate::math::Matrix4 {
                        elements: std::array::from_fn(|k| world_matrices[i * 16 + k]),
                    };
                }
            }
        }
        let items: Vec<(ObjectId, bool)> = object_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| {
                Some((
                    object_id_from_f64(id)?,
                    first_only.get(i).is_some_and(|f| *f != 0),
                ))
            })
            .collect();
        let hits = self.inner.intersect_each(&scene.inner.arena, &items);
        let out = js_sys::Array::new();
        for h in &hits {
            out.push(&intersection_to_js(h));
        }
        out
    }

    /// three.js `raycaster.ray.set(origin, direction)` from numbers (the
    /// direction is normalised).
    #[wasm_bindgen(js_name = setRay)]
    pub fn set_ray(&mut self, ox: f32, oy: f32, oz: f32, dx: f32, dy: f32, dz: f32) {
        self.inner.set(
            Vector3::new(ox, oy, oz),
            Vector3::new(dx, dy, dz).normalize(),
        );
    }

    /// The ray as `[ox, oy, oz, dx, dy, dz]`.
    #[wasm_bindgen(js_name = rayArray)]
    pub fn ray_array(&self) -> Vec<f32> {
        let r = &self.inner.ray;
        vec![
            r.origin.x,
            r.origin.y,
            r.origin.z,
            r.direction.x,
            r.direction.y,
            r.direction.z,
        ]
    }
}

/// three.js r165 `DragControls` (`src/controls/drag.rs`) for the shim: the
/// shim casts with its `Raycaster` (so objects' own `raycast` methods take
/// part, as in three.js) and hands the nearest hit in; the control moves
/// the scene node and reports three.js's events and cursor.
#[wasm_bindgen]
pub struct WebDragControls {
    inner: crate::controls::DragControls,
}

/// A drag outcome for JS: `{ events: [{ type, objectId }], cursor }`.
fn outcome_to_js(o: &crate::controls::DragOutcome) -> JsValue {
    let out = js_sys::Object::new();
    let events = js_sys::Array::new();
    for e in &o.events {
        let ev = js_sys::Object::new();
        js_set(&ev, "type", &JsValue::from_str(e.kind()));
        js_set(&ev, "objectId", &object_id_to_f64(e.object()).into());
        events.push(&ev.into());
    }
    js_set(&out, "events", &events.into());
    js_set(
        &out,
        "cursor",
        &o.cursor
            .map(JsValue::from_str)
            .unwrap_or(JsValue::UNDEFINED),
    );
    out.into()
}

fn drag_camera(matrix_world: &[f32], quaternion: &[f32]) -> crate::controls::DragCamera {
    let q = if quaternion.len() == 4 {
        Quaternion::new(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
    } else {
        Quaternion::identity()
    };
    crate::controls::DragCamera {
        matrix_world: matrix(matrix_world).unwrap_or_else(Matrix4::identity),
        quaternion: q,
    }
}

#[wasm_bindgen]
impl WebDragControls {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebDragControls {
        WebDragControls {
            inner: crate::controls::DragControls::new(Vec::new()),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn enabled(&self) -> bool {
        self.inner.enabled
    }
    #[wasm_bindgen(setter)]
    pub fn set_enabled(&mut self, on: bool) {
        self.inner.enabled = on;
    }

    #[wasm_bindgen(getter, js_name = transformGroup)]
    pub fn transform_group(&self) -> bool {
        self.inner.transform_group
    }
    #[wasm_bindgen(setter, js_name = transformGroup)]
    pub fn set_transform_group(&mut self, on: bool) {
        self.inner.transform_group = on;
    }

    /// `"translate"` or `"rotate"`.
    #[wasm_bindgen(getter)]
    pub fn mode(&self) -> String {
        match self.inner.mode {
            crate::controls::DragMode::Translate => "translate".into(),
            crate::controls::DragMode::Rotate => "rotate".into(),
        }
    }
    #[wasm_bindgen(setter)]
    pub fn set_mode(&mut self, mode: &str) {
        self.inner.mode = if mode == "rotate" {
            crate::controls::DragMode::Rotate
        } else {
            crate::controls::DragMode::Translate
        };
    }

    #[wasm_bindgen(getter, js_name = rotateSpeed)]
    pub fn rotate_speed(&self) -> f32 {
        self.inner.rotate_speed
    }
    #[wasm_bindgen(setter, js_name = rotateSpeed)]
    pub fn set_rotate_speed(&mut self, v: f32) {
        self.inner.rotate_speed = v;
    }

    /// The node being dragged (an id in the scene last passed), or undefined.
    #[wasm_bindgen(getter)]
    pub fn selected(&self) -> Option<f64> {
        self.inner.selected().map(object_id_to_f64)
    }

    /// `pointerdown` at NDC `(x, y)`: `raycaster` set from the camera there,
    /// `hit` the id of the nearest object its cast hit (undefined: none),
    /// the camera's `matrixWorld` and `quaternion`.
    #[wasm_bindgen(js_name = pointerDown)]
    pub fn pointer_down(
        &mut self,
        scene: &mut WebScene,
        x: f32,
        y: f32,
        raycaster: &WebRaycaster,
        hit: Option<f64>,
        camera_world: Vec<f32>,
        camera_quaternion: Vec<f32>,
    ) -> JsValue {
        scene.inner.update_world();
        let cam = drag_camera(&camera_world, &camera_quaternion);
        let hit = hit.and_then(object_id_from_f64);
        let o = self.inner.pointer_down(
            &mut scene.inner.arena,
            Vector2::new(x, y),
            &raycaster.inner.ray,
            hit,
            &cam,
        );
        outcome_to_js(&o)
    }

    /// `pointermove` at NDC `(x, y)`. With nothing selected and `hover`
    /// (a mouse or pen pointer), `hit` is the hover cast's nearest object.
    #[wasm_bindgen(js_name = pointerMove)]
    pub fn pointer_move(
        &mut self,
        scene: &mut WebScene,
        x: f32,
        y: f32,
        raycaster: &WebRaycaster,
        hover: bool,
        hit: Option<f64>,
        camera_world: Vec<f32>,
        camera_quaternion: Vec<f32>,
    ) -> JsValue {
        scene.inner.update_world();
        let cam = drag_camera(&camera_world, &camera_quaternion);
        let hover_hit = hover.then(|| hit.and_then(object_id_from_f64));
        let o = self.inner.pointer_move(
            &mut scene.inner.arena,
            Vector2::new(x, y),
            &raycaster.inner.ray,
            hover_hit,
            &cam,
        );
        outcome_to_js(&o)
    }

    /// `pointerup` / `pointerleave`.
    #[wasm_bindgen(js_name = pointerCancel)]
    pub fn pointer_cancel(&mut self) -> JsValue {
        outcome_to_js(&self.inner.pointer_cancel())
    }
}
