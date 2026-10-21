//! Bindings split out of `wasm.rs` by area (see the `mod` list there): the
//! three.js scene-core state the shim used to fake in JS — geometry groups
//! with a material array, `drawRange`, morph targets (`morphAttributes`,
//! `morphTargetsRelative`, `morphTargetInfluences`, `morphTargetDictionary`),
//! `matrixAutoUpdate` / an arbitrary `matrix`, Euler orders, and the
//! raycaster's layers. Everything is addressed by object handle on the scene
//! and applies to `Mesh` and `SkinnedMesh` alike.
#![allow(unused_imports)]

use super::*;
use crate::core::{BufferAttribute, BufferGeometry, ObjectKind};

/// The geometry, material, material array and morph state of a mesh-like
/// object, mutably.
fn with_mesh<R>(
    scene: &mut WebScene,
    handle: &WebObjectHandle,
    f: impl FnOnce(
        &mut Arc<BufferGeometry>,
        &mut Arc<crate::Material>,
        &mut Vec<Arc<crate::Material>>,
        &mut Vec<f32>,
        &mut std::collections::HashMap<String, usize>,
    ) -> R,
) -> Option<R> {
    match &mut scene.inner.get_mut(handle.id)?.kind {
        ObjectKind::Mesh(m) => Some(f(
            &mut m.geometry,
            &mut m.material,
            &mut m.materials,
            &mut m.morph_influences,
            &mut m.morph_target_dictionary,
        )),
        ObjectKind::SkinnedMesh(m) => Some(f(
            &mut m.geometry,
            &mut m.material,
            &mut m.materials,
            &mut m.morph_influences,
            &mut m.morph_target_dictionary,
        )),
        _ => None,
    }
}

/// Edit a mesh's geometry copy-on-write (it may be shared).
fn edit_geometry(
    scene: &mut WebScene,
    handle: &WebObjectHandle,
    f: impl FnOnce(&mut BufferGeometry),
) {
    with_mesh(scene, handle, |g, _, _, _, _| f(Arc::make_mut(g)));
}

/// JS `Infinity` (or anything past `u32`) as an unbounded draw-range count.
fn js_count(count: f64) -> u32 {
    if !count.is_finite() || count >= u32::MAX as f64 {
        u32::MAX
    } else {
        count.max(0.0) as u32
    }
}

#[wasm_bindgen]
impl WebScene {
    /// three.js `geometry.groups` for a mesh: `[start, count, materialIndex,
    /// …]` replaces them (an empty array clears them).
    #[wasm_bindgen(js_name = setMeshGroups)]
    pub fn set_mesh_groups(&mut self, handle: &WebObjectHandle, flat: Vec<u32>) {
        edit_geometry(self, handle, |g| {
            g.clear_groups();
            for c in flat.chunks_exact(3) {
                g.add_group(c[0], c[1], c[2]);
            }
        });
    }

    /// three.js `geometry.setDrawRange(start, count)`; `count` may be
    /// `Infinity`.
    #[wasm_bindgen(js_name = setMeshDrawRange)]
    pub fn set_mesh_draw_range(&mut self, handle: &WebObjectHandle, start: f64, count: f64) {
        edit_geometry(self, handle, |g| {
            g.set_draw_range(start.max(0.0) as u32, js_count(count));
        });
    }

    /// three.js `mesh.material = [a, b, …]`: clears the array. Follow with
    /// `pushMeshMaterial` per entry. An empty array makes the mesh a
    /// one-material mesh again.
    #[wasm_bindgen(js_name = clearMeshMaterials)]
    pub fn clear_mesh_materials(&mut self, handle: &WebObjectHandle) {
        with_mesh(self, handle, |_, _, mats, _, _| mats.clear());
    }

    /// Append to the mesh's material array (see `clearMeshMaterials`). The
    /// first entry also becomes the mesh's single-material fallback.
    #[wasm_bindgen(js_name = pushMeshMaterial)]
    pub fn push_mesh_material(&mut self, handle: &WebObjectHandle, mat: &WebMaterial) {
        with_mesh(self, handle, |_, single, mats, _, _| {
            if mats.is_empty() {
                *single = mat.inner.clone();
            }
            mats.push(mat.inner.clone());
        });
    }

    /// Replace entry `index` of the mesh's material array (after a setter on
    /// that material).
    #[wasm_bindgen(js_name = setMeshMaterialAt)]
    pub fn set_mesh_material_at(
        &mut self,
        handle: &WebObjectHandle,
        index: usize,
        mat: &WebMaterial,
    ) {
        with_mesh(self, handle, |_, single, mats, _, _| {
            if let Some(slot) = mats.get_mut(index) {
                *slot = mat.inner.clone();
                if index == 0 {
                    *single = mat.inner.clone();
                }
            }
        });
    }

    /// three.js `geometry.morphAttributes[name][target] = attribute` for a
    /// mesh (`name` is `position`, `normal` or `color`). Targets past the end
    /// of the list grow it with empty attributes.
    #[wasm_bindgen(js_name = setMorphAttribute)]
    pub fn set_morph_attribute(
        &mut self,
        handle: &WebObjectHandle,
        name: &str,
        target: usize,
        array: Vec<f32>,
        item_size: usize,
    ) {
        edit_geometry(self, handle, |g| {
            let mut list = g.morph_attributes.remove(name).unwrap_or_default();
            if list.len() <= target {
                list.resize(
                    target + 1,
                    BufferAttribute::new(Vec::new(), item_size.max(1)),
                );
            }
            list[target] = BufferAttribute::new(array, item_size.max(1));
            g.set_morph_attribute(name.to_string(), list);
        });
    }

    /// Drop every morph attribute of a mesh's geometry.
    #[wasm_bindgen(js_name = clearMorphAttributes)]
    pub fn clear_morph_attributes(&mut self, handle: &WebObjectHandle) {
        edit_geometry(self, handle, |g| {
            g.morph_attributes.clear();
            g.morph_target_names.clear();
            g.bump_version();
        });
    }

    /// three.js `geometry.morphTargetsRelative`.
    #[wasm_bindgen(js_name = setMorphTargetsRelative)]
    pub fn set_morph_targets_relative(&mut self, handle: &WebObjectHandle, relative: bool) {
        edit_geometry(self, handle, |g| g.morph_targets_relative = relative);
    }

    /// The morph targets' names (three.js keeps them on each morph
    /// attribute's `name`), by target index.
    #[wasm_bindgen(js_name = setMorphTargetNames)]
    pub fn set_morph_target_names(&mut self, handle: &WebObjectHandle, names: Vec<String>) {
        edit_geometry(self, handle, |g| g.morph_target_names = names);
    }

    /// three.js `mesh.updateMorphTargets()`: zeroed influences sized to the
    /// geometry's targets, and the name dictionary.
    #[wasm_bindgen(js_name = updateMorphTargets)]
    pub fn update_morph_targets(&mut self, handle: &WebObjectHandle) {
        if let Some(obj) = self.inner.get_mut(handle.id) {
            match &mut obj.kind {
                ObjectKind::Mesh(m) => m.update_morph_targets(),
                ObjectKind::SkinnedMesh(m) => m.update_morph_targets(),
                _ => {}
            }
        }
    }

    /// three.js `mesh.morphTargetInfluences` (the whole array).
    #[wasm_bindgen(js_name = setMorphTargetInfluences)]
    pub fn set_morph_target_influences(&mut self, handle: &WebObjectHandle, influences: Vec<f32>) {
        with_mesh(self, handle, |_, _, _, w, _| *w = influences);
    }

    #[wasm_bindgen(js_name = getMorphTargetInfluences)]
    pub fn get_morph_target_influences(&mut self, handle: &WebObjectHandle) -> Vec<f32> {
        with_mesh(self, handle, |_, _, _, w, _| w.clone()).unwrap_or_default()
    }

    /// three.js `mesh.morphTargetDictionary` as a plain object `{name: index}`.
    #[wasm_bindgen(js_name = getMorphTargetDictionary)]
    pub fn get_morph_target_dictionary(&mut self, handle: &WebObjectHandle) -> JsValue {
        let o = js_sys::Object::new();
        if let Some(dict) = with_mesh(self, handle, |_, _, _, _, d| d.clone()) {
            for (k, v) in dict {
                let _ = js_sys::Reflect::set(&o, &k.into(), &(v as f64).into());
            }
        }
        o.into()
    }

    /// three.js `matrixAutoUpdate`.
    #[wasm_bindgen(js_name = setMatrixAutoUpdate)]
    pub fn set_matrix_auto_update(&mut self, handle: &WebObjectHandle, on: bool) {
        if let Some(o) = self.inner.get_mut(handle.id) {
            o.matrix_auto_update = on;
        }
    }

    /// three.js `matrixWorldAutoUpdate` (honoured on the scene root, as in a
    /// three.js render).
    #[wasm_bindgen(js_name = setMatrixWorldAutoUpdate)]
    pub fn set_matrix_world_auto_update(&mut self, handle: &WebObjectHandle, on: bool) {
        if let Some(o) = self.inner.get_mut(handle.id) {
            o.matrix_world_auto_update = on;
        }
    }

    /// `object.matrix.fromArray(e); object.matrixAutoUpdate = false` — any
    /// affine matrix, column-major, shear included.
    #[wasm_bindgen(js_name = setLocalMatrix)]
    pub fn set_local_matrix(&mut self, handle: &WebObjectHandle, elements: Vec<f32>) {
        if let (Some(o), Ok(e)) = (
            self.inner.get_mut(handle.id),
            <[f32; 16]>::try_from(elements),
        ) {
            o.set_matrix(crate::math::Matrix4 { elements: e });
        }
    }

    /// three.js `applyMatrix4(m)` (column-major).
    #[wasm_bindgen(js_name = applyMatrix4)]
    pub fn apply_matrix4(&mut self, handle: &WebObjectHandle, elements: Vec<f32>) {
        if let (Some(o), Ok(e)) = (
            self.inner.get_mut(handle.id),
            <[f32; 16]>::try_from(elements),
        ) {
            o.apply_matrix4(&crate::math::Matrix4 { elements: e });
        }
    }

    /// three.js `rotation.set(x, y, z, order)` — any of the six orders
    /// (`"XYZ"` when `order` is unknown).
    #[wasm_bindgen(js_name = setRotationEuler)]
    pub fn set_rotation_euler(
        &mut self,
        handle: &WebObjectHandle,
        x: f32,
        y: f32,
        z: f32,
        order: &str,
    ) {
        let order = crate::math::EulerOrder::parse(order).unwrap_or_default();
        if let Some(o) = self.inner.get_mut(handle.id) {
            o.set_rotation(crate::math::Euler::with_order(x, y, z, order));
        }
    }

    /// three.js `rotation` as `[x, y, z]` in the object's order.
    #[wasm_bindgen(js_name = getRotationEuler)]
    pub fn get_rotation_euler(&self, handle: &WebObjectHandle) -> Vec<f32> {
        self.inner
            .get(handle.id)
            .map(|o| {
                let e = o.rotation();
                vec![e.x, e.y, e.z]
            })
            .unwrap_or_default()
    }

    /// The object's local and world matrices, column-major, 32 floats —
    /// after a world update.
    #[wasm_bindgen(js_name = getMatrices)]
    pub fn get_matrices(&mut self, handle: &WebObjectHandle) -> Vec<f32> {
        self.inner.update_world();
        self.inner
            .get(handle.id)
            .map(|o| {
                o.matrix
                    .elements
                    .iter()
                    .chain(&o.matrix_world.elements)
                    .copied()
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[wasm_bindgen]
impl WebRaycaster {
    /// three.js `raycaster.layers.mask`. Defaults to channel 0 only, as in
    /// three.js.
    #[wasm_bindgen(js_name = setLayersMask)]
    pub fn set_layers_mask(&mut self, mask: u32) {
        self.inner.layers.mask = mask;
    }

    #[wasm_bindgen(js_name = layersMask)]
    pub fn layers_mask(&self) -> u32 {
        self.inner.layers.mask
    }
}
