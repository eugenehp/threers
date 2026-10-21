//! Bindings split out of `wasm.rs` by area (see the `mod` list there).
//!
//! The exporters: three.js's `OBJExporter`, `STLExporter`, `PLYExporter` and
//! `GLTFExporter` run in Rust ([`crate::exporters`],
//! [`crate::loaders::gltf_write`]). The shim walks the JS scene graph — that
//! graph is JS objects — and feeds each object to a [`WebSceneExporter`] in
//! `traverse` order; the formats are written here.
#![allow(unused_imports)]

use super::*;

use crate::exporters::{ExportKind, ExportObject, NumberFormat, ObjOptions, PlyOptions};
use crate::loaders::{GltfExportOptions, GltfMaterialInfo, GltfMaterialKind};

struct Entry {
    parent: Option<usize>,
    name: String,
    kind: Option<ExportKind>,
    position: [f64; 3],
    quaternion: [f64; 4],
    scale: [f64; 3],
    matrix_world: [f64; 16],
    geometry: Option<Arc<crate::BufferGeometry>>,
    material: Option<usize>,
}

/// A scene graph handed over one object at a time, to be written as OBJ,
/// STL, PLY or glTF.
#[wasm_bindgen]
pub struct WebSceneExporter {
    entries: Vec<Entry>,
    materials: Vec<(Arc<crate::Material>, GltfMaterialInfo)>,
}

fn fixed<const N: usize>(v: &[f64], what: &str) -> Result<[f64; N], JsValue> {
    v.try_into()
        .map_err(|_| JsValue::from_str(&format!("{what}: expected {N} numbers, got {}", v.len())))
}

#[wasm_bindgen]
impl WebSceneExporter {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebSceneExporter {
        WebSceneExporter {
            entries: Vec::new(),
            materials: Vec::new(),
        }
    }

    /// Add an object; returns its index, which children pass as `parent`
    /// (`-1` for none). `kind` is `"mesh"`, `"line"`, `"lineSegments"`,
    /// `"lineLoop"`, `"points"` or anything else for a plain node. The local
    /// transform is `position` (3), `quaternion` (4, xyzw) and `scale` (3); the
    /// world matrix is `matrixWorld.elements` (16, column-major).
    #[wasm_bindgen(js_name = addObject)]
    pub fn add_object(
        &mut self,
        parent: i32,
        name: String,
        kind: &str,
        position: &[f64],
        quaternion: &[f64],
        scale: &[f64],
        matrix_world: &[f64],
    ) -> Result<u32, JsValue> {
        let parent = usize::try_from(parent).ok();
        if let Some(p) = parent {
            if p >= self.entries.len() {
                return Err(JsValue::from_str(&format!("addObject: no parent {p}")));
            }
        }
        let kind = match kind {
            "mesh" => Some(ExportKind::Mesh),
            "line" => Some(ExportKind::Line),
            "lineSegments" => Some(ExportKind::LineSegments),
            "lineLoop" => Some(ExportKind::LineLoop),
            "points" => Some(ExportKind::Points),
            _ => None,
        };
        self.entries.push(Entry {
            parent,
            name,
            kind,
            position: fixed(position, "position")?,
            quaternion: fixed(quaternion, "quaternion")?,
            scale: fixed(scale, "scale")?,
            matrix_world: fixed(matrix_world, "matrixWorld")?,
            geometry: None,
            material: None,
        });
        Ok((self.entries.len() - 1) as u32)
    }

    /// The object's geometry, from a JS `BufferGeometry` (`_syncWasmFromJs()`).
    #[wasm_bindgen(js_name = setBufferGeometry)]
    pub fn set_buffer_geometry(
        &mut self,
        index: u32,
        geometry: &WebBufferGeometry,
    ) -> Result<(), JsValue> {
        self.entry(index)?.geometry = Some(geometry.inner.clone());
        Ok(())
    }

    /// The object's geometry, from a built-in generator's `WebGeometry`.
    #[wasm_bindgen(js_name = setGeometry)]
    pub fn set_geometry(&mut self, index: u32, geometry: &WebGeometry) -> Result<(), JsValue> {
        self.entry(index)?.geometry = Some(geometry.inner.clone());
        Ok(())
    }

    /// Register a material; returns its index for [`set_material`]. `kind` is
    /// `"standard"` (and physical), `"basic"`, or anything else. Colours are
    /// linear RGB, as three.js's `Color` holds them; `side` is 0/1/2.
    ///
    /// [`set_material`]: Self::set_material
    #[wasm_bindgen(js_name = addMaterial)]
    pub fn add_material(
        &mut self,
        kind: &str,
        name: String,
        color: &[f32],
        opacity: f32,
        transparent: bool,
        metalness: f32,
        roughness: f32,
        emissive: &[f32],
        side: u32,
        alpha_test: f32,
    ) -> u32 {
        let rgb = |v: &[f32], d: f32| [0, 1, 2].map(|k| v.get(k).copied().unwrap_or(d));
        let color = rgb(color, 1.0);
        let emissive = rgb(emissive, 0.0);
        let colour = crate::Color::new(color[0], color[1], color[2]);
        let kind = match kind {
            "standard" => GltfMaterialKind::Standard,
            "basic" => GltfMaterialKind::Basic,
            _ => GltfMaterialKind::Other,
        };
        let material: crate::Material = match kind {
            GltfMaterialKind::Basic => crate::materials::BasicMaterial {
                transparent,
                opacity,
                side,
                alpha_test,
                ..crate::materials::BasicMaterial::new(colour)
            }
            .into(),
            _ => crate::materials::StandardMaterial {
                metalness,
                roughness,
                opacity,
                side,
                emissive: crate::Color::new(emissive[0], emissive[1], emissive[2]),
                ..crate::materials::StandardMaterial::new(colour)
            }
            .into(),
        };
        let info = GltfMaterialInfo {
            name,
            kind,
            color,
            opacity,
            transparent,
            metalness,
            roughness,
            emissive,
            alpha_test,
            double_sided: side == 2,
        };
        self.materials.push((Arc::new(material), info));
        (self.materials.len() - 1) as u32
    }

    /// Give object `index` material `material` (from [`add_material`]).
    ///
    /// [`add_material`]: Self::add_material
    #[wasm_bindgen(js_name = setMaterial)]
    pub fn set_material(&mut self, index: u32, material: u32) -> Result<(), JsValue> {
        if material as usize >= self.materials.len() {
            return Err(JsValue::from_str(&format!(
                "setMaterial: no material {material}"
            )));
        }
        self.entry(index)?.material = Some(material as usize);
        Ok(())
    }

    /// three.js `OBJExporter.parse`. `polygons`: threers extension, one face
    /// per polygon the geometry remembers.
    pub fn obj(&self, polygons: bool) -> String {
        crate::exporters::write_obj(
            &self.flat(),
            &ObjOptions {
                polygons,
                numbers: NumberFormat::JavaScript,
            },
        )
    }

    /// three.js `STLExporter.parse(…, { binary: false })`.
    #[wasm_bindgen(js_name = stlAscii)]
    pub fn stl_ascii(&self) -> String {
        crate::exporters::write_stl_ascii(&self.flat(), NumberFormat::JavaScript)
    }

    /// three.js `STLExporter.parse(…, { binary: true })`.
    #[wasm_bindgen(js_name = stlBinary)]
    pub fn stl_binary(&self) -> Vec<u8> {
        crate::exporters::write_stl_binary(&self.flat())
    }

    /// three.js `PLYExporter.parse`: the file's bytes (ASCII text when not
    /// `binary`), or `undefined` where three.js returns `null`.
    pub fn ply(
        &self,
        binary: bool,
        little_endian: bool,
        exclude_attributes: Vec<String>,
        polygons: bool,
    ) -> Option<Vec<u8>> {
        crate::exporters::write_ply(
            &self.flat(),
            &PlyOptions {
                binary,
                little_endian,
                exclude_attributes,
                polygons,
                numbers: NumberFormat::JavaScript,
                comment: None,
            },
        )
    }

    /// three.js `GLTFExporter.parse` with `binary: false`: the JSON text, its
    /// buffer inline as a `data:` URI. Roots are the objects added without a
    /// parent; `sceneName` names `scenes[0]`.
    pub fn gltf(&self, trs: bool, scene_name: Option<String>) -> String {
        let (arena, roots, options) = self.arena(trs, scene_name);
        crate::loaders::scene_to_gltf_with(
            &arena,
            &roots,
            &GltfExportOptions {
                embed_buffer: true,
                ..options
            },
        )
        .0
    }

    /// three.js `GLTFExporter.parse` with `binary: true`: a `.glb`.
    pub fn glb(&self, trs: bool, scene_name: Option<String>) -> Vec<u8> {
        let (arena, roots, options) = self.arena(trs, scene_name);
        crate::loaders::scene_to_glb_with(&arena, &roots, &options)
    }
}

impl WebSceneExporter {
    fn entry(&mut self, index: u32) -> Result<&mut Entry, JsValue> {
        self.entries
            .get_mut(index as usize)
            .ok_or_else(|| JsValue::from_str(&format!("no object {index}")))
    }

    /// The objects with something to write, in the order they were added.
    fn flat(&self) -> Vec<ExportObject> {
        self.entries
            .iter()
            .filter_map(|e| {
                Some(ExportObject {
                    name: e.name.clone(),
                    kind: e.kind?,
                    geometry: e.geometry.clone()?,
                    matrix_world: e.matrix_world,
                    material_name: e
                        .material
                        .map(|m| self.materials[m].1.name.clone())
                        .unwrap_or_default(),
                })
            })
            .collect()
    }

    /// The entries as a Rust scene graph: meshes with geometry and material,
    /// everything else (lines and points included) as plain nodes.
    fn arena(
        &self,
        trs: bool,
        scene_name: Option<String>,
    ) -> (
        crate::core::ObjectArena,
        Vec<crate::core::ObjectId>,
        GltfExportOptions,
    ) {
        let mut arena = crate::core::ObjectArena::new();
        let mut ids = Vec::with_capacity(self.entries.len());
        let mut roots = Vec::new();
        let default_material = Arc::new(crate::Material::from(
            crate::materials::StandardMaterial::new(crate::Color::WHITE),
        ));
        for e in &self.entries {
            let mut object = match (e.kind, &e.geometry) {
                (Some(ExportKind::Mesh), Some(g)) => {
                    let material = e
                        .material
                        .map(|m| self.materials[m].0.clone())
                        .unwrap_or_else(|| default_material.clone());
                    crate::core::Object3D::mesh(crate::core::Mesh::from_arc(g.clone(), material))
                }
                _ => crate::core::Object3D::group(),
            };
            object.name = e.name.clone();
            object.position = crate::Vector3::new(
                e.position[0] as f32,
                e.position[1] as f32,
                e.position[2] as f32,
            );
            object.quaternion = crate::Quaternion::new(
                e.quaternion[0] as f32,
                e.quaternion[1] as f32,
                e.quaternion[2] as f32,
                e.quaternion[3] as f32,
            );
            object.scale =
                crate::Vector3::new(e.scale[0] as f32, e.scale[1] as f32, e.scale[2] as f32);
            let id = arena.insert(object);
            match e.parent {
                Some(p) => arena.add_child(ids[p], id),
                None => roots.push(id),
            }
            ids.push(id);
        }
        let material_info = self
            .materials
            .iter()
            .map(|(m, info)| (Arc::as_ptr(m) as usize, info.clone()))
            .collect();
        let options = GltfExportOptions {
            generator: "THREE.GLTFExporter r165".into(),
            trs,
            three_js: true,
            scene_name,
            material_info,
            ..GltfExportOptions::default()
        };
        (arena, roots, options)
    }
}
