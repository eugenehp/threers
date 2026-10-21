//! `GLTFLoader` for the shim: three.js r165's parser, ported to
//! `crate::loaders::gltf`, building every object as a node of the scene
//! graph the shim mirrors into — so the shim's `Object3D`s wrap nodes the
//! loader made rather than describing a scene for JS to build again.
//!
//! The JS side drives the load: it fetches what the file names by URI
//! (`externalBuffers`, `pendingImages`), decodes the images the crate cannot
//! (`setImageBytes` decodes PNG and KTX2 here and says `false` otherwise;
//! `setImageRgba` takes a browser decode), then calls `parse` with the
//! `WebScene` the nodes go into. `parse` (and `load`, three.js's lazy
//! `parser.getDependency`) returns what is new since the last call: objects
//! with their node ids, geometries with their attribute arrays and the
//! `WebGeometry` the nodes share, materials with three.js's parameters and
//! the `WebMaterial` the nodes draw with, textures, skeletons and clips.
#![allow(unused_imports)]

use super::*;
use crate::loaders::gltf_parser::{
    Assoc, CameraParams, GltfOptions, GltfParser, ImageRequest, MaterialType, ObjectType, Param,
    TrackInterpolation, TrackKind,
};
use crate::loaders::json::Value;

fn err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

/// A JSON value as a plain JS value.
fn json_js(v: &Value) -> JsValue {
    match v {
        Value::Null => JsValue::NULL,
        Value::Bool(b) => (*b).into(),
        Value::Number(n) => (*n).into(),
        Value::String(s) => JsValue::from_str(s),
        Value::Array(a) => {
            let out = js_sys::Array::new();
            for x in a {
                out.push(&json_js(x));
            }
            out.into()
        }
        Value::Object(m) => {
            let out = js_sys::Object::new();
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            for k in keys {
                js_set(&out, k, &json_js(&m[k]));
            }
            out.into()
        }
    }
}

fn f32s(v: &[f32]) -> JsValue {
    js_sys::Float32Array::from(v).into()
}

fn numbers(v: &[f64]) -> JsValue {
    let a = js_sys::Array::new();
    for x in v {
        a.push(&(*x).into());
    }
    a.into()
}

/// `Absent` → no key; `Undefined` → -1; a record → its index.
fn assoc_js(a: Assoc) -> JsValue {
    match a {
        Assoc::Absent => JsValue::UNDEFINED,
        Assoc::Undefined => (-1.0).into(),
        Assoc::Record(r) => (r as f64).into(),
    }
}

/// What has been handed to JS so far.
#[derive(Default)]
struct Delivered {
    objects: usize,
    geometries: usize,
    materials: usize,
    textures: usize,
    skeletons: usize,
    clips: usize,
}

/// One glTF load in progress (and, after `parse`, three.js's
/// `parser.getDependency` for what no scene used).
#[wasm_bindgen]
pub struct WebGltfParser {
    parser: GltfParser,
    delivered: Delivered,
}

#[wasm_bindgen]
impl WebGltfParser {
    /// Read a `.glb` or `.gltf` (JSON text) file. Throws three.js's errors
    /// for a bad container or a pre-2.0 asset.
    #[wasm_bindgen(constructor)]
    pub fn new(data: &[u8]) -> Result<WebGltfParser, JsError> {
        let doc = crate::loaders::gltf_parser::Document::from_bytes(data).map_err(err)?;
        // three.js warns about an unknown required extension and goes on.
        let options = GltfOptions {
            refuse_unknown_required: false,
            ..Default::default()
        };
        let parser = GltfParser::new(doc, options).map_err(err)?;
        Ok(WebGltfParser {
            parser,
            delivered: Delivered::default(),
        })
    }

    /// The JSON text (a GLB's JSON chunk), for `parser.json`.
    #[wasm_bindgen(js_name = jsonText)]
    pub fn json_text(&self) -> String {
        self.parser.doc.json_text.clone()
    }

    /// Warnings three.js would have logged, since the last call.
    #[wasm_bindgen(js_name = takeWarnings)]
    pub fn take_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.parser.warnings)
    }

    /// Whether the image decoder takes WebP / AVIF (`EXT_texture_webp`,
    /// `EXT_texture_avif`).
    #[wasm_bindgen(js_name = setImageSupport)]
    pub fn set_image_support(&mut self, webp: bool, avif: bool) {
        self.parser.set_image_support(webp, avif);
    }

    /// `[[index, uri], …]`: the buffers to fetch (relative to the resource
    /// path). `data:` URIs and a GLB body are already in.
    #[wasm_bindgen(js_name = externalBuffers)]
    pub fn external_buffers(&self) -> js_sys::Array {
        let out = js_sys::Array::new();
        for (i, uri) in self.parser.doc.external_buffers() {
            let pair = js_sys::Array::new();
            pair.push(&(i as f64).into());
            pair.push(&JsValue::from_str(&uri));
            out.push(&pair);
        }
        out
    }

    #[wasm_bindgen(js_name = setBuffer)]
    pub fn set_buffer(&mut self, index: usize, bytes: &[u8]) {
        self.parser.doc.set_buffer(index, bytes.to_vec());
    }

    /// A buffer that would not load: an error once something reads it.
    #[wasm_bindgen(js_name = setBufferError)]
    pub fn set_buffer_error(&mut self, index: usize, message: String) {
        self.parser.doc.set_buffer_error(index, message);
    }

    /// The images the load needs that have not been supplied:
    /// `[{index, uri, bytes, mimeType}]` — `uri` to fetch, or the `bytes`
    /// the file holds. Call until it returns none (a KTX2 that would not
    /// decode makes its texture's fallback source needed).
    #[wasm_bindgen(js_name = pendingImages)]
    pub fn pending_images(&mut self) -> Result<js_sys::Array, JsError> {
        let reqs = self.parser.pending_images().map_err(err)?;
        Ok(requests_js(&reqs))
    }

    /// As `pendingImages`, for a lazily loaded dependency.
    #[wasm_bindgen(js_name = pendingImagesFor)]
    pub fn pending_images_for(
        &mut self,
        kind: &str,
        index: usize,
    ) -> Result<js_sys::Array, JsError> {
        let reqs = self.parser.pending_images_for(kind, index).map_err(err)?;
        Ok(requests_js(&reqs))
    }

    /// An image's encoded bytes: decoded here if PNG or KTX2 (true);
    /// otherwise false, for the caller to decode (`setImageRgba`).
    #[wasm_bindgen(js_name = setImageBytes)]
    pub fn set_image_bytes(&mut self, index: usize, bytes: &[u8]) -> bool {
        self.parser.set_image_bytes(index, bytes)
    }

    /// An image decoded by the caller: RGBA8, rows top first.
    #[wasm_bindgen(js_name = setImageRgba)]
    pub fn set_image_rgba(&mut self, index: usize, width: u32, height: u32, rgba: Vec<u8>) {
        self.parser.set_image_rgba(index, width, height, rgba);
    }

    /// An image that could not be loaded (its textures become `null`).
    #[wasm_bindgen(js_name = setImageFailed)]
    pub fn set_image_failed(&mut self, index: usize, reason: String) {
        self.parser.set_image_failed(index, reason);
    }

    /// `parser.parse`: every scene, animation and camera, as nodes of
    /// `scene`. Returns what was made, plus `scenes` (object index per
    /// scene), `scene` (the default's), `animations` (clip per animation),
    /// `cameras` (object per camera, -1 for none) and the result's `userData`.
    pub fn parse(&mut self, scene: &mut WebScene) -> Result<JsValue, JsError> {
        self.parser.parse(&mut scene.inner.arena).map_err(err)?;
        let out = self.delta(scene);
        let idx = |v: &[Option<usize>]| -> JsValue {
            numbers(
                &v.iter()
                    .map(|x| x.map(|i| i as f64).unwrap_or(-1.0))
                    .collect::<Vec<_>>(),
            )
        };
        js_set(&out, "scenes", &idx(&self.parser.scenes));
        js_set(
            &out,
            "scene",
            &self
                .parser
                .default_scene()
                .map(|s| JsValue::from(s as f64))
                .unwrap_or(JsValue::UNDEFINED),
        );
        js_set(&out, "animations", &idx(&self.parser.animations));
        js_set(&out, "cameras", &idx(&self.parser.cameras));
        js_set(&out, "userData", &json_js(&self.parser.user_data));
        Ok(out.into())
    }

    /// Lazily load `kind` (`scene`, `node`, `mesh`, `material`, `texture`,
    /// `camera`, `skin`, `animation`, `light`) number `index`: what was made
    /// (as from `parse`) and `result`, the index of the thing in its list
    /// (-1 for a `null` texture or camera).
    pub fn load(
        &mut self,
        scene: &mut WebScene,
        kind: &str,
        index: usize,
    ) -> Result<JsValue, JsError> {
        let r = self
            .parser
            .load(&mut scene.inner.arena, kind, index)
            .map_err(err)?;
        let out = self.delta(scene);
        js_set(&out, "result", &r.map(|i| i as f64).unwrap_or(-1.0).into());
        Ok(out.into())
    }

    /// `getDependency('accessor', i)`: `{itemSize, count, array}` (normalized
    /// integers as the fractions they stand for).
    pub fn accessor(&self, index: usize) -> Result<JsValue, JsError> {
        let a = self.parser.doc.accessor(index).map_err(err)?;
        let o = js_sys::Object::new();
        js_set(&o, "itemSize", &(a.item_size as f64).into());
        js_set(&o, "count", &(a.count as f64).into());
        js_set(&o, "array", &f32s(&a.values()));
        Ok(o.into())
    }

    /// `getDependency('bufferView', i)`: its bytes (decoded, for meshopt).
    #[wasm_bindgen(js_name = bufferView)]
    pub fn buffer_view(&mut self, index: usize) -> Result<Vec<u8>, JsError> {
        self.parser.doc.decode_meshopt_views().map_err(err)?;
        Ok(self.parser.doc.buffer_view(index).map_err(err)?.to_vec())
    }

    /// `getDependency('buffer', i)`.
    pub fn buffer(&self, index: usize) -> Result<Vec<u8>, JsError> {
        Ok(self.parser.doc.buffer(index).map_err(err)?.to_vec())
    }

    /// Whether object `i` is in `parser.associations` (three.js drops the
    /// objects no scene holds).
    #[wasm_bindgen(js_name = objectAssociated)]
    pub fn object_associated(&self, i: usize) -> bool {
        self.parser.object_associated(i)
    }
}

fn requests_js(reqs: &[ImageRequest]) -> js_sys::Array {
    let out = js_sys::Array::new();
    for r in reqs {
        let o = js_sys::Object::new();
        js_set(&o, "index", &(r.index as f64).into());
        js_set(&o, "mimeType", &JsValue::from_str(&r.source.mime_type));
        if let Some(uri) = &r.source.uri {
            js_set(&o, "uri", &JsValue::from_str(uri));
        }
        if let Some(b) = &r.source.bytes {
            js_set(&o, "bytes", &js_sys::Uint8Array::from(&b[..]).into());
        }
        out.push(&o);
    }
    out
}

fn param_js(p: &Param) -> JsValue {
    match p {
        Param::Num(n) => (*n).into(),
        Param::Bool(b) => (*b).into(),
        Param::Color(c) => {
            let o = js_sys::Object::new();
            js_set(
                &o,
                "color",
                &numbers(&[c[0] as f64, c[1] as f64, c[2] as f64]),
            );
            o.into()
        }
        Param::Vec2(v) => {
            let o = js_sys::Object::new();
            js_set(&o, "vec2", &numbers(&[v[0] as f64, v[1] as f64]));
            o.into()
        }
        Param::Range(v) => numbers(&[v[0] as f64, v[1] as f64]),
        Param::Texture(t) => {
            let o = js_sys::Object::new();
            js_set(&o, "texture", &(*t as f64).into());
            o.into()
        }
    }
}

impl WebGltfParser {
    /// Everything made since the last delivery.
    fn delta(&mut self, scene: &WebScene) -> js_sys::Object {
        let p = &mut self.parser;
        let d = &mut self.delivered;
        let out = js_sys::Object::new();

        let textures = js_sys::Array::new();
        for i in d.textures..p.textures.len() {
            let srgb = p.textures[i].srgb;
            let w = WebTexture {
                inner: p.textures[i].rust(srgb),
            };
            let t = &p.textures[i];
            let o = js_sys::Object::new();
            let (width, height) = t.image_data.size();
            js_set(&o, "index", &(i as f64).into());
            js_set(&o, "w", &w.into());
            js_set(&o, "width", &width.into());
            js_set(&o, "height", &height.into());
            js_set(&o, "name", &JsValue::from_str(&t.name));
            js_set(&o, "magFilter", &t.mag_filter.into());
            js_set(&o, "minFilter", &t.min_filter.into());
            js_set(&o, "wrapS", &t.wrap_s.into());
            js_set(&o, "wrapT", &t.wrap_t.into());
            js_set(&o, "channel", &t.channel.into());
            js_set(
                &o,
                "offset",
                &numbers(&[t.offset[0] as f64, t.offset[1] as f64]),
            );
            js_set(
                &o,
                "repeat",
                &numbers(&[t.repeat[0] as f64, t.repeat[1] as f64]),
            );
            js_set(&o, "rotation", &(t.rotation as f64).into());
            js_set(&o, "srgb", &t.srgb.into());
            js_set(&o, "image", &(t.image as f64).into());
            js_set(&o, "userData", &json_js(&t.user_data));
            js_set(&o, "association", &assoc_js(t.association));
            textures.push(&o);
        }
        d.textures = p.textures.len();
        js_set(&out, "textures", &textures);

        let geometries = js_sys::Array::new();
        for i in d.geometries..p.geometries.len() {
            let g = &p.geometries[i];
            let Some(arc) = g.arc.clone() else { continue };
            let o = js_sys::Object::new();
            js_set(&o, "index", &(i as f64).into());
            js_set(&o, "w", &WebGeometry { inner: arc.clone() }.into());
            js_set(
                &o,
                "wBuffer",
                &WebBufferGeometry { inner: arc.clone() }.into(),
            );
            let attrs = js_sys::Array::new();
            for name in &g.attribute_order {
                let Some(a) = g.get().get_attribute(name) else {
                    continue;
                };
                let e = js_sys::Object::new();
                js_set(&e, "name", &JsValue::from_str(name));
                js_set(&e, "itemSize", &(a.item_size as f64).into());
                js_set(&e, "array", &f32s(&a.array));
                attrs.push(&e);
            }
            js_set(&o, "attributes", &attrs);
            if let Some(index) = &g.get().index {
                js_set(&o, "indices", &js_sys::Uint32Array::from(&index[..]).into());
            }
            let morph = js_sys::Array::new();
            for name in &g.morph_order {
                let list = js_sys::Array::new();
                for a in g
                    .get()
                    .morph_attributes
                    .get(name)
                    .map(|l| l.as_slice())
                    .unwrap_or(&[])
                {
                    let e = js_sys::Object::new();
                    js_set(&e, "itemSize", &(a.item_size as f64).into());
                    js_set(&e, "array", &f32s(&a.array));
                    list.push(&e);
                }
                let e = js_sys::Object::new();
                js_set(&e, "name", &JsValue::from_str(name));
                js_set(&e, "targets", &list);
                morph.push(&e);
            }
            js_set(&o, "morphAttributes", &morph);
            js_set(
                &o,
                "morphTargetsRelative",
                &g.get().morph_targets_relative.into(),
            );
            js_set(&o, "userData", &json_js(&g.user_data));
            if let (Some(b), Some(s)) = (g.get().bounding_box, g.get().bounding_sphere) {
                let v = |x: crate::Vector3| [x.x as f64, x.y as f64, x.z as f64];
                js_set(&o, "boxMin", &numbers(&v(b.min)));
                js_set(&o, "boxMax", &numbers(&v(b.max)));
                js_set(&o, "sphereCenter", &numbers(&v(s.center)));
                js_set(&o, "sphereRadius", &(s.radius as f64).into());
            }
            geometries.push(&o);
        }
        d.geometries = p.geometries.len();
        js_set(&out, "geometries", &geometries);

        let materials = js_sys::Array::new();
        for i in d.materials..p.materials.len() {
            let m = &p.materials[i];
            let o = js_sys::Object::new();
            js_set(&o, "index", &(i as f64).into());
            js_set(&o, "type", &JsValue::from_str(m.ty.name()));
            js_set(&o, "name", &JsValue::from_str(&m.name));
            js_set(&o, "userData", &json_js(&m.user_data));
            js_set(&o, "association", &assoc_js(m.association));
            let params = js_sys::Object::new();
            for (k, v) in &m.params {
                js_set(&params, k, &param_js(v));
            }
            js_set(&o, "params", &params);
            if let Some(rust) = &m.rust {
                js_set(
                    &o,
                    "w",
                    &WebMaterial {
                        inner: rust.clone(),
                    }
                    .into(),
                );
            }
            materials.push(&o);
        }
        d.materials = p.materials.len();
        js_set(&out, "materials", &materials);

        let skeletons = js_sys::Array::new();
        for i in d.skeletons..p.skeletons.len() {
            let s = &p.skeletons[i];
            let o = js_sys::Object::new();
            js_set(&o, "index", &(i as f64).into());
            js_set(
                &o,
                "bones",
                &numbers(&s.bones.iter().map(|b| *b as f64).collect::<Vec<_>>()),
            );
            let flat: Vec<f32> = s.inverses.iter().flat_map(|m| m.iter().copied()).collect();
            js_set(&o, "inverses", &f32s(&flat));
            skeletons.push(&o);
        }
        d.skeletons = p.skeletons.len();
        js_set(&out, "skeletons", &skeletons);

        let objects = js_sys::Array::new();
        for i in d.objects..p.objects.len() {
            let ob = &p.objects[i];
            let o = js_sys::Object::new();
            js_set(&o, "index", &(i as f64).into());
            js_set(&o, "id", &object_id_to_f64(ob.id).into());
            js_set(&o, "type", &JsValue::from_str(ob.ty.name()));
            js_set(&o, "name", &JsValue::from_str(&ob.name));
            js_set(&o, "userData", &json_js(&ob.user_data));
            js_set(&o, "association", &assoc_js(ob.association));
            if let Some(node) = scene.inner.arena.get(ob.id) {
                let (pos, q, s) = (node.position, node.quaternion, node.scale);
                js_set(
                    &o,
                    "position",
                    &numbers(&[pos.x as f64, pos.y as f64, pos.z as f64]),
                );
                js_set(
                    &o,
                    "quaternion",
                    &numbers(&[q.x as f64, q.y as f64, q.z as f64, q.w as f64]),
                );
                js_set(&o, "scale", &numbers(&[s.x as f64, s.y as f64, s.z as f64]));
                if let crate::core::ObjectKind::Light(l) = &node.kind {
                    let inner = match l {
                        crate::lights::Light::Directional(x) => Some(LightInner::Directional(*x)),
                        crate::lights::Light::Point(x) => Some(LightInner::Point(*x)),
                        crate::lights::Light::Spot(x) => Some(LightInner::Spot(*x)),
                        _ => None,
                    };
                    if let Some(inner) = inner {
                        js_set(&o, "w", &WebLight { inner }.into());
                    }
                }
            }
            if let Some(g) = ob.geometry {
                js_set(&o, "geometry", &(g as f64).into());
            }
            if let Some(m) = ob.material {
                js_set(&o, "material", &(m as f64).into());
            }
            if let Some(s) = ob.skeleton {
                js_set(&o, "skeleton", &(s as f64).into());
            }
            if let Some(inf) = &ob.morph_influences {
                js_set(
                    &o,
                    "morphTargetInfluences",
                    &numbers(&inf.iter().map(|x| *x as f64).collect::<Vec<_>>()),
                );
                let dict = js_sys::Object::new();
                for (k, v) in &ob.morph_dictionary {
                    js_set(&dict, k, &(*v as f64).into());
                }
                js_set(&o, "morphTargetDictionary", &dict);
            }
            if let Some(inst) = &ob.instances {
                js_set(&o, "count", &(inst.count as f64).into());
                js_set(&o, "instanceMatrix", &f32s(&inst.matrices));
                if let Some((c, size)) = &inst.colors {
                    js_set(&o, "instanceColor", &f32s(c));
                    js_set(&o, "instanceColorSize", &(*size as f64).into());
                }
            }
            if let Some(c) = ob.camera {
                let v = match c {
                    CameraParams::Perspective {
                        fov,
                        aspect,
                        near,
                        far,
                    } => vec![fov, aspect, near, far],
                    CameraParams::Orthographic {
                        left,
                        right,
                        top,
                        bottom,
                        near,
                        far,
                    } => {
                        vec![left, right, top, bottom, near, far]
                    }
                };
                js_set(&o, "camera", &numbers(&v));
            }
            if let Some(l) = ob.light {
                let e = js_sys::Object::new();
                js_set(
                    &e,
                    "color",
                    &numbers(&[l.color[0] as f64, l.color[1] as f64, l.color[2] as f64]),
                );
                js_set(&e, "intensity", &(l.intensity as f64).into());
                js_set(&e, "distance", &(l.distance as f64).into());
                js_set(&e, "decay", &(l.decay as f64).into());
                js_set(&e, "angle", &(l.angle as f64).into());
                js_set(&e, "penumbra", &(l.penumbra as f64).into());
                js_set(&o, "light", &e);
            }
            if let Some(t) = ob.light_target {
                js_set(&o, "lightTarget", &(t as f64).into());
            }
            objects.push(&o);
        }
        d.objects = p.objects.len();
        js_set(&out, "objects", &objects);

        // Children of every object (a lazy load can re-parent an old one).
        let children = js_sys::Array::new();
        for ob in &p.objects {
            let kids = scene
                .inner
                .arena
                .get(ob.id)
                .map(|n| {
                    n.children
                        .iter()
                        .filter_map(|c| p.object_index(*c))
                        .map(|i| i as f64)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            children.push(&numbers(&kids));
        }
        js_set(&out, "children", &children);

        let records = js_sys::Array::new();
        for r in &p.records {
            let o = js_sys::Object::new();
            for (k, v) in [
                ("nodes", r.nodes),
                ("meshes", r.meshes),
                ("primitives", r.primitives),
                ("materials", r.materials),
                ("textures", r.textures),
            ] {
                if let Some(v) = v {
                    js_set(&o, k, &(v as f64).into());
                }
            }
            records.push(&o);
        }
        js_set(&out, "records", &records);

        let clips = js_sys::Array::new();
        for i in d.clips..p.clips.len() {
            let c = &p.clips[i];
            let o = js_sys::Object::new();
            js_set(&o, "index", &(i as f64).into());
            js_set(&o, "name", &JsValue::from_str(&c.name));
            let tracks = js_sys::Array::new();
            for t in &c.tracks {
                let e = js_sys::Object::new();
                js_set(&e, "target", &(t.target as f64).into());
                js_set(&e, "property", &JsValue::from_str(&t.property));
                js_set(
                    &e,
                    "kind",
                    &JsValue::from_str(match t.kind {
                        TrackKind::Number => "number",
                        TrackKind::Vector => "vector",
                        TrackKind::Quaternion => "quaternion",
                    }),
                );
                js_set(&e, "times", &f32s(&t.times));
                js_set(&e, "values", &f32s(&t.values));
                js_set(
                    &e,
                    "interpolation",
                    &JsValue::from_str(match t.interpolation {
                        TrackInterpolation::Linear => "LINEAR",
                        TrackInterpolation::Step => "STEP",
                        TrackInterpolation::CubicSpline => "CUBICSPLINE",
                    }),
                );
                tracks.push(&e);
            }
            js_set(&o, "tracks", &tracks);
            clips.push(&o);
        }
        d.clips = p.clips.len();
        js_set(&out, "clips", &clips);
        out
    }
}

#[wasm_bindgen]
impl WebScene {
    /// Move `node` — the node itself, with its geometry, materials and all —
    /// out of `from` (another scene's graph) into this one, detached; its
    /// children stay in `from`, parentless, until they are moved in turn.
    /// The handle here, or `undefined` for `from`'s root or a missing node.
    ///
    /// The shim moves an object's node this way when the object goes from one
    /// graph to another (the hidden graph to a `Scene`, one `Scene` to
    /// another), so a node the glTF loader built stays the node that object
    /// is drawn by.
    #[wasm_bindgen(js_name = adoptNode)]
    pub fn adopt_node(
        &mut self,
        from: &mut WebScene,
        node: &WebObjectHandle,
    ) -> Option<WebObjectHandle> {
        if node.id == from.inner.root {
            return None;
        }
        from.inner.arena.remove_child(node.id);
        let mut obj = from.inner.arena.nodes.remove(node.id)?;
        for c in &obj.children {
            if let Some(child) = from.inner.arena.get_mut(*c) {
                if child.parent == Some(node.id) {
                    child.parent = None;
                }
            }
        }
        obj.parent = None;
        obj.children.clear();
        Some(WebObjectHandle {
            id: self.inner.arena.insert(obj),
        })
    }
}
