//! Bindings split out of `wasm.rs` by area (see the `mod` list there): mesh bvh (feature mesh-bvh).

use super::*;

// ---- mesh bvh (feature mesh-bvh) ----
#[cfg(all(target_arch = "wasm32", feature = "mesh-bvh"))]
#[wasm_bindgen]
pub struct WebMeshBvh {
    pub(crate) inner: Arc<crate::mesh_bvh::MeshBvh>,
}

#[cfg(all(target_arch = "wasm32", feature = "mesh-bvh"))]
#[wasm_bindgen]
impl WebMeshBvh {
    #[wasm_bindgen(constructor)]
    pub fn new(geometry: &WebBufferGeometry) -> Result<WebMeshBvh, JsValue> {
        Self::new_with_options(geometry, 40, 10)
    }

    #[wasm_bindgen(js_name = newWithOptions)]
    pub fn new_with_options(
        geometry: &WebBufferGeometry,
        max_depth: u32,
        max_leaf_tris: u32,
    ) -> Result<WebMeshBvh, JsValue> {
        Self::new_with_options_full(
            geometry,
            max_depth,
            max_leaf_tris,
            crate::mesh_bvh::CENTER,
            0,
            0,
            false,
        )
    }

    #[wasm_bindgen(js_name = newWithOptionsFull)]
    pub fn new_with_options_full(
        geometry: &WebBufferGeometry,
        max_depth: u32,
        max_leaf_tris: u32,
        strategy: u32,
        offset: u32,
        count: u32,
        indirect: bool,
    ) -> Result<WebMeshBvh, JsValue> {
        // Spread the default rather than listing every field: this broke once
        // already, when `split_degenerate` was added to the struct and only the
        // wasm construction was left behind — so the browser build of mesh-BVH
        // stopped compiling while every native check stayed green. The default
        // is also what keeps these APIs behaving as the native ones do, which is
        // the point of mirroring three-mesh-bvh's interface.
        let options = crate::mesh_bvh::BuildOptions {
            strategy,
            max_depth,
            max_leaf_tris,
            offset,
            count,
            indirect,
            ..Default::default()
        };
        let bvh = crate::mesh_bvh::MeshBvh::build(&geometry.inner, options)
            .ok_or_else(|| JsValue::from_str("failed to build MeshBVH"))?;
        Ok(WebMeshBvh {
            inner: Arc::new(bvh),
        })
    }

    /// Packed hits: per entry `[distance, px, py, pz, face_index, u, v]`.
    pub fn raycast(
        &self,
        ox: f32,
        oy: f32,
        oz: f32,
        dx: f32,
        dy: f32,
        dz: f32,
        near: f32,
        far: f32,
    ) -> Vec<f32> {
        let ray = crate::Ray::new(
            crate::Vector3::new(ox, oy, oz),
            crate::Vector3::new(dx, dy, dz),
        );
        let hits = self.inner.raycast(&ray, near, far, true);
        pack_hits(&hits)
    }

    /// Closest hit packed as `[distance, px, py, pz, face_index, u, v]`, or empty vec.
    #[wasm_bindgen(js_name = raycastFirst)]
    pub fn raycast_first(
        &self,
        ox: f32,
        oy: f32,
        oz: f32,
        dx: f32,
        dy: f32,
        dz: f32,
        near: f32,
        far: f32,
        side: u32,
    ) -> Vec<f32> {
        let ray = crate::Ray::new(
            crate::Vector3::new(ox, oy, oz),
            crate::Vector3::new(dx, dy, dz),
        );
        match self.inner.raycast_first_with_side(&ray, near, far, side) {
            Some(h) => pack_hits(&[h]),
            None => Vec::new(),
        }
    }

    #[wasm_bindgen(js_name = resolveTriangleIndex)]
    pub fn resolve_triangle_index(&self, bvh_triangle_index: u32) -> i32 {
        match self
            .inner
            .resolve_triangle_index(bvh_triangle_index as usize)
        {
            Some(i) => i as i32,
            None => -1,
        }
    }

    /// Packed triangle pairs: `[ia, ib, ia, ib, ...]` (BVH-layout indices).
    #[wasm_bindgen(js_name = bvhcast)]
    pub fn bvhcast(&self, other: &WebMeshBvh, matrix: &[f32]) -> Vec<u32> {
        if matrix.len() < 16 {
            return Vec::new();
        }
        let mut elems = [0.0f32; 16];
        elems.copy_from_slice(&matrix[..16]);
        let m = crate::Matrix4 { elements: elems };
        let pairs = self.inner.bvhcast(&other.inner, &m);
        let mut out = Vec::with_capacity(pairs.len() * 2);
        for (a, b) in pairs {
            out.push(a as u32);
            out.push(b as u32);
        }
        out
    }

    #[wasm_bindgen(js_name = getBoundingBox)]
    pub fn get_bounding_box(&self) -> WebBox3 {
        WebBox3 {
            inner: self.inner.bounding_box(),
        }
    }

    #[wasm_bindgen(js_name = nodeBuffer)]
    pub fn node_buffer(&self) -> Vec<f32> {
        self.inner.node_buffer().to_vec()
    }

    #[wasm_bindgen(js_name = nodeCount)]
    pub fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    #[wasm_bindgen(js_name = triangleCount)]
    pub fn triangle_count(&self) -> usize {
        self.inner.triangle_count()
    }

    #[wasm_bindgen(js_name = triangleIndices)]
    pub fn triangle_indices(&self) -> Vec<u32> {
        self.inner
            .triangle_indices()
            .iter()
            .flat_map(|(a, b, c)| [*a, *b, *c])
            .collect()
    }

    pub fn positions(&self) -> Vec<f32> {
        self.inner.positions().to_vec()
    }

    #[wasm_bindgen(js_name = triangleOrder)]
    pub fn triangle_order(&self) -> Vec<u32> {
        self.inner
            .triangle_order()
            .iter()
            .map(|&i| i as u32)
            .collect()
    }

    #[wasm_bindgen(js_name = refit)]
    pub fn refit(&mut self, positions: &[f32]) {
        Arc::make_mut(&mut self.inner).refit(positions);
    }

    #[wasm_bindgen(js_name = intersectsBox)]
    pub fn intersects_box(
        &self,
        min_x: f32,
        min_y: f32,
        min_z: f32,
        max_x: f32,
        max_y: f32,
        max_z: f32,
    ) -> bool {
        let b = crate::Box3::new(
            crate::Vector3::new(min_x, min_y, min_z),
            crate::Vector3::new(max_x, max_y, max_z),
        );
        self.inner.intersects_box(&b)
    }

    #[wasm_bindgen(js_name = intersectsSphere)]
    pub fn intersects_sphere(&self, cx: f32, cy: f32, cz: f32, radius: f32) -> bool {
        let s = crate::Sphere::new(crate::Vector3::new(cx, cy, cz), radius);
        self.inner.intersects_sphere(&s)
    }

    /// Closest point: `[px, py, pz, distance, face_index]`.
    #[wasm_bindgen(js_name = closestPointToPoint)]
    pub fn closest_point_to_point(&self, px: f32, py: f32, pz: f32) -> Vec<f32> {
        let (point, dist, tri) = self
            .inner
            .closest_point_to_point(crate::Vector3::new(px, py, pz));
        vec![point.x, point.y, point.z, dist, tri as f32]
    }

    /// Serialize to flat arrays for JS: version, nodeBuffer, triangleOrder, triangleIndices, positions.
    #[wasm_bindgen(js_name = serialize)]
    pub fn serialize(&self) -> js_sys::Array {
        let data = self.inner.serialize();
        let arr = js_sys::Array::new();
        arr.push(&JsValue::from_f64(data.version as f64));
        arr.push(&js_sys::Float32Array::from(data.node_buffer.as_slice()).into());
        arr.push(&js_sys::Uint32Array::from(data.triangle_order.as_slice()).into());
        arr.push(&js_sys::Uint32Array::from(data.triangle_indices.as_slice()).into());
        arr.push(&js_sys::Float32Array::from(data.positions.as_slice()).into());
        arr
    }

    #[wasm_bindgen(js_name = deserialize)]
    pub fn deserialize(
        version: u32,
        node_buffer: &[f32],
        triangle_order: &[u32],
        triangle_indices: &[u32],
        positions: &[f32],
    ) -> Result<WebMeshBvh, JsValue> {
        let data = crate::mesh_bvh::SerializedMeshBvh {
            version,
            node_buffer: node_buffer.to_vec(),
            triangle_order: triangle_order.to_vec(),
            triangle_indices: triangle_indices.to_vec(),
            positions: positions.to_vec(),
        };
        let bvh = crate::mesh_bvh::MeshBvh::deserialize(data)
            .ok_or_else(|| JsValue::from_str("failed to deserialize MeshBVH"))?;
        Ok(WebMeshBvh {
            inner: Arc::new(bvh),
        })
    }
}

#[cfg(all(target_arch = "wasm32", feature = "mesh-bvh"))]
pub(super) fn pack_hits(hits: &[crate::mesh_bvh::BvhHit]) -> Vec<f32> {
    let mut out = Vec::with_capacity(hits.len() * 7);
    for h in hits {
        out.push(h.distance);
        out.push(h.point.x);
        out.push(h.point.y);
        out.push(h.point.z);
        out.push(h.face_index as f32);
        out.push(h.uv.x);
        out.push(h.uv.y);
    }
    out
}

#[cfg(all(target_arch = "wasm32", feature = "mesh-bvh"))]
#[wasm_bindgen]
impl WebBufferGeometry {
    #[wasm_bindgen(js_name = computeBoundsTree)]
    pub fn compute_bounds_tree(
        &mut self,
        max_depth: u32,
        max_leaf_tris: u32,
    ) -> Result<WebMeshBvh, JsValue> {
        self.compute_bounds_tree_full(
            max_depth,
            max_leaf_tris,
            crate::mesh_bvh::CENTER,
            0,
            0,
            false,
        )
    }

    #[wasm_bindgen(js_name = computeBoundsTreeFull)]
    pub fn compute_bounds_tree_full(
        &mut self,
        max_depth: u32,
        max_leaf_tris: u32,
        strategy: u32,
        offset: u32,
        count: u32,
        indirect: bool,
    ) -> Result<WebMeshBvh, JsValue> {
        // Spread the default rather than listing every field: this broke once
        // already, when `split_degenerate` was added to the struct and only the
        // wasm construction was left behind — so the browser build of mesh-BVH
        // stopped compiling while every native check stayed green. The default
        // is also what keeps these APIs behaving as the native ones do, which is
        // the point of mirroring three-mesh-bvh's interface.
        let options = crate::mesh_bvh::BuildOptions {
            strategy,
            max_depth,
            max_leaf_tris,
            offset,
            count,
            indirect,
            ..Default::default()
        };
        let bvh = Arc::make_mut(&mut self.inner)
            .compute_bounds_tree(options)
            .ok_or_else(|| JsValue::from_str("failed to build bounds tree"))?;
        Ok(WebMeshBvh { inner: bvh })
    }

    #[wasm_bindgen(js_name = disposeBoundsTree)]
    pub fn dispose_bounds_tree(&mut self) {
        Arc::make_mut(&mut self.inner).dispose_bounds_tree();
    }

    #[wasm_bindgen(js_name = hasBoundsTree)]
    pub fn has_bounds_tree(&self) -> bool {
        self.inner.bounds_tree.is_some()
    }
}

#[cfg(all(target_arch = "wasm32", feature = "mesh-bvh"))]
#[wasm_bindgen(js_name = mergeGeometries)]
pub fn merge_geometries(geometries: Vec<WebBufferGeometry>) -> Result<WebBufferGeometry, JsValue> {
    if geometries.is_empty() {
        return Err(JsValue::from_str("mergeGeometries: empty input"));
    }
    let refs: Vec<crate::BufferGeometry> = geometries.iter().map(|g| (*g.inner).clone()).collect();
    let merged = crate::merge_geometries(&refs)
        .ok_or_else(|| JsValue::from_str("mergeGeometries: incompatible geometries"))?;
    Ok(WebBufferGeometry {
        inner: Arc::new(merged),
    })
}

// ---------------------------------------------------------------------------
// Browser animation export (native-codec): GIF / APNG / WebM / MP4 from RGBA frames
// ---------------------------------------------------------------------------

#[cfg(feature = "native-codec")]
pub(super) fn js_rgba_frames(frames: &js_sys::Array) -> Result<Vec<Vec<u8>>, JsValue> {
    let n = frames.length() as usize;
    if n == 0 {
        return Err(JsValue::from_str("encode: need at least one frame"));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..frames.length() {
        let v = frames.get(i);
        let u8a = js_sys::Uint8Array::new(&v);
        let mut buf = vec![0u8; u8a.length() as usize];
        u8a.copy_to(&mut buf);
        out.push(buf);
    }
    Ok(out)
}

#[cfg(feature = "native-codec")]
pub(super) fn encode_browser_animation(
    width: u32,
    height: u32,
    fps: u32,
    codec: crate::BrowserCodec,
    transparent: bool,
    gif_colors: u16,
    frames: &js_sys::Array,
) -> Result<js_sys::Uint8Array, JsValue> {
    let collected = js_rgba_frames(frames)?;
    let opts = crate::AnimationEncodeOptions {
        width,
        height,
        fps: fps.max(1),
        codec,
        transparent,
        gif_colors: gif_colors.clamp(2, 256),
    };
    let bytes = crate::encode_animation_rgba(&opts, collected)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(js_sys::Uint8Array::from(bytes.as_slice()))
}

/// Encode RGBA frames to an animated GIF (`native-codec` feature).
#[cfg(feature = "native-codec")]
#[wasm_bindgen(js_name = encodeGifRgba)]
pub fn encode_gif_rgba(
    width: u32,
    height: u32,
    fps: u32,
    colors: u16,
    transparent: bool,
    frames: js_sys::Array,
) -> Result<js_sys::Uint8Array, JsValue> {
    encode_browser_animation(
        width,
        height,
        fps,
        crate::BrowserCodec::Gif,
        transparent,
        colors,
        &frames,
    )
}

/// Encode RGBA frames to an animated PNG (`native-codec` feature).
#[cfg(feature = "native-codec")]
#[wasm_bindgen(js_name = encodeApngRgba)]
pub fn encode_apng_rgba(
    width: u32,
    height: u32,
    fps: u32,
    transparent: bool,
    frames: js_sys::Array,
) -> Result<js_sys::Uint8Array, JsValue> {
    encode_browser_animation(
        width,
        height,
        fps,
        crate::BrowserCodec::Apng,
        transparent,
        256,
        &frames,
    )
}

/// Encode RGBA frames to a VP9 WebM (`native-codec` feature).
/// Width and height must be multiples of 8.
#[cfg(feature = "native-codec")]
#[wasm_bindgen(js_name = encodeWebmRgba)]
pub fn encode_webm_rgba(
    width: u32,
    height: u32,
    fps: u32,
    transparent: bool,
    frames: js_sys::Array,
) -> Result<js_sys::Uint8Array, JsValue> {
    encode_browser_animation(
        width,
        height,
        fps,
        crate::BrowserCodec::Webm,
        transparent,
        256,
        &frames,
    )
}

/// Encode RGBA frames to an H.264 MP4 (`native-codec` feature).
/// Width and height must be even; transparency is not supported.
#[cfg(feature = "native-codec")]
#[wasm_bindgen(js_name = encodeMp4Rgba)]
pub fn encode_mp4_rgba(
    width: u32,
    height: u32,
    fps: u32,
    frames: js_sys::Array,
) -> Result<js_sys::Uint8Array, JsValue> {
    encode_browser_animation(
        width,
        height,
        fps,
        crate::BrowserCodec::Mp4,
        false,
        256,
        &frames,
    )
}

/// `initThreadPool(n)`: start `n` threads, once, before anything that uses
/// them. Blocking calls such as `decodeVideo` must then run in a Worker, as a
/// browser will not let its main thread wait, and the page must be
/// cross-origin isolated (COOP and COEP headers) for `SharedArrayBuffer`.
#[cfg(all(target_arch = "wasm32", feature = "wasm-threads"))]
pub use wasm_bindgen_rayon::init_thread_pool;

/// Decode a video file to RGBA frames (`native-decode` feature): MP4/MOV
/// (H.264, HEVC, AV1), WebM (VP9, AV1), IVF (VP9, AV1), or a raw H.264/HEVC
/// Annex B stream. Returns `{ width, height, time, data }` per frame in
/// display order, where `data` is a `Uint8ClampedArray` ready for
/// `new ImageData(data, width, height)` and `time` is seconds from the first
/// frame (`undefined` for raw streams, which carry no timing).
///
/// Every frame is held at once, about 8 MB each at 1080p; for anything long use
/// [`decodeVideoStream`](decode_video_stream).
#[cfg(feature = "native-decode")]
#[wasm_bindgen(js_name = decodeVideo)]
pub fn decode_video(bytes: &[u8]) -> Result<js_sys::Array, JsValue> {
    let frames = js_sys::Array::new();
    let mut rgba = Vec::new();
    let skipped = crate::codec::decode::decode_video_in_order(bytes, |time, f| {
        frames.push(&rgba_object(time, &f, &mut rgba));
        Ok(true)
    })
    .map_err(|e| JsValue::from_str(&e))?;
    js_set(&frames, "skipped", &(skipped as u32).into());
    Ok(frames)
}

/// `decodeVideoStream(bytes, onFrame)`: [`decodeVideo`](decode_video) one
/// frame at a time, in display order, for clips too long to hold. `onFrame`
/// gets each `{ width, height, time, data }` and may keep it, or transfer
/// `data.buffer` to another thread; return `false` to stop.
///
/// Packets that will not decode are skipped, as a player would, and the
/// return value says how many; it throws only if no frame could be decoded.
/// `decodeVideo` does the same, with the count as the array's `skipped`.
///
/// ```js
/// // In a Worker, which a build with threads requires:
/// decodeVideoStream(bytes, (f) => {
///   postMessage(f, [f.data.buffer]);
/// });
/// ```
#[cfg(feature = "native-decode")]
#[wasm_bindgen(js_name = decodeVideoStream)]
pub fn decode_video_stream(bytes: &[u8], on_frame: &js_sys::Function) -> Result<u32, JsValue> {
    // A throw from `onFrame` comes back to the caller as itself.
    let mut thrown = None;
    let mut rgba = Vec::new();
    let r = crate::codec::decode::decode_video_in_order(bytes, |time, f| {
        match on_frame.call1(&JsValue::NULL, &rgba_object(time, &f, &mut rgba)) {
            Ok(v) => Ok(v != JsValue::FALSE),
            Err(e) => {
                thrown = Some(e);
                Err("onFrame threw".to_string())
            }
        }
    });
    match (thrown, r) {
        (Some(e), _) => Err(e),
        (None, r) => r.map(|n| n as u32).map_err(|e| JsValue::from_str(&e)),
    }
}

/// `{ width, height, time, data }`, straight into a JS buffer: wasm memory
/// holds one frame at a time, in `rgba`, reused from frame to frame.
#[cfg(feature = "native-decode")]
pub(super) fn rgba_object(
    time: Option<f64>,
    f: &crate::codec::decode::DecodedFrame,
    rgba: &mut Vec<u8>,
) -> js_sys::Object {
    f.to_rgba_into(rgba);
    let data = js_sys::Uint8ClampedArray::new_with_length(rgba.len() as u32);
    data.copy_from(rgba);
    let o = js_sys::Object::new();
    js_set(&o, "width", &f.width.into());
    js_set(&o, "height", &f.height.into());
    js_set(&o, "time", &time.map_or(JsValue::UNDEFINED, JsValue::from));
    js_set(&o, "data", &data);
    o
}

// ---------------------------------------------------------------------------
// OpenSCAD front end → geometry, for the in-browser gallery. Gated on the
// `openscad` feature (which pulls in the exact CSG kernel + bvh-csg fallback).
// ---------------------------------------------------------------------------

/// A parsed OpenSCAD model as a flat, flat-shaded triangle soup: `positions` and
/// per-face `normals` (9 floats per triangle each), ready to drop straight into a
/// `BufferGeometry`. `error` is set (and the arrays empty) when parsing fails.
#[cfg(feature = "openscad")]
#[wasm_bindgen]
pub struct ScadGeometry {
    pub(super) positions: Vec<f32>,
    pub(super) normals: Vec<f32>,
    pub(super) triangles: u32,
    pub(super) error: Option<String>,
}

#[cfg(feature = "openscad")]
#[wasm_bindgen]
impl ScadGeometry {
    #[wasm_bindgen(getter)]
    pub fn positions(&self) -> Vec<f32> {
        self.positions.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn normals(&self) -> Vec<f32> {
        self.normals.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn triangles(&self) -> u32 {
        self.triangles
    }
    #[wasm_bindgen(getter)]
    pub fn error(&self) -> Option<String> {
        self.error.clone()
    }
}

/// Register an in-memory file (text) so `surface`/`import`/`include`/`use` can
/// read it — the browser has no filesystem, so height-fields, meshes, etc. must
/// be supplied this way before calling `scad_geometry`.
#[cfg(feature = "openscad")]
#[wasm_bindgen]
pub fn scad_register_file(name: &str, contents: &str) {
    crate::register_file(name, contents.as_bytes().to_vec());
}

/// Register an in-memory file from raw bytes (e.g. a binary STL to `import`).
#[cfg(feature = "openscad")]
#[wasm_bindgen]
pub fn scad_register_file_bytes(name: &str, bytes: &[u8]) {
    crate::register_file(name, bytes.to_vec());
}

/// Drop every registered in-memory file.
#[cfg(feature = "openscad")]
#[wasm_bindgen]
pub fn scad_clear_files() {
    crate::clear_files();
}

/// Parse OpenSCAD source and build its solid with the exact CSG kernel (float
/// fallback behind the never-wrong gate), returning a flat-shaded triangle soup.
#[cfg(feature = "openscad")]
#[wasm_bindgen]
pub fn scad_geometry(src: &str) -> ScadGeometry {
    let solid = match crate::parse_scad(src) {
        Ok(s) => s,
        Err(e) => {
            return ScadGeometry {
                positions: vec![],
                normals: vec![],
                triangles: 0,
                error: Some(e),
            }
        }
    };
    let (positions, normals) = flat_soup(&solid.to_geometry_exact());
    let tris = (positions.len() / 9) as u32;
    ScadGeometry {
        positions,
        normals,
        triangles: tris,
        error: None,
    }
}

/// Parse OpenSCAD source and encode its solid in a mesh format for download.
/// `format` is one of `stl` / `obj` / `off` / `3mf` / `glb` / `fcstd`; returns
/// the encoded bytes (empty on parse error or unknown format).
#[cfg(feature = "openscad")]
#[wasm_bindgen(js_name = scadExport)]
pub fn scad_export(src: &str, format: &str) -> Vec<u8> {
    let solid = match crate::parse_scad(src) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let g = solid.to_geometry_exact();
    match format {
        "stl" => crate::geometry_to_stl(&g),
        "obj" => crate::geometry_to_obj(&g).into_bytes(),
        "off" => crate::geometry_to_off(&g).into_bytes(),
        "3mf" => crate::geometry_to_3mf(&g),
        "glb" => crate::geometry_to_glb(&g),
        "fcstd" => crate::geometry_to_fcstd(&g, "Model"),
        _ => Vec::new(),
    }
}

/// A geometry as a flat-shaded triangle soup: `(positions, per-face normals)`,
/// 9 floats per triangle each. Expands the index buffer when present.
#[cfg(feature = "openscad")]
pub(super) fn flat_soup(g: &crate::BufferGeometry) -> (Vec<f32>, Vec<f32>) {
    let verts: Vec<[f32; 3]> = match g.positions() {
        Some(it) => it.map(|v| [v.x, v.y, v.z]).collect(),
        None => Vec::new(),
    };
    let soup: Vec<[f32; 3]> = match &g.index {
        Some(idx) => idx.iter().map(|&i| verts[i as usize]).collect(),
        None => verts,
    };
    let (mut positions, mut normals) = (
        Vec::with_capacity(soup.len() * 3),
        Vec::with_capacity(soup.len() * 3),
    );
    for t in soup.as_chunks::<3>().0 {
        let (a, b, c) = (t[0], t[1], t[2]);
        let (u, v) = (
            [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
            [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
        );
        let mut n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 1e-20 {
            n = [n[0] / len, n[1] / len, n[2] / len];
        }
        for vtx in t {
            positions.extend_from_slice(vtx);
            normals.extend_from_slice(&n);
        }
    }
    (positions, normals)
}

// ---------------------------------------------------------------------------
// Parametric 3D-printer assembly (see examples/printer_assembly.rs). Returned to
// JS as a list of flat-shaded, coloured parts — one mesh each, no cross-part CSG.
// ---------------------------------------------------------------------------

/// A parametric printer as a list of coloured parts. `positions(i)`/`normals(i)`
/// are flat-shaded soups; `color(i)` is an `[r,g,b]`.
#[cfg(feature = "openscad")]
#[wasm_bindgen]
pub struct PrinterScene {
    pub(super) parts: Vec<(Vec<f32>, Vec<f32>, [f32; 3])>,
}

#[cfg(feature = "openscad")]
#[wasm_bindgen]
impl PrinterScene {
    #[wasm_bindgen(getter)]
    pub fn count(&self) -> usize {
        self.parts.len()
    }
    pub fn positions(&self, i: usize) -> Vec<f32> {
        self.parts.get(i).map(|p| p.0.clone()).unwrap_or_default()
    }
    pub fn normals(&self, i: usize) -> Vec<f32> {
        self.parts.get(i).map(|p| p.1.clone()).unwrap_or_default()
    }
    pub fn color(&self, i: usize) -> Vec<f32> {
        self.parts.get(i).map(|p| p.2.to_vec()).unwrap_or_default()
    }
}

/// Build the parametric printer (all lengths in mm). Mirrors the native
/// `examples/printer_assembly.rs`.
#[cfg(feature = "openscad")]
#[wasm_bindgen]
pub fn printer_scene(
    bed_x: f32,
    bed_y: f32,
    z_travel: f32,
    ext: f32,
    gantry_z: f32,
    carriage: f32,
    bed_pos: f32,
) -> PrinterScene {
    let parts = crate::build_printer(bed_x, bed_y, z_travel, ext, gantry_z, carriage, bed_pos)
        .into_iter()
        .map(|(g, c)| {
            let (p, n) = flat_soup(&g);
            (p, n, c)
        })
        .collect();
    PrinterScene { parts }
}

/// Build the detailed NEMA 17 motor (see `examples/nema17.rs`) as coloured parts.
#[cfg(feature = "openscad")]
#[wasm_bindgen]
pub fn nema17_scene() -> PrinterScene {
    let parts = crate::build_nema17()
        .into_iter()
        .map(|(g, c)| {
            let (p, n) = flat_soup(&g);
            (p, n, c)
        })
        .collect();
    PrinterScene { parts }
}

// ---------------------------------------------------------------------------
// Subtitles and captions
// ---------------------------------------------------------------------------

/// A timed-text track — the JS face of [`crate::captions::CaptionTrack`].
///
/// ```js
/// const track = CaptionTrack.parse(await (await fetch('dialogue.vtt')).text());
/// const overlay = new CaptionOverlay(track);
/// overlay.setAutoScale(true);
/// // ...then each frame:
/// renderer.renderWithCaptions(scene, camera, overlay, video.currentTime);
/// ```
#[cfg(feature = "captions")]
#[wasm_bindgen]
pub struct WebCaptionTrack {
    pub(super) inner: crate::captions::CaptionTrack,
}

#[cfg(feature = "captions")]
#[wasm_bindgen]
impl WebCaptionTrack {
    /// An empty track.
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebCaptionTrack {
        WebCaptionTrack {
            inner: crate::captions::CaptionTrack::new(),
        }
    }

    /// Parse SubRip or WebVTT, sniffing the `WEBVTT` magic.
    pub fn parse(text: &str) -> Result<WebCaptionTrack, JsValue> {
        crate::captions::CaptionTrack::parse(text)
            .map(|inner| WebCaptionTrack { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Parse SubRip (`.srt`).
    #[wasm_bindgen(js_name = parseSrt)]
    pub fn parse_srt(text: &str) -> Result<WebCaptionTrack, JsValue> {
        crate::captions::CaptionTrack::parse_srt(text)
            .map(|inner| WebCaptionTrack { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Parse WebVTT (`.vtt`) — what a `<track>` element takes.
    #[wasm_bindgen(js_name = parseVtt)]
    pub fn parse_vtt(text: &str) -> Result<WebCaptionTrack, JsValue> {
        crate::captions::CaptionTrack::parse_vtt(text)
            .map(|inner| WebCaptionTrack { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Append a cue, in seconds.
    #[wasm_bindgen(js_name = addCue)]
    pub fn add_cue(&mut self, start: f64, end: f64, text: &str) {
        self.inner.push(crate::captions::Cue::new(start, end, text));
    }

    /// BCP-47 language tag (`"en"`, `"pt-BR"`).
    #[wasm_bindgen(js_name = setLanguage)]
    pub fn set_language(&mut self, language: &str) {
        self.inner.language = language.to_string();
    }

    /// Human-readable name shown in player menus.
    #[wasm_bindgen(js_name = setLabel)]
    pub fn set_label(&mut self, label: &str) {
        self.inner.label = label.to_string();
    }

    /// Shift every cue by `seconds` (negative moves earlier).
    pub fn shift(&mut self, seconds: f64) {
        self.inner.shift(seconds);
    }

    /// Number of cues.
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.len()
    }

    /// End time of the last cue, in seconds.
    pub fn duration(&self) -> f64 {
        self.inner.duration()
    }

    /// Text showing at `time` seconds, markup stripped. Empty when nothing is.
    #[wasm_bindgen(js_name = textAt)]
    pub fn text_at(&self, time: f64) -> String {
        self.inner.text_at(time)
    }

    /// Serialize to SubRip.
    #[wasm_bindgen(js_name = toSrt)]
    pub fn to_srt(&self) -> String {
        self.inner.to_srt()
    }

    /// Serialize to WebVTT — feed this to a `<track>` via a Blob URL.
    #[wasm_bindgen(js_name = toVtt)]
    pub fn to_vtt(&self) -> String {
        self.inner.to_vtt()
    }
}

#[cfg(feature = "captions")]
impl Default for WebCaptionTrack {
    fn default() -> Self {
        Self::new()
    }
}

/// A frame-sized caption overlay — the JS face of
/// [`crate::captions::CaptionOverlay`].
///
/// Pass it to [`WebRenderer::render_with_captions`] to have the GPU blend it
/// over the 3D image, or read [`rgba`](Self::rgba) to composite it yourself
/// (e.g. into a 2D canvas `ImageData`).
#[cfg(feature = "captions")]
#[wasm_bindgen]
pub struct WebCaptionOverlay {
    pub(super) inner: crate::captions::CaptionOverlay,
}

#[cfg(feature = "captions")]
#[wasm_bindgen]
impl WebCaptionOverlay {
    /// An overlay for `track`, using the built-in bitmap face.
    #[wasm_bindgen(constructor)]
    pub fn new(track: &WebCaptionTrack) -> WebCaptionOverlay {
        WebCaptionOverlay {
            inner: crate::captions::CaptionOverlay::new(track.inner.clone()),
        }
    }

    /// Replace the cues.
    #[wasm_bindgen(js_name = setTrack)]
    pub fn set_track(&mut self, track: &WebCaptionTrack) {
        self.inner.set_track(track.inner.clone());
    }

    /// Use a TrueType font for the text (`.ttf` bytes, `glyf` outlines).
    #[wasm_bindgen(js_name = setFont)]
    pub fn set_font(&mut self, ttf: &[u8]) -> Result<(), JsValue> {
        let font = crate::captions::CaptionFont::from_ttf_bytes(ttf)
            .map_err(|e| JsValue::from_str(&format!("font could not be parsed: {e:?}")))?;
        self.inner.painter.font = font;
        self.inner.invalidate();
        Ok(())
    }

    /// Rescale the style with the frame height, treating the current values as
    /// authored for 1080p. Off by default.
    #[wasm_bindgen(js_name = setAutoScale)]
    pub fn set_auto_scale(&mut self, enabled: bool) {
        self.inner.set_auto_scale(enabled);
    }

    /// Type size in pixels.
    #[wasm_bindgen(js_name = setFontSize)]
    pub fn set_font_size(&mut self, px: f32) {
        self.restyle(|s| s.font_size(px));
    }

    /// Text fill color, 0–255 per channel.
    #[wasm_bindgen(js_name = setColor)]
    pub fn set_color(&mut self, r: u8, g: u8, b: u8, a: u8) {
        self.restyle(|s| s.color([r, g, b, a]));
    }

    /// Outline color and half-width in pixels. Width `0` removes it.
    #[wasm_bindgen(js_name = setOutline)]
    pub fn set_outline(&mut self, r: u8, g: u8, b: u8, a: u8, width: f32) {
        self.restyle(|s| s.outline([r, g, b, a], width));
    }

    /// Background box color. Alpha `0` removes the box.
    #[wasm_bindgen(js_name = setBackground)]
    pub fn set_background(&mut self, r: u8, g: u8, b: u8, a: u8) {
        self.restyle(|s| s.background([r, g, b, a]));
    }

    /// Drop shadow color and offset in pixels. Alpha `0` removes it.
    #[wasm_bindgen(js_name = setShadow)]
    pub fn set_shadow(&mut self, r: u8, g: u8, b: u8, a: u8, dx: f32, dy: f32) {
        self.restyle(|s| s.shadow([r, g, b, a], dx, dy));
    }

    /// Distance from the anchored frame edge, in pixels.
    #[wasm_bindgen(js_name = setMargin)]
    pub fn set_margin(&mut self, px: f32) {
        self.restyle(|s| s.margin(px));
    }

    /// Padding inside the background box, in pixels.
    #[wasm_bindgen(js_name = setPadding)]
    pub fn set_padding(&mut self, px: f32) {
        self.restyle(|s| s.padding(px));
    }

    /// Line advance as a multiple of the type size.
    #[wasm_bindgen(js_name = setLineHeight)]
    pub fn set_line_height(&mut self, factor: f32) {
        self.restyle(|s| s.line_height(factor));
    }

    /// Wrap width as a fraction (`0..=1`) of the frame width.
    #[wasm_bindgen(js_name = setMaxWidth)]
    pub fn set_max_width(&mut self, fraction: f32) {
        self.restyle(|s| s.max_width(fraction));
    }

    /// Horizontal alignment: `"left"`, `"center"`, or `"right"`.
    #[wasm_bindgen(js_name = setAlign)]
    pub fn set_align(&mut self, align: &str) {
        let align = match align {
            "left" | "start" => crate::captions::CaptionAlign::Left,
            "right" | "end" => crate::captions::CaptionAlign::Right,
            _ => crate::captions::CaptionAlign::Center,
        };
        self.restyle(|s| s.align(align));
    }

    /// Frame edge to anchor to: `"top"`, `"middle"`, or `"bottom"`.
    #[wasm_bindgen(js_name = setAnchor)]
    pub fn set_anchor(&mut self, anchor: &str) {
        let anchor = match anchor {
            "top" => crate::captions::CaptionAnchor::Top,
            "middle" | "center" => crate::captions::CaptionAnchor::Middle,
            _ => crate::captions::CaptionAnchor::Bottom,
        };
        self.restyle(|s| s.anchor(anchor));
    }

    /// Overlay resolution. [`WebRenderer::render_with_captions`] sets this for
    /// you; call it directly only when compositing by hand.
    #[wasm_bindgen(js_name = setSize)]
    pub fn set_size(&mut self, width: u32, height: u32) {
        self.inner.set_size(width, height);
    }

    /// Rasterize for `time` seconds if needed. Returns whether the buffer
    /// changed — the cue to skip re-uploading on unchanged frames.
    pub fn update(&mut self, time: f64) -> bool {
        self.inner.update(time)
    }

    /// Whether a cue is currently showing.
    #[wasm_bindgen(js_name = isVisible)]
    pub fn is_visible(&self) -> bool {
        self.inner.is_visible()
    }

    /// The RGBA8 overlay buffer, transparent where there is no caption.
    /// Copy it into an `ImageData` to composite in a 2D canvas.
    pub fn rgba(&self) -> Vec<u8> {
        self.inner.rgba().to_vec()
    }
}

#[cfg(feature = "captions")]
impl WebCaptionOverlay {
    /// Apply a style builder and invalidate the cached raster.
    pub(super) fn restyle(
        &mut self,
        f: impl FnOnce(crate::captions::CaptionStyle) -> crate::captions::CaptionStyle,
    ) {
        // Style off the authored values, not the auto-scaled ones, so
        // repeated setter calls do not compound the rescale.
        let style = f(self.inner.style().clone());
        self.inner.set_style(style);
    }
}

// ---------------------------------------------------------------------------
// NURBS (feature = "nurbs")
// ---------------------------------------------------------------------------
//
// The three.js parity types in `curves::nurbs` were never reachable from JS at
// all — no binding, no shim export. These are the f64 kernel types, which is
// what a caller actually wants on the web side too: exact circles, analytic
// normals, and a tessellator that answers to a tolerance instead of a segment
// count.
//
// Flat `Float64Array`s rather than arrays-of-arrays throughout: one copy across
// the wasm boundary instead of an allocation per control point.

#[cfg(feature = "nurbs")]
#[wasm_bindgen]
pub struct WebNurbsCurve {
    pub(super) inner: crate::nurbs::NurbsCurve,
}

#[cfg(feature = "nurbs")]
#[wasm_bindgen]
impl WebNurbsCurve {
    /// Build from Cartesian control points (flat `xyz`) and optional weights.
    #[wasm_bindgen(constructor)]
    pub fn new(
        degree: usize,
        knots: Vec<f64>,
        points: Vec<f64>,
        weights: Option<Vec<f64>>,
    ) -> Result<WebNurbsCurve, JsValue> {
        let pts: Vec<[f64; 3]> = points
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        crate::nurbs::NurbsCurve::new(degree, knots, &pts, weights.as_deref())
            .map(|inner| WebNurbsCurve { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Build from control points already in homogeneous form (flat `xyzw`) —
    /// the layout three.js's `NURBSCurve` and STEP both use.
    #[wasm_bindgen(js_name = fromHomogeneous)]
    pub fn from_homogeneous(
        degree: usize,
        knots: Vec<f64>,
        control: Vec<f64>,
    ) -> Result<WebNurbsCurve, JsValue> {
        let cw: Vec<[f64; 4]> = control.as_chunks::<4>().0.to_vec();
        crate::nurbs::NurbsCurve::from_homogeneous(degree, knots, cw)
            .map(|inner| WebNurbsCurve { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn degree(&self) -> usize {
        self.inner.degree()
    }

    pub fn knots(&self) -> Vec<f64> {
        self.inner.knots().to_vec()
    }

    #[wasm_bindgen(js_name = controlCount)]
    pub fn control_count(&self) -> usize {
        self.inner.n_control()
    }

    /// `[u_min, u_max]`.
    pub fn domain(&self) -> Vec<f64> {
        let (a, b) = self.inner.domain();
        vec![a, b]
    }

    /// Point at parameter `u` (clamped to the domain) as `[x, y, z]`.
    pub fn point(&self, u: f64) -> Vec<f64> {
        self.inner.point(u).to_vec()
    }

    /// Point at normalized `t ∈ [0, 1]`.
    #[wasm_bindgen(js_name = pointAt)]
    pub fn point_at(&self, t: f64) -> Vec<f64> {
        self.inner.point(self.inner.param_at(t)).to_vec()
    }

    /// Derivatives 0..=`k`, flattened: `[C, C', C'', …]`.
    pub fn derivatives(&self, u: f64, k: usize) -> Vec<f64> {
        self.inner.derivatives(u, k).into_iter().flatten().collect()
    }

    /// Unit tangent, or an empty array at a cusp where none exists.
    pub fn tangent(&self, u: f64) -> Vec<f64> {
        self.inner
            .tangent(u)
            .map(|t| t.to_vec())
            .unwrap_or_default()
    }

    /// Adaptive polyline honouring `tolerance`, flattened `xyz`.
    pub fn tessellate(&self, tolerance: f64) -> Vec<f64> {
        let opts = crate::nurbs::TessellationOptions::with_tolerance(tolerance);
        crate::nurbs::tessellate_curve(&self.inner, &opts)
            .into_iter()
            .flatten()
            .collect()
    }
}

#[cfg(feature = "nurbs")]
#[wasm_bindgen]
pub struct WebNurbsSurface {
    pub(super) inner: crate::nurbs::NurbsSurface,
}

#[cfg(feature = "nurbs")]
#[wasm_bindgen]
impl WebNurbsSurface {
    /// Build from a Cartesian control grid (flat `xyz`, **u-major**: index
    /// `i * n_v + j`) and optional weights.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        degree_u: usize,
        degree_v: usize,
        knots_u: Vec<f64>,
        knots_v: Vec<f64>,
        n_u: usize,
        n_v: usize,
        points: Vec<f64>,
        weights: Option<Vec<f64>>,
    ) -> Result<WebNurbsSurface, JsValue> {
        let pts: Vec<[f64; 3]> = points
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        crate::nurbs::NurbsSurface::new(
            degree_u,
            degree_v,
            knots_u,
            knots_v,
            n_u,
            n_v,
            &pts,
            weights.as_deref(),
        )
        .map(|inner| WebNurbsSurface { inner })
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = degreeU)]
    pub fn degree_u(&self) -> usize {
        self.inner.degree_u()
    }

    #[wasm_bindgen(js_name = degreeV)]
    pub fn degree_v(&self) -> usize {
        self.inner.degree_v()
    }

    /// `[u_min, u_max, v_min, v_max]`.
    pub fn domain(&self) -> Vec<f64> {
        let (u0, u1) = self.inner.domain_u();
        let (v0, v1) = self.inner.domain_v();
        vec![u0, u1, v0, v1]
    }

    /// Point at `(u, v)` as `[x, y, z]`.
    pub fn point(&self, u: f64, v: f64) -> Vec<f64> {
        self.inner.point(u, v).to_vec()
    }

    /// Point at normalized `(s, t) ∈ [0, 1]²`.
    #[wasm_bindgen(js_name = pointAt)]
    pub fn point_at(&self, s: f64, t: f64) -> Vec<f64> {
        let (u, v) = self.inner.param_at(s, t);
        self.inner.point(u, v).to_vec()
    }

    /// Analytic unit normal. Empty only if the surface is degenerate over a
    /// whole neighbourhood — poles are handled and do return a normal.
    pub fn normal(&self, u: f64, v: f64) -> Vec<f64> {
        self.inner
            .normal(u, v)
            .map(|n| n.to_vec())
            .unwrap_or_default()
    }

    /// Tessellate to a `BufferGeometry` at the given chord tolerance.
    pub fn tessellate(&self, tolerance: f64) -> WebBufferGeometry {
        WebBufferGeometry {
            inner: Arc::new(crate::geometries::NurbsGeometry::with_tolerance(
                &self.inner,
                tolerance,
            )),
        }
    }

    /// Tessellate on a fixed `u_segments × v_segments` grid.
    #[wasm_bindgen(js_name = tessellateGrid)]
    pub fn tessellate_grid(&self, u_segments: usize, v_segments: usize) -> WebBufferGeometry {
        WebBufferGeometry {
            inner: Arc::new(crate::geometries::NurbsGeometry::with_segments(
                &self.inner,
                u_segments,
                v_segments,
            )),
        }
    }

    /// Did the sampler actually reach `tolerance`, or did its sample ceiling
    /// bind first? A truncated grid otherwise looks identical to a converged one.
    #[wasm_bindgen(js_name = meetsTolerance)]
    pub fn meets_tolerance(&self, tolerance: f64) -> bool {
        let opts = crate::nurbs::TessellationOptions::with_tolerance(tolerance);
        let (pu, pv) = crate::nurbs::tessellate::sample_grid(&self.inner, &opts);
        crate::nurbs::tessellate::sample_grid_meets_tolerance(&self.inner, &pu, &pv, tolerance)
    }

    /// Sweep this surface's defining profile — see the free `nurbs*` builders
    /// for the constructors that produce exact quadrics.
    #[wasm_bindgen(js_name = isRational)]
    pub fn is_rational(&self) -> bool {
        self.inner.is_rational()
    }
}

/// An exact circle as a rational quadratic — not a polyline approximation.
#[cfg(feature = "nurbs")]
#[wasm_bindgen(js_name = nurbsCircle)]
pub fn nurbs_circle(cx: f64, cy: f64, cz: f64, radius: f64) -> WebNurbsCurve {
    WebNurbsCurve {
        inner: crate::nurbs::construct::circle(
            [cx, cy, cz],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            radius,
        ),
    }
}

/// An exact circular arc from `start` to `end` radians in the xy-plane.
#[cfg(feature = "nurbs")]
#[wasm_bindgen(js_name = nurbsArc)]
pub fn nurbs_arc(cx: f64, cy: f64, cz: f64, radius: f64, start: f64, end: f64) -> WebNurbsCurve {
    WebNurbsCurve {
        inner: crate::nurbs::construct::arc(
            [cx, cy, cz],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            radius,
            start,
            end,
        ),
    }
}

/// An exact sphere. Sampling it reproduces the radius to ~1e-15 at every
/// parameter, unlike a `SphereGeometry` at any segment count.
#[cfg(feature = "nurbs")]
#[wasm_bindgen(js_name = nurbsSphere)]
pub fn nurbs_sphere(cx: f64, cy: f64, cz: f64, radius: f64) -> WebNurbsSurface {
    WebNurbsSurface {
        inner: crate::nurbs::construct::sphere([cx, cy, cz], radius),
    }
}

/// An exact cylindrical side surface about `+Z` (no caps).
#[cfg(feature = "nurbs")]
#[wasm_bindgen(js_name = nurbsCylinder)]
pub fn nurbs_cylinder(cx: f64, cy: f64, cz: f64, radius: f64, height: f64) -> WebNurbsSurface {
    WebNurbsSurface {
        inner: crate::nurbs::construct::cylinder([cx, cy, cz], [0.0, 0.0, 1.0], radius, height),
    }
}

/// An exact conical side surface about `+Z` (no cap).
#[cfg(feature = "nurbs")]
#[wasm_bindgen(js_name = nurbsCone)]
pub fn nurbs_cone(
    apex_x: f64,
    apex_y: f64,
    apex_z: f64,
    base_radius: f64,
    height: f64,
) -> WebNurbsSurface {
    WebNurbsSurface {
        inner: crate::nurbs::construct::cone(
            [apex_x, apex_y, apex_z],
            [0.0, 0.0, 1.0],
            base_radius,
            height,
        ),
    }
}

/// An exact torus about `+Z`.
#[cfg(feature = "nurbs")]
#[wasm_bindgen(js_name = nurbsTorus)]
pub fn nurbs_torus(cx: f64, cy: f64, cz: f64, major: f64, minor: f64) -> WebNurbsSurface {
    WebNurbsSurface {
        inner: crate::nurbs::construct::torus([cx, cy, cz], [0.0, 0.0, 1.0], major, minor),
    }
}

/// Revolve a profile curve about an axis through `(px, py, pz)` in direction
/// `(dx, dy, dz)`, sweeping `angle` radians (0 means a full turn).
#[cfg(feature = "nurbs")]
#[wasm_bindgen(js_name = nurbsRevolve)]
#[allow(clippy::too_many_arguments)]
pub fn nurbs_revolve(
    profile: &WebNurbsCurve,
    px: f64,
    py: f64,
    pz: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    angle: f64,
) -> WebNurbsSurface {
    WebNurbsSurface {
        inner: crate::nurbs::construct::revolve(&profile.inner, [px, py, pz], [dx, dy, dz], angle),
    }
}

/// Linearly sweep a profile curve along `(dx, dy, dz)`.
#[cfg(feature = "nurbs")]
#[wasm_bindgen(js_name = nurbsExtrude)]
pub fn nurbs_extrude(profile: &WebNurbsCurve, dx: f64, dy: f64, dz: f64) -> WebNurbsSurface {
    WebNurbsSurface {
        inner: crate::nurbs::construct::extrude(&profile.inner, [dx, dy, dz]),
    }
}

/// The ruled surface between two curves, made compatible first (they may differ
/// in degree and knot vector).
#[cfg(feature = "nurbs")]
#[wasm_bindgen(js_name = nurbsRuled)]
pub fn nurbs_ruled(a: &WebNurbsCurve, b: &WebNurbsCurve) -> Result<WebNurbsSurface, JsValue> {
    crate::nurbs::construct::ruled(&a.inner, &b.inner)
        .map(|inner| WebNurbsSurface { inner })
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

// ── RLX bridge ──────────────────────────────────────────────────────────
//
// The browser half of `threers::rlx`. Everything the bridge does is pure data
// in and pure data out, so the bindings are plain typed arrays: JS hands over
// an `Uint8Array` of pixels or a `Float32Array` of positions and gets one
// back. `RLX=1 web/build.sh` turns them on; `RLX_GEO=1` adds the geometry set.
//
// Which devices a browser build finds is rlx's answer, not this crate's — ask
// `rlxDevices()` at startup rather than assuming.

/// The rlx backends this build found, comma-separated (e.g. `"WebGpu,Cpu"`).
/// Empty when rlx has none, in which case every call below will fail.
#[cfg(feature = "rlx")]
#[wasm_bindgen(js_name = rlxDevices)]
pub fn rlx_devices() -> String {
    ::rlx::available_devices()
        .iter()
        .map(|d| format!("{d:?}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// Convolve an RGBA8 frame with a named 3×3 kernel: `box`, `gaussian`,
/// `sharpen`, `laplacian`, `sobel-x`, `sobel-y` or `emboss`.
///
/// Alpha is carried through, and the edges are replicated rather than
/// zero-padded — see [`crate::rlx::ConvFilter`].
#[cfg(feature = "rlx")]
#[wasm_bindgen(js_name = rlxConvolve)]
pub fn rlx_convolve(
    rgba: &[u8],
    width: u32,
    height: u32,
    kernel: &str,
) -> Result<Vec<u8>, JsValue> {
    use crate::rlx::{preferred_device, ConvFilter, Kernel3x3};
    let kernel = match kernel {
        "box" => Kernel3x3::BOX_BLUR,
        "gaussian" => Kernel3x3::GAUSSIAN,
        "sharpen" => Kernel3x3::SHARPEN,
        "laplacian" => Kernel3x3::LAPLACIAN,
        "sobel-x" => Kernel3x3::SOBEL_X,
        "sobel-y" => Kernel3x3::SOBEL_Y,
        "emboss" => Kernel3x3::EMBOSS,
        other => return Err(JsValue::from_str(&format!("unknown kernel {other}"))),
    };
    ConvFilter::new(width, height, kernel, preferred_device())
        .apply(rgba)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Run `iterations` diffusion passes over a single-channel height grid.
#[cfg(feature = "rlx")]
#[wasm_bindgen(js_name = rlxDiffuse)]
pub fn rlx_diffuse(
    field: &[f32],
    width: u32,
    height: u32,
    rate: f32,
    iterations: u32,
) -> Result<Vec<f32>, JsValue> {
    use crate::rlx::{preferred_device, Diffusion};
    Diffusion::new(width, height, rate, preferred_device())
        .run(field, iterations as usize)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Smooth a mesh, returning new positions (`[x, y, z, …]`).
///
/// A negative `mu` runs Taubin's λ|μ filter, which keeps the volume; pass 0 for
/// plain Laplacian smoothing, which does not.
#[cfg(feature = "rlx")]
#[wasm_bindgen(js_name = rlxSmoothMesh)]
pub fn rlx_smooth_mesh(
    positions: &[f32],
    index: &[u32],
    iterations: u32,
    lambda: f32,
    mu: f32,
) -> Result<Vec<f32>, JsValue> {
    use crate::core::{BufferAttribute, BufferGeometry};
    use crate::rlx::{mesh, preferred_device};

    let mut geometry = BufferGeometry::new();
    geometry.set_attribute("position", BufferAttribute::new(positions.to_vec(), 3));
    geometry.set_index(index.to_vec());

    let device = preferred_device();
    let result = if mu < 0.0 {
        mesh::taubin_smooth(&mut geometry, iterations as usize, lambda, mu, device)
    } else {
        mesh::laplacian_smooth(&mut geometry, iterations as usize, lambda, device)
    };
    result.map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(geometry
        .get_attribute("position")
        .map(|a| a.array.clone())
        .unwrap_or_default())
}

/// Fit a colour grade carrying `source` onto `target`, both RGBA8 frames of the
/// same size. Returns twelve numbers: the 3×3 matrix row-major, then the bias.
#[cfg(feature = "rlx")]
#[wasm_bindgen(js_name = rlxFitGrade)]
pub fn rlx_fit_grade(source: &[u8], target: &[u8]) -> Result<Vec<f32>, JsValue> {
    use crate::rlx::{preferred_device, ColorGrade, FitOptions};
    let report = ColorGrade::fit(source, target, &FitOptions::default(), preferred_device())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut out = Vec::with_capacity(12);
    for row in report.grade.matrix {
        out.extend_from_slice(&row);
    }
    out.extend_from_slice(&report.grade.bias);
    Ok(out)
}

/// Apply a grade in the twelve-number form [`rlx_fit_grade`] returns.
#[cfg(feature = "rlx")]
#[wasm_bindgen(js_name = rlxApplyGrade)]
pub fn rlx_apply_grade(rgba: &[u8], grade: &[f32]) -> Result<Vec<u8>, JsValue> {
    use crate::rlx::ColorGrade;
    if grade.len() != 12 {
        return Err(JsValue::from_str("a grade is 9 matrix values then 3 bias"));
    }
    let grade = ColorGrade {
        matrix: [
            [grade[0], grade[1], grade[2]],
            [grade[3], grade[4], grade[5]],
            [grade[6], grade[7], grade[8]],
        ],
        bias: [grade[9], grade[10], grade[11]],
    };
    Ok(grade.apply(rgba))
}

/// Cluster a frame into `count` colours, returned as sRGB `[r, g, b, …]`
/// bytes. May be shorter than asked for — see [`crate::rlx::Palette::extract`].
#[cfg(feature = "rlx")]
#[wasm_bindgen(js_name = rlxPalette)]
pub fn rlx_palette(rgba: &[u8], count: u32) -> Result<Vec<u8>, JsValue> {
    use crate::rlx::{preferred_device, Palette, PaletteOptions};
    let palette = Palette::extract(
        rgba,
        count as usize,
        &PaletteOptions::default(),
        preferred_device(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(palette
        .colors
        .iter()
        .flat_map(|c| {
            let e = |v: f32| {
                let s = if v <= 0.0031308 {
                    v * 12.92
                } else {
                    1.055 * v.powf(1.0 / 2.4) - 0.055
                };
                (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
            };
            [e(c.r), e(c.g), e(c.b)]
        })
        .collect())
}

/// Snap every pixel of a frame to its nearest entry of a `count`-colour
/// palette extracted from that same frame.
#[cfg(feature = "rlx")]
#[wasm_bindgen(js_name = rlxPosterize)]
pub fn rlx_posterize(rgba: &[u8], count: u32) -> Result<Vec<u8>, JsValue> {
    use crate::rlx::{preferred_device, Palette, PaletteOptions};
    let device = preferred_device();
    let palette = Palette::extract(rgba, count as usize, &PaletteOptions::default(), device)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    palette
        .posterize(rgba, device)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// A triangulated height field, ready to hand to `BufferGeometry`.
#[cfg(feature = "rlx-geo")]
#[wasm_bindgen]
pub struct GeoGeometry {
    pub(super) positions: Vec<f32>,
    pub(super) normals: Vec<f32>,
    pub(super) uvs: Vec<f32>,
    pub(super) index: Vec<u32>,
}

#[cfg(feature = "rlx-geo")]
#[wasm_bindgen]
impl GeoGeometry {
    #[wasm_bindgen(getter)]
    pub fn positions(&self) -> Vec<f32> {
        self.positions.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn normals(&self) -> Vec<f32> {
        self.normals.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn uvs(&self) -> Vec<f32> {
        self.uvs.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn index(&self) -> Vec<u32> {
        self.index.clone()
    }
}

/// Exact Delaunay triangulation of `[x, y, …]` pairs, as triangle indices.
#[cfg(feature = "rlx-geo")]
#[wasm_bindgen(js_name = geoDelaunay)]
pub fn geo_delaunay(points: &[f32]) -> Result<Vec<u32>, JsValue> {
    use crate::math::Vector2;
    let points: Vec<Vector2> = points
        .chunks_exact(2)
        .map(|p| Vector2::new(p[0], p[1]))
        .collect();
    crate::rlx::geo::delaunay_indices(&points).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Scattered `[x, y, z, …]` samples triangulated in xz, with y as the height.
#[cfg(feature = "rlx-geo")]
#[wasm_bindgen(js_name = geoHeightfield)]
pub fn geo_heightfield(points: &[f32]) -> Result<GeoGeometry, JsValue> {
    use crate::math::Vector3;
    let points: Vec<Vector3> = points
        .chunks_exact(3)
        .map(|p| Vector3::new(p[0], p[1], p[2]))
        .collect();
    let geometry = crate::rlx::geo::heightfield_geometry(&points)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let attribute = |name: &str| {
        geometry
            .get_attribute(name)
            .map(|a| a.array.clone())
            .unwrap_or_default()
    };
    Ok(GeoGeometry {
        positions: attribute("position"),
        normals: attribute("normal"),
        uvs: attribute("uv"),
        index: geometry.index.clone().unwrap_or_default(),
    })
}

/// Which site owns each pixel, for sites given as `[x, y, …]` in pixels.
#[cfg(feature = "rlx-geo")]
#[wasm_bindgen(js_name = geoVoronoiLabels)]
pub fn geo_voronoi_labels(sites: &[f32], width: u32, height: u32) -> Vec<u32> {
    use crate::math::Vector2;
    let sites: Vec<Vector2> = sites
        .chunks_exact(2)
        .map(|s| Vector2::new(s[0], s[1]))
        .collect();
    crate::rlx::geo::voronoi_labels(&sites, width, height)
}

/// Distance from every pixel to the nearest cell wall — the field cell
/// textures want. Cost is pixels × sites; keep the site count in the hundreds.
#[cfg(feature = "rlx-geo")]
#[wasm_bindgen(js_name = geoVoronoiWallDistance)]
pub fn geo_voronoi_wall_distance(sites: &[f32], width: u32, height: u32) -> Vec<f32> {
    use crate::math::Vector2;
    let sites: Vec<Vector2> = sites
        .chunks_exact(2)
        .map(|s| Vector2::new(s[0], s[1]))
        .collect();
    crate::rlx::geo::voronoi_wall_distance(&sites, width, height)
}

/// A height grid as a tangent-space normal map, RGBA8 and linear.
#[cfg(feature = "rlx-geo")]
#[wasm_bindgen(js_name = geoNormalMap)]
pub fn geo_normal_map(field: &[f32], width: u32, height: u32, strength: f32) -> Vec<u8> {
    let texture = crate::rlx::geo::normal_map_from_height(field, width, height, strength);
    texture.data.as_ref().clone()
}

// ---------------------------------------------------------------------------
// IK + geared servo plant (planar 3R) — bodies, joints, contacts, materials.
// ---------------------------------------------------------------------------

/// One planar 3R arm with geometric IK, actuator plant, capsule solids,
/// joint reports, and world contacts (floor + obstacle).
///
/// `kind`: `"direct"` | `"qdd"` | `"high"` | `"hydraulic"` | `"tendon"`.
/// Lengths are millimetres.
#[wasm_bindgen]
pub struct WebIkServoPlant {
    pub(super) world: crate::kinematics::ArmWorld,
    pub(super) ik: crate::kinematics::Ik,
    pub(super) q_ik: Vec<f64>,
    pub(super) label: String,
}

pub(super) fn ik_for_chain(chain: &crate::kinematics::SerialChain) -> crate::kinematics::Ik {
    crate::kinematics::Ik {
        limits: (-179.0, 179.0),
        max_iters: 120,
        joint_limits: chain.joints.iter().map(|j| j.limits).collect(),
        ..Default::default()
    }
}

pub(super) fn drive_for_kind(kind: &str) -> Result<crate::kinematics::Drive, JsValue> {
    use crate::kinematics::Drive;
    Ok(match kind {
        "direct" | "dd" => Drive::direct(),
        "qdd" | "qdd-15:1" => Drive::qdd(),
        "high" | "servo" | "servo-288:1" => Drive::high_ratio(),
        "hydraulic" | "hyd" => Drive::hydraulic(),
        "tendon" | "cable" | "string" => Drive::tendon(),
        other => {
            return Err(JsValue::from_str(&format!(
                "unknown drive '{other}' (use direct|qdd|high|hydraulic|tendon)"
            )))
        }
    })
}

#[wasm_bindgen]
impl WebIkServoPlant {
    #[wasm_bindgen(constructor)]
    pub fn new(kind: &str) -> Result<WebIkServoPlant, JsValue> {
        let chain = crate::kinematics::SerialChain::planar_3r();
        let drive = drive_for_kind(kind)?;
        let label = drive.label().to_string();
        let ik = ik_for_chain(&chain);
        let mut plant = crate::kinematics::ArmPlant::from_drive(chain, drive);
        let seed = vec![-2.8, 88.9, 93.9];
        plant.seed(&seed);
        let mut world = crate::kinematics::ArmWorld::from_plant(plant);
        world.last_cmd = seed.clone();
        world.update_contacts();
        Ok(Self {
            world,
            ik,
            q_ik: seed,
            label,
        })
    }

    pub fn seed(&mut self, q_deg: &[f64]) -> Result<(), JsValue> {
        if q_deg.len() != self.world.plant.servos.len() {
            return Err(JsValue::from_str("seed length must match joint count"));
        }
        self.world.plant.seed(q_deg);
        self.q_ik = q_deg.to_vec();
        self.world.last_cmd = q_deg.to_vec();
        self.world.update_contacts();
        Ok(())
    }

    /// IK + plant step. Returns tip position error in millimetres.
    pub fn step(&mut self, tx: f64, ty: f64, tz: f64, ax: f64, ay: f64, az: f64, dt: f64) -> f64 {
        use crate::kinematics::Goal;
        let goal = Goal::PointAlong {
            at: [tx, ty, tz],
            along: [ax, ay, az],
        };
        let sol = self
            .ik
            .solve_chain(&self.q_ik, &goal, &self.world.plant.chain);
        self.q_ik = sol.joints.clone();
        self.world.last_cmd = sol.joints.clone();
        self.world.plant.step(&sol.joints, dt.max(1e-4));
        self.world.update_contacts();
        let q_act = self.world.plant.q_deg();
        let (p, _) = self.world.plant.chain.tool(&q_act);
        let dx = p[0] - tx;
        let dy = p[1] - ty;
        let dz = p[2] - tz;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    #[wasm_bindgen(getter)]
    pub fn label(&self) -> String {
        self.label.clone()
    }

    #[wasm_bindgen(getter, js_name = jointCount)]
    pub fn joint_count(&self) -> usize {
        self.world.plant.servos.len()
    }

    #[wasm_bindgen(getter, js_name = bodyCount)]
    pub fn body_count(&self) -> usize {
        self.world.plant.chain.n()
    }

    #[wasm_bindgen(getter, js_name = contactCount)]
    pub fn contact_count(&self) -> usize {
        self.world.contacts.len()
    }

    #[wasm_bindgen(js_name = qAct)]
    pub fn q_act(&self) -> Vec<f64> {
        self.world.plant.q_deg()
    }

    #[wasm_bindgen(js_name = qCmd)]
    pub fn q_cmd(&self) -> Vec<f64> {
        self.world.last_cmd.clone()
    }

    #[wasm_bindgen(js_name = skeletonAct)]
    pub fn skeleton_act(&self) -> Vec<f64> {
        flat_skeleton(&self.world.plant.chain, &self.world.plant.q_deg())
    }

    #[wasm_bindgen(js_name = skeletonCmd)]
    pub fn skeleton_cmd(&self) -> Vec<f64> {
        flat_skeleton(&self.world.plant.chain, &self.world.last_cmd)
    }

    #[wasm_bindgen(js_name = hingesAct)]
    pub fn hinges_act(&self) -> Vec<f64> {
        flat_hinges(&self.world.plant.chain, &self.world.plant.q_deg())
    }

    #[wasm_bindgen(js_name = tipAct)]
    pub fn tip_act(&self) -> Vec<f64> {
        let (p, _) = self.world.plant.chain.tool(&self.world.plant.q_deg());
        p.to_vec()
    }

    #[wasm_bindgen(js_name = tipCmd)]
    pub fn tip_cmd(&self) -> Vec<f64> {
        let (p, _) = self.world.plant.chain.tool(&self.world.last_cmd);
        p.to_vec()
    }

    #[wasm_bindgen(js_name = meanJointErr)]
    pub fn mean_joint_err(&self) -> f64 {
        let n = self.world.plant.servos.len().max(1) as f64;
        self.world
            .plant
            .servos
            .iter()
            .map(|s| s.tracking_error_deg())
            .sum::<f64>()
            / n
    }

    #[wasm_bindgen(js_name = reflectedInertia)]
    pub fn reflected_inertia(&self) -> f64 {
        self.world.plant.servos[0].drive.reflected_inertia()
    }

    /// Gear reduction N, or `1` for hydraulic / tendon.
    pub fn ratio(&self) -> f64 {
        match &self.world.plant.servos[0].drive {
            crate::kinematics::Drive::Gear(g) => g.ratio,
            _ => 1.0,
        }
    }

    #[wasm_bindgen(js_name = driveKind)]
    pub fn drive_kind(&self) -> String {
        self.world.plant.servos[0].drive.label().to_string()
    }

    /// Working fluid / cable / grease name.
    #[wasm_bindgen(js_name = materialName)]
    pub fn material_name(&self) -> String {
        self.world.plant.servos[0].drive.material_name().to_string()
    }

    #[wasm_bindgen(js_name = tempC)]
    pub fn temp_c(&self) -> f64 {
        self.world.plant.temp_c()
    }

    /// Operating temperature in °C (oil, grease, cable).
    #[wasm_bindgen(js_name = setTemp)]
    pub fn set_temp(&mut self, temp_c: f64) {
        self.world.plant.set_temp(temp_c.clamp(-20.0, 90.0));
    }

    /// Swap hydraulic fluid: `iso32|iso46|iso68|water-glycol|silicone`.
    #[wasm_bindgen(js_name = setFluid)]
    pub fn set_fluid(&mut self, name: &str) -> Result<(), JsValue> {
        use crate::kinematics::{Drive, HydraulicFluid};
        let fluid = HydraulicFluid::from_name(name).ok_or_else(|| {
            JsValue::from_str("unknown fluid (iso32|iso46|iso68|water-glycol|silicone)")
        })?;
        let temp = self.world.plant.temp_c();
        for s in &mut self.world.plant.servos {
            if matches!(s.drive, Drive::Hydraulic(_)) {
                let mut h = crate::kinematics::HydraulicDrive::with_fluid(fluid);
                h.env.temp_c = temp;
                s.drive = Drive::Hydraulic(h);
                s.tau_max = s.drive.torque_limit();
            }
        }
        Ok(())
    }

    /// Swap tendon material: `steel|uhmwpe|nylon|aramid`.
    #[wasm_bindgen(js_name = setTendon)]
    pub fn set_tendon(&mut self, name: &str) -> Result<(), JsValue> {
        use crate::kinematics::{Drive, TendonMaterial};
        let mat = TendonMaterial::from_name(name)
            .ok_or_else(|| JsValue::from_str("unknown tendon (steel|uhmwpe|nylon|aramid)"))?;
        let temp = self.world.plant.temp_c();
        for s in &mut self.world.plant.servos {
            if matches!(s.drive, Drive::Tendon(_)) {
                let mut t = crate::kinematics::TendonDrive::with_material(mat);
                t.env.temp_c = temp;
                s.drive = Drive::Tendon(t);
                s.tau_max = s.drive.torque_limit();
            }
        }
        Ok(())
    }

    /// Live agonist/antagonist tensions for joint 0, N: `[T+, T−, pretension]`.
    #[wasm_bindgen(js_name = tensions0)]
    pub fn tensions0(&self) -> Vec<f64> {
        match &self.world.plant.servos[0].drive {
            crate::kinematics::Drive::Tendon(t) => {
                vec![t.tension_plus, t.tension_minus, t.live_pretension()]
            }
            _ => vec![],
        }
    }

    /// Fluid kinematic viscosity at current temp, cSt (hydraulic only).
    #[wasm_bindgen(js_name = fluidNuCst)]
    pub fn fluid_nu_cst(&self) -> f64 {
        match &self.world.plant.servos[0].drive {
            crate::kinematics::Drive::Hydraulic(h) => h.fluid.nu_cst(h.env.temp_c),
            _ => 0.0,
        }
    }

    /// Mechanical design label (e.g. "harmonic + planetary 288:1").
    #[wasm_bindgen(js_name = designLabel)]
    pub fn design_label(&self) -> String {
        self.world.design_label()
    }

    /// Assembled mechanism parts in world mm.
    /// Stride 16: `[joint, role, ox,oy,oz, ax,ay,az, radius, length, r,g,b, metal, rough, shape]`.
    /// `role`: motor=0 … encoder=14, housing=15.
    /// `shape`: capsule=0, segment=1, disk=2.
    #[wasm_bindgen(js_name = designPartsFlat)]
    pub fn design_parts_flat(&self) -> Vec<f64> {
        crate::kinematics::parts_flat(&self.world.design_parts())
    }

    #[wasm_bindgen(js_name = designPartCount)]
    pub fn design_part_count(&self) -> usize {
        self.world.design_parts().len()
    }

    /// Flat link solids: per body
    /// `[ox,oy,oz, dx,dy,dz, radius, mass, volume, mat_id, rgb_r,rgb_g,rgb_b, metal, rough]`.
    /// `mat_id`: 0 aluminum, 1 steel, 2 plastic, 3 rubber.
    #[wasm_bindgen(js_name = bodiesFlat)]
    pub fn bodies_flat(&self) -> Vec<f64> {
        let bodies = self.world.bodies();
        let mut out = Vec::with_capacity(bodies.len() * 15);
        for b in bodies {
            let mat_id = match b.material {
                crate::kinematics::LinkMaterial::Aluminum => 0.0,
                crate::kinematics::LinkMaterial::Steel => 1.0,
                crate::kinematics::LinkMaterial::Plastic => 2.0,
                crate::kinematics::LinkMaterial::Rubber => 3.0,
            };
            let rgb = b.material.rgb();
            out.extend_from_slice(&b.origin);
            out.extend_from_slice(&b.distal);
            out.push(b.radius);
            out.push(b.mass);
            out.push(b.volume);
            out.push(mat_id);
            out.push(rgb[0] as f64);
            out.push(rgb[1] as f64);
            out.push(rgb[2] as f64);
            out.push(b.material.metalness() as f64);
            out.push(b.material.roughness() as f64);
        }
        out
    }

    /// Body names, newline-separated.
    #[wasm_bindgen(js_name = bodyNames)]
    pub fn body_names(&self) -> String {
        self.world
            .bodies()
            .into_iter()
            .map(|b| b.name)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Flat joints: per hinge
    /// `[ox,oy,oz, ax,ay,az, angle, cmd, omega, lo, hi, torque, engaged]`.
    #[wasm_bindgen(js_name = jointsFlat)]
    pub fn joints_flat(&self) -> Vec<f64> {
        let joints = self.world.joints();
        let mut out = Vec::with_capacity(joints.len() * 13);
        for j in joints {
            out.extend_from_slice(&j.origin);
            out.extend_from_slice(&j.axis);
            out.push(j.angle_deg);
            out.push(j.cmd_deg);
            out.push(j.omega);
            out.push(j.limits.0);
            out.push(j.limits.1);
            out.push(j.torque);
            out.push(if j.engaged { 1.0 } else { 0.0 });
        }
        out
    }

    /// Joint names, newline-separated.
    #[wasm_bindgen(js_name = jointNames)]
    pub fn joint_names(&self) -> String {
        self.world
            .joints()
            .into_iter()
            .map(|j| j.name)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Flat contacts: `[px,py,pz, nx,ny,nz, depth, …]` plus parallel name pairs
    /// via [`Self::contact_pairs`].
    #[wasm_bindgen(js_name = contactsFlat)]
    pub fn contacts_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.world.contacts.len() * 7);
        for c in &self.world.contacts {
            out.extend_from_slice(&c.point);
            out.extend_from_slice(&c.normal);
            out.push(c.depth);
        }
        out
    }

    /// `"a|b"` pairs, newline-separated, matching [`Self::contacts_flat`] order.
    #[wasm_bindgen(js_name = contactPairs)]
    pub fn contact_pairs(&self) -> String {
        self.world
            .contacts
            .iter()
            .map(|c| format!("{}|{}", c.a, c.b))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Obstacle AABB `[xmin,ymin,zmin, xmax,ymax,zmax]` and floor z as 7th value.
    #[wasm_bindgen(js_name = worldBounds)]
    pub fn world_bounds(&self) -> Vec<f64> {
        let o = &self.world.obstacles;
        let (mn, mx) = match (o.box_min, o.box_max) {
            (Some(a), Some(b)) => (a, b),
            _ => ([0.0; 3], [0.0; 3]),
        };
        vec![mn[0], mn[1], mn[2], mx[0], mx[1], mx[2], o.floor_z]
    }
}

pub(super) fn flat_skeleton(chain: &crate::kinematics::SerialChain, q: &[f64]) -> Vec<f64> {
    let sk = chain.skeleton(q);
    let mut out = Vec::with_capacity(sk.len() * 3);
    for p in sk {
        out.extend_from_slice(&p);
    }
    out
}

pub(super) fn flat_hinges(chain: &crate::kinematics::SerialChain, q: &[f64]) -> Vec<f64> {
    let poses = chain.poses(q);
    let mut out = Vec::with_capacity(poses.len() * 6);
    for p in poses {
        out.extend_from_slice(&p.origin);
        out.extend_from_slice(&p.axis);
    }
    out
}

// ---------------------------------------------------------------------------
// QuadView — CAD-style top / front / side / perspective panes on one canvas.
// ---------------------------------------------------------------------------

pub(super) fn pane_from_name(name: &str) -> Result<crate::helpers::Pane, JsValue> {
    crate::helpers::Pane::from_name(name).ok_or_else(|| {
        JsValue::from_str(&format!(
            "unknown pane '{name}' (use top|front|side|perspective)"
        ))
    })
}

/// A draw into the WebGL2 back buffer (`WebRenderer::gl_back_buffer`). With
/// MSAA on it goes through the multisampled path, as a draw to the canvas
/// would — a pane is resolved through its own pane-sized target there, so the
/// back buffer's other panes survive. `render_to` would be single-sampled.
/// Whether anything visible in `scene` blends with what is behind it: a
/// transparent material on a mesh (of any kind) or a sprite.
pub(super) fn scene_blends(scene: &crate::Scene) -> bool {
    use crate::core::ObjectKind;
    let mut any = false;
    scene.arena.traverse_visible(scene.root, &mut |_, obj| {
        any = any
            || match &obj.kind {
                ObjectKind::Mesh(m) => m.draws().iter().any(|d| d.material.transparent()),
                ObjectKind::InstancedMesh(m) => m.draws().iter().any(|d| d.material.transparent()),
                ObjectKind::SkinnedMesh(m) => m.draws().iter().any(|d| d.material.transparent()),
                ObjectKind::Sprite(_) => true,
                _ => false,
            };
    });
    any
}

pub(super) fn render_back(
    renderer: &mut crate::Renderer,
    scene: &mut crate::Scene,
    camera: &dyn crate::cameras::Camera,
    back: &crate::renderer::RenderTarget,
) {
    if renderer.msaa() > 1 {
        renderer.render_to_multisampled(scene, camera, back, false);
    } else {
        renderer.render_to(scene, camera, back);
    }
}

/// Where a [`WebQuadView`] draws this frame: the canvas texture, acquired
/// once for all four panes, the renderer's active render target, or the
/// WebGL2 back buffer standing in for the canvas.
pub(super) enum WebPaneTarget<'a> {
    View(&'a wgpu::TextureView),
    Target(&'a crate::renderer::RenderTarget),
    Back(&'a crate::renderer::RenderTarget),
}

pub(super) struct WebPaneSurface<'a> {
    pub(super) renderer: &'a mut crate::Renderer,
    pub(super) target: WebPaneTarget<'a>,
}

impl crate::helpers::PaneSurface for WebPaneSurface<'_> {
    fn renderer(&mut self) -> &mut dyn crate::helpers::ViewportControl {
        self.renderer
    }
    fn draw(&mut self, scene: &mut crate::Scene, camera: &dyn crate::cameras::Camera) {
        match self.target {
            WebPaneTarget::View(v) => self.renderer.render(scene, camera, v, false),
            WebPaneTarget::Target(t) => self.renderer.render_to(scene, camera, t),
            WebPaneTarget::Back(t) => render_back(self.renderer, scene, camera, t),
        }
    }
    fn clear(&mut self, color: crate::Color, alpha: f32) {
        match self.target {
            WebPaneTarget::View(v) => self.renderer.clear(v, false, color, alpha),
            WebPaneTarget::Target(t) | WebPaneTarget::Back(t) => {
                self.renderer.clear_render_target(t, color, alpha)
            }
        }
    }
}

/// [`crate::helpers::QuadView`] for the browser: three orthographic panes
/// (top, front, side — Y-up, third-angle) and the caller's perspective camera,
/// laid out on one canvas.
///
/// Rectangles are three.js's: bottom-left origin, logical (CSS) pixels — flip
/// a DOM mouse `y` (`height - y`) before `paneAt`. Pane names are
/// `"top" | "front" | "side" | "perspective"`.
#[wasm_bindgen]
pub struct WebQuadView {
    pub(super) inner: crate::helpers::QuadView,
    /// Pointer handling, built from the perspective camera on first use.
    pub(super) controls: Option<crate::helpers::QuadViewControls>,
}

/// The perspective camera inside a `WebCamera`, or a JS error naming `what`.
pub(super) fn quad_persp<'a>(
    camera: &'a mut WebCamera,
    what: &str,
) -> Result<&'a mut crate::cameras::PerspectiveCamera, JsValue> {
    match &mut camera.inner {
        CameraInner::Perspective(c) => Ok(c),
        _ => Err(JsValue::from_str(&format!(
            "QuadView.{what}: needs the perspective camera"
        ))),
    }
}

#[wasm_bindgen]
impl WebQuadView {
    /// Third-angle layout, 2 px gap, framed on the unit cube round the origin.
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebQuadView {
        WebQuadView {
            inner: crate::helpers::QuadView::new(),
            controls: None,
        }
    }

    /// Magnification of the orthographic panes over the fitted scale (one
    /// value for all three). `frameBox`/`frameScene`/`frameAll` reset it to 1.
    #[wasm_bindgen(getter)]
    pub fn zoom(&self) -> f32 {
        self.inner.zoom
    }
    #[wasm_bindgen(setter)]
    pub fn set_zoom(&mut self, zoom: f32) {
        if zoom.is_finite() && zoom > 0.0 {
            self.inner.zoom = zoom;
        }
    }

    /// Move what the orthographic panes are centred on, world units.
    #[wasm_bindgen(js_name = panBy)]
    pub fn pan_by(&mut self, x: f32, y: f32, z: f32) {
        self.inner.pan_by(crate::Vector3::new(x, y, z));
    }

    /// Draw the shadow maps once per `render` instead of once per pane
    /// (default on; the pictures are identical).
    #[wasm_bindgen(getter, js_name = shareShadows)]
    pub fn share_shadows(&self) -> bool {
        self.inner.share_shadows
    }
    #[wasm_bindgen(setter, js_name = shareShadows)]
    pub fn set_share_shadows(&mut self, on: bool) {
        self.inner.share_shadows = on;
    }

    /// Start a pointer gesture at `(x, y)` — canvas CSS pixels, **top-left**
    /// origin (DOM `offsetX`/`offsetY`) — on a `width`×`height` canvas.
    /// `button` is DOM `PointerEvent.button`. Returns the pane, or `undefined`
    /// over a gap. Primary drag in the perspective pane orbits `camera`; any
    /// other drag pans.
    #[wasm_bindgen(js_name = pointerDown)]
    pub fn pointer_down(
        &mut self,
        camera: &mut WebCamera,
        x: f32,
        y: f32,
        button: i32,
        width: f32,
        height: f32,
    ) -> Result<Option<String>, JsValue> {
        let cam = quad_persp(camera, "pointerDown")?.clone();
        let Some(button) = crate::helpers::PointerButton::from_dom(button) else {
            return Ok(None);
        };
        let mut ctl = self.controls.take();
        let c = ctl.get_or_insert_with(|| crate::helpers::QuadViewControls::new(&cam));
        c.set_size(width, height);
        let pane = c.pointer_down(&self.inner, &cam, x, y, button);
        self.controls = ctl;
        Ok(pane.map(|p| p.name().to_string()))
    }

    /// Continue the gesture. Returns whether anything moved.
    #[wasm_bindgen(js_name = pointerMove)]
    pub fn pointer_move(
        &mut self,
        camera: &mut WebCamera,
        x: f32,
        y: f32,
    ) -> Result<bool, JsValue> {
        let cam = quad_persp(camera, "pointerMove")?;
        let Some(c) = self.controls.as_mut() else {
            return Ok(false);
        };
        Ok(c.pointer_move(&mut self.inner, cam, x, y))
    }

    #[wasm_bindgen(js_name = pointerUp)]
    pub fn pointer_up(&mut self) {
        if let Some(c) = self.controls.as_mut() {
            c.pointer_up();
        }
    }

    /// Zoom at `(x, y)` (top-left origin): `delta` is DOM `WheelEvent.deltaY`,
    /// positive zooms out. One zoom for the three orthographic panes; a dolly
    /// of `camera` in the perspective pane. Returns whether anything moved.
    pub fn wheel(
        &mut self,
        camera: &mut WebCamera,
        x: f32,
        y: f32,
        delta: f32,
        width: f32,
        height: f32,
    ) -> Result<bool, JsValue> {
        let cam = quad_persp(camera, "wheel")?;
        let snapshot = cam.clone();
        let mut ctl = self.controls.take();
        let c = ctl.get_or_insert_with(|| crate::helpers::QuadViewControls::new(&snapshot));
        c.set_size(width, height);
        let moved = c.wheel(&mut self.inner, cam, x, y, delta);
        self.controls = ctl;
        Ok(moved)
    }

    /// Maximise the pane under `(x, y)` (top-left origin), or restore the
    /// four. Returns the pane, `undefined` over a gap.
    pub fn dblclick(&mut self, x: f32, y: f32, width: f32, height: f32) -> Option<String> {
        let pane = self.inner.pane_at(x, height - y, width, height)?;
        self.inner.toggle_maximize(pane);
        if let Some(c) = self.controls.as_mut() {
            c.pointer_up();
        }
        Some(pane.name().to_string())
    }

    /// Frame every visible mesh in every pane: the orthographic panes at
    /// zoom 1, and `camera` moved along its line of sight to fit. False when
    /// there is nothing to frame.
    #[wasm_bindgen(js_name = frameAll)]
    pub fn frame_all(
        &mut self,
        scene: &mut WebScene,
        camera: &mut WebCamera,
    ) -> Result<bool, JsValue> {
        let cam = quad_persp(camera, "frameAll")?;
        let snapshot = cam.clone();
        let mut ctl = self
            .controls
            .take()
            .unwrap_or_else(|| crate::helpers::QuadViewControls::new(&snapshot));
        let framed = ctl
            .frame_all(&mut self.inner, cam, &mut scene.inner)
            .is_some();
        self.controls = Some(ctl);
        Ok(framed)
    }

    /// `frameAll` for a box: frame `bounds` in every pane.
    #[wasm_bindgen(js_name = frameBoxAll)]
    pub fn frame_box_all(
        &mut self,
        bounds: &WebBox3,
        camera: &mut WebCamera,
    ) -> Result<(), JsValue> {
        let cam = quad_persp(camera, "frameBoxAll")?;
        let snapshot = cam.clone();
        let mut ctl = self
            .controls
            .take()
            .unwrap_or_else(|| crate::helpers::QuadViewControls::new(&snapshot));
        ctl.frame_box(&mut self.inner, cam, &bounds.inner);
        self.controls = Some(ctl);
        Ok(())
    }

    /// Panes in grid order: top-left, top-right, bottom-left, bottom-right.
    pub fn layout(&self) -> Vec<String> {
        self.inner
            .layout
            .iter()
            .map(|p| p.name().to_string())
            .collect()
    }

    #[wasm_bindgen(js_name = setLayout)]
    pub fn set_layout(
        &mut self,
        top_left: &str,
        top_right: &str,
        bottom_left: &str,
        bottom_right: &str,
    ) -> Result<(), JsValue> {
        self.inner.layout = [
            pane_from_name(top_left)?,
            pane_from_name(top_right)?,
            pane_from_name(bottom_left)?,
            pane_from_name(bottom_right)?,
        ];
        Ok(())
    }

    /// Space between panes, logical pixels.
    #[wasm_bindgen(getter)]
    pub fn gap(&self) -> f32 {
        self.inner.gap
    }
    #[wasm_bindgen(setter)]
    pub fn set_gap(&mut self, gap: f32) {
        self.inner.gap = gap.max(0.0);
    }

    /// Space between the panes and the canvas edge, logical pixels.
    #[wasm_bindgen(getter)]
    pub fn border(&self) -> f32 {
        self.inner.border
    }
    #[wasm_bindgen(setter)]
    pub fn set_border(&mut self, border: f32) {
        self.inner.border = border.max(0.0);
    }

    /// Padding round the framed box in the orthographic panes, as a fraction
    /// of its extent.
    #[wasm_bindgen(getter)]
    pub fn margin(&self) -> f32 {
        self.inner.margin
    }
    #[wasm_bindgen(setter)]
    pub fn set_margin(&mut self, margin: f32) {
        self.inner.margin = margin.max(0.0);
    }

    /// What the gap and border are cleared to (linear, like every `WebColor`).
    #[wasm_bindgen(js_name = setGapColor)]
    pub fn set_gap_color(&mut self, color: &WebColor) {
        self.inner.gap_color = Some(color.inner);
    }

    /// Leave the gap and border untouched.
    #[wasm_bindgen(js_name = clearGapColor)]
    pub fn clear_gap_color(&mut self) {
        self.inner.gap_color = None;
    }

    /// Per-pane background overriding `scene.background`.
    #[wasm_bindgen(js_name = setPaneBackground)]
    pub fn set_pane_background(&mut self, pane: &str, color: &WebColor) -> Result<(), JsValue> {
        let p = pane_from_name(pane)?;
        self.inner.backgrounds[p.index()] = Some(color.inner);
        Ok(())
    }

    /// Back to the scene's background for `pane`.
    #[wasm_bindgen(js_name = clearPaneBackground)]
    pub fn clear_pane_background(&mut self, pane: &str) -> Result<(), JsValue> {
        let p = pane_from_name(pane)?;
        self.inner.backgrounds[p.index()] = None;
        Ok(())
    }

    /// The pane shown full-size, or `undefined` for all four.
    pub fn maximized(&self) -> Option<String> {
        self.inner.maximized.map(|p| p.name().to_string())
    }

    /// Show `pane` alone; `undefined`/`null` goes back to all four.
    #[wasm_bindgen(js_name = setMaximized)]
    pub fn set_maximized(&mut self, pane: Option<String>) -> Result<(), JsValue> {
        self.inner.maximized = match pane {
            Some(name) => Some(pane_from_name(&name)?),
            None => None,
        };
        Ok(())
    }

    /// Show `pane` alone, or — if it already is — go back to all four.
    #[wasm_bindgen(js_name = toggleMaximize)]
    pub fn toggle_maximize(&mut self, pane: &str) -> Result<(), JsValue> {
        self.inner.toggle_maximize(pane_from_name(pane)?);
        Ok(())
    }

    /// Aim the orthographic cameras at a box.
    #[wasm_bindgen(js_name = frameBox)]
    pub fn frame_box(&mut self, bounds: &WebBox3) {
        self.inner.frame_box(&bounds.inner);
    }

    /// Frame every visible mesh in `scene`. Returns false when there is
    /// nothing to frame (the cameras are left where they were).
    #[wasm_bindgen(js_name = frameScene)]
    pub fn frame_scene(&mut self, scene: &mut WebScene) -> bool {
        self.inner.frame_scene(&mut scene.inner).is_some()
    }

    /// The framed box's centre, `[x, y, z]` — a natural orbit target.
    pub fn target(&self) -> Vec<f32> {
        let t = self.inner.target();
        vec![t.x, t.y, t.z]
    }

    /// `[x, y, width, height]` of `pane` on a `width`×`height` canvas (logical
    /// pixels, bottom-left origin), or `undefined` when it is hidden.
    #[wasm_bindgen(js_name = paneRect)]
    pub fn pane_rect(
        &self,
        pane: &str,
        width: f32,
        height: f32,
    ) -> Result<Option<Vec<f32>>, JsValue> {
        let p = pane_from_name(pane)?;
        Ok(self
            .inner
            .pane_rect(p, width, height)
            .map(|r| vec![r.x, r.y, r.width, r.height]))
    }

    /// The pane under `(x, y)` — bottom-left origin, logical pixels — or
    /// `undefined` over a gap or border.
    #[wasm_bindgen(js_name = paneAt)]
    pub fn pane_at(&self, x: f32, y: f32, width: f32, height: f32) -> Option<String> {
        self.inner
            .pane_at(x, y, width, height)
            .map(|p| p.name().to_string())
    }

    /// Size the orthographic frusta for a `width`×`height` canvas. `render`
    /// does this itself; call it before reading the cameras out.
    pub fn fit(&mut self, width: f32, height: f32) {
        self.inner.fit(width, height);
    }

    /// A copy of the orthographic camera for `"top"`, `"front"` or `"side"`
    /// (as last fitted). Throws for `"perspective"`, which is the caller's.
    #[wasm_bindgen(js_name = orthoCamera)]
    pub fn ortho_camera(&self, pane: &str) -> Result<WebCamera, JsValue> {
        let p = pane_from_name(pane)?;
        let cam = self
            .inner
            .ortho_camera(p)
            .ok_or_else(|| JsValue::from_str("the perspective pane has no orthographic camera"))?;
        Ok(WebCamera {
            inner: CameraInner::Orthographic(cam.clone()),
        })
    }

    /// Draw every visible pane: the gap colour first, then each pane cleared
    /// to its background and drawn inside its own viewport and scissor. Goes
    /// to the canvas — acquired once for all panes, presented once — or to the
    /// renderer's active render target. `perspective` must be a perspective
    /// camera; its aspect is set per pane on a copy. The renderer's viewport,
    /// scissor and auto-clear are left as they were.
    pub fn render(
        &mut self,
        renderer: &mut WebRenderer,
        scene: &mut WebScene,
        perspective: &WebCamera,
    ) -> Result<(), JsValue> {
        let CameraInner::Perspective(persp) = &perspective.inner else {
            return Err(JsValue::from_str(
                "QuadView.render: the fourth pane needs a perspective camera",
            ));
        };
        if let Some(env_rt_id) = scene.inner.environment_cube_rt
            && let Some(target) = ACTIVE_CUBE_TARGETS.with(|m| m.borrow().get(&env_rt_id).cloned())
        {
            renderer
                .renderer
                .register_cube_render_target(env_rt_id, &target);
        }
        if let Some(id) = renderer.current_target_id
            && let Some(target) = ACTIVE_TARGETS.with(|m| m.borrow().get(&id).cloned())
        {
            let mut surface = WebPaneSurface {
                renderer: &mut renderer.renderer,
                target: WebPaneTarget::Target(&target),
            };
            self.inner.render(&mut surface, &mut scene.inner, persp);
            return Ok(());
        }
        // WebGL2 page already composing by hand: keep its back buffer current.
        if let Some((id, back)) = renderer.gl_back_buffer() {
            let mut surface = WebPaneSurface {
                renderer: &mut renderer.renderer,
                target: WebPaneTarget::Back(&back),
            };
            self.inner.render(&mut surface, &mut scene.inner, persp);
            renderer.present_gl_back_buffer(id);
            return Ok(());
        }
        match renderer.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                {
                    let mut surface = WebPaneSurface {
                        renderer: &mut renderer.renderer,
                        target: WebPaneTarget::View(&view),
                    };
                    self.inner.render(&mut surface, &mut scene.inner, persp);
                }
                renderer.renderer.queue_arc().present(frame);
            }
            _ => renderer
                .surface
                .configure(&renderer.device, &renderer.config),
        }
        Ok(())
    }
}

/// Shell fur's groom and look (`threers::fur::FurMaterial`), for
/// `WebScene.addFur`. Every setter returns nothing and changes this object;
/// fur already added keeps what it was built with.
#[wasm_bindgen]
pub struct WebFurMaterial {
    pub(crate) inner: crate::fur::FurMaterial,
}

#[wasm_bindgen]
impl WebFurMaterial {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebFurMaterial {
        WebFurMaterial {
            inner: crate::fur::FurMaterial::default(),
        }
    }
    /// Full strand length, world units.
    #[wasm_bindgen(js_name = setLength)]
    pub fn set_length(&mut self, v: f32) {
        self.inner.groom.length = v;
    }
    /// Strands per UV unit along each axis.
    #[wasm_bindgen(js_name = setDensity)]
    pub fn set_density(&mut self, v: f32) {
        self.inner.groom.density = v;
    }
    /// Root radius as a fraction of a strand's cell (≤ 0.5).
    #[wasm_bindgen(js_name = setThickness)]
    pub fn set_thickness(&mut self, v: f32) {
        self.inner.groom.thickness = v;
    }
    /// Tip radius as a fraction of the root's.
    #[wasm_bindgen(js_name = setTaper)]
    pub fn set_taper(&mut self, v: f32) {
        self.inner.groom.taper = v;
    }
    /// Droop at the tips, world units per unit of length.
    #[wasm_bindgen(js_name = setGravity)]
    pub fn set_gravity(&mut self, x: f32, y: f32, z: f32) {
        self.inner.groom.gravity = crate::Vector3::new(x, y, z);
    }
    /// Turns each strand makes over its length.
    #[wasm_bindgen(js_name = setCurl)]
    pub fn set_curl(&mut self, v: f32) {
        self.inner.groom.curl = v;
    }
    /// 0..1: how far tips gather into clumps; `size` in strands.
    #[wasm_bindgen(js_name = setClumping)]
    pub fn set_clumping(&mut self, amount: f32, size: f32) {
        self.inner.groom.clumping = amount;
        self.inner.groom.clump_size = size;
    }
    /// 0..1: how much shorter than the length a strand may be.
    #[wasm_bindgen(js_name = setLengthVariation)]
    pub fn set_length_variation(&mut self, v: f32) {
        self.inner.groom.length_variation = v;
    }
    #[wasm_bindgen(js_name = setSeed)]
    pub fn set_seed(&mut self, v: u32) {
        self.inner.groom.seed = v;
    }
    /// Shells above the skin (16–48 typical).
    #[wasm_bindgen(js_name = setShells)]
    pub fn set_shells(&mut self, v: u32) {
        self.inner.shells = v;
    }
    #[wasm_bindgen(js_name = setColors)]
    pub fn set_colors(&mut self, root: &WebColor, tip: &WebColor) {
        self.inner.root_color = root.inner;
        self.inner.tip_color = tip.inner;
    }
    /// 0..1: how much darker the roots are.
    #[wasm_bindgen(js_name = setOcclusion)]
    pub fn set_occlusion(&mut self, v: f32) {
        self.inner.occlusion = v;
    }
    /// The Kajiya-Kay highlight: strength, sharpness, and shift along the strand.
    #[wasm_bindgen(js_name = setSpecular)]
    pub fn set_specular(&mut self, strength: f32, exponent: f32, shift: f32) {
        self.inner.specular = strength;
        self.inner.specular_exponent = exponent;
        self.inner.specular_shift = shift;
    }
    /// The wind's velocity (m/s; it pulls by drag, as its speed squared)
    /// and gusts a second; animate with `WebScene.stepFur`.
    #[wasm_bindgen(js_name = setWind)]
    pub fn set_wind(&mut self, x: f32, y: f32, z: f32, frequency: f32) {
        self.inner.wind = crate::Vector3::new(x, y, z);
        self.inner.wind_frequency = frequency;
    }
    /// How far each strand flutters on its own, as a fraction of its cell.
    #[wasm_bindgen(js_name = setFlutter)]
    pub fn set_flutter(&mut self, v: f32) {
        self.inner.flutter = v;
    }
    /// How far up the coat shell fur's shadow reaches, as a share of its
    /// length (0.6); 0 casts none. (Hair's shadow is `shadowStrands`.)
    #[wasm_bindgen(js_name = setShadowHeight)]
    pub fn set_shadow_height(&mut self, v: f32) {
        self.inner.shadow_height = v;
    }
    /// Air drag (pull of air going by at 1 m/s; 0.15) and the gusts'
    /// strength (the turbulent share of the wind's speed; 0.35).
    #[wasm_bindgen(js_name = setAir)]
    pub fn set_air(&mut self, drag: f32, turbulence: f32) {
        self.inner.drag = drag;
        self.inner.turbulence = turbulence;
    }
    /// Draw fewer shells where the fur is small on screen (on by default).
    #[wasm_bindgen(js_name = setLod)]
    pub fn set_lod(&mut self, on: bool) {
        self.inner.lod = on;
    }
    /// Inertia: spring rate (1/s²), damping (0..1 of critical), response (1 =
    /// tips as free masses) and the largest swing (fraction of the length).
    #[wasm_bindgen(js_name = setDynamics)]
    pub fn set_dynamics(&mut self, stiffness: f32, damping: f32, response: f32, max_lag: f32) {
        self.inner.dynamics = crate::fur::FurDynamics {
            stiffness,
            damping,
            response,
            max_lag,
        };
    }
    /// Root strands over the surface in object space, one per `cell × cell`
    /// (no UVs needed); `0` puts them back on the UVs.
    #[wasm_bindgen(js_name = setSurfacePlacement)]
    pub fn set_surface_placement(&mut self, cell: f32) {
        self.inner.groom.placement = if cell > 0.0 {
            crate::fur::FurPlacement::Surface { cell }
        } else {
            crate::fur::FurPlacement::Uv
        };
    }
}

impl Default for WebFurMaterial {
    fn default() -> Self {
        Self::new()
    }
}
