//! Bindings split out of `wasm.rs` by area (see the `mod` list there).
#![allow(unused_imports)]

use super::*;

// ======================================================================
//                     DEBUG HELPERS (three.js r165 parity)
// ======================================================================
//
// Each `helper*` function returns what three.js's helper of that name draws,
// already in world space — see `crate::helpers::three_lines`. Matrices are
// `Matrix4.elements` (column-major, 16 numbers); a malformed one reads as the
// identity.

use crate::helpers::three_lines as lines;

/// A helper's geometry: a line list, optional per-point colours, and the
/// triangles of its filled part.
#[wasm_bindgen]
pub struct WebHelperGeometry {
    inner: lines::HelperGeometry,
}

#[wasm_bindgen]
impl WebHelperGeometry {
    /// Line list, `xyz` per point, two points per segment.
    #[wasm_bindgen(getter)]
    pub fn lines(&self) -> Vec<f32> {
        self.inner.lines.clone()
    }
    /// Linear `rgb` per line point, or empty when the material colour is used.
    #[wasm_bindgen(getter)]
    pub fn colors(&self) -> Vec<f32> {
        self.inner.colors.clone()
    }
    /// Triangle list (`xyz` per corner) of the filled part, or empty.
    #[wasm_bindgen(getter)]
    pub fn mesh(&self) -> Vec<f32> {
        self.inner.mesh.clone()
    }
}

fn helper(inner: lines::HelperGeometry) -> WebHelperGeometry {
    WebHelperGeometry { inner }
}

fn mat4(m: &[f64]) -> lines::Mat4 {
    m.try_into().unwrap_or(lines::IDENTITY)
}

fn rgb(c: &WebColor) -> [f32; 3] {
    [c.inner.r, c.inner.g, c.inner.b]
}

fn view6(view: Option<Vec<f64>>) -> Option<[f64; 6]> {
    view.and_then(|v| v.as_slice().try_into().ok())
}

/// `DirectionalLightHelper(light, size)`.
#[wasm_bindgen(js_name = helperDirectionalLight)]
pub fn helper_directional_light(
    light_world: &[f64],
    tx: f64,
    ty: f64,
    tz: f64,
    size: f64,
) -> WebHelperGeometry {
    helper(lines::directional_light(
        &mat4(light_world),
        [tx, ty, tz],
        size,
    ))
}

/// `HemisphereLightHelper(light, size)`; pass a fixed colour as both.
#[wasm_bindgen(js_name = helperHemisphereLight)]
pub fn helper_hemisphere_light(
    light_world: &[f64],
    size: f64,
    sky: &WebColor,
    ground: &WebColor,
) -> WebHelperGeometry {
    helper(lines::hemisphere_light(
        &mat4(light_world),
        size,
        rgb(sky),
        rgb(ground),
    ))
}

/// `PointLightHelper(light, sphereSize)`.
#[wasm_bindgen(js_name = helperPointLight)]
pub fn helper_point_light(light_world: &[f64], sphere_size: f64) -> WebHelperGeometry {
    helper(lines::point_light(&mat4(light_world), sphere_size))
}

/// `SpotLightHelper(light)`.
#[wasm_bindgen(js_name = helperSpotLight)]
pub fn helper_spot_light(
    light_world: &[f64],
    tx: f64,
    ty: f64,
    tz: f64,
    distance: f64,
    angle: f64,
) -> WebHelperGeometry {
    helper(lines::spot_light(
        &mat4(light_world),
        [tx, ty, tz],
        distance,
        angle,
    ))
}

/// `PerspectiveCamera.updateProjectionMatrix` (`view` is `[fullWidth,
/// fullHeight, offsetX, offsetY, width, height]` when enabled).
#[wasm_bindgen(js_name = helperPerspectiveProjection)]
pub fn helper_perspective_projection(
    fov: f64,
    aspect: f64,
    near: f64,
    far: f64,
    zoom: f64,
    film_offset: f64,
    film_width: f64,
    view: Option<Vec<f64>>,
) -> Vec<f64> {
    lines::perspective_projection(
        fov,
        aspect,
        near,
        far,
        zoom,
        film_offset,
        film_width,
        view6(view),
    )
    .to_vec()
}

/// `OrthographicCamera.updateProjectionMatrix`.
#[wasm_bindgen(js_name = helperOrthographicProjection)]
pub fn helper_orthographic_projection(
    left: f64,
    right: f64,
    top: f64,
    bottom: f64,
    near: f64,
    far: f64,
    zoom: f64,
    view: Option<Vec<f64>>,
) -> Vec<f64> {
    lines::orthographic_projection(left, right, top, bottom, near, far, zoom, view6(view)).to_vec()
}

/// `CameraHelper(camera)` from the camera's projection matrix (inverted
/// here, as `projectionMatrixInverse`) and world matrix, with
/// `setColors(frustum, cone, up, target, cross)`.
#[wasm_bindgen(js_name = helperCamera)]
pub fn helper_camera(
    projection: &[f64],
    camera_world: &[f64],
    frustum: &WebColor,
    cone: &WebColor,
    up: &WebColor,
    target: &WebColor,
    cross: &WebColor,
) -> WebHelperGeometry {
    let inverse = lines::invert(&mat4(projection));
    helper(lines::camera(
        &inverse,
        &mat4(camera_world),
        [rgb(frustum), rgb(cone), rgb(up), rgb(target), rgb(cross)],
    ))
}

/// `ArrowHelper.setDirection(dir)`: the helper's quaternion `[x, y, z, w]`.
#[wasm_bindgen(js_name = helperArrowQuaternion)]
pub fn helper_arrow_quaternion(x: f64, y: f64, z: f64) -> Vec<f64> {
    lines::arrow_quaternion([x, y, z]).to_vec()
}

/// `ArrowHelper` under its own world matrix: the shaft line and the cone.
#[wasm_bindgen(js_name = helperArrow)]
pub fn helper_arrow(
    helper_world: &[f64],
    length: f64,
    head_length: f64,
    head_width: f64,
) -> WebHelperGeometry {
    helper(lines::arrow(
        &mat4(helper_world),
        length,
        head_length,
        head_width,
    ))
}

/// `PlaneHelper(plane, size)`: outline and translucent fill.
#[wasm_bindgen(js_name = helperPlane)]
pub fn helper_plane(nx: f64, ny: f64, nz: f64, constant: f64, size: f64) -> WebHelperGeometry {
    helper(lines::plane([nx, ny, nz], constant, size))
}

/// `SkeletonHelper(object)`: `bone_worlds` is every bone's `matrixWorld`,
/// 16 numbers each, in `getBoneList` order; `parents[i]` the index of bone
/// `i`'s parent bone, or -1.
#[wasm_bindgen(js_name = helperSkeleton)]
pub fn helper_skeleton(
    root_world: &[f64],
    bone_worlds: &[f64],
    parents: &[i32],
) -> WebHelperGeometry {
    let bones: Vec<lines::Mat4> = bone_worlds.chunks_exact(16).map(mat4).collect();
    helper(lines::skeleton(&mat4(root_world), &bones, parents))
}

/// `BoxHelper`: the edges of a world-space box.
#[wasm_bindgen(js_name = helperBox)]
pub fn helper_box(min: &[f64], max: &[f64]) -> WebHelperGeometry {
    let v = |a: &[f64]| -> [f64; 3] { a.try_into().unwrap_or([0.0; 3]) };
    helper(lines::box_lines(v(min), v(max)))
}

/// `BoxHelper(object)`: the edges of `Box3.setFromObject(object)`, given
/// each mesh's world matrix (16 numbers apiece in `worlds`) and its vertices
/// (`positions`, `counts[i]` floats for mesh `i`). Empty for no vertices.
#[wasm_bindgen(js_name = helperBoxOfObjects)]
pub fn helper_box_of_objects(
    worlds: &[f64],
    positions: &[f32],
    counts: &[u32],
) -> WebHelperGeometry {
    let mats: Vec<lines::Mat4> = worlds.chunks_exact(16).map(mat4).collect();
    let mut slices = Vec::with_capacity(counts.len());
    let mut at = 0usize;
    for &n in counts {
        let end = (at + n as usize).min(positions.len());
        slices.push(&positions[at..end]);
        at = end;
    }
    match lines::object_bounds(&mats, &slices) {
        Some((min, max)) => helper(lines::box_lines(min, max)),
        None => helper(lines::HelperGeometry::default()),
    }
}

/// `Box3Helper(box)`.
#[wasm_bindgen(js_name = helperBox3)]
pub fn helper_box3(min: &[f64], max: &[f64]) -> WebHelperGeometry {
    let v = |a: &[f64]| -> [f64; 3] { a.try_into().unwrap_or([0.0; 3]) };
    helper(lines::box3(v(min), v(max)))
}

/// `VertexNormalsHelper(object, size)`.
#[wasm_bindgen(js_name = helperVertexNormals)]
pub fn helper_vertex_normals(
    object_world: &[f64],
    positions: &[f32],
    normals: &[f32],
    size: f64,
) -> WebHelperGeometry {
    helper(lines::vertex_normals(
        &mat4(object_world),
        positions,
        normals,
        size,
    ))
}

/// `RectAreaLightHelper(light)`: outline and back-facing fill.
#[wasm_bindgen(js_name = helperRectAreaLight)]
pub fn helper_rect_area_light(light_world: &[f64], width: f64, height: f64) -> WebHelperGeometry {
    helper(lines::rect_area_light(&mat4(light_world), width, height))
}

/// `RectAreaLightHelper`'s colour when none is fixed: the light's linear
/// colour times its intensity, kept in range without a hue shift, as linear
/// `[r, g, b]`.
#[wasm_bindgen(js_name = helperRectAreaLightColor)]
pub fn helper_rect_area_light_color(color: &WebColor, intensity: f32) -> Vec<f32> {
    lines::rect_area_light_color(rgb(color), intensity).to_vec()
}

// ======================================================================
//                     LOADERS: KTX2 textures, Draco meshes
// ======================================================================

#[wasm_bindgen]
impl WebKtx2Loader {
    /// The whole file as one texture, mip chain included — what three.js's
    /// `KTX2Loader` returns for a non-Basis file: zlib supercompression undone,
    /// BGRA swizzled, block-compressed (BC1/BC7) levels kept as blocks. Basis
    /// Universal and zstd payloads, cube maps and arrays are refused by name.
    #[wasm_bindgen(js_name = parseTexture)]
    pub fn parse_texture(&self, bytes: &[u8]) -> Result<WebTexture, JsValue> {
        let ktx = crate::loaders::ktx2::parse(bytes)
            .map_err(|e| JsValue::from_str(&format!("KTX2Loader: {e}")))?;
        let texture = ktx
            .to_texture()
            .map_err(|e| JsValue::from_str(&format!("KTX2Loader: {e}")))?;
        Ok(WebTexture {
            inner: Arc::new(texture),
        })
    }
}

/// three.js's `DRACOLoader` over the crate's own Draco decoder
/// (`crate::loaders::draco`) — Google's decoder is not involved.
#[wasm_bindgen]
pub struct WebDracoLoader;

#[wasm_bindgen]
impl WebDracoLoader {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebDracoLoader {
        WebDracoLoader
    }

    /// A `.drc` mesh as `DRACOLoader` builds it: `position`, `normal`,
    /// `color` and `uv` (the first attribute of each type, as floats) and the
    /// index. `srgbColors` (true for `load`/`parse`, false for
    /// `decodeDracoFile`'s default) decodes vertex colours to linear.
    pub fn decode(&self, bytes: &[u8], srgb_colors: bool) -> Result<WebBufferGeometry, JsValue> {
        crate::loaders::draco::three::decode_three(bytes, srgb_colors)
            .map(|g| WebBufferGeometry { inner: Arc::new(g) })
            .map_err(|e| JsValue::from_str(&format!("DRACOLoader: {e}")))
    }
}

// ======================================================================
//              three.js's SimplexNoise (examples/jsm/math)
// ======================================================================

/// three.js r165's `SimplexNoise`, seeded with the 256 bytes three.js draws
/// as `Math.floor(r.random() * 256)` — see `crate::extras::ThreeSimplexNoise`.
#[wasm_bindgen]
pub struct WebThreeSimplexNoise {
    inner: crate::extras::ThreeSimplexNoise,
}

#[wasm_bindgen]
impl WebThreeSimplexNoise {
    /// `permutation`: 256 values in `0..=255`.
    #[wasm_bindgen(constructor)]
    pub fn new(permutation: &[u8]) -> Result<WebThreeSimplexNoise, JsValue> {
        let p: &[u8; 256] = permutation
            .try_into()
            .map_err(|_| JsValue::from_str("SimplexNoise: the permutation needs 256 entries"))?;
        Ok(WebThreeSimplexNoise {
            inner: crate::extras::ThreeSimplexNoise::from_permutation(p),
        })
    }
    pub fn noise(&self, x: f64, y: f64) -> f64 {
        self.inner.noise(x, y)
    }
    pub fn noise3d(&self, x: f64, y: f64, z: f64) -> f64 {
        self.inner.noise3d(x, y, z)
    }
    pub fn noise4d(&self, x: f64, y: f64, z: f64, w: f64) -> f64 {
        self.inner.noise4d(x, y, z, w)
    }
}
