//! Drawing splats: a [`ShaderMaterial`] that projects each one as EWA
//! splatting does, on an [`InstancedMesh`] of quads, and the sort that orders
//! them.

use std::sync::Arc;

use super::{covariance, sh_rest_coefficients, GaussianSplats, SplatError};
use crate::cameras::Camera;
use crate::core::{BufferAttribute, BufferGeometry, InstancedMesh, Object3D, ObjectId};
use crate::materials::{Material, ShaderMaterial};
use crate::math::{Color, Matrix4};
use crate::scene::Scene;

/// Splats one instance draws. The geometry is this many quads; splat `k` of
/// instance `i` is the `i * SPLATS_PER_INSTANCE + k`th in the sorted order.
/// One instance per splat would cost the renderer's per-instance matrix and
/// tint — 112 bytes, twice the splat itself — a million times over.
pub const SPLATS_PER_INSTANCE: u32 = 128;

/// Floats a splat takes in storage slot 0, as four `vec4`s: centre and
/// opacity; the covariance's `xx xy xz yy`; its `yz zz` and colour's red and
/// green; blue and three unused. Fours, because WebGL2 reads storage out of a
/// float texture a texel at a time, and `s0x4` takes a texel at once where
/// thirteen `s0` calls fetched thirteen.
const STRIDE: usize = 16;

/// wgpu's default `max_storage_buffer_binding_size`, which is what the
/// renderer asks for: the most any one storage slot may hold.
pub const DEFAULT_MAX_STORAGE_BYTES: u64 = 128 << 20;

/// How splats are drawn.
#[derive(Debug, Clone, PartialEq)]
pub struct SplatOptions {
    /// The highest spherical-harmonic degree uploaded and evaluated (0..=3).
    /// Lower costs memory and vertex time and loses view-dependent colour.
    pub max_sh_degree: u32,
    /// Scales every splat's opacity.
    pub opacity: f32,
    /// Mip-Splatting's opacity compensation: keep a splat's total energy
    /// when the screen-space low-pass widens it, so small splats fade
    /// rather than bloat. Right for scenes trained with it (Mip-Splatting,
    /// gsplat's `antialiased` mode); the reference trainer was not, and its
    /// scenes look thinner with it. On for every splat set whose
    /// [`GaussianSplats::antialiased`](super::GaussianSplats::antialiased)
    /// is, whatever this says.
    pub antialiased: bool,
    /// The largest a splat's quad may reach from its centre, in pixels: a
    /// guard against a near, huge splat filling the screen.
    pub max_screen_radius: f32,
    /// The most one storage slot may hold, in bytes: the device's
    /// `max_storage_buffer_binding_size`. Splats that do not fit are an
    /// error: 16 floats a splat in one slot, and at degree 1, 2 and 3, 4, 12
    /// and 16 in another — about 2.1 million splats at the default, at any
    /// degree.
    pub max_storage_bytes: u64,
}

impl Default for SplatOptions {
    fn default() -> Self {
        Self {
            max_sh_degree: 3,
            opacity: 1.0,
            antialiased: false,
            max_screen_radius: 1024.0,
            max_storage_bytes: DEFAULT_MAX_STORAGE_BYTES,
        }
    }
}

/// Floats a splat takes in storage slot 2 at `degree`: a scale, then one per
/// coefficient (see [`pack_sh`]), padded to whole `vec4`s.
fn sh_stride(degree: u32) -> usize {
    match sh_rest_coefficients(degree) {
        0 => 0,
        n => (n + 1).next_multiple_of(4),
    }
}

/// The higher bands, packed: per splat the largest coefficient's magnitude
/// `m`, then each coefficient's red, green and blue quantised to bytes over
/// `-m..=m` and stored as the integer `r << 16 | g << 8 | b`, which an `f32`
/// holds exactly. A coefficient is off by at most `m / 255` — finer than the
/// fixed-range bytes compressed splat formats store — at a third of the
/// floats.
fn pack_sh(rest: &[f32], degree: u32) -> Vec<f32> {
    let n = sh_rest_coefficients(degree);
    if n == 0 {
        return Vec::new();
    }
    let stride = sh_stride(degree);
    let mut out = Vec::with_capacity(rest.len() / (n * 3) * stride);
    for splat in rest.chunks_exact(n * 3) {
        let m = splat.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        out.push(m);
        let q = |v: f32| {
            if m > 0.0 {
                ((v / m) * 127.5 + 127.5).round().clamp(0.0, 255.0) as u32
            } else {
                128
            }
        };
        for c in splat.as_chunks::<3>().0 {
            out.push((q(c[0]) << 16 | q(c[1]) << 8 | q(c[2])) as f32);
        }
        out.extend(std::iter::repeat_n(0.0, stride - (n + 1)));
    }
    out
}

/// The material [`splat_object`] draws `splats` with: their data in its
/// storage, the shader below, transparent and double-sided. The draw order
/// (slot 1) starts as the splats' own; a [`SplatSorter`] made from the same
/// material replaces it.
///
/// Fails if `splats` fails [`validate`](GaussianSplats::validate), holds more
/// than 2²⁴ splats (the order is stored as exact `f32`s), or more than fit in
/// [`SplatOptions::max_storage_bytes`].
pub fn splat_material(
    splats: &GaussianSplats,
    options: &SplatOptions,
) -> Result<ShaderMaterial, SplatError> {
    splats.validate()?;
    let n = splats.len();
    if n > 1 << 24 {
        return Err(SplatError::new(format!(
            "{n} splats is past the 2^24 a sort order can index"
        )));
    }
    let fits = |floats_per_splat: usize| (n * floats_per_splat * 4) as u64 <= options.max_storage_bytes;
    if !fits(STRIDE) {
        return Err(SplatError::new(format!(
            "{n} splats take {} bytes, past the {}-byte storage limit",
            n * STRIDE * 4,
            options.max_storage_bytes
        )));
    }
    // The bands never take more than slot 0 does, so they fit if it did.
    let degree = splats.sh_degree.min(options.max_sh_degree);
    debug_assert!(fits(sh_stride(degree)));
    let mut data = Vec::with_capacity(n * STRIDE);
    for i in 0..n {
        let [x, y, z] = splats.positions[i];
        let [xx, xy, xz, yy, yz, zz] = covariance(splats.scales[i], splats.rotations[i]);
        let [r, g, b] = splats.colors[i];
        data.extend_from_slice(&[x, y, z, splats.opacities[i]]);
        data.extend_from_slice(&[xx, xy, xz, yy, yz, zz, r, g, b, 0.0, 0.0, 0.0]);
    }
    let sh = if degree == splats.sh_degree {
        pack_sh(&splats.sh, degree)
    } else {
        let mut s = splats.clone();
        s.truncate_sh(degree);
        pack_sh(&s.sh, degree)
    };
    debug_assert_eq!(sh.len(), n * sh_stride(degree));
    let mut m = ShaderMaterial::new(SPLAT_FRAGMENT_WGSL);
    m.vertex = Some(SPLAT_VERTEX_WGSL.to_string());
    m.data = vec![
        [
            n as f32,
            degree as f32,
            SPLATS_PER_INSTANCE as f32,
            if options.antialiased || splats.antialiased { 1.0 } else { 0.0 },
        ],
        // The bands' stride in texels.
        [options.opacity, options.max_screen_radius, (sh_stride(degree) / 4) as f32, 0.0],
    ];
    m.storage0 = data;
    m.storage1 = (0..n).map(|i| i as f32).collect();
    m.storage2 = sh;
    m.transparent = true;
    m.side = 2;
    // The shader decodes the trained (display-space) colours to linear and
    // `framebuffer_encode` encodes them back: tone mapping would change them.
    m.tone_mapped = false;
    m.render_state.depth_write = Some(false);
    m.fallback_color = Color::new(0.5, 0.5, 0.5);
    Ok(m)
}

/// The spherical-harmonic degree `material` draws — [`splat_material`]'s:
/// the splats' own, capped at [`SplatOptions::max_sh_degree`].
pub fn splat_sh_degree(material: &ShaderMaterial) -> u32 {
    material.data.first().map_or(0, |d| d[1] as u32)
}

/// `splats` as an object to add to a scene, and the sorter that keeps them in
/// order: call [`SplatSorter::update`] once a frame before rendering. Fails as
/// [`splat_material`] does.
///
/// The object is an [`InstancedMesh`] of [`SPLATS_PER_INSTANCE`] quads whose
/// corners all sit at the origin — the vertex stage places them — so it casts
/// no shadow and a raycast passes through it.
pub fn splat_object(
    splats: &GaussianSplats,
    options: &SplatOptions,
) -> Result<(Object3D, SplatSorter), SplatError> {
    let material = splat_material(splats, options)?;
    let sorter = SplatSorter::new(splats, &material);
    let instances = splats.len().div_ceil(SPLATS_PER_INSTANCE as usize);
    let mut mesh = InstancedMesh::from_arc(
        Arc::new(quads(SPLATS_PER_INSTANCE as usize)),
        Arc::new(Material::Shader(material)),
        instances,
    );
    // The instances are identities and never change: upload them once.
    mesh.instance_version = Some(0);
    let mut object = Object3D::instanced_mesh(mesh);
    object.name = "GaussianSplats".into();
    object.cast_shadow = false;
    object.receive_shadow = false;
    Ok((object, sorter))
}

/// `count` quads, two triangles each, every corner at the origin: the vertex
/// stage reads which corner it is from its index.
fn quads(count: usize) -> BufferGeometry {
    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(vec![0.0; count * 4 * 3], 3));
    let index = (0..count as u32)
        .flat_map(|q| [0, 1, 2, 0, 2, 3].map(|c| q * 4 + c))
        .collect();
    g.set_index(index);
    g
}

/// Keeps splats sorted back to front, by their depth along the camera's view
/// axis, and uploads the order.
///
/// Holds the splats' centres and a handle on their material's live storage,
/// so it reaches the material however many clones of it there are.
pub struct SplatSorter {
    material: ShaderMaterial,
    /// The centres and the radix sort over them (`threers-splat-sort`).
    sort: threers_splat_sort::DepthSort,
    /// The (normalised) object-space view axis of the last sort. The order
    /// depends on nothing else: not on where along it the camera stands, nor
    /// on any move across it.
    last_axis: Option<[f32; 3]>,
}

impl SplatSorter {
    /// A sorter for `splats` drawn with `material` — [`splat_material`]'s, or
    /// any clone of it.
    pub fn new(splats: &GaussianSplats, material: &ShaderMaterial) -> Self {
        Self {
            material: material.clone(),
            sort: threers_splat_sort::DepthSort::new(&splats.positions),
            last_axis: None,
        }
    }

    /// Sort for `camera` viewing the splat object `id` in `scene`, updating
    /// the scene's world matrices first. Returns whether the order was
    /// re-sorted (and is uploaded with the next frame).
    pub fn update<C: Camera + ?Sized>(&mut self, scene: &mut Scene, id: ObjectId, camera: &C) -> bool {
        scene.update_world();
        let Some(object) = scene.get(id) else {
            return false;
        };
        let model_view = camera.view_matrix().multiply(&object.matrix_world);
        self.sort(&model_view)
    }

    /// The axis [`update`](Self::update) would sort along for `camera`
    /// viewing object `id`: the unit view direction in the splats' own
    /// space. For sorting elsewhere — on another thread, with
    /// [`threers_splat_sort::DepthSort`] over [`centres`](Self::centres) —
    /// and handing the order back with [`set_order`](Self::set_order).
    pub fn view_axis<C: Camera + ?Sized>(scene: &mut Scene, id: ObjectId, camera: &C) -> Option<[f32; 3]> {
        scene.update_world();
        let object = scene.get(id)?;
        axis_of(&camera.view_matrix().multiply(&object.matrix_world))
    }

    /// The splats' centres, as `x`, `y` and `z` columns.
    pub fn centres(&self) -> [&[f32]; 3] {
        self.sort.columns()
    }

    /// Draw in `order` (splat indices as floats, back to front), sorted
    /// elsewhere. [`order`](Self::order) still reports this sorter's own last
    /// sort; the next [`update`](Self::update) sorts again whatever the view.
    ///
    /// # Panics
    ///
    /// If `order` does not hold one index per splat.
    pub fn set_order(&mut self, order: Vec<f32>) {
        assert_eq!(order.len(), self.sort.len(), "an order of the wrong length");
        self.last_axis = None;
        self.material.set_storage(1, order);
    }

    /// Sort for the object-to-view matrix `model_view` (the camera's view
    /// matrix times the object's world matrix). Returns whether the view axis
    /// had changed since the last sort and so the order was redone.
    pub fn sort(&mut self, model_view: &Matrix4) -> bool {
        let Some(axis) = axis_of(model_view) else {
            return false;
        };
        if self.last_axis == Some(axis) {
            return false;
        }
        self.last_axis = Some(axis);
        // Ascending view-space z, which is negative in front of the camera:
        // farthest first.
        self.sort.sort(axis);
        let mut order = Vec::with_capacity(self.sort.len());
        self.sort.order_f32(&mut order);
        self.material.set_storage(1, order);
        true
    }

    /// The current draw order: splat indices, back to front.
    pub fn order(&self) -> &[u32] {
        self.sort.order()
    }
}

/// Row 2 of `model_view`, normalised: view-space z of a centre `p` is that
/// row applied to it.
fn axis_of(model_view: &Matrix4) -> Option<[f32; 3]> {
    let e = &model_view.elements;
    let row = [e[2], e[6], e[10]];
    let len = row.iter().map(|v| v * v).sum::<f32>().sqrt();
    (len > 0.0 && len.is_finite()).then(|| row.map(|v| v / len))
}

/// The fragment: the Gaussian's falloff at this point of the quad, whose `uv`
/// is the offset from the centre in standard deviations along the
/// footprint's axes. Fog, tone mapping and encoding stay here rather than
/// in the vertex stage, though they are the same across a splat: most
/// splats cover fewer pixels than their quad has corners, and moving them
/// to the corners made a scene of small splats 20% slower.
pub(crate) const SPLAT_FRAGMENT_WGSL: &str = crate::wgsl!(
    r#"
@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
    let a = min(in.vertex_color.a * exp(-0.5 * dot(in.uv, in.uv)), 0.99);
    // Below one step of an 8-bit target it would change nothing; written as
    // zero rather than discarded, which is far cheaper under MSAA.
    let rgb = framebuffer_encode(apply_tone_mapping(apply_fog(in.vertex_color.rgb, in.view_z)));
    return vec4<f32>(rgb, select(0.0, a, a >= 1.0 / 255.0));
}
"#
);

/// The vertex stage: one quad corner of one splat, projected as the
/// reference rasteriser projects it.
pub(crate) const SPLAT_VERTEX_WGSL: &str = crate::wgsl!(
    r#"
// Coefficient k of the splat whose packed bands start at `base`: bytes
// r, g, b of an exact integer, over -m..=m with m the float at `base`. The
// float is converted as it is: adding 0.5 to round it, as this once did,
// cannot be represented past 2^23 and rounds to even — off by one in blue,
// and at blue 255 a carry that turned green and blue to their opposites.
fn splat_sh_k(packed : f32, m : f32) -> vec3<f32> {
    let v = u32(packed);
    let q = vec3<f32>(f32(v >> 16u), f32((v >> 8u) & 255u), f32(v & 255u));
    return (q - vec3<f32>(127.5)) * (m / 127.5);
}

// The bands above 0 towards the unit object-space direction `d`, as the
// reference rasteriser's computeColorFromSH has them. Splat `idx`'s bands
// are `stride` texels of slot 2: the scale m, then the coefficients.
fn splat_sh(idx : u32, degree : u32, stride : u32, d : vec3<f32>) -> vec3<f32> {
    let b = idx * stride;
    let x = d.x;
    let y = d.y;
    let z = d.z;
    let t0 = s2x4(b);
    let m = t0.x;
    var c = 0.48860251 * (-y * splat_sh_k(t0.y, m) + z * splat_sh_k(t0.z, m) - x * splat_sh_k(t0.w, m));
    if (degree >= 2u) {
        let t1 = s2x4(b + 1u);
        let t2 = s2x4(b + 2u);
        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        c = c + 1.09254843 * x * y * splat_sh_k(t1.x, m)
            - 1.09254843 * y * z * splat_sh_k(t1.y, m)
            + 0.31539157 * (2.0 * zz - xx - yy) * splat_sh_k(t1.z, m)
            - 1.09254843 * x * z * splat_sh_k(t1.w, m)
            + 0.54627421 * (xx - yy) * splat_sh_k(t2.x, m);
        if (degree >= 3u) {
            let t3 = s2x4(b + 3u);
            c = c - 0.59004359 * y * (3.0 * xx - yy) * splat_sh_k(t2.y, m)
                + 2.89061144 * x * y * z * splat_sh_k(t2.z, m)
                - 0.45704580 * y * (4.0 * zz - xx - yy) * splat_sh_k(t2.w, m)
                + 0.37317633 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * splat_sh_k(t3.x, m)
                - 0.45704580 * x * (4.0 * zz - xx - yy) * splat_sh_k(t3.y, m)
                + 1.44530572 * z * (xx - yy) * splat_sh_k(t3.z, m)
                - 0.59004359 * x * (xx - 3.0 * yy) * splat_sh_k(t3.w, m);
        }
    }
    return c;
}

fn splat_inverse3(m : mat3x3<f32>) -> mat3x3<f32> {
    let c0 = cross(m[1], m[2]);
    let c1 = cross(m[2], m[0]);
    let c2 = cross(m[0], m[1]);
    return transpose(mat3x3<f32>(c0, c1, c2)) * (1.0 / dot(m[0], c0));
}

fn splat_srgb_decode(c : vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

@vertex
fn vs_custom(in : InstancedCustomIn, @builtin(vertex_index) vid : u32,
             @builtin(instance_index) iid : u32) -> VsOut {
    var out : VsOut;
    // Outside the clip volume on every axis: what a skipped splat returns.
    out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    let count = u32(u_data.data[0].x);
    let slot = iid * u32(u_data.data[0].z) + vid / 4u;
    if (slot >= count) {
        return out;
    }
    // An exact integer below 2^24, converted as it is (see `splat_sh_k`).
    let idx = u32(s1(slot));
    let p0 = s0x4(idx * 4u);
    let p1 = s0x4(idx * 4u + 1u);
    let p2 = s0x4(idx * 4u + 2u);
    let local = p0.xyz;
    let world = mesh.model * vec4<f32>(local, 1.0);
    let view = frame.view * world;
    let clip = frame.projection * view;
    if (clip.w <= 0.0) {
        return out;
    }
    // Centres well off screen are skipped, as the reference does.
    let ndc = clip.xy / clip.w;
    if (abs(ndc.x) > 1.3 || abs(ndc.y) > 1.3) {
        return out;
    }

    // The covariance into view space, then through the projection's local
    // affine approximation (its Jacobian) to pixels.
    let sigma = mat3x3<f32>(
        vec3<f32>(p1.x, p1.y, p1.z),
        vec3<f32>(p1.y, p1.w, p2.x),
        vec3<f32>(p1.z, p2.x, p2.y));
    let m3 = mat3x3<f32>(mesh.model[0].xyz, mesh.model[1].xyz, mesh.model[2].xyz);
    let v3 = mat3x3<f32>(frame.view[0].xyz, frame.view[1].xyz, frame.view[2].xyz);
    let w = v3 * m3;
    let cv = w * sigma * transpose(w);
    let fx = frame.projection[0][0] * 0.5 * frame.viewport_size.x;
    let fy = frame.projection[1][1] * 0.5 * frame.viewport_size.y;
    var j0 = vec3<f32>(fx, 0.0, 0.0);
    var j1 = vec3<f32>(0.0, fy, 0.0);
    if (frame.projection[3][3] < 0.5) {
        // Perspective. The point is held inside 1.3 times the frustum, as the
        // reference holds it, so a splat at the edge does not stretch without
        // bound.
        let d = -view.z;
        let lx = 1.3 / frame.projection[0][0];
        let ly = 1.3 / frame.projection[1][1];
        let tx = clamp(view.x / d, -lx, lx) * d;
        let ty = clamp(view.y / d, -ly, ly) * d;
        j0 = vec3<f32>(fx / d, 0.0, fx * tx / (d * d));
        j1 = vec3<f32>(0.0, fy / d, fy * ty / (d * d));
    }
    let a0 = dot(j0, cv * j0);
    let bb = dot(j0, cv * j1);
    let c0 = dot(j1, cv * j1);
    // The low-pass: every footprint at least a third of a pixel across.
    let a = a0 + 0.3;
    let c = c0 + 0.3;
    var alpha = p0.w * u_data.data[1].x;
    if (u_data.data[0].w > 0.5) {
        alpha = alpha * sqrt(max(a0 * c0 - bb * bb, 0.0) / (a * c - bb * bb));
    }
    // The fragment writes nothing below 1/255: past r standard deviations,
    // with alpha·exp(-r²/2) = 1/255, there is nothing to draw. A faint splat
    // gets a smaller quad than three deviations, and one that never reaches
    // 1/255 none.
    let reach2 = 2.0 * log(255.0 * min(alpha, 0.99));
    if (reach2 <= 0.0) {
        return out;
    }
    let reach = min(3.0, sqrt(reach2));

    // The footprint's axes: the 2x2 covariance's eigenvectors.
    let mid = 0.5 * (a + c);
    let rad = length(vec2<f32>(0.5 * (a - c), bb));
    let l1 = mid + rad;
    let l2 = max(mid - rad, 1e-4);
    var e1 = select(vec2<f32>(bb, l1 - a), vec2<f32>(l1 - c, bb), a >= c);
    let e1_len = length(e1);
    e1 = select(vec2<f32>(1.0, 0.0), e1 / e1_len, e1_len > 1e-12);
    let e2 = vec2<f32>(-e1.y, e1.x);
    // Three standard deviations at most, within the screen-radius guard.
    let r1 = min(reach * sqrt(l1), u_data.data[1].y);
    let r2 = min(reach * sqrt(l2), u_data.data[1].y);
    let k = vid & 3u;
    let corner = vec2<f32>(select(-1.0, 1.0, k == 1u || k == 2u), select(-1.0, 1.0, k >= 2u));
    let px = e1 * (corner.x * r1) + e2 * (corner.y * r2);
    out.clip_pos = clip + vec4<f32>(px * 2.0 / frame.viewport_size.xy * clip.w, 0.0, 0.0);
    out.uv = corner * vec2<f32>(r1 / sqrt(l1), r2 / sqrt(l2));

    var rgb = vec3<f32>(p2.z, p2.w, s0x4(idx * 4u + 3u).x);
    let degree = u32(u_data.data[0].y);
    if (degree > 0u) {
        // From the camera to the splat, in the frame the splats were trained
        // in: the object's own.
        let eye = splat_inverse3(m3) * (frame.camera_position.xyz - mesh.model[3].xyz);
        rgb = rgb + splat_sh(idx, degree, u32(u_data.data[1].z), normalize(local - eye));
    }
    let linear = splat_srgb_decode(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)));
    out.vertex_color = vec4<f32>(linear, alpha);
    out.view_z = -view.z;
    out.clip_zw = out.clip_pos.zw;
    return out;
}
"#
);
