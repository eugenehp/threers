//! What [`OverlayPainter`](crate::ui::OverlayPainter) works out a pixel at
//! a time, by the GPU where there is much of it: the warp of a turned box (a
//! canvas's pixels looked up through a projective map), shadows, gradients
//! and see-through images, mixed over the overlay's pixels. The results come
//! back to those pixels, so a UI drawn anywhere — a window, a texture
//! RealityKit shows — has them. `false` from [`warp`] or [`fill`] leaves it
//! to the CPU.

use std::sync::{Mutex, OnceLock};

use super::device::compile_library;
use super::objc::{error_message, msg0, msg1, msg2, msg3, sel, AutoreleasePool, Id, Owned, NIL};
use super::resources::library_function;

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Params {
    float4 inv0; float4 inv1; float4 inv2;   // rows of the inverse map
    float4 eye;                              // the map's last row
    float4 clip;                             // x0, y0, x1, y1
    float2 origin;                           // the canvas's, in its pixels
    int2 at;                                 // the piece's corner on the overlay
    uint2 size;                              // the piece's
    uint2 canvas;                            // the canvas's
    uint looks; uint rounds; uint linear; uint pad;
};

static float encode_one(float v, bool linear) {
    v = clamp(v, 0.0f, 1.0f);
    if (linear) { v = v <= 0.0031308f ? v * 12.92f : 1.055f * precise::pow(v, 1.0f / 2.4f) - 0.055f; }
    return v;
}

static float4 decode4(uchar4 v, constant float* decode) {
    const float a = float(v.a) / 255.0f;
    return float4(float3(decode[v.r], decode[v.g], decode[v.b]) * a, a);
}

/// Premultiplied `c`, as much as `cover` leaves of it, over `was`.
static uchar4 over(float4 c, float cover, uchar4 was, constant float* decode, bool lin) {
    const float sa = c.a * cover;
    const float da = float(was.a) / 255.0f;
    const float a = sa + da * (1.0f - sa);
    const float3 under = float3(decode[was.r], decode[was.g], decode[was.b]) * da * (1.0f - sa);
    const float3 rgb = (c.rgb * cover + under) / max(a, 1e-6f);
    return uchar4(uchar3(float3(encode_one(rgb.r, lin), encode_one(rgb.g, lin), encode_one(rgb.b, lin)) * 255.0f + 0.5f),
                  uchar(clamp(a, 0.0f, 1.0f) * 255.0f + 0.5f));
}

static float round_cover(float2 mid, uint n, constant float4* rounds) {
    float cover = 1.0f;
    for (uint k = 0; k < n; k++) {
        const float4 r = rounds[k * 2];
        const float radius = rounds[k * 2 + 1].x;
        const float2 d = max(max(r.xy + radius - mid, mid - (r.xy + r.zw - radius)), 0.0f);
        if (all(d > 0.0f)) { cover *= clamp(radius - length(d) + 0.5f, 0.0f, 1.0f); }
    }
    return cover;
}

// what else is filled a pixel at a time: `v` as `Fill` gives it
struct FillParams {
    float4 clip;
    int2 at; uint2 size;
    uint rounds; uint linear; uint kind; uint n;
    float4 v[5];
};

static float erf_approx(float x) {
    const float s = x < 0.0f ? -1.0f : 1.0f, a = abs(x);
    float t = 1.0f + (0.278393f + (0.230389f + 0.078108f * (a * a)) * a) * a;
    t = t * t;
    return s - s / (t * t);
}

static float shadow_cover(float4 b, float2 at, float sigma, float radius) {
    const float2 mid = (b.xy + b.zw) * 0.5f, h = (b.zw - b.xy) * 0.5f;
    const float corner = clamp(radius, 0.0f, min(h.x, h.y));
    const float2 p = at - mid;
    const float low = p.y - h.y, high = p.y + h.y;
    const float start = clamp(-3.0f * sigma, low, high), end = clamp(3.0f * sigma, low, high);
    const float step = (end - start) / 4.0f;
    const float k = 0.70710678f / sigma;
    float v = start + step * 0.5f, sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        const float y = p.y - v;
        const float delta = min(h.y - corner - abs(y), 0.0f);
        const float curved = h.x - corner + sqrt(max(corner * corner - delta * delta, 0.0f));
        const float row = 0.5f * erf_approx((p.x + curved) * k) - 0.5f * erf_approx((p.x - curved) * k);
        sum += row * exp(-(v * v) / (2.0f * sigma * sigma)) / (2.5066283f * sigma) * step;
        v += step;
    }
    return clamp(sum, 0.0f, 1.0f);
}

static float box_cover(float4 b, float r, float2 p) {
    r = clamp(r, 0.0f, min(b.z, b.w) / 2.0f);
    const float2 d = abs(p - b.xy - b.zw / 2.0f) - (b.zw / 2.0f - r);
    return clamp(0.5f - (length(max(d, 0.0f)) + min(max(d.x, d.y), 0.0f) - r), 0.0f, 1.0f);
}

static float to_linear(float c) {
    c = clamp(c, 0.0f, 1.0f);
    return c <= 0.04045f ? c / 12.92f : precise::pow((c + 0.055f) / 1.055f, 2.4f);
}

kernel void fill_pixels(
    constant FillParams& p [[buffer(0)]],
    device const uchar4* src [[buffer(1)]],
    device uchar4* dst [[buffer(2)]],
    constant float* decode [[buffer(3)]],
    constant float4* data [[buffer(4)]],     // a gradient's stops: place, then colour
    constant float4* rounds [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (any(gid >= p.size)) { return; }
    const float2 px = float2(p.at + int2(gid)), mid = px + 0.5f;
    const float2 in = clamp(min(p.clip.zw, px + 1.0f) - max(p.clip.xy, px), 0.0f, 1.0f);
    float cover = in.x * in.y;
    if (cover <= 0.0f) { return; }
    cover *= round_cover(mid, p.rounds, rounds);
    if (cover <= 0.0f) { return; }
    const bool lin = p.linear != 0;
    float4 c = 0.0f;
    if (p.kind == 0) {
        // a shadow: its box; sigma, radius, whether there is a hole; the hole; its radius; the colour
        float a = shadow_cover(p.v[0], mid, p.v[1].x, p.v[1].y) * p.v[4].a;
        if (p.v[1].z > 0.0f) { a *= 1.0f - box_cover(p.v[2], p.v[3].x, mid); }
        if (a <= 1.0f / 512.0f) { return; }
        c = float4(p.v[4].rgb * a, a);
    } else if (p.kind == 1) {
        // a gradient: its box; its shape, and that shape's numbers
        const float4 r = p.v[0], g = p.v[1];
        float t;
        if (g.x == 0.0f) {
            t = dot(mid - (r.xy + r.zw / 2.0f), g.yz) / g.w + 0.5f;
        } else if (g.x == 1.0f) {
            t = length(mid - g.yz) / g.w;
        } else {
            const float2 d = mid - g.yz;
            t = (atan2(d.x, -d.y) - g.w) / 6.28318531f;
            t = t - floor(t);
        }
        float4 s = data[(p.n - 1) * 2 + 1];
        if (t <= data[0].x) {
            s = data[1];
        } else {
            for (uint i = 0; i + 1 < p.n; i++) {
                const float t0 = data[i * 2].x, t1 = data[i * 2 + 2].x;
                if (t >= t0 && t < t1) {
                    s = mix(data[i * 2 + 1], data[i * 2 + 3], (t - t0) / (t1 - t0));
                    break;
                }
            }
        }
        if (s.a <= 0.0f) { return; }
        const float3 rgb = lin ? float3(to_linear(s.r), to_linear(s.g), to_linear(s.b)) : clamp(s.rgb, 0.0f, 1.0f);
        c = float4(rgb * s.a, s.a);
    } else {
        // an image: where it goes; its size, its opacity
        const float4 r = p.v[0];
        const float2 n = p.v[1].xy;
        const float2 q = clamp((mid - r.xy) / r.zw * n - 0.5f, 0.0f, n - 1.0f);
        const uint2 i = uint2(q);
        const uint2 j = min(i + 1, uint2(n) - 1);
        const float2 t = q - float2(i);
        const uint w = uint(n.x);
        const float4 top = mix(decode4(src[i.y * w + i.x], decode), decode4(src[i.y * w + j.x], decode), t.x);
        const float4 foot = mix(decode4(src[j.y * w + i.x], decode), decode4(src[j.y * w + j.x], decode), t.x);
        c = mix(top, foot, t.y) * p.v[1].z;
    }
    if (c.a * cover <= 0.0f) { return; }
    device uchar4& out = dst[gid.y * p.size.x + gid.x];
    out = over(c, cover, out, decode, lin);
}

kernel void warp_turned(
    constant Params& p [[buffer(0)]],
    device const uchar4* src [[buffer(1)]],
    device uchar4* dst [[buffer(2)]],
    constant float* decode [[buffer(3)]],
    constant float4* looks [[buffer(4)]],
    constant float4* rounds [[buffer(5)]],   // box, then radius in the next's x
    uint2 gid [[thread_position_in_grid]])
{
    if (any(gid >= p.size)) { return; }
    const float2 px = float2(p.at + int2(gid));
    // how much of the pixel the clip leaves
    const float2 in = clamp(min(p.clip.zw, px + 1.0f) - max(p.clip.xy, px), 0.0f, 1.0f);
    float cover = in.x * in.y;
    if (cover <= 0.0f) { return; }
    float4 c = 0.0f;
    const float2 most = float2(p.canvas) - 1.0f;
    for (uint k = 0; k < p.looks; k++) {
        const float3 s = float3(px + looks[k].xy, 1.0f);
        const float w = 1.0f / dot(p.inv2.xyz, s);
        const float2 uv = float2(dot(p.inv0.xyz, s), dot(p.inv1.xyz, s)) * w;
        const float2 q = uv - p.origin - 0.5f;
        if (!(all(q >= 0.0f) && all(q < most) && dot(p.eye.xyz, float3(uv, 1.0f)) > 0.0f)) { continue; }
        const uint2 i = uint2(q);
        const float2 t = q - float2(i);
        const uint n = i.y * p.canvas.x + i.x;
        const uint4 at = uint4(n, n + 1, n + p.canvas.x, n + p.canvas.x + 1);
        const float4 wt = float4((1 - t.x) * (1 - t.y), t.x * (1 - t.y), (1 - t.x) * t.y, t.x * t.y);
        for (uint m = 0; m < 4; m++) { c += decode4(src[at[m]], decode) * wt[m]; }
    }
    c /= float(p.looks);
    if (c.a <= 0.0f) { return; }
    cover *= round_cover(px + 0.5f, p.rounds, rounds);
    if (cover <= 0.0f) { return; }
    device uchar4& out = dst[gid.y * p.size.x + gid.x];
    out = over(c, cover, out, decode, p.linear != 0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    inv: [[f32; 4]; 3],
    eye: [f32; 4],
    clip: [f32; 4],
    origin: [f32; 2],
    at: [i32; 2],
    size: [u32; 2],
    canvas: [u32; 2],
    looks: u32,
    rounds: u32,
    linear: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Size {
    width: usize,
    height: usize,
    depth: usize,
}

/// What to warp. Places are the overlay's pixels.
pub(crate) struct Warp<'a> {
    /// From the overlay's pixels to the canvas's (3 × 3 by rows), and the
    /// last row of the map back.
    pub inverse: [f32; 9],
    pub eye: [f32; 3],
    /// The canvas: RGBA8, straight alpha, clear at its edges; its size and
    /// where its first pixel is.
    pub canvas: &'a [u8],
    pub canvas_size: (u32, u32),
    pub origin: (f32, f32),
    /// Where in a pixel to look.
    pub looks: &'a [(f32, f32)],
    /// A byte as the overlay mixes it, and whether that is linear light.
    pub decode: &'a [f32; 256],
    pub linear: bool,
    /// `x0, y0, x1, y1`, and rounded boxes (`x, y, width, height`, radius).
    pub clip: [f32; 4],
    pub rounds: &'a [([f32; 4], f32)],
    /// The piece of the overlay to draw on: its corner and size.
    pub at: (i32, i32),
    pub size: (u32, u32),
}

struct Context {
    device: Owned,
    queue: Owned,
    warp: Owned,
    fill: Owned,
    /// The canvas's and the piece's, kept (and their bytes).
    buffers: [Option<(Owned, usize)>; 2],
}

// SAFETY: a device, a queue, a pipeline state and buffers may be used from
// any thread; the mutex lets one at a time at the buffers.
unsafe impl Send for Context {}

#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> Id;
}

impl Context {
    fn new() -> Option<Self> {
        let _pool = AutoreleasePool::new();
        // SAFETY: Metal's own calls, their results checked for null.
        unsafe {
            let device = Owned::from_retained(MTLCreateSystemDefaultDevice())?;
            let queue = Owned::from_retained(msg0(device.id(), sel!("newCommandQueue")))?;
            let library = compile_library(device.id(), SOURCE)
                .map_err(|e| eprintln!("threers: no GPU painting: {e}"))
                .ok()?;
            let pipeline = |name: &str| -> Option<Owned> {
                let f = library_function(library.id(), name).ok()?;
                let mut err: Id = NIL;
                let made: Id = msg2(
                    device.id(),
                    sel!("newComputePipelineStateWithFunction:error:"),
                    f.id(),
                    &mut err as *mut Id,
                );
                let made = Owned::from_retained(made);
                if made.is_none() {
                    eprintln!("threers: no GPU painting: {}", error_message(err));
                }
                made
            };
            let (warp, fill) = (pipeline("warp_turned")?, pipeline("fill_pixels")?);
            Some(Self { device, queue, warp, fill, buffers: [None, None] })
        }
    }

    /// Buffer `n`, of `bytes` or more, shared with the CPU.
    fn buffer(&mut self, n: usize, bytes: usize) -> Option<Id> {
        if self.buffers[n].as_ref().is_none_or(|b| b.1 < bytes) {
            let len = bytes.next_power_of_two();
            // SAFETY: a new buffer, 0 its options (shared).
            let b = unsafe { Owned::from_retained(msg2(self.device.id(), sel!("newBufferWithLength:options:"), len, 0usize)) }?;
            self.buffers[n] = Some((b, len));
        }
        self.buffers[n].as_ref().map(|b| b.0.id())
    }

    fn run(&mut self, job: &Job, piece: &mut [u8]) -> Option<()> {
        let _pool = AutoreleasePool::new();
        // (nothing, or nothing to read: a byte or two all the same, for a
        // buffer of nothing cannot be bound)
        let (from, to) = (self.buffer(0, job.src.len().max(16))?, self.buffer(1, piece.len())?);
        if size_of_val(job.data) > 4096 || size_of_val(job.rounds) > 4096 {
            return None;
        }
        let pipeline = if job.warps { self.warp.id() } else { self.fill.id() };
        // SAFETY: the buffers are as long as what is copied in and out of
        // them, and the GPU is done with them before it is read; bytes set
        // are copied at once.
        unsafe {
            let p: *mut u8 = msg0(from, sel!("contents"));
            let q: *mut u8 = msg0(to, sel!("contents"));
            if p.is_null() || q.is_null() {
                return None;
            }
            std::ptr::copy_nonoverlapping(job.src.as_ptr(), p, job.src.len());
            std::ptr::copy_nonoverlapping(piece.as_ptr(), q, piece.len());
            let commands: Id = msg0(self.queue.id(), sel!("commandBuffer"));
            let enc: Id = if commands.is_null() { NIL } else { msg0(commands, sel!("computeCommandEncoder")) };
            if enc.is_null() {
                return None;
            }
            let () = msg1(enc, sel!("setComputePipelineState:"), pipeline);
            let bytes = |data: *const u8, len: usize, at: usize| {
                let () = msg3(enc, sel!("setBytes:length:atIndex:"), data, len, at);
            };
            bytes(job.params.as_ptr(), job.params.len(), 0);
            let () = msg3(enc, sel!("setBuffer:offset:atIndex:"), from, 0usize, 1usize);
            let () = msg3(enc, sel!("setBuffer:offset:atIndex:"), to, 0usize, 2usize);
            bytes(job.decode.as_ptr().cast(), size_of_val(job.decode), 3);
            bytes(job.data.as_ptr().cast(), size_of_val(job.data), 4);
            bytes(job.rounds.as_ptr().cast(), size_of_val(job.rounds), 5);
            let groups = Size {
                width: (job.size.0 as usize).div_ceil(16),
                height: (job.size.1 as usize).div_ceil(16),
                depth: 1,
            };
            let each = Size { width: 16, height: 16, depth: 1 };
            let () = msg2(enc, sel!("dispatchThreadgroups:threadsPerThreadgroup:"), groups, each);
            let () = msg0(enc, sel!("endEncoding"));
            let () = msg0(commands, sel!("commit"));
            let () = msg0(commands, sel!("waitUntilCompleted"));
            let status: usize = msg0(commands, sel!("status"));
            if status != 4 {
                return None;
            }
            std::ptr::copy_nonoverlapping(q, piece.as_mut_ptr(), piece.len());
        }
        Some(())
    }
}

/// What the GPU is given.
struct Job<'a> {
    warps: bool,
    params: &'a [u8],
    src: &'a [u8],
    decode: &'a [f32; 256],
    data: &'a [[f32; 4]],
    rounds: &'a [[f32; 4]],
    size: (u32, u32),
}

#[cfg(test)]
thread_local! {
    /// Leaves it all to the CPU, to compare with.
    pub(crate) static OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// What has been drawn here.
    pub(crate) static DRAWN: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

fn run(job: &Job, piece: &mut [u8]) -> bool {
    #[cfg(test)]
    if OFF.get() {
        return false;
    }
    static CONTEXT: OnceLock<Option<Mutex<Context>>> = OnceLock::new();
    if job.size.0 == 0 || job.size.1 == 0 || piece.len() != long(job.size) {
        return false;
    }
    let Some(context) = CONTEXT.get_or_init(|| Context::new().map(Mutex::new)) else {
        return false;
    };
    let drawn = context.lock().unwrap_or_else(|e| e.into_inner()).run(job, piece).is_some();
    #[cfg(test)]
    DRAWN.set(DRAWN.get() + u32::from(drawn));
    drawn
}

fn long((x, y): (u32, u32)) -> usize {
    x as usize * y as usize * 4
}

/// A struct's bytes, for the GPU.
fn bytes_of<T: Copy>(v: &T) -> &[u8] {
    // SAFETY: `repr(C)`, of four-byte numbers with no gaps between them.
    unsafe { std::slice::from_raw_parts((&raw const *v).cast(), size_of::<T>()) }
}

fn rounds_of(rounds: &[([f32; 4], f32)]) -> Vec<[f32; 4]> {
    let mut v: Vec<[f32; 4]> = rounds.iter().flat_map(|(r, radius)| [*r, [*radius, 0.0, 0.0, 0.0]]).collect();
    if v.is_empty() {
        v.push([0.0; 4]);
    }
    v
}

/// How much of the overlay, in pixels, is worth handing to the GPU.
pub(crate) const WORTH: f32 = 40_000.0;

#[repr(C)]
#[derive(Clone, Copy)]
struct FillParams {
    clip: [f32; 4],
    at: [i32; 2],
    size: [u32; 2],
    rounds: u32,
    linear: u32,
    kind: u32,
    n: u32,
    v: [[f32; 4]; 5],
}

/// Something filled a pixel at a time. Places are the overlay's pixels.
pub(crate) enum Fill<'a> {
    /// A box (`x0, y0, x1, y1`) blurred, none of it in `hole` (a box, `x, y,
    /// width, height`, and its radius); the colour as the overlay mixes it.
    Shadow { rect: [f32; 4], sigma: f32, radius: f32, hole: Option<([f32; 4], f32)>, color: [f32; 4] },
    /// Over `rect` (`x, y, width, height`), its stops' colours encoded.
    Gradient { rect: [f32; 4], shape: crate::ui::GradientShape, stops: &'a [(f32, crate::ui::Rgba)] },
    /// RGBA8, straight alpha, scaled to `rect`.
    Image { rect: [f32; 4], rgba: &'a [u8], size: (u32, u32), opacity: f32 },
}

/// Where a [`Fill`] goes, and how it is mixed.
pub(crate) struct Onto<'a> {
    pub clip: [f32; 4],
    pub rounds: &'a [([f32; 4], f32)],
    pub at: (i32, i32),
    pub size: (u32, u32),
    pub decode: &'a [f32; 256],
    pub linear: bool,
}

/// Fills `piece` (the overlay's pixels at `onto`): whether it did.
pub(crate) fn fill(what: &Fill, onto: &Onto, piece: &mut [u8]) -> bool {
    use crate::ui::GradientShape as G;
    let mut v = [[0.0f32; 4]; 5];
    let (mut data, mut src): (Vec<[f32; 4]>, &[u8]) = (vec![[0.0; 4]], &[]);
    let (kind, n) = match *what {
        Fill::Shadow { rect, sigma, radius, hole, color } => {
            v[0] = rect;
            v[1] = [sigma, radius, 0.0, 0.0];
            if let Some((h, r)) = hole {
                v[1][2] = 1.0;
                v[2] = h;
                v[3][0] = r;
            }
            v[4] = color;
            (0, 0)
        }
        Fill::Gradient { rect, shape, stops } => {
            // (what `GradientShape::t` works out for every pixel alike)
            let [x, y, w, h] = rect;
            let at = |cx: f32, cy: f32| (x + w * cx, y + h * cy);
            v[0] = rect;
            v[1] = match shape {
                G::Linear(deg) => {
                    let a = deg.to_radians();
                    let (dx, dy) = (a.sin(), -a.cos());
                    [0.0, dx, dy, ((w * dx).abs() + (h * dy).abs()).max(1e-6)]
                }
                G::Radial { cx, cy } => {
                    let (px, py) = at(cx, cy);
                    let far = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
                        .iter()
                        .map(|&(qx, qy)| (qx - px).hypot(qy - py))
                        .fold(0.0f32, f32::max);
                    [1.0, px, py, far.max(1e-6)]
                }
                G::Conic { cx, cy, from } => {
                    let (px, py) = at(cx, cy);
                    [2.0, px, py, from.to_radians()]
                }
            };
            if stops.is_empty() {
                return false;
            }
            data = stops.iter().flat_map(|(t, c)| [[*t, 0.0, 0.0, 0.0], *c]).collect();
            (1, stops.len() as u32)
        }
        Fill::Image { rect, rgba, size, opacity } => {
            if rgba.len() != long(size) || rgba.is_empty() {
                return false;
            }
            v[0] = rect;
            v[1] = [size.0 as f32, size.1 as f32, opacity, 0.0];
            src = rgba;
            (2, 0)
        }
    };
    let rounds = rounds_of(onto.rounds);
    let params = FillParams {
        clip: onto.clip,
        at: [onto.at.0, onto.at.1],
        size: [onto.size.0, onto.size.1],
        rounds: onto.rounds.len() as u32,
        linear: u32::from(onto.linear),
        kind,
        n,
        v,
    };
    let job = Job {
        warps: false,
        params: bytes_of(&params),
        src,
        decode: onto.decode,
        data: &data,
        rounds: &rounds,
        size: onto.size,
    };
    run(&job, piece)
}

/// Warps `w` over `piece` (the overlay's pixels there, `w.size`, RGBA8):
/// whether it did.
pub(crate) fn warp(w: &Warp, piece: &mut [u8]) -> bool {
    if w.canvas.len() != long(w.canvas_size) || w.looks.is_empty() {
        return false;
    }
    let row = |i: usize| [w.inverse[i], w.inverse[i + 1], w.inverse[i + 2], 0.0];
    let params = Params {
        inv: [row(0), row(3), row(6)],
        eye: [w.eye[0], w.eye[1], w.eye[2], 0.0],
        clip: w.clip,
        origin: [w.origin.0, w.origin.1],
        at: [w.at.0, w.at.1],
        size: [w.size.0, w.size.1],
        canvas: [w.canvas_size.0, w.canvas_size.1],
        looks: w.looks.len() as u32,
        rounds: w.rounds.len() as u32,
        linear: u32::from(w.linear),
        pad: 0,
    };
    let looks: Vec<[f32; 4]> = w.looks.iter().map(|l| [l.0, l.1, 0.0, 0.0]).collect();
    let rounds = rounds_of(w.rounds);
    let job = Job {
        warps: true,
        params: bytes_of(&params),
        src: w.canvas,
        decode: w.decode,
        data: &looks,
        rounds: &rounds,
        size: w.size,
    };
    run(&job, piece)
}
