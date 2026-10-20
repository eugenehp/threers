//! The three-dimensional solver, as one WGSL module.
//!
//! # Why the volume is a 2D texture
//!
//! A pass has to *write* a whole field, and a render pass writes one
//! attachment. A `texture_3d` attachment is one z-slice at a time, so a 64³
//! volume would be sixty-four render passes per stage and about eighteen
//! hundred a frame — the pass setup alone would cost more than the arithmetic.
//!
//! So the volume is flattened: the z-slices are tiled across a 2D texture, and
//! every stage is one fullscreen pass over that atlas, exactly as in two
//! dimensions. It also keeps the property the rest of this module is built
//! around — no compute stage, so it runs on WebGL2, which has no 3D storage
//! textures at all.
//!
//! The cost is that trilinear filtering is not free. `sample_grid` does it by
//! hand: two bilinear fetches from adjacent slices and a lerp, with the
//! in-tile coordinate **clamped half a texel from the tile edge**. That clamp
//! is not optional — without it a fetch near a tile boundary averages in the
//! neighbouring slice's opposite face, which looks like the fluid leaking
//! diagonally through itself and is very hard to read as an addressing bug.

/// Every stage of the 3D fluid. Pairs with [`crate::renderer::fullscreen::VS`].
pub(crate) const SHADER: &str = r#"
// One field's atlas: how big the volume is and how its slices are packed.
//
// There are three of them in the block because a pass can touch grids of
// different sizes — advecting the dye reads the velocity, and the dye is
// usually the finer of the two. Sampling one with the other's tiling is the
// kind of bug that still produces a plausible-looking plume, so each texture
// carries its own.
struct Grid {
    // n, n, n, 1/n
    size  : vec4<f32>,
    // tiles across, tiles down, and their reciprocals
    tiles : vec4<f32>,
};

struct Vol {
    // The grid this pass writes, which is what `voxel_of` walks.
    out_grid      : Grid,
    // `tex_a`'s and `tex_b`'s. Equal to `out_grid` in every pass but the two
    // that mix the velocity with the dye.
    a_grid        : Grid,
    b_grid        : Grid,
    // dt in w; xyz unused
    texel         : vec4<f32>,
    // splat centre in uvw, w = radius
    point         : vec4<f32>,
    // splat colour, or the velocity it adds
    color         : vec4<f32>,
    // dissipation, vorticity, a spare scalar, flags
    params        : vec4<f32>,
    bounds_min    : vec4<f32>,
    bounds_max    : vec4<f32>,
    // march steps, absorption, emission, exposure
    march         : vec4<f32>,
    // what is behind the volume, and w > 0.5 to keep the alpha instead
    background    : vec4<f32>,
    cam_pos       : vec4<f32>,
    inv_view_proj : mat4x4<f32>,
    // Last frame's, for the resolve to reproject through.
    prev_view_proj : mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> v : Vol;
@group(0) @binding(1) var tex_a : texture_2d<f32>;
@group(0) @binding(2) var tex_b : texture_2d<f32>;
@group(0) @binding(3) var samp  : sampler;
// Only the resolve reads this; see the layout for why it is always bound.
@group(0) @binding(4) var tex_c : texture_2d<f32>;

// ---------------------------------------------------------------------------
// The atlas
// ---------------------------------------------------------------------------

// Which voxel this fragment is, as a position in the unit cube.
//
// `uv` is the atlas; the integer part in tile space says which slice, and the
// fraction is the position within it.
fn voxel_of(uv : vec2<f32>) -> vec3<f32> {
    let g = v.out_grid;
    let t = uv * g.tiles.xy;
    let tile = floor(t);
    let z = tile.y * g.tiles.x + tile.x;
    return vec3<f32>(t - tile, (z + 0.5) * g.size.w);
}

// One z-slice of `g`, sampled bilinearly.
fn slice_at(t : texture_2d<f32>, g : Grid, uv : vec2<f32>, z : f32) -> vec4<f32> {
    // Half a texel in from the tile's edge. A bilinear fetch any closer
    // averages in whatever is packed beside this slice, which is a different
    // part of the volume entirely — it looks like the fluid leaking
    // diagonally through itself.
    let half = 0.5 * g.size.w;
    let c = clamp(uv, vec2<f32>(half), vec2<f32>(1.0 - half));
    let zi = clamp(z, 0.0, g.size.z - 1.0);
    let row = floor(zi * g.tiles.z);
    let tile = vec2<f32>(zi - row * g.tiles.x, row);
    // `textureSampleLevel`, not `textureSample`: this is reached from inside
    // the march's loop, and `textureSample` needs implicit derivatives, so
    // WGSL only allows it in uniform control flow. naga does not enforce
    // that; Tint does, and rejects the whole module — so the volume drew
    // nothing in any real WebGPU browser while WebGL2 was fine. The atlas
    // has one level, so naming it changes no pixels.
    return textureSampleLevel(t, samp, (tile + c) * g.tiles.zw, 0.0);
}

// Trilinear, by hand: the two slices either side of `p.z`, lerped.
fn sample_grid(t : texture_2d<f32>, g : Grid, p : vec3<f32>) -> vec4<f32> {
    let w = clamp(p.z * g.size.z - 0.5, 0.0, g.size.z - 1.0);
    let z0 = floor(w);
    return mix(slice_at(t, g, p.xy, z0), slice_at(t, g, p.xy, z0 + 1.0), w - z0);
}

fn sample_a(p : vec3<f32>) -> vec4<f32> {
    return sample_grid(tex_a, v.a_grid, p);
}

fn sample_b(p : vec3<f32>) -> vec4<f32> {
    return sample_grid(tex_b, v.b_grid, p);
}

// A neighbour in `tex_a`, `d` voxels of its own grid away.
fn neighbour(p : vec3<f32>, d : vec3<f32>) -> vec4<f32> {
    return sample_a(p + d * v.a_grid.size.w);
}

// ---------------------------------------------------------------------------
// Exact taps
// ---------------------------------------------------------------------------
//
// The solver reads its neighbours at *whole voxel* offsets, and a whole voxel
// of the atlas is one texel. Going through `sample_grid` for those pays for
// two filtered fetches and a lerp to arrive at a value the filtering was
// never going to change: both sets of weights come out 1 and 0. `textureLoad`
// fetches the texel. Half the memory traffic in the pressure sweep, which is
// where the solve spends most of its time, and the arithmetic is identical —
// `the_exact_tap_is_the_filtered_tap` holds the two forms to the same answer.
//
// This only applies where the grid being read is the grid being written.
// Advecting the dye reads the velocity at a dye voxel, which falls between
// velocity texels and genuinely needs the filter.

// Where voxel `c` of `g` sits in the atlas. Out-of-range voxels clamp to the
// wall, which is what the half-texel clamp in `slice_at` does for the
// filtered path.
fn atlas_texel(g : Grid, c : vec3<i32>) -> vec2<i32> {
    let n = i32(g.size.x);
    let p = clamp(c, vec3<i32>(0), vec3<i32>(n - 1));
    let across = i32(g.tiles.x);
    let row = p.z / across;
    let col = p.z - row * across;
    return vec2<i32>(col * n + p.x, row * n + p.y);
}

// This fragment's voxel. One fragment writes one texel, so its own position
// *is* the voxel index — no round trip through uv and no rounding to undo.
fn voxel_index(clip : vec2<f32>) -> vec3<i32> {
    let g = v.out_grid;
    let n = i32(g.size.x);
    let px = vec2<i32>(clip);
    let tile = px / n;
    return vec3<i32>(
        px.x - tile.x * n,
        px.y - tile.y * n,
        tile.y * i32(g.tiles.x) + tile.x,
    );
}

fn load_a(c : vec3<i32>) -> vec4<f32> {
    return textureLoad(tex_a, atlas_texel(v.a_grid, c), 0);
}

fn load_b(c : vec3<i32>) -> vec4<f32> {
    return textureLoad(tex_b, atlas_texel(v.b_grid, c), 0);
}

// The centre of this voxel in the unit cube, for the passes that need both.
fn voxel_centre(c : vec3<i32>) -> vec3<f32> {
    return (vec3<f32>(c) + 0.5) * v.out_grid.size.w;
}

// The voxel of `tex_b` containing `p`, unfiltered. Nearest, deliberately: the
// caller is asking *which cell am I in*, and an interpolated answer to that
// is not an answer.
fn load_b_at(p : vec3<f32>) -> vec4<f32> {
    let g = v.b_grid;
    let c = vec3<i32>(floor(p * g.size.x));
    return textureLoad(tex_b, atlas_texel(g, c), 0);
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

// Buoyancy: dye is hot, and hot fluid rises.
//
// Proportional to the dye *there*, which is what makes it buoyancy rather
// than gravity. A uniform upward push — which is what this was before the
// grids were separated — accelerates the empty fluid too, and the whole
// volume ends up piled against the ceiling in a rectangular slab.
//
// `tex_a` velocity, `tex_b` dye.
@fragment
fn fs_buoyancy3(i : FullscreenVertex) -> @location(0) vec4<f32> {
    let c = voxel_index(i.clip.xy);
    // The dye is on its own, finer grid, so this one is a real filtered
    // lookup rather than a tap.
    let d = sample_b(voxel_centre(c)).xyz;
    let density = max(max(d.x, d.y), d.z);
    // Along `color.xyz`, which the caller normalised: up by default, but a
    // volume has no privileged axis and the scene it sits in may not agree
    // about which way that is.
    let vel = load_a(c).xyz + v.color.xyz * (density * v.params.y * v.texel.w);
    return vec4<f32>(vel, 0.0);
}

@fragment
fn fs_fill3(i : FullscreenVertex) -> @location(0) vec4<f32> {
    return v.color;
}

// A Gaussian ball added to whatever is there. The same pass seeds both fields:
// into velocity `color` is a push, into dye it is a hue.
@fragment
fn fs_splat3(i : FullscreenVertex) -> @location(0) vec4<f32> {
    let c = voxel_index(i.clip.xy);
    let d = voxel_centre(c) - v.point.xyz;
    let blob = exp(-dot(d, d) / v.point.w);
    let base = load_a(c);
    return vec4<f32>(base.xyz + blob * v.color.xyz, 1.0);
}

// ---------------------------------------------------------------------------
// The solver
// ---------------------------------------------------------------------------

// Semi-Lagrangian advection, as in two dimensions: look back along the
// velocity for where this voxel's contents came from.
//
// `tex_a` velocity, `tex_b` the field being carried.
@fragment
fn fs_advect3(i : FullscreenVertex) -> @location(0) vec4<f32> {
    let p = voxel_of(i.uv);
    // The velocity is in *its own* voxels a second, so the step is scaled by
    // the velocity grid's texel and not by the advected field's.
    let u = sample_a(p).xyz;
    let back = p - u * (v.texel.w * v.a_grid.size.w);
    let result = sample_b(back);
    let decay = 1.0 + v.params.x * v.texel.w;
    return result / decay;
}

// Advection where the velocity *is* the field being carried, so the velocity
// at this voxel is a tap rather than a lookup. Only the back-trace, which
// lands between voxels by construction, needs the filter.
@fragment
fn fs_advect3_self(i : FullscreenVertex) -> @location(0) vec4<f32> {
    let g = voxel_index(i.clip.xy);
    let u = load_a(g).xyz;
    let back = voxel_centre(g) - u * (v.texel.w * v.a_grid.size.w);
    let result = sample_b(back);
    let decay = 1.0 + v.params.x * v.texel.w;
    return result / decay;
}

// How much the field is creating or destroying fluid at this voxel.
@fragment
fn fs_divergence3(i : FullscreenVertex) -> @location(0) vec4<f32> {
    let g = voxel_index(i.clip.xy);
    let c = load_a(g).xyz;
    var l = load_a(g + vec3<i32>(-1, 0, 0)).x;
    var r = load_a(g + vec3<i32>(1, 0, 0)).x;
    var d = load_a(g + vec3<i32>(0, -1, 0)).y;
    var u = load_a(g + vec3<i32>(0, 1, 0)).y;
    var b = load_a(g + vec3<i32>(0, 0, -1)).z;
    var f = load_a(g + vec3<i32>(0, 0, 1)).z;
    // Free-slip walls on all six faces: outside the box the normal component
    // is mirrored, so the boundary neither leaks nor drags.
    let last = i32(v.a_grid.size.x) - 1;
    if (g.x == 0) { l = -c.x; }
    if (g.x == last) { r = -c.x; }
    if (g.y == 0) { d = -c.y; }
    if (g.y == last) { u = -c.y; }
    if (g.z == 0) { b = -c.z; }
    if (g.z == last) { f = -c.z; }
    return vec4<f32>(0.5 * ((r - l) + (u - d) + (f - b)), 0.0, 0.0, 1.0);
}

// Vorticity is a *vector* in three dimensions, not a scalar: the curl of the
// velocity, which is the axis the fluid is spinning about and how fast.
@fragment
fn fs_curl3(i : FullscreenVertex) -> @location(0) vec4<f32> {
    let g = voxel_index(i.clip.xy);
    let l = load_a(g + vec3<i32>(-1, 0, 0)).xyz;
    let r = load_a(g + vec3<i32>(1, 0, 0)).xyz;
    let d = load_a(g + vec3<i32>(0, -1, 0)).xyz;
    let u = load_a(g + vec3<i32>(0, 1, 0)).xyz;
    let b = load_a(g + vec3<i32>(0, 0, -1)).xyz;
    let f = load_a(g + vec3<i32>(0, 0, 1)).xyz;
    let curl = 0.5 * vec3<f32>(
        (u.z - d.z) - (f.y - b.y),
        (f.x - b.x) - (r.z - l.z),
        (r.y - l.y) - (u.x - d.x),
    );
    return vec4<f32>(curl, 0.0);
}

// Vorticity confinement (Steinhoff & Underhill 1994), in three dimensions.
//
// Where two dimensions needed a quarter-turn of the gradient, three needs the
// cross product it was standing in for: `f = e * (N x w)`, with `N` the unit
// gradient of the vorticity's magnitude. That is the only part of the solver
// whose form genuinely changes between 2D and 3D.
//
// `tex_a` velocity, `tex_b` curl.
@fragment
fn fs_vorticity3(i : FullscreenVertex) -> @location(0) vec4<f32> {
    let g = voxel_index(i.clip.xy);
    let w = load_b(g).xyz;
    let gx = length(load_b(g + vec3<i32>(1, 0, 0)).xyz)
        - length(load_b(g + vec3<i32>(-1, 0, 0)).xyz);
    let gy = length(load_b(g + vec3<i32>(0, 1, 0)).xyz)
        - length(load_b(g + vec3<i32>(0, -1, 0)).xyz);
    let gz = length(load_b(g + vec3<i32>(0, 0, 1)).xyz)
        - length(load_b(g + vec3<i32>(0, 0, -1)).xyz);
    let grad = 0.5 * vec3<f32>(gx, gy, gz);
    let n = grad / (length(grad) + 0.0001);
    let force = cross(n, w) * v.params.y;
    var vel = load_a(g).xyz + force * v.texel.w;
    // The clamp is load-bearing for the same reason as in 2D: the field is
    // half float, and one unbounded voxel reaches infinity and then NaN,
    // which advection smears through the whole volume in a few frames.
    vel = clamp(vel, vec3<f32>(-1000.0), vec3<f32>(1000.0));
    return vec4<f32>(vel, 0.0);
}

// One Jacobi sweep of grad^2 p = div u. The seven-point stencil, so the
// divisor is six rather than two dimensions' four.
//
// `params.z` scales the pressure this sweep reads, which folds two passes
// that did almost nothing into the one that does the work:
//
// * **1** is an ordinary sweep.
// * **The `pressure` setting** starts a solve from a fraction of last
//   frame's answer. That used to be a pass of its own over the whole grid,
//   to multiply a field by a constant — and scaling the six neighbours is
//   the same thing, because the stencil is linear.
// * **0** is a coarse level starting from nothing. The whole sweep collapses
//   to `-rhs / 6`: no loads at all, and the clear-only pass that used to set
//   up for it is gone too.
//
// `tex_a` pressure, `tex_b` divergence.
@fragment
fn fs_pressure3(i : FullscreenVertex) -> @location(0) vec4<f32> {
    let g = voxel_index(i.clip.xy);
    let scale = v.params.z;
    // Uniform across the draw, so every lane takes the same side.
    if (scale == 0.0) {
        return vec4<f32>(-load_b(g).x / 6.0, 0.0, 0.0, 1.0);
    }
    let l = load_a(g + vec3<i32>(-1, 0, 0)).x;
    let r = load_a(g + vec3<i32>(1, 0, 0)).x;
    let d = load_a(g + vec3<i32>(0, -1, 0)).x;
    let u = load_a(g + vec3<i32>(0, 1, 0)).x;
    let b = load_a(g + vec3<i32>(0, 0, -1)).x;
    let f = load_a(g + vec3<i32>(0, 0, 1)).x;
    let div = load_b(g).x;
    return vec4<f32>((scale * (l + r + d + u + b + f) - div) / 6.0, 0.0, 0.0, 1.0);
}

// The projection: subtracting the pressure gradient removes exactly the part
// of the velocity that was compressing the fluid.
//
// `tex_a` pressure, `tex_b` velocity.
@fragment
fn fs_gradient3(i : FullscreenVertex) -> @location(0) vec4<f32> {
    let g = voxel_index(i.clip.xy);
    let l = load_a(g + vec3<i32>(-1, 0, 0)).x;
    let r = load_a(g + vec3<i32>(1, 0, 0)).x;
    let d = load_a(g + vec3<i32>(0, -1, 0)).x;
    let u = load_a(g + vec3<i32>(0, 1, 0)).x;
    let b = load_a(g + vec3<i32>(0, 0, -1)).x;
    let f = load_a(g + vec3<i32>(0, 0, 1)).x;
    let vel = load_b(g).xyz - v.params.z * vec3<f32>(r - l, u - d, f - b);
    return vec4<f32>(vel, 0.0);
}

// ---------------------------------------------------------------------------
// Multigrid
// ---------------------------------------------------------------------------
//
// A Jacobi sweep moves information one voxel. The pressure from a splat
// reaches the whole volume, so a 64³ grid needs on the order of 64 sweeps
// before the far side of the box knows anything happened — and in three
// dimensions it is worse than in two, because the pressure falls off as 1/r
// rather than log r and there are six neighbours sharing the correction
// instead of four. Measured: twenty sweeps leave 82% of the divergence.
//
// A V-cycle solves the smooth part of the problem on a grid where one voxel
// *is* the whole domain, and hands the answer back. The three stages below are
// all it needs; the smoother is the Jacobi sweep already here.

// What this level's pressure fails to satisfy: r = rhs - (sum - 6p).
//
// `tex_a` pressure, `tex_b` the right-hand side.
@fragment
fn fs_residual3(i : FullscreenVertex) -> @location(0) vec4<f32> {
    let g = voxel_index(i.clip.xy);
    let l = load_a(g + vec3<i32>(-1, 0, 0)).x;
    let r = load_a(g + vec3<i32>(1, 0, 0)).x;
    let d = load_a(g + vec3<i32>(0, -1, 0)).x;
    let u = load_a(g + vec3<i32>(0, 1, 0)).x;
    let b = load_a(g + vec3<i32>(0, 0, -1)).x;
    let f = load_a(g + vec3<i32>(0, 0, 1)).x;
    let c = load_a(g).x;
    let rhs = load_b(g).x;
    return vec4<f32>(rhs - (l + r + d + u + b + f - 6.0 * c), 0.0, 0.0, 1.0);
}

// The residual onto a grid of half the side, as the coarse problem's
// right-hand side.
@fragment
fn fs_restrict3(i : FullscreenVertex) -> @location(0) vec4<f32> {
    // A coarse voxel's centre lands exactly on the corner its eight fine
    // voxels share, so one trilinear fetch *is* their full-weighted average —
    // two bilinear taps rather than the eight the explicit form would cost,
    // and the same number.
    //
    // The factor of four is the change of units: the stencil is written with
    // no 1/h², so doubling the spacing quadruples what the same equation's
    // right-hand side has to be. Four in three dimensions as in two — it is
    // h², not h³.
    let p = voxel_centre(voxel_index(i.clip.xy));
    return vec4<f32>(4.0 * sample_a(p).x, 0.0, 0.0, 1.0);
}

// Add the coarse correction back into the fine pressure.
//
// The interpolation is the sampler's: reading a half-size volume at a fine
// voxel centre with a linear filter *is* trilinear prolongation, so there is
// no arithmetic here beyond the add.
//
// `tex_a` coarse correction, `tex_b` fine pressure.
@fragment
fn fs_prolong3(i : FullscreenVertex) -> @location(0) vec4<f32> {
    let g = voxel_index(i.clip.xy);
    let coarse = sample_a(voxel_centre(g)).x;
    let fine = load_b(g).x;
    return vec4<f32>(fine + coarse, 0.0, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Empty space
// ---------------------------------------------------------------------------
//
// Four fifths of a settled plume is nothing at all — measured, 80% of its 8³
// bricks and 85% of its 4³ bricks hold no dye a ray could see
// (`examples/fluid3d_bench.rs` prints it). A ray marching through that spends
// four fifths of its samples fetching zero and adding it to the total.
//
// So: a small volume holding the *largest* density in each brick, and a march
// that steps a whole brick at a time wherever that largest is nothing. It is
// built by halving twice, because sixty-four loads in one fragment is a worse
// deal than eight loads in each of two passes.

// Half the side, keeping the largest of each eight. `tex_a` the finer grid.
@fragment
fn fs_maxdown3(i : FullscreenVertex) -> @location(0) vec4<f32> {
    let c = voxel_index(i.clip.xy) * 2;
    var peak = 0.0;
    for (var k = 0; k < 8; k = k + 1) {
        let d = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        let s = load_a(c + d).xyz;
        // The same density the march uses: the brightest channel, so a
        // coloured wisp is as visible here as a white one.
        peak = max(peak, max(s.x, max(s.y, s.z)));
    }
    return vec4<f32>(peak, 0.0, 0.0, 1.0);
}

// Blend this frame's march with what previous frames left, and reject the
// history where the fluid has moved out from under it.
//
// Averaging frames is only sound while they are frames of the same thing. A
// still camera over still smoke converges; a plume that is rising means the
// history is a picture of where it *was*, and mixing it in is ghosting —
// measured, eight accumulated frames of a moving plume are thirty times
// further from the truth than one march with no history at all.
//
// So the history is clamped to the range this frame's own neighbourhood
// covers before it is mixed. Where the fluid has not moved, the history is
// inside that range and survives untouched, and the averaging happens. Where
// it has, the history is outside it and is pulled to the edge — which is the
// nearest value the scene could plausibly have here now. It is the colour
// clamp every temporal anti-aliasing resolve does, and it works here for the
// same reason: a 3x3 neighbourhood is a cheap stand-in for "what is this
// pixel allowed to be".
//
// `tex_a` this frame's march, `tex_b` the history. `params.z` is the weight.
struct Resolved {
    @location(0) history : vec4<f32>,
    @location(1) shown   : vec4<f32>,
};

@fragment
fn fs_resolve3(i : FullscreenVertex) -> Resolved {
    let px = vec2<i32>(i.clip.xy);
    let current = textureLoad(tex_a, px, 0);
    // Mean and spread of the neighbourhood, not its extremes. The march is
    // jittered, so each pixel carries noise of its own, and the min and max
    // of nine noisy samples is a box wide enough to let most of a stale
    // history through — measured, clamping to it left a moving plume eleven
    // times further from the truth than not accumulating at all. A box of
    // `k` standard deviations is the shape of the signal rather than its
    // worst two samples.
    var m1 = vec4<f32>(0.0);
    var m2 = vec4<f32>(0.0);
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let s = textureLoad(tex_a, px + vec2<i32>(x, y), 0);
            m1 = m1 + s;
            m2 = m2 + s * s;
        }
    }
    m1 = m1 / 9.0;
    m2 = m2 / 9.0;
    let sigma = sqrt(max(m2 - m1 * m1, vec4<f32>(0.0)));
    let k = v.params.w;
    // Where this pixel's picture came from, last frame.
    //
    // Without this the history only lines up while the camera is still, and
    // any movement throws it away — which is most of when a fluid is being
    // looked at. With it, the depth the march reported is unprojected to a
    // world point and projected again through the previous frame's camera,
    // which is where that point was on screen then.
    var weight = v.params.z;
    var at = vec2<i32>(px);
    let depth = textureLoad(tex_c, px, 0).x;
    if (depth > 0.0) {
        let ndc = vec2<f32>(i.uv.x * 2.0 - 1.0, 1.0 - i.uv.y * 2.0);
        let near = v.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
        let far = v.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
        let o = near.xyz / near.w;
        let dir = normalize(far.xyz / far.w - o);
        let world = o + dir * depth;
        let was = v.prev_view_proj * vec4<f32>(world, 1.0);
        if (was.w > 0.0) {
            let prev = was.xy / was.w;
            let uv = vec2<f32>(prev.x * 0.5 + 0.5, 0.5 - prev.y * 0.5);
            if (all(uv > vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0))) {
                at = vec2<i32>(uv * vec2<f32>(textureDimensions(tex_b)));
            } else {
                // It was off screen last frame, so there is no history for
                // it and this frame is all there is.
                weight = 1.0;
            }
        } else {
            weight = 1.0;
        }
    }
    let history = clamp(textureLoad(tex_b, at, 0), m1 - k * sigma, m1 + k * sigma);
    let out = mix(history, current, weight);
    return Resolved(out, out);
}

// The accumulated picture, onto the caller's target.
//
// A pass rather than a texture copy: the accumulation is a half float so it
// can hold what many frames average to, and the target is whatever the
// surface handed over.
@fragment
fn fs_blit3(i : FullscreenVertex) -> @location(0) vec4<f32> {
    return textureSampleLevel(tex_a, samp, i.uv, 0.0);
}

// ---------------------------------------------------------------------------
// The read-out
// ---------------------------------------------------------------------------
//
// The glyphs are in `super::read_out`, shared with the plane's shader —
// `params.x` is the number.
@fragment
fn fs_overlay3(i : FullscreenVertex) -> @location(0) vec4<f32> {
    let a = read_out(i.clip.xy, v.params.x);
    if (a > 0.0) {
        return vec4<f32>(1.0, 1.0, 1.0, a);
    }
    // `tex_a` is the frame-time history for this pass, not the dye.
    let bar = chart(i.clip.xy, tex_a, v.params.y, v.params.z);
    if (bar.a <= 0.0) {
        discard;
    }
    return bar;
}

// ---------------------------------------------------------------------------
// Seeing it
// ---------------------------------------------------------------------------

// Emission-absorption ray marching: the only way to look at a volume.
//
// A surface renderer asks "what is the nearest thing along this ray"; a volume
// has no nearest thing, so this walks the ray and accumulates what each step
// emits, attenuated by everything already passed through. Front to back, so
// the loop can stop once the remaining transmittance cannot change the pixel —
// which on a thick plume is most of the volume.
// What one ray came back with: the pixel, and how far along itself it saw
// what it saw.
struct MarchResult {
    colour : vec4<f32>,
    depth  : f32,
};

fn march(uv : vec2<f32>, clip : vec2<f32>) -> MarchResult {
    // The ray, from the inverse view-projection: unproject the near and far
    // plane at this pixel and the line between them is what the camera sees.
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let near = v.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let far = v.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let o = near.xyz / near.w;
    let dir = normalize(far.xyz / far.w - o);

    // Slab test against the box the volume occupies. The divisions by a zero
    // component give infinities, which min and max then handle correctly —
    // that is the whole reason this form of the test is the usual one.
    let t0 = (v.bounds_min.xyz - o) / dir;
    let t1 = (v.bounds_max.xyz - o) / dir;
    let lo = min(t0, t1);
    let hi = max(t0, t1);
    let t_in = max(max(lo.x, lo.y), max(lo.z, 0.0));
    let t_out = min(min(hi.x, hi.y), hi.z);
    if (t_out <= t_in) {
        if (v.background.w > 0.5) {
            return MarchResult(vec4<f32>(0.0), 0.0);
        }
        return MarchResult(vec4<f32>(v.background.xyz, 1.0), 0.0);
    }

    let steps = max(i32(v.march.x), 4);
    let dt = (t_out - t_in) / f32(steps);
    // Jitter the first sample by a fraction of a step. Without it every ray
    // samples at the same distances, so the sample planes line up across the
    // image and the volume comes out in stripes — wood-grain banding, the
    // classic ray-marching artefact. Interleaved gradient noise decorrelates
    // them per pixel, which turns the banding into fine noise the eye reads
    // as texture instead.
    //
    // `params.y` counts frames, and moving the jitter with it is what makes
    // accumulation worth anything: without it every frame samples at the
    // same depths and averaging a hundred of them gives back the one. The
    // golden ratio is the step that fills the gap most evenly however many
    // frames you happen to stop at.
    let ign = fract(52.9829189 * fract(dot(clip, vec2<f32>(0.06711056, 0.00583715))));
    let jitter = fract(ign + v.params.y * 0.6180339887);

    // How far a ray can advance and still be inside the brick it started in.
    // A brick is `1 / n` of the box per axis, and moving that distance along
    // *any* direction leaves at most that much on one axis — a diagonal ray
    // covers 1/sqrt(3) of it per axis — so stepping by it can never jump a
    // brick.
    //
    // `params.x` is the density a brick has to beat to be worth sampling,
    // and zero means there is no occupancy volume: skipping is off and every
    // step is taken.
    let skip_floor = v.params.x;
    let size3 = v.bounds_max.xyz - v.bounds_min.xyz;
    // One dye voxel short of a whole brick. The dye is read with a filter, so
    // a sample within half a voxel of the far face reads part of whatever is
    // in the *next* brick — which is the one case where an empty brick is not
    // empty. Landing a voxel inside the face samples it normally; the cost is
    // a quarter of the skip, and it takes the worst pixel of a 128×128 render
    // from 38 levels out to 11.
    let brick = min(size3.x, min(size3.y, size3.z));
    let skip = select(
        0.0,
        brick * (v.b_grid.size.w - v.a_grid.size.w),
        skip_floor > 0.0,
    );

    let size = v.bounds_max.xyz - v.bounds_min.xyz;
    var acc = vec3<f32>(0.0);
    var transmittance = 1.0;
    // Where the picture at this pixel came from: each sample weighted by
    // what it contributed, which is its own opacity times the transmittance
    // that reached it. A volume has no surface, but it has an expected
    // depth, and for a plume that is a few voxels inside the smoke — which
    // is the thing the eye tracks when the camera moves.
    var weighted = 0.0;
    var total = 0.0;
    var t = t_in + jitter * dt;
    // Bounded by the samples a solid volume would take plus the bricks a ray
    // can cross, because a `loop` that advances by two different amounts has
    // no trip count the compiler can see.
    let budget = steps + i32(v.b_grid.size.x) * 2;
    for (var k = 0; k < budget; k = k + 1) {
        if (t >= t_out || transmittance < 0.003) {
            break;
        }
        let pos = o + dir * t;
        let uvw = (pos - v.bounds_min.xyz) / size;
        // Nothing in this brick: step over the whole of it.
        if (skip > 0.0 && load_b_at(uvw).x < skip_floor) {
            // Forward to the next sample the ray *would* have taken, rather
            // than to wherever the skip lands. Off the lattice, two
            // neighbouring pixels that skipped a different number of bricks
            // sample at different phases, and the seam between them reads as
            // a cross-hatch at the brick's scale — the one artefact that
            // averaging frames cannot remove, because the bricks do not move.
            let steps_done = ceil((t + skip - t_in) / dt - jitter);
            t = t_in + (steps_done + jitter) * dt;
            continue;
        }
        let s = sample_a(uvw).xyz;
        // Density from the brightest channel, so coloured dye is as opaque as
        // white dye of the same intensity rather than a third as opaque.
        let density = max(max(s.x, s.y), s.z) * v.march.y;
        let a = 1.0 - exp(-density * dt);
        let contribution = transmittance * a;
        acc = acc + contribution * s * v.march.z;
        weighted = weighted + contribution * t;
        total = total + contribution;
        transmittance = transmittance * (1.0 - a);
        t = t + dt;
    }
    let lit = acc * v.march.w;
    let depth = select(0.0, weighted / total, total > 1e-5);
    if (v.background.w > 0.5) {
        // Premultiplied, for compositing over whatever is already there.
        return MarchResult(vec4<f32>(lit, 1.0 - transmittance), depth);
    }
    // Opaque, over the flat background the volume is notionally sitting in —
    // which is what a capture wants, since a PNG of premultiplied dye over
    // nothing reads as a pale smudge in anything that shows transparency.
    return MarchResult(vec4<f32>(lit + v.background.xyz * transmittance, 1.0), depth);
}

@fragment
fn fs_raymarch(i : FullscreenVertex) -> @location(0) vec4<f32> {
    return march(i.uv, i.clip.xy).colour;
}

// The march again, also reporting the depth, for the resolve to reproject
// by. One loop, two attachments — not a second march.
struct Marched {
    @location(0) colour : vec4<f32>,
    @location(1) depth  : vec4<f32>,
};

@fragment
fn fs_raymarch_depth(i : FullscreenVertex) -> Marched {
    let r = march(i.uv, i.clip.xy);
    return Marched(r.colour, vec4<f32>(r.depth, 0.0, 0.0, 1.0));
}
"#;
