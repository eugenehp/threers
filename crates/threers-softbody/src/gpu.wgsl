// The soft body's substep on the GPU: integration, the hand, the lattice's
// edges and volumes by the parity classes of its cubes (cubes alike in parity
// share no node), the glue, the floor and walls, and damping toward each
// block's mean velocity. Fifteen dispatches a substep; see gpu.rs.
//
// Two storage buffers, under the four of WebGPU's downlevel floor: `state`
// (positions, previous positions, velocities, block means) and `data`, every
// constant and per-frame table as 32-bit words in sections the frame names.

const MAX_BLOCKS: u32 = 64u;

struct Frame {
    dt: f32,
    gravity: f32,
    alpha_edge: f32,     // edge compliance / dt², for the body (parts take 0.3 of it)
    alpha_vol: f32,      // volume compliance / dt²
    alpha_glue: f32,
    damp_k: f32,         // exp(−damping · dt)
    time: f32,
    n: u32,              // particles
    walls_on: u32,
    grab_n: u32,
    grab_cos: f32,
    grab_sin: f32,
    grab_target: vec4<f32>,
    grab_axis: vec4<f32>,
    walls: array<vec4<f32>, 4>,
    // where each table starts in `data`, in words
    node_at: u32,        // 2 a node: inverse mass, block
    cell_at: u32,        // 4 a cube of the lattice: block, first tet, first edge, edges
    tet_at: u32,         // 5 a tetrahedron: four nodes, rest volume
    edge_at: u32,        // 5 an edge: a, b, body edge (or ~0), rest, phase
    link_at: u32,        // 8 a glue group: part node, links, then 3 body nodes and 3 links
    grab_at: u32,        // 5 a held node: node, weight, offset
    muscle_at: u32,      // 1 a body edge: its rest-length change
    glue_at: u32,        // 1 a glue link: its rest length
    links: u32,          // glue groups
    body_n: u32,         // the body's particles
    _pad0: u32,
    _pad1: u32,
};
struct Block {
    off: u32,
    n: u32,
    alive: u32,
    leg: i32,            // −1 the body
    scale: f32,
    twitch: f32,
    alpha_mul: f32,      // 1 for the body, 0.3 for parts
    _pad: f32,
};
struct Blocks { b: array<Block, MAX_BLOCKS> };
struct Range { start: u32, count: u32 };

@group(0) @binding(0) var<uniform> f: Frame;
@group(0) @binding(1) var<uniform> bl: Blocks;
// [0, n) positions · [n, 2n) previous · [2n, 3n) velocities · [3n, 3n + blocks) means
@group(0) @binding(2) var<storage, read_write> state: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> data: array<u32>;
// the glue's pull on the body, summed in fixed point (see `solve_glue`)
@group(0) @binding(4) var<storage, read_write> pull: array<atomic<i32>>;
// which colour batch a constraint dispatch covers (a dynamic offset)
@group(1) @binding(0) var<uniform> range: Range;

fn word(i: u32) -> u32 { return data[i]; }
fn real(i: u32) -> f32 { return bitcast<f32>(data[i]); }
fn pos(i: u32) -> vec3<f32> { return state[i].xyz; }
fn set_pos(i: u32, p: vec3<f32>) { state[i] = vec4<f32>(p, 0.0); }
fn w(i: u32) -> f32 { return real(f.node_at + 2u * i); }
fn block_of(i: u32) -> u32 { return word(f.node_at + 2u * i + 1u); }
fn live(i: u32) -> bool { return bl.b[block_of(i)].alive != 0u; }

@compute @workgroup_size(128)
fn integrate(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= f.n || !live(i)) { return; }
    var v = state[2u * f.n + i];
    v.y = v.y - f.gravity * f.dt;
    state[2u * f.n + i] = v;
    state[f.n + i] = state[i];
    set_pos(i, pos(i) + v.xyz * f.dt);
}

@compute @workgroup_size(64)
fn grab(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x;
    if (k >= f.grab_n) { return; }
    let at = f.grab_at + 5u * k;
    let i = word(at);
    let o = vec3<f32>(real(at + 2u), real(at + 3u), real(at + 4u));
    let a = f.grab_axis.xyz;
    let d = dot(a, o);
    let r = o * f.grab_cos + cross(a, o) * f.grab_sin + a * d * (1.0 - f.grab_cos);
    set_pos(i, pos(i) + (f.grab_target.xyz + r - pos(i)) * (0.5 * real(at + 1u)));
}

// One parity class of the lattice's cubes: cubes whose grid coordinates are
// alike in parity share no node, so each thread takes a cube and solves its
// own edges, then its five tetrahedra, in order — Gauss-Seidel within it.
@compute @workgroup_size(64)
fn solve_cells(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= range.count) { return; }
    let at = f.cell_at + 4u * (range.start + id.x);
    let b = bl.b[word(at)];
    if (b.alive == 0u) { return; }
    let e0 = word(at + 2u);
    let ne = word(at + 3u);
    for (var e = 0u; e < ne; e = e + 1u) {
        edge(f.edge_at + 5u * (e0 + e), b);
    }
    let t0 = word(at + 1u);
    for (var t = 0u; t < 5u; t = t + 1u) {
        tet(f.tet_at + 5u * (t0 + t), b);
    }
}

fn edge(at: u32, b: Block) {
    let qa = word(at);
    let qb = word(at + 1u);
    let wa = w(qa);
    let wc = w(qb);
    let ws = wa + wc;
    let d = pos(qa) - pos(qb);
    let len = length(d);
    if (len == 0.0 || ws == 0.0) { return; }
    var factor = 1.0;
    if (b.leg < 0) {
        factor = 1.0 + real(f.muscle_at + word(at + 2u));
    } else if (b.twitch != 0.0) {
        factor = 1.0 + b.twitch * sin(f.time * 17.0 + real(at + 4u)) * 0.22;
    }
    let rest = real(at + 3u) * b.scale * factor;
    let k = -(len - rest) / (ws + f.alpha_edge * b.alpha_mul) / len;
    set_pos(qa, pos(qa) + d * k * wa);
    set_pos(qb, pos(qb) - d * k * wc);
}

fn tet(at: u32, b: Block) {
    let n = vec4<u32>(word(at), word(at + 1u), word(at + 2u), word(at + 3u));
    let p0 = pos(n.x);
    let a = pos(n.y) - p0;
    let bb = pos(n.z) - p0;
    let c = pos(n.w) - p0;
    let g1 = cross(bb, c) / 6.0;
    let g2 = cross(c, a) / 6.0;
    let g3 = cross(a, bb) / 6.0;
    let g0 = -(g1 + g2 + g3);
    let vol = dot(a, g1);
    let w0 = w(n.x);
    let w1 = w(n.y);
    let w2 = w(n.z);
    let w3 = w(n.w);
    let ws = w0 * dot(g0, g0) + w1 * dot(g1, g1) + w2 * dot(g2, g2) + w3 * dot(g3, g3);
    if (ws == 0.0) { return; }
    let k = -(vol - real(at + 4u) * b.scale * b.scale * b.scale) / (ws + f.alpha_vol);
    set_pos(n.x, pos(n.x) + g0 * k * w0);
    set_pos(n.y, pos(n.y) + g1 * k * w1);
    set_pos(n.z, pos(n.z) + g2 * k * w2);
    set_pos(n.w, pos(n.w) + g3 * k * w3);
}

// Glue, two passes for all of it. A thread owns a part node and solves its
// links in order, moving the part node as Gauss-Seidel would; what each link
// does to its body node is summed and applied after. (Averaging the sum over
// the links a body node is in made the glue softer than the CPU solver's —
// a tentacle tore at frame 25 against 35; the plain sum tears at 39 and
// leaves the body within 0.02 mm of the CPU's.)
const PULL_SCALE: f32 = 1073741824.0;   // 2^30: a correction is far under a unit

@compute @workgroup_size(64)
fn solve_glue(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= f.links) { return; }
    let at = f.link_at + 8u * id.x;
    let qa = word(at);
    let wa = w(qa);
    var pa = pos(qa);
    for (var l = 0u; l < word(at + 1u); l = l + 1u) {
        let qb = word(at + 2u + l);
        let wc = w(qb);
        let d = pa - pos(qb);
        let len = length(d);
        if (len == 0.0) { continue; }
        let k = -(len - real(f.glue_at + word(at + 5u + l))) / (wa + wc + f.alpha_glue) / len;
        pa = pa + d * k * wa;
        let push = -d * k * wc;
        atomicAdd(&pull[3u * qb], i32(round(push.x * PULL_SCALE)));
        atomicAdd(&pull[3u * qb + 1u], i32(round(push.y * PULL_SCALE)));
        atomicAdd(&pull[3u * qb + 2u], i32(round(push.z * PULL_SCALE)));
    }
    set_pos(qa, pa);
}

@compute @workgroup_size(128)
fn apply_glue(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= f.body_n) { return; }
    let p = vec3<f32>(f32(atomicExchange(&pull[3u * i], 0)), f32(atomicExchange(&pull[3u * i + 1u], 0)), f32(atomicExchange(&pull[3u * i + 2u], 0)));
    if (any(p != vec3<f32>(0.0))) {
        set_pos(i, pos(i) + p / PULL_SCALE);
    }
}

@compute @workgroup_size(128)
fn contact(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= f.n || !live(i)) { return; }
    var p = pos(i);
    var pp = state[f.n + i].xyz;
    if (p.y < 0.0) {
        let pen = -p.y;
        p.y = 0.0;
        let t = vec2<f32>(p.x - pp.x, p.z - pp.z);
        let tl = length(t);
        if (tl > 0.0) {
            let k = min(1.0, 0.7 * pen / tl + 0.35);
            p.x = p.x - t.x * k;
            p.z = p.z - t.y * k;
        }
    }
    if (f.walls_on != 0u) {
        for (var k = 0u; k < 4u; k = k + 1u) {
            let pl = f.walls[k];
            let s = dot(pl.xyz, p) + pl.w;
            if (s < 0.0) {
                let m = min(-s, 0.006);
                p = p + pl.xyz * m;
                pp = pp + pl.xyz * m * 0.5;
            }
        }
        state[f.n + i] = vec4<f32>(pp, 0.0);
    }
    set_pos(i, p);
    // and the velocity the substep's move implies
    state[2u * f.n + i] = vec4<f32>((p - pp) / f.dt, 0.0);
}

// each block's mean velocity: one workgroup a block
var<workgroup> partial: array<vec4<f32>, 128>;

@compute @workgroup_size(128)
fn block_means(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let b = bl.b[wg.x];
    var s = vec4<f32>(0.0);
    if (b.alive != 0u) {
        for (var i = li; i < b.n; i = i + 128u) {
            s = s + vec4<f32>(state[2u * f.n + b.off + i].xyz, 0.0);
        }
    }
    partial[li] = s;
    workgroupBarrier();
    for (var k = 64u; k > 0u; k = k >> 1u) {
        if (li < k) { partial[li] = partial[li] + partial[li + k]; }
        workgroupBarrier();
    }
    if (li == 0u) { state[3u * f.n + wg.x] = partial[0] / f32(max(b.n, 1u)); }
}

@compute @workgroup_size(128)
fn damp(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= f.n || !live(i)) { return; }
    let m = state[3u * f.n + block_of(i)].xyz;
    let v = state[2u * f.n + i].xyz;
    state[2u * f.n + i] = vec4<f32>(m + (v - m) * f.damp_k, 0.0);
}
