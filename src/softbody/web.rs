//! [`SoftBody`] for JavaScript (the shim's `THREE.SoftBody`).
//!
//! Built with `WebSoftBuilder` from `WebSdf` shapes; driven a frame at a time;
//! read back as flat arrays. Skinning is here too: a skin, or any other points
//! bound to a part's lattice, is carried along in Rust and handed back as
//! positions and normals, so the page never touches the lattice.

use wasm_bindgen::prelude::*;

use super::creature::{ChainSpec, Creature, Marks, ShapeSpec};
use threers_softbody::gpu::GpuSolver;
use threers_softbody::{Embedding, Event, Hit, Muscles, Pulse, SoftBody};
use crate::sdf::WebSdf;
use crate::sdf::SurfaceNets;

/// Collects a creature's shapes, then builds it.
#[wasm_bindgen]
pub struct WebSoftBuilder {
    body: ShapeSpec,
    parts: Vec<(String, ShapeSpec)>,
    chains: Vec<ChainSpec>,
    marks: Marks,
}

#[wasm_bindgen]
impl WebSoftBuilder {
    /// The body: its shape, the skin's surface-nets cell, the lattice spacing.
    #[wasm_bindgen(constructor)]
    pub fn new(body: &WebSdf, cell: f32, h: f64) -> WebSoftBuilder {
        WebSoftBuilder {
            body: ShapeSpec { shape: body.shape().clone(), cell, h },
            parts: Vec::new(),
            chains: Vec::new(),
            marks: Marks { head: [[0.0; 3]; 3], span: [[0.0; 3]; 2] },
        }
    }
    /// A part that can tear away; it is glued where it reaches into the body.
    #[wasm_bindgen(js_name = addPart)]
    pub fn add_part(&mut self, name: String, shape: &WebSdf, cell: f32, h: f64) {
        self.parts.push((name, ShapeSpec { shape: shape.shape().clone(), cell, h }));
    }
    /// A chain of particles: rest points `[x, y, z, …]` from the root (in the
    /// body) to the tip, radius `r0` to `r1`, and how firmly it keeps shape.
    #[wasm_bindgen(js_name = addChain)]
    pub fn add_chain(&mut self, points: &[f64], r0: f64, r1: f64, stiff: f64) {
        let points = points.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
        self.chains.push(ChainSpec { points, r0, r1, stiff });
    }
    /// Three rest points for the head's frame, then two for the size readout.
    #[wasm_bindgen(js_name = setMarks)]
    pub fn set_marks(&mut self, m: &[f64]) -> Result<(), JsError> {
        if m.len() != 15 {
            return Err(JsError::new("setMarks: three head points and two span points, 15 numbers"));
        }
        let p = |i: usize| [m[3 * i], m[3 * i + 1], m[3 * i + 2]];
        self.marks = Marks { head: [p(0), p(1), p(2)], span: [p(3), p(4)] };
        Ok(())
    }
    /// Build it; `seed` seeds the goo's randomness.
    pub fn build(&self, seed: u32) -> Result<WebSoftBody, JsError> {
        let c = Creature::from_shapes(&self.body, &self.parts, &self.chains, self.marks).map_err(|e| JsError::new(&e))?;
        let skins = c.skins.clone();
        let mut bindings = Vec::new();
        for (i, s) in skins.iter().enumerate() {
            let emb = if i == 0 { c.body.skin.clone() } else { c.limbs[i - 1].part.skin.clone() };
            bindings.push(Binding::new(i, emb, flat(&s.normal)));
        }
        Ok(WebSoftBody { sim: c.into_soft_body(seed), skins, bindings, gpu: None, gpu_on: false })
    }
}

fn flat(v: &[crate::math::Vector3]) -> Vec<f32> {
    v.iter().flat_map(|p| [p.x, p.y, p.z]).collect()
}

/// Points bound to one part's lattice (`topo` 0 the body, `1 + i` part `i`):
/// the tetrahedron each lies in and its barycentric weights there, with rest
/// normals if they are a surface.
struct Binding {
    topo: usize,
    emb: Embedding,
    rest_normal: Vec<f32>,
}

impl Binding {
    fn new(topo: usize, emb: Embedding, rest_normal: Vec<f32>) -> Binding {
        Binding { topo, emb, rest_normal }
    }
}

/// A running soft body.
#[wasm_bindgen]
pub struct WebSoftBody {
    sim: SoftBody,
    skins: Vec<SurfaceNets>,
    bindings: Vec<Binding>,
    /// The GPU solver, once made; kept while the CPU one runs so that
    /// switching back is an upload, not a rebuild.
    gpu: Option<GpuSolver>,
    gpu_on: bool,
}

#[wasm_bindgen]
impl WebSoftBody {
    pub fn step(&mut self, dt: f64) {
        match &mut self.gpu {
            Some(g) if self.gpu_on => g.step(&mut self.sim, dt),
            _ => self.sim.step(dt),
        }
    }
    pub fn reset(&mut self) {
        self.sim.reset();
        if let (Some(g), true) = (&mut self.gpu, self.gpu_on) {
            g.upload_all(&self.sim);
        }
    }
    /// Switch between the solvers on `device`/`queue` (made on first use).
    fn switch_gpu(&mut self, on: bool, make: impl FnOnce() -> (std::sync::Arc<wgpu::Device>, std::sync::Arc<wgpu::Queue>)) {
        if on == self.gpu_on {
            return;
        }
        match (&mut self.gpu, on) {
            (None, true) => {
                let (d, q) = make();
                self.gpu = Some(GpuSolver::new(d, q, &mut self.sim));
            }
            (Some(g), true) => g.resume(&mut self.sim),
            (Some(g), false) => g.suspend(&mut self.sim),
            (None, false) => {}
        }
        self.gpu_on = on;
    }
    /// How many frames the GPU solver's readbacks are behind (0 off it).
    #[wasm_bindgen(getter, js_name = gpuLag)]
    pub fn gpu_lag(&self) -> usize {
        match &self.gpu {
            Some(g) if self.gpu_on => g.lag(),
            _ => 0,
        }
    }
    /// Solve the lattice on `renderer`'s GPU from now on (WebGPU compute; the
    /// positions come back a frame later), or, with `on` false, on the CPU.
    #[wasm_bindgen(js_name = useGpu)]
    pub fn use_gpu(&mut self, renderer: &crate::wasm::WebRenderer, on: bool) {
        self.switch_gpu(on, || renderer.device_queue());
    }
    /// [`use_gpu`](Self::use_gpu) on a device of its own — for a worker,
    /// which has no renderer to share one with.
    #[wasm_bindgen(js_name = useGpuDevice)]
    pub fn use_gpu_device(&mut self, device: &WebComputeDevice, on: bool) {
        self.switch_gpu(on, || (device.device.clone(), device.queue.clone()));
    }
    #[wasm_bindgen(getter, js_name = onGpu)]
    pub fn on_gpu(&self) -> bool {
        self.gpu_on
    }
    pub fn flick(&mut self) {
        self.sim.flick();
    }
    pub fn poke(&mut self) {
        self.sim.poke();
    }
    #[wasm_bindgen(js_name = setParams)]
    pub fn set_params(&mut self, firmness: f64, damping: f64, regrow: f64) {
        let p = &mut self.sim.params;
        (p.firmness, p.damping, p.regrow) = (firmness, damping, regrow);
    }
    #[wasm_bindgen(js_name = setSubsteps)]
    pub fn set_substeps(&mut self, n: usize) {
        self.sim.params.substeps = n.max(1);
    }
    /// Four planes `(nx, ny, nz, d)` the particles are kept inside; empty for none.
    #[wasm_bindgen(js_name = setWalls)]
    pub fn set_walls(&mut self, w: &[f32]) {
        self.sim.walls = (w.len() == 16).then(|| w.try_into().unwrap());
    }
    /// How far part `i`'s glue stretches before a held part tears (0.07).
    #[wasm_bindgen(js_name = setTearLimit)]
    pub fn set_tear_limit(&mut self, part: usize, limit: f64) {
        if let Some(l) = self.sim.tear_limit.get_mut(part) {
            *l = limit;
        }
    }
    /// Muscles: each body edge shortens by `weight · pulse` — see `Pulse`.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = setPulse)]
    pub fn set_pulse(&mut self, weight: Vec<f32>, phase: Vec<f32>, a: f64, calm: f64, omega_a: f64, phase_a: f64, power_a: f64, b: f64, omega_b: f64, phase_b: f64, power_b: f64) {
        let pulse = Pulse { a, calm, omega_a, phase_a, power_a, b, omega_b, phase_b, power_b };
        self.sim.muscles = Muscles::Pulse { weight, phase, pulse };
    }
    /// Muscles: the travelling sway down a tail toward −x (the default).
    #[wasm_bindgen(js_name = setSway)]
    pub fn set_sway(&mut self) {
        self.sim.muscles = Muscles::Sway;
    }
    /// No muscles: it only wobbles.
    #[wasm_bindgen(js_name = setStill)]
    pub fn set_still(&mut self) {
        self.sim.muscles = Muscles::None;
    }
    /// The body's edges at rest, as midpoints and unit directions
    /// `[mx, my, mz, dx, dy, dz, …]`, for weighing muscles.
    #[wasm_bindgen(js_name = bodyEdges)]
    pub fn body_edges(&self) -> Vec<f32> {
        let t = &self.sim.topos[0];
        let mut out = Vec::with_capacity(t.rest_len.len() * 6);
        for e in 0..t.rest_len.len() {
            let (a, b, p, l) = (3 * t.edges[2 * e] as usize, 3 * t.edges[2 * e + 1] as usize, &t.rest, t.rest_len[e]);
            for c in 0..3 {
                out.push((p[a + c] + p[b + c]) / 2.0);
            }
            for c in 0..3 {
                out.push((p[b + c] - p[a + c]) / l);
            }
        }
        out
    }

    // ── what there is ──
    #[wasm_bindgen(getter, js_name = partCount)]
    pub fn part_count(&self) -> usize {
        self.sim.limbs.len()
    }
    #[wasm_bindgen(js_name = partName)]
    pub fn part_name(&self, i: usize) -> String {
        self.sim.limbs.get(i).map_or(String::new(), |l| l.name.clone())
    }
    #[wasm_bindgen(getter, js_name = blockCount)]
    pub fn block_count(&self) -> usize {
        self.sim.blocks.len()
    }
    /// Per block: `[active, part (−1 the body), slot, grow, scale, strain, twitch]`.
    pub fn blocks(&self) -> Vec<f64> {
        self.sim
            .blocks
            .iter()
            .flat_map(|b| [b.active as u8 as f64, b.leg as f64, b.slot as f64, b.grow, b.scale, b.strain, b.twitch])
            .collect()
    }
    /// The skin of the body (0) or part `i` (`1 + i`), at rest.
    #[wasm_bindgen(js_name = skinPositions)]
    pub fn skin_positions(&self, topo: usize) -> Vec<f32> {
        flat(&self.skins[topo].position)
    }
    #[wasm_bindgen(js_name = skinNormals)]
    pub fn skin_normals(&self, topo: usize) -> Vec<f32> {
        flat(&self.skins[topo].normal)
    }
    #[wasm_bindgen(js_name = skinIndex)]
    pub fn skin_index(&self, topo: usize) -> Vec<u32> {
        self.skins[topo].index.clone()
    }

    // ── carrying points along ──
    /// Bind points `[x, y, z, …]` (with rest normals, or empty) to the body's
    /// (0) or part `i`'s (`1 + i`) lattice; returns a handle for `carry`.
    /// The skins are bound already: handle `t` is skin `t`.
    pub fn bind(&mut self, topo: usize, points: &[f32], normals: Vec<f32>) -> usize {
        let emb = if topo == 0 { self.sim.body.embed(points) } else { self.sim.limbs[topo - 1].part.embed(points) };
        self.bindings.push(Binding::new(topo, emb, normals));
        self.bindings.len() - 1
    }
    /// Where bound points are now on `block`, into `pos` (and their normals
    /// into `nrm`, when it is not empty), each normal carried by the cofactor
    /// of the deformation gradient there.
    pub fn carry(&self, binding: usize, block: usize, pos: &mut [f32], nrm: &mut [f32]) {
        let b = &self.bindings[binding];
        threers_softbody::carry(&self.sim, b.topo, &b.emb, &b.rest_normal, block, pos, nrm);
    }

    // ── particles ──
    /// Every particle's position, `[x, y, z, …]`, by block offset.
    pub fn positions(&self) -> Vec<f32> {
        self.sim.x.clone()
    }
    /// The chains' particles.
    #[wasm_bindgen(js_name = chainPositions)]
    pub fn chain_positions(&self) -> Vec<f32> {
        self.sim.gx.clone()
    }
    /// Per chain segment (child particle with a parent):
    /// `[ax, ay, az, ra, bx, by, bz, rb, …]`, at rest when `rest` is true.
    pub fn segments(&self, rest: bool) -> Vec<f32> {
        let c = &self.sim.chains;
        let g = if rest { &c.rest } else { &self.sim.gx };
        let mut out = Vec::new();
        for i in 0..c.parent.len() {
            let p = c.parent[i];
            if p < 0 {
                continue;
            }
            let p = p as usize;
            out.extend([g[3 * p], g[3 * p + 1], g[3 * p + 2], c.seg_radius[i], g[3 * i], g[3 * i + 1], g[3 * i + 2], c.radius[i]]);
        }
        out
    }
    /// The goo: `[x, y, z, r, …]`.
    pub fn drops(&self) -> Vec<f32> {
        (0..self.sim.d_count).flat_map(|i| [self.sim.dp[3 * i], self.sim.dp[3 * i + 1], self.sim.dp[3 * i + 2], self.sim.dr[i]]).collect()
    }
    /// The lattice edges of every active block, as segment endpoints.
    #[wasm_bindgen(js_name = latticeLines)]
    pub fn lattice_lines(&self) -> Vec<f32> {
        let (s, x) = (&self.sim, &self.sim.x);
        let mut out = Vec::new();
        for b in s.blocks.iter().filter(|b| b.active) {
            for e in s.topos[b.topo].edges.iter() {
                let i = b.off + *e as usize;
                out.extend([x[3 * i], x[3 * i + 1], x[3 * i + 2]]);
            }
        }
        out
    }

    // ── hands ──
    /// `[kind, which, t, x, y, z]`: kind 0 a block (which = block), 1 a drop
    /// (which = drop), −1 nothing.
    pub fn raycast(&self, ox: f64, oy: f64, oz: f64, dx: f64, dy: f64, dz: f64) -> Vec<f64> {
        match self.sim.raycast([ox, oy, oz], [dx, dy, dz]) {
            Some(Hit::Part { block, t, point }) => vec![0.0, block as f64, t, point[0], point[1], point[2]],
            Some(Hit::Drop { index, t, point }) => vec![1.0, index as f64, t, point[0], point[1], point[2]],
            None => vec![-1.0],
        }
    }
    /// Take hold of `block` at a point; `axis` is what twisting turns about.
    #[wasm_bindgen(js_name = grab)]
    #[allow(clippy::too_many_arguments)]
    pub fn grab(&mut self, block: usize, x: f64, y: f64, z: f64, ax: f64, ay: f64, az: f64) {
        self.sim.start_grab([x, y, z], block);
        if let Some(g) = &mut self.sim.grab {
            g.axis = [ax, ay, az];
        }
    }
    #[wasm_bindgen(js_name = moveGrab)]
    pub fn move_grab(&mut self, x: f64, y: f64, z: f64) {
        if let Some(g) = &mut self.sim.grab {
            g.target = [x, y, z];
        }
    }
    #[wasm_bindgen(js_name = twistGrab)]
    pub fn twist_grab(&mut self, angle: f64) {
        if let Some(g) = &mut self.sim.grab {
            g.angle = angle;
        }
    }
    /// The held block's index, its twist, and its glue strain;
    /// `[-1]` when nothing is held.
    #[wasm_bindgen(js_name = held)]
    pub fn held(&self) -> Vec<f64> {
        self.sim.grab.as_ref().map_or(vec![-1.0], |g| vec![g.block as f64, g.angle, self.sim.blocks[g.block].strain])
    }
    pub fn release(&mut self) {
        self.sim.grab = None;
    }
    #[wasm_bindgen(js_name = grabDrop)]
    pub fn grab_drop(&mut self, i: usize, x: f64, y: f64, z: f64) {
        self.sim.drop_grab = Some((i, [x, y, z]));
    }
    #[wasm_bindgen(js_name = moveDrop)]
    pub fn move_drop(&mut self, x: f64, y: f64, z: f64) {
        if let Some((_, t)) = &mut self.sim.drop_grab {
            *t = [x, y, z];
        }
    }
    /// Let go of a held drop; returns how fast it was moving.
    #[wasm_bindgen(js_name = releaseDrop)]
    pub fn release_drop(&mut self) -> f64 {
        let Some((i, _)) = self.sim.drop_grab.take() else { return 0.0 };
        let v = &self.sim.dv;
        ((v[3 * i] as f64).powi(2) + (v[3 * i + 1] as f64).powi(2) + (v[3 * i + 2] as f64).powi(2)).sqrt()
    }

    // ── readouts ──
    /// `[volume, length, attached, growing (−1 none), loose]`.
    pub fn measure(&self) -> Vec<f64> {
        let m = self.sim.measure();
        vec![m.volume, m.length, m.attached as f64, m.growing.unwrap_or(-1.0), m.loose as f64]
    }
    /// What happened since the last call: `[kind, part, r, speed, …]`, kind
    /// 0 tick, 1 plop, 2 plip, 3 tear, 4 regrown.
    pub fn events(&mut self) -> Vec<f64> {
        self.sim
            .events
            .drain(..)
            .flat_map(|e| match e {
                Event::Tick { speed } => [0.0, -1.0, 0.0, speed],
                Event::Plop { r, speed } => [1.0, -1.0, r, speed],
                Event::Plip { r, speed } => [2.0, -1.0, r, speed],
                Event::Tear { leg } => [3.0, leg as f64, 0.0, 0.0],
                Event::Regrown { leg } => [4.0, leg as f64, 0.0, 0.0],
            })
            .collect()
    }
    /// For the sounds: the last step's landing speed, the startle, the time,
    /// and how much the body bounces (the top node's vertical speed against
    /// the body's mean).
    pub fn feel(&self) -> Vec<f64> {
        let t = &self.sim.topos[0];
        let (mut top, mut vy) = (0usize, 0.0f64);
        for i in 0..t.n {
            if t.rest[3 * i].abs() < 0.3 && t.rest[3 * i + 1] > t.rest[3 * top + 1] {
                top = i;
            }
            vy += self.sim.v[3 * i + 1] as f64;
        }
        vec![self.sim.impact, self.sim.startle, self.sim.time, self.sim.v[3 * top + 1] as f64 - vy / t.n as f64]
    }
    #[wasm_bindgen(getter)]
    pub fn regrown(&self) -> usize {
        self.sim.regrown
    }
    #[wasm_bindgen(getter)]
    pub fn time(&self) -> f64 {
        self.sim.time
    }
}

/// A WebGPU device for computing on, with no canvas — what a worker solves on.
#[wasm_bindgen]
pub struct WebComputeDevice {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
}

/// Open a [`WebComputeDevice`]; fails where there is no WebGPU.
#[wasm_bindgen(js_name = requestComputeDevice)]
pub async fn request_compute_device() -> Result<WebComputeDevice, JsError> {
    let gpu = crate::gpu::GpuContext::headless().await.map_err(|e| JsError::new(&e))?;
    Ok(WebComputeDevice { device: gpu.device, queue: gpu.queue })
}
