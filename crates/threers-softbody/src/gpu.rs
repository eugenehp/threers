//! The solver's substeps as WebGPU compute: [`GpuSolver`].
//!
//! The lattice — integration, the hand, edges, volumes and glue, the floor and
//! walls, damping — runs on the GPU, every substep of every frame; what is
//! left (growth, muscles, chains, goo, tearing, the deformation gradients
//! skins are carried by) runs on the CPU once a frame on positions read back,
//! with the same code as the CPU solver.
//!
//! Gauss-Seidel solves constraints one after another, which a GPU cannot.
//! Here the lattice's cubes are split by the parity of their grid coordinates
//! — cubes alike in parity share no node — so each of the eight classes runs
//! in parallel, a thread a cube solving that cube's own edges and then its
//! five tetrahedra in order; the classes run one after another. Glue, which
//! piles many links onto a few body nodes, is solved a part node to a thread,
//! its pull on the body summed in fixed point and applied after. Fifteen
//! dispatches a substep, against more than seventy for colouring each
//! constraint. It is the same physics in a different order: not bit for bit
//! the CPU solver's (which is the JavaScript's), but the same shapes and the
//! same tears (`tests/softbody_gpu.rs`).
//!
//! Reading back is asynchronous in a browser: the CPU half of a frame then
//! sees the positions of the frame before. Whatever the CPU writes (a tear,
//! a reset) is uploaded before the next substeps, and a readback taken before
//! it is dropped rather than let it undo the write.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::sim::SoftBody;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Frame {
    dt: f32,
    gravity: f32,
    alpha_edge: f32,
    alpha_vol: f32,
    alpha_glue: f32,
    damp_k: f32,
    time: f32,
    n: u32,
    walls_on: u32,
    grab_n: u32,
    grab_cos: f32,
    grab_sin: f32,
    grab_target: [f32; 4],
    grab_axis: [f32; 4],
    walls: [[f32; 4]; 4],
    node_at: u32,
    cell_at: u32,
    tet_at: u32,
    edge_at: u32,
    link_at: u32,
    grab_at: u32,
    muscle_at: u32,
    glue_at: u32,
    links: u32,
    body_n: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct GpuBlock {
    off: u32,
    n: u32,
    active: u32,
    leg: i32,
    scale: f32,
    twitch: f32,
    alpha_mul: f32,
    _pad: f32,
}

struct Pending {
    epoch: u64,
    ready: Arc<AtomicBool>,
}

const MAX_BLOCKS: usize = 64;

/// A [`SoftBody`]'s substeps on a GPU. Make one for a body, then call
/// [`step`](Self::step) in place of [`SoftBody::step`]; after changing the
/// body's positions yourself (`reset`), call [`upload_all`](Self::upload_all).
pub struct GpuSolver {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipes: Pipes,
    frame_buf: wgpu::Buffer,
    blocks_buf: wgpu::Buffer,
    state: wgpu::Buffer,
    /// Every table as words (see the shader's `Frame`): nodes, edges, tets,
    /// glue, the hand and the muscles, at `at`.
    data: wgpu::Buffer,
    at: [u32; 10],
    range_stride: u32,
    bind: wgpu::BindGroup,
    range_bind: wgpu::BindGroup,
    /// Cubes in each parity class, then glue groups in each colour batch: a
    /// dispatch each per substep.
    cell_batches: Vec<u32>,
    links: u32,
    /// Kept alive for the bind group.
    #[allow(dead_code)]
    pull: wgpu::Buffer,
    /// Per glue link: which part and which of its links.
    glue_of: Vec<(usize, usize)>,
    glue_scale: Vec<f64>,
    staging: [wgpu::Buffer; 2],
    pending: [Option<Pending>; 2],
    next_staging: usize,
    /// Bumped by every write from the CPU; a readback from before is dropped.
    epoch: u64,
    uploaded_grab: Option<(usize, usize)>,
    n: usize,
    n_blocks: usize,
    /// Frames whose CPU half has not run yet (their positions not back).
    behind: Vec<f64>,
    /// Whether to wait for each readback (native, tests) rather than pick it
    /// up the next frame (a browser, where waiting would block the page).
    pub wait: bool,
}

struct Pipes {
    integrate: wgpu::ComputePipeline,
    grab: wgpu::ComputePipeline,
    cells: wgpu::ComputePipeline,
    glue: wgpu::ComputePipeline,
    apply_glue: wgpu::ComputePipeline,
    contact: wgpu::ComputePipeline,
    means: wgpu::ComputePipeline,
    damp: wgpu::ComputePipeline,
}

fn vec4s(a: &[f32]) -> Vec<[f32; 4]> {
    a.chunks_exact(3).map(|p| [p[0], p[1], p[2], 0.0]).collect()
}

impl GpuSolver {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, sim: &mut SoftBody) -> GpuSolver {
        assert!(sim.blocks.len() <= MAX_BLOCKS, "the GPU solver takes up to {MAX_BLOCKS} blocks (21 parts)");
        let d = &device;
        sim.drops_push_nodes = false;
        let n = sim.n;
        let nb = sim.blocks.len();
        let storage = |label: &str, bytes: &[u8], extra: wgpu::BufferUsages| {
            d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: if bytes.is_empty() { &[0u8; 16] } else { bytes },
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
            })
        };
        let state = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("softbody state"),
            size: (16 * (3 * n + nb)) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut node = vec![0u32; 2 * n];
        for (bi, b) in sim.blocks.iter().enumerate() {
            for i in b.off..b.off + sim.topos[b.topo].n {
                node[2 * i] = sim.w[i].to_bits();
                node[2 * i + 1] = bi as u32;
            }
        }

        // the lattice's cubes, five tetrahedra each in creation order, sorted
        // by the parity of their grid coordinates; each with the edges it
        // owns (those its tetrahedra first name)
        let lattice_of = |topo: usize| if topo == 0 { &sim.body.lattice } else { &sim.limbs[topo - 1].part.lattice };
        let (mut tets, mut edges): (Vec<[u32; 5]>, Vec<[u32; 5]>) = (Vec::new(), Vec::new());
        let mut cells: Vec<(u32, [u32; 4])> = Vec::new();
        for (bi, b) in sim.blocks.iter().enumerate() {
            let t = &sim.topos[b.topo];
            let lat = lattice_of(b.topo);
            let off = b.off as u32;
            // which cube first names each edge
            let mut owner = std::collections::HashMap::new();
            for k in 0..t.rest_vol.len() {
                let q = &t.tets[4 * k..4 * k + 4];
                for i in 0..4 {
                    for j in i + 1..4 {
                        owner.entry((q[i].min(q[j]), q[i].max(q[j]))).or_insert(k / 5);
                    }
                }
            }
            let mut owned: Vec<Vec<usize>> = vec![Vec::new(); t.rest_vol.len() / 5];
            for e in 0..t.rest_len.len() {
                let (a, c) = (t.edges[2 * e], t.edges[2 * e + 1]);
                owned[owner[&(a.min(c), a.max(c))]].push(e);
            }
            for (cell, list) in owned.iter().enumerate() {
                // the cube's grid coordinates, from its lowest corner
                let corner = (0..4 * 5)
                    .map(|q| t.tets[20 * cell + q] as usize)
                    .fold([f64::MAX; 3], |m, i| [0, 1, 2].map(|c| m[c].min(lat.nodes[3 * i + c] as f64)));
                let g = [0, 1, 2].map(|c| ((corner[c] - lat.origin[c]) / lat.h).round() as i64);
                let parity = ((g[0] & 1) | ((g[1] & 1) << 1) | ((g[2] & 1) << 2)) as u32;
                let first_tet = tets.len() as u32;
                for k in 5 * cell..5 * cell + 5 {
                    let q = [0, 1, 2, 3].map(|c| off + t.tets[4 * k + c]);
                    tets.push([q[0], q[1], q[2], q[3], t.rest_vol[k].to_bits()]);
                }
                let first_edge = edges.len() as u32;
                for &e in list {
                    let body_edge = if b.topo == 0 { e as u32 } else { u32::MAX };
                    let phase = if b.leg >= 0 { sim.leg_phase(b.leg as usize)[e] } else { 0.0 };
                    edges.push([off + t.edges[2 * e], off + t.edges[2 * e + 1], body_edge, t.rest_len[e].to_bits(), phase.to_bits()]);
                }
                cells.push((parity, [bi as u32, first_tet, first_edge, list.len() as u32]));
            }
        }
        cells.sort_by_key(|c| c.0);
        let mut cell_batches = vec![0u32; 8];
        for c in &cells {
            cell_batches[c.0 as usize] += 1;
        }
        // glue, grouped by part node and coloured over the body nodes
        let mut glue_of = Vec::new();
        let mut groups: Vec<[u32; 8]> = Vec::new();
        for (i, l) in sim.limbs.iter().enumerate() {
            let off = sim.blocks[sim.leg_slots[i][0]].off as u32;
            let mut g = 0;
            while g < l.glue_part.len() {
                let a = l.glue_part[g];
                let mut rec = [off + a, 0, 0, 0, 0, 0, 0, 0];
                while g < l.glue_part.len() && l.glue_part[g] == a && rec[1] < 3 {
                    rec[2 + rec[1] as usize] = l.glue_body[g];
                    rec[5 + rec[1] as usize] = glue_of.len() as u32;
                    glue_of.push((i, g));
                    rec[1] += 1;
                    g += 1;
                }
                groups.push(rec);
            }
        }
        let body_n = sim.topos[0].n;

        // the word buffer: nodes, cubes, tets, edges, glue, the hand, the
        // muscles, the glue's rest lengths
        let mut words: Vec<u32> = node;
        let cell_at = words.len() as u32;
        words.extend(cells.iter().flat_map(|c| c.1));
        let tet_at = words.len() as u32;
        words.extend(tets.iter().flatten());
        let edge_at = words.len() as u32;
        words.extend(edges.iter().flatten());
        let link_at = words.len() as u32;
        words.extend(groups.iter().flatten());
        let grab_at = words.len() as u32;
        words.extend(std::iter::repeat_n(0, 5 * n));
        let muscle_at = words.len() as u32;
        words.extend(std::iter::repeat_n(0, sim.topos[0].rest_len.len()));
        let glue_at = words.len() as u32;
        words.extend(std::iter::repeat_n(0, glue_of.len()));
        // the dispatch ranges: the eight parity classes
        let mut ranges: Vec<(u32, u32)> = Vec::new();
        let mut s0 = 0;
        for &c in &cell_batches {
            ranges.push((s0, c));
            s0 += c;
        }
        let links = groups.len() as u32;
        let pull = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("softbody glue pull"),
            size: (12 * body_n.max(1)) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let data = storage("softbody tables", bytemuck::cast_slice(&words), wgpu::BufferUsages::empty());
        let at = [0, cell_at, tet_at, edge_at, link_at, grab_at, muscle_at, glue_at, links, body_n as u32];
        let uniform = |label: &str, size: usize| {
            d.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let frame_buf = uniform("softbody frame", std::mem::size_of::<Frame>());
        let blocks_buf = uniform("softbody blocks", 32 * MAX_BLOCKS);
        // every batch's range, at the device's uniform offset alignment
        let stride = d.limits().min_uniform_buffer_offset_alignment.max(8);
        let mut range_words = vec![0u8; stride as usize * ranges.len().max(1)];
        for (k, (s, c)) in ranges.iter().enumerate() {
            let o = k * stride as usize;
            range_words[o..o + 4].copy_from_slice(&s.to_ne_bytes());
            range_words[o + 4..o + 8].copy_from_slice(&c.to_ne_bytes());
        }
        let range_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("softbody ranges"),
            contents: &range_words,
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let layout0 = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("softbody"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, wgpu::BufferBindingType::Uniform),
                entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
                entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let layout1 = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("softbody range"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(8),
                },
                count: None,
            }],
        });
        let bufs = [&frame_buf, &blocks_buf, &state, &data, &pull];
        let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softbody"),
            layout: &layout0,
            entries: &bufs
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
                .collect::<Vec<_>>(),
        });
        let range_bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softbody range"),
            layout: &layout1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &range_buf, offset: 0, size: wgpu::BufferSize::new(8) }),
            }],
        });
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("softbody"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu.wgsl").into()),
        });
        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("softbody"),
            bind_group_layouts: &[Some(&layout0), Some(&layout1)],
            immediate_size: 0,
        });
        let pipe = |entry: &str| {
            d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipes = Pipes {
            integrate: pipe("integrate"),
            grab: pipe("grab"),
            cells: pipe("solve_cells"),
            glue: pipe("solve_glue"),
            apply_glue: pipe("apply_glue"),
            contact: pipe("contact"),
            means: pipe("block_means"),
            damp: pipe("damp"),
        };
        let staging_buf = || {
            d.create_buffer(&wgpu::BufferDescriptor {
                label: Some("softbody readback"),
                size: (32 * n.max(1)) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let mut solver = GpuSolver {
            pipes,
            frame_buf,
            blocks_buf,
            state,
            data,
            at,
            range_stride: stride,
            bind,
            range_bind,
            cell_batches,
            links,
            glue_of,
            pull,
            glue_scale: vec![f64::NAN; sim.limbs.len()],
            staging: [staging_buf(), staging_buf()],
            pending: [None, None],
            next_staging: 0,
            epoch: 0,
            uploaded_grab: None,
            n,
            n_blocks: nb,
            behind: Vec::new(),
            wait: !cfg!(target_arch = "wasm32"),
            device,
            queue,
        };
        solver.upload_all(sim);
        solver
    }

    /// Dispatches a substep takes: integration, the hand, the lattice's
    /// eight parity classes, the glue's two passes, contact and damping's two.
    pub fn batch_count(&self) -> usize {
        2 + self.cell_batches.iter().filter(|&&c| c > 0).count() + 2 + 3
    }

    /// Hand the body back to the CPU solver, keeping this one (its buffers
    /// and pipelines) for [`resume`](Self::resume): readbacks still on their
    /// way are dropped, so the body stays where the last one left it.
    pub fn suspend(&mut self, sim: &mut SoftBody) {
        self.epoch += 1;
        self.behind.clear();
        sim.drops_push_nodes = true;
    }

    /// Take the body back from the CPU solver: its positions, velocities and
    /// hand go up before the next substeps.
    pub fn resume(&mut self, sim: &mut SoftBody) {
        sim.drops_push_nodes = false;
        self.behind.clear();
        self.upload_all(sim);
    }

    /// How many frames the CPU half is behind the GPU: 0 or 1 when the
    /// readbacks keep up, more when the GPU is busy with other work.
    pub fn lag(&self) -> usize {
        self.behind.len()
    }

    /// Positions the CPU has changed (a tear, a bud, a reset): to the GPU
    /// before the next substeps, and any readback from before is dropped.
    pub fn upload_all(&mut self, sim: &SoftBody) {
        let mut data = vec4s(&sim.x);
        data.extend(vec4s(&sim.prev));
        data.extend(vec4s(&sim.v));
        self.queue.write_buffer(&self.state, 0, bytemuck::cast_slice(&data));
        self.epoch += 1;
        self.uploaded_grab = None;
    }

    /// Take a finished readback into the body and run the CPU half of its
    /// frame. True if one was taken.
    fn collect(&mut self, sim: &mut SoftBody) -> bool {
        for k in 0..2 {
            let Some(p) = &self.pending[k] else { continue };
            if !p.ready.load(Ordering::Acquire) {
                continue;
            }
            let stale = p.epoch != self.epoch;
            if !stale {
                let range = self.staging[k].slice(..);
                let data = range.get_mapped_range().expect("the readback is mapped");
                let v: &[[f32; 4]] = bytemuck::cast_slice(&data);
                let (xs, vs) = v.split_at(self.n);
                for i in 0..self.n {
                    sim.x[3 * i..3 * i + 3].copy_from_slice(&xs[i][..3]);
                    sim.v[3 * i..3 * i + 3].copy_from_slice(&vs[i][..3]);
                }
            }
            self.staging[k].unmap();
            self.pending[k] = None;
            let dt = if self.behind.is_empty() { 0.0 } else { self.behind.remove(0) };
            if stale {
                continue;
            }
            // the CPU half: chains and goo through the frame's substeps, then
            // tearing and the gradients
            let s = sim.params.substeps.max(1);
            for _ in 0..s {
                sim.substep_chains(dt / s as f64);
                sim.substep_drops(dt / s as f64);
            }
            let before: Vec<(bool, f64)> = sim.blocks.iter().map(|b| (b.active, b.grow)).collect();
            for i in 0..sim.leg_slots.len() {
                let bi = sim.leg_slots[i][0];
                sim.blocks[bi].strain = sim.glue_strain(i);
            }
            sim.end_frame();
            // a tear copied a part and set a bud: send them
            if sim.blocks.iter().zip(&before).any(|(b, &(a, g))| b.active != a || b.grow < g) {
                self.upload_all(sim);
            }
            return true;
        }
        false
    }

    /// One frame: the CPU's prologue, the substeps on the GPU, and the
    /// positions back (now, or for the next frame).
    pub fn step(&mut self, sim: &mut SoftBody, frame_dt: f64) {
        self.collect(sim);
        sim.begin_frame(frame_dt);
        let s = sim.params.substeps.max(1);
        let dt = frame_dt / s as f64;
        let q = &self.queue;
        // this frame's blocks, muscles, glue lengths and hand
        let blocks: Vec<GpuBlock> = sim
            .blocks
            .iter()
            .map(|b| GpuBlock {
                off: b.off as u32,
                n: sim.topos[b.topo].n as u32,
                active: b.active as u32,
                leg: b.leg,
                scale: b.scale as f32,
                twitch: b.twitch as f32,
                alpha_mul: if b.leg < 0 { 1.0 } else { 0.3 },
                _pad: 0.0,
            })
            .collect();
        q.write_buffer(&self.blocks_buf, 0, bytemuck::cast_slice(&blocks));
        let _ = MAX_BLOCKS;
        let word = |k: usize| 4 * self.at[k] as u64;
        if !sim.topos[0].modv.is_empty() {
            q.write_buffer(&self.data, word(6), bytemuck::cast_slice(&sim.topos[0].modv));
        }
        let scales: Vec<f64> = (0..sim.leg_slots.len()).map(|i| sim.blocks[sim.leg_slots[i][0]].scale).collect();
        if scales != self.glue_scale {
            let rest: Vec<f32> = self.glue_of.iter().map(|&(i, g)| sim.glue_rest(i, g, scales[i]) as f32).collect();
            if !rest.is_empty() {
                q.write_buffer(&self.data, word(7), bytemuck::cast_slice(&rest));
            }
            self.glue_scale = scales;
        }
        let grab_n = match &sim.grab {
            Some(g) => {
                let key = (g.block, g.nodes.len());
                if self.uploaded_grab != Some(key) {
                    let mut held: Vec<u32> = Vec::with_capacity(5 * g.nodes.len());
                    for k in 0..g.nodes.len() {
                        held.extend([g.nodes[k] as u32, (g.weights[k] as f32).to_bits()]);
                        held.extend((0..3).map(|c| (g.offsets[3 * k + c] as f32).to_bits()));
                    }
                    if !held.is_empty() {
                        q.write_buffer(&self.data, word(5), bytemuck::cast_slice(&held));
                    }
                    self.uploaded_grab = Some(key);
                }
                g.nodes.len() as u32
            }
            None => {
                self.uploaded_grab = None;
                0
            }
        };
        let (gcos, gsin, target, axis) = sim.grab.as_ref().map_or((1.0, 0.0, [0.0; 4], [0.0, 0.0, 1.0, 0.0]), |g| {
            (g.angle.cos() as f32, g.angle.sin() as f32, [g.target[0] as f32, g.target[1] as f32, g.target[2] as f32, 0.0], [
                g.axis[0] as f32,
                g.axis[1] as f32,
                g.axis[2] as f32,
                0.0,
            ])
        });
        let walls = sim.walls.unwrap_or([0.0; 16]);
        let alpha = sim.edge_compliance() / (dt * dt);
        let frame = Frame {
            dt: dt as f32,
            gravity: sim.params.gravity as f32,
            alpha_edge: alpha as f32,
            alpha_vol: (1e-8 / (dt * dt)) as f32,
            alpha_glue: (alpha * 3.0) as f32,
            damp_k: crate::js_math::exp(-sim.damp_rate() * dt) as f32,
            time: sim.time as f32,
            n: self.n as u32,
            walls_on: sim.walls.is_some() as u32,
            grab_n,
            grab_cos: gcos,
            grab_sin: gsin,
            grab_target: target,
            grab_axis: axis,
            walls: [0, 1, 2, 3].map(|k| [walls[4 * k], walls[4 * k + 1], walls[4 * k + 2], walls[4 * k + 3]]),
            node_at: self.at[0],
            cell_at: self.at[1],
            tet_at: self.at[2],
            edge_at: self.at[3],
            link_at: self.at[4],
            grab_at: self.at[5],
            muscle_at: self.at[6],
            glue_at: self.at[7],
            links: self.at[8],
            body_n: self.at[9],
            _pad: [0; 2],
        };
        q.write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&frame));

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("softbody step") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("softbody substeps"), timestamp_writes: None });
            pass.set_bind_group(0, &self.bind, &[]);
            pass.set_bind_group(1, &self.range_bind, &[0]);
            let nodes = (self.n as u32).div_ceil(128);
            let p = &self.pipes;
            let stride = self.range_stride;
            for _ in 0..s {
                pass.set_pipeline(&p.integrate);
                pass.dispatch_workgroups(nodes, 1, 1);
                if grab_n > 0 {
                    pass.set_pipeline(&p.grab);
                    pass.dispatch_workgroups(grab_n.div_ceil(64), 1, 1);
                }
                pass.set_pipeline(&p.cells);
                for (slot, &count) in self.cell_batches.iter().enumerate() {
                    if count > 0 {
                        pass.set_bind_group(1, &self.range_bind, &[slot as u32 * stride]);
                        pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
                    }
                }
                if self.links > 0 {
                    pass.set_pipeline(&p.glue);
                    pass.dispatch_workgroups(self.links.div_ceil(64), 1, 1);
                    pass.set_pipeline(&p.apply_glue);
                    pass.dispatch_workgroups(self.at[9].div_ceil(128), 1, 1);
                }
                // the floor, the walls, and the velocities the move implies
                pass.set_pipeline(&p.contact);
                pass.dispatch_workgroups(nodes, 1, 1);
                pass.set_pipeline(&p.means);
                pass.dispatch_workgroups(self.n_blocks as u32, 1, 1);
                pass.set_pipeline(&p.damp);
                pass.dispatch_workgroups(nodes, 1, 1);
            }
        }
        // positions and velocities back, if a staging buffer is free
        let k = self.next_staging;
        let free = self.pending[k].is_none();
        let bytes = (16 * self.n) as u64;
        if free {
            enc.copy_buffer_to_buffer(&self.state, 0, &self.staging[k], 0, bytes);
            enc.copy_buffer_to_buffer(&self.state, 2 * bytes, &self.staging[k], bytes, bytes);
        }
        self.queue.submit(Some(enc.finish()));
        self.behind.push(frame_dt);
        if free {
            let ready = Arc::new(AtomicBool::new(false));
            let r = ready.clone();
            self.staging[k].slice(..).map_async(wgpu::MapMode::Read, move |res| {
                if res.is_ok() {
                    r.store(true, Ordering::Release);
                }
            });
            self.pending[k] = Some(Pending { epoch: self.epoch, ready });
            self.next_staging = 1 - k;
        } else {
            // no room to read this frame back: its CPU half folds into the next
            let dt = self.behind.pop().unwrap_or(0.0);
            if let Some(last) = self.behind.last_mut() {
                *last += dt;
            }
        }
        if self.wait {
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            self.collect(sim);
        }
    }
}
