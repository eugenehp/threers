//! The XPBD solver: a body made of parts that can come apart and grow back.
//!
//! A port of Axolotl Jelly's `AxoSim` (Dream Meridian, MIT), arithmetic for
//! arithmetic — f64 working values stored to f32 where the original stores
//! into a `Float32Array`, fdlibm for its `Math.sin`/`exp`/`pow`, V8's
//! `Math.hypot` — so that given the same parts and the same inputs it computes
//! the same particles (`tests/softbody_parity.rs` holds it to the original).
//!
//! All particles live in one array, cut into *blocks*: the body, and for each
//! part one attached slot plus two loose slots for torn-off copies.
//!
//! | | |
//! |---|---|
//! | glue | distance constraints from part root nodes to nearby body nodes |
//! | tearing | when a held part's glue stretches past its limit, the part is copied into a loose slot and the glue lets go |
//! | regrowth | the attached slot restarts as a bud at the socket, its rest lengths and volumes scaled from 5 % back to 100 % |
//! | muscles | body edges change rest length: a travelling sway, a pulse, or nothing |
//! | chains | particle chains pulled toward their rest shape, turned with the head |
//! | drops | goo flung off by a tear, and thrown by hand |

use crate::js_math::{self as js, Random};
use crate::lattice::{Embedding, Part};

/// A part glued to the body: its lattice, which of its nodes hold on to which
/// body nodes, the point it grows back from, and the body node it turns with.
#[derive(Debug, Clone)]
pub struct Limb {
    pub part: Part,
    pub name: String,
    pub glue_part: Vec<u32>,
    pub glue_body: Vec<u32>,
    pub pivot: [f64; 3],
    pub anchor: usize,
}

/// Particle chains (gills, fringes, crests): `parent` −1 marks a root, which
/// rides on the body where `root_emb` binds it; every other particle is pulled
/// toward its rest offset from `anchor`.
#[derive(Debug, Clone, Default)]
pub struct Chains {
    pub rest: Vec<f32>,
    pub parent: Vec<i32>,
    pub anchor: Vec<i32>,
    pub stiff: Vec<f32>,
    pub radius: Vec<f32>,
    pub seg_radius: Vec<f32>,
    pub roots: Vec<i32>,
    pub root_emb: Embedding,
}

/// Body nodes the solver reads: three for the head's frame (the chains turn
/// with it), and two whose distance is the creature's size.
#[derive(Debug, Clone, Copy)]
pub struct Landmarks {
    pub head: [usize; 3],
    pub snout: usize,
    pub tail: usize,
}

/// One part's (or the body's) topology and rest state.
#[derive(Debug, Clone)]
pub struct Topo {
    pub n: usize,
    pub rest: Vec<f32>,
    pub tets: Vec<u32>,
    pub edges: Vec<u32>,
    pub rest_len: Vec<f32>,
    pub rest_vol: Vec<f32>,
    pub dm_inv: Vec<f32>,
    pub w: Vec<f32>,
    pub node_fw: Vec<f32>,
    pub boundary: Vec<u32>,
    pub radius: Vec<f32>,
    pub total_vol: f64,
    /// Per-edge rest-length change, from the muscles.
    pub modv: Vec<f32>,
}

impl Topo {
    fn new(p: &Part) -> Topo {
        let n = p.node_count();
        let (mut mass, mut node_fw) = (vec![0.0f32; n], vec![0.0f32; n]);
        let tets = p.lattice.tets.clone();
        for t in 0..p.rest_vol.len() {
            for c in 0..4 {
                let i = tets[4 * t + c] as usize;
                mass[i] = (mass[i] as f64 + p.rest_vol[t] as f64 / 4.0) as f32;
                node_fw[i] = (node_fw[i] as f64 + p.rest_vol[t] as f64) as f32;
            }
        }
        let r = &p.lattice.nodes;
        let rest_len = (0..p.edges.len() / 2)
            .map(|e| {
                let (a, b) = (3 * p.edges[2 * e] as usize, 3 * p.edges[2 * e + 1] as usize);
                js::hypot3(r[a] as f64 - r[b] as f64, r[a + 1] as f64 - r[b + 1] as f64, r[a + 2] as f64 - r[b + 2] as f64) as f32
            })
            .collect::<Vec<_>>();
        let w = mass.iter().map(|&m| if m > 0.0 { (1.0 / m as f64) as f32 } else { 0.0 }).collect();
        let total_vol = p.rest_vol.iter().fold(0.0f64, |a, &b| a + b as f64);
        Topo {
            n,
            rest: r.clone(),
            tets,
            edges: p.edges.clone(),
            modv: vec![0.0; rest_len.len()],
            rest_len,
            rest_vol: p.rest_vol.clone(),
            dm_inv: p.dm_inv.clone(),
            w,
            node_fw,
            boundary: p.boundary.clone(),
            radius: p.radius.clone(),
            total_vol,
        }
    }
}

/// A slice of the particle array: the body, or one slot of a part.
#[derive(Debug, Clone)]
pub struct Block {
    /// 0 the body, `1 + i` part `i`.
    pub topo: usize,
    pub off: usize,
    pub active: bool,
    pub scale: f64,
    pub grow: f64,
    pub twitch: f64,
    pub born: f64,
    /// The part this is a slot of, −1 for the body.
    pub leg: i32,
    /// 0 attached, 1 and 2 loose.
    pub slot: u8,
    /// How far the glue is stretched, while held.
    pub strain: f64,
    glue_rest: Option<(f64, Vec<f32>)>,
}

/// Something that happened in a step, for sound and for words.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Event {
    Tick { speed: f64 },
    Plop { r: f64, speed: f64 },
    Plip { r: f64, speed: f64 },
    Tear { leg: usize },
    Regrown { leg: usize },
}

/// A hand on a block: the nodes it holds, how firmly, and where.
#[derive(Debug, Clone)]
pub struct Grab {
    pub block: usize,
    pub nodes: Vec<usize>,
    pub weights: Vec<f64>,
    pub offsets: Vec<f64>,
    pub target: [f64; 3],
    pub angle: f64,
    pub axis: [f64; 3],
}

/// What the body's edges do on their own.
#[derive(Debug, Clone)]
pub enum Muscles {
    /// The original's: flank edges toward −x lengthen and shorten in a
    /// travelling wave (an idle sway; a fast flick when startled).
    Sway,
    /// Each edge shortens by `weight · pulse(t, phase, startle)`.
    Pulse { weight: Vec<f32>, phase: Vec<f32>, pulse: Pulse },
    None,
}

/// `a · (1 − calm · s) · wave(ωa t − φa q, pa) + b · s · wave(ωb t − φb q, pb)`,
/// with `s` the startle squared, `q` an edge's phase, and `wave(x, p)` `sin x`
/// for `p = 0`, else `max(0, sin x)^p` — a beat.
#[derive(Debug, Clone, Copy, Default)]
pub struct Pulse {
    pub a: f64,
    pub calm: f64,
    pub omega_a: f64,
    pub phase_a: f64,
    pub power_a: f64,
    pub b: f64,
    pub omega_b: f64,
    pub phase_b: f64,
    pub power_b: f64,
}

impl Pulse {
    fn at(&self, t: f64, q: f64, st: f64) -> f64 {
        let wave = |x: f64, p: f64| {
            let s = js::sin(x);
            if p == 0.0 { s } else { js::pow(s.max(0.0), p) }
        };
        self.a * (1.0 - self.calm * st) * wave(t * self.omega_a - q * self.phase_a, self.power_a)
            + self.b * st * wave(t * self.omega_b - q * self.phase_b, self.power_b)
    }
}

/// The solver's knobs, as the original's sliders set them.
#[derive(Debug, Clone, Copy)]
pub struct Params {
    pub firmness: f64,
    pub damping: f64,
    pub regrow: f64,
    pub substeps: usize,
    pub gravity: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params { firmness: 0.5, damping: 0.4, regrow: 0.5, substeps: 12, gravity: 9.8 }
    }
}

/// What the readout shows.
#[derive(Debug, Clone, Copy)]
pub struct Measure {
    /// Body volume against rest.
    pub volume: f64,
    /// Between the two span landmarks.
    pub length: f64,
    pub attached: usize,
    /// The first growing bud's progress, if one is growing.
    pub growing: Option<f64>,
    pub loose: usize,
}

/// Where a ray met the creature.
#[derive(Debug, Clone, Copy)]
pub enum Hit {
    Part { block: usize, t: f64, point: [f64; 3] },
    Drop { index: usize, t: f64, point: [f64; 3] },
}

pub const MAX_DROPS: usize = 120;

/// Where a step's time went, in the clock's units, summed over steps — when
/// [`SoftBody::clock`] is set. See `examples/softbody_bench.rs`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Profile {
    pub steps: u64,
    pub integrate: f64,
    pub grab: f64,
    pub edges: f64,
    pub volumes: f64,
    pub glue: f64,
    pub contact: f64,
    pub damping: f64,
    pub chains: f64,
    pub drops: f64,
    pub frame: f64,
}

/// The soft body: a body, parts glued on that tear away and regrow, chains
/// and goo. See the [crate docs](crate).
pub struct SoftBody {
    /// The body's lattice, kept to bind more points to it.
    pub body: Part,
    pub topos: Vec<Topo>,
    pub limbs: Vec<Limb>,
    pub chains: Chains,
    pub landmarks: Landmarks,
    pub blocks: Vec<Block>,
    /// `leg_slots[i]` — the three blocks of part `i`, attached first.
    pub leg_slots: Vec<[usize; 3]>,
    pub n: usize,
    pub x: Vec<f32>,
    pub prev: Vec<f32>,
    pub v: Vec<f32>,
    pub w: Vec<f32>,
    pub rad: Vec<f32>,
    /// Per node, its deformation gradient's three columns (xyz_ each).
    pub node_f: Vec<f32>,
    muscle: Vec<f32>,
    muscle_x: Vec<f32>,
    pub muscles: Muscles,
    leg_phase: Vec<Vec<f32>>,
    pub gn: usize,
    pub gx: Vec<f32>,
    gprev: Vec<f32>,
    gv: Vec<f32>,
    g_rest_len: Vec<f32>,
    head_rest: [[f64; 3]; 3],
    r: [f64; 9],
    pub dp: Vec<f32>,
    dprev: Vec<f32>,
    pub dv: Vec<f32>,
    pub dr: Vec<f32>,
    pub d_count: usize,
    d_next: usize,
    d_touch: Vec<u8>,
    touch: Vec<u8>,
    pub params: Params,
    pub time: f64,
    pub startle: f64,
    pub impact: f64,
    pub grab: Option<Grab>,
    /// A drop held by the hand: its index and where it is pulled.
    pub drop_grab: Option<(usize, [f64; 3])>,
    pub events: Vec<Event>,
    pub regrown: usize,
    /// Per part, the glue stretch at which a held part tears (0.07).
    pub tear_limit: Vec<f64>,
    /// Four inward-facing planes `(n, d)` the particles are kept inside.
    pub walls: Option<[f32; 16]>,
    hash: SpatialHash,
    active_idx: Vec<i32>,
    n_active: usize,
    acc: Vec<f32>,
    pub random: Box<dyn Random>,
    /// A clock for [`profile`](Self::profile) (milliseconds, say); `None` to
    /// skip timing.
    pub clock: Option<fn() -> f64>,
    pub profile: Profile,
    /// Whether goo landing on the body pushes it (true). The GPU solver turns
    /// it off: there the body moves on the GPU and the goo on the CPU.
    pub drops_push_nodes: bool,
}

/// JavaScript's `Math.sign`: zero (of either sign) stays as it is.
fn js_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x
    }
}

impl SoftBody {
    pub fn new(body: &Part, limbs: Vec<Limb>, chains: Chains, landmarks: Landmarks, random: Box<dyn Random>) -> SoftBody {
        let mut topos = vec![Topo::new(body)];
        topos.extend(limbs.iter().map(|l| Topo::new(&l.part)));
        let mut blocks = Vec::new();
        let mut off = 0;
        let mut add = |topo: usize, leg: i32, slot: u8, blocks: &mut Vec<Block>| {
            blocks.push(Block {
                topo, off, active: slot == 0, scale: 1.0, grow: 1.0, twitch: 0.0, born: 0.0, leg, slot, strain: 0.0, glue_rest: None,
            });
            off += topos[topo].n;
            blocks.len() - 1
        };
        add(0, -1, 0, &mut blocks);
        let leg_slots: Vec<[usize; 3]> =
            (0..limbs.len()).map(|i| [0u8, 1, 2].map(|slot| add(1 + i, i as i32, slot, &mut blocks))).collect();
        let n = off;
        let (mut w, mut rad) = (vec![0.0f32; n], vec![0.0f32; n]);
        for b in &blocks {
            let t = &topos[b.topo];
            w[b.off..b.off + t.n].copy_from_slice(&t.w);
            rad[b.off..b.off + t.n].copy_from_slice(&t.radius);
        }

        // body muscles: lengthwise edges on each flank, weighted toward the tail
        let t = &topos[0];
        let (mut muscle, mut muscle_x) = (vec![0.0f32; t.rest_len.len()], vec![0.0f32; t.rest_len.len()]);
        for e in 0..t.rest_len.len() {
            let (a, b, r) = (3 * t.edges[2 * e] as usize, 3 * t.edges[2 * e + 1] as usize, &t.rest);
            let mx = (r[a] as f64 + r[b] as f64) / 2.0;
            let mz = (r[a + 2] as f64 + r[b + 2] as f64) / 2.0;
            let ax = (r[a] as f64 - r[b] as f64).abs() / t.rest_len[e] as f64;
            let tail = 1f64.min(0f64.max((0.25 - mx) / 0.9));
            muscle[e] = (js_sign(mz) * 1f64.min(mz.abs() / 0.06) * ax * tail) as f32;
            muscle_x[e] = mx as f32;
        }
        // loose-part twitch: phase by distance from the stump
        let leg_phase = limbs
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let t = &topos[1 + i];
                (0..t.rest_len.len())
                    .map(|e| {
                        let a = 3 * t.edges[2 * e] as usize;
                        (js::hypot3(
                            t.rest[a] as f64 - l.pivot[0],
                            t.rest[a + 1] as f64 - l.pivot[1],
                            t.rest[a + 2] as f64 - l.pivot[2],
                        ) * 30.0) as f32
                    })
                    .collect()
            })
            .collect();

        let gn = chains.rest.len() / 3;
        let mut g_rest_len = vec![0.0f32; gn];
        for i in 0..gn {
            let p = chains.parent[i];
            if p >= 0 {
                let (g, p) = (&chains.rest, p as usize);
                g_rest_len[i] = js::hypot3(
                    g[3 * i] as f64 - g[3 * p] as f64,
                    g[3 * i + 1] as f64 - g[3 * p + 1] as f64,
                    g[3 * i + 2] as f64 - g[3 * p + 2] as f64,
                ) as f32;
            }
        }
        let head_rest = frame_of(&landmarks, &topos[0].rest, 0);
        let n_limbs = limbs.len();
        let mut sim = SoftBody {
            body: body.clone(),
            topos,
            limbs,
            gx: chains.rest.clone(),
            gprev: chains.rest.clone(),
            gv: vec![0.0; gn * 3],
            chains,
            landmarks,
            blocks,
            leg_slots,
            n,
            x: vec![0.0; n * 3],
            prev: vec![0.0; n * 3],
            v: vec![0.0; n * 3],
            w,
            rad,
            node_f: vec![0.0; n * 12],
            muscle,
            muscle_x,
            muscles: Muscles::Sway,
            leg_phase,
            gn,
            g_rest_len,
            head_rest,
            r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            dp: vec![0.0; MAX_DROPS * 3],
            dprev: vec![0.0; MAX_DROPS * 3],
            dv: vec![0.0; MAX_DROPS * 3],
            dr: vec![0.0; MAX_DROPS],
            d_count: 0,
            d_next: 0,
            d_touch: vec![0; MAX_DROPS],
            touch: vec![0; MAX_DROPS],
            params: Params::default(),
            time: 0.0,
            startle: 0.0,
            impact: 0.0,
            grab: None,
            drop_grab: None,
            events: Vec::new(),
            regrown: 0,
            tear_limit: vec![0.07; n_limbs],
            walls: None,
            hash: SpatialHash::new(0.12, n),
            active_idx: vec![0; n],
            n_active: 0,
            acc: vec![0.0; n * 9],
            random,
            clock: None,
            profile: Profile::default(),
            drops_push_nodes: true,
        };
        sim.reset();
        sim
    }

    /// Everything back to rest; parts reattached, loose copies gone.
    pub fn reset(&mut self) {
        for b in &mut self.blocks {
            b.active = b.slot == 0;
            b.scale = 1.0;
            b.grow = 1.0;
            b.twitch = 0.0;
            let t = &self.topos[b.topo];
            self.x[3 * b.off..3 * (b.off + t.n)].copy_from_slice(&t.rest);
        }
        self.prev.copy_from_slice(&self.x);
        self.v.fill(0.0);
        self.gx.copy_from_slice(&self.chains.rest);
        self.gprev.copy_from_slice(&self.gx);
        self.gv.fill(0.0);
        self.d_count = 0;
        self.d_next = 0;
        self.grab = None;
        self.drop_grab = None;
        self.regrown = 0;
        self.update_node_f();
    }

    pub fn edge_compliance(&self) -> f64 {
        js::pow(10.0, 0.6 - 1.6 * self.params.firmness)
    }
    pub fn damp_rate(&self) -> f64 {
        0.4 + 22.0 * js::pow(self.params.damping, 1.6)
    }
    pub fn grow_seconds(&self) -> f64 {
        14.0 - 11.0 * self.params.regrow
    }
    pub fn body(&self) -> usize {
        0
    }

    // ── frame ───────────────────────────────────────────────────────────────
    #[inline]
    fn tick(&self) -> f64 {
        self.clock.map_or(0.0, |c| c())
    }

    pub fn step(&mut self, frame_dt: f64) {
        let t0 = self.tick();
        let s = self.params.substeps;
        let dt = frame_dt / s as f64;
        self.begin_frame(frame_dt);
        let t1 = self.tick();
        self.profile.frame += t1 - t0;
        for _ in 0..s {
            self.substep(dt);
        }
        let t2 = self.tick();
        self.end_frame();
        self.profile.frame += self.tick() - t2;
        self.profile.steps += 1;
    }

    /// The part of a frame before its substeps: the clock, growth, muscles,
    /// the head's frame, and the hash the goo collides against.
    pub(crate) fn begin_frame(&mut self, frame_dt: f64) {
        self.time += frame_dt;
        self.impact = 0.0;
        self.startle = 0f64.max(self.startle - frame_dt / 1.4);
        self.update_growth(frame_dt);
        self.update_muscles();
        self.head_frame();
        let mut na = 0;
        for b in &self.blocks {
            if b.active {
                for i in 0..self.topos[b.topo].n {
                    self.active_idx[na] = (b.off + i) as i32;
                    na += 1;
                }
            }
        }
        self.n_active = na;
        self.hash.build(&self.x, &self.active_idx, na);
    }

    /// The part of a frame after its substeps: tearing, and the deformation
    /// gradients the skins are carried by.
    pub(crate) fn end_frame(&mut self) {
        self.check_tear();
        self.update_node_f();
    }

    fn substep(&mut self, dt: f64) {
        let mut t = self.tick();
        let mut lap = |s: &mut Self, slot: fn(&mut Profile) -> &mut f64| {
            let now = s.tick();
            *slot(&mut s.profile) += now - t;
            t = now;
        };
        let g = self.params.gravity;
        for b in 0..self.blocks.len() {
            if !self.blocks[b].active {
                continue;
            }
            let (s, e) = (self.blocks[b].off, self.blocks[b].off + self.topos[self.blocks[b].topo].n);
            for i in s..e {
                self.v[3 * i + 1] = (self.v[3 * i + 1] as f64 - g * dt) as f32;
                for c in 0..3 {
                    self.prev[3 * i + c] = self.x[3 * i + c];
                    self.x[3 * i + c] = (self.x[3 * i + c] as f64 + self.v[3 * i + c] as f64 * dt) as f32;
                }
            }
        }
        lap(self, |p| &mut p.integrate);
        self.solve_grab();
        lap(self, |p| &mut p.grab);
        let alpha = self.edge_compliance() / (dt * dt);
        for b in 0..self.blocks.len() {
            if self.blocks[b].active {
                self.solve_edges(b, if b == 0 { alpha } else { alpha * 0.3 });
                lap(self, |p| &mut p.edges);
                self.solve_volumes(b, 1e-8 / (dt * dt));
                lap(self, |p| &mut p.volumes);
            }
        }
        for i in 0..self.leg_slots.len() {
            self.solve_glue(i, alpha * 3.0);
        }
        lap(self, |p| &mut p.glue);
        self.solve_floor(dt);
        self.solve_walls();
        lap(self, |p| &mut p.contact);
        let inv = 1.0 / dt;
        let k = js::exp(-self.damp_rate() * dt);
        for b in 0..self.blocks.len() {
            if !self.blocks[b].active {
                continue;
            }
            let n = self.topos[self.blocks[b].topo].n;
            let (s, e) = (self.blocks[b].off, self.blocks[b].off + n);
            let (mut mx, mut my, mut mz) = (0.0f64, 0.0f64, 0.0f64);
            for i in s..e {
                for c in 0..3 {
                    self.v[3 * i + c] = ((self.x[3 * i + c] as f64 - self.prev[3 * i + c] as f64) * inv) as f32;
                }
                mx += self.v[3 * i] as f64;
                my += self.v[3 * i + 1] as f64;
                mz += self.v[3 * i + 2] as f64;
            }
            let nf = n as f64;
            mx /= nf;
            my /= nf;
            mz /= nf;
            for i in s..e {
                self.v[3 * i] = (mx + (self.v[3 * i] as f64 - mx) * k) as f32;
                self.v[3 * i + 1] = (my + (self.v[3 * i + 1] as f64 - my) * k) as f32;
                self.v[3 * i + 2] = (mz + (self.v[3 * i + 2] as f64 - mz) * k) as f32;
            }
        }
        lap(self, |p| &mut p.damping);
        self.substep_chains(dt);
        lap(self, |p| &mut p.chains);
        self.substep_drops(dt);
        lap(self, |p| &mut p.drops);
    }

    fn solve_edges(&mut self, bi: usize, alpha: f64) {
        let b = &self.blocks[bi];
        let t = &self.topos[b.topo];
        let (o, s) = (b.off, b.scale);
        let is_body = bi == 0;
        for e in 0..t.rest_len.len() {
            let (a, c) = (o + t.edges[2 * e] as usize, o + t.edges[2 * e + 1] as usize);
            let (wa, wc) = (self.w[a] as f64, self.w[c] as f64);
            let ws = wa + wc;
            let dx = self.x[3 * a] as f64 - self.x[3 * c] as f64;
            let dy = self.x[3 * a + 1] as f64 - self.x[3 * c + 1] as f64;
            let dz = self.x[3 * a + 2] as f64 - self.x[3 * c + 2] as f64;
            let len = (dx * dx + dy * dy + dz * dz).sqrt();
            if len == 0.0 || ws == 0.0 {
                continue;
            }
            let factor = if is_body {
                1.0 + t.modv[e] as f64
            } else if b.twitch != 0.0 && !b.twitch.is_nan() {
                1.0 + b.twitch * js::sin(self.time * 17.0 + self.leg_phase[b.leg as usize][e] as f64) * 0.22
            } else {
                1.0
            };
            let rest = t.rest_len[e] as f64 * s * factor;
            let k = -(len - rest) / (ws + alpha) / len;
            self.x[3 * a] = (self.x[3 * a] as f64 + dx * k * wa) as f32;
            self.x[3 * a + 1] = (self.x[3 * a + 1] as f64 + dy * k * wa) as f32;
            self.x[3 * a + 2] = (self.x[3 * a + 2] as f64 + dz * k * wa) as f32;
            self.x[3 * c] = (self.x[3 * c] as f64 - dx * k * wc) as f32;
            self.x[3 * c + 1] = (self.x[3 * c + 1] as f64 - dy * k * wc) as f32;
            self.x[3 * c + 2] = (self.x[3 * c + 2] as f64 - dz * k * wc) as f32;
        }
    }

    fn solve_volumes(&mut self, bi: usize, alpha: f64) {
        let b = &self.blocks[bi];
        let t = &self.topos[b.topo];
        let (o, s3) = (b.off, js::pow(b.scale, 3.0));
        let x = &mut self.x;
        let w = &self.w;
        let mut g = [0.0f32; 12];
        for tt in 0..t.rest_vol.len() {
            let n = [o + t.tets[4 * tt] as usize, o + t.tets[4 * tt + 1] as usize, o + t.tets[4 * tt + 2] as usize, o + t.tets[4 * tt + 3] as usize];
            let i = n.map(|q| 3 * q);
            let d = |p: usize, q: usize, c: usize| x[p + c] as f64 - x[q + c] as f64;
            let (ax, ay, az) = (d(i[1], i[0], 0), d(i[1], i[0], 1), d(i[1], i[0], 2));
            let (bx, by, bz) = (d(i[2], i[0], 0), d(i[2], i[0], 1), d(i[2], i[0], 2));
            let (cx, cy, cz) = (d(i[3], i[0], 0), d(i[3], i[0], 1), d(i[3], i[0], 2));
            g[3] = ((by * cz - bz * cy) / 6.0) as f32;
            g[4] = ((bz * cx - bx * cz) / 6.0) as f32;
            g[5] = ((bx * cy - by * cx) / 6.0) as f32;
            g[6] = ((cy * az - cz * ay) / 6.0) as f32;
            g[7] = ((cz * ax - cx * az) / 6.0) as f32;
            g[8] = ((cx * ay - cy * ax) / 6.0) as f32;
            g[9] = ((ay * bz - az * by) / 6.0) as f32;
            g[10] = ((az * bx - ax * bz) / 6.0) as f32;
            g[11] = ((ax * by - ay * bx) / 6.0) as f32;
            g[0] = (-(g[3] as f64 + g[6] as f64 + g[9] as f64)) as f32;
            g[1] = (-(g[4] as f64 + g[7] as f64 + g[10] as f64)) as f32;
            g[2] = (-(g[5] as f64 + g[8] as f64 + g[11] as f64)) as f32;
            let gf = |k: usize| g[k] as f64;
            let vol = ax * gf(3) + ay * gf(4) + az * gf(5);
            let wn = n.map(|q| w[q] as f64);
            let ws = wn[0] * (gf(0) * gf(0) + gf(1) * gf(1) + gf(2) * gf(2))
                + wn[1] * (gf(3) * gf(3) + gf(4) * gf(4) + gf(5) * gf(5))
                + wn[2] * (gf(6) * gf(6) + gf(7) * gf(7) + gf(8) * gf(8))
                + wn[3] * (gf(9) * gf(9) + gf(10) * gf(10) + gf(11) * gf(11));
            if ws == 0.0 {
                continue;
            }
            let k = -(vol - t.rest_vol[tt] as f64 * s3) / (ws + alpha);
            for c in 0..4 {
                for q in 0..3 {
                    x[i[c] + q] = (x[i[c] + q] as f64 + gf(3 * c + q) * k * wn[c]) as f32;
                }
            }
        }
    }

    /// Glue rest lengths follow the bud as it grows: the part's rest shape is
    /// scaled about its pivot, the body end stays put.
    pub(crate) fn glue_rest(&self, i: usize, g: usize, s: f64) -> f64 {
        let l = &self.limbs[i];
        let (lr, br, p) = (&self.topos[1 + i].rest, &self.topos[0].rest, l.pivot);
        let (a, b) = (3 * l.glue_part[g] as usize, 3 * l.glue_body[g] as usize);
        js::hypot3(
            p[0] + s * (lr[a] as f64 - p[0]) - br[b] as f64,
            p[1] + s * (lr[a + 1] as f64 - p[1]) - br[b + 1] as f64,
            p[2] + s * (lr[a + 2] as f64 - p[2]) - br[b + 2] as f64,
        )
    }

    fn solve_glue(&mut self, i: usize, alpha: f64) {
        let bi = self.leg_slots[i][0];
        let s = self.blocks[bi].scale;
        let cached = matches!(&self.blocks[bi].glue_rest, Some((cs, _)) if *cs == s);
        if !cached {
            let rest: Vec<f32> = (0..self.limbs[i].glue_part.len()).map(|g| self.glue_rest(i, g, s) as f32).collect();
            self.blocks[bi].glue_rest = Some((s, rest));
        }
        let off = self.blocks[bi].off;
        let mut stretch = 0.0f64;
        // disjoint fields: the rest lengths are read while x is written
        let rest = &self.blocks[bi].glue_rest.as_ref().unwrap().1;
        let (gl, gb) = (&self.limbs[i].glue_part, &self.limbs[i].glue_body);
        for g in 0..gl.len() {
            let (a, c) = (off + gl[g] as usize, gb[g] as usize);
            let (wa, wc) = (self.w[a] as f64, self.w[c] as f64);
            let ws = wa + wc;
            let dx = self.x[3 * a] as f64 - self.x[3 * c] as f64;
            let dy = self.x[3 * a + 1] as f64 - self.x[3 * c + 1] as f64;
            let dz = self.x[3 * a + 2] as f64 - self.x[3 * c + 2] as f64;
            let len = (dx * dx + dy * dy + dz * dz).sqrt();
            if len == 0.0 {
                continue;
            }
            let rg = rest[g] as f64;
            stretch = stretch.max(len - rg);
            let k = -(len - rg) / (ws + alpha) / len;
            self.x[3 * a] = (self.x[3 * a] as f64 + dx * k * wa) as f32;
            self.x[3 * a + 1] = (self.x[3 * a + 1] as f64 + dy * k * wa) as f32;
            self.x[3 * a + 2] = (self.x[3 * a + 2] as f64 + dz * k * wa) as f32;
            self.x[3 * c] = (self.x[3 * c] as f64 - dx * k * wc) as f32;
            self.x[3 * c + 1] = (self.x[3 * c + 1] as f64 - dy * k * wc) as f32;
            self.x[3 * c + 2] = (self.x[3 * c + 2] as f64 - dz * k * wc) as f32;
        }
        self.blocks[bi].strain = stretch;
    }

    /// How far part `i`'s glue is stretched now, at its current scale — what
    /// `solve_glue` measures in passing, for a solver that does not.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn glue_strain(&self, i: usize) -> f64 {
        let bi = self.leg_slots[i][0];
        let (off, s) = (self.blocks[bi].off, self.blocks[bi].scale);
        let (gl, gb) = (&self.limbs[i].glue_part, &self.limbs[i].glue_body);
        let mut stretch = 0.0f64;
        for g in 0..gl.len() {
            let (a, c) = (off + gl[g] as usize, gb[g] as usize);
            let d = [0, 1, 2].map(|q| self.x[3 * a + q] as f64 - self.x[3 * c + q] as f64);
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            stretch = stretch.max(len - self.glue_rest(i, g, s) as f32 as f64);
        }
        stretch
    }

    /// The twitch phase of each edge of part `i`.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn leg_phase(&self, i: usize) -> &[f32] {
        &self.leg_phase[i]
    }

    fn solve_floor(&mut self, dt: f64) {
        for q in 0..self.n_active {
            let i = self.active_idx[q] as usize;
            if self.x[3 * i + 1] >= 0.0 {
                continue;
            }
            // how fast it came down, if it was airborne a moment ago (for the thud)
            if self.prev[3 * i + 1] > 0.004 {
                self.impact = self.impact.max((self.prev[3 * i + 1] as f64 - self.x[3 * i + 1] as f64) / dt);
            }
            let pen = -(self.x[3 * i + 1] as f64);
            self.x[3 * i + 1] = 0.0;
            let tx = self.x[3 * i] as f64 - self.prev[3 * i] as f64;
            let tz = self.x[3 * i + 2] as f64 - self.prev[3 * i + 2] as f64;
            let tl = js::hypot(&[tx, tz]);
            if tl > 0.0 {
                let k = 1f64.min(0.7 * pen / tl + 0.35);
                self.x[3 * i] = (self.x[3 * i] as f64 - tx * k) as f32;
                self.x[3 * i + 2] = (self.x[3 * i + 2] as f64 - tz * k) as f32;
            }
        }
    }

    /// Soft walls at the screen's edges: a node outside is eased back a little
    /// each substep (prev moved half as far), so it bumps rather than bounces.
    fn solve_walls(&mut self) {
        let Some(wl) = self.walls else { return };
        for q in 0..self.n_active {
            let i = self.active_idx[q] as usize;
            for k in (0..16).step_by(4) {
                let s = wl[k] as f64 * self.x[3 * i] as f64
                    + wl[k + 1] as f64 * self.x[3 * i + 1] as f64
                    + wl[k + 2] as f64 * self.x[3 * i + 2] as f64
                    + wl[k + 3] as f64;
                if s >= 0.0 {
                    continue;
                }
                let m = (-s).min(0.006);
                for c in 0..3 {
                    self.x[3 * i + c] = (self.x[3 * i + c] as f64 + wl[k + c] as f64 * m) as f32;
                    self.prev[3 * i + c] = (self.prev[3 * i + c] as f64 + wl[k + c] as f64 * m * 0.5) as f32;
                }
            }
        }
    }

    fn walls_drop(&mut self, i: usize) {
        let Some(wl) = self.walls else { return };
        let r = self.dr[i] as f64;
        for k in (0..16).step_by(4) {
            let s = wl[k] as f64 * self.dp[3 * i] as f64
                + wl[k + 1] as f64 * self.dp[3 * i + 1] as f64
                + wl[k + 2] as f64 * self.dp[3 * i + 2] as f64
                + wl[k + 3] as f64
                - r;
            if s >= 0.0 {
                continue;
            }
            for c in 0..3 {
                self.dp[3 * i + c] = (self.dp[3 * i + c] as f64 - wl[k + c] as f64 * s) as f32;
                self.dprev[3 * i + c] = (self.dprev[3 * i + c] as f64 - wl[k + c] as f64 * s * 0.6) as f32;
            }
        }
    }

    // ── muscles ─────────────────────────────────────────────────────────────
    fn update_muscles(&mut self) {
        let st = self.startle * self.startle;
        let t = self.time;
        let m = &mut self.topos[0].modv;
        match &self.muscles {
            Muscles::Sway => {
                let amp_idle = 0.03 * (1.0 - st);
                let amp_flick = 0.2 * st;
                for e in 0..m.len() {
                    if self.muscle[e] == 0.0 {
                        m[e] = 0.0;
                        continue;
                    }
                    let mx = self.muscle_x[e] as f64;
                    m[e] = (self.muscle[e] as f64
                        * (amp_idle * js::sin(t * 1.7 + mx * 3.2) + amp_flick * js::sin(t * 13.0 + mx * 4.5)))
                        as f32;
                }
            }
            Muscles::Pulse { weight, phase, pulse } => {
                for e in 0..m.len() {
                    let wgt = weight[e] as f64;
                    m[e] = if wgt != 0.0 { (-wgt * pulse.at(t, phase[e] as f64, st)) as f32 } else { 0.0 };
                }
            }
            Muscles::None => m.fill(0.0),
        }
    }

    // ── grabbing ────────────────────────────────────────────────────────────
    pub fn start_grab(&mut self, hit: [f64; 3], block: usize) {
        let radius = if block == 0 { 0.2 } else { 0.16 };
        let b = &self.blocks[block];
        let (mut nodes, mut weights, mut offsets) = (Vec::new(), Vec::new(), Vec::new());
        for i in b.off..b.off + self.topos[b.topo].n {
            let dx = self.x[3 * i] as f64 - hit[0];
            let dy = self.x[3 * i + 1] as f64 - hit[1];
            let dz = self.x[3 * i + 2] as f64 - hit[2];
            let d = js::hypot3(dx, dy, dz);
            if d > radius {
                continue;
            }
            let f = 1.0 - d / radius;
            nodes.push(i);
            weights.push(f * f);
            offsets.extend([dx, dy, dz]);
        }
        self.grab = Some(Grab { block, nodes, weights, offsets, target: hit, angle: 0.0, axis: [0.0, 0.0, 1.0] });
        if block == 0 {
            self.poke();
        }
    }

    fn solve_grab(&mut self) {
        let Some(g) = &self.grab else { return };
        let (c, s) = (js::cos(g.angle), js::sin(g.angle));
        let [ax, ay, az] = g.axis;
        for k in 0..g.nodes.len() {
            let i = g.nodes[k];
            let (ox, oy, oz) = (g.offsets[3 * k], g.offsets[3 * k + 1], g.offsets[3 * k + 2]);
            let d = ax * ox + ay * oy + az * oz;
            let rx = ox * c + (ay * oz - az * oy) * s + ax * d * (1.0 - c);
            let ry = oy * c + (az * ox - ax * oz) * s + ay * d * (1.0 - c);
            let rz = oz * c + (ax * oy - ay * ox) * s + az * d * (1.0 - c);
            let k2 = 0.5 * g.weights[k];
            self.x[3 * i] = (self.x[3 * i] as f64 + (g.target[0] + rx - self.x[3 * i] as f64) * k2) as f32;
            self.x[3 * i + 1] = (self.x[3 * i + 1] as f64 + (g.target[1] + ry - self.x[3 * i + 1] as f64) * k2) as f32;
            self.x[3 * i + 2] = (self.x[3 * i + 2] as f64 + (g.target[2] + rz - self.x[3 * i + 2] as f64) * k2) as f32;
        }
    }

    pub fn poke(&mut self) {
        self.startle = self.startle.max(0.55);
    }
    pub fn flick(&mut self) {
        self.startle = 1.0;
    }

    // ── tearing and regrowth ────────────────────────────────────────────────
    fn check_tear(&mut self) {
        let Some(g) = &self.grab else { return };
        let b = &self.blocks[g.block];
        if b.leg < 0 || b.slot != 0 || b.grow < 1.0 {
            return;
        }
        let leg = b.leg as usize;
        if b.strain > self.tear_limit[leg] {
            self.tear(leg);
        }
    }

    /// Tear part `i` off: copy it into a loose slot, which keeps simulating
    /// (and twitches), and restart the attached slot as a bud.
    pub fn tear(&mut self, i: usize) {
        let a = self.leg_slots[i][0];
        let n = self.topos[self.blocks[a].topo].n;
        // the free slot, else the one torn off longest ago
        let mut free = [self.leg_slots[i][1], self.leg_slots[i][2]];
        free.sort_by(|&p, &q| {
            let (p, q) = (&self.blocks[p], &self.blocks[q]);
            (p.active as i32 - q.active as i32)
                .cmp(&0)
                .then(p.born.partial_cmp(&q.born).unwrap_or(std::cmp::Ordering::Equal))
        });
        let f = free[0];
        let (ao, fo) = (self.blocks[a].off, self.blocks[f].off);
        for arr in [&mut self.x, &mut self.prev, &mut self.v] {
            arr.copy_within(3 * ao..3 * (ao + n), 3 * fo);
        }
        let time = self.time;
        let fb = &mut self.blocks[f];
        fb.active = true;
        fb.scale = 1.0;
        fb.grow = 1.0;
        fb.twitch = 1.0;
        fb.born = time;
        if let Some(g) = &mut self.grab {
            if g.block == a {
                for node in &mut g.nodes {
                    *node = *node - ao + fo;
                }
                g.block = f;
            }
        }
        // a burst of goo from the stump
        let p = self.pivot_now(i);
        for _ in 0..14 {
            let ang = self.random.next() * std::f64::consts::PI * 2.0;
            let u = self.random.next();
            let pos = [p[0] + js::cos(ang) * 0.04, p[1] + 0.06 + u * 0.05, p[2] + js::sin(ang) * 0.04];
            let vy = 1.4 + self.random.next() * 1.6;
            let vel = [js::cos(ang) * (0.6 + u), vy, js::sin(ang) * (0.6 + u)];
            let r = 0.014 + self.random.next() * 0.02;
            self.add_drop(pos, vel, r);
        }
        self.blocks[a].grow = 0.0;
        self.blocks[a].scale = 0.05;
        self.place_bud(i);
        self.poke();
        self.events.push(Event::Tear { leg: i });
    }

    /// The body's rotation near part `i`'s socket, from the deformation
    /// gradient there (column-major).
    pub fn socket_rotation(&self, i: usize) -> [f64; 9] {
        let o = 12 * self.limbs[i].anchor;
        let f = &self.node_f;
        let mut c0 = [f[o] as f64, f[o + 1] as f64, f[o + 2] as f64];
        let mut c1 = [f[o + 4] as f64, f[o + 5] as f64, f[o + 6] as f64];
        let l0 = js::hypot(&c0);
        let l0 = if l0 == 0.0 || l0.is_nan() { 1.0 } else { l0 };
        c0 = c0.map(|v| v / l0);
        let d = c0[0] * c1[0] + c0[1] * c1[1] + c0[2] * c1[2];
        c1 = [c1[0] - c0[0] * d, c1[1] - c0[1] * d, c1[2] - c0[2] * d];
        let l1 = js::hypot(&c1);
        let l1 = if l1 == 0.0 || l1.is_nan() { 1.0 } else { l1 };
        c1 = c1.map(|v| v / l1);
        let c2 = [c0[1] * c1[2] - c0[2] * c1[1], c0[2] * c1[0] - c0[0] * c1[2], c0[0] * c1[1] - c0[1] * c1[0]];
        [c0[0], c0[1], c0[2], c1[0], c1[1], c1[2], c2[0], c2[1], c2[2]]
    }

    /// Where part `i`'s pivot is now, carried by the body round its socket.
    pub fn pivot_now(&self, i: usize) -> [f64; 3] {
        let l = &self.limbs[i];
        let (a, r, br) = (l.anchor, self.socket_rotation(i), &self.topos[0].rest);
        let d = [l.pivot[0] - br[3 * a] as f64, l.pivot[1] - br[3 * a + 1] as f64, l.pivot[2] - br[3 * a + 2] as f64];
        let x = &self.x;
        [
            x[3 * a] as f64 + r[0] * d[0] + r[3] * d[1] + r[6] * d[2],
            x[3 * a + 1] as f64 + r[1] * d[0] + r[4] * d[1] + r[7] * d[2],
            x[3 * a + 2] as f64 + r[2] * d[0] + r[5] * d[1] + r[8] * d[2],
        ]
    }

    fn place_bud(&mut self, i: usize) {
        let a = self.leg_slots[i][0];
        let (r, p) = (self.socket_rotation(i), self.pivot_now(i));
        let (off, s) = (self.blocks[a].off, self.blocks[a].scale);
        let t = &self.topos[self.blocks[a].topo];
        let pv = self.limbs[i].pivot;
        for k in 0..t.n {
            let d = [
                (t.rest[3 * k] as f64 - pv[0]) * s,
                (t.rest[3 * k + 1] as f64 - pv[1]) * s,
                (t.rest[3 * k + 2] as f64 - pv[2]) * s,
            ];
            let j = 3 * (off + k);
            self.x[j] = (p[0] + r[0] * d[0] + r[3] * d[1] + r[6] * d[2]) as f32;
            self.x[j + 1] = (p[1] + r[1] * d[0] + r[4] * d[1] + r[7] * d[2]) as f32;
            self.x[j + 2] = (p[2] + r[2] * d[0] + r[5] * d[1] + r[8] * d[2]) as f32;
        }
        let range = 3 * off..3 * (off + t.n);
        self.prev[range.clone()].copy_from_slice(&self.x[range.clone()]);
        self.v[range].fill(0.0);
    }

    fn update_growth(&mut self, dt: f64) {
        let grow_seconds = self.grow_seconds();
        for li in 0..self.leg_slots.len() {
            let a = self.leg_slots[li][0];
            let b = &mut self.blocks[a];
            if b.grow < 1.0 {
                b.grow = 1f64.min(b.grow + dt / grow_seconds);
                // ease out with a little overshoot, like a bud swelling then settling
                let (t, c1) = (b.grow - 1.0, 1.5);
                let c3 = c1 + 1.0;
                b.scale = if b.grow >= 1.0 { 1.0 } else { 0.05 + 0.95 * (1.0 + c3 * t * t * t + c1 * t * t) };
                if b.grow >= 1.0 {
                    self.regrown += 1;
                    self.events.push(Event::Regrown { leg: li });
                }
            }
            for k in 1..3 {
                let time = self.time;
                let l = &mut self.blocks[self.leg_slots[li][k]];
                if l.active {
                    let age = time - l.born;
                    l.twitch = 0f64.max(1.0 - age / 9.0) * (0.55 + 0.45 * js::sin(age * 2.1));
                }
            }
        }
    }

    // ── chains ──────────────────────────────────────────────────────────────
    fn head_frame(&mut self) {
        let now = frame_of(&self.landmarks, &self.x, 0);
        let rest = self.head_rest;
        for r in 0..3 {
            for c in 0..3 {
                self.r[3 * r + c] = now[0][r] * rest[0][c] + now[1][r] * rest[1][c] + now[2][r] * rest[2][c];
            }
        }
    }

    pub(crate) fn substep_chains(&mut self, dt: f64) {
        let g = &self.chains;
        let (tets, emb, x, r) = (&self.topos[0].tets, &g.root_emb, &self.x, self.r);
        let gx = &mut self.gx;
        // roots ride on the head
        for (ri, &root) in g.roots.iter().enumerate() {
            let i = root as usize;
            let t = emb.tet_of[ri] as usize;
            let (b1, b2, b3) = (emb.bary[3 * ri] as f64, emb.bary[3 * ri + 1] as f64, emb.bary[3 * ri + 2] as f64);
            let b0 = 1.0 - b1 - b2 - b3;
            for c in 0..3 {
                let at = |k: usize| x[3 * tets[4 * t + k] as usize + c] as f64;
                gx[3 * i + c] = (b0 * at(0) + b1 * at(1) + b2 * at(2) + b3 * at(3)) as f32;
            }
        }
        let t = self.time;
        let breath = 1.0 + 0.07 * js::sin(t * 2.2) + 0.12 * self.startle;
        let grav = self.params.gravity;
        for i in 0..self.gn {
            if g.parent[i] < 0 {
                continue;
            }
            let gv = &mut self.gv;
            gv[3 * i + 1] = (gv[3 * i + 1] as f64 - grav * 0.3 * dt) as f32;
            gv[3 * i] = (gv[3 * i] as f64 + js::sin(t * 3.1 + i as f64 * 0.7) * 0.6 * dt) as f32;
            gv[3 * i + 2] = (gv[3 * i + 2] as f64 + js::cos(t * 2.6 + i as f64 * 1.3) * 0.6 * dt) as f32;
            for c in 0..3 {
                self.gprev[3 * i + c] = gx[3 * i + c];
                gx[3 * i + c] = (gx[3 * i + c] as f64 + gv[3 * i + c] as f64 * dt) as f32;
            }
        }
        // parents come before children in the array
        for i in 0..self.gn {
            let p = g.parent[i];
            if p < 0 {
                continue;
            }
            let (p, a, s) = (p as usize, g.anchor[i] as usize, &g.rest);
            // soft pull toward the rest pose, carried round by the head's rotation
            let rx = (s[3 * i] as f64 - s[3 * a] as f64) * breath;
            let ry = (s[3 * i + 1] as f64 - s[3 * a + 1] as f64) * breath;
            let rz = (s[3 * i + 2] as f64 - s[3 * a + 2] as f64) * breath;
            let k = g.stiff[i] as f64;
            for c in 0..3 {
                let target = gx[3 * a + c] as f64 + r[3 * c] * rx + r[3 * c + 1] * ry + r[3 * c + 2] * rz;
                gx[3 * i + c] = (gx[3 * i + c] as f64 + (target - gx[3 * i + c] as f64) * k) as f32;
            }
            // inextensible link to the parent (only the child moves)
            let dx = gx[3 * i] as f64 - gx[3 * p] as f64;
            let dy = gx[3 * i + 1] as f64 - gx[3 * p + 1] as f64;
            let dz = gx[3 * i + 2] as f64 - gx[3 * p + 2] as f64;
            let len = (dx * dx + dy * dy + dz * dz).sqrt();
            let len = if len == 0.0 || len.is_nan() { 1.0 } else { len };
            let f = self.g_rest_len[i] as f64 * breath / len;
            gx[3 * i] = (gx[3 * p] as f64 + dx * f) as f32;
            gx[3 * i + 1] = (gx[3 * p + 1] as f64 + dy * f) as f32;
            gx[3 * i + 2] = (gx[3 * p + 2] as f64 + dz * f) as f32;
            if gx[3 * i + 1] < g.radius[i] {
                gx[3 * i + 1] = g.radius[i];
            }
        }
        let inv = 1.0 / dt;
        let kd = js::exp(-3.0 * dt);
        for i in 0..self.gn {
            for c in 0..3 {
                self.gv[3 * i + c] = ((gx[3 * i + c] as f64 - self.gprev[3 * i + c] as f64) * inv * kd) as f32;
            }
        }
    }

    // ── drops ───────────────────────────────────────────────────────────────
    pub fn add_drop(&mut self, p: [f64; 3], vel: [f64; 3], r: f64) {
        let i = self.d_next;
        self.d_next = (self.d_next + 1) % MAX_DROPS;
        self.d_count = (self.d_count + 1).min(MAX_DROPS);
        for c in 0..3 {
            self.dp[3 * i + c] = p[c] as f32;
            self.dprev[3 * i + c] = p[c] as f32;
            self.dv[3 * i + c] = vel[c] as f32;
        }
        self.dr[i] = r as f32;
        if matches!(self.drop_grab, Some((k, _)) if k == i) {
            self.drop_grab = None;
        }
    }

    pub(crate) fn substep_drops(&mut self, dt: f64) {
        let g = self.params.gravity;
        let m = self.d_count;
        for i in 0..m {
            self.dv[3 * i + 1] = (self.dv[3 * i + 1] as f64 - g * dt) as f32;
            for c in 0..3 {
                self.dprev[3 * i + c] = self.dp[3 * i + c];
                self.dp[3 * i + c] = (self.dp[3 * i + c] as f64 + self.dv[3 * i + c] as f64 * dt) as f32;
            }
        }
        if let Some((i, t)) = self.drop_grab {
            for c in 0..3 {
                self.dp[3 * i + c] = (self.dp[3 * i + c] as f64 + (t[c] - self.dp[3 * i + c] as f64) * 0.35) as f32;
            }
        }
        self.touch.fill(0);
        let (dp, dv, dr) = (&mut self.dp, &self.dv, &self.dr);
        let dprev = &mut self.dprev;
        for i in 0..m {
            for j in i + 1..m {
                let dx = dp[3 * i] as f64 - dp[3 * j] as f64;
                let dy = dp[3 * i + 1] as f64 - dp[3 * j + 1] as f64;
                let dz = dp[3 * i + 2] as f64 - dp[3 * j + 2] as f64;
                let rr = dr[i] as f64 + dr[j] as f64;
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 >= rr * rr || d2 == 0.0 {
                    continue;
                }
                if self.d_touch[i] & 2 == 0 && self.touch[i] & 2 == 0 {
                    let rv = js::hypot3(
                        dv[3 * i] as f64 - dv[3 * j] as f64,
                        dv[3 * i + 1] as f64 - dv[3 * j + 1] as f64,
                        dv[3 * i + 2] as f64 - dv[3 * j + 2] as f64,
                    );
                    if rv > 0.6 && self.events.len() < 40 {
                        self.events.push(Event::Tick { speed: rv });
                    }
                }
                self.touch[i] |= 2;
                self.touch[j] |= 2;
                let d = d2.sqrt();
                let s = 0.5 * (rr - d).min(0.01) / d;
                let k = 0.8 * s;
                for (c, dd) in [dx, dy, dz].into_iter().enumerate() {
                    dp[3 * i + c] = (dp[3 * i + c] as f64 + dd * s) as f32;
                    dp[3 * j + c] = (dp[3 * j + c] as f64 - dd * s) as f32;
                }
                for (c, dd) in [dx, dy, dz].into_iter().enumerate() {
                    dprev[3 * i + c] = (dprev[3 * i + c] as f64 + dd * k) as f32;
                    dprev[3 * j + c] = (dprev[3 * j + c] as f64 - dd * k) as f32;
                }
            }
        }
        for i in 0..m {
            let r = dr[i] as f64;
            let wd = 1.0 / (r * r * r * 4.2);
            let cnt = self.hash.query(dp[3 * i] as f64, dp[3 * i + 1] as f64, dp[3 * i + 2] as f64);
            for q in 0..cnt {
                let n = self.hash.query_ids[q] as usize;
                let rr = r + self.rad[n] as f64;
                if rr <= 0.0 {
                    continue;
                }
                let dx = dp[3 * i] as f64 - self.x[3 * n] as f64;
                let dy = dp[3 * i + 1] as f64 - self.x[3 * n + 1] as f64;
                let dz = dp[3 * i + 2] as f64 - self.x[3 * n + 2] as f64;
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 >= rr * rr || d2 == 0.0 {
                    continue;
                }
                if self.d_touch[i] & 1 == 0 && self.touch[i] & 1 == 0 {
                    let sp = js::hypot3(dv[3 * i] as f64, dv[3 * i + 1] as f64, dv[3 * i + 2] as f64);
                    if sp > 0.8 && self.events.len() < 40 {
                        self.events.push(Event::Plop { r, speed: sp });
                    }
                }
                self.touch[i] |= 1;
                let d = d2.sqrt();
                let cc = (rr - d).min(0.01) / d;
                let wn = self.w[n] as f64;
                let ws = wd + wn;
                let kd = 0.6 * cc * wd / ws;
                for (c, dd) in [dx, dy, dz].into_iter().enumerate() {
                    dp[3 * i + c] = (dp[3 * i + c] as f64 + dd * cc * wd / ws) as f32;
                }
                for (c, dd) in [dx, dy, dz].into_iter().enumerate() {
                    dprev[3 * i + c] = (dprev[3 * i + c] as f64 + dd * kd) as f32;
                }
                if self.drops_push_nodes {
                    for (c, dd) in [dx, dy, dz].into_iter().enumerate() {
                        self.x[3 * n + c] = (self.x[3 * n + c] as f64 - dd * cc * wn / ws) as f32;
                    }
                }
            }
        }
        self.d_touch.copy_from_slice(&self.touch);
        let inv = 1.0 / dt;
        for i in 0..m {
            let r = self.dr[i] as f64;
            if (self.dp[3 * i + 1] as f64) < r {
                if self.dprev[3 * i + 1] as f64 > r + 0.002 && (self.dv[3 * i + 1] as f64) < -0.8 && self.events.len() < 40 {
                    self.events.push(Event::Plip { r, speed: -(self.dv[3 * i + 1] as f64) });
                }
                self.dp[3 * i + 1] = r as f32;
                self.dp[3 * i] = (self.dp[3 * i] as f64 - (self.dp[3 * i] as f64 - self.dprev[3 * i] as f64) * 0.3) as f32;
                self.dp[3 * i + 2] =
                    (self.dp[3 * i + 2] as f64 - (self.dp[3 * i + 2] as f64 - self.dprev[3 * i + 2] as f64) * 0.3) as f32;
            }
            self.walls_drop(i);
            for c in 0..3 {
                self.dv[3 * i + c] = ((self.dp[3 * i + c] as f64 - self.dprev[3 * i + c] as f64) * inv) as f32;
            }
            let rr = js::hypot(&[self.dp[3 * i] as f64, self.dp[3 * i + 2] as f64]);
            if rr > 4.0 {
                let f = 4.0 / rr;
                self.dp[3 * i] = (self.dp[3 * i] as f64 * f) as f32;
                self.dp[3 * i + 2] = (self.dp[3 * i + 2] as f64 * f) as f32;
                self.dv[3 * i] = (self.dv[3 * i] as f64 * -0.3) as f32;
                self.dv[3 * i + 2] = (self.dv[3 * i + 2] as f64 * -0.3) as f32;
            }
        }
    }

    // ── per-node deformation gradient, for normals ──────────────────────────
    pub fn update_node_f(&mut self) {
        let mut f = [0.0f32; 9];
        let mut e = [0.0f32; 9];
        for b in &self.blocks {
            if !b.active {
                continue;
            }
            let t = &self.topos[b.topo];
            let (d, o) = (&t.dm_inv, b.off);
            self.acc[9 * o..9 * (o + t.n)].fill(0.0);
            for tt in 0..t.rest_vol.len() {
                let i0 = 3 * (o + t.tets[4 * tt] as usize);
                let dd = 9 * tt;
                for c in 1..=3 {
                    let ic = 3 * (o + t.tets[4 * tt + c] as usize);
                    for q in 0..3 {
                        e[3 * c - 3 + q] = (self.x[ic + q] as f64 - self.x[i0 + q] as f64) as f32;
                    }
                }
                for col in 0..3 {
                    for row in 0..3 {
                        f[3 * col + row] = (e[row] as f64 * d[dd + 3 * col] as f64
                            + e[3 + row] as f64 * d[dd + 3 * col + 1] as f64
                            + e[6 + row] as f64 * d[dd + 3 * col + 2] as f64) as f32;
                    }
                }
                let wv = t.rest_vol[tt] as f64;
                for c in 0..4 {
                    let a = 9 * (o + t.tets[4 * tt + c] as usize);
                    for k in 0..9 {
                        self.acc[a + k] = (self.acc[a + k] as f64 + f[k] as f64 * wv) as f32;
                    }
                }
            }
            for i in 0..t.n {
                let s = 1.0 / t.node_fw[i] as f64;
                let (a, q) = (9 * (o + i), 12 * (o + i));
                for c in 0..3 {
                    for k in 0..3 {
                        self.node_f[q + 4 * c + k] = (self.acc[a + 3 * c + k] as f64 * s) as f32;
                    }
                }
            }
        }
    }

    pub fn measure(&self) -> Measure {
        let t = &self.topos[0];
        let x = &self.x;
        let mut v = 0.0f64;
        for tt in 0..t.rest_vol.len() {
            let q: Vec<usize> = (0..4).map(|c| 3 * t.tets[4 * tt + c] as usize).collect();
            let d = |p: usize, c: usize| x[q[p] + c] as f64 - x[q[0] + c] as f64;
            let (ux, uy, uz) = (d(1, 0), d(1, 1), d(1, 2));
            let (vx, vy, vz) = (d(2, 0), d(2, 1), d(2, 2));
            let (wx, wy, wz) = (d(3, 0), d(3, 1), d(3, 2));
            v += ((uy * vz - uz * vy) * wx + (uz * vx - ux * vz) * wy + (ux * vy - uy * vx) * wz) / 6.0;
        }
        let (s, tl) = (3 * self.landmarks.snout, 3 * self.landmarks.tail);
        let length = js::hypot3(x[s] as f64 - x[tl] as f64, x[s + 1] as f64 - x[tl + 1] as f64, x[s + 2] as f64 - x[tl + 2] as f64);
        let attached = self.leg_slots.iter().filter(|s| self.blocks[s[0]].grow >= 1.0).count();
        let growing = self.leg_slots.iter().map(|s| self.blocks[s[0]].grow).find(|&g| g < 1.0);
        let loose = self.blocks.iter().filter(|b| b.slot > 0 && b.active).count();
        Measure { volume: v / t.total_vol, length, attached, growing, loose }
    }

    /// The nearest part or drop along the ray from `o` along unit `d`. Parts
    /// win close calls against the body, so a click on a foot grabs the foot.
    pub fn raycast(&self, o: [f64; 3], d: [f64; 3]) -> Option<Hit> {
        let x = &self.x;
        let mut best_t = f64::INFINITY;
        let mut best: Option<Hit> = None;
        for (bi, blk) in self.blocks.iter().enumerate() {
            if !blk.active {
                continue;
            }
            let (bd, off) = (&self.topos[blk.topo].boundary, blk.off);
            for f in (0..bd.len()).step_by(3) {
                let (a, b, c) = (3 * (off + bd[f] as usize), 3 * (off + bd[f + 1] as usize), 3 * (off + bd[f + 2] as usize));
                let g = |i: usize| x[i] as f64;
                let (e1x, e1y, e1z) = (g(b) - g(a), g(b + 1) - g(a + 1), g(b + 2) - g(a + 2));
                let (e2x, e2y, e2z) = (g(c) - g(a), g(c + 1) - g(a + 1), g(c + 2) - g(a + 2));
                let (px, py, pz) = (d[1] * e2z - d[2] * e2y, d[2] * e2x - d[0] * e2z, d[0] * e2y - d[1] * e2x);
                let det = e1x * px + e1y * py + e1z * pz;
                if det.abs() < 1e-10 {
                    continue;
                }
                let inv = 1.0 / det;
                let (tx, ty, tz) = (o[0] - g(a), o[1] - g(a + 1), o[2] - g(a + 2));
                let u = (tx * px + ty * py + tz * pz) * inv;
                if !(0.0..=1.0).contains(&u) {
                    continue;
                }
                let (qx, qy, qz) = (ty * e1z - tz * e1y, tz * e1x - tx * e1z, tx * e1y - ty * e1x);
                let v = (d[0] * qx + d[1] * qy + d[2] * qz) * inv;
                if v < 0.0 || u + v > 1.0 {
                    continue;
                }
                let t = (e2x * qx + e2y * qy + e2z * qz) * inv;
                let tt = if bi == 0 { t + 0.03 } else { t };
                if t > 0.0 && tt < best_t {
                    best_t = tt;
                    best = Some(Hit::Part { block: bi, t: tt, point: [0.0; 3] });
                }
            }
        }
        for i in 0..self.d_count {
            let r = self.dr[i] as f64 * 1.8;
            let (ox, oy, oz) = (o[0] - self.dp[3 * i] as f64, o[1] - self.dp[3 * i + 1] as f64, o[2] - self.dp[3 * i + 2] as f64);
            let b = ox * d[0] + oy * d[1] + oz * d[2];
            let c = ox * ox + oy * oy + oz * oz - r * r;
            let h = b * b - c;
            if h < 0.0 {
                continue;
            }
            let t = -b - h.sqrt();
            if t > 0.0 && t < best_t {
                best_t = t;
                best = Some(Hit::Drop { index: i, t, point: [0.0; 3] });
            }
        }
        let at = |t: f64| [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t];
        best.map(|h| match h {
            Hit::Part { block, t, .. } => Hit::Part { block, t, point: at(t) },
            Hit::Drop { index, t, .. } => Hit::Drop { index, t, point: at(t) },
        })
    }

    /// The active blocks' node ranges, for whoever draws them.
    pub fn block_range(&self, b: usize) -> std::ops::Range<usize> {
        let blk = &self.blocks[b];
        blk.off..blk.off + self.topos[blk.topo].n
    }
}

/// An orthonormal frame from the three head landmarks of the block at `off`.
fn frame_of(lm: &Landmarks, p: &[f32], off: usize) -> [[f64; 3]; 3] {
    let [a, b, c] = lm.head.map(|i| 3 * (off + i));
    let g = |i: usize| p[i] as f64;
    let mut e1 = [g(b) - g(a), g(b + 1) - g(a + 1), g(b + 2) - g(a + 2)];
    let l1 = js::hypot(&e1);
    e1 = e1.map(|v| v / l1);
    let mut e2 = [g(c) - g(a), g(c + 1) - g(a + 1), g(c + 2) - g(a + 2)];
    let d = e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2];
    e2 = [e2[0] - e1[0] * d, e2[1] - e1[1] * d, e2[2] - e1[2] * d];
    let l2 = js::hypot(&e2);
    e2 = e2.map(|v| v / l2);
    let e3 = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    [e1, e2, e3]
}

/// The original's spatial hash over the active particles, rebuilt each frame
/// and queried by the drops; its key reproduces JavaScript's int32 XOR.
struct SpatialHash {
    s: f64,
    size: usize,
    start: Vec<i32>,
    entries: Vec<i32>,
    query_ids: Vec<i32>,
}

impl SpatialHash {
    fn new(spacing: f64, max_items: usize) -> Self {
        let size = 2 * max_items;
        SpatialHash { s: spacing, size, start: vec![0; size + 1], entries: vec![0; max_items], query_ids: vec![0; max_items] }
    }
    fn key(&self, xi: f64, yi: f64, zi: f64) -> usize {
        let h = js::to_int32(xi * 92837111.0) ^ js::to_int32(yi * 689287499.0) ^ js::to_int32(zi * 283923481.0);
        ((h as i64).abs() % self.size as i64) as usize
    }
    fn cell(&self, p: &[f32], i: usize) -> usize {
        self.key((p[3 * i] as f64 / self.s).floor(), (p[3 * i + 1] as f64 / self.s).floor(), (p[3 * i + 2] as f64 / self.s).floor())
    }
    fn build(&mut self, p: &[f32], idx: &[i32], n: usize) {
        self.start.fill(0);
        for &i in &idx[..n] {
            let k = self.cell(p, i as usize);
            self.start[k] += 1;
        }
        let mut acc = 0;
        for i in 0..self.size {
            acc += self.start[i];
            self.start[i] = acc;
        }
        self.start[self.size] = acc;
        for &i in &idx[..n] {
            let k = self.cell(p, i as usize);
            self.start[k] -= 1;
            self.entries[self.start[k] as usize] = i;
        }
    }
    fn query(&mut self, x: f64, y: f64, z: f64) -> usize {
        let (xi, yi, zi) = ((x / self.s).floor(), (y / self.s).floor(), (z / self.s).floor());
        let mut n = 0;
        for a in [xi - 1.0, xi, xi + 1.0] {
            for b in [yi - 1.0, yi, yi + 1.0] {
                for c in [zi - 1.0, zi, zi + 1.0] {
                    let h = self.key(a, b, c);
                    let (s, e) = (self.start[h], self.start[h + 1]);
                    let mut k = s;
                    while k < e && n < self.query_ids.len() {
                        self.query_ids[n] = self.entries[k as usize];
                        n += 1;
                        k += 1;
                    }
                }
            }
        }
        n
    }
}
