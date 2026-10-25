//! The physics: verlet strands rooted on the shape's outline, the state
//! machine that sends them seeking, grabbing, holding and snapping, the mound
//! spring, spikes, droplets, and the coat creeping over the cursor.

use super::{VenomState, VenomStyle};
use crate::fx::{Cursor, Outline, Shape};
use crate::math::Vector2;

pub(super) const MS: usize = crate::fx::surface::MAX_STRANDS;
pub(super) const NP: usize = crate::fx::surface::STRAND_POINTS;
const MSPK: usize = crate::fx::surface::MAX_SPIKES;
const MDROP: usize = crate::fx::surface::MAX_DROPS;

/// A small deterministic generator, so runs are repeatable.
struct Rng(u32);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 >> 8) as f32 / (1u32 << 24) as f32
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum Mode {
    Idle,
    Seek,
    Grow,
    Hold,
    Snap,
}

pub(super) struct Strand {
    pub x: [f32; NP],
    pub y: [f32; NP],
    px: [f32; NP],
    py: [f32; NP],
    spread: f32,
    /// Where the root sits, as arc length round the outline.
    pub s: f32,
    slack: f32,
    pub width: f32,
    /// The cursor grips it reaches for: one for a strand, the midpoint of two
    /// neighbours for a hair.
    pub grip: (usize, usize),
    /// The grip it coats, `None` for a hair.
    pub gi: Option<usize>,
    reach_k: f32,
    pub ex: f32,
    pub ey: f32,
    follow: f32,
    seg: f32,
    pub phase: f32,
    pub hair: bool,
}

pub(super) struct Spike {
    pub s: f32,
    pub tilt: f32,
    pub len: f32,
    pub age: f32,
    pub life: f32,
}

pub(super) struct Drop {
    pub x: f32,
    pub y: f32,
    vx: f32,
    vy: f32,
    pub r: f32,
    age: f32,
}

pub(super) struct Sim {
    pub shape: Shape,
    pub outline: Outline,
    /// The shape's inradius, kept because a polygon's is not free.
    pub inradius: f32,
    pub cursor: Cursor,
    rng: Rng,
    pub mode: Mode,
    violent_snap: bool,
    grow: f32,
    pub seek_f: f32,
    pub list: Vec<Strand>,
    /// Strand a, strand b, point in a, point in b, kink.
    pub webs: Vec<(usize, usize, usize, usize, f32)>,
    pub spikes: Vec<Spike>,
    pub drops: Vec<Drop>,
    near_s: f32,
    pub hand: Vector2,
    pub live: bool,
    pub stretch: f32,
    pub mound: f32,
    mound_v: f32,
    pub mound_s: f32,
    pub mound_dir: Vector2,
    pub lean: f32,
    /// How far the goo has crept over each cursor grip, 0 to 1. Kept apart
    /// from the strands so it can keep receding after they are gone.
    pub coat: [f32; Cursor::GRIPS],
    pub time: f32,
    pub spread_px: f32,
}

impl Sim {
    pub fn new() -> Self {
        let shape = Shape::circle(Vector2::ZERO, 1.0);
        Self {
            outline: Outline::new(&shape),
            inradius: 1.0,
            shape,
            cursor: Cursor::default(),
            rng: Rng(0x9e37_79b9),
            mode: Mode::Idle,
            violent_snap: false,
            grow: 0.0,
            seek_f: 0.0,
            list: Vec::new(),
            webs: Vec::new(),
            spikes: Vec::new(),
            drops: Vec::new(),
            near_s: 0.0,
            hand: Vector2::new(-999.0, -999.0),
            live: false,
            stretch: 0.0,
            mound: 0.0,
            mound_v: 0.0,
            mound_s: 0.0,
            mound_dir: Vector2::new(0.0, 1.0),
            lean: 0.0,
            coat: [0.0; Cursor::GRIPS],
            time: 0.0,
            spread_px: 14.0,
        }
    }

    /// Follow the element's shape. Strands already out keep the same fraction
    /// of the way round.
    pub fn sync_shape(&mut self, shape: &Shape) {
        if *shape == self.shape {
            return;
        }
        let old = self.outline.perimeter();
        self.outline = Outline::new(shape);
        let k = self.outline.perimeter() / old;
        for st in &mut self.list {
            st.s *= k;
        }
        for sp in &mut self.spikes {
            sp.s *= k;
        }
        self.mound_s *= k;
        // A press only rescales; the inradius scales with it, and a polygon's
        // is not worth recomputing every frame of the spring.
        let (a, b) = (self.shape.bounds(), shape.bounds());
        let (wa, wb) = ((a.1 - a.0).length(), (b.1 - b.0).length());
        let same_kind = std::mem::discriminant(&self.shape) == std::mem::discriminant(shape);
        self.inradius = match shape {
            Shape::Polygon { points, .. } if same_kind && wa > 0.0 && matches!(&self.shape, Shape::Polygon { points: p, .. } if p.len() == points.len()) => {
                self.inradius * wb / wa
            }
            _ => shape.inradius(),
        };
        self.shape = shape.clone();
    }

    pub fn state(&self) -> VenomState {
        match self.mode {
            Mode::Idle => VenomState::Idle,
            Mode::Seek => VenomState::Seeking,
            Mode::Grow => VenomState::Grabbing,
            Mode::Hold => VenomState::Holding,
            Mode::Snap => VenomState::Snapping,
        }
    }

    pub fn grip_at(&self, grip: (usize, usize)) -> Vector2 {
        let (a, b) = (self.cursor.grip(self.hand, grip.0), self.cursor.grip(self.hand, grip.1));
        (a + b) * 0.5
    }

    pub fn holding(&self) -> bool {
        matches!(self.mode, Mode::Hold | Mode::Grow)
    }

    pub fn thin(&self) -> f32 {
        1.0 - 0.5 * self.stretch
    }

    fn spawn(&mut self, style: &VenomStyle) {
        self.list.clear();
        let main = style.strands.clamp(2, MS);
        let total = if style.hairs { main * 2 - 1 } else { main };
        let root = self.outline.at(self.near_s).point;
        for k in 0..total {
            let hair = style.hairs && k % 2 == 1;
            let mi = if style.hairs { k / 2 } else { k };
            // each thick strand claims a grip; each hair a spot between two of them
            let grip = if hair { (mi, mi + 1) } else { (mi, mi) };
            let r = &mut self.rng;
            let spread = if total > 1 { (k as f32 / (total - 1) as f32) * 2.0 - 1.0 } else { 0.0 };
            self.list.push(Strand {
                x: [root.x; NP],
                y: [root.y; NP],
                px: [root.x; NP],
                py: [root.y; NP],
                spread: spread + (r.f() - 0.5) * 0.15,
                s: self.near_s,
                slack: 1.02 + r.f() * if hair { 0.2 } else { 0.12 },
                width: if hair { 0.5 } else { 0.8 + r.f() * 0.45 },
                grip,
                gi: (!hair).then_some(mi % Cursor::GRIPS),
                reach_k: if hair { 0.35 } else { 0.55 + r.f() * 0.4 },
                ex: root.x,
                ey: root.y,
                follow: 9.0 + r.f() * 8.0,
                seg: 0.0,
                phase: r.f() * 10.0,
                hair,
            });
        }
        self.webs.clear();
        if style.webs {
            for k in 0..self.list.len().saturating_sub(1) {
                let count = 1 + (self.rng.f() * 2.0) as usize;
                for _ in 0..count {
                    let i = 3 + (self.rng.f() * (NP - 5) as f32) as usize;
                    let j = ((i as f32 + ((self.rng.f() - 0.5) * 3.0).round()) as usize).clamp(2, NP - 2);
                    let kink = (self.rng.f() - 0.5) * 5.0;
                    self.webs.push((k, k + 1, i, j, kink));
                }
            }
        }
        self.grow = 0.0;
        self.mode = Mode::Seek;
    }

    fn snap(&mut self, style: &VenomStyle, violent: bool) {
        if matches!(self.mode, Mode::Idle | Mode::Snap) {
            return;
        }
        self.mode = Mode::Snap;
        self.violent_snap = violent;
        if !violent {
            return;
        }
        self.mound_v -= 90.0 * style.mound; // the body recoils
        let mut seen = Vec::new();
        for k in 0..self.list.len() {
            let st = &self.list[k];
            if st.hair {
                continue;
            }
            let key = (st.s / 8.0).round() as i64;
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
            let s = st.s;
            if style.spikes && self.spikes.len() < MSPK {
                let (tilt, len, life) = ((self.rng.f() - 0.5) * 0.8, 10.0 + self.rng.f() * 16.0, 1.1 + self.rng.f() * 0.6);
                self.spikes.push(Spike { s, tilt, len, age: 0.0, life });
            }
            if style.droplets && self.rng.f() < 0.6 && self.drops.len() < MDROP {
                let i = 4 + (self.rng.f() * 4.0) as usize;
                let root = self.outline.at(s).point;
                let st = &self.list[k];
                let (x, y) = (st.x[i], st.y[i]);
                let r = 2.0 + self.rng.f() * 2.2;
                self.drops.push(Drop { x, y, vx: (root.x - x) * 2.0, vy: (root.y - y) * 2.0 - 40.0, r, age: 0.0 });
            }
        }
    }

    /// One frame. `hand` is the pointer's hotspot, `None` when there is none.
    pub fn step(&mut self, style: &VenomStyle, dt: f32, hand: Option<Vector2>) {
        self.time += dt;
        let time = self.time;
        self.live = hand.is_some();
        if let Some(h) = hand {
            self.hand = h;
        }
        // distance is measured from the middle of what the cursor draws
        let anchor = self.cursor.center(self.hand);
        let near_s = self.outline.nearest(anchor);
        self.near_s = near_s;
        let d = self.shape.distance(anchor).max(0.0);
        let (sn, rc, rl) = (style.sense, style.reach, style.release.max(style.reach));
        let (lo, hi) = self.shape.bounds();
        let (cx, width) = ((lo.x + hi.x) * 0.5, (hi.x - lo.x).max(1.0));
        let lean_t = if self.live { ((self.hand.x - cx) / (width * 1.5)).clamp(-1.0, 1.0) } else { 0.0 };
        self.lean += (lean_t - self.lean) * (dt * 4.0).min(1.0);

        let live = self.live;
        match self.mode {
            Mode::Idle if live && d < sn => self.spawn(style),
            // changed its mind
            Mode::Snap if !self.violent_snap && live && d < sn => self.mode = Mode::Seek,
            Mode::Seek if !live || d > sn + 25.0 => self.snap(style, false),
            Mode::Seek if d < rc => {
                self.mode = Mode::Grow;
                self.grow = 0.0;
            }
            Mode::Grow | Mode::Hold if !live || d > rl => self.snap(style, true),
            _ => {}
        }
        let holding = self.holding();
        let seeking = self.mode == Mode::Seek;
        // how hard it is reaching: nothing at the edge of its senses, everything at grabbing range
        let f_t = if seeking {
            ((sn - d) / (sn - rc).max(1.0)).clamp(0.0, 1.0)
        } else if holding {
            1.0
        } else {
            0.0
        };
        self.seek_f += (f_t - self.seek_f) * (dt * 5.0).min(1.0);
        self.stretch = if holding { (d / rl).min(1.0) } else { 0.0 };
        if self.mode == Mode::Grow {
            self.grow = (self.grow + dt / 0.3).min(1.0);
            if self.grow >= 1.0 {
                self.mode = Mode::Hold;
            }
        }

        // ---- the strands ----
        // Roots spread along at most an eighth of the outline.
        let spread_px = (d * 0.4 + 14.0).min(self.outline.perimeter() * 0.13).max(14.0) * if seeking { 0.65 } else { 1.0 };
        self.spread_px = spread_px;
        let mut alive = 0;
        let mode = self.mode;
        let (grow, seek_f) = (self.grow, self.seek_f);
        for idx in 0..self.list.len() {
            let gp = self.grip_at(self.list[idx].grip);
            let outline = &self.outline;
            let st = &mut self.list[idx];
            let target = near_s + st.spread * spread_px;
            st.s += outline.loop_diff(st.s, target) * (dt * if mode == Mode::Snap { 1.5 } else { 6.0 }).min(1.0);
            let root = outline.at(st.s);
            let (rx, ry) = (root.point.x - root.normal.x * 3.0, root.point.y - root.normal.y * 3.0);
            let mut pin_tip = false;
            if seeking {
                // a pseudopod reaching for its grip: part of the way there, feeling about as it goes
                let (vx, vy) = (gp.x - rx, gp.y - ry);
                let vl = vx.hypot(vy).max(1e-6);
                let ease = seek_f * seek_f * (3.0 - 2.0 * seek_f);
                // it hesitates, then darts a little further, then draws back
                let dart = (time * 1.9 + st.phase * 3.1).sin().max(0.0).powi(10) * 0.28;
                let len = (vl * st.reach_k * 0.85 * ease * (1.0 + dart)).max(3.0);
                let sway = (time * 2.7 + st.phase).sin() * (3.0 + 9.0 * ease)
                    + (time * 5.3 + st.phase * 2.0).sin() * 2.0 * ease;
                let tx = rx + (vx / vl) * len - (vy / vl) * sway;
                let ty = ry + (vy / vl) * len + (vx / vl) * sway;
                let k = 1.0 - (-dt * 6.0).exp();
                st.ex += (tx - st.ex) * k;
                st.ey += (ty - st.ey) * k;
                pin_tip = true;
                st.seg = ((st.ex - rx).hypot(st.ey - ry) * (st.slack + 0.06)) / (NP - 1) as f32;
            } else if holding {
                // lunge for the grip and coil in around it, then hold on and trail it like syrup
                let (mut gx, mut gy) = (gp.x, gp.y);
                if mode == Mode::Grow {
                    let coil = (1.0 - grow).powf(1.6) * if st.hair { 7.0 } else { 13.0 };
                    let a = st.phase + grow * std::f32::consts::PI * 2.6;
                    gx += a.cos() * coil;
                    gy += a.sin() * coil * 0.8;
                }
                let rate = if mode == Mode::Grow { 20.0 + st.follow } else { st.follow * 1.4 };
                let k = 1.0 - (-dt * rate).exp();
                st.ex += (gx - st.ex) * k;
                st.ey += (gy - st.ey) * k;
                pin_tip = true;
                st.seg = ((st.ex - rx).hypot(st.ey - ry) * st.slack) / (NP - 1) as f32;
            } else {
                st.seg *= (-dt * 5.5).exp(); // reel back in, slowly, like something heavy
                if st.seg * (NP - 1) as f32 > 1.5 {
                    alive += 1;
                }
            }
            // verlet: heavy damping, gravity and a slow wander
            let damp = if mode == Mode::Snap { style.damping - 0.06 } else { style.damping };
            for i in 1..NP {
                let vx = (st.x[i] - st.px[i]) * damp;
                let vy = (st.y[i] - st.py[i]) * damp;
                st.px[i] = st.x[i];
                st.py[i] = st.y[i];
                let mid = ((i as f32 / (NP - 1) as f32) * std::f32::consts::PI).sin();
                let wx = (time * 1.6 + st.phase + i as f32 * 0.55).sin() * 14.0 * mid;
                let wy = (time * 1.3 + st.phase * 1.3 + i as f32 * 0.45).cos() * 14.0 * mid;
                st.x[i] += vx + wx * dt * dt;
                st.y[i] += vy + (style.gravity + wy) * dt * dt;
            }
            st.x[0] = rx;
            st.px[0] = rx;
            st.y[0] = ry;
            st.py[0] = ry;
            if pin_tip {
                st.x[NP - 1] = st.ex;
                st.y[NP - 1] = st.ey;
            }
            let len = st.seg.max(0.01);
            for _ in 0..14 {
                for i in 0..NP - 1 {
                    let (dx, dy) = (st.x[i + 1] - st.x[i], st.y[i + 1] - st.y[i]);
                    let dd = dx.hypot(dy).max(1e-4);
                    let diff = (dd - len) / dd;
                    let a = if i == 0 { 0.0 } else { 0.5 };
                    let b = if pin_tip && i + 1 == NP - 1 { 0.0 } else if i == 0 { 1.0 } else { 0.5 };
                    let m = if a + b > 0.0 { a + b } else { 1.0 };
                    st.x[i] += dx * diff * (a / m);
                    st.y[i] += dy * diff * (a / m);
                    st.x[i + 1] -= dx * diff * (b / m);
                    st.y[i + 1] -= dy * diff * (b / m);
                }
            }
            // a little bending stiffness, so a strand curves like goo instead of folding into kinks
            for _ in 0..2 {
                for i in 1..NP - 1 {
                    st.x[i] += ((st.x[i - 1] + st.x[i + 1]) / 2.0 - st.x[i]) * 0.22;
                    st.y[i] += ((st.y[i - 1] + st.y[i + 1]) / 2.0 - st.y[i]) * 0.22;
                }
            }
            if !st.x[NP - 1].is_finite() {
                st.x = [rx; NP];
                st.y = [ry; NP];
                st.px = [rx; NP];
                st.py = [ry; NP];
            }
        }
        if self.mode == Mode::Snap && alive == 0 {
            self.mode = Mode::Idle;
            self.list.clear();
            self.webs.clear();
        }

        // ---- the mound, a spring ----
        let mound_target = style.mound
            * if holding {
                5.0 + 16.0 * self.stretch
            } else if seeking {
                3.0 + 9.0 * self.seek_f
            } else {
                0.0
            };
        self.mound_v += ((mound_target - self.mound) * 150.0 - self.mound_v * 9.0) * dt;
        self.mound += self.mound_v * dt;
        if holding || seeking {
            self.mound_s += self.outline.loop_diff(self.mound_s, near_s) * (dt * 6.0).min(1.0);
            let e = self.outline.at(self.mound_s);
            let to = (anchor - e.point).normalize();
            self.mound_dir = e.normal * 0.55 + to * 0.45;
        }

        self.spikes.retain_mut(|s| {
            s.age += dt;
            s.age <= s.life
        });
        let shape = &self.shape;
        self.drops.retain_mut(|d| {
            d.age += dt;
            d.vy += 520.0 * dt;
            d.vx *= (-dt * 2.0).exp();
            d.vy *= (-dt).exp();
            d.x += d.vx * dt;
            d.y += d.vy * dt;
            d.r *= (-dt * 1.5).exp();
            !(d.age > 0.9 || shape.distance(Vector2::new(d.x, d.y)) < -d.r)
        });

        // ---- the goo creeping over the cursor ----
        for g in 0..self.coat.len() {
            let caught = style.coat
                && holding
                && self.list.iter().any(|st| st.gi == Some(g) && (Vector2::new(st.ex, st.ey) - self.grip_at(st.grip)).length() < 6.0);
            // it spreads slowly while it holds on, and slides off quickly when it lets go
            self.coat[g] = if caught { (self.coat[g] + dt * 0.5).min(1.0) } else { (self.coat[g] - dt * 1.4).max(0.0) };
        }
    }
}
