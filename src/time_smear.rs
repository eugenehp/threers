//! Time smear: every moving thing gets echoes of itself at other moments in
//! time — before it arrives as well as after it has gone.
//!
//! Motion blur, trails and afterimages smear light forwards only: frame `t`
//! can reach `t + 1, t + 2, …`. This sends each pixel's brightness over a
//! 64-frame window through a small interference transform instead, and the
//! echoes land at specific, non-adjacent frames on *both* sides of the event,
//! spaced by the bits of a Gray code rather than spread smoothly.
//!
//! The model is the "quantum time-smear" one: per pixel,
//!
//! 1. the 64 luma values become amplitudes `sqrt(v[t] / Σv)` on six qubits,
//!    addressed by the Gray code of `t` (neighbouring frames differ in one bit);
//! 2. qubit `k` turns by `θₖ = π·strength·((1 − reach)·W₀[k] + reach)` — at
//!    `reach = 0` the low bits (near frames) turn most, at `reach = 1` all six
//!    turn alike and the echoes jump far;
//! 3. the probabilities, scaled back by `Σv`, are the new brightness.
//!
//! Light is moved in time, never created — each window keeps its total — and
//! a pixel that never changes is a fixed point, so a static background comes
//! back bit-identical and only moving parts are ghosted. The transform is not
//! linear: smearing `a + b` differs from smearing each and adding.
//!
//! As in the original, only luma at a reduced working width is smeared. The
//! *echo* (smeared − source) is upsampled and laid over the untouched
//! full-resolution frame, so edges and text come from the source and only the
//! ghost layer is soft. Clips of any length run as 64-frame windows hopping by
//! 32, crossfaded with `sin²` weights that sum to one.
//!
//! The result depends on frames that have not been shown yet, so this is an
//! offline filter over a frame sequence rather than a live post pass. It
//! streams: [`TimeSmear::wrap`] turns any frame closure — a
//! [`HeadlessRenderer`](crate::HeadlessRenderer), decoded video — into one
//! that returns smeared frames, holding at most 64 source frames at a time.
//!
//! ```no_run
//! use threers::time_smear::{SmearLook, TimeSmear};
//! # fn render(_i: usize) -> Vec<u8> { vec![0; 640 * 360 * 4] }
//! let smear = TimeSmear::new().strength(0.3).reach(0.55).look(SmearLook::Faithful);
//! let mut frame = smear.wrap(640, 360, 240, render);
//! for i in 0..240 {
//!     let rgba = frame(i); // ascending order, each index once
//!     # let _ = rgba;
//! }
//! ```
//!
//! Colour is handled the way the original does: the echo is computed on
//! Rec. 709 luma of the *encoded* values and added equally to R, G and B, so
//! hue survives and grey stays grey. [`SmearLook::Ghost`] and
//! [`SmearLook::Echo`] paint it with an invented palette instead.

use std::collections::BTreeMap;
use std::f64::consts::PI;

use crate::utils::parallel::par_map_range;

/// Frames per window: six qubits.
pub const WINDOW: usize = 64;
const QUBITS: usize = 6;
const HOP: usize = WINDOW / 2;
/// Tile edge, in working pixels, for [`Adaptive`] strength.
const TILE: usize = 36;

/// Per-qubit turn, as a fraction of `π·strength`, at `reach = 0` — fitted to
/// the reference engine; low bits (adjacent frames) turn most.
pub const LOCAL_WEIGHTS: [f64; QUBITS] = [1.0, 0.5, 0.2424, 0.1212, 0.0606, 0.0455];

/// Which rotation each qubit gets. `X` mixes with a quarter-turn phase, so
/// echoes interfere differently from `Y`'s real rotation; the pairs apply
/// both, in order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Rotation {
    #[default]
    X,
    Y,
    XY,
    YX,
}

impl Rotation {
    fn gates(self) -> &'static [bool] {
        // true = Rx, false = Ry
        match self {
            Rotation::X => &[true],
            Rotation::Y => &[false],
            Rotation::XY => &[true, false],
            Rotation::YX => &[false, true],
        }
    }
}

impl std::str::FromStr for Rotation {
    type Err = String;

    /// `"x"`, `"y"`, `"xy"` or `"yx"`, any case.
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "x" => Ok(Rotation::X),
            "y" => Ok(Rotation::Y),
            "xy" => Ok(Rotation::XY),
            "yx" => Ok(Rotation::YX),
            _ => Err(format!("unknown rotation '{s}' (x, y, xy, yx)")),
        }
    }
}

/// How the echo is laid over the source.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SmearLook {
    /// Source plus its own echo on every channel: no invented colour.
    #[default]
    Faithful,
    /// Added light as an iridescent glow, removed light as a darkening.
    Ghost,
    /// The smeared luma alone, in grey — what the transform returned.
    Raw,
    /// Only the echo: warm where light was added, cool where it was taken,
    /// black where nothing changed.
    Echo,
}

impl std::str::FromStr for SmearLook {
    type Err = String;

    /// `"faithful"`, `"ghost"`, `"raw"` or `"echo"`, any case.
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "faithful" => Ok(SmearLook::Faithful),
            "ghost" => Ok(SmearLook::Ghost),
            "raw" => Ok(SmearLook::Raw),
            "echo" => Ok(SmearLook::Echo),
            _ => Err(format!("unknown look '{s}' (faithful, ghost, raw, echo)")),
        }
    }
}

/// Scale strength and reach per tile by how much that tile moves, so busy
/// regions can be pushed hard without ghosting calm ones.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Adaptive {
    /// Mean per-pixel temporal standard deviation (grey levels) at which a
    /// tile gets the full strength and reach.
    pub motion_ref: f32,
    /// Fraction a motionless tile still gets.
    pub floor: f32,
}

impl Default for Adaptive {
    fn default() -> Self {
        Self { motion_ref: 40.0, floor: 0.15 }
    }
}

/// The time-smear filter. See the [module docs](self).
#[derive(Clone, Debug, PartialEq)]
pub struct TimeSmear {
    /// Overall turn, 0 (identity) to 1.
    pub strength: f32,
    /// 0: echoes stay near the event; 1: they reach across the window.
    pub reach: f32,
    pub rotation: Rotation,
    /// Width the luma is smeared at; height follows the aspect.
    pub work_width: u32,
    pub look: SmearLook,
    /// Echo gain; `None` takes the look's own (1 faithful, 3 ghost, 4 echo).
    pub gain: Option<f32>,
    /// Ghost look: how much removed light darkens the source.
    pub dark: f32,
    pub adaptive: Option<Adaptive>,
}

impl Default for TimeSmear {
    fn default() -> Self {
        Self {
            strength: 0.3,
            reach: 0.55,
            rotation: Rotation::X,
            work_width: 160,
            look: SmearLook::Faithful,
            gain: None,
            dark: 0.25,
            adaptive: None,
        }
    }
}

impl TimeSmear {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn strength(mut self, v: f32) -> Self {
        self.strength = v;
        self
    }
    pub fn reach(mut self, v: f32) -> Self {
        self.reach = v;
        self
    }
    pub fn rotation(mut self, v: Rotation) -> Self {
        self.rotation = v;
        self
    }
    pub fn work_width(mut self, v: u32) -> Self {
        self.work_width = v.max(1);
        self
    }
    pub fn look(mut self, v: SmearLook) -> Self {
        self.look = v;
        self
    }
    pub fn gain(mut self, v: f32) -> Self {
        self.gain = Some(v);
        self
    }
    pub fn dark(mut self, v: f32) -> Self {
        self.dark = v;
        self
    }
    pub fn adaptive(mut self, v: Adaptive) -> Self {
        self.adaptive = Some(v);
        self
    }

    /// Working size for a `width × height` source.
    pub fn work_size(&self, width: u32, height: u32) -> (usize, usize) {
        let ww = self.work_width.clamp(1, width.max(1)) as usize;
        let wh = ((height as f64 * ww as f64 / width.max(1) as f64).round() as usize).max(1);
        (ww, wh)
    }

    /// Echo (smeared − source) of a luma volume, frame-major
    /// `[frames][height][width]`, any number of frames. This is the whole
    /// computation; [`wrap`](Self::wrap) streams the same thing.
    pub fn echo_volume(&self, luma: &[f32], width: usize, height: usize) -> Vec<f32> {
        let px = width * height;
        assert!(px > 0 && luma.len().is_multiple_of(px), "luma is not whole frames");
        let n = luma.len() / px;
        if n == 0 {
            return Vec::new();
        }
        let padded = n.max(WINDOW);
        let frame = |t: usize| &luma[t.min(n - 1) * px..][..px];
        let weights = window_weights();
        let mut acc = vec![0f32; padded * px];
        let mut wsum = vec![0f32; padded];
        for start in plan_windows(padded) {
            let win: Vec<&[f32]> = (start..start + WINDOW).map(frame).collect();
            let echo = self.window_echo(&win, width, height);
            for (j, e) in echo.chunks(px).enumerate() {
                let t = start + j;
                let w = weights[j];
                for (a, &v) in acc[t * px..][..px].iter_mut().zip(e) {
                    *a += w * v;
                }
                wsum[t] += w;
            }
        }
        for (t, chunk) in acc.chunks_mut(px).enumerate() {
            let inv = 1.0 / wsum[t].max(1e-6);
            chunk.iter_mut().for_each(|v| *v *= inv);
        }
        acc.truncate(n * px);
        acc
    }

    /// Echo of one 64-frame window, `[64][height][width]`.
    fn window_echo(&self, win: &[&[f32]], width: usize, height: usize) -> Vec<f32> {
        debug_assert_eq!(win.len(), WINDOW);
        let px = width * height;
        let tiles_x = width.div_ceil(TILE);
        let factors: Vec<f32> = match self.adaptive {
            None => vec![1.0],
            Some(a) => (0..tiles_x * height.div_ceil(TILE))
                .map(|i| {
                    let (tx, ty) = (i % tiles_x, i / tiles_x);
                    let motion = tile_motion(win, width, height, tx * TILE, ty * TILE);
                    a.floor + (1.0 - a.floor) * (motion / a.motion_ref.max(1e-6)).min(1.0)
                })
                .collect(),
        };
        let kernels: Vec<Kernel> = factors
            .iter()
            .map(|&f| Kernel::new(self.strength * f, self.reach * f, self.rotation))
            .collect();
        let columns = par_map_range(px, |p| {
            let mut col = [0f64; WINDOW];
            for (c, f) in col.iter_mut().zip(win) {
                *c = f[p].max(0.0) as f64;
            }
            let kernel = match self.adaptive {
                None => &kernels[0],
                Some(_) => &kernels[(p / width / TILE) * tiles_x + (p % width) / TILE],
            };
            let mut out = col;
            kernel.apply(&mut out);
            let mut echo = [0f32; WINDOW];
            for t in 0..WINDOW {
                echo[t] = (out[t] - col[t]) as f32;
            }
            echo
        });
        let mut vol = vec![0f32; WINDOW * px];
        for (p, echo) in columns.iter().enumerate() {
            for t in 0..WINDOW {
                vol[t * px + p] = echo[t];
            }
        }
        vol
    }

    /// Smear one pixel's 64-frame history in place. Values must be ≥ 0;
    /// the sum is kept.
    pub fn smear_column(&self, column: &mut [f64; WINDOW]) {
        Kernel::new(self.strength, self.reach, self.rotation).apply(column);
    }

    /// Smear a whole clip held in memory: RGBA8 frames, `width × height`.
    pub fn apply(&self, frames: &[Vec<u8>], width: u32, height: u32) -> Vec<Vec<u8>> {
        let mut next = self.wrap(width, height, frames.len(), |i| frames[i].clone());
        (0..frames.len()).map(&mut next).collect()
    }

    /// Wrap a frame closure (RGBA8, `width × height`, `frames` long) into one
    /// returning smeared frames. Call it with `0, 1, 2, …` in order — exactly
    /// what [`render_sequence`](crate::render_sequence) and, with the `video`
    /// feature, `VideoExporter::export` do. It renders up to 63
    /// frames ahead and keeps those source frames in memory.
    pub fn wrap<F>(
        &self,
        width: u32,
        height: u32,
        frames: usize,
        render: F,
    ) -> impl FnMut(usize) -> Vec<u8>
    where
        F: FnMut(usize) -> Vec<u8>,
    {
        let mut stream = Stream::new(self.clone(), width, height, frames, render);
        move |i| stream.frame(i)
    }
}

/// Window starts covering `[0, n)`, `n ≥ 64`: every frame in at least one,
/// interior frames in two, the last window flush with the end.
fn plan_windows(n: usize) -> Vec<usize> {
    let mut starts: Vec<usize> = (0..=n - WINDOW).step_by(HOP).collect();
    if starts.last().unwrap() + WINDOW < n {
        starts.push(n - WINDOW);
    }
    starts
}

/// `sin²` ramp; `w(i) + w(i + 32) == 1`, so half-overlapping windows sum to one.
fn window_weights() -> [f32; WINDOW] {
    std::array::from_fn(|i| {
        let s = (PI * (i as f64 + 0.5) / WINDOW as f64).sin();
        (s * s) as f32
    })
}

/// Mean over a tile of each pixel's temporal standard deviation.
fn tile_motion(win: &[&[f32]], width: usize, height: usize, x0: usize, y0: usize) -> f32 {
    let (x1, y1) = ((x0 + TILE).min(width), (y0 + TILE).min(height));
    let mut total = 0f64;
    for y in y0..y1 {
        for x in x0..x1 {
            let p = y * width + x;
            let mean = win.iter().map(|f| f[p] as f64).sum::<f64>() / WINDOW as f64;
            let var = win.iter().map(|f| (f[p] as f64 - mean).powi(2)).sum::<f64>() / WINDOW as f64;
            total += var.sqrt();
        }
    }
    (total / ((x1 - x0) * (y1 - y0)) as f64) as f32
}

const fn gray(t: usize) -> usize {
    t ^ (t >> 1)
}

/// Per-qubit `cos(θ/2)`, `sin(θ/2)` and the gate sequence.
struct Kernel {
    cs: [(f64, f64); QUBITS],
    gates: &'static [bool],
}

impl Kernel {
    fn new(strength: f32, reach: f32, rotation: Rotation) -> Self {
        let (s, r) = (strength as f64, reach as f64);
        Self {
            cs: std::array::from_fn(|k| {
                let half = 0.5 * PI * s * ((1.0 - r) * LOCAL_WEIGHTS[k] + r);
                (half.cos(), half.sin())
            }),
            gates: rotation.gates(),
        }
    }

    fn apply(&self, col: &mut [f64; WINDOW]) {
        let (lo, hi) = col.iter().fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        let sum: f64 = col.iter().sum();
        // A constant column is the uniform state, which every rotation only
        // rephases: leave it exactly as it is.
        if lo == hi || sum <= 0.0 {
            return;
        }
        let mut re = [0f64; WINDOW];
        let mut im = [0f64; WINDOW];
        for t in 0..WINDOW {
            re[gray(t)] = (col[t] / sum).sqrt();
        }
        for (k, &(c, s)) in self.cs.iter().enumerate() {
            let bit = 1 << k;
            for &rx in self.gates {
                for i in (0..WINDOW).filter(|i| i & bit == 0) {
                    let j = i | bit;
                    let (ar, ai, br, bi) = (re[i], im[i], re[j], im[j]);
                    if rx {
                        // [c, −is; −is, c]
                        re[i] = c * ar + s * bi;
                        im[i] = c * ai - s * br;
                        re[j] = c * br + s * ai;
                        im[j] = c * bi - s * ar;
                    } else {
                        // [c, −s; s, c]
                        re[i] = c * ar - s * br;
                        im[i] = c * ai - s * bi;
                        re[j] = s * ar + c * br;
                        im[j] = s * ai + c * bi;
                    }
                }
            }
        }
        for (t, c) in col.iter_mut().enumerate() {
            let g = gray(t);
            *c = (re[g] * re[g] + im[g] * im[g]) * sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming

struct Stream<F> {
    smear: TimeSmear,
    width: usize,
    height: usize,
    frames: usize,
    work: (usize, usize),
    render: F,
    starts: Vec<usize>,
    next_window: usize,
    next_input: usize,
    next_output: usize,
    luma: BTreeMap<usize, Vec<f32>>,
    sources: BTreeMap<usize, Vec<u8>>,
    acc: BTreeMap<usize, (Vec<f32>, f32)>,
    weights: [f32; WINDOW],
}

impl<F: FnMut(usize) -> Vec<u8>> Stream<F> {
    fn new(smear: TimeSmear, width: u32, height: u32, frames: usize, render: F) -> Self {
        let work = smear.work_size(width, height);
        Self {
            smear,
            width: width as usize,
            height: height as usize,
            frames,
            work,
            render,
            starts: plan_windows(frames.max(WINDOW)),
            next_window: 0,
            next_input: 0,
            next_output: 0,
            luma: BTreeMap::new(),
            sources: BTreeMap::new(),
            acc: BTreeMap::new(),
            weights: window_weights(),
        }
    }

    /// Bring input frame `t` (past the end: the last frame repeated) in.
    fn load(&mut self, t: usize) {
        if t >= self.frames {
            let last = self.luma[&(self.frames - 1)].clone();
            self.luma.insert(t, last);
            return;
        }
        let rgba = (self.render)(t);
        assert_eq!(rgba.len(), self.width * self.height * 4, "frame {t}: wrong size");
        self.luma.insert(t, downsample_luma(&rgba, self.width, self.height, self.work));
        self.sources.insert(t, rgba);
    }

    fn frame(&mut self, i: usize) -> Vec<u8> {
        assert_eq!(i, self.next_output, "TimeSmear frames must be requested in order");
        assert!(i < self.frames, "frame {i} past the end ({})", self.frames);
        let (ww, wh) = self.work;
        let px = ww * wh;
        while self.next_window < self.starts.len() && self.starts[self.next_window] <= i {
            let start = self.starts[self.next_window];
            while self.next_input < start + WINDOW {
                self.load(self.next_input);
                self.next_input += 1;
            }
            let win: Vec<&[f32]> = (start..start + WINDOW).map(|t| &self.luma[&t][..]).collect();
            let echo = self.smear.window_echo(&win, ww, wh);
            for (j, e) in echo.chunks(px).enumerate() {
                let w = self.weights[j];
                let slot = self.acc.entry(start + j).or_insert_with(|| (vec![0.0; px], 0.0));
                for (a, &v) in slot.0.iter_mut().zip(e) {
                    *a += w * v;
                }
                slot.1 += w;
            }
            self.next_window += 1;
        }
        let (mut echo, wsum) = self.acc.remove(&i).expect("frame covered by a window");
        let inv = 1.0 / wsum.max(1e-6);
        echo.iter_mut().for_each(|v| *v *= inv);
        let source = self.sources.remove(&i).expect("source frame loaded");
        let out = composite(&self.smear, &source, &self.luma[&i], &echo, self.width, self.height, self.work);
        self.next_output += 1;
        // Luma is needed by windows still to come and by the frames still to
        // be written (the raw look draws it).
        let keep = self.starts.get(self.next_window).copied().unwrap_or(usize::MAX).min(self.next_output);
        self.luma.retain(|&t, _| t >= keep);
        out
    }
}

// ---------------------------------------------------------------------------
// Resampling and looks

fn luma_of(px: &[u8]) -> f32 {
    0.2126 * px[0] as f32 + 0.7152 * px[1] as f32 + 0.0722 * px[2] as f32
}

/// Box-filtered Rec. 709 luma at the working size.
pub(crate) fn downsample_luma(rgba: &[u8], width: usize, height: usize, (ww, wh): (usize, usize)) -> Vec<f32> {
    let span = |i: usize, n: usize, full: usize| {
        let a = i * full / n;
        (a, ((i + 1) * full / n).max(a + 1).min(full))
    };
    let rows = par_map_range(wh, |y| {
        let (y0, y1) = span(y, wh, height);
        (0..ww)
            .map(|x| {
                let (x0, x1) = span(x, ww, width);
                let mut sum = 0f32;
                for yy in y0..y1 {
                    let row = &rgba[(yy * width + x0) * 4..(yy * width + x1) * 4];
                    sum += row.chunks_exact(4).map(luma_of).sum::<f32>();
                }
                sum / ((y1 - y0) * (x1 - x0)) as f32
            })
            .collect::<Vec<f32>>()
    });
    rows.concat()
}

/// Bilinear taps for upsampling `n` samples to `full`, pixel centres aligned.
fn taps(n: usize, full: usize) -> Vec<(usize, usize, f32)> {
    (0..full)
        .map(|x| {
            let s = ((x as f32 + 0.5) * n as f32 / full as f32 - 0.5).clamp(0.0, (n - 1) as f32);
            let a = s.floor() as usize;
            (a, (a + 1).min(n - 1), s - a as f32)
        })
        .collect()
}

const WARM: [[f32; 4]; 4] = [
    [0.00, 0.00, 0.00, 0.00],
    [0.35, 0.55, 0.05, 0.00],
    [0.70, 1.00, 0.55, 0.05],
    [1.00, 1.00, 0.92, 0.65],
];
const COOL: [[f32; 4]; 4] = [
    [0.00, 0.00, 0.00, 0.00],
    [0.35, 0.02, 0.12, 0.45],
    [0.70, 0.05, 0.55, 0.95],
    [1.00, 0.70, 0.95, 1.00],
];

/// Piecewise-linear colour ramp; rows are `(position, r, g, b)`.
fn ramp(m: f32, stops: &[[f32; 4]; 4]) -> [f32; 3] {
    let i = stops[1..3].iter().take_while(|s| s[0] <= m).count();
    let (a, b) = (stops[i], stops[i + 1]);
    let t = ((m - a[0]) / (b[0] - a[0]).max(1e-6)).clamp(0.0, 1.0);
    std::array::from_fn(|c| a[c + 1] * (1.0 - t) + b[c + 1] * t)
}

/// Thin-film-like cosine palette: hue rolls with `x`.
fn iridescent(x: f32) -> [f32; 3] {
    [0.0, 0.33, 0.67].map(|o| 0.5 + 0.5 * (std::f32::consts::TAU * (1.1 * x + o)).cos())
}

pub(crate) fn composite(
    smear: &TimeSmear,
    source: &[u8],
    luma: &[f32],
    echo: &[f32],
    width: usize,
    height: usize,
    (ww, wh): (usize, usize),
) -> Vec<u8> {
    let tx = taps(ww, width);
    let ty = taps(wh, height);
    let sample = |plane: &[f32], x: usize, (y0, y1, fy): (usize, usize, f32)| {
        let (x0, x1, fx) = tx[x];
        let top = plane[y0 * ww + x0] * (1.0 - fx) + plane[y0 * ww + x1] * fx;
        let bot = plane[y1 * ww + x0] * (1.0 - fx) + plane[y1 * ww + x1] * fx;
        top * (1.0 - fy) + bot * fy
    };
    let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    let rows = par_map_range(height, |y| {
        let mut row = source[y * width * 4..(y + 1) * width * 4].to_vec();
        for (x, px) in row.chunks_exact_mut(4).enumerate() {
            let e = sample(echo, x, ty[y]);
            match smear.look {
                SmearLook::Faithful => {
                    // Added in byte space, so a zero echo returns the source exactly.
                    let g = smear.gain.unwrap_or(1.0);
                    for c in &mut px[..3] {
                        *c = (*c as f32 + g * e).round().clamp(0.0, 255.0) as u8;
                    }
                }
                SmearLook::Raw => {
                    let v = to_u8((sample(luma, x, ty[y]) + e) / 255.0);
                    px[..3].fill(v);
                }
                SmearLook::Echo => {
                    let g = smear.gain.unwrap_or(4.0);
                    let m = (e.abs() / 255.0 * g).clamp(0.0, 1.0).powf(0.6);
                    let rgb = ramp(if e > 0.0 { m } else { 0.0 }, &WARM);
                    let cool = ramp(if e < 0.0 { m } else { 0.0 }, &COOL);
                    for c in 0..3 {
                        px[c] = to_u8(rgb[c] + cool[c]);
                    }
                }
                SmearLook::Ghost => {
                    let g = smear.gain.unwrap_or(3.0);
                    let pos = (e / 255.0 * g).clamp(0.0, 1.0);
                    let neg = (-e / 255.0 * g).clamp(0.0, 1.0);
                    let glow = iridescent(pos);
                    for c in 0..3 {
                        let s = px[c] as f32 / 255.0;
                        px[c] = to_u8(s * (1.0 - smear.dark * neg) + glow[c] * pos);
                    }
                }
            }
        }
        row
    });
    rows.concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The transform as a dense 64×64 unitary, built from the tensor product
    /// of single-qubit matrices — an independent route to the same numbers.
    fn dense(strength: f64, reach: f64, rotation: Rotation, col: &[f64; WINDOW]) -> [f64; WINDOW] {
        type C = (f64, f64);
        let mul = |a: C, b: C| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
        let gate = |rx: bool, th: f64| -> [[C; 2]; 2] {
            let (c, s) = ((th / 2.0).cos(), (th / 2.0).sin());
            if rx {
                [[(c, 0.0), (0.0, -s)], [(0.0, -s), (c, 0.0)]]
            } else {
                [[(c, 0.0), (-s, 0.0)], [(s, 0.0), (c, 0.0)]]
            }
        };
        let per_qubit: Vec<[[C; 2]; 2]> = (0..QUBITS)
            .map(|k| {
                let th = PI * strength * ((1.0 - reach) * LOCAL_WEIGHTS[k] + reach);
                let mut m = [[(1.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)]];
                for &rx in rotation.gates() {
                    let g = gate(rx, th);
                    let prev = m;
                    for r in 0..2 {
                        for c in 0..2 {
                            let a = mul(g[r][0], prev[0][c]);
                            let b = mul(g[r][1], prev[1][c]);
                            m[r][c] = (a.0 + b.0, a.1 + b.1);
                        }
                    }
                }
                m
            })
            .collect();
        let sum: f64 = col.iter().sum();
        let mut psi = [0f64; WINDOW];
        for t in 0..WINDOW {
            psi[gray(t)] = (col[t] / sum).sqrt();
        }
        let mut out = [0f64; WINDOW];
        for (t, o) in out.iter_mut().enumerate() {
            let j = gray(t);
            let mut acc = (0.0, 0.0);
            for i in 0..WINDOW {
                let mut u = (1.0, 0.0);
                for (k, m) in per_qubit.iter().enumerate() {
                    u = mul(u, m[(j >> k) & 1][(i >> k) & 1]);
                }
                acc.0 += u.0 * psi[i];
                acc.1 += u.1 * psi[i];
            }
            *o = (acc.0 * acc.0 + acc.1 * acc.1) * sum;
        }
        out
    }

    fn columns() -> Vec<[f64; WINDOW]> {
        let mut seed = 7u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed % 256) as f64
        };
        vec![
            std::array::from_fn(|t| {
                40.0 + 30.0 * (t as f64 / 7.0).sin() + if (20..26).contains(&t) { 150.0 } else { 0.0 }
            }),
            std::array::from_fn(|t| if t == 37 { 230.0 } else { 20.0 }),
            std::array::from_fn(|_| rnd()),
        ]
    }

    #[test]
    fn butterflies_match_the_dense_unitary() {
        for rotation in [Rotation::X, Rotation::Y, Rotation::XY, Rotation::YX] {
            for (s, r) in [(0.3, 0.0), (0.5, 0.6), (1.0, 1.0)] {
                let smear = TimeSmear::new().strength(s).reach(r).rotation(rotation);
                for col in columns() {
                    let want = dense(s as f64, r as f64, rotation, &col);
                    let mut got = col;
                    smear.smear_column(&mut got);
                    for t in 0..WINDOW {
                        assert!((got[t] - want[t]).abs() < 1e-9, "{rotation:?} s={s} r={r} t={t}");
                    }
                }
            }
        }
    }

    #[test]
    fn light_is_moved_not_made() {
        let smear = TimeSmear::new().strength(0.7);
        for col in columns() {
            let mut out = col;
            smear.smear_column(&mut out);
            let (a, b): (f64, f64) = (col.iter().sum(), out.iter().sum());
            assert!((a - b).abs() < 1e-9 * a);
            assert!(out.iter().all(|&v| v >= 0.0));
        }
    }

    #[test]
    fn a_flash_echoes_before_it_happens() {
        let smear = TimeSmear::new();
        let mut col: [f64; WINDOW] = std::array::from_fn(|t| if t == 37 { 230.0 } else { 20.0 });
        smear.smear_column(&mut col);
        // Gray(37) neighbours one bit away include frames well before 37.
        let before = (0..37).filter(|&t| col[t] > 20.5).count();
        let after = (38..WINDOW).filter(|&t| col[t] > 20.5).count();
        assert!(before > 0 && after > 0, "before {before}, after {after}");
        assert!(col[37] < 230.0);
    }

    #[test]
    fn not_linear() {
        let smear = TimeSmear::new().strength(0.5);
        let cols = columns();
        let (mut a, mut b) = (cols[0], cols[1]);
        let mut ab: [f64; WINDOW] = std::array::from_fn(|t| a[t] + b[t]);
        smear.smear_column(&mut a);
        smear.smear_column(&mut b);
        smear.smear_column(&mut ab);
        let err: f64 = (0..WINDOW).map(|t| (ab[t] - a[t] - b[t]).abs()).sum();
        assert!(err > 1.0, "superposition error {err}");
    }

    #[test]
    fn names_parse() {
        assert_eq!("XY".parse::<Rotation>(), Ok(Rotation::XY));
        assert_eq!("ghost".parse::<SmearLook>(), Ok(SmearLook::Ghost));
        assert!("blur".parse::<SmearLook>().is_err());
    }

    #[test]
    fn crossfade_weights_sum_to_one() {
        let w = window_weights();
        for i in 0..HOP {
            assert!((w[i] + w[i + HOP] - 1.0).abs() < 1e-6);
        }
        assert_eq!(plan_windows(64), vec![0]);
        assert_eq!(plan_windows(130), vec![0, 32, 64, 66]);
    }

    /// A small clip: a bright square crossing a gradient, plus a still corner.
    fn clip(w: usize, h: usize, n: usize) -> Vec<Vec<u8>> {
        (0..n)
            .map(|i| {
                let cx = 4 + i * (w - 8) / n;
                let mut f = vec![0u8; w * h * 4];
                for y in 0..h {
                    for x in 0..w {
                        let p = (y * w + x) * 4;
                        let v = if x.abs_diff(cx) < 3 && y.abs_diff(h / 2) < 3 { 240 } else { (30 + x) as u8 };
                        f[p..p + 4].copy_from_slice(&[v, v / 2, 255 - v, 255]);
                    }
                }
                f
            })
            .collect()
    }

    #[test]
    fn a_still_clip_comes_back_unchanged() {
        let still = vec![clip(48, 24, 1)[0].clone(); 70];
        let out = TimeSmear::new().strength(0.9).work_width(16).apply(&still, 48, 24);
        assert!(out == still);
    }

    #[test]
    fn streaming_matches_the_whole_volume() {
        let (w, h, n) = (40usize, 20usize, 100usize);
        let frames = clip(w, h, n);
        let smear = TimeSmear::new().work_width(20).look(SmearLook::Raw);
        let (ww, wh) = smear.work_size(w as u32, h as u32);
        let luma: Vec<f32> = frames.iter().flat_map(|f| downsample_luma(f, w, h, (ww, wh))).collect();
        let echo = smear.echo_volume(&luma, ww, wh);
        let streamed = smear.apply(&frames, w as u32, h as u32);
        let px = ww * wh;
        for i in 0..n {
            let want = composite(&smear, &frames[i], &luma[i * px..][..px], &echo[i * px..][..px], w, h, (ww, wh));
            assert!(streamed[i] == want, "frame {i}");
        }
        // Something moved, so something was smeared.
        assert!(echo.iter().any(|e| e.abs() > 1.0));
        // Short clips are padded to a window rather than refused.
        assert_eq!(smear.apply(&frames[..10], w as u32, h as u32).len(), 10);
    }

    #[test]
    fn adaptive_tones_down_slow_tiles() {
        // A faint drift: well under the reference motion, so every tile is
        // scaled down towards the floor.
        let (w, h, n) = (72usize, 36usize, 64usize);
        let luma: Vec<f32> = (0..n)
            .flat_map(|t| (0..w * h).map(move |p| 60.0 + 8.0 * ((p % w) as f32 / 5.0 + t as f32 / 9.0).sin()))
            .collect();
        let plain = TimeSmear::new().strength(0.6);
        let adaptive = plain.clone().adaptive(Adaptive::default());
        let total = |e: Vec<f32>| e.iter().map(|v| v.abs() as f64).sum::<f64>();
        let (a, b) = (total(plain.echo_volume(&luma, w, h)), total(adaptive.echo_volume(&luma, w, h)));
        assert!(b < 0.5 * a, "adaptive {b} vs plain {a}");
        assert!(b > 0.0);
    }
}
