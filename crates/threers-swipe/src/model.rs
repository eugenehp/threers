//! The network: a small temporal convolution net that never learns where
//! the keys are.
//!
//! ```text
//! features [T × F] ─ linear ─► h ─┬─ dilated conv (k 3) ─ ReLU ─ 1×1 ─(+)─► … ─► head
//!                                 └──────────────────────────────────┘ ×L
//! head, per point: gate g, blank b, aim offset (ox, oy), spread s
//! logits: blank = b,  key k = g − |aim − key_k|² / 2σ²,  aim = point + (ox, oy),  σ = 0.1 + softplus(s)
//! ```
//!
//! The key centres come in with the swipe, so the class count is the
//! keyboard's key count and any keyboard works: the network says *where*
//! the finger was aiming and *how sure* it is, the keyboard says which
//! keys are there. Trained with [`ctc`](crate::ctc); the backward pass is
//! written out by hand (no autodiff), checked against finite differences.
//!
//! Weights save as safetensors (`input.weight`, `blocks.N.conv.weight`, …,
//! the [`ModelConfig`] as JSON in the metadata), so other runtimes, RLX
//! included, can load them.

use std::collections::HashMap;
use std::ops::Range;

use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use serde::{Deserialize, Serialize};

use crate::synth::Rng;
use crate::trace::FEATURES;

/// Head outputs per point: gate, blank, aim x, aim y, spread.
const HEAD: usize = 5;
const KERNEL: usize = 3;
const MIN_SIGMA: f32 = 0.1;

/// The network's size.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Width of every layer.
    pub channels: usize,
    /// One residual block per entry, its convolution this dilated.
    pub dilations: Vec<usize>,
}

impl ModelConfig {
    /// About 223 K parameters, 0.9 MB f32; each point sees 32 either side
    /// of it, half the swipe.
    pub fn small() -> Self {
        Self {
            channels: 96,
            dilations: vec![1, 2, 4, 8, 16, 1],
        }
    }

    /// For tests and quick tries.
    pub fn tiny() -> Self {
        Self {
            channels: 24,
            dilations: vec![1, 2, 4],
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::small()
    }
}

#[derive(Debug, Clone)]
struct Block {
    conv_w: Range<usize>,
    conv_b: Range<usize>,
    mix_w: Range<usize>,
    mix_b: Range<usize>,
    dilation: usize,
}

#[derive(Debug, Clone)]
struct Offsets {
    in_w: Range<usize>,
    in_b: Range<usize>,
    blocks: Vec<Block>,
    head_w: Range<usize>,
    head_b: Range<usize>,
    len: usize,
}

impl Offsets {
    fn new(cfg: &ModelConfig) -> Self {
        let c = cfg.channels;
        let mut at = 0;
        let mut take = |n: usize| {
            let r = at..at + n;
            at += n;
            r
        };
        let in_w = take(c * FEATURES);
        let in_b = take(c);
        let blocks = cfg
            .dilations
            .iter()
            .map(|&d| Block {
                conv_w: take(KERNEL * c * c),
                conv_b: take(c),
                mix_w: take(c * c),
                mix_b: take(c),
                dilation: d.max(1),
            })
            .collect();
        let head_w = take(HEAD * c);
        let head_b = take(HEAD);
        Self {
            in_w,
            in_b,
            blocks,
            head_w,
            head_b,
            len: at,
        }
    }
}

/// What a forward pass keeps for the backward one.
#[derive(Debug, Clone)]
pub struct Activations {
    /// `h` before each block and after the last: `[T × C]` each.
    h: Vec<Vec<f32>>,
    /// Each block's convolution, before ReLU.
    u: Vec<Vec<f32>>,
    /// `[T × HEAD]`.
    head: Vec<f32>,
    /// `[T × (keys + 1)]`, blank first.
    pub logits: Vec<f32>,
}

/// The network and its weights.
#[derive(Debug, Clone)]
pub struct Model {
    pub config: ModelConfig,
    /// Every weight, one after another (see [`tensors`](Self::tensors)).
    pub params: Vec<f32>,
    offs: Offsets,
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `a · b` in eight running sums, which the compiler keeps in vector
/// registers (one sum in order cannot be vectorised).
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let (ac, ar) = a.as_chunks::<8>();
    let (bc, br) = b.as_chunks::<8>();
    let mut acc = [0.0f32; 8];
    for (x, y) in ac.iter().zip(bc) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for (x, y) in ar.iter().zip(br) {
        s += x * y;
    }
    s
}

/// Four dot products of `w` at once: each load of `w` used four times.
#[inline]
fn dot4(w: &[f32], x: [&[f32]; 4]) -> [f32; 4] {
    let (wc, wr) = w.as_chunks::<8>();
    let xc = x.map(|v| v.as_chunks::<8>().0);
    let mut acc = [[0.0f32; 8]; 4];
    for (j, wv) in wc.iter().enumerate() {
        for f in 0..4 {
            let xv = &xc[f][j];
            for k in 0..8 {
                acc[f][k] += wv[k] * xv[k];
            }
        }
    }
    let tail = wc.len() * 8;
    let mut out = acc.map(|a| a.iter().sum::<f32>());
    for (i, wv) in wr.iter().enumerate() {
        for f in 0..4 {
            out[f] += wv * x[f][tail + i];
        }
    }
    out
}

/// For every frame `t` with `t + shift` in range:
/// `y[t][o] += Σ_i w[o][i] · x[t + shift][i]` (rows of `n_in` and `n_out`),
/// four frames to a load of each weight row.
fn frames_add(w: &[f32], x: &[f32], y: &mut [f32], n_in: usize, n_out: usize, shift: isize) {
    let frames = (x.len() / n_in) as isize;
    let (t0, t1) = ((-shift).max(0), (frames - shift).min(frames));
    if t0 >= t1 {
        return;
    }
    let row = |t: isize| &x[(t + shift) as usize * n_in..(t + shift + 1) as usize * n_in];
    for o in 0..n_out {
        let wr = &w[o * n_in..(o + 1) * n_in];
        let mut t = t0;
        while t + 4 <= t1 {
            let d = dot4(wr, [row(t), row(t + 1), row(t + 2), row(t + 3)]);
            for (f, v) in d.iter().enumerate() {
                y[(t as usize + f) * n_out + o] += v;
            }
            t += 4;
        }
        while t < t1 {
            y[t as usize * n_out + o] += dot(wr, row(t));
            t += 1;
        }
    }
}

/// `y[o] += Σ_i w[o·n_in + i] · x[i]`.
#[inline]
fn matvec_add(w: &[f32], x: &[f32], y: &mut [f32]) {
    let n_in = x.len();
    for (o, yo) in y.iter_mut().enumerate() {
        *yo += dot(&w[o * n_in..(o + 1) * n_in], x);
    }
}

/// `dx[i] += Σ_o w[o·n_in + i] · dy[o]`.
#[inline]
fn matvec_t_add(w: &[f32], dy: &[f32], dx: &mut [f32]) {
    let n_in = dx.len();
    for (o, &g) in dy.iter().enumerate() {
        if g == 0.0 {
            continue;
        }
        let row = &w[o * n_in..(o + 1) * n_in];
        for (d, a) in dx.iter_mut().zip(row) {
            *d += a * g;
        }
    }
}

/// `dw[o·n_in + i] += dy[o] · x[i]`.
#[inline]
fn outer_add(dw: &mut [f32], dy: &[f32], x: &[f32]) {
    let n_in = x.len();
    for (o, &g) in dy.iter().enumerate() {
        if g == 0.0 {
            continue;
        }
        let row = &mut dw[o * n_in..(o + 1) * n_in];
        for (d, a) in row.iter_mut().zip(x) {
            *d += g * a;
        }
    }
}

impl Model {
    /// A fresh network, weights drawn from `seed`.
    pub fn new(config: ModelConfig, seed: u64) -> Self {
        let offs = Offsets::new(&config);
        let mut params = vec![0.0f32; offs.len];
        let mut rng = Rng::new(seed);
        let c = config.channels as f32;
        let depth = config.dilations.len().max(1) as f32;
        let mut fill = |r: &Range<usize>, std: f32| {
            for p in &mut params[r.clone()] {
                *p = rng.normal() * std;
            }
        };
        fill(&offs.in_w, (2.0 / FEATURES as f32).sqrt());
        for b in &offs.blocks {
            fill(&b.conv_w, (2.0 / (KERNEL as f32 * c)).sqrt());
            fill(&b.mix_w, (1.0 / c).sqrt() * 0.5 / depth.sqrt());
        }
        fill(&offs.head_w, 0.01);
        // σ starts near 0.45 keys: each point favours the keys around it.
        params[offs.head_b.start + 4] = -0.86;
        Self {
            config,
            params,
            offs,
        }
    }

    /// How many weights.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Logits for a swipe's [`features`](crate::trace::features) against
    /// `keys` (centres in the same units): `[FRAMES × (keys + 1)]`, blank
    /// first, then the keys in order.
    pub fn logits(&self, x: &[f32], keys: &[(f32, f32)]) -> Vec<f32> {
        self.forward(x, keys).logits
    }

    pub fn forward(&self, x: &[f32], keys: &[(f32, f32)]) -> Activations {
        let c = self.config.channels;
        let t_len = x.len() / FEATURES;
        let p = &self.params;
        let mut h0 = vec![0.0f32; t_len * c];
        for t in 0..t_len {
            let y = &mut h0[t * c..(t + 1) * c];
            y.copy_from_slice(&p[self.offs.in_b.clone()]);
            matvec_add(
                &p[self.offs.in_w.clone()],
                &x[t * FEATURES..(t + 1) * FEATURES],
                y,
            );
        }
        let mut hs = vec![h0];
        let mut us = vec![];
        for b in &self.offs.blocks {
            let h = hs.last().unwrap();
            let mut u = vec![0.0f32; t_len * c];
            let w = &p[b.conv_w.clone()];
            for ut in u.chunks_mut(c) {
                ut.copy_from_slice(&p[b.conv_b.clone()]);
            }
            for tap in 0..KERNEL {
                let shift = (tap as isize - 1) * b.dilation as isize;
                frames_add(&w[tap * c * c..(tap + 1) * c * c], h, &mut u, c, c, shift);
            }
            let relu: Vec<f32> = u.iter().map(|v| v.max(0.0)).collect();
            let mut next = h.clone();
            for y in next.chunks_mut(c) {
                for (yi, bi) in y.iter_mut().zip(&p[b.mix_b.clone()]) {
                    *yi += bi;
                }
            }
            frames_add(&p[b.mix_w.clone()], &relu, &mut next, c, c, 0);
            us.push(u);
            hs.push(next);
        }
        let last = hs.last().unwrap();
        let mut head = vec![0.0f32; t_len * HEAD];
        for t in 0..t_len {
            let y = &mut head[t * HEAD..(t + 1) * HEAD];
            y.copy_from_slice(&p[self.offs.head_b.clone()]);
            matvec_add(&p[self.offs.head_w.clone()], &last[t * c..(t + 1) * c], y);
        }
        let classes = keys.len() + 1;
        let mut logits = vec![0.0f32; t_len * classes];
        for t in 0..t_len {
            let o = &head[t * HEAD..(t + 1) * HEAD];
            let sigma = MIN_SIGMA + softplus(o[4]);
            let inv = 1.0 / (2.0 * sigma * sigma);
            let (px, py) = (x[t * FEATURES] + o[2], x[t * FEATURES + 1] + o[3]);
            let row = &mut logits[t * classes..(t + 1) * classes];
            row[0] = o[1];
            for (r, &(kx, ky)) in row[1..].iter_mut().zip(keys) {
                let d2 = (px - kx) * (px - kx) + (py - ky) * (py - ky);
                *r = o[0] - d2 * inv;
            }
        }
        Activations {
            h: hs,
            u: us,
            head,
            logits,
        }
    }

    /// Adds the gradient of the loss to `grad` (as long as
    /// [`params`](Self::params)), given the loss's gradient with respect
    /// to the logits.
    pub fn backward(
        &self,
        x: &[f32],
        keys: &[(f32, f32)],
        act: &Activations,
        dlogits: &[f32],
        grad: &mut [f32],
    ) {
        let c = self.config.channels;
        let t_len = x.len() / FEATURES;
        let classes = keys.len() + 1;
        let p = &self.params;

        // Logits → head.
        let mut dhead = vec![0.0f32; t_len * HEAD];
        for t in 0..t_len {
            let o = &act.head[t * HEAD..(t + 1) * HEAD];
            let sigma = MIN_SIGMA + softplus(o[4]);
            let s2 = sigma * sigma;
            let (px, py) = (x[t * FEATURES] + o[2], x[t * FEATURES + 1] + o[3]);
            let dl = &dlogits[t * classes..(t + 1) * classes];
            let (mut dg, mut dpx, mut dpy, mut dsig) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for (&g, &(kx, ky)) in dl[1..].iter().zip(keys) {
                let (ex, ey) = (px - kx, py - ky);
                dg += g;
                dpx -= g * ex / s2;
                dpy -= g * ey / s2;
                dsig += g * (ex * ex + ey * ey) / (s2 * sigma);
            }
            let d = &mut dhead[t * HEAD..(t + 1) * HEAD];
            d[0] = dg;
            d[1] = dl[0];
            d[2] = dpx;
            d[3] = dpy;
            d[4] = dsig * sigmoid(o[4]);
        }

        // Head → last h.
        let last = act.h.last().unwrap();
        let mut dh = vec![0.0f32; t_len * c];
        {
            let (hw, hb) = (self.offs.head_w.clone(), self.offs.head_b.clone());
            for t in 0..t_len {
                let dy = &dhead[t * HEAD..(t + 1) * HEAD];
                for (g, d) in grad[hb.clone()].iter_mut().zip(dy) {
                    *g += d;
                }
                outer_add(&mut grad[hw.clone()], dy, &last[t * c..(t + 1) * c]);
                matvec_t_add(&p[hw.clone()], dy, &mut dh[t * c..(t + 1) * c]);
            }
        }

        // Blocks, last first: h' = h + mix(relu(conv(h))).
        let mut a = vec![0.0f32; c];
        let mut da = vec![0.0f32; c];
        for (l, b) in self.offs.blocks.iter().enumerate().rev() {
            let h = &act.h[l];
            let u = &act.u[l];
            let mut du = vec![0.0f32; t_len * c];
            for t in 0..t_len {
                let dv = &dh[t * c..(t + 1) * c];
                let ut = &u[t * c..(t + 1) * c];
                for (ai, &ui) in a.iter_mut().zip(ut) {
                    *ai = ui.max(0.0);
                }
                for (g, d) in grad[b.mix_b.clone()].iter_mut().zip(dv) {
                    *g += d;
                }
                outer_add(&mut grad[b.mix_w.clone()], dv, &a);
                da.fill(0.0);
                matvec_t_add(&p[b.mix_w.clone()], dv, &mut da);
                for ((d, &ai), &ui) in du[t * c..(t + 1) * c].iter_mut().zip(&da).zip(ut) {
                    *d = if ui > 0.0 { ai } else { 0.0 };
                }
            }
            // The residual passes dh on; the convolution adds to it.
            let w = &p[b.conv_w.clone()];
            for t in 0..t_len {
                let dut = &du[t * c..(t + 1) * c];
                for (g, d) in grad[b.conv_b.clone()].iter_mut().zip(dut) {
                    *g += d;
                }
                for tap in 0..KERNEL {
                    let src = t as isize + (tap as isize - 1) * b.dilation as isize;
                    if src < 0 || src >= t_len as isize {
                        continue;
                    }
                    let s = src as usize;
                    let range = b.conv_w.start + tap * c * c..b.conv_w.start + (tap + 1) * c * c;
                    outer_add(&mut grad[range], dut, &h[s * c..(s + 1) * c]);
                    matvec_t_add(
                        &w[tap * c * c..(tap + 1) * c * c],
                        dut,
                        &mut dh[s * c..(s + 1) * c],
                    );
                }
            }
        }

        // Input layer.
        let (iw, ib) = (self.offs.in_w.clone(), self.offs.in_b.clone());
        for t in 0..t_len {
            let dy = &dh[t * c..(t + 1) * c];
            for (g, d) in grad[ib.clone()].iter_mut().zip(dy) {
                *g += d;
            }
            outer_add(
                &mut grad[iw.clone()],
                dy,
                &x[t * FEATURES..(t + 1) * FEATURES],
            );
        }
    }

    /// Every weight tensor: its name, shape and place in
    /// [`params`](Self::params).
    pub fn tensors(&self) -> Vec<(String, Vec<usize>, Range<usize>)> {
        let c = self.config.channels;
        let o = &self.offs;
        let mut out = vec![
            ("input.weight".into(), vec![c, FEATURES], o.in_w.clone()),
            ("input.bias".into(), vec![c], o.in_b.clone()),
        ];
        for (i, b) in o.blocks.iter().enumerate() {
            out.push((
                format!("blocks.{i}.conv.weight"),
                vec![KERNEL, c, c],
                b.conv_w.clone(),
            ));
            out.push((format!("blocks.{i}.conv.bias"), vec![c], b.conv_b.clone()));
            out.push((
                format!("blocks.{i}.mix.weight"),
                vec![c, c],
                b.mix_w.clone(),
            ));
            out.push((format!("blocks.{i}.mix.bias"), vec![c], b.mix_b.clone()));
        }
        out.push(("head.weight".into(), vec![HEAD, c], o.head_w.clone()));
        out.push(("head.bias".into(), vec![HEAD], o.head_b.clone()));
        out
    }

    /// The weights as safetensors, the config in the metadata.
    pub fn to_safetensors(&self) -> Result<Vec<u8>, String> {
        let bytes: Vec<(String, Vec<usize>, Vec<u8>)> = self
            .tensors()
            .into_iter()
            .map(|(n, s, r)| {
                (
                    n,
                    s,
                    self.params[r]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect(),
                )
            })
            .collect();
        let views = bytes
            .iter()
            .map(|(n, s, b)| {
                Ok((
                    n.clone(),
                    TensorView::new(Dtype::F32, s.clone(), b).map_err(|e| e.to_string())?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut meta = HashMap::new();
        meta.insert(
            "config".into(),
            serde_json::to_string(&self.config).map_err(|e| e.to_string())?,
        );
        meta.insert("format".into(), "threers-swipe/1".into());
        safetensors::serialize(views, Some(meta)).map_err(|e| e.to_string())
    }

    pub fn from_safetensors(bytes: &[u8]) -> Result<Self, String> {
        let (_, meta) = SafeTensors::read_metadata(bytes).map_err(|e| e.to_string())?;
        let config: ModelConfig = meta
            .metadata()
            .as_ref()
            .and_then(|m| m.get("config"))
            .ok_or("no config in the metadata: not a threers-swipe model")
            .and_then(|s| serde_json::from_str(s).map_err(|_| "unreadable config"))?;
        let st = SafeTensors::deserialize(bytes).map_err(|e| e.to_string())?;
        let mut model = Self::new(config, 0);
        for (name, shape, range) in model.tensors() {
            let t = st.tensor(&name).map_err(|e| format!("{name}: {e}"))?;
            if t.dtype() != Dtype::F32 || t.shape() != shape.as_slice() {
                return Err(format!(
                    "{name}: expected f32 {shape:?}, found {:?} {:?}",
                    t.dtype(),
                    t.shape()
                ));
            }
            for (p, b) in model.params[range]
                .iter_mut()
                .zip(t.data().as_chunks::<4>().0)
            {
                *p = f32::from_le_bytes(*b);
            }
        }
        Ok(model)
    }

    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        std::fs::write(path, self.to_safetensors()?).map_err(|e| e.to_string())
    }

    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        Self::from_safetensors(&std::fs::read(path).map_err(|e| e.to_string())?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ctc;
    use crate::trace::{features, TouchPoint, FRAMES};

    fn sample() -> (Vec<f32>, Vec<(f32, f32)>, Vec<usize>) {
        let pts: Vec<_> = (0..30)
            .map(|i| {
                let s = i as f32 / 29.0;
                TouchPoint::new(-2.0 + 4.0 * s, (s * 6.0).sin() * 0.5, s * 0.6)
            })
            .collect();
        let keys = vec![(-2.0, 0.0), (0.0, 0.5), (2.0, 0.0), (0.0, -1.0)];
        (features(&pts), keys, vec![1, 2, 3])
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let (x, keys, target) = sample();
        let mut m = Model::new(ModelConfig::tiny(), 5);
        // Make the head matter, so its gradients are not all tiny.
        for v in &mut m.params[m.offs.head_w.clone()] {
            *v *= 30.0;
        }
        let loss = |m: &Model| {
            let z = m.logits(&x, &keys);
            ctc::loss(&z, keys.len() + 1, &target).unwrap().0 as f64
        };
        let act = m.forward(&x, &keys);
        let (_, dz) = ctc::loss(&act.logits, keys.len() + 1, &target).unwrap();
        let mut g = vec![0.0f32; m.len()];
        m.backward(&x, &keys, &act, &dz, &mut g);
        // A ReLU input a hair from zero is crossed by any finite step, and
        // every weight feeding it then reads the average of two slopes; a
        // wrong backward pass would show on far more than a few.
        let (mut checked, mut off) = (0, vec![]);
        let mut rng = Rng::new(11);
        for _ in 0..400 {
            let i = rng.below(m.len());
            let h = 1e-3f32;
            let orig = m.params[i];
            m.params[i] = orig + h;
            let up = loss(&m);
            m.params[i] = orig - h;
            let down = loss(&m);
            m.params[i] = orig;
            let fd = (up - down) / (2.0 * h as f64);
            let tol = 2e-2 * fd.abs().max(g[i].abs() as f64) + 2e-3;
            if (fd - g[i] as f64).abs() >= tol {
                off.push((i, fd, g[i]));
            }
            checked += (fd.abs() > 1e-3) as usize;
        }
        assert!(
            off.len() <= 8,
            "{} of 400 gradients off: {:?}",
            off.len(),
            &off[..off.len().min(8)]
        );
        assert!(checked > 50, "only {checked} weights moved the loss");
    }

    #[test]
    fn logits_shape_and_saving() {
        let (x, keys, _) = sample();
        let m = Model::new(ModelConfig::tiny(), 1);
        assert_eq!(m.logits(&x, &keys).len(), FRAMES * 5);
        let back = Model::from_safetensors(&m.to_safetensors().unwrap()).unwrap();
        assert_eq!(back.config, m.config);
        assert_eq!(back.params, m.params);
        assert!(Model::from_safetensors(b"nonsense").is_err());
    }
}
