//! NRC-style tiny MLP: per-pixel world features → irradiance residual.
//!
//! Implemented as 1×1 convolutions so it tiles like the U-Net and runs in RLX
//! on the same batch shapes.

use anyhow::{ensure, Result};
use rlx::{grad_with_loss, DType, Device, Graph, GraphExt, Session, Shape};

use crate::model::{ParamSpec, ALBEDO_FLOOR};
use crate::pack::geometry_hits;
use crate::dataset::Dataset;
use crate::train::{Batch, TrainConfig, TRAIN_EPSILON};
use crate::{compress, expand};

/// normal(3) + world(3) + albedo(3) + depth(1) + irradiance(3) [+ ddgi(3)].
pub const NRC_IN: usize = 13;
pub const NRC_IN_DDGI: usize = 16;

/// When `NRC_DDGI=1`, NRC sees DDGI planes 16–18 as extra features.
pub fn nrc_input_channels() -> usize {
    if std::env::var("NRC_DDGI")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    {
        NRC_IN_DDGI
    } else {
        NRC_IN
    }
}

/// Linear irradiance residual per pixel.
pub const NRC_OUT: usize = 3;

/// Cap on the NRC irradiance residual in linear light.
pub const NRC_CAP: f32 = 0.12;

/// Skip NRC on dark surfaces — the irradiance residual is unstable there.
pub const NRC_MIN_ALBEDO: f32 = 0.12;

/// Skip when probe already matches the base colour (relative squared error).
pub const NRC_PROBE_CONFIDENCE: f32 = 0.04;
/// Looser gate for the hops stack — Cornell still differs on colored walls.
pub const HOPS_NRC_PROBE_CONFIDENCE: f32 = 0.022;
/// Slightly larger residual cap when fused on the hops stack.
pub const HOPS_NRC_CAP: f32 = 0.14;

/// Skip on very bright surfaces (furnace / lamp) — residual there is energy leak.
pub const NRC_MAX_PROBE: f32 = 0.72;

const WIDTHS: [usize; 3] = [24, 24, NRC_OUT];

#[derive(Debug, Clone)]
pub struct NrcArch {
    in_channels: usize,
    params: Vec<ParamSpec>,
}

impl Default for NrcArch {
    fn default() -> Self {
        Self::new()
    }
}

impl NrcArch {
    pub fn new() -> Self {
        Self::with_inputs(nrc_input_channels())
    }

    pub fn with_inputs(in_channels: usize) -> Self {
        let mut params = Vec::new();
        let mut in_ch = in_channels;
        for (i, &out) in WIDTHS.iter().enumerate() {
            params.push(ParamSpec {
                name: match i {
                    0 => "nrc0",
                    1 => "nrc1",
                    _ => "nrc_out",
                },
                shape: [out, in_ch, 1, 1],
            });
            in_ch = out;
        }
        Self {
            in_channels,
            params,
        }
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    pub fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    pub fn parameter_count(&self) -> usize {
        self.params.iter().map(|p| p.elems()).sum()
    }

    pub fn forward(&self, graph: &mut Graph, input: rlx::NodeId) -> rlx::NodeId {
        let p: Vec<_> = self
            .params
            .iter()
            .map(|s| graph.param(s.name, Shape::new(&s.shape, DType::F32)))
            .collect();
        let mut x = input;
        for (i, weight) in p.iter().enumerate() {
            x = graph.conv2d(x, *weight, [1, 1], [1, 1], [0, 0], [1, 1], 1);
            if i + 1 < p.len() {
                x = graph.relu(x);
            }
        }
        clamp_irradiance(graph, x, NRC_CAP)
    }
}

fn clamp_irradiance(graph: &mut Graph, x: rlx::NodeId, cap: f32) -> rlx::NodeId {
    soft_clamp_irradiance(graph, x, cap)
}

fn soft_clamp_irradiance(graph: &mut Graph, x: rlx::NodeId, cap: f32) -> rlx::NodeId {
    let c = graph.constant(cap as f64, DType::F32);
    let scaled = graph.div(x, c);
    let t = graph.tanh(scaled);
    graph.mul(c, t)
}

pub fn init_params(arch: &NrcArch, seed: u64) -> Vec<Vec<f32>> {
    arch.params()
        .iter()
        .enumerate()
        .map(|(layer, spec)| {
            if spec.name == "nrc_out" {
                return vec![0.0; spec.elems()];
            }
            let scale = (2.0 / spec.fan_in() as f32).sqrt();
            (0..spec.elems())
                .map(|i| {
                    let h = crate::model::mix64(seed ^ layer as u64 ^ i as u64);
                    ((h >> 40) as f32 / 16_777_216.0 - 0.5) * scale * 2.0
                })
                .collect()
        })
        .collect()
}

/// Mean albedo at a pixel (planes 3–5).
pub fn mean_albedo(input: &[f32], pixels: usize, i: usize) -> f32 {
    (input[3 * pixels + i] + input[4 * pixels + i] + input[5 * pixels + i]) / 3.0
}

/// Mean compressed probe colour (planes 0–2).
pub fn mean_probe(input: &[f32], pixels: usize, i: usize) -> f32 {
    (input[i] + input[pixels + i] + input[2 * pixels + i]) / 3.0
}

/// Per-pixel NRC features from a packed U-Net input.
pub fn nrc_features(input: &[f32], pixels: usize) -> Vec<f32> {
    let in_ch = nrc_input_channels();
    let mut feat = vec![0.0f32; in_ch * pixels];
    for i in 0..pixels {
        for c in 0..3 {
            feat[c * pixels + i] = input[(6 + c) * pixels + i];
            feat[(3 + c) * pixels + i] = input[(19 + c) * pixels + i];
            feat[(6 + c) * pixels + i] = input[(3 + c) * pixels + i];
            feat[(10 + c) * pixels + i] = input[(10 + c) * pixels + i];
        }
        feat[9 * pixels + i] = input[9 * pixels + i];
        if in_ch >= NRC_IN_DDGI {
            for c in 0..3 {
                feat[(13 + c) * pixels + i] = input[(16 + c) * pixels + i];
            }
        }
    }
    feat
}

/// Relative squared error between probe colour and another compressed beauty buffer.
pub fn probe_colour_relative_sq(input: &[f32], other: &[f32], pixels: usize, i: usize) -> f32 {
    let mut err = 0.0f32;
    let mut energy = 0.0f32;
    for c in 0..3 {
        let probe = expand(input[c * pixels + i]);
        let other = expand(other[c * pixels + i]);
        let d = probe - other;
        err += d * d;
        energy += other * other;
    }
    err / (energy + 1e-4)
}

/// Geometry hit with enough albedo for a stable irradiance residual.
pub fn nrc_eligible(input: &[f32], pixels: usize, i: usize) -> bool {
    geometry_hits(input, pixels)[i]
        && mean_albedo(input, pixels, i) >= NRC_MIN_ALBEDO
        && mean_probe(input, pixels, i) < NRC_MAX_PROBE
}

/// Training/infer mask: eligible surface and probe not already close to the target.
pub fn nrc_apply_mask(
    input: &[f32],
    base_colour: &[f32],
    pixels: usize,
) -> Vec<bool> {
    nrc_apply_mask_with(input, base_colour, pixels, NRC_PROBE_CONFIDENCE)
}

pub fn nrc_apply_mask_with(
    input: &[f32],
    base_colour: &[f32],
    pixels: usize,
    min_gap: f32,
) -> Vec<bool> {
    (0..pixels)
        .map(|i| {
            nrc_eligible(input, pixels, i)
                && probe_colour_relative_sq(input, base_colour, pixels, i) >= min_gap
        })
        .collect()
}

/// Build NRC inputs and irradiance targets from a packed frame.
///
/// The MLP learns `(reference − U-Net) / albedo` on pixels where the probe
/// still differs from the U-Net output — the same mask [`fuse_beauty`] uses at
/// inference.
pub fn pack_nrc_batch(
    input: &[f32],
    unet: &[f32],
    reference: &[f32],
    pixels: usize,
) -> (Vec<f32>, Vec<f32>) {
    pack_nrc_batch_with(input, unet, reference, pixels, NRC_PROBE_CONFIDENCE)
}

pub fn pack_nrc_batch_with(
    input: &[f32],
    unet: &[f32],
    reference: &[f32],
    pixels: usize,
    min_gap: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mask = nrc_apply_mask_with(input, unet, pixels, min_gap);
    let feat = nrc_features(input, pixels);
    let mut tgt = vec![0.0f32; NRC_OUT * pixels];
    for i in 0..pixels {
        if !mask[i] {
            continue;
        }
        for c in 0..3 {
            let alb = input[(3 + c) * pixels + i].max(ALBEDO_FLOOR);
            let base = expand(unet[c * pixels + i]);
            let reference = expand(reference[c * pixels + i]);
            tgt[c * pixels + i] = ((reference - base) / alb).clamp(-NRC_CAP, NRC_CAP);
        }
    }
    (feat, tgt)
}

/// Add capped NRC irradiance on top of compressed beauty planes.
///
/// Only eligible geometry pixels where the probe differs from the U-Net output
/// are modified.
pub fn fuse_beauty(
    beauty: &mut [f32],
    nrc_irradiance: &[f32],
    input: &[f32],
    pixels: usize,
) {
    fuse_beauty_with(beauty, nrc_irradiance, input, pixels, NRC_PROBE_CONFIDENCE);
}

pub fn fuse_beauty_with(
    beauty: &mut [f32],
    nrc_irradiance: &[f32],
    input: &[f32],
    pixels: usize,
    min_gap: f32,
) {
    fuse_beauty_with_cap(beauty, nrc_irradiance, input, pixels, min_gap, NRC_CAP);
}

pub fn fuse_beauty_with_cap(
    beauty: &mut [f32],
    nrc_irradiance: &[f32],
    input: &[f32],
    pixels: usize,
    min_gap: f32,
    cap: f32,
) {
    let mask = nrc_apply_mask_with(input, beauty, pixels, min_gap);
    let albedo = &input[3 * pixels..6 * pixels];
    for i in 0..pixels {
        if !mask[i] {
            continue;
        }
        for c in 0..3 {
            let alb = albedo[c * pixels + i].max(ALBEDO_FLOOR);
            let base = expand(beauty[c * pixels + i]);
            let delta = nrc_irradiance[c * pixels + i].clamp(-cap, cap);
            beauty[c * pixels + i] = compress((base + alb * delta).max(0.0));
        }
    }
}

/// Train the NRC MLP on shuffled tiles from a [`Dataset`].
pub fn train_nrc_dataset(
    arch: &NrcArch,
    set: &Dataset,
    batch_n: usize,
    device: Device,
    config: TrainConfig,
    epochs: usize,
    seed: u64,
) -> Result<Vec<Vec<f32>>> {
    let tile = set.tile();
    let batch_n = batch_n.max(1).min(set.len().max(1));
    let mut trainer = NrcTrainer::new(arch.clone(), batch_n, tile, tile, device, config)?;
    let mut indices: Vec<usize> = (0..set.len()).collect();
    let mut rng = seed;
    for _ in 0..epochs {
        for i in (1..indices.len()).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = ((rng >> 32) as usize) % (i + 1);
            indices.swap(i, j);
        }
        for chunk in indices.chunks(batch_n) {
            if chunk.len() != batch_n {
                break;
            }
            let (input, target) = set.gather(chunk)?;
            trainer.step(&Batch {
                n: batch_n,
                h: tile,
                w: tile,
                input: &input,
                target: &target,
            })?;
        }
    }
    Ok(trainer.into_params())
}

/// Train the NRC MLP with the same Adam loop as [`crate::train::Trainer`].
#[allow(clippy::too_many_arguments)]
pub fn train_nrc(
    arch: &NrcArch,
    n: usize,
    h: usize,
    w: usize,
    device: Device,
    config: TrainConfig,
    epochs: usize,
    input: &[f32],
    target: &[f32],
) -> Result<Vec<Vec<f32>>> {
    train_nrc_from(arch, &init_params(arch, 0x4E52_4301), n, h, w, device, config, epochs, input, target)
}

/// Continue training from existing weights (online / per-scene fine-tune).
#[allow(clippy::too_many_arguments)]
pub fn train_nrc_from(
    arch: &NrcArch,
    params: &[Vec<f32>],
    n: usize,
    h: usize,
    w: usize,
    device: Device,
    config: TrainConfig,
    epochs: usize,
    input: &[f32],
    target: &[f32],
) -> Result<Vec<Vec<f32>>> {
    ensure!(
        params.len() == arch.params().len(),
        "NRC params length mismatch"
    );
    let mut trainer = NrcTrainer::from_params(arch.clone(), n, h, w, device, config, params)?;
    let batch = Batch {
        n,
        h,
        w,
        input,
        target,
    };
    for _ in 0..epochs {
        trainer.step(&batch)?;
    }
    Ok(trainer.into_params())
}

struct NrcTrainer {
    arch: NrcArch,
    graph: rlx::CompiledGraph,
    params: Vec<Vec<f32>>,
    moment1: Vec<Vec<f32>>,
    moment2: Vec<Vec<f32>>,
    config: TrainConfig,
    step: u32,
}

impl NrcTrainer {
    fn new(arch: NrcArch, n: usize, h: usize, w: usize, device: Device, config: TrainConfig) -> Result<Self> {
        let params = init_params(&arch, 0x4E52_4301);
        Self::from_params(arch, n, h, w, device, config, &params)
    }

    fn from_params(
        arch: NrcArch,
        n: usize,
        h: usize,
        w: usize,
        device: Device,
        config: TrainConfig,
        params: &[Vec<f32>],
    ) -> Result<Self> {
        ensure!(
            params.len() == arch.params().len(),
            "NRC trainer params length mismatch"
        );
        let mut forward = Graph::new("nrc_loss");
        let in_ch = arch.in_channels();
        let input = forward.input("input", Shape::new(&[n, in_ch, h, w], DType::F32));
        let target = forward.input("target", Shape::new(&[n, NRC_OUT, h, w], DType::F32));
        let predicted = arch.forward(&mut forward, input);
        let residual = forward.sub(predicted, target);
        let squared = forward.mul(residual, residual);
        let scale = forward.mul(target, target);
        let eps = forward.input("loss_eps", Shape::new(&[1], DType::F32));
        let denom = forward.add(scale, eps);
        let relative = forward.div(squared, denom);
        let loss = forward.mean(relative, vec![0, 1, 2, 3], false);
        forward.set_outputs(vec![loss]);
        let wrt: Vec<_> = arch
            .params()
            .iter()
            .map(|s| forward.param(s.name, Shape::new(s.shape.as_ref(), DType::F32)))
            .collect();
        let backward = grad_with_loss(&forward, &wrt);
        let graph = Session::new(device).compile(backward);
        let params: Vec<Vec<f32>> = params
            .iter()
            .zip(arch.params())
            .map(|(values, spec)| {
                ensure!(
                    values.len() == spec.elems(),
                    "NRC {} param size mismatch",
                    spec.name
                );
                Ok(values.clone())
            })
            .collect::<Result<_>>()?;
        let moment1 = params.iter().map(|p| vec![0.0; p.len()]).collect();
        let moment2 = params.iter().map(|p| vec![0.0; p.len()]).collect();
        Ok(Self {
            arch,
            graph,
            params,
            moment1,
            moment2,
            config,
            step: 0,
        })
    }

    fn into_params(self) -> Vec<Vec<f32>> {
        self.params
    }

    fn step(&mut self, batch: &Batch<'_>) -> Result<f32> {
        let in_ch = self.arch.in_channels();
        ensure!(batch.input.len() == batch.n * batch.h * batch.w * in_ch);
        ensure!(batch.target.len() == batch.n * batch.h * batch.w * NRC_OUT);
        for (spec, values) in self.arch.params().iter().zip(&self.params) {
            self.graph.set_param(spec.name, values);
        }
        let seed = [1.0f32];
        let eps = [TRAIN_EPSILON];
        let out = self.graph.run(&[
            ("input", batch.input),
            ("target", batch.target),
            ("loss_eps", &eps),
            ("d_output", &seed),
        ]);
        let loss = out.first().and_then(|v| v.first()).copied().unwrap_or(f32::NAN);
        if !loss.is_finite() {
            return Ok(loss);
        }
        self.step += 1;
        let lr = self.config.learning_rate;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let correction1 = 1.0 - beta1.powi(self.step as i32);
        let correction2 = 1.0 - beta2.powi(self.step as i32);
        for i in 0..self.params.len() {
            let grad = &out[i + 1];
            let p = &mut self.params[i];
            let m = &mut self.moment1[i];
            let v = &mut self.moment2[i];
            for j in 0..p.len() {
                let g = grad.get(j).copied().unwrap_or(0.0);
                m[j] = beta1 * m[j] + (1.0 - beta1) * g;
                v[j] = beta2 * v[j] + (1.0 - beta2) * g * g;
                let m_hat = m[j] / correction1;
                let v_hat = v[j] / correction2;
                p[j] -= lr * m_hat / (v_hat.sqrt() + self.config.epsilon);
            }
        }
        Ok(loss)
    }
}

pub struct NrcNet {
    arch: NrcArch,
    graph: rlx::CompiledGraph,
    params: Vec<Vec<f32>>,
    shape: (usize, usize, usize),
}

impl NrcNet {
    pub fn new(arch: NrcArch, params: Vec<Vec<f32>>, n: usize, h: usize, w: usize, device: Device) -> Result<Self> {
        let mut graph = Graph::new("nrc_forward");
        let in_ch = arch.in_channels();
        let input = graph.input("input", Shape::new(&[n, in_ch, h, w], DType::F32));
        let out = arch.forward(&mut graph, input);
        graph.set_outputs(vec![out]);
        let graph = Session::new(device).compile(graph);
        Ok(Self {
            arch,
            graph,
            params,
            shape: (n, h, w),
        })
    }

    pub fn run(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let (_n, _h, _w) = self.shape;
        for (spec, values) in self.arch.params().iter().zip(&self.params) {
            self.graph.set_param(spec.name, values);
        }
        let out = self.graph.run(&[("input", input)]);
        Ok(out.into_iter().next().unwrap_or_default())
    }

    /// Sequential per-frame fine-tune (online NRC) on enclosed training scenes.
    pub fn finetune_scenes(
        &mut self,
        device: Device,
        side: usize,
        epochs_per_scene: usize,
        config: TrainConfig,
        min_gap: f32,
        scenes: &[(&[f32], &[f32], &[f32])],
    ) -> Result<()> {
        if epochs_per_scene == 0 || scenes.is_empty() {
            return Ok(());
        }
        let pixels = side * side;
        for (input, base, reference) in scenes {
            let (nrc_in, nrc_tgt) =
                pack_nrc_batch_with(input, base, reference, pixels, min_gap);
            self.params = train_nrc_from(
                &self.arch,
                &self.params,
                1,
                side,
                side,
                device,
                config,
                epochs_per_scene,
                &nrc_in,
                &nrc_tgt,
            )?;
            self.shape = (1, side, side);
        }
        Ok(())
    }

    pub fn params(&self) -> &[Vec<f32>] {
        &self.params
    }

    /// Compile and load NRC weights from disk.
    pub fn load(
        path: impl AsRef<std::path::Path>,
        n: usize,
        h: usize,
        w: usize,
        device: Device,
    ) -> Result<Self> {
        let (arch, params) = load_nrc(path)?;
        Self::new(arch, params, n, h, w, device)
    }

    pub fn load_bytes(
        bytes: &[u8],
        n: usize,
        h: usize,
        w: usize,
        device: Device,
    ) -> Result<Self> {
        let (arch, params) = load_nrc_bytes(bytes)?;
        Self::new(arch, params, n, h, w, device)
    }
}

const NRC_MAGIC_V1: &[u8; 8] = b"THRSN001";
const NRC_MAGIC: &[u8; 8] = b"THRSN002";

/// Save trained NRC weights.
pub fn save_nrc(arch: &NrcArch, params: &[Vec<f32>], path: impl AsRef<std::path::Path>) -> Result<()> {
    ensure!(
        params.len() == arch.params().len(),
        "{} tensors for an NRC with {}",
        params.len(),
        arch.params().len()
    );
    let path = path.as_ref();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(NRC_MAGIC);
    bytes.extend_from_slice(&(arch.in_channels() as u32).to_le_bytes());
    bytes.extend_from_slice(&(params.len() as u32).to_le_bytes());
    for (spec, values) in arch.params().iter().zip(params) {
        ensure!(
            values.len() == spec.elems(),
            "{}: {} values for {:?}",
            spec.name,
            values.len(),
            spec.shape
        );
        bytes.extend_from_slice(&(values.len() as u32).to_le_bytes());
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path, bytes)
        .map_err(|e| anyhow::anyhow!("probe: cannot write {}: {e}", path.display()))?;
    Ok(())
}

/// Load NRC weights written by [`save_nrc`].
pub fn load_nrc(path: impl AsRef<std::path::Path>) -> Result<(NrcArch, Vec<Vec<f32>>)> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|e| anyhow::anyhow!("probe: cannot read {}: {e}", path.as_ref().display()))?;
    load_nrc_bytes(&bytes)
}

pub fn load_nrc_bytes(bytes: &[u8]) -> Result<(NrcArch, Vec<Vec<f32>>)> {
    ensure!(bytes.len() >= 16, "NRC weights are {} bytes, too short", bytes.len());
    let magic = &bytes[..8];
    ensure!(
        magic == NRC_MAGIC || magic == NRC_MAGIC_V1,
        "NRC weights do not start with THRSN001/THRSN002"
    );
    let (arch, file_in, mut off) = if magic == NRC_MAGIC {
        ensure!(bytes.len() >= 16, "THRSN002 header truncated");
        let n = u32::from_le_bytes(bytes[8..12].try_into()?) as usize;
        (NrcArch::with_inputs(n), n, 12usize)
    } else {
        (NrcArch::with_inputs(13), 13, 8usize)
    };
    let count = u32::from_le_bytes(bytes[off..off + 4].try_into()?) as usize;
    off += 4;
    ensure!(
        count == arch.params().len(),
        "file has {count} tensors, NRC expects {}",
        arch.params().len()
    );
    let _ = file_in;
    let mut params = Vec::with_capacity(count);
    for spec in arch.params() {
        ensure!(off + 4 <= bytes.len(), "truncated NRC header at {}", spec.name);
        let elems = u32::from_le_bytes(bytes[off..off + 4].try_into()?) as usize;
        off += 4;
        ensure!(
            elems == spec.elems(),
            "{}: file has {elems}, shape {:?} wants {}",
            spec.name,
            spec.shape,
            spec.elems()
        );
        ensure!(off + elems * 4 <= bytes.len(), "truncated NRC values at {}", spec.name);
        let mut values = Vec::with_capacity(elems);
        for _ in 0..elems {
            values.push(f32::from_le_bytes(bytes[off..off + 4].try_into()?));
            off += 4;
        }
        params.push(values);
    }
    Ok((arch, params))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::OUT_CHANNELS;

    #[test]
    fn fuse_skips_low_albedo_and_confident_probe() {
        let pixels = 3;
        let mut beauty = vec![0.5f32; OUT_CHANNELS * pixels];
        let nrc = vec![0.1f32; NRC_OUT * pixels];
        let mut input = vec![0.0f32; crate::IN_CHANNELS * pixels];
        // Geometry + albedo + probe differs from beauty → apply.
        input[9 * pixels] = 0.4;
        input[3 * pixels] = 0.5;
        input[4 * pixels] = 0.5;
        input[5 * pixels] = 0.5;
        input[0] = crate::compress(0.1);
        // Dark surface → skip.
        input[9 * pixels + 1] = 0.4;
        input[3 * pixels + 1] = 0.05;
        // Probe already matches beauty on all channels → skip.
        input[9 * pixels + 2] = 0.4;
        input[3 * pixels + 2] = 0.5;
        input[4 * pixels + 2] = 0.5;
        input[5 * pixels + 2] = 0.5;
        input[2 * pixels + 2] = 0.5;
        input[pixels + 2] = 0.5;
        input[2] = 0.5;
        fuse_beauty(&mut beauty, &nrc, &input, pixels);
        assert!((beauty[0] - 0.5).abs() > 1e-4);
        assert!((beauty[pixels + 1] - 0.5).abs() < 1e-6);
        assert!((beauty[2 * pixels + 2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn fuse_skips_sky_pixels() {
        let pixels = 4;
        let mut beauty = vec![0.5f32; OUT_CHANNELS * pixels];
        let nrc = vec![0.1f32; NRC_OUT * pixels];
        let mut input = vec![0.0f32; crate::IN_CHANNELS * pixels];
        // Hit: compressed depth well below sky threshold.
        input[9 * pixels] = 0.4;
        input[3 * pixels] = 0.8;
        // Sky: miss encodes as ~1.0.
        input[9 * pixels + 1] = 0.999;
        fuse_beauty(&mut beauty, &nrc, &input, pixels);
        assert!((beauty[0] - 0.5).abs() > 1e-4, "geometry pixel should change");
        assert!((beauty[1] - 0.5).abs() < 1e-6, "sky pixel must stay untouched");
    }

    #[test]
    fn pack_nrc_targets_unet_residual() {
        let pixels = 2;
        let mut input = vec![0.0f32; crate::IN_CHANNELS * pixels];
        let mut unet = vec![0.0f32; OUT_CHANNELS * pixels];
        let mut reference = vec![0.0f32; OUT_CHANNELS * pixels];
        for c in 0..3 {
            input[c * pixels] = crate::compress(2.0);
            unet[c * pixels] = crate::compress(0.5);
            reference[c * pixels] = crate::compress(4.0);
        }
        input[3 * pixels] = 0.5;
        input[4 * pixels] = 0.5;
        input[5 * pixels] = 0.5;
        input[9 * pixels] = 0.3;
        input[19 * pixels] = 0.2;
        input[9 * pixels + 1] = 0.999;
        reference[1] = crate::compress(99.0);
        let (_, tgt) = pack_nrc_batch(&input, &unet, &reference, pixels);
        assert!(tgt[0].abs() > 1e-6);
        for c in 0..3 {
            assert!(tgt[c * pixels + 1].abs() < 1e-6);
        }
    }

    #[test]
    fn nrc_weights_round_trip() {
        let arch = NrcArch::new();
        let params = init_params(&arch, 9);
        let path = std::env::temp_dir().join("threers-nrc-weights.bin");
        save_nrc(&arch, &params, &path).unwrap();
        let (_, loaded) = load_nrc(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded, params);
    }

    #[test]
    fn nrc_starts_at_zero_residual() {
        let arch = NrcArch::new();
        let params = init_params(&arch, 1);
        let mut net = NrcNet::new(arch, params, 1, 4, 4, Device::Cpu).unwrap();
        let mut input = vec![0.0f32; NRC_IN * 16];
        for i in 0..16 {
            input[i] = 0.5;
            input[16 + i] = 0.2;
        }
        let out = net.run(&input).unwrap();
        assert!(out.iter().all(|v| v.abs() < 1e-6));
    }
}
