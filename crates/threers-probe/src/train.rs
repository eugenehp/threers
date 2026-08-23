//! Training: relative L2, autodiff, Adam.
//!
//! One graph does both passes. The forward graph ends at a scalar loss;
//! `grad_with_loss` rewrites it so every step is bind, run, Adam.

use anyhow::{ensure, Result};
use rlx::{grad_with_loss, DType, Device, Graph, GraphExt, Session, Shape};

use crate::model::{init_params, ProbeArch, OUT_CHANNELS};

/// Floor in the relative-L2 denominator, for reporting.
pub const LOSS_EPSILON: f32 = 0.01;
/// Floor used while training — low enough that dark pixels stay relative.
pub const TRAIN_EPSILON: f32 = 1e-3;
/// Extra weight on pixels where the probe already disagrees with the reference
/// (i.e. where GI lives). 0 is uniform relative L2.
pub const GI_LOSS_BOOST: f32 = 2.5;
/// Extra GI pixel weight when training the hops head.
pub const HOPS_GI_LOSS_BOOST: f32 = 3.0;
/// Extra weight on saturated albedo (colored Cornell walls) when training hops.
pub const HOPS_CHROMA_LOSS_BOOST: f32 = 1.25;
/// Extra weight on neutral albedo (floor/ceiling) — where Cornell bleed error lives.
pub const HOPS_NEUTRAL_LOSS_BOOST: f32 = 1.5;

/// Optimiser settings.
#[derive(Debug, Clone, Copy)]
pub struct TrainConfig {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    /// Total steps for cosine decay. 0 holds the rate constant.
    pub total_steps: u32,
    pub final_lr_fraction: f32,
    /// Clip the global gradient norm. 0 disables.
    pub grad_clip: f32,
    /// Weight on pixels where probe ≠ reference. 0 = uniform relative L2.
    pub gi_loss_boost: f32,
    /// Extra weight on high-chroma albedo pixels (colored walls). 0 disables.
    pub chroma_loss_boost: f32,
    /// Extra weight on low-chroma albedo pixels (neutral floor/ceiling). 0 disables.
    pub neutral_loss_boost: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            total_steps: 0,
            final_lr_fraction: 0.05,
            grad_clip: 1.0,
            gi_loss_boost: GI_LOSS_BOOST,
            chroma_loss_boost: 0.0,
            neutral_loss_boost: 0.0,
        }
    }
}

/// `n` tiles of `h × w`. `input` is `[n, IN, h, w]`, `target` `[n, 3, h, w]`.
pub struct Batch<'a> {
    pub n: usize,
    pub h: usize,
    pub w: usize,
    pub input: &'a [f32],
    pub target: &'a [f32],
}

impl Batch<'_> {
    fn check(&self) -> Result<()> {
        let pixels = self.n * self.h * self.w;
        ensure!(
            self.input.len() == pixels * crate::IN_CHANNELS,
            "input is {} floats, expected {}",
            self.input.len(),
            pixels * crate::IN_CHANNELS
        );
        ensure!(
            self.target.len() == pixels * OUT_CHANNELS,
            "target is {} floats, expected {}",
            self.target.len(),
            pixels * OUT_CHANNELS
        );
        Ok(())
    }
}

/// A compiled training step, its parameters, and Adam state.
pub struct Trainer {
    net: ProbeArch,
    graph: rlx::CompiledGraph,
    params: Vec<Vec<f32>>,
    moment1: Vec<Vec<f32>>,
    moment2: Vec<Vec<f32>>,
    config: TrainConfig,
    tile: (usize, usize, usize),
    step: u32,
}

impl Trainer {
    pub fn new(
        net: ProbeArch,
        n: usize,
        h: usize,
        w: usize,
        device: Device,
        config: TrainConfig,
    ) -> Result<Self> {
        ensure!(n > 0 && h > 0 && w > 0, "empty batch shape {n}x{h}x{w}");
        ensure!(
            h.is_multiple_of(ProbeArch::TILE_MULTIPLE) && w.is_multiple_of(ProbeArch::TILE_MULTIPLE),
            "tile {h}x{w} must be a multiple of {}",
            ProbeArch::TILE_MULTIPLE
        );

        let mut forward = Graph::new("probe_loss");
        let input = forward.input(
            "input",
            Shape::new(&[n, crate::IN_CHANNELS, h, w], DType::F32),
        );
        let target = forward.input("target", Shape::new(&[n, OUT_CHANNELS, h, w], DType::F32));
        let predicted = net.forward(&mut forward, input, n, h, w);

        let residual = forward.sub(predicted, target);
        let squared = forward.mul(residual, residual);
        let scale = forward.mul(target, target);
        let eps = forward.input("loss_eps", Shape::new(&[1], DType::F32));
        let denom = forward.add(scale, eps);
        let relative = forward.div(squared, denom);
        let probe = forward.narrow_(input, 1, 0, OUT_CHANNELS);
        let gap = forward.sub(target, probe);
        let gap2 = forward.mul(gap, gap);
        let rel_gap = forward.div(gap2, denom);
        let one = forward.constant(1.0, DType::F32);
        let boost = forward.constant(config.gi_loss_boost as f64, DType::F32);
        let boosted = forward.mul(boost, rel_gap);
        let mut weight = forward.add(one, boosted);
        let albedo_var = |forward: &mut Graph, input: rlx::NodeId| -> rlx::NodeId {
            let ar = forward.narrow_(input, 1, 3, 1);
            let ag = forward.narrow_(input, 1, 4, 1);
            let ab = forward.narrow_(input, 1, 5, 1);
            let three = forward.constant(3.0, DType::F32);
            let sum_a = forward.add(ar, ag);
            let sum_a = forward.add(sum_a, ab);
            let mean_a = forward.div(sum_a, three);
            let dr = forward.sub(ar, mean_a);
            let dg = forward.sub(ag, mean_a);
            let db = forward.sub(ab, mean_a);
            let dr2 = forward.mul(dr, dr);
            let dg2 = forward.mul(dg, dg);
            let db2 = forward.mul(db, db);
            let chroma_sum = forward.add(dr2, dg2);
            let chroma_sum = forward.add(chroma_sum, db2);
            forward.div(chroma_sum, three)
        };
        if config.chroma_loss_boost > 0.0 {
            let chroma = albedo_var(&mut forward, input);
            let chroma_boost = forward.constant(config.chroma_loss_boost as f64, DType::F32);
            let chroma_w = forward.mul(chroma_boost, chroma);
            weight = forward.add(weight, chroma_w);
        }
        if config.neutral_loss_boost > 0.0 {
            let var = albedo_var(&mut forward, input);
            let var_thresh = forward.constant(0.012f64, DType::F32);
            let var_delta = forward.sub(var_thresh, var);
            let neutral = forward.relu(var_delta);
            let inv = forward.constant(1.0 / 0.012, DType::F32);
            let neutral = forward.mul(neutral, inv);
            let neutral_boost = forward.constant(config.neutral_loss_boost as f64, DType::F32);
            let neutral_w = forward.mul(neutral_boost, neutral);
            weight = forward.add(weight, neutral_w);
        }
        let weighted = forward.mul(relative, weight);
        let loss = forward.mean(weighted, vec![0, 1, 2, 3], false);
        forward.set_outputs(vec![loss]);

        let wrt: Vec<_> = net
            .params()
            .iter()
            .map(|s| forward.param(s.name, Shape::new(s.shape.as_ref(), DType::F32)))
            .collect();
        let backward = grad_with_loss(&forward, &wrt);
        let graph = Session::new(device).compile(backward);

        let params = init_params(&net, 0x5eed_1234);
        let moment1 = params.iter().map(|p| vec![0.0; p.len()]).collect();
        let moment2 = params.iter().map(|p| vec![0.0; p.len()]).collect();

        Ok(Self {
            net,
            graph,
            params,
            moment1,
            moment2,
            config,
            tile: (n, h, w),
            step: 0,
        })
    }

    pub fn net(&self) -> &ProbeArch {
        &self.net
    }

    pub fn params(&self) -> &[Vec<f32>] {
        &self.params
    }

    pub fn set_params(&mut self, params: Vec<Vec<f32>>) -> Result<()> {
        ensure!(
            params.len() == self.params.len()
                && params
                    .iter()
                    .zip(&self.params)
                    .all(|(a, b)| a.len() == b.len()),
            "parameter shapes do not match this network"
        );
        self.params = params;
        Ok(())
    }

    pub fn steps_taken(&self) -> u32 {
        self.step
    }

    /// One optimiser step. Returns the loss of the weights that produced the
    /// gradient (before the update).
    pub fn step(&mut self, batch: &Batch<'_>) -> Result<f32> {
        batch.check()?;
        let (n, h, w) = self.tile;
        ensure!(
            batch.n == n && batch.h == h && batch.w == w,
            "batch is {}x{}x{}, the graph was compiled for {n}x{h}x{w}",
            batch.n,
            batch.h,
            batch.w
        );

        for (spec, values) in self.net.params().iter().zip(&self.params) {
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

        let loss = out
            .first()
            .and_then(|v| v.first())
            .copied()
            .unwrap_or(f32::NAN);
        if !loss.is_finite() {
            return Ok(loss);
        }
        ensure!(
            out.len() == self.params.len() + 1,
            "expected {} gradients, got {}",
            self.params.len(),
            out.len().saturating_sub(1)
        );

        let scale = self.gradient_scale(&out[1..]);
        self.step += 1;
        let TrainConfig {
            beta1,
            beta2,
            epsilon,
            ..
        } = self.config;
        let learning_rate = self.current_learning_rate();
        let correction1 = 1.0 - beta1.powi(self.step as i32);
        let correction2 = 1.0 - beta2.powi(self.step as i32);

        for i in 0..self.params.len() {
            let grad = &out[i + 1];
            let p = &mut self.params[i];
            let m = &mut self.moment1[i];
            let v = &mut self.moment2[i];
            for j in 0..p.len() {
                let g = grad.get(j).copied().unwrap_or(0.0) * scale;
                m[j] = beta1 * m[j] + (1.0 - beta1) * g;
                v[j] = beta2 * v[j] + (1.0 - beta2) * g * g;
                let m_hat = m[j] / correction1;
                let v_hat = v[j] / correction2;
                p[j] -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
            }
        }
        Ok(loss)
    }

    pub fn current_learning_rate(&self) -> f32 {
        let total = self.config.total_steps;
        if total == 0 {
            return self.config.learning_rate;
        }
        let t = (self.step.saturating_sub(1) as f32 / total as f32).clamp(0.0, 1.0);
        let floor = self.config.final_lr_fraction.clamp(0.0, 1.0);
        let cosine = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
        self.config.learning_rate * (floor + (1.0 - floor) * cosine)
    }

    fn gradient_scale(&self, grads: &[Vec<f32>]) -> f32 {
        if self.config.grad_clip <= 0.0 {
            return 1.0;
        }
        let sum_sq: f64 = grads
            .iter()
            .flat_map(|g| g.iter())
            .map(|g| (*g as f64) * (*g as f64))
            .sum();
        let norm = sum_sq.sqrt() as f32;
        if norm > self.config.grad_clip && norm.is_finite() {
            self.config.grad_clip / norm
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Widths, IN_CHANNELS};

    fn synthetic(n: usize, h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
        let pixels = h * w;
        let mut input = vec![0.0f32; n * IN_CHANNELS * pixels];
        let mut target = vec![0.0f32; n * OUT_CHANNELS * pixels];
        for b in 0..n {
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let clean = 0.25 + 0.5 * (x as f32 / w as f32) + 0.2 * (b as f32);
                    let noise = if (x + y) % 2 == 0 { 0.18 } else { -0.18 };
                    let noisy = (clean + noise).max(0.0);
                    for c in 0..3 {
                        input[(b * IN_CHANNELS + c) * pixels + i] = crate::compress(noisy);
                        input[(b * IN_CHANNELS + 3 + c) * pixels + i] = clean;
                        input[(b * IN_CHANNELS + 6 + c) * pixels + i] = 0.5;
                        input[(b * IN_CHANNELS + 10 + c) * pixels + i] =
                            crate::compress(noisy / clean.max(crate::ALBEDO_FLOOR));
                        input[(b * IN_CHANNELS + 13 + c) * pixels + i] = crate::compress(noisy);
                        target[(b * OUT_CHANNELS + c) * pixels + i] = crate::compress(clean);
                    }
                    input[(b * IN_CHANNELS + 9) * pixels + i] = 0.4;
                }
            }
        }
        (input, target)
    }

    /// Target is a 50/50 blend with the east neighbour — one hop tap.
    fn synthetic_east(n: usize, h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
        let pixels = h * w;
        let mut input = vec![0.0f32; n * IN_CHANNELS * pixels];
        let mut target = vec![0.0f32; n * OUT_CHANNELS * pixels];
        for b in 0..n {
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let probe = [
                        (x as f32 / w as f32) * 0.8 + 0.1,
                        (y as f32 / h as f32) * 0.8 + 0.1,
                        if (x + y) % 2 == 0 { 0.72 } else { 0.18 },
                    ];
                    let xe = (x + 1).min(w - 1);
                    let ie = y * w + xe;
                    let east = [
                        (xe as f32 / w as f32) * 0.8 + 0.1,
                        (y as f32 / h as f32) * 0.8 + 0.1,
                        if (xe + y) % 2 == 0 { 0.72 } else { 0.18 },
                    ];
                    for c in 0..OUT_CHANNELS {
                        input[(b * IN_CHANNELS + c) * pixels + i] = probe[c];
                        target[(b * OUT_CHANNELS + c) * pixels + i] = 0.5 * probe[c] + 0.5 * east[c];
                        input[(b * IN_CHANNELS + 3 + c) * pixels + i] = 0.5;
                        input[(b * IN_CHANNELS + 6 + c) * pixels + i] = 0.5;
                        input[(b * IN_CHANNELS + 10 + c) * pixels + i] = probe[c];
                        input[(b * IN_CHANNELS + 13 + c) * pixels + i] = probe[c];
                        let _ = ie;
                    }
                    input[(b * IN_CHANNELS + 9) * pixels + i] = 0.4;
                }
            }
        }
        (input, target)
    }

    #[test]
    fn an_untrained_network_is_the_identity() {
        let net = ProbeArch::new(Widths::tiny());
        let (n, h, w) = (1, 16, 16);
        let mut trainer =
            Trainer::new(net, n, h, w, Device::Cpu, TrainConfig::default()).expect("compile");
        let (input, target) = synthetic(n, h, w);
        let loss = trainer
            .step(&Batch {
                n,
                h,
                w,
                input: &input,
                target: &target,
            })
            .expect("step");

        let mut expected = 0.0f64;
        let pixels = h * w;
        for c in 0..OUT_CHANNELS {
            for i in 0..pixels {
                let y = input[c * pixels + i] as f64;
                let t = target[c * pixels + i] as f64;
                expected += (y - t) * (y - t) / (t * t + TRAIN_EPSILON as f64);
            }
        }
        expected /= (OUT_CHANNELS * pixels) as f64;
        // Untrained Direct: predicted = probe, so relative = rel_gap and the
        // GI boost turns the mean into E[r] + boost·E[r²].
        let mut r2 = 0.0f64;
        for c in 0..OUT_CHANNELS {
            for i in 0..pixels {
                let y = input[c * pixels + i] as f64;
                let t = target[c * pixels + i] as f64;
                let r = (y - t) * (y - t) / (t * t + TRAIN_EPSILON as f64);
                r2 += r * r;
            }
        }
        r2 /= (OUT_CHANNELS * pixels) as f64;
        expected += GI_LOSS_BOOST as f64 * r2;
        assert!(
            (loss as f64 - expected).abs() < 1e-4 * expected.max(1e-6),
            "loss {loss} but identity should give {expected}"
        );
    }

    // The point of this one is that the constants are constant: it pins the
    // relationship between them so a later edit to either has to come here.
    #[allow(clippy::assertions_on_constants)]
    #[test]
    fn the_reporting_floor_is_not_the_training_floor() {
        assert_eq!(LOSS_EPSILON, 0.01);
        assert!(TRAIN_EPSILON < LOSS_EPSILON);
        assert!(TRAIN_EPSILON >= 1e-4);
    }

    fn train_until_half(net: ProbeArch, input: Vec<f32>, target: Vec<f32>, lr: f32, steps: usize) {
        let (n, h, w) = (2, 16, 16);
        let mut trainer = Trainer::new(
            net,
            n,
            h,
            w,
            Device::Cpu,
            TrainConfig {
                learning_rate: lr,
                ..Default::default()
            },
        )
        .expect("compile");
        let batch = Batch {
            n,
            h,
            w,
            input: &input,
            target: &target,
        };
        let first = trainer.step(&batch).expect("step");
        let mut last = first;
        for _ in 0..steps {
            last = trainer.step(&batch).expect("step");
            assert!(last.is_finite(), "diverged to {last}");
        }
        assert!(
            last < first * 0.5,
            "loss did not fall: {first} → {last}"
        );
    }

    #[test]
    fn training_reduces_the_loss() {
        let (n, h, w) = (2, 16, 16);
        let (input, target) = synthetic(n, h, w);
        train_until_half(ProbeArch::new(Widths::tiny()), input, target, 3e-3, 40);
    }

    #[test]
    fn a_hop_head_also_trains() {
        let (n, h, w) = (2, 16, 16);
        let (input, target) = synthetic_east(n, h, w);
        train_until_half(ProbeArch::hops(), input, target, 3e-3, 80);
    }
}
