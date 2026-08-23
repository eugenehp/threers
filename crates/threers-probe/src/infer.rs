//! Running a trained network: forward graph only.

use anyhow::{ensure, Result};
use rlx::{DType, Device, Graph, Session, Shape};

use crate::model::{ProbeArch, OUT_CHANNELS};
use crate::NrcNet;

/// Overlap between adjacent inference tiles. Must be less than half the compiled
/// tile side. A 7×7 kernel only needs a few pixels of halo; sixteen also covers
/// most of the U-Net receptive field on a 32² tile.
pub const RECON_MARGIN: usize = 16;

/// A compiled forward pass and the weights behind it.
pub struct ProbeNet {
    net: ProbeArch,
    graph: rlx::CompiledGraph,
    params: Vec<Vec<f32>>,
    shape: (usize, usize, usize),
}

impl ProbeNet {
    pub fn new(
        net: ProbeArch,
        params: Vec<Vec<f32>>,
        n: usize,
        h: usize,
        w: usize,
        device: Device,
    ) -> Result<Self> {
        ensure!(n > 0 && h > 0 && w > 0, "empty shape {n}x{h}x{w}");
        ensure!(
            h.is_multiple_of(ProbeArch::TILE_MULTIPLE) && w.is_multiple_of(ProbeArch::TILE_MULTIPLE),
            "{h}x{w} must be a multiple of {}",
            ProbeArch::TILE_MULTIPLE
        );
        ensure!(
            params.len() == net.params().len()
                && params
                    .iter()
                    .zip(net.params())
                    .all(|(v, s)| v.len() == s.elems()),
            "parameter shapes do not match this network"
        );

        let mut graph = Graph::new("probe_forward");
        let input = graph.input(
            "input",
            Shape::new(&[n, crate::IN_CHANNELS, h, w], DType::F32),
        );
        let out = net.forward(&mut graph, input, n, h, w);
        graph.set_outputs(vec![out]);
        let graph = Session::new(device).compile(graph);

        Ok(Self {
            net,
            graph,
            params,
            shape: (n, h, w),
        })
    }

    pub fn net(&self) -> &ProbeArch {
        &self.net
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

    /// Reconstruct one batch. `input` is `[n, IN, h, w]`; the result is `[n, 3, h, w]`.
    pub fn run(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let (n, h, w) = self.shape;
        let pixels = n * h * w;
        ensure!(
            input.len() == pixels * crate::IN_CHANNELS,
            "input is {} floats, this graph was compiled for {}",
            input.len(),
            pixels * crate::IN_CHANNELS
        );
        for (spec, values) in self.net.params().iter().zip(&self.params) {
            self.graph.set_param(spec.name, values);
        }
        let out = self.graph.run(&[("input", input)]);
        let result = out.into_iter().next().unwrap_or_default();
        ensure!(
            result.len() == pixels * OUT_CHANNELS,
            "network returned {} floats, expected {}",
            result.len(),
            pixels * OUT_CHANNELS
        );
        Ok(result)
    }

    /// Sum of `(y - t)² / (t² + ε)` and the count behind it.
    pub fn relative_error_sum(predicted: &[f32], target: &[f32], epsilon: f32) -> (f64, usize) {
        if predicted.len() != target.len() {
            return (f64::NAN, 0);
        }
        let mut sum = 0.0f64;
        for (&y, &t) in predicted.iter().zip(target) {
            let (y, t) = (y as f64, t as f64);
            let d = y - t;
            sum += d * d / (t * t + epsilon as f64);
        }
        (sum, target.len())
    }

    pub fn relative_error(predicted: &[f32], target: &[f32], epsilon: f32) -> f32 {
        let (sum, n) = Self::relative_error_sum(predicted, target, epsilon);
        if n == 0 {
            return f32::NAN;
        }
        (sum / n as f64).sqrt() as f32
    }

    /// Reconstruct a planar frame `[IN, H, W]` into `[3, H, W]`.
    ///
    /// When `H` and `W` match the compiled tile size, one forward pass is
    /// enough. Larger frames are covered in overlapping tiles with a feathered
    /// blend so kernels can gather across tile borders.
    pub fn reconstruct(&mut self, input: &[f32], height: usize, width: usize) -> Result<Vec<f32>> {
        let (n, th, tw) = self.shape;
        ensure!(n == 1, "reconstruct wants a batch of 1, this graph is {n}");
        ensure!(
            th == tw,
            "reconstruct needs a square tile, this graph is {th}x{tw}"
        );
        let pixels = height * width;
        ensure!(
            input.len() == pixels * crate::IN_CHANNELS,
            "input is {} floats, expected {}",
            input.len(),
            pixels * crate::IN_CHANNELS
        );
        if height == th && width == tw {
            let mut out = self.run(input)?;
            post_process(input, &mut out, pixels, self.net.head().is_hops());
            return Ok(out);
        }
        let margin = RECON_MARGIN.min(th / 2 - 1).max(1);
        let step = th.saturating_sub(2 * margin).max(1);
        let mut sum = vec![0.0f32; crate::OUT_CHANNELS * pixels];
        let mut weight = vec![0.0f32; pixels];
        let mut y0 = 0usize;
        loop {
            let mut x0 = 0usize;
            loop {
                self.accumulate_tile(input, height, width, th, (x0, y0), margin, &mut sum, &mut weight)?;
                if x0 + th >= width {
                    break;
                }
                x0 = (x0 + step).min(width.saturating_sub(th));
            }
            if y0 + th >= height {
                break;
            }
            y0 = (y0 + step).min(height.saturating_sub(th));
        }
        let mut out = vec![0.0f32; crate::OUT_CHANNELS * pixels];
        for i in 0..pixels {
            let w = weight[i];
            if w <= 0.0 {
                continue;
            }
            for c in 0..crate::OUT_CHANNELS {
                out[c * pixels + i] = sum[c * pixels + i] / w;
            }
        }
        post_process(input, &mut out, pixels, self.net.head().is_hops());
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn accumulate_tile(
        &mut self,
        input: &[f32],
        height: usize,
        width: usize,
        tile: usize,
        origin: (usize, usize),
        margin: usize,
        sum: &mut [f32],
        weight: &mut [f32],
    ) -> Result<()> {
        let (x0, y0) = origin;
        let patch = crate::pack::extract_region(
            input,
            crate::IN_CHANNELS,
            height,
            width,
            tile,
            x0,
            y0,
        );
        let pred = self.run(&patch)?;
        let plane = tile * tile;
        for ty in 0..tile {
            let sy = y0 + ty;
            if sy >= height {
                continue;
            }
            for tx in 0..tile {
                let sx = x0 + tx;
                if sx >= width {
                    continue;
                }
                let f = feather(tx, tile, margin, x0 == 0, x0 + tile >= width)
                    * feather(ty, tile, margin, y0 == 0, y0 + tile >= height);
                if f <= 0.0 {
                    continue;
                }
                let i = sy * width + sx;
                weight[i] += f;
                for c in 0..crate::OUT_CHANNELS {
                    sum[c * height * width + i] += f * pred[c * plane + ty * tile + tx];
                }
            }
        }
        Ok(())
    }

    /// Load weights and compile a forward pass.
    pub fn load(
        path: impl AsRef<std::path::Path>,
        n: usize,
        h: usize,
        w: usize,
        device: Device,
    ) -> Result<Self> {
        let (net, params) = crate::checkpoint::load(path)?;
        Self::new(net, params, n, h, w, device)
    }
}

/// Hybrid U-Net + optional NRC — one entry point for fused reconstruction.
pub struct ProbeGi {
    unet: ProbeNet,
    nrc: Option<NrcNet>,
    height: usize,
    width: usize,
    nrc_min_gap: f32,
    nrc_cap: f32,
}

impl ProbeGi {
    pub fn new(unet: ProbeNet, nrc: Option<NrcNet>, height: usize, width: usize) -> Self {
        Self::with_nrc_gap(unet, nrc, height, width, crate::nrc::NRC_PROBE_CONFIDENCE)
    }

    pub fn with_nrc_gap(
        unet: ProbeNet,
        nrc: Option<NrcNet>,
        height: usize,
        width: usize,
        nrc_min_gap: f32,
    ) -> Self {
        Self::with_nrc_fuse(unet, nrc, height, width, nrc_min_gap, crate::nrc::NRC_CAP)
    }

    pub fn with_nrc_fuse(
        unet: ProbeNet,
        nrc: Option<NrcNet>,
        height: usize,
        width: usize,
        nrc_min_gap: f32,
        nrc_cap: f32,
    ) -> Self {
        Self {
            unet,
            nrc,
            height,
            width,
            nrc_min_gap,
            nrc_cap,
        }
    }

    pub fn load(
        probe_path: impl AsRef<std::path::Path>,
        nrc_path: Option<impl AsRef<std::path::Path>>,
        height: usize,
        width: usize,
        device: Device,
    ) -> Result<Self> {
        let unet = ProbeNet::load(probe_path, 1, height, width, device)?;
        let nrc = match nrc_path {
            Some(path) => Some(NrcNet::load(path, 1, height, width, device)?),
            None => None,
        };
        Ok(Self::new(unet, nrc, height, width))
    }

    pub fn load_with_nrc_gap(
        probe_path: impl AsRef<std::path::Path>,
        nrc_path: Option<impl AsRef<std::path::Path>>,
        height: usize,
        width: usize,
        device: Device,
        nrc_min_gap: f32,
    ) -> Result<Self> {
        let unet = ProbeNet::load(probe_path, 1, height, width, device)?;
        let nrc = match nrc_path {
            Some(path) => Some(NrcNet::load(path, 1, height, width, device)?),
            None => None,
        };
        Ok(Self::with_nrc_fuse(unet, nrc, height, width, nrc_min_gap, crate::nrc::NRC_CAP))
    }

    pub fn load_with_nrc_fuse(
        probe_path: impl AsRef<std::path::Path>,
        nrc_path: Option<impl AsRef<std::path::Path>>,
        height: usize,
        width: usize,
        device: Device,
        nrc_min_gap: f32,
        nrc_cap: f32,
    ) -> Result<Self> {
        let unet = ProbeNet::load(probe_path, 1, height, width, device)?;
        let nrc = match nrc_path {
            Some(path) => Some(NrcNet::load(path, 1, height, width, device)?),
            None => None,
        };
        Ok(Self::with_nrc_fuse(unet, nrc, height, width, nrc_min_gap, nrc_cap))
    }

    /// Load from in-memory weight blobs (wasm / embedded assets).
    pub fn load_from_bytes(
        probe_bytes: &[u8],
        nrc_bytes: Option<&[u8]>,
        height: usize,
        width: usize,
        device: Device,
        nrc_min_gap: f32,
        nrc_cap: f32,
    ) -> Result<Self> {
        let (net, params) = crate::checkpoint::load_bytes(probe_bytes)?;
        let unet = ProbeNet::new(net, params, 1, height, width, device)?;
        let nrc = match nrc_bytes {
            Some(bytes) => Some(NrcNet::load_bytes(bytes, 1, height, width, device)?),
            None => None,
        };
        Ok(Self::with_nrc_fuse(unet, nrc, height, width, nrc_min_gap, nrc_cap))
    }

    pub fn unet(&self) -> &ProbeNet {
        &self.unet
    }

    pub fn unet_mut(&mut self) -> &mut ProbeNet {
        &mut self.unet
    }

    /// U-Net reconstruction, optionally fused with NRC on gated pixels.
    pub fn reconstruct(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let beauty = self.unet.reconstruct(input, self.height, self.width)?;
        let Some(nrc) = &mut self.nrc else {
            return Ok(beauty);
        };
        let pixels = self.height * self.width;
        let feat = crate::nrc::nrc_features(input, pixels);
        let nrc_out = nrc.run(&feat)?;
        let mut fused = beauty;
        crate::nrc::fuse_beauty_with_cap(&mut fused, &nrc_out, input, pixels, self.nrc_min_gap, self.nrc_cap);
        Ok(fused)
    }
}

/// Fraction of sky pixels above which we start blending back to the probe.
pub const OPEN_SKY_START: f32 = 0.05;
/// At this sky fraction the output is the probe (no invented bounce outdoors).
pub const OPEN_SKY_FULL: f32 = 0.22;
/// Hit pixels whose shading normal is nearly +Y — a ground plane filling the view.
const GROUND_NORMAL_Y: f32 = 0.82;
const GROUND_FRACTION: f32 = 0.58;

/// Exterior frames are mostly sky and/or a single ground plane. GI residuals
/// trained on rooms hallucinate bounce onto the floor; blend toward the probe.
/// Gentle DDGI pull on neutral albedo — helps canonical Cornell floor bleed.
#[allow(dead_code)] // parked ablation variant — see the module's env-var switches
const CORNELL_DDGI_BLEND: f32 = 0.14;

fn wall_bleed_amount() -> Option<f32> {
    let raw = std::env::var("WALL_BLEED").ok()?;
    if raw == "1" || raw.eq_ignore_ascii_case("true") {
        return Some(0.28);
    }
    let v: f32 = raw.parse().ok()?;
    (v > 0.0).then_some(v.clamp(0.0, 1.0))
}

fn post_process(input: &[f32], pred: &mut [f32], pixels: usize, hops: bool) {
    protect_open_exteriors(input, pred, pixels);
    if hops {
        if probe_is_emissive_shell(input, pixels) {
            copy_probe_colour(input, pred, pixels);
            return;
        }
        protect_uniform_bright_probe(input, pred, pixels);
        let lift = floor_lift_rgb();
        if lift != [0.0; 3] {
            lift_neutral_floors(input, pred, pixels, lift);
        }
    }
    if let Some(amount) = wall_bleed_amount() {
        blend_wall_albedo_bleed(input, pred, pixels, amount);
    }
    if let Some(amount) = wall_hue_amount() {
        blend_wall_probe_hue(input, pred, pixels, amount);
    }
}

fn floor_lift_rgb() -> [f32; 3] {
    if let Ok(raw) = std::env::var("FLOOR_LIFT") {
        if raw.trim() == "0" {
            return [0.0; 3];
        }
        let parts: Vec<f32> = raw
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        if parts.len() == 3 {
            return [parts[0], parts[1], parts[2]];
        }
        if parts.len() == 1 {
            return [parts[0]; 3];
        }
    }
    // Per-channel bias on Cornell floor (compressed space).
    [0.012, 0.011, 0.010]
}

fn lift_neutral_floors(input: &[f32], pred: &mut [f32], pixels: usize, lift: [f32; 3]) {
    let hits = crate::pack::geometry_hits(input, pixels);
    for i in 0..pixels {
        if !hits[i] {
            continue;
        }
        let ny = input[7 * pixels + i];
        if ny < 0.72 {
            continue;
        }
        let ar = input[3 * pixels + i];
        let ag = input[4 * pixels + i];
        let ab = input[5 * pixels + i];
        if albedo_spread(ar, ag, ab) > 0.10 || (ar + ag + ab) / 3.0 < 0.40 {
            continue;
        }
        for c in 0..OUT_CHANNELS {
            pred[c * pixels + i] = (pred[c * pixels + i] + lift[c]).clamp(0.0, 0.999);
        }
    }
}

fn wall_hue_amount() -> Option<f32> {
    let raw = std::env::var("WALL_HUE").ok()?;
    if raw == "1" || raw.eq_ignore_ascii_case("true") {
        return Some(0.12);
    }
    let v: f32 = raw.parse().ok()?;
    (v > 0.0).then_some(v.clamp(0.0, 1.0))
}

/// Add left/right wall probe chroma onto floor/ceiling, keeping luma.
fn blend_wall_probe_hue(input: &[f32], pred: &mut [f32], pixels: usize, amount: f32) {
    let hits = crate::pack::geometry_hits(input, pixels);
    let mut left = [0.0f32; 3];
    let mut right = [0.0f32; 3];
    let mut left_n = 0.0f32;
    let mut right_n = 0.0f32;
    for i in 0..pixels {
        if !hits[i] {
            continue;
        }
        let nx = input[6 * pixels + i];
        let ny = input[7 * pixels + i];
        if ny.abs() > 0.55 {
            continue;
        }
        if nx < -0.65 {
            for c in 0..3 {
                left[c] += crate::expand(input[c * pixels + i]);
            }
            left_n += 1.0;
        } else if nx > 0.65 {
            for c in 0..3 {
                right[c] += crate::expand(input[c * pixels + i]);
            }
            right_n += 1.0;
        }
    }
    if left_n < 24.0 || right_n < 24.0 {
        return;
    }
    for c in 0..3 {
        left[c] /= left_n;
        right[c] /= right_n;
    }
    if rgb_spread(left).max(rgb_spread(right)) < 0.04 {
        return;
    }
    for i in 0..pixels {
        if !hits[i] {
            continue;
        }
        let ny = input[7 * pixels + i];
        if ny.abs() < 0.72 {
            continue;
        }
        let ar = input[3 * pixels + i];
        let ag = input[4 * pixels + i];
        let ab = input[5 * pixels + i];
        if albedo_spread(ar, ag, ab) > 0.10 || (ar + ag + ab) / 3.0 < 0.40 {
            continue;
        }
        let t = input[19 * pixels + i].clamp(0.0, 1.0);
        let mut wall = [0.0f32; 3];
        for c in 0..3 {
            wall[c] = left[c] * (1.0 - t) + right[c] * t;
        }
        let wall_mean = (wall[0] + wall[1] + wall[2]) / 3.0;
        if rgb_spread(wall) < 0.03 {
            continue;
        }
        for c in 0..OUT_CHANNELS {
            let pred_lin = crate::expand(pred[c * pixels + i]);
            let chroma = wall[c] - wall_mean;
            pred[c * pixels + i] = crate::compress((pred_lin + amount * chroma).max(0.0));
        }
    }
}

fn rgb_spread(rgb: [f32; 3]) -> f32 {
    let mean = (rgb[0] + rgb[1] + rgb[2]) / 3.0;
    (rgb[0] - mean)
        .abs()
        .max((rgb[1] - mean).abs())
        .max((rgb[2] - mean).abs())
}

/// Lateral colour bleed: floor/ceiling pick up left/right wall albedo, scaled
/// by local luma. Gated by `WALL_BLEED` so the default hops path stays unchanged.
fn blend_wall_albedo_bleed(input: &[f32], pred: &mut [f32], pixels: usize, amount: f32) {
    let hits = crate::pack::geometry_hits(input, pixels);
    let mut left = [0.0f32; 3];
    let mut right = [0.0f32; 3];
    let mut left_n = 0.0f32;
    let mut right_n = 0.0f32;
    for i in 0..pixels {
        if !hits[i] {
            continue;
        }
        let nx = input[6 * pixels + i];
        let ny = input[7 * pixels + i];
        if ny.abs() > 0.55 {
            continue;
        }
        if nx < -0.65 {
            for c in 0..3 {
                left[c] += input[(3 + c) * pixels + i];
            }
            left_n += 1.0;
        } else if nx > 0.65 {
            for c in 0..3 {
                right[c] += input[(3 + c) * pixels + i];
            }
            right_n += 1.0;
        }
    }
    if left_n < 24.0 || right_n < 24.0 {
        return;
    }
    for c in 0..3 {
        left[c] /= left_n;
        right[c] /= right_n;
    }
    if albedo_spread(left[0], left[1], left[2]).max(albedo_spread(right[0], right[1], right[2]))
        < 0.12
    {
        return;
    }
    for i in 0..pixels {
        if !hits[i] {
            continue;
        }
        let ny = input[7 * pixels + i];
        if ny.abs() < 0.72 {
            continue;
        }
        let ar = input[3 * pixels + i];
        let ag = input[4 * pixels + i];
        let ab = input[5 * pixels + i];
        let mean_alb = (ar + ag + ab) / 3.0;
        if albedo_spread(ar, ag, ab) > 0.10 || mean_alb < 0.40 {
            continue;
        }
        let t = input[19 * pixels + i].clamp(0.0, 1.0);
        let mut wall = [0.0f32; 3];
        for c in 0..3 {
            wall[c] = left[c] * (1.0 - t) + right[c] * t;
        }
        let wall_mean = (wall[0] + wall[1] + wall[2]) / 3.0;
        if wall_mean < 0.05 || albedo_spread(wall[0], wall[1], wall[2]) < 0.08 {
            continue;
        }
        let mut pred_lin = [0.0f32; 3];
        for c in 0..OUT_CHANNELS {
            pred_lin[c] = crate::expand(pred[c * pixels + i]);
        }
        let luma = (pred_lin[0] + pred_lin[1] + pred_lin[2]) / 3.0;
        if !(0.02..=0.85).contains(&luma) {
            continue;
        }
        for c in 0..OUT_CHANNELS {
            let tinted = wall[c] / wall_mean * luma;
            let out = pred_lin[c] * (1.0 - amount) + tinted * amount;
            pred[c * pixels + i] = crate::compress(out.max(0.0));
        }
    }
}

fn albedo_spread(ar: f32, ag: f32, ab: f32) -> f32 {
    let mean = (ar + ag + ab) / 3.0;
    (ar - mean)
        .abs()
        .max((ag - mean).abs())
        .max((ab - mean).abs())
}

#[allow(dead_code)] // parked ablation variant — see the module's env-var switches
fn lin_chroma(r: f32, g: f32, b: f32) -> f32 {
    let mean = (r + g + b) / 3.0;
    (r - mean)
        .abs()
        .max((g - mean).abs())
        .max((b - mean).abs())
}

/// Linear DDGI pull on neutral surfaces when DDGI carries more wall chroma.
#[allow(dead_code)] // parked ablation variant — see the module's env-var switches
fn blend_room_context_on_neutral(input: &[f32], pred: &mut [f32], pixels: usize) {
    let hits = crate::pack::geometry_hits(input, pixels);
    for i in 0..pixels {
        if !hits[i] {
            continue;
        }
        let ar = input[3 * pixels + i];
        let ag = input[4 * pixels + i];
        let ab = input[5 * pixels + i];
        let mean = (ar + ag + ab) / 3.0;
        if albedo_spread(ar, ag, ab) > 0.10 || mean < 0.45 {
            continue;
        }
        let probe_avg =
            (input[i] + input[pixels + i] + input[2 * pixels + i]) / 3.0;
        if !(0.12..=0.68).contains(&probe_avg) {
            continue;
        }
        let mut pred_lin = [0.0f32; 3];
        let mut tint_lin = [0.0f32; 3];
        for c in 0..OUT_CHANNELS {
            pred_lin[c] = crate::expand(pred[c * pixels + i]);
            tint_lin[c] = crate::expand(input[(16 + c) * pixels + i]);
        }
        if lin_chroma(tint_lin[0], tint_lin[1], tint_lin[2])
            <= lin_chroma(pred_lin[0], pred_lin[1], pred_lin[2]) + 0.004
        {
            continue;
        }
        let t = CORNELL_TINT;
        let keep = 1.0 - t;
        for c in 0..OUT_CHANNELS {
            let out_lin = (pred_lin[c] * keep + tint_lin[c] * t).max(0.0);
            pred[c * pixels + i] = crate::compress(out_lin);
        }
    }
}

/// Used by unit tests; production path is [`crate::model::GRAPH_ROOM_TINT`] in-graph.
#[allow(dead_code)] // parked ablation variant — see the module's env-var switches
const CORNELL_TINT: f32 = 0.22;

/// Uniform bright probe with almost no spatial contrast — emissive furnace shell.
pub(crate) fn probe_is_emissive_shell(input: &[f32], pixels: usize) -> bool {
    let hits = crate::pack::geometry_hits(input, pixels);
    let mut hit_n = 0usize;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for i in 0..pixels {
        if !hits[i] {
            continue;
        }
        hit_n += 1;
        let avg = (input[i] + input[pixels + i] + input[2 * pixels + i]) as f64 / 3.0;
        sum += avg;
        sum_sq += avg * avg;
    }
    if hit_n < pixels / 8 {
        return false;
    }
    let mean = sum / hit_n as f64;
    let var = (sum_sq / hit_n as f64 - mean * mean).max(0.0);
    mean > 0.40 && var < 0.014
}

fn copy_probe_colour(input: &[f32], pred: &mut [f32], pixels: usize) {
    for c in 0..OUT_CHANNELS {
        pred[c * pixels..(c + 1) * pixels].copy_from_slice(&input[c * pixels..(c + 1) * pixels]);
    }
}

fn protect_open_exteriors(input: &[f32], pred: &mut [f32], pixels: usize) {
    let hits = crate::pack::geometry_hits(input, pixels);
    let mut sky = 0usize;
    let mut ground = 0usize;
    let mut hit_n = 0usize;
    for i in 0..pixels {
        if !hits[i] {
            sky += 1;
            continue;
        }
        hit_n += 1;
        if input[7 * pixels + i] > GROUND_NORMAL_Y {
            ground += 1;
        }
    }
    let sky_f = sky as f32 / pixels.max(1) as f32;
    let ground_f = if hit_n == 0 {
        0.0
    } else {
        ground as f32 / hit_n as f32
    };
    let span = (OPEN_SKY_FULL - OPEN_SKY_START).max(1e-3);
    let mut t = if sky_f < OPEN_SKY_START {
        0.0
    } else {
        ((sky_f - OPEN_SKY_START) / span).clamp(0.0, 1.0)
    };
    if ground_f >= GROUND_FRACTION {
        t = t.max(0.9);
    }
    if t <= 0.0 {
        return;
    }
    let keep = 1.0 - t;
    for c in 0..OUT_CHANNELS {
        for i in 0..pixels {
            let p = input[c * pixels + i];
            pred[c * pixels + i] = pred[c * pixels + i] * keep + p * t;
        }
    }
}

/// Uniformly bright hit pixels (emissive shell / furnace) — blend toward probe.
const BRIGHT_PROBE: f32 = 0.48;
const BRIGHT_FRACTION: f32 = 0.78;
const BRIGHT_BLEND: f32 = 0.75;

fn protect_uniform_bright_probe(input: &[f32], pred: &mut [f32], pixels: usize) {
    let hits = crate::pack::geometry_hits(input, pixels);
    let mut bright = 0usize;
    let mut hit_n = 0usize;
    for i in 0..pixels {
        if !hits[i] {
            continue;
        }
        hit_n += 1;
        let r = input[i];
        let g = input[pixels + i];
        let b = input[2 * pixels + i];
        if r > BRIGHT_PROBE && g > BRIGHT_PROBE && b > BRIGHT_PROBE {
            bright += 1;
        }
    }
    if hit_n == 0 {
        return;
    }
    let frac = bright as f32 / hit_n as f32;
    if frac < BRIGHT_FRACTION {
        return;
    }
    let t = BRIGHT_BLEND * ((frac - BRIGHT_FRACTION) / (1.0 - BRIGHT_FRACTION)).clamp(0.0, 1.0);
    if t <= 0.0 {
        return;
    }
    let keep = 1.0 - t;
    for c in 0..OUT_CHANNELS {
        for i in 0..pixels {
            let p = input[c * pixels + i];
            pred[c * pixels + i] = pred[c * pixels + i] * keep + p * t;
        }
    }
}

fn feather(i: usize, tile: usize, margin: usize, at_start: bool, at_end: bool) -> f32 {
    let from_start = i;
    let from_end = tile - 1 - i;
    let lead = if at_start { margin } else { from_start };
    let trail = if at_end { margin } else { from_end };
    let edge = lead.min(trail);
    if edge >= margin {
        1.0
    } else {
        let t = (edge as f32 + 0.5) / margin as f32;
        t * t * (3.0 - 2.0 * t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{init_params, Widths, IN_CHANNELS};
    use crate::LOSS_EPSILON;

    #[test]
    fn an_untrained_forward_pass_returns_the_probe_colour() {
        let net = ProbeArch::new(Widths::tiny());
        let params = init_params(&net, 0x5eed_1234);
        let (n, h, w) = (1usize, 16usize, 16usize);
        let mut infer = ProbeNet::new(net, params, n, h, w, Device::Cpu).expect("compile");
        let pixels = h * w;
        let mut input = vec![0.0f32; n * IN_CHANNELS * pixels];
        for i in 0..pixels {
            input[i] = 0.31;
            input[pixels + i] = 0.17;
            input[2 * pixels + i] = 0.09;
            input[3 * pixels + i] = 0.4;
            input[4 * pixels + i] = 0.4;
            input[5 * pixels + i] = 0.4;
            input[9 * pixels + i] = 0.4;
        }
        let out = infer.run(&input).expect("run");
        for i in 0..pixels {
            assert!((out[i] - 0.31).abs() < 1e-5);
            assert!((out[pixels + i] - 0.17).abs() < 1e-5);
            assert!((out[2 * pixels + i] - 0.09).abs() < 1e-5);
        }
        let err = ProbeNet::relative_error(&out, &input[..3 * pixels], LOSS_EPSILON);
        assert!(err < 1e-5, "identity error {err}");
    }

    #[test]
    fn an_untrained_hop_head_is_the_identity() {
        let net = ProbeArch::hops();
        let params = init_params(&net, 0x5eed_1234);
        let (n, h, w) = (1usize, 16usize, 16usize);
        let mut infer = ProbeNet::new(net, params, n, h, w, Device::Cpu).expect("compile");
        let pixels = h * w;
        let mut input = vec![0.0f32; n * IN_CHANNELS * pixels];
        for i in 0..pixels {
            input[i] = 0.31;
            input[pixels + i] = 0.17;
            input[2 * pixels + i] = 0.09;
            input[16 * pixels + i] = 0.05;
            input[17 * pixels + i] = 0.04;
            input[18 * pixels + i] = 0.03;
        }
        let out = infer.run(&input).expect("run");
        for i in 0..pixels {
            assert!((out[i] - 0.31).abs() < 0.02, "hop identity {}", out[i]);
            assert!((out[pixels + i] - 0.17).abs() < 0.02);
            assert!((out[2 * pixels + i] - 0.09).abs() < 0.02);
        }
    }

    #[test]
    fn a_kernel_head_on_a_flat_probe_is_the_identity() {
        let net = ProbeArch::gathering(Widths::tiny());
        let params = init_params(&net, 0x5eed_1234);
        let (n, h, w) = (1usize, 16usize, 16usize);
        let mut infer = ProbeNet::new(net, params, n, h, w, Device::Cpu).expect("compile");
        let pixels = h * w;
        let mut input = vec![0.0f32; n * IN_CHANNELS * pixels];
        for i in 0..pixels {
            input[i] = 0.31;
            input[pixels + i] = 0.17;
            input[2 * pixels + i] = 0.09;
            input[3 * pixels + i] = 0.4;
            input[4 * pixels + i] = 0.4;
            input[5 * pixels + i] = 0.4;
            input[9 * pixels + i] = 0.4;
        }
        let out = infer.run(&input).expect("run");
        for i in 0..pixels {
            assert!((out[i] - 0.31).abs() < 1e-5);
            assert!((out[pixels + i] - 0.17).abs() < 1e-5);
            assert!((out[2 * pixels + i] - 0.09).abs() < 1e-5);
        }
    }

    #[test]
    fn an_untrained_kernel_head_is_nearly_the_identity() {
        let net = ProbeArch::gathering(Widths::tiny());
        let params = init_params(&net, 11);
        let (n, h, w) = (1usize, 16usize, 16usize);
        let mut infer = ProbeNet::new(net, params, n, h, w, Device::Cpu).expect("compile");
        let pixels = h * w;
        let mut input = vec![0.0f32; n * IN_CHANNELS * pixels];
        for (i, v) in input.iter_mut().enumerate() {
            *v = ((i % 29) as f32) / 29.0;
        }
        let out = infer.run(&input).expect("run");
        // exp(8)/(exp(8)+48) ≈ 0.984, so a few percent may come from neighbours.
        // Anything far outside that is a uniform box (lost centre bias) or the
        // softmax on the wrong axis.
        for c in 0..OUT_CHANNELS {
            for p in 0..pixels {
                let got = out[c * pixels + p];
                let want = input[c * pixels + p];
                assert!(
                    (got - want).abs() < 0.08,
                    "channel {c} pixel {p}: {got} but the probe was {want}"
                );
            }
        }
    }

    #[test]
    fn a_kernel_head_cannot_leave_its_neighbourhood() {
        const RADIUS: usize = 3;
        let net = ProbeArch::gathering(Widths::tiny());
        let mut params = init_params(&net, 3);
        // Break the delta so the filter actually mixes, otherwise this test
        // would pass for a Direct head too.
        if let Some(out) = params.last_mut() {
            for (i, v) in out.iter_mut().enumerate() {
                *v = ((i % 17) as f32 - 8.0) * 0.05;
            }
        }
        let (n, h, w) = (1usize, 16usize, 16usize);
        let mut infer = ProbeNet::new(net, params, n, h, w, Device::Cpu).expect("compile");
        let pixels = h * w;
        let mut input = vec![0.0f32; n * IN_CHANNELS * pixels];
        for (i, v) in input.iter_mut().enumerate() {
            *v = ((i % 23) as f32) / 23.0;
        }
        let out = infer.run(&input).expect("run");
        for c in 0..OUT_CHANNELS {
            for y in 0..h {
                for x in 0..w {
                    let mut lo = f32::INFINITY;
                    let mut hi = f32::NEG_INFINITY;
                    for dy in 0..=2 * RADIUS {
                        for dx in 0..=2 * RADIUS {
                            let yy = (y as i32 + dy as i32 - RADIUS as i32)
                                .clamp(0, h as i32 - 1) as usize;
                            let xx = (x as i32 + dx as i32 - RADIUS as i32)
                                .clamp(0, w as i32 - 1) as usize;
                            let v = input[c * pixels + yy * w + xx];
                            lo = lo.min(v);
                            hi = hi.max(v);
                        }
                    }
                    let got = out[c * pixels + y * w + x];
                    assert!(
                        got >= lo - 1e-4 && got <= hi + 1e-4,
                        "channel {c} at ({x},{y}): {got} escaped [{lo}, {hi}]"
                    );
                }
            }
        }
    }

    #[test]
    fn room_context_blend_tints_neutral_floor_toward_the_room_average() {
        let pixels = 2;
        let mut input = vec![0.0f32; crate::IN_CHANNELS * pixels];
        let mut pred = vec![0.5f32; OUT_CHANNELS * pixels];
        // Hit, neutral albedo, moderate probe.
        input[9 * pixels] = 0.4;
        input[3 * pixels] = 0.72;
        input[4 * pixels] = 0.70;
        input[5 * pixels] = 0.71;
        input[0] = 0.45;
        input[pixels] = 0.44;
        input[2 * pixels] = 0.43;
        input[16 * pixels] = 0.62;
        input[17 * pixels] = 0.40;
        input[18 * pixels] = 0.38;
        blend_room_context_on_neutral(&input, &mut pred, pixels);
        assert!(pred[0] > 0.5);
        assert!(pred[pixels] < 0.5);
        // Colored wall pixel — skip.
        input[3 * pixels + 1] = 0.65;
        input[4 * pixels + 1] = 0.05;
        input[5 * pixels + 1] = 0.05;
        input[6 * pixels + 1] = 0.0;
        input[7 * pixels + 1] = 0.2;
        input[8 * pixels + 1] = 0.95;
        pred[1] = 0.2;
        pred[pixels + 1] = 0.1;
        pred[2 * pixels + 1] = 0.1;
        blend_room_context_on_neutral(&input, &mut pred, pixels);
        assert!((pred[1] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn overlapping_reconstruct_matches_a_single_full_frame_pass() {
        let net = ProbeArch::new(Widths::tiny());
        let params = init_params(&net, 0x5eed_1234);
        let tile = 16usize;
        let (h, w) = (16usize, 32usize);
        let mut whole = ProbeNet::new(net.clone(), params.clone(), 1, h, w, Device::Cpu).expect("whole");
        let mut tiled = ProbeNet::new(net, params, 1, tile, tile, Device::Cpu).expect("tile");
        let pixels = h * w;
        let mut input = vec![0.0f32; IN_CHANNELS * pixels];
        for i in 0..pixels {
            input[i] = 0.4;
            input[pixels + i] = 0.2;
            input[2 * pixels + i] = 0.1;
            input[3 * pixels + i] = 0.5;
            input[4 * pixels + i] = 0.5;
            input[5 * pixels + i] = 0.5;
        }
        let whole_out = whole.run(&input).expect("whole");
        let tiled_out = tiled.reconstruct(&input, h, w).expect("tiled");
        for (a, b) in whole_out.iter().zip(&tiled_out) {
            assert!((a - b).abs() < 1e-4, "whole {a} vs tiled {b}");
        }
    }

    #[test]
    fn emissive_shell_detection_is_uniform_and_bright() {
        let pixels = 64usize;
        let mut uniform = vec![0.0f32; IN_CHANNELS * pixels];
        for i in 0..pixels {
            uniform[i] = 0.62;
            uniform[pixels + i] = 0.61;
            uniform[2 * pixels + i] = 0.63;
            uniform[3 * pixels + i] = 0.5;
            uniform[4 * pixels + i] = 0.5;
            uniform[5 * pixels + i] = 0.5;
            uniform[9 * pixels + i] = 0.4;
        }
        assert!(probe_is_emissive_shell(&uniform, pixels));

        let mut cornell = uniform.clone();
        for i in 0..pixels / 2 {
            cornell[i] = 0.08;
            cornell[pixels + i] = 0.07;
            cornell[2 * pixels + i] = 0.06;
        }
        assert!(!probe_is_emissive_shell(&cornell, pixels));
    }

    #[test]
    fn reconstruct_covers_a_two_tile_frame() {
        let net = ProbeArch::new(Widths::tiny());
        let params = init_params(&net, 0x5eed_1234);
        let tile = 16usize;
        let mut infer = ProbeNet::new(net, params, 1, tile, tile, Device::Cpu).expect("compile");
        let (h, w) = (16usize, 32usize);
        let pixels = h * w;
        let mut input = vec![0.0f32; IN_CHANNELS * pixels];
        for i in 0..pixels {
            input[i] = 0.4;
            input[pixels + i] = 0.2;
            input[2 * pixels + i] = 0.1;
            input[3 * pixels + i] = 0.5;
            input[4 * pixels + i] = 0.5;
            input[5 * pixels + i] = 0.5;
        }
        let out = infer.reconstruct(&input, h, w).expect("reconstruct");
        assert_eq!(out.len(), 3 * pixels);
        assert!((out[0] - 0.4).abs() < 1e-5);
        assert!((out[pixels + w] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn a_mostly_sky_frame_blends_back_to_the_probe() {
        let net = ProbeArch::new(Widths::tiny());
        let mut params = init_params(&net, 0x5eed_1234);
        // Non-zero residual so Direct would otherwise leave the probe.
        if let Some(out) = params.last_mut() {
            for v in out.iter_mut() {
                *v = 0.4;
            }
        }
        let (h, w) = (16usize, 16usize);
        let mut infer = ProbeNet::new(net, params, 1, h, w, Device::Cpu).expect("compile");
        let pixels = h * w;
        let mut input = vec![0.0f32; IN_CHANNELS * pixels];
        for i in 0..pixels {
            input[i] = 0.31;
            input[pixels + i] = 0.17;
            input[2 * pixels + i] = 0.09;
            input[3 * pixels + i] = 0.5;
            input[9 * pixels + i] = 0.999; // sky
        }
        let out = infer.reconstruct(&input, h, w).expect("reconstruct");
        for value in &out[..pixels] {
            assert!((value - 0.31).abs() < 1e-4, "sky frame must copy the probe");
        }
    }
}
