//! A guided reconstructor: probe + G-buffer + demodulated irradiance in,
//! reconstructed colour out.
//!
//! * [`Head::Direct`] / [`Head::Kernel`] / [`Head::Hybrid`] — tiny U-Net.
//! * [`Head::Hops`] — à-trous light jumps (cross taps + world-grid teleport), no encoder.

use rlx::ir::PadMode;
use rlx::{DType, Graph, GraphExt, NodeId, Shape};

/// Probe RGB, albedo, shading normal, depth, demodulated irradiance, and a
/// quarter-resolution probe for long-range context.
///
/// | planes | |
/// |---|---|
/// | 0-2 | probe colour, compressed |
/// | 3-5 | first-hit albedo |
/// | 6-8 | first-hit shading normal |
/// | 9 | first-hit depth / scene scale, compressed |
/// | 10-12 | probe ÷ albedo, compressed |
/// | 13-15 | probe colour box-averaged to ¼ res, then nearest-upsampled |
/// | 16-18 | DDGI-lite: probe binned into an 8×4×8 world grid |
/// | 19-21 | encoded world position (scene-centred, in `[0, 1)`) |
pub const IN_CHANNELS: usize = 22;
/// Reconstructed colour, compressed the same way as the probe.
pub const OUT_CHANNELS: usize = 3;
/// Floor on albedo when demodulating, so a black texel does not explode.
pub const ALBEDO_FLOOR: f32 = 0.05;
const K: usize = 3;

/// Default cap on the hybrid / direct irradiance residual (linear light).
/// Applied in shadow: bounce is multiplied by `(1 − compressed probe)`, so
/// bright furnace walls stay near identity even if the cap is this large.
pub const HYBRID_IRRADIANCE_CAP: f32 = 0.18;
/// Tighter cap for the hops residual — gather already moves probe energy.
pub const HOPS_IRRADIANCE_CAP: f32 = 0.17;

/// À-trous dilations 1 … 16 — five propagation hops.
pub const HOP_COUNT: usize = 5;
/// Centre + cardinals + diagonals + DDGI + quarter-res + demodulated irradiance.
pub const HOP_TAPS: usize = 12;
const HOP_FEAT: usize = 16;

/// How the last layer turns features into a colour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Head {
    /// Predict a linear irradiance residual, multiply by albedo, add the probe.
    Direct,
    /// Predict a normalised `(2r+1)²` filter per pixel and apply it to the
    /// probe colour — kernel-predicting network, after Bako et al. 2017.
    Kernel { radius: usize },
    /// KPN gather, then add albedo × clamp(ΔE, ±cap). Untrained ≈ probe.
    Hybrid { radius: usize, cap: f32 },
    /// Recursive screen-space hops in linear light, then a shadow-gated
    /// irradiance residual. Convex gather; untrained ≈ probe.
    Hops { hops: usize, cap: f32 },
}

impl Head {
    /// Channels the final convolution has to emit.
    pub fn out_channels(self) -> usize {
        match self {
            Head::Direct => OUT_CHANNELS,
            Head::Hops { hops, .. } => hops * HOP_TAPS,
            Head::Kernel { radius } | Head::Hybrid { radius, .. } => {
                let side = 2 * radius + 1;
                side * side + if matches!(self, Head::Hybrid { .. }) {
                    OUT_CHANNELS
                } else {
                    0
                }
            }
        }
    }

    pub fn code(self) -> u32 {
        match self {
            Head::Direct => 0,
            Head::Kernel { .. } => 1,
            Head::Hybrid { .. } => 2,
            Head::Hops { .. } => 3,
        }
    }

    pub fn from_code(code: u32, radius: u32, cap: f32) -> Option<Self> {
        match code {
            0 => Some(Head::Direct),
            1 if radius >= 1 => Some(Head::Kernel {
                radius: radius as usize,
            }),
            2 if radius >= 1 => Some(Head::Hybrid {
                radius: radius as usize,
                cap: if cap > 0.0 { cap } else { HYBRID_IRRADIANCE_CAP },
            }),
            3 if radius >= 1 => Some(Head::Hops {
                hops: radius as usize,
                cap: if cap > 0.0 {
                    cap
                } else {
                    HOPS_IRRADIANCE_CAP
                },
            }),
            _ => None,
        }
    }

    pub fn radius(self) -> u32 {
        match self {
            Head::Direct => 0,
            Head::Kernel { radius } | Head::Hybrid { radius, .. } => radius as u32,
            Head::Hops { hops, .. } => hops as u32,
        }
    }

    pub fn cap(self) -> f32 {
        match self {
            Head::Hybrid { cap, .. } | Head::Hops { cap, .. } => cap,
            _ => 0.0,
        }
    }

    pub fn is_hops(self) -> bool {
        matches!(self, Head::Hops { .. })
    }
}

/// Channel widths at the four resolutions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Widths {
    pub level0: usize,
    pub level1: usize,
    pub level2: usize,
    pub level3: usize,
}

impl Widths {
    /// The proof-of-concept size: ~24k parameters (direct) / ~27k (7×7 kernel).
    pub fn tiny() -> Self {
        Self {
            level0: 8,
            level1: 12,
            level2: 16,
            level3: 20,
        }
    }
}

impl Default for Widths {
    fn default() -> Self {
        Self::tiny()
    }
}

/// One convolution's parameter: name and `[out, in, kh, kw]`.
#[derive(Debug, Clone)]
pub struct ParamSpec {
    pub name: &'static str,
    pub shape: [usize; 4],
}

impl ParamSpec {
    pub fn elems(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn fan_in(&self) -> usize {
        self.shape[1] * self.shape[2] * self.shape[3]
    }
}

/// Architecture: widths, head, and the parameter list they imply.
#[derive(Debug, Clone)]
pub struct ProbeArch {
    widths: Widths,
    head: Head,
    params: Vec<ParamSpec>,
}

impl ProbeArch {
    /// Direct residual head. Untrained weights copy the probe exactly.
    pub fn new(widths: Widths) -> Self {
        Self::with_head(widths, Head::Direct)
    }

    /// 7×7 kernel-predicting head. Untrained weights are a near-delta.
    pub fn gathering(widths: Widths) -> Self {
        Self::with_head(widths, Head::Kernel { radius: 3 })
    }

    /// KPN + capped irradiance residual — the production head for this crate.
    pub fn hybrid(widths: Widths) -> Self {
        Self::with_head(
            widths,
            Head::Hybrid {
                radius: 3,
                cap: HYBRID_IRRADIANCE_CAP,
            },
        )
    }

    /// À-trous light hops. A few hundred parameters; no encoder/decoder.
    pub fn hops() -> Self {
        Self::with_head(
            Widths::tiny(),
            Head::Hops {
                hops: HOP_COUNT,
                cap: HOPS_IRRADIANCE_CAP,
            },
        )
    }

    pub fn with_head(widths: Widths, head: Head) -> Self {
        let params = match head {
            Head::Hops { hops, .. } => hop_params(hops),
            _ => unet_params(widths, head),
        };
        Self {
            widths,
            head,
            params,
        }
    }

    pub fn widths(&self) -> Widths {
        self.widths
    }

    pub fn head(&self) -> Head {
        self.head
    }

    pub fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    pub fn parameter_count(&self) -> usize {
        self.params.iter().map(ParamSpec::elems).sum()
    }

    /// Both spatial dimensions must be divisible by this: the encoder halves
    /// three times and the skips have to line up.
    pub const TILE_MULTIPLE: usize = 8;

    /// Add the forward pass, returning reconstructed colour `[n, 3, h, w]`.
    pub fn forward(
        &self,
        graph: &mut Graph,
        input: NodeId,
        n: usize,
        h: usize,
        w: usize,
    ) -> NodeId {
        assert!(
            h.is_multiple_of(Self::TILE_MULTIPLE) && w.is_multiple_of(Self::TILE_MULTIPLE),
            "probe: {h}x{w} is not a multiple of {}",
            Self::TILE_MULTIPLE
        );
        let p: Vec<NodeId> = self
            .params
            .iter()
            .map(|s| graph.param(s.name, Shape::new(&s.shape, DType::F32)))
            .collect();

        if let Head::Hops { hops, cap } = self.head {
            return hop_forward(graph, input, &p, hops, cap, h, w);
        }

        let same = [K / 2, K / 2];
        let stride = [1, 1];
        let Widths {
            level0: w0,
            level1: w1,
            level2: w2,
            level3: w3,
        } = self.widths;

        let enc0 = conv_relu(graph, input, p[0], stride, same);
        let enc1 = down(graph, enc0, p[1], w0);
        let enc2 = down(graph, enc1, p[2], w1);
        let enc3 = down(graph, enc2, p[3], w2);
        let bottleneck = conv_relu(graph, enc3, p[4], stride, same);

        let u3 = up(graph, bottleneck, p[5], w3);
        let d3 = graph.concat_(vec![u3, enc2], 1);
        let d3 = conv_relu(graph, d3, p[6], stride, same);
        let u2 = up(graph, d3, p[7], w2);
        let d2 = graph.concat_(vec![u2, enc1], 1);
        let d2 = conv_relu(graph, d2, p[8], stride, same);
        let u1 = up(graph, d2, p[9], w1);
        let d1 = graph.concat_(vec![u1, enc0], 1);
        let d1 = conv_relu(graph, d1, p[10], stride, same);

        let predicted = graph.conv2d(d1, p[11], [K, K], stride, same, stride, 1);
        match self.head {
            Head::Direct => beauty_from_irradiance(graph, input, predicted, None),
            Head::Kernel { radius } => {
                let colour = graph.narrow_(input, 1, 0, OUT_CHANNELS);
                apply_predicted_kernel(graph, colour, predicted, radius, n, h, w)
            }
            Head::Hybrid { radius, cap } => {
                let taps = (2 * radius + 1).pow(2);
                let colour = graph.narrow_(input, 1, 0, OUT_CHANNELS);
                let kernel_logits = graph.narrow_(predicted, 1, 0, taps);
                let gathered = apply_predicted_kernel(graph, colour, kernel_logits, radius, n, h, w);
                let residual = graph.narrow_(predicted, 1, taps, OUT_CHANNELS);
                let capped = soft_clamp_irradiance(graph, residual, cap);
                beauty_from_gathered(graph, input, gathered, capped)
            }
            Head::Hops { .. } => unreachable!("hops skip the U-Net"),
        }
    }
}

fn beauty_from_irradiance(
    graph: &mut Graph,
    input: NodeId,
    predicted: NodeId,
    cap: Option<f32>,
) -> NodeId {
    let colour = graph.narrow_(input, 1, 0, OUT_CHANNELS);
    let irradiance = match cap {
        Some(c) => soft_clamp_irradiance(graph, predicted, c),
        None => predicted,
    };
    beauty_from_gathered(graph, input, colour, irradiance)
}

fn beauty_from_hops_gathered(
    graph: &mut Graph,
    input: NodeId,
    gathered: NodeId,
    irradiance: NodeId,
) -> NodeId {
    let probe = graph.narrow_(input, 1, 0, OUT_CHANNELS);
    let albedo = graph.narrow_(input, 1, 3, OUT_CHANNELS);
    let one = graph.constant(1.0, DType::F32);
    let shadow = hops_bounce_shadow(graph, input, probe);
    let denom_g = graph.sub(one, gathered);
    let probe_lin = graph.div(gathered, denom_g);
    let bounce = graph.mul(albedo, irradiance);
    let bounce = graph.mul(bounce, shadow);
    let beauty_lin = graph.add(probe_lin, bounce);
    let beauty_lin = graph.relu(beauty_lin);
    let beauty_den = graph.add(one, beauty_lin);
    graph.div(beauty_lin, beauty_den)
}

fn floor_shadow_soften_amount() -> f32 {
    std::env::var("FLOOR_SHADOW")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0)
}

fn graph_up_floor_weight(graph: &mut Graph, input: NodeId) -> NodeId {
    let ny = graph.narrow_(input, 1, 7, 1);
    let thresh = graph.constant(0.72, DType::F32);
    let delta = graph.sub(ny, thresh);
    let up = graph.relu(delta);
    let scale = graph.constant(1.0 / 0.28, DType::F32);
    let up = graph.mul(up, scale);
    let surface = neutral_surface_weight(graph, input);
    graph.mul(up, surface)
}

/// Lit floors still need irradiance residual for wall bleed; soften `(1 − probe)`.
fn hops_bounce_shadow(graph: &mut Graph, input: NodeId, probe: NodeId) -> NodeId {
    let one = graph.constant(1.0, DType::F32);
    let shadow = graph.sub(one, probe);
    let soften = floor_shadow_soften_amount();
    if soften <= 0.0 {
        return shadow;
    }
    let floor_w = graph_up_floor_weight(graph, input);
    let soften_c = graph.constant(soften as f64, DType::F32);
    let t = graph.mul(floor_w, soften_c);
    let t3 = graph.concat_(vec![t, t, t], 1);
    let gap = graph.sub(one, shadow);
    let add = graph.mul(t3, gap);
    graph.add(shadow, add)
}

fn beauty_from_gathered(
    graph: &mut Graph,
    input: NodeId,
    base_colour: NodeId,
    irradiance: NodeId,
) -> NodeId {
    let albedo = graph.narrow_(input, 1, 3, OUT_CHANNELS);
    let one = graph.constant(1.0, DType::F32);
    let denom = graph.sub(one, base_colour);
    let probe_lin = graph.div(base_colour, denom);
    // GI belongs in shadow. `1 − compress(probe)` = `1/(1+L)`: furnace and
    // directly lit walls keep the gather, dark Cornell corners get the residual.
    let bounce = graph.mul(albedo, irradiance);
    let bounce = graph.mul(bounce, denom);
    let beauty_lin = graph.add(probe_lin, bounce);
    let beauty_lin = graph.relu(beauty_lin);
    let beauty_den = graph.add(one, beauty_lin);
    graph.div(beauty_lin, beauty_den)
}

/// Differentiable soft clamp via `cap * tanh(x / cap)`.
fn soft_clamp_irradiance(graph: &mut Graph, x: NodeId, cap: f32) -> NodeId {
    let c = graph.constant(cap as f64, DType::F32);
    let scaled = graph.div(x, c);
    let t = graph.tanh(scaled);
    graph.mul(c, t)
}

/// Softmax the predicted logits into a per-pixel filter and apply it to
/// compressed probe colour.
///
/// `logits` is `[n, (2r+1)², h, w]`. A constant is added to the centre tap so
/// that an untrained network (zero `out` weights → zero logits) starts as a
/// near-delta rather than a uniform box.
fn apply_predicted_kernel(
    graph: &mut Graph,
    colour: NodeId,
    logits: NodeId,
    radius: usize,
    _n: usize,
    h: usize,
    w: usize,
) -> NodeId {
    /// `exp(8) / (exp(8) + 48)` ≈ 0.984 of the weight on the centre tap of a 7×7.
    const CENTRE_LOGIT: f64 = 6.0;

    let side = 2 * radius + 1;
    let taps = side * side;
    let centre = taps / 2;

    // Bump the centre channel with a scalar add, not a concatenated constant
    // tensor: autodiff has been seen to fold large `full` constants together,
    // which would turn this back into a uniform box.
    let left = graph.narrow_(logits, 1, 0, centre);
    let mid = graph.narrow_(logits, 1, centre, 1);
    let right = graph.narrow_(logits, 1, centre + 1, taps - centre - 1);
    let bump = graph.constant(CENTRE_LOGIT, DType::F32);
    let mid = graph.add(mid, bump);
    let logits = graph.concat_(vec![left, mid, right], 1);
    // Softmax in rlx 0.2.14 is a contiguous-row kernel: taps have to be the
    // last axis or it normalises across width instead of across the filter.
    let taps_last = graph.transpose_(logits, vec![0, 2, 3, 1]);
    let weights_last = graph.sm(taps_last, -1);
    let weights = graph.transpose_(weights_last, vec![0, 3, 1, 2]);

    let padded = graph.pad_(
        colour,
        vec![[0, 0], [0, 0], [radius, radius], [radius, radius]],
        PadMode::Replicate,
    );

    let mut sum: Option<NodeId> = None;
    for dy in 0..side {
        for dx in 0..side {
            let rows = graph.narrow_(padded, 2, dy, h);
            let shifted = graph.narrow_(rows, 3, dx, w);
            let tap = graph.narrow_(weights, 1, dy * side + dx, 1);
            let tap = graph.concat_(vec![tap, tap, tap], 1);
            let term = graph.mul(shifted, tap);
            sum = Some(match sum {
                None => term,
                Some(acc) => graph.add(acc, term),
            });
        }
    }
    sum.expect("a kernel head has at least one tap")
}

fn spec(name: &'static str, shape: [usize; 4]) -> ParamSpec {
    ParamSpec { name, shape }
}

fn unet_params(widths: Widths, head: Head) -> Vec<ParamSpec> {
    let Widths {
        level0: w0,
        level1: w1,
        level2: w2,
        level3: w3,
    } = widths;
    vec![
        spec("enc0", [w0, IN_CHANNELS, K, K]),
        spec("down1", [w1, w0, K, K]),
        spec("down2", [w2, w1, K, K]),
        spec("down3", [w3, w2, K, K]),
        spec("bottleneck", [w3, w3, K, K]),
        spec("up3", [w2, w3, K, K]),
        spec("dec3", [w2, w2 * 2, K, K]),
        spec("up2", [w1, w2, K, K]),
        spec("dec2", [w1, w1 * 2, K, K]),
        spec("up1", [w0, w1, K, K]),
        spec("dec1", [w0, w0 * 2, K, K]),
        spec("out", [head.out_channels(), w0, K, K]),
    ]
}

fn decode_compressed(graph: &mut Graph, colour: NodeId) -> NodeId {
    let one = graph.constant(1.0, DType::F32);
    let denom = graph.sub(one, colour);
    graph.div(colour, denom)
}

fn encode_compressed(graph: &mut Graph, lin: NodeId) -> NodeId {
    let one = graph.constant(1.0, DType::F32);
    let positive = graph.relu(lin);
    let denom = graph.add(one, positive);
    graph.div(positive, denom)
}

/// On already-bright probe texels, keep the probe instead of a smoothed hop field.
fn blend_toward_probe_on_bright(graph: &mut Graph, input: NodeId, gathered: NodeId) -> NodeId {
    let probe = graph.narrow_(input, 1, 0, OUT_CHANNELS);
    let r = graph.narrow_(probe, 1, 0, 1);
    let g = graph.narrow_(probe, 1, 1, 1);
    let b = graph.narrow_(probe, 1, 2, 1);
    let rg = graph.add(r, g);
    let rgb = graph.add(rg, b);
    let three = graph.constant(3.0, DType::F32);
    let avg = graph.div(rgb, three);
    let thresh = graph.constant(0.50, DType::F32);
    let delta = graph.sub(avg, thresh);
    let over = graph.relu(delta);
    let scale = graph.constant(2.8, DType::F32);
    let t = graph.mul(over, scale);
    let t3 = graph.concat_(vec![t, t, t], 1);
    let one = graph.constant(1.0, DType::F32);
    let keep = graph.sub(one, t3);
    let from_probe = graph.mul(probe, t3);
    let from_gather = graph.mul(gathered, keep);
    graph.add(from_probe, from_gather)
}

/// Differentiable DDGI pull on neutral albedo — Cornell floor/ceiling wall bleed.
pub const GRAPH_ROOM_TINT: f32 = 0.26;

fn graph_room_tint() -> f32 {
    std::env::var("ROOM_TINT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(GRAPH_ROOM_TINT)
}

fn neutral_surface_weight(graph: &mut Graph, input: NodeId) -> NodeId {
    let ar = graph.narrow_(input, 1, 3, 1);
    let ag = graph.narrow_(input, 1, 4, 1);
    let ab = graph.narrow_(input, 1, 5, 1);
    let three = graph.constant(3.0, DType::F32);
    let sum = graph.add(ar, ag);
    let sum = graph.add(sum, ab);
    let mean = graph.div(sum, three);
    let dr = graph.sub(ar, mean);
    let dg = graph.sub(ag, mean);
    let db = graph.sub(ab, mean);
    let dr2 = graph.mul(dr, dr);
    let dg2 = graph.mul(dg, dg);
    let db2 = graph.mul(db, db);
    let var_sum = graph.add(dr2, dg2);
    let var_sum = graph.add(var_sum, db2);
    let var = graph.div(var_sum, three);
    let var_thresh = graph.constant(0.012f64, DType::F32);
    let var_delta = graph.sub(var_thresh, var);
    let neutral = graph.relu(var_delta);
    let inv = graph.constant(1.0 / 0.012, DType::F32);
    let neutral = graph.mul(neutral, inv);
    let alb_floor = graph.constant(0.45, DType::F32);
    let alb_delta = graph.sub(mean, alb_floor);
    let alb_relu = graph.relu(alb_delta);
    let alb_scale = graph.constant(4.0, DType::F32);
    let alb = graph.mul(alb_relu, alb_scale);
    let pr = graph.narrow_(input, 1, 0, 1);
    let pg = graph.narrow_(input, 1, 1, 1);
    let pb = graph.narrow_(input, 1, 2, 1);
    let psum = graph.add(pr, pg);
    let psum = graph.add(psum, pb);
    let probe_avg = graph.div(psum, three);
    let lo_floor = graph.constant(0.12, DType::F32);
    let lo_delta = graph.sub(probe_avg, lo_floor);
    let lo = graph.relu(lo_delta);
    let hi_ceil = graph.constant(0.68, DType::F32);
    let hi_delta = graph.sub(hi_ceil, probe_avg);
    let hi = graph.relu(hi_delta);
    let probe = graph.mul(lo, hi);
    let probe_scale = graph.constant(25.0, DType::F32);
    let probe = graph.mul(probe, probe_scale);
    let na = graph.mul(neutral, alb);
    graph.mul(na, probe)
}

fn linear_chroma_var(graph: &mut Graph, rgb: NodeId) -> NodeId {
    let r = graph.narrow_(rgb, 1, 0, 1);
    let g = graph.narrow_(rgb, 1, 1, 1);
    let b = graph.narrow_(rgb, 1, 2, 1);
    let three = graph.constant(3.0, DType::F32);
    let sum = graph.add(r, g);
    let sum = graph.add(sum, b);
    let mean = graph.div(sum, three);
    let dr = graph.sub(r, mean);
    let dg = graph.sub(g, mean);
    let db = graph.sub(b, mean);
    let dr2 = graph.mul(dr, dr);
    let dg2 = graph.mul(dg, dg);
    let db2 = graph.mul(db, db);
    let var_sum = graph.add(dr2, dg2);
    let var_sum = graph.add(var_sum, db2);
    graph.div(var_sum, three)
}

fn graph_blend_albedo_context(
    graph: &mut Graph,
    input: NodeId,
    gathered: NodeId,
    tint_src: NodeId,
    tint_amount: f32,
) -> NodeId {
    let gathered_lin = decode_compressed(graph, gathered);
    let tint_lin = decode_compressed(graph, tint_src);
    let tint_c = linear_chroma_var(graph, tint_lin);
    let pred_c = linear_chroma_var(graph, gathered_lin);
    let gap = graph.sub(tint_c, pred_c);
    let gap_relu = graph.relu(gap);
    let chroma_scale = graph.constant(250.0, DType::F32);
    let chroma = graph.mul(gap_relu, chroma_scale);
    let surface = neutral_surface_weight(graph, input);
    let mix_sc = graph.mul(surface, chroma);
    let tint = graph.constant(tint_amount as f64, DType::F32);
    let mix = graph.mul(mix_sc, tint);
    let mix3 = graph.concat_(vec![mix, mix, mix], 1);
    let one = graph.constant(1.0, DType::F32);
    let keep = graph.sub(one, mix3);
    let from_gather = graph.mul(gathered_lin, keep);
    let from_tint = graph.mul(tint_lin, mix3);
    let out_lin = graph.add(from_gather, from_tint);
    encode_compressed(graph, out_lin)
}

fn graph_blend_room_context(
    graph: &mut Graph,
    input: NodeId,
    gathered: NodeId,
    ddgi: NodeId,
) -> NodeId {
    graph_blend_albedo_context(graph, input, gathered, ddgi, graph_room_tint())
}

fn hop_params(hops: usize) -> Vec<ParamSpec> {
    let mut params = vec![
        spec("hop_feat0", [HOP_FEAT, IN_CHANNELS, 1, 1]),
        spec("hop_feat1", [HOP_FEAT, HOP_FEAT, 1, 1]),
    ];
    let names = ["hop0", "hop1", "hop2", "hop3", "hop4", "hop5"];
    for name in names.iter().take(hops.min(6)) {
        params.push(spec(name, [HOP_TAPS, HOP_FEAT + 1, 1, 1]));
    }
    params.push(spec("hop_res", [OUT_CHANNELS, HOP_FEAT + 1, 1, 1]));
    params
}

fn hop_forward(
    graph: &mut Graph,
    input: NodeId,
    p: &[NodeId],
    hops: usize,
    cap: f32,
    h: usize,
    w: usize,
) -> NodeId {
    let colour = graph.narrow_(input, 1, 0, OUT_CHANNELS);
    let ddgi = graph.narrow_(input, 1, 16, OUT_CHANNELS);
    let qprobe = graph.narrow_(input, 1, 13, OUT_CHANNELS);
    let demod = graph.narrow_(input, 1, 10, OUT_CHANNELS);
    let mut feat = graph.conv2d(input, p[0], [1, 1], [1, 1], [0, 0], [1, 1], 1);
    feat = graph.relu(feat);
    feat = graph.conv2d(feat, p[1], [1, 1], [1, 1], [0, 0], [1, 1], 1);
    feat = graph.relu(feat);
    // A constant-1 plane so each hop can learn a spatially uniform mix
    // (rlx conv2d has no bias).
    let ones = {
        let r = graph.narrow_(input, 1, 0, 1);
        let zero = graph.constant(0.0, DType::F32);
        let z = graph.mul(r, zero);
        let one = graph.constant(1.0, DType::F32);
        graph.add(z, one)
    };
    feat = graph.concat_(vec![feat, ones], 1);
    let mut light = decode_compressed(graph, colour);
    for i in 0..hops {
        let logits = graph.conv2d(feat, p[2 + i], [1, 1], [1, 1], [0, 0], [1, 1], 1);
        light = apply_cross_hop(graph, input, light, ddgi, qprobe, demod, logits, 1 << i, h, w);
    }
    let gathered = encode_compressed(graph, light);
    let gathered = blend_toward_probe_on_bright(graph, input, gathered);
    let gathered = graph_blend_room_context(graph, input, gathered, ddgi);
    let residual = graph.conv2d(feat, p[2 + hops], [1, 1], [1, 1], [0, 0], [1, 1], 1);
    let capped = soft_clamp_irradiance(graph, residual, cap);
    beauty_from_hops_gathered(graph, input, gathered, capped)
}

/// Dot-product coherence in `[0, 1]` — blocks hops across facing discontinuities.
fn normal_coherence(graph: &mut Graph, a: NodeId, b: NodeId) -> NodeId {
    let dot = normal_dot(graph, a, b);
    graph.relu(dot)
}

fn normal_dot(graph: &mut Graph, a: NodeId, b: NodeId) -> NodeId {
    let prod = graph.mul(a, b);
    let c0 = graph.narrow_(prod, 1, 0, 1);
    let c1 = graph.narrow_(prod, 1, 1, 1);
    let c2 = graph.narrow_(prod, 1, 2, 1);
    let s01 = graph.add(c0, c1);
    graph.add(s01, c2)
}

fn graph_abs(graph: &mut Graph, x: NodeId) -> NodeId {
    let zero = graph.constant(0.0, DType::F32);
    let pos = graph.relu(x);
    let neg_x = graph.sub(zero, x);
    let neg = graph.relu(neg_x);
    graph.add(pos, neg)
}

fn ortho_bleed_amount() -> f32 {
    std::env::var("ORTHO_BLEED")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0)
}

/// Aligned surfaces keep `relu(n·n')`. Near-orthogonal neighbours with similar
/// depth leak a little energy — wall→floor colour bleed that a normal gate zeros.
fn hop_tap_gate(
    graph: &mut Graph,
    centre_n: NodeId,
    nbr_n: NodeId,
    centre_d: NodeId,
    nbr_d: NodeId,
) -> NodeId {
    let aligned = normal_coherence(graph, centre_n, nbr_n);
    let bleed = ortho_bleed_amount();
    if bleed <= 0.0 {
        return aligned;
    }
    let dot = normal_dot(graph, centre_n, nbr_n);
    let absdot = graph_abs(graph, dot);
    let near = graph.constant(0.40, DType::F32);
    let ortho_gap = graph.sub(near, absdot);
    let ortho = graph.relu(ortho_gap);
    let dsub = graph.sub(centre_d, nbr_d);
    let dd = graph_abs(graph, dsub);
    let close_t = graph.constant(0.10, DType::F32);
    let close_gap = graph.sub(close_t, dd);
    let close = graph.relu(close_gap);
    let close_scale = graph.constant(10.0, DType::F32);
    let close = graph.mul(close, close_scale);
    let leak = graph.mul(ortho, close);
    let bleed_c = graph.constant(bleed as f64, DType::F32);
    let leak = graph.mul(leak, bleed_c);
    graph.add(aligned, leak)
}

fn luma_gate_on() -> bool {
    std::env::var("LUMA_GATE")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

fn rgb_luma(graph: &mut Graph, rgb: NodeId) -> NodeId {
    let r = graph.narrow_(rgb, 1, 0, 1);
    let g = graph.narrow_(rgb, 1, 1, 1);
    let b = graph.narrow_(rgb, 1, 2, 1);
    let s01 = graph.add(r, g);
    let sum = graph.add(s01, b);
    let three = graph.constant(3.0, DType::F32);
    graph.div(sum, three)
}

/// Neighbours darker than the centre cannot dump shadow onto a lit floor.
fn bright_neighbor_keep(graph: &mut Graph, centre: NodeId, nbr: NodeId) -> NodeId {
    let c = rgb_luma(graph, centre);
    let n = rgb_luma(graph, nbr);
    let eps = graph.constant(0.08, DType::F32);
    let den = graph.add(c, eps);
    let ratio = graph.div(n, den);
    let ratio = graph.relu(ratio);
    let one = graph.constant(1.0, DType::F32);
    let over = graph.sub(ratio, one);
    let over = graph.relu(over);
    graph.sub(ratio, over)
}

/// 12-tap hop in linear light with normal-gated spatial taps.
#[allow(clippy::too_many_arguments)]
fn apply_cross_hop(
    graph: &mut Graph,
    input: NodeId,
    light: NodeId,
    ddgi: NodeId,
    qprobe: NodeId,
    demod: NodeId,
    logits: NodeId,
    dilation: usize,
    h: usize,
    w: usize,
) -> NodeId {
    const CENTRE_LOGIT: f64 = 8.0;
    let centre = graph.narrow_(logits, 1, 0, 1);
    let rest = graph.narrow_(logits, 1, 1, HOP_TAPS - 1);
    let bump = graph.constant(CENTRE_LOGIT, DType::F32);
    let centre = graph.add(centre, bump);
    let logits = graph.concat_(vec![centre, rest], 1);
    let taps_last = graph.transpose_(logits, vec![0, 2, 3, 1]);
    let weights_last = graph.sm(taps_last, -1);
    let weights = graph.transpose_(weights_last, vec![0, 3, 1, 2]);

    let d = dilation;
    let pad_spec = vec![[0, 0], [0, 0], [d, d], [d, d]];
    let padded = graph.pad_(light, pad_spec.clone(), PadMode::Replicate);
    let normal = graph.narrow_(input, 1, 6, OUT_CHANNELS);
    let padded_n = graph.pad_(normal, pad_spec.clone(), PadMode::Replicate);
    let depth = graph.narrow_(input, 1, 9, 1);
    let padded_d = graph.pad_(depth, pad_spec, PadMode::Replicate);
    let sample = |graph: &mut Graph, padded: NodeId, py: usize, px: usize| {
        let rows = graph.narrow_(padded, 2, py, h);
        graph.narrow_(rows, 3, px, w)
    };
    let centre_px = sample(graph, padded, d, d);
    let centre_n = sample(graph, padded_n, d, d);
    let centre_d = sample(graph, padded_d, d, d);
    let north = sample(graph, padded, 0, d);
    let east = sample(graph, padded, d, 2 * d);
    let south = sample(graph, padded, 2 * d, d);
    let west = sample(graph, padded, d, 0);
    let ne = sample(graph, padded, 0, 2 * d);
    let nw = sample(graph, padded, 0, 0);
    let se = sample(graph, padded, 2 * d, 2 * d);
    let sw = sample(graph, padded, 2 * d, 0);
    let north_n = sample(graph, padded_n, 0, d);
    let east_n = sample(graph, padded_n, d, 2 * d);
    let south_n = sample(graph, padded_n, 2 * d, d);
    let west_n = sample(graph, padded_n, d, 0);
    let ne_n = sample(graph, padded_n, 0, 2 * d);
    let nw_n = sample(graph, padded_n, 0, 0);
    let se_n = sample(graph, padded_n, 2 * d, 2 * d);
    let sw_n = sample(graph, padded_n, 2 * d, 0);
    let north_d = sample(graph, padded_d, 0, d);
    let east_d = sample(graph, padded_d, d, 2 * d);
    let south_d = sample(graph, padded_d, 2 * d, d);
    let west_d = sample(graph, padded_d, d, 0);
    let ne_d = sample(graph, padded_d, 0, 2 * d);
    let nw_d = sample(graph, padded_d, 0, 0);
    let se_d = sample(graph, padded_d, 2 * d, 2 * d);
    let sw_d = sample(graph, padded_d, 2 * d, 0);
    let ddgi_lin = decode_compressed(graph, ddgi);
    let qprobe_lin = decode_compressed(graph, qprobe);
    let demod_lin = decode_compressed(graph, demod);
    let spatial = [
        (centre_px, None, None),
        (north, Some(north_n), Some(north_d)),
        (east, Some(east_n), Some(east_d)),
        (south, Some(south_n), Some(south_d)),
        (west, Some(west_n), Some(west_d)),
        (ne, Some(ne_n), Some(ne_d)),
        (nw, Some(nw_n), Some(nw_d)),
        (se, Some(se_n), Some(se_d)),
        (sw, Some(sw_n), Some(sw_d)),
    ];
    let global = [ddgi_lin, qprobe_lin, demod_lin];
    let mut sum: Option<NodeId> = None;
    for (t, (src, nbr, nbr_d)) in spatial.into_iter().enumerate() {
        let tap = graph.narrow_(weights, 1, t, 1);
        let tap = graph.concat_(vec![tap, tap, tap], 1);
        let mut term = graph.mul(src, tap);
        if let (Some(nbr_n), Some(nbr_d)) = (nbr, nbr_d) {
            let coh = hop_tap_gate(graph, centre_n, nbr_n, centre_d, nbr_d);
            term = graph.mul(term, coh);
            if luma_gate_on() {
                let keep = bright_neighbor_keep(graph, centre_px, src);
                let keep3 = graph.concat_(vec![keep, keep, keep], 1);
                term = graph.mul(term, keep3);
            }
        }
        sum = Some(match sum {
            None => term,
            Some(acc) => graph.add(acc, term),
        });
    }
    for (i, src) in global.into_iter().enumerate() {
        let t = spatial.len() + i;
        let tap = graph.narrow_(weights, 1, t, 1);
        let tap = graph.concat_(vec![tap, tap, tap], 1);
        let term = graph.mul(src, tap);
        sum = Some(match sum {
            None => term,
            Some(acc) => graph.add(acc, term),
        });
    }
    sum.expect("hop taps")
}

fn conv_relu(
    graph: &mut Graph,
    x: NodeId,
    weight: NodeId,
    stride: [usize; 2],
    padding: [usize; 2],
) -> NodeId {
    let y = graph.conv2d(x, weight, [K, K], stride, padding, [1, 1], 1);
    graph.relu(y)
}

/// Average-pool 2×, then a stride-1 3×3. The pool is a constant, not a parameter.
fn down(graph: &mut Graph, x: NodeId, weight: NodeId, channels: usize) -> NodeId {
    let pooled = {
        let k = graph.full(&[channels, 1, 2, 2], 0.25, DType::F32);
        graph.conv2d(x, k, [2, 2], [2, 2], [0, 0], [1, 1], channels)
    };
    conv_relu(graph, pooled, weight, [1, 1], [K / 2, K / 2])
}

/// Nearest-neighbour 2×, then a stride-1 3×3.
fn up(graph: &mut Graph, x: NodeId, weight: NodeId, channels: usize) -> NodeId {
    let doubled = {
        let k = graph.full(&[channels, 1, 2, 2], 1.0, DType::F32);
        graph.conv_transpose2d(x, k, [2, 2], [2, 2], [0, 0], [1, 1], [0, 0], channels)
    };
    conv_relu(graph, doubled, weight, [1, 1], [K / 2, K / 2])
}

/// He initialisation, with the output layer at exactly zero.
pub fn init_params(net: &ProbeArch, seed: u64) -> Vec<Vec<f32>> {
    net.params()
        .iter()
        .enumerate()
        .map(|(layer, spec)| {
            if spec.name == "out"
                || spec.name == "hop_res"
                || (spec.name.starts_with("hop") && !spec.name.starts_with("hop_feat"))
            {
                return vec![0.0; spec.elems()];
            }
            let scale = (2.0 / spec.fan_in() as f32).sqrt();
            (0..spec.elems())
                .map(|i| {
                    let h = mix64(seed ^ ((layer as u64) << 40) ^ i as u64);
                    let a = (h >> 40) as f32 / 16_777_216.0;
                    let b = ((h >> 16) & 0xff_ffff) as f32 / 16_777_216.0;
                    (a + b - 1.0) * scale * 2.449_5
                })
                .collect()
        })
        .collect()
}

pub(crate) fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn in_band(net: &ProbeArch) {
        let by_hand: usize = net.params().iter().map(|s| s.elems()).sum();
        assert_eq!(net.parameter_count(), by_hand);
        assert!(
            (15_000..55_000).contains(&net.parameter_count()),
            "unexpected size {}",
            net.parameter_count()
        );
    }

    #[test]
    fn tiny_is_tens_of_thousands_of_parameters() {
        in_band(&ProbeArch::new(Widths::tiny()));
        in_band(&ProbeArch::gathering(Widths::tiny()));
        in_band(&ProbeArch::hybrid(Widths::tiny()));
        let hops = ProbeArch::hops();
        assert!(
            hops.parameter_count() < 5_000,
            "hops should be tiny, got {}",
            hops.parameter_count()
        );
        assert!(hops.head().is_hops());
        assert!(
            ProbeArch::gathering(Widths::tiny()).parameter_count()
                > ProbeArch::new(Widths::tiny()).parameter_count()
        );
    }

    #[test]
    fn decoder_channels_match_the_skips() {
        let net = ProbeArch::gathering(Widths::tiny());
        let w = net.widths();
        let find = |name: &str| {
            net.params()
                .iter()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("missing {name}"))
                .shape
        };
        assert_eq!(find("dec3")[1], w.level2 * 2);
        assert_eq!(find("dec2")[1], w.level1 * 2);
        assert_eq!(find("dec1")[1], w.level0 * 2);
        assert_eq!(find("up3")[0], w.level2);
        assert_eq!(find("up2")[0], w.level1);
        assert_eq!(find("up1")[0], w.level0);
        assert_eq!(find("out")[0], 7 * 7);
        let hybrid = ProbeArch::hybrid(Widths::tiny());
        assert_eq!(
            hybrid
                .params()
                .iter()
                .find(|s| s.name == "out")
                .unwrap()
                .shape[0],
            7 * 7 + OUT_CHANNELS
        );
    }

    #[test]
    fn the_output_layer_starts_at_zero() {
        for net in [
            ProbeArch::new(Widths::tiny()),
            ProbeArch::gathering(Widths::tiny()),
            ProbeArch::hops(),
        ] {
            let init = init_params(&net, 7);
            assert!(init.last().unwrap().iter().all(|v| *v == 0.0));
            assert!(init[0].iter().any(|v| *v != 0.0));
        }
    }
}
