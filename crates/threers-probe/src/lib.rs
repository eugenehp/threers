//! Screen-space neural GI from a G-buffer and a lighting probe.
//!
//! The path tracer is the teacher. At inference the network sees first-hit
//! albedo, normal and depth, a multi-bounce lighting probe, quarter-res probe
//! context, DDGI-lite world bins, and encoded hit positions. Heads: Direct
//! irradiance residual, 7×7 kernel gather, hybrid (KPN + capped residual),
//! hops (à-trous light jumps + world-grid teleport), plus an optional
//! NRC-style 1×1 MLP for per-pixel world residuals. The training family
//! includes jittered furnace-like shells so hops learn uniform-bright identity.
//!
//! Built from `conv2d` / `relu` / `concat` / a fixed 2× pool, so the same
//! graph trains on Metal or CUDA and runs on the CPU the browser already has.

pub mod checkpoint;
pub mod dataset;
pub mod ddgi;
pub mod infer;
pub mod live_cornell;
pub mod model;
pub mod nrc;
pub mod pack;
pub mod train;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

/// Procedural rooms and the hostile hold-out (`generate` feature).
#[cfg(feature = "generate")]
pub mod generate;

pub use dataset::Dataset;
pub use infer::{ProbeGi, ProbeNet};
pub use nrc::{
    fuse_beauty, fuse_beauty_with, init_params as init_nrc_params, load_nrc, load_nrc_bytes, mean_albedo, nrc_apply_mask,
    nrc_eligible, nrc_features, nrc_input_channels, pack_nrc_batch, pack_nrc_batch_with, probe_colour_relative_sq,
    save_nrc, train_nrc, train_nrc_from, train_nrc_dataset, NrcArch, NrcNet, NRC_CAP, NRC_IN, NRC_IN_DDGI,
    NRC_MAX_PROBE, NRC_MIN_ALBEDO, NRC_OUT, NRC_PROBE_CONFIDENCE, HOPS_NRC_PROBE_CONFIDENCE, HOPS_NRC_CAP,
};
pub use ddgi::{
    bounds_from_world, encode_frame, encode_world_planes, DdgiContext, DDGI_GRID, DDGI_GRID_Y,
    DDGI_VIS_PLANE,
};
pub use pack::{colour_planes, geometry_hits, planes, PlaneExtras, DEPTH_PLANE, SKY_DEPTH_COMPRESSED};
pub use model::{
    init_params, Head, ProbeArch, Widths, ALBEDO_FLOOR, GRAPH_ROOM_TINT, HOPS_IRRADIANCE_CAP,
    HYBRID_IRRADIANCE_CAP,
    IN_CHANNELS, OUT_CHANNELS,
};
pub use train::{Batch, TrainConfig, Trainer, GI_LOSS_BOOST, HOPS_CHROMA_LOSS_BOOST, HOPS_GI_LOSS_BOOST, HOPS_NEUTRAL_LOSS_BOOST, LOSS_EPSILON, TRAIN_EPSILON};

use rlx::Device;

/// The device this crate actually trains and infers on.
///
/// Metal when the `metal` feature is on and a device is present; otherwise the
/// CPU. The GPU feature is rlx's wgpu, a different device from the renderer.
pub fn preferred_device() -> Device {
    #[cfg(feature = "metal")]
    {
        if rlx::is_available(Device::Metal) {
            return Device::Metal;
        }
    }
    #[cfg(feature = "gpu")]
    {
        if rlx::is_available(Device::Gpu) {
            return Device::Gpu;
        }
    }
    Device::Cpu
}

/// Compress unbounded radiance into `[0, 1)` for the convolution stack.
///
/// `x / (1 + x)` is monotone and invertible as [`expand`]. Non-finite values
/// would poison every layer that touches them: +∞ is maximum energy (1),
/// anything else (NaN, −∞) is 0.
pub fn compress(v: f32) -> f32 {
    if !v.is_finite() {
        return if v > 0.0 { 1.0 } else { 0.0 };
    }
    (v.max(0.0) / (1.0 + v.max(0.0))).min(0.999)
}

/// Inverse of [`compress`]. Values at 1 (compressed infinity) stay large
/// rather than dividing by zero.
pub fn expand(v: f32) -> f32 {
    let v = v.clamp(0.0, 0.999_999);
    v / (1.0 - v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_round_trips_finite_radiance() {
        for v in [0.0, 0.01, 0.5, 1.0, 8.0, 100.0] {
            let back = expand(compress(v));
            assert!(
                (back - v).abs() / v.max(1e-6) < 1e-5,
                "{v} -> {back}"
            );
        }
        assert_eq!(compress(f32::INFINITY), 1.0);
        assert_eq!(compress(f32::NAN), 0.0);
        assert_eq!(compress(-3.0), 0.0);
    }
}
