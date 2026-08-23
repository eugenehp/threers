//! Browser inference — load trained weights and reconstruct Cornell (or any packed frame).

use wasm_bindgen::prelude::*;

use rlx::Device;

use crate::infer::ProbeGi;
use crate::live_cornell;
use crate::{expand, HOPS_NRC_CAP, HOPS_NRC_PROBE_CONFIDENCE, IN_CHANNELS, OUT_CHANNELS};

fn js_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

#[wasm_bindgen]
pub fn probe_in_channels() -> u32 {
    IN_CHANNELS as u32
}

#[wasm_bindgen]
pub fn probe_out_channels() -> u32 {
    OUT_CHANNELS as u32
}

/// Hops stack + NRC fused reconstructor for one square frame.
#[wasm_bindgen]
pub struct WebProbeGi {
    stack: ProbeGi,
    side: usize,
}

#[wasm_bindgen]
impl WebProbeGi {
    /// `probe_weights` is a `THRSW003` blob; `nrc_weights` is a `THRSN002` blob (empty = hops only).
    #[wasm_bindgen(constructor)]
    pub fn new(probe_weights: &[u8], nrc_weights: &[u8], side: u32) -> Result<WebProbeGi, JsValue> {
        let side = side as usize;
        if side < 8 || side % 8 != 0 {
            return Err(JsValue::from_str("side must be a multiple of 8"));
        }
        let nrc = if nrc_weights.is_empty() {
            None
        } else {
            Some(nrc_weights)
        };
        let stack = ProbeGi::load_from_bytes(
            probe_weights,
            nrc,
            side,
            side,
            Device::Cpu,
            HOPS_NRC_PROBE_CONFIDENCE,
            HOPS_NRC_CAP,
        )
        .map_err(js_err)?;
        Ok(Self { stack, side })
    }

    #[wasm_bindgen(getter)]
    pub fn side(&self) -> u32 {
        self.side as u32
    }

    /// Planar input `[IN, H, W]` → planar beauty `[3, H, W]` (compressed, same as training pack).
    pub fn reconstruct(&mut self, input: &[f32]) -> Result<Vec<f32>, JsValue> {
        let need = IN_CHANNELS * self.side * self.side;
        if input.len() != need {
            return Err(JsValue::from_str(&format!(
                "input length {} != {need} (IN={IN_CHANNELS}, side={})",
                input.len(),
                self.side
            )));
        }
        self.stack.reconstruct(input).map_err(js_err)
    }

    /// Hops output only (no NRC fuse).
    pub fn reconstruct_hops(&mut self, input: &[f32]) -> Result<Vec<f32>, JsValue> {
        let need = IN_CHANNELS * self.side * self.side;
        if input.len() != need {
            return Err(JsValue::from_str(&format!(
                "input length {} != {need} (side={})",
                input.len(),
                self.side
            )));
        }
        self.stack
            .unet_mut()
            .reconstruct(input, self.side, self.side)
            .map_err(js_err)
    }

    /// Path-trace the canonical Cornell box for this camera, pack 22 planes, run fused GI.
    ///
    /// Returns planar `[3, side, side]` compressed beauty. `spp`/`bounces` control the
    /// lighting probe only (G-buffer is always 1 ray per pixel).
    pub fn infer_cornell(
        &mut self,
        cam_x: f32,
        cam_y: f32,
        cam_z: f32,
        tgt_x: f32,
        tgt_y: f32,
        tgt_z: f32,
        fov_deg: f32,
        spp: u32,
        bounces: u32,
    ) -> Result<Vec<f32>, JsValue> {
        let input = live_cornell::pack_view(
            self.side,
            [cam_x, cam_y, cam_z],
            [tgt_x, tgt_y, tgt_z],
            fov_deg,
            spp,
            bounces,
        )
        .map_err(js_err)?;
        self.stack.reconstruct(&input).map_err(js_err)
    }

    /// Same as [`Self::infer_cornell`] but returns hops only (no NRC).
    pub fn infer_cornell_hops(
        &mut self,
        cam_x: f32,
        cam_y: f32,
        cam_z: f32,
        tgt_x: f32,
        tgt_y: f32,
        tgt_z: f32,
        fov_deg: f32,
        spp: u32,
        bounces: u32,
    ) -> Result<Vec<f32>, JsValue> {
        let input = live_cornell::pack_view(
            self.side,
            [cam_x, cam_y, cam_z],
            [tgt_x, tgt_y, tgt_z],
            fov_deg,
            spp,
            bounces,
        )
        .map_err(js_err)?;
        self.stack
            .unet_mut()
            .reconstruct(&input, self.side, self.side)
            .map_err(js_err)
    }

    /// Probe RGB planes only (compressed) for the same view — for side-by-side UI.
    pub fn pack_cornell_probe(
        &mut self,
        cam_x: f32,
        cam_y: f32,
        cam_z: f32,
        tgt_x: f32,
        tgt_y: f32,
        tgt_z: f32,
        fov_deg: f32,
        spp: u32,
        bounces: u32,
    ) -> Result<Vec<f32>, JsValue> {
        let input = live_cornell::pack_view(
            self.side,
            [cam_x, cam_y, cam_z],
            [tgt_x, tgt_y, tgt_z],
            fov_deg,
            spp,
            bounces,
        )
        .map_err(js_err)?;
        let n = self.side * self.side;
        Ok(input[..3 * n].to_vec())
    }
}

/// Spherical orbit helper: yaw/pitch/distance → camera position (target fixed at room centre).
#[wasm_bindgen]
pub fn cornell_orbit_camera(yaw: f32, pitch: f32, distance: f32) -> Vec<f32> {
    let (pos, tgt) = live_cornell::orbit_camera(yaw, pitch, distance);
    vec![pos[0], pos[1], pos[2], tgt[0], tgt[1], tgt[2]]
}

/// Expand compressed planar RGB to linear `[r,g,b,a, …]` for canvas display.
#[wasm_bindgen]
pub fn expand_planar_rgb(compressed: &[f32], side: u32) -> Vec<f32> {
    let side = side as usize;
    let n = side * side;
    if compressed.len() < OUT_CHANNELS * n {
        return Vec::new();
    }
    let mut out = vec![0.0f32; n * 4];
    for i in 0..n {
        let o = i * 4;
        out[o] = expand(compressed[i]).clamp(0.0, 1.0);
        out[o + 1] = expand(compressed[n + i]).clamp(0.0, 1.0);
        out[o + 2] = expand(compressed[2 * n + i]).clamp(0.0, 1.0);
        out[o + 3] = 1.0;
    }
    out
}
