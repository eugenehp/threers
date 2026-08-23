//! Browser bindings for the trained denoiser, shaped for web workers.
//!
//! The network is a couple of gigaflops a frame, which is more than the main
//! thread can spend without dropping the UI. The way round that is not a faster
//! kernel — it is running the tiles at the same time, and a browser's unit of
//! "at the same time" is a worker.
//!
//! So the API is deliberately split rather than offering one `denoise(frame)`
//! call:
//!
//! | | runs on | |
//! |---|---|---|
//! | [`WebDenoiser::new`] | any thread | parse the weights once |
//! | [`WebDenoiser::plan`] | main | decide the tiles |
//! | [`WebDenoiser::extract`] | main | cut one tile out, to post to a worker |
//! | [`WebDenoiser::run`] | **worker** | the expensive part |
//! | [`WebDenoiser::merge`] | main | put the result back |
//!
//! Every tile is independent: none reads another's output and none shares
//! state, so N workers each taking a share of the list produce the same frame
//! as doing them one at a time — which the native test
//! `tiling_a_frame_matches_denoising_it_whole` pins.
//!
//! # Wiring it up
//!
//! Each worker needs its own `WebDenoiser`, because a `wasm-bindgen` object
//! cannot cross a `postMessage`. The weights can: they are a plain
//! `Uint8Array`, so fetch them once on the main thread and post the same buffer
//! to every worker, which is a copy of a few megabytes rather than a refetch.
//!
//! ```js
//! // main.js
//! const weights = new Uint8Array(await (await fetch('denoise.bin')).arrayBuffer());
//! const pool = Array.from({length: navigator.hardwareConcurrency}, () => {
//!   const w = new Worker('denoise-worker.js', {type: 'module'});
//!   w.postMessage({weights}, [/* keep weights, they are reused */]);
//!   return w;
//! });
//!
//! const den = new WebDenoiser(weights);
//! const plan = den.plan(width, height, 128, 24);
//! // Hand tile i to worker i % pool.length, transferring the patch so the
//! // copy is free.
//! for (let i = 0; i < plan.count(); i++) {
//!   const patch = den.extract(frame, width, height, plan, i);
//!   pool[i % pool.length].postMessage({i, patch, w: plan.tileWidth(i), h: plan.tileHeight(i)},
//!                                     [patch.buffer]);
//! }
//! // As each result comes back:
//! //   den.merge(out, width, height, plan, i, result);
//! ```
//!
//! ```js
//! // denoise-worker.js
//! import init, {WebDenoiser} from './threers.js';
//! let den;
//! onmessage = async (e) => {
//!   if (e.data.weights) { await init(); den = new WebDenoiser(e.data.weights); return; }
//!   const {i, patch, w, h} = e.data;
//!   const out = den.run(patch, w, h);
//!   postMessage({i, out}, [out.buffer]);
//! };
//! ```

use wasm_bindgen::prelude::*;

use crate::raytrace::denoise_net::{Denoiser, TilePlan};

/// A tiling of one frame. Opaque to JS; index into it with the accessors.
#[wasm_bindgen]
pub struct WebTilePlan {
    inner: TilePlan,
    /// Padded tile sizes, which is what [`WebDenoiser::run`] must be told.
    sizes: Vec<(usize, usize)>,
}

#[wasm_bindgen]
impl WebTilePlan {
    /// How many tiles the frame was cut into.
    pub fn count(&self) -> usize {
        self.inner.tiles.len()
    }

    /// Width to pass to `run` for tile `i` — padded up to the network's
    /// multiple, so not always the same as the region it came from.
    #[wasm_bindgen(js_name = tileWidth)]
    pub fn tile_width(&self, i: usize) -> usize {
        self.sizes.get(i).map(|s| s.0).unwrap_or(0)
    }

    /// Height to pass to `run` for tile `i`.
    #[wasm_bindgen(js_name = tileHeight)]
    pub fn tile_height(&self, i: usize) -> usize {
        self.sizes.get(i).map(|s| s.1).unwrap_or(0)
    }
}

/// The denoiser, as a browser sees it.
#[wasm_bindgen]
pub struct WebDenoiser {
    net: Denoiser,
}

#[wasm_bindgen]
impl WebDenoiser {
    /// Parse a checkpoint. The bytes are not retained, so the caller may reuse
    /// or transfer the array afterwards.
    #[wasm_bindgen(constructor)]
    pub fn new(weights: &[u8]) -> Result<WebDenoiser, JsError> {
        let net = Denoiser::from_bytes(weights).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { net })
    }

    /// Input planes this checkpoint expects, so JS can lay out the buffer.
    pub fn inputs(&self) -> usize {
        self.net.inputs()
    }

    /// Weights in the network, for reporting.
    #[wasm_bindgen(js_name = parameterCount)]
    pub fn parameter_count(&self) -> usize {
        self.net.parameter_count()
    }

    /// Cut a frame into tiles of about `target` pixels a side, overlapping by
    /// `margin`.
    ///
    /// A margin smaller than the network's receptive field lets the zero
    /// padding at a tile's edge reach the part that is kept, which shows up as
    /// a faint grid. 24 is comfortable for the shipped widths.
    pub fn plan(&self, width: usize, height: usize, target: usize, margin: usize) -> WebTilePlan {
        let inner = TilePlan::new(width, height, target, margin);
        let sizes = inner
            .tiles
            .iter()
            // The padded size is a function of the tile's own extent, so it can
            // be computed without touching the frame.
            .map(|t| (round_up8(t.width), round_up8(t.height)))
            .collect();
        WebTilePlan { inner, sizes }
    }

    /// Cut tile `index` out of a frame, ready to post to a worker.
    pub fn extract(
        &self,
        frame: &[f32],
        width: usize,
        height: usize,
        plan: &WebTilePlan,
        index: usize,
    ) -> Result<Vec<f32>, JsError> {
        let tile = plan
            .inner
            .tiles
            .get(index)
            .ok_or_else(|| JsError::new("tile index out of range"))?;
        if frame.len() != self.net.inputs() * width * height {
            return Err(JsError::new(
                "frame is not inputs() * width * height floats",
            ));
        }
        let (patch, _, _) = self.net.extract_tile(frame, width, height, tile);
        Ok(patch)
    }

    /// Denoise one tile. This is the part that belongs on a worker.
    pub fn run(&self, patch: &[f32], width: usize, height: usize) -> Result<Vec<f32>, JsError> {
        self.net
            .denoise(patch, width, height)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Write a worker's result into the frame buffer.
    pub fn merge(
        &self,
        frame: &mut [f32],
        width: usize,
        height: usize,
        plan: &WebTilePlan,
        index: usize,
        denoised: &[f32],
    ) -> Result<(), JsError> {
        let tile = plan
            .inner
            .tiles
            .get(index)
            .ok_or_else(|| JsError::new("tile index out of range"))?;
        let tw = plan.tile_width(index);
        self.net
            .merge_tile(denoised, tw, tile, frame, width, height);
        Ok(())
    }

    /// Denoise a whole frame on the calling thread.
    ///
    /// Correct and simple, and it will block whichever thread runs it for as
    /// long as the whole frame takes — fine inside a single worker, wrong on
    /// the main thread of a page that also wants to stay responsive.
    #[wasm_bindgen(js_name = denoiseFrame)]
    pub fn denoise_frame(
        &self,
        frame: &[f32],
        width: usize,
        height: usize,
        target: usize,
        margin: usize,
    ) -> Result<Vec<f32>, JsError> {
        let plan = TilePlan::new(width, height, target, margin);
        self.net
            .denoise_frame(frame, width, height, &plan)
            .map_err(|e| JsError::new(&e.to_string()))
    }
}

fn round_up8(v: usize) -> usize {
    v.div_ceil(8) * 8
}
