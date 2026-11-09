//! Bindings split out of `wasm.rs` by area (see the `mod` list there): stats, clock.

use super::*;

// ---- stats ----
#[wasm_bindgen]
pub struct WebStats {
    pub(super) inner: crate::Stats,
}
#[wasm_bindgen]
impl WebStats {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebStats {
        WebStats {
            inner: crate::Stats::new(),
        }
    }
    pub fn begin(&mut self) {
        self.inner.begin();
    }
    pub fn end(&mut self) {
        self.inner.end();
    }
    #[wasm_bindgen(getter)]
    pub fn fps(&self) -> f32 {
        self.inner.fps
    }
    #[wasm_bindgen(js_name = frameMs, getter)]
    pub fn frame_ms(&self) -> f32 {
        self.inner.frame_ms
    }
}

// ---- clock ----
#[wasm_bindgen]
pub struct WebClock {
    pub(super) inner: crate::Clock,
}
#[wasm_bindgen]
impl WebClock {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebClock {
        WebClock {
            inner: crate::Clock::new(true),
        }
    }
    #[wasm_bindgen(js_name = getDelta)]
    pub fn get_delta(&mut self) -> f64 {
        self.inner.get_delta()
    }
    #[wasm_bindgen(js_name = getElapsedTime)]
    pub fn get_elapsed_time(&mut self) -> f64 {
        self.inner.get_elapsed_time()
    }
}
