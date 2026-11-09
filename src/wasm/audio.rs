//! Bindings split out of `wasm.rs` by area (see the `mod` list there): audio.

use super::*;

// ---- audio ----
#[cfg(feature = "audio")]
#[wasm_bindgen]
pub struct WebAudioListener {
    pub(super) inner: crate::AudioListener,
}
#[cfg(feature = "audio")]
#[wasm_bindgen]
impl WebAudioListener {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebAudioListener {
        WebAudioListener {
            inner: crate::AudioListener::default(),
        }
    }
    #[wasm_bindgen(js_name = setMasterVolume)]
    pub fn set_master_volume(&mut self, v: f32) {
        self.inner.master_volume = v;
    }
}

#[cfg(feature = "audio")]
#[wasm_bindgen]
pub struct WebAudio {
    pub(super) inner: crate::Audio,
}
#[cfg(feature = "audio")]
#[wasm_bindgen]
impl WebAudio {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebAudio {
        WebAudio {
            inner: crate::Audio::default(),
        }
    }
    #[wasm_bindgen(js_name = setVolume)]
    pub fn set_volume(&mut self, v: f32) {
        self.inner.volume = v;
    }
    #[wasm_bindgen(js_name = setLoop)]
    pub fn set_loop(&mut self, l: bool) {
        self.inner.loop_ = l;
    }
    pub fn play(&mut self) {
        self.inner.playing = true;
    }
    pub fn stop(&mut self) {
        self.inner.playing = false;
    }
}
