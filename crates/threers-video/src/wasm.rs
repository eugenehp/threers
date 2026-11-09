//! The video player for JavaScript, beside the rest of the `THREE.*`
//! bindings in `threers::wasm`.

use threers::wasm::WebDataTexture;
use wasm_bindgen::prelude::*;

/// A video playing into a texture, for JavaScript: `WebVideoPlayer.open(bytes)`,
/// put [`texture`](Self::texture) in a material, call [`update`](Self::update)
/// each frame with the seconds that passed.
///
/// Decoding is the browser's (WebCodecs), in hardware where it has it, and
/// its frames are copied into the texture GPU to GPU; a browser without
/// WebCodecs, or one that refuses the codec, gets the native decoders.
#[wasm_bindgen]
pub struct WebVideoPlayer {
    inner: crate::player::VideoPlayer,
}

#[wasm_bindgen]
impl WebVideoPlayer {
    /// Open MP4/MOV, WebM/Matroska or IVF bytes.
    pub fn open(data: Vec<u8>) -> Result<WebVideoPlayer, JsValue> {
        crate::player::VideoPlayer::open(data)
            .map(|inner| WebVideoPlayer { inner })
            .map_err(|e| JsValue::from_str(&e))
    }

    /// The texture the video plays in, for a material's map.
    pub fn texture(&self) -> WebDataTexture {
        WebDataTexture::from_texture(self.inner.texture())
    }

    /// Move playback on by `seconds` and show the picture due.
    pub fn update(&mut self, seconds: f64) {
        self.inner.update(seconds);
    }

    /// Where playback is, in seconds.
    #[wasm_bindgen(getter)]
    pub fn time(&self) -> f64 {
        self.inner.time()
    }

    /// The clip's length in seconds.
    #[wasm_bindgen(getter)]
    pub fn duration(&self) -> f64 {
        self.inner.duration()
    }

    /// Pictures shown so far.
    #[wasm_bindgen(getter, js_name = framesShown)]
    pub fn frames_shown(&self) -> f64 {
        self.inner.video_texture().stream().frames() as f64
    }

    /// `"browser"` when WebCodecs decodes, `"inline"` when the native
    /// decoders do.
    #[wasm_bindgen(getter)]
    pub fn decoder(&self) -> String {
        format!("{:?}", self.inner.decoder()).to_lowercase()
    }

    /// Why decoding stopped, if it did.
    #[wasm_bindgen(getter)]
    pub fn error(&self) -> Option<String> {
        self.inner.error().map(str::to_owned)
    }

    #[wasm_bindgen(js_name = setLooping)]
    pub fn set_looping(&mut self, looping: bool) {
        self.inner.set_looping(looping);
    }

    #[wasm_bindgen(js_name = setSpeed)]
    pub fn set_speed(&mut self, speed: f64) {
        self.inner.set_speed(speed);
    }

    pub fn play(&mut self) {
        self.inner.play();
    }

    pub fn pause(&mut self) {
        self.inner.pause();
    }

    pub fn seek(&mut self, seconds: f64) {
        self.inner.seek(seconds);
    }
}
