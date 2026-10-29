//! `WebTimeSmear`, behind the shim's `THREE.TimeSmear`. It holds a clip,
//! takes its settings as a plain options object and its looks by name,
//! smears lazily (once per change of clip or settings, on the next frame
//! asked for), and composites any frame in any look on demand — so a look or
//! gain change during playback costs one frame, not the clip. The shim only
//! drives the capture, which has to go through the JavaScript renderer.

use wasm_bindgen::prelude::*;
use wasm_bindgen::Clamped;

use crate::time_smear::{composite, downsample_luma, Adaptive, SmearLook, TimeSmear};

#[wasm_bindgen]
pub struct WebTimeSmear {
    smear: TimeSmear,
    width: usize,
    height: usize,
    frames: Vec<Vec<u8>>,
    /// Working-size luma of every frame, and the size it was taken at.
    work: (usize, usize),
    luma: Vec<f32>,
    /// `None` until smeared, and again after any change.
    echo: Option<Vec<f32>>,
}

#[wasm_bindgen]
impl WebTimeSmear {
    /// An empty clip of `width × height` frames; `frames`, if given, is RGBA8
    /// frames back to back.
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32, frames: Option<Vec<u8>>) -> Result<WebTimeSmear, JsError> {
        if width == 0 || height == 0 {
            return Err(JsError::new("THREE.TimeSmear: width and height must be positive"));
        }
        let mut s = WebTimeSmear {
            smear: TimeSmear::new(),
            width: width as usize,
            height: height as usize,
            frames: Vec::new(),
            work: (0, 0),
            luma: Vec::new(),
            echo: None,
        };
        if let Some(flat) = frames {
            let size = s.frame_size();
            if flat.len() % size != 0 {
                return Err(JsError::new("THREE.TimeSmear: frames are not whole width × height RGBA frames"));
            }
            for f in flat.chunks(size) {
                s.push(f)?;
            }
        }
        Ok(s)
    }

    /// Append one RGBA8 frame.
    pub fn push(&mut self, rgba: &[u8]) -> Result<(), JsError> {
        if rgba.len() != self.frame_size() {
            return Err(JsError::new(&format!(
                "THREE.TimeSmear: frame is {} bytes, expected {}×{}×4",
                rgba.len(),
                self.width,
                self.height
            )));
        }
        if self.work != (0, 0) {
            self.luma.extend(downsample_luma(rgba, self.width, self.height, self.work));
        }
        self.frames.push(rgba.to_vec());
        self.echo = None;
        Ok(())
    }

    #[wasm_bindgen(getter, js_name = frameCount)]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width as u32
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height as u32
    }

    /// Change any of `{ strength, reach, rotation, workWidth, adaptive }`;
    /// the rest keep their values. `rotation` is `'x'`, `'y'`, `'xy'` or
    /// `'yx'`; `adaptive` is a bool or `{ motionRef, floor }`. Re-smears now
    /// and returns the milliseconds it took.
    pub fn set(&mut self, params: JsValue) -> Result<f64, JsError> {
        let get = |key: &str| js_sys::Reflect::get(&params, &key.into()).ok().filter(|v| !v.is_undefined());
        let num = |key: &str| get(key).and_then(|v| v.as_f64());
        if let Some(v) = num("strength") {
            self.smear.strength = v as f32;
        }
        if let Some(v) = num("reach") {
            self.smear.reach = v as f32;
        }
        if let Some(v) = num("workWidth") {
            self.smear.work_width = (v as u32).max(1);
        }
        if let Some(v) = get("rotation") {
            let name = v.as_string().unwrap_or_default();
            self.smear.rotation = name.parse().map_err(|e: String| JsError::new(&format!("THREE.TimeSmear: {e}")))?;
        }
        if let Some(v) = get("adaptive") {
            self.smear.adaptive = if let Some(on) = v.as_bool() {
                on.then(Adaptive::default)
            } else if v.is_object() {
                let field = |k: &str| js_sys::Reflect::get(&v, &k.into()).ok().and_then(|x| x.as_f64());
                let d = Adaptive::default();
                Some(Adaptive {
                    motion_ref: field("motionRef").map_or(d.motion_ref, |x| x as f32),
                    floor: field("floor").map_or(d.floor, |x| x as f32),
                })
            } else {
                None
            };
        }
        if let Some(v) = num("dark") {
            self.smear.dark = v as f32;
        }
        self.echo = None;
        let t0 = js_sys::Date::now();
        self.ensure_smeared();
        Ok(js_sys::Date::now() - t0)
    }

    /// The current settings, as the object `set` takes.
    pub fn params(&self) -> JsValue {
        let o = js_sys::Object::new();
        let put = |k: &str, v: JsValue| {
            let _ = js_sys::Reflect::set(&o, &k.into(), &v);
        };
        let s = &self.smear;
        put("strength", s.strength.into());
        put("reach", s.reach.into());
        put("rotation", format!("{:?}", s.rotation).to_lowercase().into());
        put("workWidth", s.work_width.into());
        put("adaptive", s.adaptive.is_some().into());
        put("dark", s.dark.into());
        o.into()
    }

    /// Frame `i` as RGBA8 in a look: `'faithful'`, `'ghost'`, `'raw'`,
    /// `'echo'` or `'source'`. `gain` omitted: the look's own.
    pub fn frame(&mut self, i: usize, look: Option<String>, gain: Option<f32>) -> Result<Vec<u8>, JsError> {
        if self.frames.is_empty() {
            return Err(JsError::new("THREE.TimeSmear: no frames"));
        }
        let i = i.min(self.frames.len() - 1);
        let look = look.unwrap_or_else(|| "faithful".into());
        if look.eq_ignore_ascii_case("source") {
            return Ok(self.frames[i].clone());
        }
        self.smear.look = look.parse::<SmearLook>().map_err(|e| JsError::new(&format!("THREE.TimeSmear: {e}")))?;
        self.smear.gain = gain;
        self.ensure_smeared();
        let px = self.work.0 * self.work.1;
        let echo = self.echo.as_ref().expect("smeared");
        Ok(composite(
            &self.smear,
            &self.frames[i],
            &self.luma[i * px..][..px],
            &echo[i * px..][..px],
            self.width,
            self.height,
            self.work,
        ))
    }

    /// The same, as an `ImageData` for a 2D canvas's `putImageData`.
    #[wasm_bindgen(js_name = imageData)]
    pub fn image_data(&mut self, i: usize, look: Option<String>, gain: Option<f32>) -> Result<web_sys::ImageData, JsError> {
        let rgba = self.frame(i, look, gain)?;
        web_sys::ImageData::new_with_u8_clamped_array_and_sh(Clamped(&rgba), self.width as u32, self.height as u32)
            .map_err(|_| JsError::new("THREE.TimeSmear: ImageData rejected the frame"))
    }
}

impl WebTimeSmear {
    fn frame_size(&self) -> usize {
        self.width * self.height * 4
    }

    fn ensure_smeared(&mut self) {
        if self.frames.is_empty() {
            return;
        }
        let work = self.smear.work_size(self.width as u32, self.height as u32);
        if work != self.work {
            self.work = work;
            self.luma = self
                .frames
                .iter()
                .flat_map(|f| downsample_luma(f, self.width, self.height, work))
                .collect();
            self.echo = None;
        }
        if self.echo.is_none() {
            self.echo = Some(self.smear.echo_volume(&self.luma, work.0, work.1));
        }
    }
}
