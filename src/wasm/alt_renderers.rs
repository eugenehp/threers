//! Bindings split out of `wasm.rs` by area (see the `mod` list there): alt renderers.

use super::*;

// ---- alt renderers ----
#[wasm_bindgen]
pub struct WebCss2dRenderer {
    pub(super) inner: crate::Css2dRenderer,
}
#[wasm_bindgen]
impl WebCss2dRenderer {
    #[wasm_bindgen(constructor)]
    pub fn new(w: u32, h: u32) -> WebCss2dRenderer {
        WebCss2dRenderer {
            inner: crate::Css2dRenderer::new(w, h),
        }
    }
}

#[wasm_bindgen]
pub struct WebSvgRenderer {
    pub(super) inner: crate::SvgRenderer,
}
#[wasm_bindgen]
impl WebSvgRenderer {
    #[wasm_bindgen(constructor)]
    pub fn new(w: u32, h: u32) -> WebSvgRenderer {
        WebSvgRenderer {
            inner: crate::SvgRenderer::new(w, h),
        }
    }

    /// `0` lit (default), `1` flat material colour, `2` wireframe.
    #[wasm_bindgen(js_name = setShading)]
    pub fn set_shading(&mut self, mode: u32) {
        self.inner.options.shading = match mode {
            1 => crate::SvgShading::Flat,
            2 => crate::SvgShading::Wireframe,
            _ => crate::SvgShading::Lit,
        };
    }

    /// Same three.js constants as
    /// [`WebRenderer::setToneMapping`](WebRenderer::set_tone_mapping): `0`
    /// NoToneMapping, `1` Linear, `4` ACESFilmic.
    #[wasm_bindgen(js_name = setToneMapping)]
    pub fn set_tone_mapping(&mut self, mode: u32, exposure: f32) {
        // three.js's constants: 0 NoToneMapping, 1 Linear, 4 ACESFilmic,
        // 6 AgX, 7 Neutral. 2 (Reinhard) and 3 (Cineon) have no curve here and
        // fall through to linear, as they always have.
        let mapping = match mode {
            0 => crate::ToneMapping::None,
            4 => crate::ToneMapping::AcesFilmic,
            6 => crate::ToneMapping::AgX,
            7 => crate::ToneMapping::Neutral,
            _ => crate::ToneMapping::Linear,
        };
        self.inner.set_tone_mapping(mapping, exposure);
    }

    /// Emit `scene.background` as a full-canvas rect. Off gives a transparent
    /// document that sits on whatever is behind it in the page.
    #[wasm_bindgen(js_name = setBackground)]
    pub fn set_background(&mut self, on: bool) {
        self.inner.options.background = on;
    }

    #[wasm_bindgen(js_name = setCullBackfaces)]
    pub fn set_cull_backfaces(&mut self, on: bool) {
        self.inner.options.cull_backfaces = on;
    }

    /// Hairline stroke per face, in pixels, that hides the light seams SVG
    /// leaves along shared edges. 0 disables it.
    #[wasm_bindgen(js_name = setSeamStroke)]
    pub fn set_seam_stroke(&mut self, px: f32) {
        self.inner.options.seam_stroke = px.max(0.0);
    }

    /// Decimal places kept on coordinates. Lower is a smaller document.
    #[wasm_bindgen(js_name = setPrecision)]
    pub fn set_precision(&mut self, places: u32) {
        self.inner.options.precision = places.min(8) as usize;
    }

    /// Draw a face's edges as Bézier curves when a straight one would miss the
    /// real surface by more than this many pixels. Pass a value of 0 or less to
    /// keep every edge straight.
    #[wasm_bindgen(js_name = setCurveTolerance)]
    pub fn set_curve_tolerance(&mut self, px: f32) {
        self.inner.options.curve_tolerance = (px > 0.0).then_some(px);
    }

    /// Split faces spanning more than this fraction of their own distance from
    /// the camera, so the depth sort has something meaningful to order. 0 or
    /// less disables splitting.
    #[wasm_bindgen(js_name = setDepthSplit)]
    pub fn set_depth_split(&mut self, tolerance: f32) {
        self.inner.options.depth_split = (tolerance > 0.0).then_some(tolerance);
    }

    /// Depth-sort faces back to front. Off only makes sense when the caller has
    /// already ordered the scene.
    #[wasm_bindgen(js_name = setSort)]
    pub fn set_sort(&mut self, on: bool) {
        self.inner.options.sort = on;
    }

    /// Resize the canvas, keeping every option set so far.
    #[wasm_bindgen(js_name = setSize)]
    pub fn set_size(&mut self, width: u32, height: u32) {
        self.inner.set_size(width, height);
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.inner.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.inner.height
    }

    /// Background colour as `0xRRGGBB`, overriding `scene.background`. Mirrors
    /// three.js's `setClearColor`.
    #[wasm_bindgen(js_name = setClearColor)]
    pub fn set_clear_color(&mut self, hex: u32, alpha: f32) {
        self.inner.set_clear_color(
            Some(crate::Color::from_hex(hex)),
            Some(alpha.clamp(0.0, 1.0)),
        );
    }

    /// Go back to taking the background from the scene.
    #[wasm_bindgen(js_name = clearClearColor)]
    pub fn clear_clear_color(&mut self) {
        self.inner.set_clear_color(None, None);
    }

    #[wasm_bindgen(js_name = renderToString)]
    pub fn render_to_string(&self, scene: &mut WebScene, camera: &WebCamera) -> String {
        match &camera.inner {
            CameraInner::Perspective(c) => self.inner.render_to_string(&mut scene.inner, c),
            CameraInner::Orthographic(c) => self.inner.render_to_string(&mut scene.inner, c),
        }
    }
}

/// Wrap an RGBA frame — typically read back off a canvas — in an SVG document
/// as an embedded PNG. The counterpart to [`WebSvgRenderer`] for when the scene
/// needs the real renderer's textures, shadows or post-fx and a per-face
/// painter's algorithm will not do.
#[wasm_bindgen(js_name = svgFromRgba)]
pub fn svg_from_rgba(width: u32, height: u32, rgba: &[u8]) -> Result<String, JsValue> {
    let want = width as usize * height as usize * 4;
    if rgba.len() != want {
        return Err(JsValue::from_str(&format!(
            "svgFromRgba: expected {want} bytes for {width}x{height}, got {}",
            rgba.len()
        )));
    }
    Ok(crate::svg_from_rgba(width, height, rgba))
}
