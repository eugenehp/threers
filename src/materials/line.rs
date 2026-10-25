use crate::math::Color;

/// Flat-color line material. Mirrors three.js's `LineBasicMaterial`.
#[derive(Debug, Clone, Copy)]
pub struct LineBasicMaterial {
    pub color: Color,
    pub opacity: f32,
    /// three.js `alphaTest`: fragments whose alpha (opacity × map alpha) is
    /// below this are discarded. 0 (the default) keeps every fragment.
    pub alpha_test: f32,
    /// three.js `transparent`. `Some` is three.js's flag as given — `false`
    /// draws opaquely whatever the opacity. `None` (the default) blends when
    /// `opacity < 1`, the rule this crate used before it had the flag.
    pub transparent: Option<bool>,
    /// three.js `linewidth`, in pixels (× the renderer's pixel ratio, as a
    /// point's size is). 1 draws line primitives; wider, each segment is a
    /// round-capped quad (`vs_wide_lines`) — what three.js's `linewidth`
    /// never did under WebGL. Thinner and transparent, the quad too, each
    /// pixel taking the share of it the line covers, so a 0.3-pixel line
    /// draws at 0.3 of its colour instead of lighting whole pixels or none
    /// (opaque, it stays a line primitive). On the WebGPU/WebGL2 renderer; the Metal
    /// renderer has its own `set_line_width`.
    pub line_width: f32,
    pub dashed: bool,
    pub dash_scale: f32,
    pub dash_size: f32,
    pub gap_size: f32,
    /// three.js `toneMapped` (default `true`): whether the renderer's tone
    /// mapping applies to this material's output.
    pub tone_mapped: bool,
    /// three.js `depthTest` / `depthWrite` / `depthFunc` / `colorWrite`.
    /// `depth_test: false` draws the line over the depth buffer (an x-ray
    /// or overlay stroke).
    pub render_state: super::RenderState,
}

impl Default for LineBasicMaterial {
    fn default() -> Self {
        Self {
            color: Color::WHITE,
            opacity: 1.0,
            alpha_test: 0.0,
            transparent: None,
            line_width: 1.0,
            dashed: false,
            dash_scale: 1.0,
            dash_size: 0.0,
            gap_size: 0.0,
            tone_mapped: true,
            render_state: super::RenderState::default(),
        }
    }
}

impl LineBasicMaterial {
    pub fn new(color: Color) -> Self {
        Self {
            color,
            ..Default::default()
        }
    }

    /// Draw through occluders (active highlights, HUD strokes): three.js
    /// `depthTest`. Draws in order like any other, so an overlay wants a
    /// later `render_order` than what it overlays.
    pub fn with_depth_test(mut self, enabled: bool) -> Self {
        self.render_state.depth_test = enabled;
        self
    }
}
