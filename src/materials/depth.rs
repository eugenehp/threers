/// Debug material that displays linearized depth as grayscale.
/// Matches three.js's `MeshDepthMaterial`.
#[derive(Debug, Clone, Copy)]
pub struct DepthMaterial {
    pub opacity: f32,
    /// three.js `alphaTest`: fragments whose alpha (opacity × map alpha) is
    /// below this are discarded. 0 (the default) keeps every fragment.
    pub alpha_test: f32,
    /// three.js `transparent`. `Some` is three.js's flag as given — `false`
    /// draws opaquely whatever the opacity. `None` (the default) blends when
    /// `opacity < 1`, the rule this crate used before it had the flag.
    pub transparent: Option<bool>,
    pub near: f32,
    pub far: f32,
    pub wireframe: bool,
    /// three.js `polygonOffset` / `polygonOffsetFactor` / `polygonOffsetUnits`:
    /// `Some` enables the depth offset (see [`PolygonOffset`](super::PolygonOffset)).
    pub polygon_offset: Option<super::PolygonOffset>,
    /// three.js `depthTest` / `depthWrite` / `depthFunc` / `colorWrite`.
    pub render_state: super::RenderState,
}

impl Default for DepthMaterial {
    fn default() -> Self {
        Self {
            opacity: 1.0,
            alpha_test: 0.0,
            transparent: None,
            near: 0.1,
            far: 100.0,
            wireframe: false,
            polygon_offset: None,
            render_state: super::RenderState::DEFAULT,
        }
    }
}

impl DepthMaterial {
    pub fn new() -> Self {
        Self::default()
    }
}
