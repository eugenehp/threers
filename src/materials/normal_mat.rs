/// Debug material that displays world-space normals as colors.
/// Matches three.js's `MeshNormalMaterial`.
#[derive(Debug, Clone, Copy, Default)]
pub struct NormalMaterial {
    pub opacity: f32,
    /// three.js `alphaTest`: fragments whose alpha (opacity × map alpha) is
    /// below this are discarded. 0 (the default) keeps every fragment.
    pub alpha_test: f32,
    /// three.js `transparent`. `Some` is three.js's flag as given — `false`
    /// draws opaquely whatever the opacity. `None` (the default) blends when
    /// `opacity < 1`, the rule this crate used before it had the flag.
    pub transparent: Option<bool>,
    pub wireframe: bool,
    /// three.js `polygonOffset` / `polygonOffsetFactor` / `polygonOffsetUnits`:
    /// `Some` enables the depth offset (see [`PolygonOffset`](super::PolygonOffset)).
    pub polygon_offset: Option<super::PolygonOffset>,
    /// three.js `depthTest` / `depthWrite` / `depthFunc` / `colorWrite`.
    pub render_state: super::RenderState,
}

impl NormalMaterial {
    pub const fn new() -> Self {
        Self {
            opacity: 1.0,
            alpha_test: 0.0,
            transparent: None,
            wireframe: false,
            polygon_offset: None,
            render_state: super::RenderState::DEFAULT,
        }
    }
}
