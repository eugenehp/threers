use crate::math::Color;
use crate::textures::Texture;
use std::sync::Arc;

/// MatCap (material capture) sampling: viewspace normal → UV into a matcap
/// texture. Matches three.js's `MeshMatcapMaterial`.
#[derive(Debug, Clone)]
pub struct MatcapMaterial {
    pub color: Color,
    pub matcap: Option<Arc<Texture>>,
    /// three.js `map`: multiplies the diffuse colour before the matcap
    /// (`meshmatcap.glsl`'s `map_fragment`).
    pub map: Option<Arc<Texture>>,
    pub opacity: f32,
    /// three.js `alphaTest`: fragments whose alpha (opacity × map alpha) is
    /// below this are discarded. 0 (the default) keeps every fragment.
    pub alpha_test: f32,
    /// three.js `alphaMap`: its green channel multiplies the opacity, before
    /// `alphaTest` (`alphamap_fragment`). Shows only with `transparent` or
    /// `alpha_test`, as in three.js.
    pub alpha_map: Option<Arc<Texture>>,
    /// three.js `transparent`. `Some` is three.js's flag as given — `false`
    /// draws opaquely whatever the opacity. `None` (the default) blends when
    /// `opacity < 1`, the rule this crate used before it had the flag.
    pub transparent: Option<bool>,
    pub wireframe: bool,
    /// three.js `polygonOffset` / `polygonOffsetFactor` / `polygonOffsetUnits`:
    /// `Some` enables the depth offset (see [`PolygonOffset`](super::PolygonOffset)).
    pub polygon_offset: Option<super::PolygonOffset>,
    /// three.js `toneMapped` (default `true`): whether the renderer's tone
    /// mapping applies to this material's output.
    pub tone_mapped: bool,
    /// three.js `depthTest` / `depthWrite` / `depthFunc` / `colorWrite`.
    pub render_state: super::RenderState,
}

impl Default for MatcapMaterial {
    fn default() -> Self {
        Self {
            color: Color::WHITE,
            matcap: None,
            map: None,
            opacity: 1.0,
            alpha_test: 0.0,
            alpha_map: None,
            transparent: None,
            wireframe: false,
            polygon_offset: None,
            tone_mapped: true,
            render_state: super::RenderState::default(),
        }
    }
}

impl MatcapMaterial {
    pub fn new(matcap: Arc<Texture>) -> Self {
        Self {
            matcap: Some(matcap),
            map: None,
            ..Default::default()
        }
    }
}
