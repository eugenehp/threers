use crate::math::Color;
use crate::textures::Texture;
use std::sync::Arc;

/// Sprite billboard material. Mirrors three.js's `SpriteMaterial`.
#[derive(Debug, Clone)]
pub struct SpriteMaterial {
    pub color: Color,
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
    /// Turn of the quad in the view plane, radians, counter-clockwise.
    pub rotation: f32,
    pub map: Option<Arc<Texture>>,
    /// three.js `sizeAttenuation`. `true` (the default): the quad is
    /// world-sized and shrinks with distance. `false`: under a perspective
    /// camera the quad is scaled by its view depth, as three.js's
    /// `sprite_vert` does, so it keeps one size on screen. No effect under an
    /// orthographic camera.
    pub size_attenuation: bool,
    /// three.js `toneMapped` (default `true`): whether the renderer's tone
    /// mapping applies to this material's output.
    pub tone_mapped: bool,
    /// three.js `depthTest` / `depthWrite` / `depthFunc` / `colorWrite`.
    pub render_state: super::RenderState,
}

impl Default for SpriteMaterial {
    fn default() -> Self {
        Self {
            color: Color::WHITE,
            opacity: 1.0,
            alpha_test: 0.0,
            alpha_map: None,
            transparent: None,
            rotation: 0.0,
            map: None,
            size_attenuation: true,
            tone_mapped: true,
            render_state: super::RenderState::default(),
        }
    }
}

impl SpriteMaterial {
    pub fn new(color: Color) -> Self {
        Self {
            color,
            ..Default::default()
        }
    }

    pub fn with_map(mut self, map: Arc<Texture>) -> Self {
        self.map = Some(map);
        self
    }
}
