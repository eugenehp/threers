use crate::math::Color;
use crate::textures::Texture;
use std::sync::Arc;

/// Point material. Mirrors three.js's `PointsMaterial`. `size_attenuation`
/// scales size by 1/depth in perspective; ignored under orthographic.
#[derive(Debug, Clone)]
pub struct PointsMaterial {
    pub color: Color,
    pub opacity: f32,
    /// three.js `alphaTest`: fragments whose alpha (opacity × map alpha) is
    /// below this are discarded. 0 (the default) keeps every fragment.
    pub alpha_test: f32,
    /// three.js `transparent`. `Some` is three.js's flag as given — `false`
    /// draws opaquely whatever the opacity. `None` (the default) blends when
    /// `opacity < 1`, the rule this crate used before it had the flag.
    pub transparent: Option<bool>,
    pub size: f32,
    pub size_attenuation: bool,
    /// three.js `map`, sampled at each point's coordinate across its quad.
    pub map: Option<Arc<Texture>>,
    /// three.js `alphaMap` (green channel), likewise.
    pub alpha_map: Option<Arc<Texture>>,
    /// three.js `toneMapped` (default `true`): whether the renderer's tone
    /// mapping applies to this material's output.
    pub tone_mapped: bool,
    /// three.js `depthTest` / `depthWrite` / `depthFunc` / `colorWrite`.
    pub render_state: super::RenderState,
}

impl Default for PointsMaterial {
    fn default() -> Self {
        Self {
            color: Color::WHITE,
            opacity: 1.0,
            alpha_test: 0.0,
            transparent: None,
            size: 1.0,
            size_attenuation: true,
            map: None,
            alpha_map: None,
            tone_mapped: true,
            render_state: super::RenderState::default(),
        }
    }
}

impl PointsMaterial {
    pub fn new(color: Color, size: f32) -> Self {
        Self {
            color,
            size,
            ..Default::default()
        }
    }
}
