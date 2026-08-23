/// Shadow-map configuration carried by lights that can cast shadows.
/// Mirrors three.js's `LightShadow` for the parts the renderer reads.
///
/// These DO drive pixels: the renderer runs a depth pre-pass per casting light
/// and samples it with a 3x3 PCF kernel. (An earlier version of this note said
/// otherwise, long after it stopped being true, which is worth more scepticism
/// than a comment usually earns.)
#[derive(Debug, Clone, Copy)]
pub struct ShadowSettings {
    /// Currently advisory: the directional map is a fixed 4096 built once at
    /// renderer construction, before any light exists to ask for a size.
    /// Resolution is set by `camera_size` instead — halve the frustum and every
    /// texel covers half the distance.
    pub map_size: u32,
    pub bias: f32,
    pub normal_bias: f32,
    /// Orthographic frustum size (in world units) around the light target.
    pub camera_size: f32,
    pub camera_near: f32,
    pub camera_far: f32,
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self {
            map_size: 4096,
            bias: 0.0,
            normal_bias: 0.0,
            camera_size: 5.0,
            camera_near: 0.1,
            camera_far: 500.0,
        }
    }
}
