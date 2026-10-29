/// One frame's worth of pointer input. The caller is responsible for
/// translating its platform's events into this shape and calling
/// `OrbitControls::update`.
#[derive(Debug, Clone, Copy, Default)]
pub struct PointerEvent {
    /// Cumulative pointer delta in pixels.
    pub dx: f32,
    pub dy: f32,
    /// Scroll delta (positive = zoom in / scroll up).
    pub wheel: f32,
    /// True while the rotate button is held (left mouse).
    pub rotating: bool,
    /// True while the pan button is held (right mouse or shift+left).
    pub panning: bool,
    /// Pointer position in NDC (−1..1, +y up), for dolly-to-cursor. Read only
    /// when `wheel != 0` and the controls have `zoom_to_cursor` on.
    pub cursor: (f32, f32),
}
