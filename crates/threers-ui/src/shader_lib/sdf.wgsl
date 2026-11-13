// Shapes by their distance (threers_ui::shader_lib::SDF): negative inside.

// How much of a pixel `aa` across the shape `d` away covers.
fn ui_cover(d: f32, aa: f32) -> f32 {
    return clamp(0.5 - d / aa, 0.0, 1.0);
}

// The distance from `p` to the segment from `a` to `b` (a capsule: minus
// its radius).
fn ui_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let h = clamp(dot(p - a, ab) / max(dot(ab, ab), 0.000001), 0.0, 1.0);
    return length(p - a - ab * h);
}

// A box round `c`, `half` its size each way, its corners rounded `r`.
fn ui_rbox(p: vec2<f32>, c: vec2<f32>, half: vec2<f32>, r: f32) -> f32 {
    let rr = min(r, min(half.x, half.y));
    let q = abs(p - c) - (half - vec2<f32>(rr));
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - rr;
}

// A ring round `c`, `radius` to the middle of a stroke `width` wide.
fn ui_ring(p: vec2<f32>, c: vec2<f32>, radius: f32, width: f32) -> f32 {
    return abs(length(p - c) - radius) - width * 0.5;
}
