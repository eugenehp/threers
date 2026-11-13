// How the object is turned for this pixel (set by `pfx_motion`): about the
// vertical (yaw), tipped (pitch), and in the screen's plane (roll).
var<private> pfx_pose: vec3<f32>;
// Whether it is drawn on a light background (the entry's `reach` given
// negative).
var<private> pfx_on_light: bool;

fn pfx_motion(p: vec2<f32>, size: vec2<f32>, now: f32, m: vec4<f32>, flat: bool) -> vec2<f32> {
    let t = now * m.y;
    let amt = m.z;
    let unit = m.w;
    var yaw = 0.0;
    var pitch = 0.0;
    var roll = 0.0;
    var off = vec2<f32>(0.0);
    var sc = vec2<f32>(1.0);
    let k = i32(m.x + 0.5);
    if (k == 1) {
        yaw = t * 2.4;
    } else if (k == 2) {
        yaw = sin(t * 2.2) * 0.65 * amt;
        pitch = sin(t * 1.1 + 1.0) * 0.18 * amt;
    } else if (k == 3) {
        roll = t * 2.6;
    } else if (k == 4) {
        // A hop, squashed where it lands and stretched as it leaves.
        let ph = fract(t * 1.25);
        let hop = 4.0 * ph * (1.0 - ph);
        let land = min(ph, 1.0 - ph) * 14.0;
        let s = 0.2 * amt * exp(-land * land);
        sc = vec2<f32>(1.0 + s, 1.0 - s);
        off = vec2<f32>(0.0, -hop * unit * amt + size.y * 0.5 * s);
    } else if (k == 5) {
        // Buzzing in bursts, as a phone does.
        let b = fract(t * 0.9);
        let burst = smoothstep(0.0, 0.03, b) * (1.0 - smoothstep(0.3, 0.36, b));
        off = vec2<f32>(sin(t * 97.0), cos(t * 83.0)) * unit * 0.12 * amt * burst;
        roll = sin(t * 71.0) * 0.05 * amt * burst;
    } else if (k == 6) {
        sc = vec2<f32>(1.0 + 0.07 * amt * sin(t * 3.4));
    } else if (k == 7) {
        off = vec2<f32>(0.0, sin(t * 1.7) * unit * 0.3 * amt);
        roll = sin(t * 1.2) * 0.06 * amt;
        pitch = sin(t * 1.4) * 0.15 * amt;
        yaw = sin(t * 0.9) * 0.2 * amt;
    } else if (k == 8) {
        pitch = t * 2.6;
    }
    pfx_pose = vec3<f32>(yaw, pitch, roll);
    let c = size * 0.5;
    var q = (p - c - off) / sc;
    if (flat) {
        let cr = cos(roll);
        let sr = sin(roll);
        q = vec2<f32>(cr * q.x + sr * q.y, -sr * q.x + cr * q.y);
        let cy = cos(yaw);
        let cp = cos(pitch);
        q = vec2<f32>(q.x / select(min(cy, -0.02), max(cy, 0.02), cy >= 0.0), q.y / select(min(cp, -0.02), max(cp, 0.02), cp >= 0.0));
    }
    return q + c;
}
