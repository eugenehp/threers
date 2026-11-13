// The object in the view (x right, y up, z toward the eye): tipped back by
// `tilt` and the pose's pitch, turned by its yaw, rolled.
fn pfx_orient(tilt: f32) -> mat3x3<f32> {
    let a = tilt + pfx_pose.y;
    let b = pfx_pose.x;
    let g = pfx_pose.z;
    let rx = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, cos(a), -sin(a)), vec3<f32>(0.0, sin(a), cos(a)));
    let ry = mat3x3<f32>(vec3<f32>(cos(b), 0.0, -sin(b)), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(sin(b), 0.0, cos(b)));
    let rz = mat3x3<f32>(vec3<f32>(cos(g), -sin(g), 0.0), vec3<f32>(sin(g), cos(g), 0.0), vec3<f32>(0.0, 0.0, 1.0));
    return rz * ry * rx;
}

// Glass at a surface with normal `n`, met by the ray `rd` (both in the
// view), over `behind` (premultiplied light seen through it): Fresnel
// reflection of the studio and two speculars on top, the rest let
// through, tinted by `tint` more where the glass is seen edge on (more of
// it to look through). Premultiplied.
fn pfx_glass(n: vec3<f32>, rd: vec3<f32>, behind: vec4<f32>, tint: vec3<f32>) -> vec4<f32> {
    let ndv = clamp(-dot(n, rd), 0.0, 1.0);
    let f = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
    let sun = normalize(vec3<f32>(-0.4, 0.8, 0.45));
    let nh = max(dot(n, normalize(sun - rd)), 0.0);
    let spec = pow(nh, 220.0) * 3.0 + pow(nh, 28.0) * 0.1;
    let reflected = studio_env(reflect(rd, n)) * f + vec3<f32>(spec);
    let edge = pow(1.0 - ndv, 2.0);
    let own = 0.05 + 0.3 * edge;
    let through = behind.rgb * mix(vec3<f32>(1.0), tint, 0.25 + 0.5 * edge) * (1.0 - f) * (1.0 - own);
    let a = clamp(f + own + min(spec, 1.0) + behind.a * (1.0 - f) * (1.0 - own), 0.0, 1.0);
    return vec4<f32>(reflected + tint * own * 0.35 + through, a);
}

// The nearest point of the segment from `a` to `b` to `q`.
fn pfx_on_segment(q: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    return a + ab * clamp(dot(q - a, ab) / max(dot(ab, ab), 0.000001), 0.0, 1.0);
}

// How near the ray (`ro`, unit `rd`) comes to the segment from `a` to `b`:
// (the distance, how far along the ray, how far along the segment 0..1).
fn pfx_ray_segment(ro: vec3<f32>, rd: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let ba = b - a;
    let oa = ro - a;
    let rb = dot(rd, ba);
    let bb = dot(ba, ba);
    let den = bb - rb * rb;
    let s = select(0.0, clamp((dot(oa, ba) - rb * dot(oa, rd)) / max(den, 0.000001), 0.0, 1.0), den > 0.000001);
    let q = a + ba * s;
    let t = dot(q - ro, rd);
    return vec3<f32>(length(q - ro - rd * t), t, s);
}

// Where the ray first meets the capsule from `a` to `b` of radius `r`, or
// -1 (Inigo Quilez's intersection).
fn pfx_capsule_t(ro: vec3<f32>, rd: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let ba = b - a;
    let oa = ro - a;
    let baba = dot(ba, ba);
    let bard = dot(ba, rd);
    let baoa = dot(ba, oa);
    let rdoa = dot(rd, oa);
    let oaoa = dot(oa, oa);
    let qa = baba - bard * bard;
    if (qa > 0.000001) {
        let qb = baba * rdoa - baoa * bard;
        let qc = baba * oaoa - baoa * baoa - r * r * baba;
        let h = qb * qb - qa * qc;
        if (h >= 0.0) {
            let t = (-qb - sqrt(h)) / qa;
            let y = baoa + t * bard;
            if (y > 0.0 && y < baba) {
                return t;
            }
        }
    }
    // A cap (or a capsule seen end on).
    var best = -1.0;
    for (var i = 0; i < 2; i = i + 1) {
        let oc = select(ro - b, oa, i == 0);
        let b2 = dot(rd, oc);
        let c2 = dot(oc, oc) - r * r;
        let h2 = b2 * b2 - c2;
        if (h2 > 0.0) {
            let t = -b2 - sqrt(h2);
            if (t > 0.0 && (best < 0.0 || t < best)) {
                best = t;
            }
        }
    }
    return best;
}
