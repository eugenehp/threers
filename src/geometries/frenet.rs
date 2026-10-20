//! three.js r165 `Curve.computeFrenetFrames`, computed in `f64` as JavaScript
//! does, from tangents the caller has already sampled.
//!
//! Taking tangents rather than a curve is what lets the browser side hand over
//! a curve defined in JavaScript — a `Curve` subclass with its own `getPoint` —
//! by sampling its `getTangentAt` there, while the frame construction itself
//! runs here. [`TubeGeometry`](super::TubeGeometry) and
//! [`ExtrudeGeometry`](super::ExtrudeGeometry)'s `extrudePath` both use it.

/// Frenet frames along a curve: one tangent, normal and binormal per sample.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FrenetFrames {
    pub tangents: Vec<[f64; 3]>,
    pub normals: Vec<[f64; 3]>,
    pub binormals: Vec<[f64; 3]>,
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn length(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// three.js `Vector3.normalize`: `divideScalar(length || 1)`, which is a
/// multiply by the reciprocal — not a division, which can round differently.
fn normalize(a: [f64; 3]) -> [f64; 3] {
    let l = length(a);
    let inv = 1.0 / if l == 0.0 { 1.0 } else { l };
    [a[0] * inv, a[1] * inv, a[2] * inv]
}

/// `v.applyMatrix4(new Matrix4().makeRotationAxis(axis, angle))`, operation
/// for operation.
fn rotate(v: [f64; 3], axis: [f64; 3], angle: f64) -> [f64; 3] {
    let c = angle.cos();
    let s = angle.sin();
    let t = 1.0 - c;
    let [x, y, z] = axis;
    let tx = t * x;
    let ty = t * y;
    // Row-major, as `Matrix4.set` takes it.
    let m = [
        [tx * x + c, tx * y - s * z, tx * z + s * y],
        [tx * y + s * z, ty * y + c, ty * z - s * x],
        [tx * z - s * y, ty * z + s * x, t * z * z + c],
    ];
    // `applyMatrix4` divides by w, which is exactly 1 for a rotation.
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// `curve.computeFrenetFrames(segments, closed)` given
/// `tangents[i] = curve.getTangentAt(i / segments)` for `i` in `0..=segments`.
pub fn compute_frenet_frames(tangents: &[[f64; 3]], closed: bool) -> FrenetFrames {
    let n = tangents.len();
    if n == 0 {
        return FrenetFrames::default();
    }
    let segments = n - 1;
    let t0 = tangents[0];
    let mut min = f64::MAX;
    let mut normal = [0.0; 3];
    let (tx, ty, tz) = (t0[0].abs(), t0[1].abs(), t0[2].abs());
    if tx <= min {
        min = tx;
        normal = [1.0, 0.0, 0.0];
    }
    if ty <= min {
        min = ty;
        normal = [0.0, 1.0, 0.0];
    }
    if tz <= min {
        normal = [0.0, 0.0, 1.0];
    }
    let vec = normalize(cross(t0, normal));
    let mut normals = Vec::with_capacity(n);
    let mut binormals = Vec::with_capacity(n);
    normals.push(cross(t0, vec));
    binormals.push(cross(t0, normals[0]));

    for i in 1..n {
        let mut ni = normals[i - 1];
        let mut vec = cross(tangents[i - 1], tangents[i]);
        if length(vec) > f64::EPSILON {
            vec = normalize(vec);
            let theta = dot(tangents[i - 1], tangents[i]).clamp(-1.0, 1.0).acos();
            ni = rotate(ni, vec, theta);
        }
        normals.push(ni);
        binormals.push(cross(tangents[i], ni));
    }

    if closed && segments > 0 {
        let mut theta = dot(normals[0], normals[segments]).clamp(-1.0, 1.0).acos();
        theta /= segments as f64;
        if dot(tangents[0], cross(normals[0], normals[segments])) > 0.0 {
            theta = -theta;
        }
        for i in 1..=segments {
            normals[i] = rotate(normals[i], tangents[i], theta * i as f64);
            binormals[i] = cross(tangents[i], normals[i]);
        }
    }

    FrenetFrames {
        tangents: tangents.to_vec(),
        normals,
        binormals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_line_keeps_one_frame() {
        let t = vec![[1.0, 0.0, 0.0]; 4];
        let f = compute_frenet_frames(&t, false);
        // y and z tie for the smallest component and the last wins, so the
        // seed is z: vec = t × z = (0, -1, 0), normal = t × vec = (0, 0, -1).
        for n in &f.normals {
            assert_eq!(*n, [0.0, 0.0, -1.0]);
        }
        for b in &f.binormals {
            assert_eq!(*b, [0.0, 1.0, 0.0]);
        }
    }
}
