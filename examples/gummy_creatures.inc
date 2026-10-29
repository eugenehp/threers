// The gummy creatures of `web/examples/creatures.js`, from the same shapes:
// the jellyfish and the octopus. Shared, by `include!`, between
// `examples/gummy.rs`, `examples/softbody_bench.rs` and `tests/softbody_gpu.rs`.
//
// Needs in scope: `PI` (f32), `Vector3`, `Shape`, `ChainSpec`, `Creature`,
// `Marks`, `ShapeSpec`.

fn v(x: f32, y: f32, z: f32) -> Vector3 {
    Vector3::new(x, y, z)
}
fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
fn tube(pts: &[Vector3], radii: &[f32], k: f32) -> Shape {
    let parts = pts.windows(2).zip(radii.windows(2)).map(|(p, r)| Shape::tapered_capsule(p[0], p[1], r[0], r[1])).collect();
    Shape::union_all(parts, k)
}

const ORD: [&str; 8] = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"];

pub fn jellyfish() -> Creature {
    let (r, h, y0) = (0.62f32, 0.46f32, 0.04f32);
    let mut bell = Shape::ellipsoid(v(r, h, r))
        .translate(v(0.0, y0, 0.0))
        .intersect(Shape::plane(v(0.0, -1.0, 0.0), y0), 0.0)
        .subtract(Shape::ellipsoid(v(0.48, 0.25, 0.48)).translate(v(0.0, y0, 0.0)), 0.03);
    for k in 0..16 {
        let a = (k as f32 + 0.5) / 16.0 * PI * 2.0;
        bell = bell.subtract(Shape::sphere(0.055).translate(v((r + 0.03) * a.cos(), y0, (r + 0.03) * a.sin())), 0.02);
    }
    let parts = (0..8)
        .map(|i| {
            let a = (i as f32 + 0.5) / 8.0 * PI * 2.0;
            let (c, s, curl) = (a.cos(), a.sin(), if i % 2 == 1 { 1.0 } else { -1.0 });
            let len = 0.62 + 0.08 * ((i * 5) % 3) as f32;
            let pts: Vec<Vector3> = (0..=7)
                .map(|k| {
                    let t = k as f32 / 7.0;
                    let (out, side) = (r - 0.07 + len * t, curl * 0.2 * (t * 2.4).sin() * t);
                    v(c * out - s * side, mix(0.075, 0.026, smoothstep(0.0, 0.35, t)), s * out + c * side)
                })
                .collect();
            let radii: Vec<f32> = (0..=7).map(|k| mix(0.042, 0.013, k as f32 / 7.0)).collect();
            (format!("{} tentacle", ORD[i]), ShapeSpec { shape: tube(&pts, &radii, 0.012), cell: 0.0095, h: 0.04 })
        })
        .collect::<Vec<_>>();
    let chains = (0..16)
        .map(|j| {
            let a = (j as f64 + 0.5 * if j % 2 == 1 { 0.6 } else { 1.4 }) / 16.0 * std::f64::consts::TAU + std::f64::consts::PI / 8.0;
            let (c, s, curl) = (a.cos(), a.sin(), (j % 3) as f64 - 1.0);
            let len = 0.42 + 0.12 * ((j * 7) % 4) as f64 / 3.0;
            let points = (0..10)
                .map(|k| {
                    let t = k as f64 / 9.0;
                    let (out, side) = (r as f64 - 0.02 + len * t, 0.12 * curl * t * t);
                    [c * out - s * side, 0.06 + (0.008 - 0.06) * smoothstep(0.0, 0.3, t as f32) as f64, s * out + c * side]
                })
                .collect();
            ChainSpec { points, r0: 0.009, r1: 0.0035, stiff: 0.16 }
        })
        .collect::<Vec<_>>();
    let top = (y0 + h - 0.08) as f64;
    let marks = Marks {
        head: [[0.0, top, 0.0], [0.25, top - 0.06, 0.0], [0.0, top - 0.06, 0.25]],
        span: [[r as f64, y0 as f64 + 0.03, 0.0], [-r as f64, y0 as f64 + 0.03, 0.0]],
    };
    Creature::from_shapes(&ShapeSpec { shape: bell, cell: 0.016, h: 0.07 }, &parts, &chains, marks).expect("jellyfish")
}

pub fn octopus() -> Creature {
    let body = Shape::union_all(
        vec![Shape::ellipsoid(v(0.27, 0.2, 0.26)).translate(v(0.0, 0.2, 0.0)), Shape::ellipsoid(v(0.25, 0.26, 0.23)).translate(v(-0.1, 0.42, 0.0))],
        0.12,
    )
    .union(Shape::sphere(0.07).translate(v(0.17, 0.3, 0.16)), 0.04)
    .union(Shape::sphere(0.07).translate(v(0.17, 0.3, -0.16)), 0.04);
    let parts = (0..8)
        .map(|i| {
            let a = (i as f32 + 0.5) / 8.0 * PI * 2.0;
            let curl = if i % 2 == 1 { 1.0 } else { -1.0 } * (0.25 + 0.1 * (i % 3) as f32);
            let (mut pts, mut radii) = (Vec::new(), Vec::new());
            for k in 0..=9 {
                let t = k as f32 / 9.0;
                let (out, ang, rr) = (0.13 + 0.85 * t, a + curl * t * t, mix(0.075, 0.014, t));
                let tip = smoothstep(0.7, 1.0, t) * 0.1 * curl.signum();
                pts.push(v(ang.cos() * out - ang.sin() * tip, if k == 0 { 0.12 } else { rr + 0.004 }, ang.sin() * out + ang.cos() * tip));
                radii.push(rr);
            }
            (format!("{} arm", ORD[i]), ShapeSpec { shape: tube(&pts, &radii, 0.02), cell: 0.0095, h: 0.038 })
        })
        .collect::<Vec<_>>();
    let marks = Marks { head: [[0.0, 0.35, 0.0], [0.2, 0.3, 0.0], [0.0, 0.3, 0.2]], span: [[0.26, 0.2, 0.0], [-0.34, 0.42, 0.0]] };
    Creature::from_shapes(&ShapeSpec { shape: body, cell: 0.015, h: 0.065 }, &parts, &[], marks).expect("octopus")
}
