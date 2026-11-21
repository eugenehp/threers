//! A 3D model shown in the UI and turned with the pointer, a finger or the
//! keys: [`model_view`] draws a [`Mesh3`] into an image node with a small
//! software renderer (a depth buffer, faces lit by the UI's
//! [`Light`](crate::depth::Light)), so it shows with any painter, and turns
//! it by an [`Orbit`] kept in the state. A renderer of its own can stand in
//! for the drawing (the threers renderer's `ModelView` does) with the same
//! orbit.
//!
//! ```
//! use threers_ui::input::Input;
//! use threers_ui::model3d::{model_view, Mesh3, Orbit};
//! use threers_ui::{binding, FixedWidthText, Ui};
//!
//! #[derive(Default)]
//! struct S { orbit: Orbit }
//!
//! let mut ui = Ui::new();
//! let mut input: Input<S> = Input::new();
//! let mut s = S::default();
//! let root = ui.root();
//! let view = model_view(&mut ui, &mut input, &s, root, "A cube", Mesh3::cube(1.0, [0.2, 0.5, 0.9]), binding!(|s: S| s.orbit), (120, 90));
//! ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
//! input.set_focus(&mut ui, &mut s, Some(view));
//! input.simulate_keystrokes(&mut ui, &mut s, "right right").unwrap();
//! assert!(s.orbit.yaw > 0.1);
//! ```

use std::rc::Rc;

use crate::binding::Binding;
use crate::image::Image;
use crate::input::{Event, Input, Phase};
use crate::tree::{NodeId, Ui};
use crate::{style, Length, Role, Semantics};

/// Where a camera looks at a model from: turned `yaw` about the vertical,
/// tipped `pitch` (radians, positive from above), `distance` away.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Orbit {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

impl Default for Orbit {
    fn default() -> Self {
        Self {
            yaw: 0.6,
            pitch: 0.35,
            distance: 4.0,
        }
    }
}

impl Orbit {
    /// The camera's place, looking at the origin.
    pub fn eye(&self) -> [f32; 3] {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        [
            self.distance * cp * sy,
            self.distance * sp,
            self.distance * cp * cy,
        ]
    }

    /// Turned by `dyaw` and `dpitch` (pitch kept short of straight up or
    /// down).
    pub fn turned(self, dyaw: f32, dpitch: f32) -> Self {
        Self {
            yaw: self.yaw + dyaw,
            pitch: (self.pitch + dpitch).clamp(-1.45, 1.45),
            ..self
        }
    }

    /// Nearer by `factor` (below 1) or further (above), within reason.
    pub fn zoomed(self, factor: f32) -> Self {
        Self {
            distance: (self.distance * factor).clamp(1.2, 30.0),
            ..self
        }
    }
}

/// A model of triangles, each its own colour (linear RGB, 0 to 1).
#[derive(Debug, Clone, Default)]
pub struct Mesh3 {
    pub positions: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,
    pub colors: Vec<[f32; 3]>,
}

impl Mesh3 {
    /// A cube `size` across, centred on the origin.
    pub fn cube(size: f32, color: [f32; 3]) -> Self {
        let h = size / 2.0;
        let p = |x: f32, y: f32, z: f32| [x * h, y * h, z * h];
        let positions = vec![
            p(-1., -1., -1.),
            p(1., -1., -1.),
            p(1., 1., -1.),
            p(-1., 1., -1.),
            p(-1., -1., 1.),
            p(1., -1., 1.),
            p(1., 1., 1.),
            p(-1., 1., 1.),
        ];
        let quads = [
            [4, 5, 6, 7],
            [1, 0, 3, 2],
            [5, 1, 2, 6],
            [0, 4, 7, 3],
            [7, 6, 2, 3],
            [0, 1, 5, 4],
        ];
        let mut triangles = vec![];
        for q in quads {
            triangles.push([q[0], q[1], q[2]]);
            triangles.push([q[0], q[2], q[3]]);
        }
        let colors = vec![color; triangles.len()];
        Self {
            positions,
            triangles,
            colors,
        }
    }

    /// A torus around the vertical axis: `radius` to its tube's middle,
    /// `tube` thick, `segments` round and `sides` round its tube.
    pub fn torus(radius: f32, tube: f32, segments: u32, sides: u32, color: [f32; 3]) -> Self {
        let (n, m) = (segments.max(3), sides.max(3));
        let mut positions = vec![];
        for i in 0..n {
            let u = i as f32 / n as f32 * std::f32::consts::TAU;
            for j in 0..m {
                let v = j as f32 / m as f32 * std::f32::consts::TAU;
                let r = radius + tube * v.cos();
                positions.push([r * u.cos(), tube * v.sin(), r * u.sin()]);
            }
        }
        let at = |i: u32, j: u32| (i % n) * m + (j % m);
        let mut triangles = vec![];
        for i in 0..n {
            for j in 0..m {
                let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
                triangles.push([a, c, b]);
                triangles.push([a, d, c]);
            }
        }
        let colors = vec![color; triangles.len()];
        Self {
            positions,
            triangles,
            colors,
        }
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: [f32; 3]) -> [f32; 3] {
    let l = dot(a, a).sqrt().max(1e-9);
    a.map(|c| c / l)
}

/// `mesh` seen from `orbit`, `width × height` pixels, lit by `light` (a
/// unit vector toward it, in the camera's space: x right, y down, z toward
/// the viewer): RGBA8, sRGB, transparent where it is not. One sample a
/// pixel, at its centre; [`render_smooth`] for smooth edges.
pub fn render(mesh: &Mesh3, orbit: &Orbit, width: u32, height: u32, light: [f32; 3]) -> Vec<u8> {
    render_smooth(mesh, orbit, width, height, light, 1)
}

/// Where in a pixel its samples are, from its centre: the usual patterns
/// for 1, 4 and 8 samples (rotated, so near-vertical and near-horizontal
/// edges get as many steps as there are samples).
fn sample_pattern(samples: u32) -> &'static [(f32, f32)] {
    const ONE: [(f32, f32); 1] = [(0.0, 0.0)];
    const FOUR: [(f32, f32); 4] = [
        (-2.0 / 16.0, -6.0 / 16.0),
        (6.0 / 16.0, -2.0 / 16.0),
        (-6.0 / 16.0, 2.0 / 16.0),
        (2.0 / 16.0, 6.0 / 16.0),
    ];
    const EIGHT: [(f32, f32); 8] = [
        (1.0 / 16.0, -3.0 / 16.0),
        (-1.0 / 16.0, 3.0 / 16.0),
        (5.0 / 16.0, 1.0 / 16.0),
        (-3.0 / 16.0, -5.0 / 16.0),
        (-5.0 / 16.0, 5.0 / 16.0),
        (-7.0 / 16.0, -1.0 / 16.0),
        (3.0 / 16.0, 7.0 / 16.0),
        (7.0 / 16.0, -7.0 / 16.0),
    ];
    match samples {
        0 | 1 => &ONE,
        2..=4 => &FOUR,
        _ => &EIGHT,
    }
}

/// [`render`] with smooth edges: `samples` (1, 4 or 8) depth-tested points
/// in each pixel, averaged as light. As a GPU multisamples, a face is
/// shaded once and only its edges are tested sample by sample: a pixel
/// wholly inside a face costs one test, so smooth edges cost little more
/// than rough ones.
pub fn render_smooth(
    mesh: &Mesh3,
    orbit: &Orbit,
    width: u32,
    height: u32,
    light: [f32; 3],
    samples: u32,
) -> Vec<u8> {
    render_rows(mesh, orbit, width, height, light, samples, 0..height)
}

/// One face ready to rasterise: its edge functions (positive inside), the
/// depth across it as a plane, its box, its colour's index.
struct Face {
    e: [(f32, f32, f32); 3],
    z: (f32, f32, f32),
    /// Its colour as lit, red, green and blue, each a plane across it in
    /// the screen (linear light).
    rgb: [(f32, f32, f32); 3],
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
}

/// Faces meeting at a corner at less than this are shaded as one smooth
/// surface there; at more, the edge between them shows (a cube's do).
const CREASE: f32 = 40.0;

/// [`render_smooth`]'s rows `rows` alone (`width × rows.len()` pixels): a
/// band of the picture, so that several can be drawn at once (each as its
/// own job) and stacked. Every band sees the same faces, so bands meet
/// without a seam.
pub fn render_rows(
    mesh: &Mesh3,
    orbit: &Orbit,
    width: u32,
    height: u32,
    light: [f32; 3],
    samples: u32,
    rows: std::ops::Range<u32>,
) -> Vec<u8> {
    render_rows_with(
        mesh,
        &corner_normals(mesh),
        orbit,
        width,
        height,
        light,
        samples,
        rows,
    )
}

/// The normal of the surface at each face's corners: the face's own,
/// averaged with those of the faces round the corner that meet it at less
/// than the crease angle (bigger faces counting for more), so a curved
/// surface shades smoothly and a cube's edges stay sharp. They do not
/// change as the model turns: worked out once a model.
pub fn corner_normals(mesh: &Mesh3) -> Vec<[[f32; 3]; 3]> {
    // Each face's own normal (its length twice its area), and the faces
    // at each corner.
    let normals: Vec<[f32; 3]> = mesh
        .triangles
        .iter()
        .map(|t| {
            let p = t.map(|i| mesh.positions.get(i as usize).copied().unwrap_or_default());
            cross(sub(p[1], p[0]), sub(p[2], p[0]))
        })
        .collect();
    let corners = mesh.positions.len();
    let mut start = vec![0u32; corners + 1];
    for t in &mesh.triangles {
        for &i in t {
            if (i as usize) < corners {
                start[i as usize + 1] += 1;
            }
        }
    }
    for i in 0..corners {
        start[i + 1] += start[i];
    }
    let mut at_corner = vec![0u32; start[corners] as usize];
    let mut fill = start.clone();
    for (f, t) in mesh.triangles.iter().enumerate() {
        for &i in t {
            if (i as usize) < corners {
                at_corner[fill[i as usize] as usize] = f as u32;
                fill[i as usize] += 1;
            }
        }
    }
    let crease = CREASE.to_radians().cos();
    mesh.triangles
        .iter()
        .zip(&normals)
        .map(|(t, &n)| {
            let own = norm(n);
            t.map(|v| {
                let Some(around) = start.get(v as usize..v as usize + 2) else {
                    return own;
                };
                let mut sum = [0.0f32; 3];
                for &f in &at_corner[around[0] as usize..around[1] as usize] {
                    let m = normals[f as usize];
                    if dot(norm(m), own) >= crease {
                        sum = [sum[0] + m[0], sum[1] + m[1], sum[2] + m[2]];
                    }
                }
                let s = norm(sum);
                if s.iter().all(|c| c.is_finite()) && dot(s, s) > 0.5 {
                    s
                } else {
                    own
                }
            })
        })
        .collect()
}

/// [`render_rows`] with the corners' normals worked out already
/// ([`corner_normals`]): what each band of a model shares.
#[allow(clippy::too_many_arguments)]
pub fn render_rows_with(
    mesh: &Mesh3,
    normals: &[[[f32; 3]; 3]],
    orbit: &Orbit,
    width: u32,
    height: u32,
    light: [f32; 3],
    samples: u32,
    rows: std::ops::Range<u32>,
) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let (r0, r1) = (
        rows.start.min(height) as usize,
        rows.end.min(height) as usize,
    );
    if r1 <= r0 || w == 0 {
        return vec![];
    }
    // The camera's frame: forward to the origin, y up.
    let eye = orbit.eye();
    let fwd = norm(sub([0.0; 3], eye));
    let right = norm(cross(fwd, [0.0, 1.0, 0.0]));
    let up = cross(right, fwd);
    let f = 1.0 / (35f32.to_radians() / 2.0).tan();
    let aspect = width as f32 / height as f32;
    // Each corner into the camera's space and onto the screen once, however
    // many faces share it.
    let screen: Vec<Option<(f32, f32, f32)>> = mesh
        .positions
        .iter()
        .map(|&p| {
            let d = sub(p, eye);
            let v = [dot(d, right), dot(d, up), dot(d, fwd)];
            (v[2] > 1e-3).then(|| {
                (
                    (v[0] / v[2] * f / aspect * 0.5 + 0.5) * w as f32,
                    (0.5 - v[1] / v[2] * f * 0.5) * h as f32,
                    v[2],
                )
            })
        })
        .collect();
    // The light, from the camera's space (y down) to the world's.
    let l_cam = norm([light[0], -light[1], light[2]]);
    let l = norm([
        right[0] * l_cam[0] + up[0] * l_cam[1] - fwd[0] * l_cam[2],
        right[1] * l_cam[0] + up[1] * l_cam[1] - fwd[1] * l_cam[2],
        right[2] * l_cam[0] + up[2] * l_cam[1] - fwd[2] * l_cam[2],
    ]);
    // The faces this band can show.
    let mut faces = Vec::with_capacity(mesh.triangles.len() / 2);
    for (t, tri) in mesh.triangles.iter().enumerate() {
        let (Some(a), Some(b), Some(c)) = (
            screen[tri[0] as usize],
            screen[tri[1] as usize],
            screen[tri[2] as usize],
        ) else {
            continue;
        };
        let s = [a, b, c];
        let (lo_y, hi_y) = (
            s[0].1.min(s[1].1).min(s[2].1),
            s[0].1.max(s[1].1).max(s[2].1),
        );
        if hi_y < r0 as f32 || lo_y > r1 as f32 {
            continue;
        }
        // Facing away (wound the other way on screen): not drawn, and
        // nothing more worked out for it.
        let wind = (s[1].0 - s[0].0) * (s[2].1 - s[0].1) - (s[2].0 - s[0].0) * (s[1].1 - s[0].1);
        if wind >= 0.0 {
            continue;
        }
        // Edge functions, each zero along one edge and the whole area at
        // the corner across from it; turned to be positive inside.
        let edge =
            |a: (f32, f32, f32), b: (f32, f32, f32)| (a.1 - b.1, b.0 - a.0, a.0 * b.1 - b.0 * a.1);
        let mut e = [edge(s[1], s[2]), edge(s[2], s[0]), edge(s[0], s[1])];
        let mut area = e[0].2 + e[1].2 + e[2].2;
        if area.abs() < 1e-6 || !area.is_finite() {
            continue;
        }
        if area < 0.0 {
            e = e.map(|(a, b, c)| (-a, -b, -c));
            area = -area;
        }
        let x0 = s
            .iter()
            .map(|v| v.0)
            .fold(f32::MAX, f32::min)
            .floor()
            .max(0.0);
        let x1 = s
            .iter()
            .map(|v| v.0)
            .fold(f32::MIN, f32::max)
            .ceil()
            .min(w as f32);
        let y0 = lo_y.floor().max(r0 as f32);
        let y1 = hi_y.ceil().min(r1 as f32);
        if x0 >= x1 || y0 >= y1 {
            continue;
        }
        // Lit at each corner by the normal of the surface there.
        let c = mesh.colors.get(t).copied().unwrap_or([0.8; 3]);
        let lit_at = |n: [f32; 3]| {
            let lit = 0.25 + 0.75 * dot(n, l).max(0.0);
            [c[0] * lit, c[1] * lit, c[2] * lit].map(|c| c.clamp(0.0, 1.0))
        };
        let corner = normals.get(t).copied().unwrap_or_default().map(lit_at);
        // Depth and colour across the face as planes in the screen.
        let plane = |v: [f32; 3]| {
            let k = |f: fn(&(f32, f32, f32)) -> f32| {
                (f(&e[0]) * v[0] + f(&e[1]) * v[1] + f(&e[2]) * v[2]) / area
            };
            (k(|e| e.0), k(|e| e.1), k(|e| e.2))
        };
        let z = plane([s[0].2, s[1].2, s[2].2]);
        let rgb = [0, 1, 2].map(|ch| plane([corner[0][ch], corner[1][ch], corner[2][ch]]));
        faces.push(Face {
            e,
            z,
            rgb,
            x0: x0 as usize,
            x1: x1 as usize,
            y0: y0 as usize,
            y1: y1 as usize,
        });
    }
    let band = (w, r0, r1);
    match sample_pattern(samples).len() {
        1 => raster::<1>(&faces, band, [(0.0, 0.0)]),
        4 => raster::<4>(&faces, band, sample_pattern(4).try_into().expect("four")),
        _ => raster::<8>(&faces, band, sample_pattern(8).try_into().expect("eight")),
    }
}

/// The faces into a band `(width, first row, end row)` with `N` samples a
/// pixel at `pattern`, then each pixel the mean of its samples: one colour
/// as it is, several averaged as light, coverage as alpha. A face is shaded
/// once a pixel, at its centre, as a GPU multisamples. `N` is a constant so
/// the per-sample loops unroll.
fn raster<const N: usize>(
    faces: &[Face],
    (w, r0, r1): (usize, usize, usize),
    pattern: [(f32, f32); N],
) -> Vec<u8> {
    use threers_math::srgb::{linear_to_srgb, srgb_to_linear};
    let rows = r1 - r0;
    // Each sample's depth, and its colour (sRGB bytes, opaque; 0: none).
    let mut depth = vec![[f32::INFINITY; N]; w * rows];
    let mut color = vec![[0u32; N]; w * rows];
    let pack = |rgb: [f32; 3]| {
        u32::from_le_bytes([
            linear_to_srgb(rgb[0]),
            linear_to_srgb(rgb[1]),
            linear_to_srgb(rgb[2]),
            255,
        ])
    };
    for fc in faces {
        let e = fc.e;
        let (za, zb, zc) = fc.z;
        // How each edge function and the depth change from a pixel's centre
        // to each of its samples.
        let off: [[f32; N]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|k| e[i].0 * pattern[k].0 + e[i].1 * pattern[k].1)
        });
        let zoff: [f32; N] = std::array::from_fn(|k| za * pattern[k].0 + zb * pattern[k].1);
        // How far an edge function changes across half a pixel: a centre
        // further than that inside every edge has the whole pixel inside.
        let reach = e.map(|(a, b, _)| (a.abs() + b.abs()) * 0.5);
        let shade = |x: f32, y: f32| pack(fc.rgb.map(|(a, b, c)| a * x + b * y + c));
        for y in fc.y0..fc.y1 {
            let fy = y as f32 + 0.5;
            // The part of the row the face can touch: where each edge
            // function is above minus its reach (a pixel either side, for
            // rounding; the tests below decide).
            let (mut from, mut to) = (fc.x0 as f32, fc.x1 as f32);
            for (i, &(a, b, c)) in e.iter().enumerate() {
                let at = (-reach[i] - b * fy - c) / a - 0.5;
                if a > 0.0 {
                    from = from.max(at.floor() - 1.0);
                } else if a < 0.0 {
                    to = to.min(at.ceil() + 2.0);
                } else if b * fy + c < -reach[i] {
                    to = from;
                }
            }
            // (NaN bounds draw nothing.)
            if from.partial_cmp(&to) != Some(std::cmp::Ordering::Less) {
                continue;
            }
            let (x0, x1) = (from as usize, to as usize);
            let fx = x0 as f32 + 0.5;
            let mut c = e.map(|(a, b, c)| a * fx + b * fy + c);
            let mut z = za * fx + zb * fy + zc;
            let row = (y - r0) * w;
            for x in x0..x1 {
                let at = row + x;
                let inner = c[0] >= reach[0] && c[1] >= reach[1] && c[2] >= reach[2];
                if inner || (c[0] >= -reach[0] && c[1] >= -reach[1] && c[2] >= -reach[2]) {
                    // Wholly inside: every sample, by its depth; on an edge,
                    // those inside. Shaded once, if any sample takes it.
                    let (d, col) = (&mut depth[at], &mut color[at]);
                    let mut here = None;
                    for k in 0..N {
                        let inside = inner
                            || (c[0] + off[0][k] >= 0.0
                                && c[1] + off[1][k] >= 0.0
                                && c[2] + off[2][k] >= 0.0);
                        let zs = z + zoff[k];
                        if inside && zs < d[k] {
                            d[k] = zs;
                            col[k] = *here.get_or_insert_with(|| shade(x as f32 + 0.5, fy));
                        }
                    }
                }
                c[0] += e[0].0;
                c[1] += e[1].0;
                c[2] += e[2].0;
                z += za;
            }
        }
    }
    let mut rgba = vec![0u8; w * rows * 4];
    for (out, samples) in rgba.as_chunks_mut::<4>().0.iter_mut().zip(&color) {
        let first = samples[0];
        if samples.iter().all(|&s| s == first) {
            *out = first.to_le_bytes();
            continue;
        }
        let (mut sum, mut cover) = ([0.0f32; 3], 0u32);
        for &s in samples {
            if s != 0 {
                let b = s.to_le_bytes();
                for i in 0..3 {
                    sum[i] += srgb_to_linear(b[i]);
                }
                cover += 1;
            }
        }
        if cover > 0 {
            for i in 0..3 {
                out[i] = linear_to_srgb(sum[i] / cover as f32);
            }
            out[3] = (cover as f32 / N as f32 * 255.0).round() as u8;
        }
    }
    rgba
}

/// A model under `parent`, `size` pixels, turned by `orbit`: dragged with
/// the pointer or a finger (sideways turns it round, up and down tips it),
/// the wheel or `+` and `-` bring it nearer, the arrow keys turn it. It is
/// drawn again whenever the orbit or the light changes, at the display's
/// resolution with smooth edges, off the UI thread when the app has a
/// [job runner](crate::jobs). The orbit's distance is taken relative to
/// the model's size: at the default distance all of it shows. Assistive
/// technology hears an image named `label`, with how to turn it.
#[allow(clippy::too_many_arguments)]
pub fn model_view<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &S,
    parent: NodeId,
    label: &str,
    mesh: Mesh3,
    orbit: Binding<S, Orbit>,
    size: (u32, u32),
) -> NodeId {
    let (w, h) = (size.0.max(1), size.1.max(1));
    let blank = Image::new(w, h, vec![0; (w * h * 4) as usize]).expect("sized");
    let mut st = style! { cursor: grab; border-radius: 10px; };
    st.width = Length::Px(w as f32);
    st.height = Length::Px(h as f32);
    let node = ui.add_image(parent, st, blank);
    let mut sem = Semantics::new(Role::Image).label(label).tab_index(0);
    sem.description = Some("Drag or use the arrow keys to turn it; plus and minus to zoom".into());
    ui.set_semantics(node, sem);
    ui.focus_visible_style(node, "box-shadow: 0 0 0 2px #93c5fd");
    // Dragged: turned by how far.
    let drag = Rc::new(std::cell::Cell::new(None::<(f32, f32)>));
    let (o, d) = (orbit.clone(), drag.clone());
    input.on(node, Phase::Bubble, move |s, _, e, cx| match *e {
        Event::PointerDown { x, y, .. } => {
            d.set(Some((x, y)));
            cx.capture_pointer();
        }
        Event::PointerMove { x, y } => {
            if let Some((x0, y0)) = d.get() {
                d.set(Some((x, y)));
                let now = o.get(s).turned((x - x0) * 0.012, (y - y0) * 0.012);
                o.set(s, now);
            }
        }
        Event::PointerUp { .. } => d.set(None),
        Event::Wheel { dy, .. } => {
            let now = o.get(s).zoomed((dy * 0.002).exp());
            o.set(s, now);
            cx.prevent_default();
        }
        _ => {}
    });
    let o = orbit.clone();
    input.on_key(node, move |s, _, key, _, _| {
        let now = o.get(s);
        let next = match key {
            "ArrowLeft" => now.turned(-0.2, 0.0),
            "ArrowRight" => now.turned(0.2, 0.0),
            "ArrowUp" => now.turned(0.0, 0.15),
            "ArrowDown" => now.turned(0.0, -0.15),
            "+" | "=" => now.zoomed(0.85),
            "-" => now.zoomed(1.0 / 0.85),
            _ => return false,
        };
        o.set(s, next);
        true
    });
    // Drawn again when what it shows changes: off the UI thread when the
    // app has a job runner, one drawing at a time, the newest view next.
    let view = Rc::new(View {
        mesh: encode_mesh(&mesh),
        faces: mesh.triangles.len(),
        fit: fit_scale(&mesh, w as f32 / h as f32),
        node,
        size: (w, h),
        wanted: std::cell::Cell::new(None),
        drawn: std::cell::Cell::new(None),
        busy: std::cell::Cell::new(false),
    });
    let draw = move |ui: &mut Ui, s: &S| {
        view.wanted.set(Some(Seen {
            orbit: orbit.get(s),
            light: ui.light().direction(),
            ratio: ui.pixel_ratio,
        }));
        view.clone().start(ui);
    };
    draw(ui, state);
    input.on_after_dispatch(move |_, ui, s| {
        if !ui.contains(node) {
            return false;
        }
        draw(ui, s);
        true
    });
    node
}

/// A model view's drawing: what it shows and what it is drawing.
struct View {
    mesh: Vec<u8>,
    /// How many faces it has: a dense one is drawn in bands, at once.
    faces: usize,
    /// What the orbit's distance is multiplied by, so that at the default
    /// distance the whole model is in view (see [`fit_scale`]).
    fit: f32,
    node: NodeId,
    size: (u32, u32),
    wanted: std::cell::Cell<Option<Seen>>,
    drawn: std::cell::Cell<Option<Seen>>,
    busy: std::cell::Cell<bool>,
}

/// What a drawing shows: the orbit and the light's direction, compared by
/// their bits (so a NaN from the app is the same NaN, not a new view every
/// time).
#[derive(Clone, Copy)]
struct Seen {
    orbit: Orbit,
    light: [f32; 3],
    /// Device pixels per UI pixel: drawn at the display's resolution.
    ratio: f32,
}

impl Seen {
    fn bits(&self) -> [u32; 7] {
        let o = self.orbit;
        [
            o.yaw,
            o.pitch,
            o.distance,
            self.light[0],
            self.light[1],
            self.light[2],
            self.ratio,
        ]
        .map(f32::to_bits)
    }
}

impl View {
    /// Draws the view wanted, unless it is drawn or a drawing is running
    /// (which starts the next when it lands).
    fn start(self: Rc<Self>, ui: &mut Ui) {
        let Some(now) = self.wanted.get() else { return };
        if self.busy.get() || self.drawn.get().is_some_and(|d| d.bits() == now.bits()) {
            return;
        }
        self.busy.set(true);
        self.drawn.set(Some(now));
        // At the display's resolution, multisampled for smooth edges (more
        // samples where pixels are large).
        let (w, h) = self.size;
        let ratio = if now.ratio.is_finite() {
            now.ratio.clamp(0.5, 4.0)
        } else {
            1.0
        };
        let (pw, ph) = (
            ((w as f32 * ratio).round() as u32).clamp(1, 2048),
            ((h as f32 * ratio).round() as u32).clamp(1, 2048),
        );
        let samples: u32 = if ratio < 1.5 { 8 } else { 4 };
        let orbit = Orbit {
            distance: now.orbit.distance * self.fit,
            ..now.orbit
        };
        // A dense model or a large picture in bands of rows, each its own
        // job: with workers, drawn side by side; stacked when all are in.
        let bands = if self.faces >= 4096 || pw * ph >= 250_000 {
            4.min(ph)
        } else {
            1
        };
        let parts = Rc::new(std::cell::RefCell::new(vec![
            None::<Vec<u8>>;
            bands as usize
        ]));
        for b in 0..bands {
            let rows = (ph * b / bands, ph * (b + 1) / bands);
            let mut input = Vec::with_capacity(self.mesh.len() + 56);
            for v in [
                orbit.yaw,
                orbit.pitch,
                orbit.distance,
                now.light[0],
                now.light[1],
                now.light[2],
            ] {
                input.extend_from_slice(&v.to_le_bytes());
            }
            for v in [pw, ph, samples, rows.0, rows.1] {
                input.extend_from_slice(&v.to_le_bytes());
            }
            input.extend_from_slice(&self.mesh);
            let (view, parts) = (self.clone(), parts.clone());
            ui.spawn_job(render_job, input, move |ui, px| {
                parts.borrow_mut()[b as usize] = Some(px);
                if parts.borrow().iter().any(Option::is_none) {
                    return;
                }
                let px: Vec<u8> = parts
                    .borrow_mut()
                    .iter_mut()
                    .flat_map(|p| p.take().unwrap_or_default())
                    .collect();
                view.busy.set(false);
                if ui.contains(view.node)
                    && let Ok(img) = Image::new(pw, ph, px)
                {
                    ui.set_image(view.node, img);
                }
                view.start(ui);
            });
        }
    }
}

/// `mesh` as bytes: counts, then positions, triangles and colours, little
/// endian.
fn encode_mesh(mesh: &Mesh3) -> Vec<u8> {
    let mut b = encode_geometry(mesh);
    // Then the corners' normals, worked out here once rather than in every
    // band of every drawing.
    for n in corner_normals(mesh).iter().flatten().flatten() {
        b.extend_from_slice(&n.to_le_bytes());
    }
    b
}

/// The mesh alone: counts, then positions, colours and triangles.
fn encode_geometry(mesh: &Mesh3) -> Vec<u8> {
    let mut b = Vec::new();
    for n in [
        mesh.positions.len(),
        mesh.triangles.len(),
        mesh.colors.len(),
    ] {
        b.extend_from_slice(&(n as u32).to_le_bytes());
    }
    for v in mesh.positions.iter().chain(&mesh.colors).flatten() {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for i in mesh.triangles.iter().flatten() {
        b.extend_from_slice(&i.to_le_bytes());
    }
    b
}

/// Reads little-endian words off the front of bytes.
struct Words<'a>(&'a [u8]);

impl Words<'_> {
    fn u32(&mut self) -> u32 {
        let (w, rest) = self
            .0
            .split_at_checked(4)
            .unwrap_or((&[0u8; 4][..], &[][..]));
        self.0 = rest;
        u32::from_le_bytes(w.try_into().expect("four bytes"))
    }
    fn f32(&mut self) -> f32 {
        f32::from_bits(self.u32())
    }
    fn vec3(&mut self) -> [f32; 3] {
        [self.f32(), self.f32(), self.f32()]
    }
}

/// [`render_rows`] as a [`Job`](crate::jobs::Job): the orbit, the light,
/// the size, the samples per pixel, the rows and the mesh in; the rows'
/// pixels out.
/// Indices outside the mesh are dropped rather than trusted.
fn render_job(input: &[u8]) -> Vec<u8> {
    let mut r = Words(input);
    let orbit = Orbit {
        yaw: r.f32(),
        pitch: r.f32(),
        distance: r.f32(),
    };
    let light = r.vec3();
    let (w, h) = (r.u32().clamp(1, 2048), r.u32().clamp(1, 2048));
    let samples = r.u32().clamp(1, 8);
    let rows = r.u32().min(h)..r.u32().min(h);
    let (np, nt, nc) = (r.u32() as usize, r.u32() as usize, r.u32() as usize);
    // No more of anything than the bytes could hold.
    let room = r.0.len() / 12;
    let positions: Vec<[f32; 3]> = (0..np.min(room)).map(|_| r.vec3()).collect();
    let colors: Vec<[f32; 3]> = (0..nc.min(room)).map(|_| r.vec3()).collect();
    let triangles = (0..nt.min(r.0.len() / 12))
        .map(|_| [r.u32(), r.u32(), r.u32()])
        .filter(|t| t.iter().all(|&i| (i as usize) < positions.len()))
        .collect();
    let mesh = Mesh3 {
        positions,
        triangles,
        colors,
    };
    // The corners' normals, sent with the mesh (one per corner of every
    // face it was sent with); worked out here if they did not come, or do
    // not match.
    let sent = (r.0.len() / 36 == nt && nt == mesh.triangles.len()).then(|| {
        (0..nt)
            .map(|_| [r.vec3(), r.vec3(), r.vec3()])
            .collect::<Vec<_>>()
    });
    let normals = sent.unwrap_or_else(|| corner_normals(&mesh));
    render_rows_with(&mesh, &normals, &orbit, w, h, light, samples, rows)
}

/// How far a camera must be, as a multiple of the default orbit's
/// distance, for all of `mesh` (its bounding sphere about the origin, which
/// the camera looks at) to show in a view `aspect` wide for each unit high,
/// with a little room.
fn fit_scale(mesh: &Mesh3, aspect: f32) -> f32 {
    let radius = mesh
        .positions
        .iter()
        .map(|p| dot(*p, *p))
        .fold(0.0f32, f32::max)
        .sqrt();
    // None, or NaN: as it is.
    if radius.is_nan() || radius <= 0.0 {
        return 1.0;
    }
    let half_v = 35f32.to_radians() / 2.0;
    // The narrower of the two half-angles the view spans.
    let half_h = (half_v.tan() * aspect.max(0.05)).atan();
    let half = half_v.min(half_h);
    let need = radius * 1.08 / half.sin();
    need / Orbit::default().distance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cube_is_drawn_lit_and_turns() {
        let cube = Mesh3::cube(1.5, [0.8, 0.3, 0.2]);
        let o = Orbit::default();
        let a = render(&cube, &o, 64, 48, [-0.4, -0.6, 1.0]);
        let covered = a.chunks(4).filter(|p| p[3] == 255).count();
        assert!(covered > 300, "{covered} pixels");
        // Its faces lit differently.
        let mut shades: Vec<u8> = a.chunks(4).filter(|p| p[3] == 255).map(|p| p[0]).collect();
        shades.sort();
        shades.dedup();
        assert!(shades.len() >= 2, "{shades:?}");
        let b = render(&cube, &o.turned(0.5, 0.0), 64, 48, [-0.4, -0.6, 1.0]);
        assert_ne!(a, b, "turned, it looks different");
        // Nearer, it covers more.
        let c = render(&cube, &o.zoomed(0.6), 64, 48, [-0.4, -0.6, 1.0]);
        assert!(c.chunks(4).filter(|p| p[3] == 255).count() > covered);
    }

    #[test]
    fn multisampling_smooths_edges_and_leaves_insides_alone() {
        let cube = Mesh3::cube(1.5, [0.8, 0.3, 0.2]);
        let o = Orbit::default().turned(0.4, 0.3);
        let one = render(&cube, &o, 96, 72, [-0.4, -0.6, 1.0]);
        for n in [4, 8] {
            let smooth = render_smooth(&cube, &o, 96, 72, [-0.4, -0.6, 1.0], n);
            let partial = smooth.chunks(4).filter(|p| p[3] > 0 && p[3] < 255).count();
            assert!(
                partial > 60,
                "{n} samples: {partial} edge pixels part covered"
            );
            // Coverage in steps of a sample.
            assert!(smooth
                .chunks(4)
                .all(|p| (0..=n).any(|k| (k as f32 / n as f32 * 255.0).round() as u8 == p[3])));
            // Away from edges, as one sample draws it.
            let w = 96;
            let inside = |img: &[u8], i: usize| {
                (img[i * 4 + 3] == 255)
                    && [1, w, w + 1].iter().all(|&d| {
                        i >= d
                            && img[(i - d) * 4 + 3] == 255
                            && img.get((i + d) * 4 + 3) == Some(&255)
                    })
            };
            let mut same = 0;
            for i in 0..96 * 72 {
                if inside(&one, i)
                    && inside(&smooth, i)
                    && one[i * 4..i * 4 + 3] == smooth[i * 4..i * 4 + 3]
                {
                    same += 1;
                }
            }
            let full = (0..96 * 72).filter(|&i| inside(&one, i)).count();
            assert!(same as f32 > full as f32 * 0.9, "{same} of {full}");
            // About as much covered: the edges move by less than a pixel.
            let a: f32 = one.chunks(4).map(|p| p[3] as f32 / 255.0).sum();
            let b: f32 = smooth.chunks(4).map(|p| p[3] as f32 / 255.0).sum();
            assert!((a - b).abs() < a * 0.03, "{a} against {b}");
        }
    }

    #[test]
    fn curved_surfaces_shade_smoothly_and_creases_stay_sharp() {
        let l = [-0.4, -0.6, 1.0];
        // A coarse torus (faces 22.5° and 30° apart, under the crease),
        // seen from above so nothing stands in front of anything:
        // flat-shaded, its facets would step by many levels.
        let torus = Mesh3::torus(1.0, 0.4, 16, 12, [0.6, 0.6, 0.6]);
        let img = render(
            &torus,
            &Orbit {
                pitch: 1.4,
                ..Orbit::default()
            },
            160,
            120,
            l,
        );
        let mut worst = 0u8;
        for y in 0..120 {
            for x in 1..160 {
                let (a, b) = (
                    &img[(y * 160 + x - 1) * 4..][..4],
                    &img[(y * 160 + x) * 4..][..4],
                );
                // Neighbours both inside, on the same side of a silhouette.
                if a[3] == 255 && b[3] == 255 {
                    worst = worst.max(a[0].abs_diff(b[0]));
                }
            }
        }
        // Flat, the worst step here is 94 levels; smooth, what is left is the
        // light changing fast (but evenly) round the tube's edge.
        assert!(worst < 60, "no step between facets bigger than {worst}");
        // A cube's faces meet at 90°: each stays one colour.
        let cube = Mesh3::cube(1.5, [0.6, 0.6, 0.6]);
        let img = render(&cube, &Orbit::default().turned(0.6, 0.4), 120, 90, l);
        let mut shades: Vec<u8> = img
            .chunks(4)
            .filter(|p| p[3] == 255)
            .map(|p| p[0])
            .collect();
        shades.sort();
        shades.dedup();
        assert!(shades.len() <= 3, "three faces, three shades: {shades:?}");
    }

    #[test]
    fn bands_stacked_are_the_whole_picture() {
        let torus = Mesh3::torus(1.0, 0.4, 40, 20, [0.2, 0.4, 0.9]);
        let o = Orbit::default().turned(0.7, 0.4);
        for n in [1, 4, 8] {
            let whole = render_smooth(&torus, &o, 97, 61, [-0.4, -0.6, 1.0], n);
            let mut stacked = vec![];
            for rows in [0..13, 13..14, 14..40, 40..61] {
                stacked.extend(render_rows(&torus, &o, 97, 61, [-0.4, -0.6, 1.0], n, rows));
            }
            assert!(stacked == whole, "{n} samples");
        }
        assert!(render_rows(&torus, &o, 97, 61, [0.0, 0.0, 1.0], 4, 70..80).is_empty());
    }

    #[test]
    fn a_job_draws_what_a_call_draws() {
        let torus = Mesh3::torus(1.0, 0.4, 24, 12, [0.2, 0.4, 0.9]);
        let o = Orbit::default().turned(0.3, 0.1);
        let mut input = Vec::new();
        for v in [o.yaw, o.pitch, o.distance, -0.4, -0.6, 1.0] {
            input.extend_from_slice(&f32::to_le_bytes(v));
        }
        for v in [80u32, 60, 1, 0, 60] {
            input.extend_from_slice(&v.to_le_bytes());
        }
        input.extend_from_slice(&encode_mesh(&torus));
        assert_eq!(
            render_job(&input),
            render(&torus, &o, 80, 60, [-0.4, -0.6, 1.0])
        );
        // Cut short or nonsense, it still draws (nothing) rather than fail.
        for n in [0, 7, 30, input.len() - 5] {
            assert_eq!(render_job(&input[..n]).len() % 4, 0);
        }
    }

    // Worker threads exist natively only (`jobs::Threads`).
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn on_threads_the_view_lands_later_and_follows_the_newest_turn() {
        use crate::{binding, FixedWidthText};
        #[derive(Default)]
        struct S {
            orbit: Orbit,
        }
        let mut ui = Ui::new();
        ui.set_job_runner(crate::jobs::Threads::new(2));
        let mut input: Input<S> = Input::new();
        let mut s = S::default();
        let root = ui.root();
        let view = model_view(
            &mut ui,
            &mut input,
            &s,
            root,
            "A cube",
            Mesh3::cube(1.5, [0.8, 0.3, 0.2]),
            binding!(|s: S| s.orbit),
            (64, 48),
        );
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
        let pixels = |ui: &Ui| match ui.kind(view) {
            crate::NodeKind::Image { image } => image.rgba.clone(),
            _ => vec![],
        };
        let covered = |ui: &Ui| pixels(ui).chunks(4).filter(|p| p[3] == 255).count();
        assert_eq!(covered(&ui), 0, "blank until the drawing lands");
        let settle = |ui: &mut Ui| {
            let start = std::time::Instant::now();
            while ui.jobs_pending() {
                ui.poll_jobs();
                assert!(start.elapsed().as_secs() < 10);
            }
        };
        settle(&mut ui);
        assert!(covered(&ui) > 300);
        // Turned twice while the first turn draws: the last one shows.
        input.set_focus(&mut ui, &mut s, Some(view));
        input
            .simulate_keystrokes(&mut ui, &mut s, "right right right")
            .unwrap();
        settle(&mut ui);
        // As it is drawn on the spot, turned the same.
        let mut here = Ui::new();
        let mut input2: Input<S> = Input::new();
        let root2 = here.root();
        let view2 = model_view(
            &mut here,
            &mut input2,
            &s,
            root2,
            "A cube",
            Mesh3::cube(1.5, [0.8, 0.3, 0.2]),
            binding!(|s: S| s.orbit),
            (64, 48),
        );
        let want = match here.kind(view2) {
            crate::NodeKind::Image { image } => image.rgba.clone(),
            _ => vec![],
        };
        assert_eq!(pixels(&ui), want);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn a_dense_model_is_drawn_in_bands_on_threads_as_on_the_spot() {
        use crate::binding;
        #[derive(Default)]
        struct S {
            orbit: Orbit,
        }
        let dense = || Mesh3::torus(1.0, 0.4, 96, 48, [0.2, 0.5, 0.8]);
        assert!(dense().triangles.len() >= 4096);
        let s = S {
            orbit: Orbit::default().turned(0.5, 0.3),
        };
        let image = |ui: &Ui, n: NodeId| match ui.kind(n) {
            crate::NodeKind::Image { image } => image.rgba.clone(),
            _ => vec![],
        };
        let mut threaded = Ui::new();
        threaded.set_job_runner(crate::jobs::Threads::new(4));
        let mut input: Input<S> = Input::new();
        let root = threaded.root();
        let view = model_view(
            &mut threaded,
            &mut input,
            &s,
            root,
            "A torus",
            dense(),
            binding!(|s: S| s.orbit),
            (90, 70),
        );
        let start = std::time::Instant::now();
        while threaded.jobs_pending() {
            threaded.poll_jobs();
            assert!(start.elapsed().as_secs() < 20);
        }
        let mut here = Ui::new();
        let mut input2: Input<S> = Input::new();
        let root2 = here.root();
        let view2 = model_view(
            &mut here,
            &mut input2,
            &s,
            root2,
            "A torus",
            dense(),
            binding!(|s: S| s.orbit),
            (90, 70),
        );
        let (a, b) = (image(&threaded, view), image(&here, view2));
        assert_eq!(a.len(), 90 * 70 * 4);
        assert!(a.chunks(4).filter(|p| p[3] == 255).count() > 1000);
        assert!(a == b, "bands from threads stack as drawn on the spot");
    }

    fn drawn(ui: &Ui, node: NodeId) -> (u32, u32, Vec<u8>) {
        match ui.kind(node) {
            crate::NodeKind::Image { image } => (image.width, image.height, image.rgba.clone()),
            _ => panic!("not an image"),
        }
    }

    #[test]
    fn the_whole_model_shows_at_the_screens_resolution_with_smooth_edges() {
        use crate::binding;
        #[derive(Default)]
        struct S {
            orbit: Orbit,
        }
        for (ratio, size) in [(1.0, (300, 180)), (2.0, (300, 180)), (1.0, (120, 200))] {
            let mut ui = Ui::new();
            ui.set_pixel_ratio(ratio);
            let mut input: Input<S> = Input::new();
            let mut s = S::default();
            let root = ui.root();
            let torus = Mesh3::torus(1.0, 0.38, 36, 18, [0.15, 0.39, 0.92]);
            let view = model_view(
                &mut ui,
                &mut input,
                &s,
                root,
                "A torus",
                torus,
                binding!(|s: S| s.orbit),
                size,
            );
            for turn in [0.0, 0.8, 1.6] {
                s.orbit = Orbit::default().turned(turn, turn - 0.5);
                input.set_focus(&mut ui, &mut s, Some(view));
                input.simulate_keystrokes(&mut ui, &mut s, "right").unwrap();
                let (w, h, px) = drawn(&ui, view);
                assert_eq!(
                    (w, h),
                    (
                        (size.0 as f32 * ratio) as u32,
                        (size.1 as f32 * ratio) as u32
                    )
                );
                // Nothing on the edges: none of it is cut off.
                let at = |x: u32, y: u32| px[((y * w + x) * 4 + 3) as usize];
                let edge = (0..w).any(|x| at(x, 0) > 0 || at(x, h - 1) > 0)
                    || (0..h).any(|y| at(0, y) > 0 || at(w - 1, y) > 0);
                assert!(
                    !edge,
                    "{size:?} at {ratio}x, turned {turn}: the model reaches the edge"
                );
                // Its outline in between: partly covered pixels.
                let partial = px.chunks(4).filter(|p| p[3] > 20 && p[3] < 235).count();
                assert!(partial > 100, "{partial} smooth edge pixels");
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn a_view_it_cannot_compare_is_still_drawn_once() {
        use crate::{binding, depth::Light};
        struct S {
            orbit: Orbit,
        }
        for threads in [false, true] {
            let mut ui = Ui::new();
            if threads {
                ui.set_job_runner(crate::jobs::Threads::new(1));
            }
            // NaN is never equal to itself: a view that kept asking would
            // draw for ever (on the spot, without end).
            ui.set_light(Light {
                x: f32::NAN,
                ..Light::default()
            });
            let mut input: Input<S> = Input::new();
            let s = S {
                orbit: Orbit {
                    yaw: f32::NAN,
                    ..Orbit::default()
                },
            };
            let root = ui.root();
            model_view(
                &mut ui,
                &mut input,
                &s,
                root,
                "A cube",
                Mesh3::cube(1.0, [0.5; 3]),
                binding!(|s: S| s.orbit),
                (16, 16),
            );
            let start = std::time::Instant::now();
            while ui.jobs_pending() {
                ui.poll_jobs();
                assert!(start.elapsed().as_secs() < 5, "still drawing");
            }
        }
    }
}
