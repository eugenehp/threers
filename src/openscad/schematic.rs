//! Orthographic **schematic projection** — turn a solid into technical-drawing
//! line art (silhouette + feature edges, with hidden-line removal), the way a
//! CAD package produces front/top/right/iso views.
//!
//! This is the missing counterpart to the mesh exporters: [`crate::openscad::schematic::schematic_to_svg`] writes the
//! *surface*, this writes the *drawing*. It exists so a model can be compared
//! one-to-one against an existing engineering drawing — project to the same
//! view, export SVG, overlay.
//!
//! ```ignore
//! use threers::{parse_scad_file, schematic_project_view, schematic_to_svg, SchematicOptions, View};
//!
//! let geom = parse_scad_file("part.scad").unwrap().to_geometry_exact();
//! let opts = SchematicOptions { hidden: true, ..Default::default() };
//! let drawing = schematic_project_view(&geom, View::Front, &opts);
//! std::fs::write("front.svg", schematic_to_svg(&drawing, 0.35)).unwrap();
//! ```
//!
//! Edges are kept when they are a **boundary** (used by one triangle), a
//! **silhouette** (the two triangles face opposite ways relative to the view),
//! or a **crease** (dihedral angle over [`crate::openscad::schematic::SchematicOptions::crease_deg`]) — the
//! tessellation of a smooth cylinder therefore does not explode into a hundred
//! meridian lines. Visibility is resolved against a depth buffer rasterised
//! from the same projection, so the cost is linear in triangles and independent
//! of how many edges survive.

use crate::core::BufferGeometry;

/// A projected 2D drawing, in model units (mm), y up.
#[derive(Debug, Clone, Default)]
pub struct Schematic {
    /// Segments the viewer can see.
    pub visible: Vec<[[f32; 2]; 2]>,
    /// Segments occluded by the solid (conventionally drawn dashed).
    pub hidden: Vec<[[f32; 2]; 2]>,
    /// Drawing extents, `[min_x, min_y]` / `[max_x, max_y]`.
    pub min: [f32; 2],
    pub max: [f32; 2],
}

impl Schematic {
    /// Width and height of the drawing, in model units.
    pub fn size(&self) -> [f32; 2] {
        [self.max[0] - self.min[0], self.max[1] - self.min[1]]
    }

    /// Total segment count across both layers.
    pub fn len(&self) -> usize {
        self.visible.len() + self.hidden.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The six orthographic views plus an isometric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    Front,
    Back,
    Left,
    Right,
    Top,
    Bottom,
    Iso,
}

impl View {
    /// Parse a view name (`front`, `top`, `iso`, …), case-insensitive.
    pub fn parse(s: &str) -> Option<View> {
        match s.to_ascii_lowercase().as_str() {
            "front" => Some(View::Front),
            "back" => Some(View::Back),
            "left" => Some(View::Left),
            "right" => Some(View::Right),
            "top" => Some(View::Top),
            "bottom" => Some(View::Bottom),
            "iso" => Some(View::Iso),
            _ => None,
        }
    }

    /// View direction (from the eye toward the model) and the up hint.
    pub fn basis(&self) -> ([f32; 3], [f32; 3]) {
        match self {
            View::Front => ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
            View::Back => ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
            View::Left => ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            View::Right => ([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            View::Top => ([0.0, -1.0, 0.0], [0.0, 0.0, -1.0]),
            View::Bottom => ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            View::Iso => ([-1.0, -1.0, -1.0], [0.0, 1.0, 0.0]),
        }
    }
}

/// Knobs for [`schematic_project`].
#[derive(Debug, Clone)]
pub struct SchematicOptions {
    /// Keep an interior edge when the angle between its two faces exceeds this
    /// many degrees. 20-30 keeps real corners and drops tessellation seams.
    pub crease_deg: f32,
    /// Resolve occlusion and emit the [`Schematic::hidden`] layer. Costs one
    /// depth-buffer rasterisation plus a walk along every edge.
    pub hidden: bool,
    /// Depth-buffer resolution used for visibility. Higher = finer separation
    /// of near-coincident edges; 2048 is plenty for a full-page drawing.
    pub raster: usize,
    /// Depth bias as a fraction of the model's depth range, applied when
    /// testing an edge against the buffer. Too small speckles silhouettes,
    /// too large lets hidden edges leak through.
    pub bias: f32,
}

impl Default for SchematicOptions {
    fn default() -> Self {
        Self {
            crease_deg: 22.0,
            hidden: false,
            raster: 2048,
            bias: 2e-3,
        }
    }
}

// --- small vector helpers (kept local so this module stays dependency-free) ---

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
fn norm(v: [f32; 3]) -> [f32; 3] {
    let l = dot(v, v).sqrt();
    if l > 1e-20 {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Quantised vertex key, so triangles that share a corner share an edge entry
/// even when the mesh is a non-indexed soup (which the exact kernel emits).
fn key(p: [f32; 3]) -> (i64, i64, i64) {
    const Q: f32 = 1e4;
    (
        (p[0] * Q).round() as i64,
        (p[1] * Q).round() as i64,
        (p[2] * Q).round() as i64,
    )
}

fn triangles(geom: &BufferGeometry) -> Vec<[[f32; 3]; 3]> {
    let Some(pos) = geom.get_attribute("position") else {
        return Vec::new();
    };
    let p = |i: usize| [pos.array[i * 3], pos.array[i * 3 + 1], pos.array[i * 3 + 2]];
    match &geom.index {
        Some(ix) => ix
            .chunks_exact(3)
            .map(|t| [p(t[0] as usize), p(t[1] as usize), p(t[2] as usize)])
            .collect(),
        None => (0..pos.count() / 3)
            .map(|t| [p(t * 3), p(t * 3 + 1), p(t * 3 + 2)])
            .collect(),
    }
}

/// Project `geom` along `dir` (eye → model) into 2D line art.
///
/// `up` is a hint; the component along `dir` is removed. Returns an empty
/// drawing for empty input.
pub fn schematic_project(
    geom: &BufferGeometry,
    dir: [f32; 3],
    up: [f32; 3],
    opts: &SchematicOptions,
) -> Schematic {
    let tris = triangles(geom);
    if tris.is_empty() {
        return Schematic::default();
    }

    // View basis: f into the screen, r right, u up.
    let f = norm(dir);
    let mut r = cross(f, up);
    if dot(r, r) < 1e-12 {
        // up parallel to the view direction — pick any perpendicular.
        r = cross(f, [1.0, 0.0, 0.0]);
        if dot(r, r) < 1e-12 {
            r = cross(f, [0.0, 0.0, 1.0]);
        }
    }
    let r = norm(r);
    let u = norm(cross(r, f));
    let to2 = |p: [f32; 3]| [dot(p, r), dot(p, u)];
    let depth = |p: [f32; 3]| dot(p, f);

    // Per-triangle projected corners, depths and facing.
    let mut proj: Vec<[[f32; 2]; 3]> = Vec::with_capacity(tris.len());
    let mut zs: Vec<[f32; 3]> = Vec::with_capacity(tris.len());
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(tris.len());
    let (mut min, mut max) = ([f32::MAX; 2], [f32::MIN; 2]);
    let (mut zmin, mut zmax) = (f32::MAX, f32::MIN);
    for t in &tris {
        let a = to2(t[0]);
        let b = to2(t[1]);
        let c = to2(t[2]);
        for p in [a, b, c] {
            min[0] = min[0].min(p[0]);
            min[1] = min[1].min(p[1]);
            max[0] = max[0].max(p[0]);
            max[1] = max[1].max(p[1]);
        }
        let z = [depth(t[0]), depth(t[1]), depth(t[2])];
        for v in z {
            zmin = zmin.min(v);
            zmax = zmax.max(v);
        }
        proj.push([a, b, c]);
        zs.push(z);
        normals.push(norm(cross(sub(t[1], t[0]), sub(t[2], t[0]))));
    }

    // --- edge classification -------------------------------------------------
    // edge key -> (endpoints, adjacent triangle ids)
    use std::collections::HashMap;
    type PointKey = (i64, i64, i64);
    /// Snapped edge -> its endpoints, and the triangles either side of it.
    type EdgeUse = HashMap<(PointKey, PointKey), ([[f32; 3]; 2], Vec<usize>)>;
    let mut edges: EdgeUse = HashMap::new();
    for (ti, t) in tris.iter().enumerate() {
        for e in 0..3 {
            let (p, q) = (t[e], t[(e + 1) % 3]);
            let (kp, kq) = (key(p), key(q));
            let k = if kp <= kq { (kp, kq) } else { (kq, kp) };
            let ends = if kp <= kq { [p, q] } else { [q, p] };
            edges
                .entry(k)
                .or_insert_with(|| (ends, Vec::new()))
                .1
                .push(ti);
        }
    }

    let cos_crease = opts.crease_deg.to_radians().cos();
    let mut keep: Vec<[[f32; 3]; 2]> = Vec::new();
    for (ends, adj) in edges.values() {
        let draw = match adj.len() {
            0 => false,
            1 => true, // boundary of an open mesh
            _ => {
                // Silhouette: the adjacent faces disagree about facing the eye.
                let mut any_front = false;
                let mut any_back = false;
                for &t in adj {
                    if dot(normals[t], f) < 0.0 {
                        any_front = true;
                    } else {
                        any_back = true;
                    }
                }
                let silhouette = any_front && any_back;
                // Crease: the faces meet at a real angle.
                let mut crease = false;
                for i in 0..adj.len() {
                    for j in i + 1..adj.len() {
                        if dot(normals[adj[i]], normals[adj[j]]) < cos_crease {
                            crease = true;
                        }
                    }
                }
                silhouette || crease
            }
        };
        if draw {
            keep.push(*ends);
        }
    }

    if !opts.hidden {
        let visible = keep.iter().map(|e| [to2(e[0]), to2(e[1])]).collect();
        return Schematic {
            visible,
            hidden: Vec::new(),
            min,
            max,
        };
    }

    // --- depth buffer --------------------------------------------------------
    let n = opts.raster.max(64);
    let span = (max[0] - min[0]).max(max[1] - min[1]).max(1e-6);
    let scale = (n - 1) as f32 / span;
    let px = |p: [f32; 2]| ((p[0] - min[0]) * scale, (p[1] - min[1]) * scale);
    let mut zbuf = vec![f32::MAX; n * n];
    for (ti, tp) in proj.iter().enumerate() {
        let (a, b, c) = (px(tp[0]), px(tp[1]), px(tp[2]));
        let (za, zb, zc) = (zs[ti][0], zs[ti][1], zs[ti][2]);
        let minx = a.0.min(b.0).min(c.0).floor().max(0.0) as usize;
        let maxx = (a.0.max(b.0).max(c.0).ceil() as usize).min(n - 1);
        let miny = a.1.min(b.1).min(c.1).floor().max(0.0) as usize;
        let maxy = (a.1.max(b.1).max(c.1).ceil() as usize).min(n - 1);
        let area = (b.0 - a.0) * (c.1 - a.1) - (c.0 - a.0) * (b.1 - a.1);
        if area.abs() < 1e-12 {
            continue;
        }
        for y in miny..=maxy {
            for x in minx..=maxx {
                let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
                let w0 = ((b.0 - a.0) * (fy - a.1) - (fx - a.0) * (b.1 - a.1)) / area;
                let w1 = ((fx - a.0) * (c.1 - a.1) - (c.0 - a.0) * (fy - a.1)) / area;
                let w2 = 1.0 - w0 - w1;
                const E: f32 = -1e-4;
                if w0 < E || w1 < E || w2 < E {
                    continue;
                }
                let z = za * w2 + zb * w1 + zc * w0;
                let slot = &mut zbuf[y * n + x];
                if z < *slot {
                    *slot = z;
                }
            }
        }
    }

    // --- walk each edge, splitting into visible and hidden runs ---------------
    let bias = (zmax - zmin).max(1e-6) * opts.bias;
    let mut visible = Vec::new();
    let mut hidden = Vec::new();
    for e in &keep {
        let (p2, q2) = (to2(e[0]), to2(e[1]));
        let (pz, qz) = (depth(e[0]), depth(e[1]));
        let (pp, qp) = (px(p2), px(q2));
        let len_px = ((qp.0 - pp.0).hypot(qp.1 - pp.1)).max(1.0);
        let steps = (len_px.ceil() as usize).clamp(2, 4096);
        let mut run_start = 0usize;
        let mut run_vis: Option<bool> = None;
        let lerp2 = |t: f32| [p2[0] + (q2[0] - p2[0]) * t, p2[1] + (q2[1] - p2[1]) * t];
        for s in 0..=steps {
            let t = s as f32 / steps as f32;
            let (sx, sy) = (pp.0 + (qp.0 - pp.0) * t, pp.1 + (qp.1 - pp.1) * t);
            let z = pz + (qz - pz) * t;
            let (ix, iy) = (sx.round() as isize, sy.round() as isize);
            let vis = if ix < 0 || iy < 0 || ix as usize >= n || iy as usize >= n {
                true
            } else {
                z <= zbuf[iy as usize * n + ix as usize] + bias
            };
            match run_vis {
                None => {
                    run_vis = Some(vis);
                    run_start = s;
                }
                Some(prev) if prev != vis => {
                    let seg = [lerp2(run_start as f32 / steps as f32), lerp2(t)];
                    if prev {
                        visible.push(seg)
                    } else {
                        hidden.push(seg)
                    }
                    run_vis = Some(vis);
                    run_start = s;
                }
                _ => {}
            }
        }
        if let Some(prev) = run_vis {
            if run_start < steps {
                let seg = [lerp2(run_start as f32 / steps as f32), q2];
                if prev {
                    visible.push(seg)
                } else {
                    hidden.push(seg)
                }
            }
        }
    }

    Schematic {
        visible,
        hidden,
        min,
        max,
    }
}

/// [`schematic_project`] using one of the standard [`View`]s.
pub fn schematic_project_view(
    geom: &BufferGeometry,
    view: View,
    opts: &SchematicOptions,
) -> Schematic {
    let (dir, up) = view.basis();
    schematic_project(geom, dir, up, opts)
}

/// Render a [`Schematic`] to SVG — visible solid, hidden dashed — with a
/// `margin` (model units) around the drawing.
pub fn schematic_to_svg(s: &Schematic, stroke: f32) -> schematic_svg::Svg {
    schematic_svg::to_svg(s, stroke, stroke * 8.0)
}

/// Rasterise a [`Schematic`] to 8-bit RGBA at `px` pixels on its longest side:
/// black visible lines, grey dashed hidden lines, white ground. Pairs with
/// [`crate::encode_png`] to write the drawing without a vector round-trip.
///
/// Returns `(width, height, rgba)`.
pub fn schematic_to_rgba(s: &Schematic, px: usize, line_px: f32) -> (u32, u32, Vec<u8>) {
    let margin = line_px * 4.0;
    let size = s.size();
    let span = size[0].max(size[1]).max(1e-6);
    let px = px.max(16);
    let scale = (px as f32 - 2.0 * margin) / span;
    let w = ((size[0] * scale + 2.0 * margin).ceil() as usize).max(1);
    let h = ((size[1] * scale + 2.0 * margin).ceil() as usize).max(1);
    let mut buf = vec![255u8; w * h * 4];

    // Anti-aliased line stamping: cover the segment's neighbourhood and shade
    // by distance to the segment.
    let mut stroke = |a: [f32; 2], b: [f32; 2], shade: [u8; 3], half: f32| {
        let map = |p: [f32; 2]| {
            [
                (p[0] - s.min[0]) * scale + margin,
                (s.max[1] - p[1]) * scale + margin,
            ]
        };
        let (p, q) = (map(a), map(b));
        let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
        let len2 = dx * dx + dy * dy;
        let pad = half + 1.0;
        let x0 = ((p[0].min(q[0]) - pad).floor().max(0.0)) as usize;
        let x1 = ((p[0].max(q[0]) + pad).ceil().min(w as f32 - 1.0)).max(0.0) as usize;
        let y0 = ((p[1].min(q[1]) - pad).floor().max(0.0)) as usize;
        let y1 = ((p[1].max(q[1]) + pad).ceil().min(h as f32 - 1.0)).max(0.0) as usize;
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
                let t = if len2 > 1e-12 {
                    (((fx - p[0]) * dx + (fy - p[1]) * dy) / len2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let d = (fx - (p[0] + dx * t)).hypot(fy - (p[1] + dy * t));
                let cov = (half + 0.5 - d).clamp(0.0, 1.0);
                if cov <= 0.0 {
                    continue;
                }
                let o = (y * w + x) * 4;
                for c in 0..3 {
                    let dstv = buf[o + c] as f32;
                    buf[o + c] = (dstv + (shade[c] as f32 - dstv) * cov).round() as u8;
                }
            }
        }
    };

    // Hidden first so visible lines win where they overlap.
    for e in &s.hidden {
        // dash the segment in view space
        let (a, b) = (e[0], e[1]);
        let len = (b[0] - a[0]).hypot(b[1] - a[1]);
        let dash = (line_px * 5.0 / scale).max(1e-4);
        let n = ((len / dash).ceil() as usize).max(1);
        for i in 0..n {
            let (t0, t1) = (i as f32 / n as f32, (i as f32 + 0.55) / n as f32);
            let p = [a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0];
            let q = [
                a[0] + (b[0] - a[0]) * t1.min(1.0),
                a[1] + (b[1] - a[1]) * t1.min(1.0),
            ];
            stroke(p, q, [150, 150, 150], line_px * 0.5);
        }
    }
    for e in &s.visible {
        stroke(e[0], e[1], [0, 0, 0], line_px * 0.5);
    }
    (w as u32, h as u32, buf)
}

/// SVG emission, kept separate so callers can tune margin/precision.
pub mod schematic_svg {
    use super::Schematic;

    /// An SVG document. `Display`/`Into<String>` give the markup.
    pub type Svg = String;

    pub fn to_svg(s: &Schematic, stroke: f32, margin: f32) -> Svg {
        let w = (s.max[0] - s.min[0]) + margin * 2.0;
        let h = (s.max[1] - s.min[1]) + margin * 2.0;
        let mut out = String::with_capacity(64 * s.len() + 512);
        out.push_str(&format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w:.3}mm\" height=\"{h:.3}mm\" \
             viewBox=\"0 0 {w:.3} {h:.3}\">\n\
             <rect width=\"{w:.3}\" height=\"{h:.3}\" fill=\"white\"/>\n"
        ));
        // SVG y grows downward; flip so the drawing reads the right way up.
        let fx = |x: f32| x - s.min[0] + margin;
        let fy = |y: f32| (s.max[1] - y) + margin;
        let mut layer = |segs: &Vec<[[f32; 2]; 2]>, dash: bool| {
            if segs.is_empty() {
                return;
            }
            out.push_str(&format!(
                "<g stroke=\"black\" stroke-width=\"{stroke:.3}\" stroke-linecap=\"round\" \
                 fill=\"none\"{}>\n",
                if dash {
                    format!(
                        " stroke-dasharray=\"{:.2},{:.2}\" stroke-opacity=\"0.45\"",
                        stroke * 6.0,
                        stroke * 4.0
                    )
                } else {
                    String::new()
                }
            ));
            for e in segs {
                out.push_str(&format!(
                    "<line x1=\"{:.3}\" y1=\"{:.3}\" x2=\"{:.3}\" y2=\"{:.3}\"/>\n",
                    fx(e[0][0]),
                    fy(e[0][1]),
                    fx(e[1][0]),
                    fy(e[1][1])
                ));
            }
            out.push_str("</g>\n");
        };
        layer(&s.hidden, true);
        layer(&s.visible, false);
        out.push_str("</svg>\n");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openscad::cube;

    #[test]
    fn cube_front_view_is_a_square() {
        let g = cube([10.0, 20.0, 30.0]).to_geometry_exact();
        let s = schematic_project_view(&g, View::Front, &SchematicOptions::default());
        let size = s.size();
        assert!((size[0] - 10.0).abs() < 1e-3, "width {}", size[0]);
        assert!((size[1] - 20.0).abs() < 1e-3, "height {}", size[1]);
        // A cube's front view: 4 silhouette edges, no creases facing us.
        assert!(!s.visible.is_empty());
    }

    #[test]
    fn hidden_layer_appears_only_when_requested() {
        let g = cube([10.0, 10.0, 10.0]).to_geometry_exact();
        let plain = schematic_project_view(&g, View::Iso, &SchematicOptions::default());
        assert!(plain.hidden.is_empty());
        let opts = SchematicOptions {
            hidden: true,
            ..Default::default()
        };
        let hlr = schematic_project_view(&g, View::Iso, &opts);
        // An isometric cube hides its three far edges.
        assert!(
            !hlr.hidden.is_empty(),
            "expected hidden edges in an iso cube view"
        );
        assert!(!hlr.visible.is_empty());
    }
}
