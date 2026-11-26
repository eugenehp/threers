//! Outlines drawn without the system: flattened into contours and filled
//! into coverage (non-zero, anti-aliased, by accumulated signed area as
//! font-rs does). For an app that draws text from the faces it shaped with.
//!
//! ```no_run
//! use threers_font::{raster, Face, Glyph, Size};
//! # let data = std::fs::read("/System/Library/Fonts/Supplemental/Georgia.ttf").unwrap();
//! let face = Face::new(data, 0).unwrap();
//! let size = Size::char_size(&face, 0, 32 * 64, 72, 72); // 32 pixels an em
//! let g = Glyph::load(&face, &size, 36).unwrap();
//! let mask = raster::Mask::of(&raster::contours(&g.outline, 1.0 / 64.0, 0.0), 0.0, 0.0);
//! // mask.alpha: mask.width × mask.height, placed at (mask.left, mask.top) from the pen
//! ```

use crate::glyph::{Outline, Segment};

/// Closed contours: points, y down.
pub type Contours = Vec<Vec<(f32, f32)>>;

/// An outline as contours, y turned down: each point `scale` of a unit
/// (`1.0 / 64.0` for a glyph's 26.6 pixels), `slant` across per unit up
/// (a made-up italic), curves cut into segments a quarter pixel or so from
/// the curve.
pub fn contours(outline: &Outline, scale: f32, slant: f32) -> Contours {
    let Some(segments) = outline.decompose() else { return Vec::new() };
    let pt = |p: (i64, i64)| {
        let (u, v) = (p.0 as f32 * scale, p.1 as f32 * scale);
        (u + v * slant, -v)
    };
    let mut out: Contours = Vec::new();
    let mut at = (0.0, 0.0);
    // enough pieces for a quarter pixel's error at the curve's size
    let steps = |len: f32| ((len / 2.0).sqrt().ceil() as usize).clamp(2, 24);
    for s in segments {
        match s {
            Segment::MoveTo(p) => {
                at = pt(p);
                out.push(vec![at]);
            }
            Segment::LineTo(p) => {
                at = pt(p);
                if let Some(c) = out.last_mut() {
                    c.push(at);
                }
            }
            Segment::ConicTo(c, p) => {
                let (c, p) = (pt(c), pt(p));
                let n = steps((c.0 - at.0).abs() + (c.1 - at.1).abs() + (p.0 - c.0).abs() + (p.1 - c.1).abs());
                let c0 = at;
                if let Some(cur) = out.last_mut() {
                    for i in 1..=n {
                        let t = i as f32 / n as f32;
                        let u = 1.0 - t;
                        cur.push((u * u * c0.0 + 2.0 * u * t * c.0 + t * t * p.0, u * u * c0.1 + 2.0 * u * t * c.1 + t * t * p.1));
                    }
                }
                at = p;
            }
            Segment::CubicTo(c1, c2, p) => {
                let (c1, c2, p) = (pt(c1), pt(c2), pt(p));
                let n = steps((c1.0 - at.0).abs() + (c1.1 - at.1).abs() + (c2.0 - c1.0).abs() + (c2.1 - c1.1).abs() + (p.0 - c2.0).abs() + (p.1 - c2.1).abs());
                let c0 = at;
                if let Some(cur) = out.last_mut() {
                    for i in 1..=n {
                        let t = i as f32 / n as f32;
                        let u = 1.0 - t;
                        let (a, b, c, d) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
                        cur.push((a * c0.0 + b * c1.0 + c * c2.0 + d * p.0, a * c0.1 + b * c1.1 + c * c2.1 + d * p.1));
                    }
                }
                at = p;
            }
        }
    }
    out
}

/// Coverage, 0–255, `width × height` from `(left, top)`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Mask {
    pub left: i32,
    pub top: i32,
    pub width: usize,
    pub height: usize,
    pub alpha: Vec<u8>,
}

impl Mask {
    /// Contours filled, moved by `(dx, dy)` first (a pen's fraction of a
    /// pixel), in a box just round them.
    pub fn of(contours: &[Vec<(f32, f32)>], dx: f32, dy: f32) -> Mask {
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for &(x, y) in contours.iter().flatten() {
            x0 = x0.min(x + dx);
            y0 = y0.min(y + dy);
            x1 = x1.max(x + dx);
            y1 = y1.max(y + dy);
        }
        if x0 > x1 {
            return Mask::default();
        }
        let left = x0.floor() as i32 - 1;
        let top = y0.floor() as i32 - 1;
        let width = (x1.ceil() as i32 - left + 2) as usize;
        let height = (y1.ceil() as i32 - top + 1) as usize;
        let moved: Contours = contours.iter().map(|c| c.iter().map(|&(x, y)| (x + dx, y + dy)).collect()).collect();
        let mut acc = Vec::new();
        accumulate(&moved, left as f32, top as f32, width, height, &mut acc);
        let mut alpha = vec![0u8; width * height];
        for (row, out) in alpha.chunks_exact_mut(width).enumerate() {
            let mut sum = 0.0f32;
            for (col, a) in out.iter_mut().enumerate() {
                sum += acc[row * (width + 2) + col];
                *a = (sum.abs().min(1.0) * 255.0).round() as u8;
            }
        }
        Mask { left, top, width, height, alpha }
    }
}

/// The signed area each pixel's left edge gains from `contours` (moved by
/// `-ox, -oy`), in rows of `w + 2` of `acc`: summed across a row, a pixel's
/// coverage (its absolute value, at most 1). The fill under any canvas's
/// paths (see [`Mask::of`]).
pub fn accumulate(contours: &[Vec<(f32, f32)>], ox: f32, oy: f32, w: usize, h: usize, acc: &mut Vec<f32>) {
    let stride = w + 2;
    acc.clear();
    acc.resize(stride * h + 2, 0.0);
    for c in contours {
        for i in 0..c.len() {
            let (x0, y0) = c[i];
            let (x1, y1) = c[(i + 1) % c.len()];
            line(acc, stride, h, x0 - ox, y0 - oy, x1 - ox, y1 - oy);
        }
    }
}

fn line(acc: &mut [f32], stride: usize, h: usize, x0: f32, y0: f32, x1: f32, y1: f32) {
    if (y0 - y1).abs() < f32::EPSILON {
        return;
    }
    let (dir, x0, y0, x1, y1) = if y0 < y1 { (1.0, x0, y0, x1, y1) } else { (-1.0, x1, y1, x0, y0) };
    let dxdy = (x1 - x0) / (y1 - y0);
    let y0c = y0.max(0.0);
    let y1c = y1.min(h as f32);
    if y0c >= y1c {
        return;
    }
    let mut x = x0 + (y0c - y0) * dxdy;
    let start = y0c as usize;
    let end = (y1c.ceil() as usize).min(h);
    for row in start..end {
        let base = row * stride;
        let dy = ((row + 1) as f32).min(y1c) - (row as f32).max(y0c);
        let xnext = x + dxdy * dy;
        let d = dy * dir;
        let (xa, xb) = if x < xnext { (x, xnext) } else { (xnext, x) };
        let xa = xa.max(0.0);
        let xb = xb.max(0.0);
        let xa_floor = xa.floor();
        let xai = xa_floor as usize;
        let xb_ceil = xb.ceil();
        let xbi = xb_ceil as usize;
        if xbi <= xai + 1 {
            let xmf = 0.5 * (x + xnext) - xa_floor;
            // past the row's end: nothing (not the next row's start, which
            // undid that row's left edge: a shape running off the right
            // filled nothing)
            if xai < stride && base + xai < acc.len() {
                acc[base + xai] += d - d * xmf.clamp(0.0, 1.0);
            }
            if xai + 1 < stride && base + xai + 1 < acc.len() {
                acc[base + xai + 1] += d * xmf.clamp(0.0, 1.0);
            }
        } else {
            let s = (xb - xa).recip();
            let x0f = xa - xa_floor;
            let a0 = 0.5 * s * (1.0 - x0f) * (1.0 - x0f);
            let x1f = xb - xb_ceil + 1.0;
            let am = 0.5 * s * x1f * x1f;
            let at = |acc: &mut [f32], i: usize, v: f32| {
                if base + i < acc.len() && i < stride {
                    acc[base + i] += v;
                }
            };
            at(acc, xai, d * a0);
            if xbi == xai + 2 {
                at(acc, xai + 1, d * (1.0 - a0 - am));
            } else {
                let a1 = s * (1.5 - x0f);
                at(acc, xai + 1, d * (a1 - a0));
                for xi in xai + 2..xbi - 1 {
                    at(acc, xi, d * s);
                }
                let a2 = a1 + (xbi - xai - 3) as f32 * s;
                at(acc, xbi - 1, d * (1.0 - a2 - am));
            }
            at(acc, xbi, d * am);
        }
        x = xnext;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_square_with_a_hole_is_covered_round_it() {
        let outer = vec![(2.0, 2.0), (18.0, 2.0), (18.0, 18.0), (2.0, 18.0)];
        let inner = vec![(6.0, 6.0), (6.0, 14.0), (14.0, 14.0), (14.0, 6.0)];
        let m = Mask::of(&vec![outer, inner], 0.0, 0.0);
        let at = |x: i32, y: i32| m.alpha[((y - m.top) as usize) * m.width + (x - m.left) as usize];
        assert_eq!(at(3, 3), 255);
        assert_eq!(at(10, 10), 0);
        // an edge half across a pixel: half covered
        let half = Mask::of(&vec![vec![(0.5, 0.0), (4.0, 0.0), (4.0, 4.0), (0.5, 4.0)]], 0.0, 0.0);
        assert_eq!(half.alpha[(1 - half.top) as usize * half.width + (0 - half.left) as usize], 128);
    }
}

#[cfg(test)]
mod past_the_edge {
    use super::accumulate;

    /// A rectangle running off the right of the area filled is filled to
    /// its edge, every row (its right edge, past the end, was added to the
    /// next row's start).
    #[test]
    fn a_shape_off_the_right_still_fills() {
        let (w, h) = (10usize, 4usize);
        let rect = vec![(2.0, 0.0), (30.0, 0.0), (30.0, 4.0), (2.0, 4.0)];
        let mut acc = Vec::new();
        accumulate(&[rect], 0.0, 0.0, w, h, &mut acc);
        for row in 0..h {
            let mut sum = 0.0f32;
            for col in 0..w {
                sum += acc[row * (w + 2) + col];
                let want = if col >= 2 { 1.0 } else { 0.0 };
                assert!((sum.abs() - want).abs() < 1e-3, "row {row} col {col}: {sum}");
            }
        }
    }
}
