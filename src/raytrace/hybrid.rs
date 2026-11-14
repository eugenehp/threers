//! Hybrid raster + path-traced compositing.
//!
//! The raster frame is sharp, full resolution and instant; the path tracer is right about light (soft shadows,
//! inter-reflections, emitters lighting their surroundings) but noisy and, to stay interactive, rendered at a
//! fraction of the resolution. [`HybridCompositor`] puts the two together:
//!
//! 1. **Joint bilateral upsampling** of the traced image with the raster frame as the guide: every output pixel
//!    takes the 4x4 nearest traced pixels, weighted by (Gaussian) distance *and* by how much the raster agrees
//!    with them (log-luminance + chroma), so traced light does not bleed across silhouettes.
//! 2. **Detail transfer**: the traced image is multiplied by the raster's own high-frequency ratio
//!    `raster / lowpass(raster)` (luminance, clamped), restoring texture and edge detail the low-res trace lacks.
//! 3. **Mix** with the raster frame (`mix`, typically ramped up as the trace converges).
//! 4. **Bloom**: soft-knee threshold, a dual-filter (down 2x2, up tent) pyramid, energy added back on top - glow
//!    for emitters in the raster and in the traced image alike (e.g. an LED's reflection in a nearby surface).
//!
//! All maths is in linear light; inputs and output are RGBA8 sRGB, row-major, top row first.

/// Compositing parameters. `Default` is a sensible "hybrid" look.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HybridParams {
    /// Weight of the traced light (0 = raster only, 1 = traced light everywhere).
    pub mix: f32,
    /// Detail transfer amount (0 = plain upsampled trace, 1 = full raster detail).
    pub detail: f32,
    /// Bilateral range sigma in log-luminance units (smaller = sharper edges, more aliasing).
    pub edge_sigma: f32,
    /// Bloom strength added on top (0 = off).
    pub bloom_strength: f32,
    /// Bloom threshold on linear luminance and its soft knee width.
    pub bloom_threshold: f32,
    pub bloom_knee: f32,
    /// Pyramid levels (each halves the resolution; 1..=8). More = wider glow.
    pub bloom_levels: u32,
}

impl Default for HybridParams {
    fn default() -> Self {
        Self {
            mix: 1.0,
            detail: 1.0,
            edge_sigma: 0.35,
            bloom_strength: 0.0,
            bloom_threshold: 0.75,
            bloom_knee: 0.25,
            bloom_levels: 6,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum HybridError {
    Size {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    Empty,
}

impl std::fmt::Display for HybridError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Size {
                what,
                expected,
                got,
            } => write!(f, "{what}: expected {expected} bytes, got {got}"),
            Self::Empty => write!(f, "zero-sized image"),
        }
    }
}
impl std::error::Error for HybridError {}

/// Reusable compositor (keeps its scratch buffers between frames).
#[derive(Default)]
pub struct HybridCompositor {
    pub params: HybridParams,
    lut: Vec<f32>,
}

#[inline]
fn lum(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

#[inline]
fn linear_to_srgb8(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let s = if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u8
}

/// A linear RGB float image.
#[derive(Clone)]
struct Img {
    w: usize,
    h: usize,
    px: Vec<[f32; 3]>,
}

impl Img {
    fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            px: vec![[0.0; 3]; w * h],
        }
    }
    #[inline]
    fn at(&self, x: usize, y: usize) -> [f32; 3] {
        self.px[y * self.w + x]
    }
    /// area-average resample to (tw, th) (exact box filter, any ratio)
    fn box_down(&self, tw: usize, th: usize) -> Img {
        let mut out = Img::new(tw, th);
        let sx = self.w as f32 / tw as f32;
        let sy = self.h as f32 / th as f32;
        for ty in 0..th {
            let y0 = ty as f32 * sy;
            let y1 = y0 + sy;
            for tx in 0..tw {
                let x0 = tx as f32 * sx;
                let x1 = x0 + sx;
                let (mut acc, mut wsum) = ([0.0f32; 3], 0.0f32);
                let mut y = y0.floor() as usize;
                while (y as f32) < y1 && y < self.h {
                    let wy = (y1.min(y as f32 + 1.0) - y0.max(y as f32)).max(0.0);
                    let mut x = x0.floor() as usize;
                    while (x as f32) < x1 && x < self.w {
                        let wx = (x1.min(x as f32 + 1.0) - x0.max(x as f32)).max(0.0);
                        let c = self.at(x, y);
                        let wgt = wx * wy;
                        for k in 0..3 {
                            acc[k] += c[k] * wgt;
                        }
                        wsum += wgt;
                        x += 1;
                    }
                    y += 1;
                }
                let inv = if wsum > 0.0 { 1.0 / wsum } else { 0.0 };
                out.px[ty * tw + tx] = [acc[0] * inv, acc[1] * inv, acc[2] * inv];
            }
        }
        out
    }
    /// 2x2 box halving (odd edges clamp)
    fn half(&self) -> Img {
        let (w, h) = (self.w.div_ceil(2).max(1), self.h.div_ceil(2).max(1));
        let mut out = Img::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let mut acc = [0.0f32; 3];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let c = self.at((2 * x + dx).min(self.w - 1), (2 * y + dy).min(self.h - 1));
                    for k in 0..3 {
                        acc[k] += 0.25 * c[k];
                    }
                }
                out.px[y * w + x] = acc;
            }
        }
        out
    }
    /// bilinear sample at continuous pixel coordinates (centres at +0.5)
    fn bilinear(&self, u: f32, v: f32) -> [f32; 3] {
        let fx = (u - 0.5).clamp(0.0, (self.w - 1) as f32);
        let fy = (v - 0.5).clamp(0.0, (self.h - 1) as f32);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (ax, ay) = (fx - x0 as f32, fy - y0 as f32);
        let mut c = [0.0f32; 3];
        for k in 0..3 {
            let top = self.at(x0, y0)[k] * (1.0 - ax) + self.at(x1, y0)[k] * ax;
            let bot = self.at(x0, y1)[k] * (1.0 - ax) + self.at(x1, y1)[k] * ax;
            c[k] = top * (1.0 - ay) + bot * ay;
        }
        c
    }
    /// tent-filtered 2x upsample of `self` to (w, h), added into `dst`
    fn add_up_into(&self, dst: &mut Img) {
        let (sx, sy) = (self.w as f32 / dst.w as f32, self.h as f32 / dst.h as f32);
        for y in 0..dst.h {
            for x in 0..dst.w {
                let u = (x as f32 + 0.5) * sx;
                let v = (y as f32 + 0.5) * sy;
                // 3x3 tent around the bilinear tap (a softer, rounder kernel than plain bilinear)
                let mut acc = [0.0f32; 3];
                for (dx, dy, wgt) in [
                    (-1.0, -1.0, 1.0),
                    (0.0, -1.0, 2.0),
                    (1.0, -1.0, 1.0),
                    (-1.0, 0.0, 2.0),
                    (0.0, 0.0, 4.0),
                    (1.0, 0.0, 2.0),
                    (-1.0, 1.0, 1.0),
                    (0.0, 1.0, 2.0),
                    (1.0, 1.0, 1.0),
                ] {
                    let c = self.bilinear(u + dx * 0.5, v + dy * 0.5);
                    for k in 0..3 {
                        acc[k] += c[k] * (wgt / 16.0);
                    }
                }
                let d = &mut dst.px[y * dst.w + x];
                for k in 0..3 {
                    d[k] += acc[k];
                }
            }
        }
    }
}

impl HybridCompositor {
    pub fn new(params: HybridParams) -> Self {
        Self {
            params,
            lut: Vec::new(),
        }
    }

    fn decode(&mut self, rgba: &[u8], w: usize, h: usize) -> Img {
        if self.lut.is_empty() {
            self.lut = (0..256)
                .map(|i| {
                    let s = i as f32 / 255.0;
                    if s <= 0.04045 {
                        s / 12.92
                    } else {
                        ((s + 0.055) / 1.055).powf(2.4)
                    }
                })
                .collect();
        }
        let mut img = Img::new(w, h);
        for (i, p) in img.px.iter_mut().enumerate() {
            *p = [
                self.lut[rgba[4 * i] as usize],
                self.lut[rgba[4 * i + 1] as usize],
                self.lut[rgba[4 * i + 2] as usize],
            ];
        }
        img
    }

    fn check(
        what: &'static str,
        buf: &[u8],
        w: u32,
        h: u32,
    ) -> Result<(usize, usize), HybridError> {
        if w == 0 || h == 0 {
            return Err(HybridError::Empty);
        }
        let n = w as usize * h as usize * 4;
        if buf.len() != n {
            return Err(HybridError::Size {
                what,
                expected: n,
                got: buf.len(),
            });
        }
        Ok((w as usize, h as usize))
    }

    /// Composite a (w, h) raster frame with a (tw, th) traced frame -> (w, h) RGBA8. Alpha is the raster's.
    pub fn composite(
        &mut self,
        raster: &[u8],
        w: u32,
        h: u32,
        trace: &[u8],
        tw: u32,
        th: u32,
    ) -> Result<Vec<u8>, HybridError> {
        let (w, h) = Self::check("raster", raster, w, h)?;
        let (tw, th) = Self::check("trace", trace, tw, th)?;
        let p = self.params;
        let r = self.decode(raster, w, h);
        let t = self.decode(trace, tw, th);
        let mut out = if p.mix <= 0.0 {
            r.clone()
        } else {
            let rl = r.box_down(tw, th); // the raster as the tracer "sees" it
            let inv2s2 = 1.0 / (2.0 * p.edge_sigma.max(1e-3).powi(2));
            let llog = |c: [f32; 3]| (lum(c) + 1e-3).ln();
            let (sx, sy) = (tw as f32 / w as f32, th as f32 / h as f32);
            let mut o = Img::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    let g = r.at(x, y);
                    let gl = llog(g);
                    let gs = lum(g) + 1e-3;
                    let gch = [g[0] / gs, g[1] / gs, g[2] / gs];
                    let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, (tw - 1) as f32);
                    let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, (th - 1) as f32);
                    let (x0, y0) = (fx.floor() as isize, fy.floor() as isize);
                    let (mut tc, mut rc, mut ws) = ([0.0f32; 3], [0.0f32; 3], 0.0f32);
                    let (mut tb, mut rb, mut wbs) = ([0.0f32; 3], [0.0f32; 3], 0.0f32); // spatial-only fallback
                                                                                        // 4x4 window: a pixel at an edge can reach a coarse pixel that is wholly on its own side
                    for dy in -1..=2isize {
                        for dx in -1..=2isize {
                            let (xx, yy) = (
                                (x0 + dx).clamp(0, tw as isize - 1) as usize,
                                (y0 + dy).clamp(0, th as isize - 1) as usize,
                            );
                            let (ddx, ddy) = ((x0 + dx) as f32 - fx, (y0 + dy) as f32 - fy);
                            let wb = (-(ddx * ddx + ddy * ddy) * (1.0 / (2.0 * 0.6 * 0.6))).exp();
                            let n = rl.at(xx, yy);
                            let ns = lum(n) + 1e-3;
                            let dch = (0..3).map(|k| (n[k] / ns - gch[k]).abs()).sum::<f32>();
                            let d = (llog(n) - gl).abs() + 0.5 * dch;
                            let wr = wb * (-d * d * inv2s2).exp();
                            let tv = t.at(xx, yy);
                            for k in 0..3 {
                                tc[k] += tv[k] * wr;
                                rc[k] += n[k] * wr;
                                tb[k] += tv[k] * wb;
                                rb[k] += n[k] * wb;
                            }
                            ws += wr;
                            wbs += wb;
                        }
                    }
                    let (tb, rb) = (
                        [tb[0] / wbs, tb[1] / wbs, tb[2] / wbs],
                        [rb[0] / wbs, rb[1] / wbs, rb[2] / wbs],
                    );
                    let (tu, ru) = if ws > 1e-6 {
                        (
                            [tc[0] / ws, tc[1] / ws, tc[2] / ws],
                            [rc[0] / ws, rc[1] / ws, rc[2] / ws],
                        )
                    } else {
                        (tb, rb) // no neighbour agrees: thin feature
                    };
                    // detail transfer: the raster's own ratio to its low-pass, per channel - equivalently the raster
                    // relit by traced / raster-low-res light, so coloured traced light (an LED's tint) still shows
                    let px = &mut o.px[y * w + x];
                    for k in 0..3 {
                        let ratio = ((g[k] + 2e-3) / (ru[k] + 2e-3)).clamp(0.2, 5.0);
                        let c = tu[k] * (1.0 + p.detail * (ratio - 1.0));
                        px[k] = g[k] + p.mix.min(1.0) * (c - g[k]);
                    }
                }
            }
            o
        };
        if p.bloom_strength > 0.0 {
            Self::bloom(&mut out, &p);
        }
        let mut bytes = vec![0u8; w * h * 4];
        for (i, c) in out.px.iter().enumerate() {
            bytes[4 * i] = linear_to_srgb8(c[0]);
            bytes[4 * i + 1] = linear_to_srgb8(c[1]);
            bytes[4 * i + 2] = linear_to_srgb8(c[2]);
            bytes[4 * i + 3] = raster[4 * i + 3];
        }
        Ok(bytes)
    }

    /// Bloom only (for the raster path): RGBA8 in -> RGBA8 out.
    pub fn bloom_rgba(&mut self, rgba: &[u8], w: u32, h: u32) -> Result<Vec<u8>, HybridError> {
        let (wu, hu) = Self::check("image", rgba, w, h)?;
        let mut img = self.decode(rgba, wu, hu);
        let p = self.params;
        if p.bloom_strength > 0.0 {
            Self::bloom(&mut img, &p);
        }
        let mut out = rgba.to_vec();
        for (i, c) in img.px.iter().enumerate() {
            out[4 * i] = linear_to_srgb8(c[0]);
            out[4 * i + 1] = linear_to_srgb8(c[1]);
            out[4 * i + 2] = linear_to_srgb8(c[2]);
        }
        Ok(out)
    }

    fn bloom(img: &mut Img, p: &HybridParams) {
        // soft-knee prefilter (Karis): keeps the part of each pixel above threshold, smoothly
        let (t, k) = (p.bloom_threshold, p.bloom_knee.max(1e-4));
        let mut bright = Img::new(img.w, img.h);
        for (b, c) in bright.px.iter_mut().zip(&img.px) {
            let l = lum(*c).max(1e-6);
            let soft = ((l - t + k).clamp(0.0, 2.0 * k)).powi(2) / (4.0 * k);
            let f = (soft.max(l - t)).max(0.0) / l;
            *b = [c[0] * f, c[1] * f, c[2] * f];
        }
        // down pyramid
        let levels = p.bloom_levels.clamp(1, 8) as usize;
        let mut pyr = vec![bright.half()];
        while pyr.len() < levels && pyr.last().map_or(false, |m| m.w > 1 || m.h > 1) {
            let n = pyr.last().unwrap().half();
            pyr.push(n);
        }
        // up: each level adds the next coarser one (tent), so wide and tight glow both survive
        for i in (1..pyr.len()).rev() {
            let (lo, hi) = pyr.split_at_mut(i);
            hi[0].add_up_into(&mut lo[i - 1]);
        }
        let mut glow = Img::new(img.w, img.h);
        pyr[0].add_up_into(&mut glow);
        let norm = p.bloom_strength / pyr.len() as f32; // the sum over levels -> stable brightness vs levels
        for (c, g) in img.px.iter_mut().zip(&glow.px) {
            for k in 0..3 {
                c[k] += g[k] * norm;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: usize, h: usize, f: impl Fn(usize, usize) -> [u8; 3]) -> Vec<u8> {
        let mut v = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let c = f(x, y);
                v[4 * (y * w + x)..4 * (y * w + x) + 4].copy_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        v
    }
    fn g(v: &[u8], w: usize, x: usize, y: usize) -> u8 {
        v[4 * (y * w + x) + 1]
    }

    #[test]
    fn mix_zero_returns_the_raster() {
        let r = img(16, 12, |x, y| [(x * 13) as u8, (y * 17) as u8, 90]);
        let t = img(4, 3, |_, _| [0, 0, 0]);
        let mut c = HybridCompositor::new(HybridParams {
            mix: 0.0,
            ..Default::default()
        });
        assert_eq!(c.composite(&r, 16, 12, &t, 4, 3).unwrap(), r);
    }

    #[test]
    fn a_trace_that_agrees_with_the_raster_changes_nothing() {
        // the trace is exactly the raster at quarter resolution: the composite must give the raster back
        let (w, h) = (32, 24);
        let r = img(w, h, |x, y| {
            [
                (40 + x * 5) as u8,
                (60 + y * 6) as u8,
                (100 + ((x + y) % 3) * 30) as u8,
            ]
        });
        let mut c = HybridCompositor::new(HybridParams::default());
        let lin = c.decode(&r, w, h).box_down(8, 6);
        let t: Vec<u8> = lin
            .px
            .iter()
            .flat_map(|p| {
                [
                    linear_to_srgb8(p[0]),
                    linear_to_srgb8(p[1]),
                    linear_to_srgb8(p[2]),
                    255,
                ]
            })
            .collect();
        let o = c.composite(&r, w as u32, h as u32, &t, 8, 6).unwrap();
        let worst = o
            .iter()
            .zip(&r)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap();
        assert!(worst <= 3, "max deviation {worst}");
    }

    #[test]
    fn traced_shadow_is_applied_and_raster_texture_survives() {
        // raster: uniform grey with a fine 1-px checker; trace: left half in shadow (quarter brightness)
        let (w, h) = (64, 32);
        let r = img(
            w,
            h,
            |x, y| if (x + y) % 2 == 0 { [200; 3] } else { [150; 3] },
        );
        let t = img(16, 8, |x, _| if x < 8 { [90; 3] } else { [178; 3] });
        let mut c = HybridCompositor::new(HybridParams::default());
        let o = c.composite(&r, w as u32, h as u32, &t, 16, 8).unwrap();
        let (l0, l1) = (g(&o, w, 10, 16), g(&o, w, 11, 16));
        let (r0, r1) = (g(&o, w, 50, 16), g(&o, w, 51, 16));
        assert!(
            l0.max(l1) < r0.min(r1),
            "shadow side must be darker: {l0} {l1} vs {r0} {r1}"
        );
        assert!(
            l0.abs_diff(l1) >= 15 && r0.abs_diff(r1) >= 20,
            "checker detail lost: {l0} {l1} / {r0} {r1}"
        );
    }

    #[test]
    fn traced_light_does_not_bleed_across_a_raster_edge() {
        // raster: black left / white right with the edge mid-cell of the coarse trace; trace: lit white side
        let (w, h) = (64, 16);
        let r = img(w, h, |x, _| if x < 30 { [5; 3] } else { [230; 3] });
        let t = img(8, 2, |x, _| if x < 4 { [5; 3] } else { [230; 3] });
        let mut c = HybridCompositor::new(HybridParams {
            detail: 0.0,
            ..Default::default()
        });
        let o = c.composite(&r, w as u32, h as u32, &t, 8, 2).unwrap();
        assert!(
            g(&o, w, 30, 8) > 180,
            "white side next to the edge went dark: {}",
            g(&o, w, 30, 8)
        );
        assert!(
            g(&o, w, 29, 8) < 40,
            "black side next to the edge picked up light: {}",
            g(&o, w, 29, 8)
        );
    }

    #[test]
    fn bloom_spreads_a_bright_pixel_and_is_off_at_zero_strength() {
        let (w, h) = (33, 33);
        let r = img(w, h, |x, y| {
            if x == 16 && y == 16 {
                [255; 3]
            } else {
                [10; 3]
            }
        });
        let mut off = HybridCompositor::new(HybridParams {
            mix: 0.0,
            bloom_strength: 0.0,
            ..Default::default()
        });
        assert_eq!(off.bloom_rgba(&r, 33, 33).unwrap(), r);
        let mut on = HybridCompositor::new(HybridParams {
            mix: 0.0,
            bloom_strength: 4.0,
            bloom_threshold: 0.5,
            ..Default::default()
        });
        let o = on.bloom_rgba(&r, 33, 33).unwrap();
        assert!(g(&o, w, 18, 16) > g(&r, w, 18, 16) + 3, "no glow 2 px away");
        assert!(g(&o, w, 16, 22) > g(&r, w, 16, 22), "no glow 6 px away");
        assert!(
            g(&o, w, 18, 16) >= g(&o, w, 22, 16),
            "glow must fall off with distance"
        );
        assert_eq!(g(&o, w, 0, 0), g(&r, w, 0, 0).max(g(&o, w, 0, 0))); // never darkens
    }

    #[test]
    fn dim_pixels_below_the_threshold_do_not_glow() {
        let r = img(16, 16, |_, _| [120; 3]); // linear ~0.19, far below the 0.75 threshold
        let mut c = HybridCompositor::new(HybridParams {
            bloom_strength: 3.0,
            ..Default::default()
        });
        assert_eq!(c.bloom_rgba(&r, 16, 16).unwrap(), r);
    }

    #[test]
    fn size_errors_are_reported() {
        let mut c = HybridCompositor::default();
        assert_eq!(
            c.composite(&[0; 8], 2, 2, &[0; 4], 1, 1),
            Err(HybridError::Size {
                what: "raster",
                expected: 16,
                got: 8
            })
        );
        assert_eq!(
            c.composite(&[], 0, 2, &[0; 4], 1, 1),
            Err(HybridError::Empty)
        );
    }
}
