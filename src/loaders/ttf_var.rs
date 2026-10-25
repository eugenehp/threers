//! OpenType font variations for [`TtfFont`](super::TtfFont): the `fvar` axes
//! and named instances, `avar` remapping, `gvar` outline deltas, `HVAR`
//! advance deltas, and the item variation store those and `GDEF` share.
//!
//! A variable font's `glyf` table holds one instance — the default — and every
//! other weight or width is that outline moved by deltas. Reading only `glyf`
//! is not "a reasonable approximation" of the font: Apple's SF Compact ships
//! its *Black* as the default, so a reader that ignores `gvar` draws every
//! caption in the heaviest weight the family has.

use crate::math::Vector2;

/// One axis of a variable font, in its own user units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VariationAxis {
    /// `wght`, `wdth`, `opsz`, …
    pub tag: [u8; 4],
    pub min: f32,
    pub default: f32,
    pub max: f32,
}

/// A named instance: a point in the design space the font gives a name to.
#[derive(Debug, Clone, PartialEq)]
pub struct NamedInstance {
    /// The subfamily name from the `name` table ("Regular", "Bold", …), empty
    /// when the font names it in a way this reader does not decode.
    pub name: String,
    /// One user-unit coordinate per axis, in [`VariationAxis`] order.
    pub coords: Vec<f32>,
}

fn u16_at(b: &[u8], p: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*b.get(p)?, *b.get(p + 1)?]))
}
fn i16_at(b: &[u8], p: usize) -> Option<i16> {
    u16_at(b, p).map(|v| v as i16)
}
fn u32_at(b: &[u8], p: usize) -> Option<u32> {
    Some(u32::from_be_bytes([*b.get(p)?, *b.get(p + 1)?, *b.get(p + 2)?, *b.get(p + 3)?]))
}
fn fixed_at(b: &[u8], p: usize) -> Option<f32> {
    u32_at(b, p).map(|v| v as i32 as f32 / 65536.0)
}
fn f2dot14_at(b: &[u8], p: usize) -> Option<f32> {
    i16_at(b, p).map(|v| v as f32 / 16384.0)
}

/// `fvar`: the axes, and the named instances with their `name`-table names.
pub(crate) fn parse_fvar(
    bytes: &[u8],
    fvar: usize,
    name: Option<usize>,
) -> Option<(Vec<VariationAxis>, Vec<NamedInstance>)> {
    let axes_off = fvar + u16_at(bytes, fvar + 4)? as usize;
    let axis_count = u16_at(bytes, fvar + 8)? as usize;
    let axis_size = u16_at(bytes, fvar + 10)? as usize;
    let instance_count = u16_at(bytes, fvar + 12)? as usize;
    let instance_size = u16_at(bytes, fvar + 14)? as usize;
    if axis_size < 20 || instance_size < 4 + 4 * axis_count {
        return None;
    }
    let mut axes = Vec::with_capacity(axis_count);
    for i in 0..axis_count {
        let p = axes_off + i * axis_size;
        let tag = bytes.get(p..p + 4)?.try_into().ok()?;
        axes.push(VariationAxis {
            tag,
            min: fixed_at(bytes, p + 4)?,
            default: fixed_at(bytes, p + 8)?,
            max: fixed_at(bytes, p + 12)?,
        });
    }
    let names = name.map(|n| parse_names(bytes, n)).unwrap_or_default();
    let mut instances = Vec::with_capacity(instance_count);
    let mut p = axes_off + axis_count * axis_size;
    for _ in 0..instance_count {
        let name_id = u16_at(bytes, p)?;
        let coords = (0..axis_count)
            .map(|a| fixed_at(bytes, p + 4 + 4 * a))
            .collect::<Option<Vec<_>>>()?;
        let name = names.get(&name_id).cloned().unwrap_or_default();
        instances.push(NamedInstance { name, coords });
        p += instance_size;
    }
    Some((axes, instances))
}

/// The `name` table's Windows Unicode (English) strings, by name id.
fn parse_names(bytes: &[u8], off: usize) -> std::collections::HashMap<u16, String> {
    let mut out = std::collections::HashMap::new();
    let (Some(count), Some(storage)) = (u16_at(bytes, off + 2), u16_at(bytes, off + 4)) else {
        return out;
    };
    for i in 0..count as usize {
        let r = off + 6 + i * 12;
        let (Some(platform), Some(language), Some(id), Some(len), Some(at)) = (
            u16_at(bytes, r),
            u16_at(bytes, r + 4),
            u16_at(bytes, r + 6),
            u16_at(bytes, r + 8),
            u16_at(bytes, r + 10),
        ) else {
            break;
        };
        // Windows, US English; UTF-16BE.
        if platform != 3 || language != 0x0409 {
            continue;
        }
        let start = off + storage as usize + at as usize;
        let Some(raw) = bytes.get(start..start + len as usize) else { continue };
        let units: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_be_bytes([c[0], c[1]])).collect();
        out.entry(id).or_insert_with(|| String::from_utf16_lossy(&units));
    }
    out
}

/// The `wght` coordinate for a CSS font weight (100–900), `None` without a
/// weight axis.
///
/// Where the font names its instances the usual way ("Thin" … "Black"), the
/// weight goes through them: SF puts its Regular at 457.9 and its Bold at 790
/// on its own scale, which CSS `400` and `700` must land on to look like the
/// system's text. Elsewhere it is the weight itself, which is what `wght` is
/// defined to be.
pub fn wght_for_css_weight(axes: &[VariationAxis], instances: &[NamedInstance], weight: f32) -> Option<f32> {
    let axis = axes.iter().position(|a| &a.tag == b"wght")?;
    let a = axes[axis];
    // CSS's names. Apple's SF calls its lightest "Ultralight" and the next
    // "Thin", the other way round; the ordering filter below drops
    // whichever disagrees with the weights.
    const NAMES: [(&str, f32); 11] = [
        ("Thin", 100.0),
        ("Ultralight", 200.0),
        ("UltraLight", 200.0),
        ("ExtraLight", 200.0),
        ("Light", 300.0),
        ("Regular", 400.0),
        ("Medium", 500.0),
        ("Semibold", 600.0),
        ("SemiBold", 600.0),
        ("Bold", 700.0),
        ("Heavy", 800.0),
    ];
    // (css, wght) for every instance with a weight name and every other
    // axis at its default — the upright, default-width ones.
    let mut points: Vec<(f32, f32)> = instances
        .iter()
        .filter(|i| {
            i.coords.iter().enumerate().all(|(k, &c)| k == axis || c == axes[k].default)
        })
        .filter_map(|i| {
            // An italic family names them "Bold Italic", and its regular
            // just "Italic".
            let name = i.name.trim_end_matches(" Italic").trim_end_matches(" Oblique");
            let name = if matches!(name, "Italic" | "Oblique") { "Regular" } else { name };
            let css = if name == "Black" {
                900.0
            } else {
                NAMES.iter().find(|(n, _)| *n == name)?.1
            };
            Some((css, i.coords[axis]))
        })
        .collect();
    points.sort_by(|p, q| p.0.total_cmp(&q.0));
    points.dedup_by(|p, q| p.0 == q.0);
    let mut kept: Vec<(f32, f32)> = Vec::with_capacity(points.len());
    for p in points {
        if kept.last().is_none_or(|q| p.1 > q.1) {
            kept.push(p);
        }
    }
    let points = kept;
    let w = weight.clamp(1.0, 1000.0);
    let v = if points.len() >= 2 {
        let i = points.windows(2).position(|s| w <= s[1].0).unwrap_or(points.len() - 2);
        let (p, q) = (points[i], points[i + 1]);
        p.1 + (w - p.0) / (q.0 - p.0) * (q.1 - p.1)
    } else {
        w
    };
    Some(v.clamp(a.min, a.max))
}

/// User coordinates to normalized ones in `-1..=1`, with `avar` applied.
///
/// Axes `user` does not mention sit at their default.
pub(crate) fn normalize(
    bytes: &[u8],
    axes: &[VariationAxis],
    avar: Option<usize>,
    user: &[([u8; 4], f32)],
) -> Vec<f32> {
    let mut coords: Vec<f32> = axes
        .iter()
        .map(|a| {
            let v = user
                .iter()
                .rev()
                .find(|(t, _)| *t == a.tag)
                .map_or(a.default, |&(_, v)| v)
                .clamp(a.min, a.max);
            if v < a.default && a.default > a.min {
                (v - a.default) / (a.default - a.min)
            } else if v > a.default && a.max > a.default {
                (v - a.default) / (a.max - a.default)
            } else {
                0.0
            }
        })
        .collect();
    if let Some(avar) = avar {
        apply_avar(bytes, avar, &mut coords);
    }
    coords
}

/// `avar` version 1: a piecewise-linear remap of each normalized axis.
fn apply_avar(bytes: &[u8], off: usize, coords: &mut [f32]) {
    let Some(axis_count) = u16_at(bytes, off + 6) else { return };
    let mut p = off + 8;
    for (a, c) in coords.iter_mut().enumerate() {
        if a >= axis_count as usize {
            break;
        }
        let Some(n) = u16_at(bytes, p) else { return };
        let map: Option<Vec<(f32, f32)>> = (0..n as usize)
            .map(|i| Some((f2dot14_at(bytes, p + 2 + 4 * i)?, f2dot14_at(bytes, p + 4 + 4 * i)?)))
            .collect();
        p += 2 + 4 * n as usize;
        let Some(map) = map else { return };
        if map.len() < 2 {
            continue;
        }
        let v = *c;
        if let Some(i) = map.windows(2).position(|w| v >= w[0].0 && v <= w[1].0) {
            let (a, b) = (map[i], map[i + 1]);
            *c = if b.0 > a.0 { a.1 + (v - a.0) / (b.0 - a.0) * (b.1 - a.1) } else { a.1 };
        }
    }
}

/// How much a region with this `peak` (and `start`..`end`) applies at
/// `coord`, on one axis — 1 at the peak, falling to 0 at the region's ends.
fn axis_scalar(coord: f32, start: f32, peak: f32, end: f32) -> f32 {
    if peak == 0.0 || coord == peak {
        return 1.0;
    }
    if start > peak || peak > end || (start < 0.0 && end > 0.0) {
        return 1.0;
    }
    if coord <= start || coord >= end {
        return 0.0;
    }
    if coord < peak {
        (coord - start) / (peak - start)
    } else {
        (end - coord) / (end - peak)
    }
}

/// The `gvar` table, ready to vary glyphs at one location.
pub(crate) struct Gvar<'a> {
    bytes: &'a [u8],
    off: usize,
    axis_count: usize,
    shared_tuples: usize,
    shared_count: usize,
    glyph_count: usize,
    long_offsets: bool,
    data_off: usize,
    coords: Vec<f32>,
}

impl<'a> Gvar<'a> {
    pub(crate) fn new(bytes: &'a [u8], off: usize, coords: Vec<f32>) -> Option<Self> {
        let axis_count = u16_at(bytes, off + 4)? as usize;
        if axis_count != coords.len() {
            return None;
        }
        Some(Self {
            bytes,
            off,
            axis_count,
            shared_count: u16_at(bytes, off + 6)? as usize,
            shared_tuples: off + u32_at(bytes, off + 8)? as usize,
            glyph_count: u16_at(bytes, off + 12)? as usize,
            long_offsets: u16_at(bytes, off + 14)? & 1 != 0,
            data_off: off + u32_at(bytes, off + 16)? as usize,
            coords,
        })
    }

    /// Whether every axis sits at its default — nothing to vary.
    pub(crate) fn is_default(&self) -> bool {
        self.coords.iter().all(|&c| c == 0.0)
    }

    fn glyph_data(&self, gid: usize) -> Option<(usize, usize)> {
        if gid >= self.glyph_count {
            return None;
        }
        let at = |i: usize| -> Option<usize> {
            if self.long_offsets {
                u32_at(self.bytes, self.off + 20 + 4 * i).map(|v| v as usize)
            } else {
                u16_at(self.bytes, self.off + 20 + 2 * i).map(|v| v as usize * 2)
            }
        };
        let (a, b) = (at(gid)?, at(gid + 1)?);
        (b > a).then_some((self.data_off + a, self.data_off + b))
    }

    /// The summed deltas for glyph `gid`'s `n` points (its outline points,
    /// or one per component for a composite, then the four phantom points).
    ///
    /// `contour_ends` drives the interpolation of points a tuple leaves
    /// untouched; a composite passes `None`, whose untouched points stay put.
    pub(crate) fn deltas(
        &self,
        gid: usize,
        original: &[Vector2],
        contour_ends: Option<&[usize]>,
    ) -> Option<Vec<Vector2>> {
        if self.is_default() {
            return None;
        }
        let (start, end) = self.glyph_data(gid)?;
        let b = &self.bytes[..end.min(self.bytes.len())];
        let n = original.len();
        let count_word = u16_at(b, start)?;
        let tuple_count = (count_word & 0x0FFF) as usize;
        let mut data = start + u16_at(b, start + 2)? as usize;
        let shared_points = if count_word & 0x8000 != 0 {
            let (pts, next) = read_points(b, data)?;
            data = next;
            Some(pts)
        } else {
            None
        };
        let mut total = vec![Vector2::new(0.0, 0.0); n];
        let mut header = start + 4;
        for _ in 0..tuple_count {
            let size = u16_at(b, header)? as usize;
            let index = u16_at(b, header + 2)?;
            header += 4;
            let peak: Vec<f32> = if index & 0x8000 != 0 {
                let v = (0..self.axis_count)
                    .map(|a| f2dot14_at(b, header + 2 * a))
                    .collect::<Option<Vec<_>>>()?;
                header += 2 * self.axis_count;
                v
            } else {
                let i = (index & 0x0FFF) as usize;
                if i >= self.shared_count {
                    return None;
                }
                (0..self.axis_count)
                    .map(|a| f2dot14_at(self.bytes, self.shared_tuples + 2 * (i * self.axis_count + a)))
                    .collect::<Option<Vec<_>>>()?
            };
            let intermediate = if index & 0x4000 != 0 {
                let s = (0..self.axis_count)
                    .map(|a| f2dot14_at(b, header + 2 * a))
                    .collect::<Option<Vec<_>>>()?;
                let e = (0..self.axis_count)
                    .map(|a| f2dot14_at(b, header + 2 * (self.axis_count + a)))
                    .collect::<Option<Vec<_>>>()?;
                header += 4 * self.axis_count;
                Some((s, e))
            } else {
                None
            };
            let this = data;
            data += size;
            let mut scalar = 1.0;
            for (a, &c) in self.coords.iter().enumerate() {
                let p = peak[a];
                let (s, e) = match &intermediate {
                    Some((s, e)) => (s[a], e[a]),
                    None => (p.min(0.0), p.max(0.0)),
                };
                scalar *= axis_scalar(c, s, p, e);
                if scalar == 0.0 {
                    break;
                }
            }
            if scalar == 0.0 {
                continue;
            }
            let mut q = this;
            let points = if index & 0x2000 != 0 {
                let (pts, next) = read_points(b, q)?;
                q = next;
                pts
            } else {
                shared_points.clone()?
            };
            // `None`: every point, in order.
            let m = points.as_ref().map_or(n, Vec::len);
            let (xs, next) = read_deltas(b, q, m)?;
            let (ys, _) = read_deltas(b, next, m)?;
            match points {
                None => {
                    for i in 0..n {
                        total[i].x += xs[i] * scalar;
                        total[i].y += ys[i] * scalar;
                    }
                }
                Some(pts) => {
                    let mut d = vec![None; n];
                    for (k, &i) in pts.iter().enumerate() {
                        if let Some(slot) = d.get_mut(i as usize) {
                            *slot = Some(Vector2::new(xs[k], ys[k]));
                        }
                    }
                    if let Some(ends) = contour_ends {
                        interpolate_untouched(original, ends, &mut d);
                    }
                    for (t, v) in total.iter_mut().zip(d) {
                        if let Some(v) = v {
                            t.x += v.x * scalar;
                            t.y += v.y * scalar;
                        }
                    }
                }
            }
        }
        Some(total)
    }
}

/// Packed point numbers: `None` for "all points".
fn read_points(b: &[u8], mut p: usize) -> Option<(Option<Vec<u16>>, usize)> {
    let first = *b.get(p)?;
    p += 1;
    if first == 0 {
        return Some((None, p));
    }
    let count = if first & 0x80 != 0 {
        let v = ((first as usize & 0x7F) << 8) | *b.get(p)? as usize;
        p += 1;
        v
    } else {
        first as usize
    };
    let mut out = Vec::with_capacity(count);
    let mut last = 0u16;
    while out.len() < count {
        let control = *b.get(p)?;
        p += 1;
        let run = (control & 0x7F) as usize + 1;
        for _ in 0..run {
            let v = if control & 0x80 != 0 {
                let v = u16_at(b, p)?;
                p += 2;
                v
            } else {
                let v = *b.get(p)? as u16;
                p += 1;
                v
            };
            last = last.wrapping_add(v);
            out.push(last);
        }
    }
    out.truncate(count);
    Some((Some(out), p))
}

/// `count` packed deltas.
fn read_deltas(b: &[u8], mut p: usize, count: usize) -> Option<(Vec<f32>, usize)> {
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let control = *b.get(p)?;
        p += 1;
        let run = (control & 0x3F) as usize + 1;
        for _ in 0..run {
            let v = match control & 0xC0 {
                0x80 => 0.0,
                0x40 => {
                    let v = i16_at(b, p)? as f32;
                    p += 2;
                    v
                }
                0xC0 => {
                    let v = u32_at(b, p)? as i32 as f32;
                    p += 4;
                    v
                }
                _ => {
                    let v = *b.get(p)? as i8 as f32;
                    p += 1;
                    v
                }
            };
            out.push(v);
        }
    }
    out.truncate(count);
    Some((out, p))
}

/// Fill the deltas a tuple leaves out (IUP): along each axis, a point
/// between two touched neighbours on its contour moves by the interpolation
/// of theirs, and one outside them by the nearer one's. A contour with no
/// touched point stays where it is; the phantom points past the last
/// contour are never interpolated.
fn interpolate_untouched(original: &[Vector2], ends: &[usize], d: &mut [Option<Vector2>]) {
    let mut start = 0;
    for &end in ends {
        if end >= d.len() || end < start {
            return;
        }
        let touched: Vec<usize> = (start..=end).filter(|&i| d[i].is_some()).collect();
        if touched.is_empty() {
            start = end + 1;
            continue;
        }
        if touched.len() == 1 {
            let v = d[touched[0]];
            for slot in &mut d[start..=end] {
                *slot = v;
            }
            start = end + 1;
            continue;
        }
        let len = end - start + 1;
        for (k, &a) in touched.iter().enumerate() {
            let b = touched[(k + 1) % touched.len()];
            // The untouched run after `a`, up to `b`, wrapping round.
            let mut i = if a == end { start } else { a + 1 };
            let mut guard = 0;
            while i != b && guard < len {
                let (da, db) = (d[a].unwrap(), d[b].unwrap());
                let (oa, ob, o) = (original[a], original[b], original[i]);
                let x = iup_axis(o.x, oa.x, ob.x, da.x, db.x);
                let y = iup_axis(o.y, oa.y, ob.y, da.y, db.y);
                d[i] = Some(Vector2::new(x, y));
                i = if i == end { start } else { i + 1 };
                guard += 1;
            }
        }
        start = end + 1;
    }
}

fn iup_axis(p: f32, a: f32, b: f32, da: f32, db: f32) -> f32 {
    let (lo, hi, dlo, dhi) = if a <= b { (a, b, da, db) } else { (b, a, db, da) };
    if p <= lo {
        dlo
    } else if p >= hi {
        dhi
    } else if hi == lo {
        if dlo == dhi { dlo } else { 0.0 }
    } else {
        dlo + (p - lo) / (hi - lo) * (dhi - dlo)
    }
}

/// An item variation store, resolved at one location: each region's scalar
/// computed once, deltas read on demand.
#[derive(Clone, Debug)]
pub(crate) struct ItemVariationStore<'a> {
    bytes: &'a [u8],
    off: usize,
    scalars: Vec<f32>,
}

impl<'a> ItemVariationStore<'a> {
    pub(crate) fn new(bytes: &'a [u8], off: usize, coords: &[f32]) -> Option<Self> {
        if u16_at(bytes, off)? != 1 {
            return None;
        }
        let regions = off + u32_at(bytes, off + 2)? as usize;
        let axis_count = u16_at(bytes, regions)? as usize;
        let region_count = u16_at(bytes, regions + 2)? as usize;
        let mut scalars = Vec::with_capacity(region_count);
        for r in 0..region_count {
            let mut s = 1.0;
            for a in 0..axis_count {
                let p = regions + 4 + (r * axis_count + a) * 6;
                let c = coords.get(a).copied().unwrap_or(0.0);
                s *= axis_scalar(c, f2dot14_at(bytes, p)?, f2dot14_at(bytes, p + 2)?, f2dot14_at(bytes, p + 4)?);
            }
            scalars.push(s);
        }
        Some(Self { bytes, off, scalars })
    }

    /// The region scalars of subtable `outer`, in that subtable's own region order.
    ///
    /// `CFF2`'s `blend` operator needs these rather than a delta: a charstring carries its
    /// *own* deltas on the stack and multiplies them by the scalars itself, one set per
    /// `vsindex`. The scalars are computed once per location in [`Self::new`]; this maps a
    /// subtable's `regionIndexes` through them, which is the same walk [`Self::try_delta`]
    /// does inline while reading deltas.
    pub(crate) fn region_scalars(&self, outer: u16) -> Option<Vec<f32>> {
        let b = self.bytes;
        if outer >= u16_at(b, self.off + 6)? {
            return None;
        }
        let data = self.off + u32_at(b, self.off + 8 + 4 * outer as usize)? as usize;
        let region_count = u16_at(b, data + 4)? as usize;
        (0..region_count)
            .map(|k| {
                let region = u16_at(b, data + 6 + 2 * k)? as usize;
                Some(self.scalars.get(region).copied().unwrap_or(0.0))
            })
            .collect()
    }

    /// The delta for item `inner` of subtable `outer`, in font units.
    pub(crate) fn delta(&self, outer: u16, inner: u16) -> f32 {
        self.try_delta(outer, inner).unwrap_or(0.0)
    }

    fn try_delta(&self, outer: u16, inner: u16) -> Option<f32> {
        let b = self.bytes;
        let count = u16_at(b, self.off + 6)?;
        if outer >= count {
            return None;
        }
        let data = self.off + u32_at(b, self.off + 8 + 4 * outer as usize)? as usize;
        let item_count = u16_at(b, data)?;
        if inner >= item_count {
            return None;
        }
        let word_word = u16_at(b, data + 2)?;
        let long = word_word & 0x8000 != 0;
        let words = (word_word & 0x7FFF) as usize;
        let region_count = u16_at(b, data + 4)? as usize;
        let (big, small) = if long { (4, 2) } else { (2, 1) };
        let row = words * big + (region_count - words.min(region_count)) * small;
        let mut p = data + 6 + 2 * region_count + inner as usize * row;
        let mut sum = 0.0;
        for k in 0..region_count {
            let region = u16_at(b, data + 6 + 2 * k)? as usize;
            let v = match (k < words, long) {
                (true, true) => {
                    let v = u32_at(b, p)? as i32 as f32;
                    p += 4;
                    v
                }
                (true, false) | (false, true) => {
                    let v = i16_at(b, p)? as f32;
                    p += 2;
                    v
                }
                (false, false) => {
                    let v = *b.get(p)? as i8 as f32;
                    p += 1;
                    v
                }
            };
            sum += v * self.scalars.get(region).copied().unwrap_or(0.0);
        }
        Some(sum)
    }
}

/// `HVAR`'s advance-width delta for each glyph, in font units.
pub(crate) fn hvar_advance_deltas(bytes: &[u8], off: usize, coords: &[f32], n_glyphs: usize) -> Option<Vec<f32>> {
    let store = ItemVariationStore::new(bytes, off + u32_at(bytes, off + 4)? as usize, coords)?;
    let map_off = u32_at(bytes, off + 8)? as usize;
    let map = if map_off == 0 { None } else { Some(off + map_off) };
    let mut out = Vec::with_capacity(n_glyphs);
    for gid in 0..n_glyphs {
        let (outer, inner) = match map {
            None => (0, gid as u16),
            Some(m) => delta_set_index(bytes, m, gid)?,
        };
        out.push(store.delta(outer, inner));
    }
    Some(out)
}

/// A DeltaSetIndexMap entry: `(outer, inner)` for item `i`, the last entry
/// standing in for any past the end.
fn delta_set_index(b: &[u8], off: usize, i: usize) -> Option<(u16, u16)> {
    let format = *b.get(off)?;
    let entry_format = *b.get(off + 1)?;
    let (count, data) = if format == 0 {
        (u16_at(b, off + 2)? as usize, off + 4)
    } else {
        (u32_at(b, off + 2)? as usize, off + 6)
    };
    if count == 0 {
        return Some((0, 0));
    }
    let size = ((entry_format & 0x30) >> 4) as usize + 1;
    let inner_bits = (entry_format & 0x0F) as u32 + 1;
    let p = data + i.min(count - 1) * size;
    let mut v = 0u32;
    for k in 0..size {
        v = (v << 8) | *b.get(p + k)? as u32;
    }
    Some(((v >> inner_bits) as u16, (v & ((1 << inner_bits) - 1)) as u16))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_region_peaks_at_its_peak_and_falls_off_linearly() {
        assert_eq!(axis_scalar(1.0, 0.0, 1.0, 1.0), 1.0);
        assert_eq!(axis_scalar(0.5, 0.0, 1.0, 1.0), 0.5);
        assert_eq!(axis_scalar(-0.5, 0.0, 1.0, 1.0), 0.0);
        assert_eq!(axis_scalar(0.0, 0.0, 1.0, 1.0), 0.0);
        // An intermediate region from 0.2 through 0.6 to 1.0.
        assert!((axis_scalar(0.4, 0.2, 0.6, 1.0) - 0.5).abs() < 1e-6);
        assert!((axis_scalar(0.8, 0.2, 0.6, 1.0) - 0.5).abs() < 1e-6);
        // A peak of 0 does not constrain the axis.
        assert_eq!(axis_scalar(0.7, 0.0, 0.0, 0.0), 1.0);
    }

    #[test]
    fn packed_points_accumulate_and_zero_means_all() {
        // Three points as byte deltas 1, +2, +3 → 1, 3, 6.
        let b = [3u8, 0x02, 1, 2, 3];
        assert_eq!(read_points(&b, 0), Some((Some(vec![1, 3, 6]), 5)));
        assert_eq!(read_points(&[0], 0), Some((None, 1)));
        // A count past 127 takes two bytes; a run of words.
        let b = [0x80, 2, 0x81, 0x01, 0x00, 0x00, 0x05];
        assert_eq!(read_points(&b, 0), Some((Some(vec![256, 261]), 7)));
    }

    #[test]
    fn packed_deltas_read_zero_byte_word_and_long_runs() {
        let b = [
            0x81, // two zeros
            0x01, 0xFF, 0x05, // two bytes: -1, 5
            0x40, 0x01, 0x00, // one word: 256
            0xC0, 0x00, 0x01, 0x00, 0x00, // one long: 65536
        ];
        let (d, end) = read_deltas(&b, 0, 6).unwrap();
        assert_eq!(d, vec![0.0, 0.0, -1.0, 5.0, 256.0, 65536.0]);
        assert_eq!(end, b.len());
    }

    /// The IUP example the spec draws: a square whose two left corners move
    /// and two right ones are untouched take the moved ones' deltas, and a
    /// point between two touched ones interpolates.
    #[test]
    fn untouched_points_follow_their_touched_neighbours() {
        let o = [
            Vector2::new(0.0, 0.0),
            Vector2::new(50.0, 0.0),
            Vector2::new(100.0, 0.0),
            Vector2::new(100.0, 100.0),
            Vector2::new(0.0, 100.0),
        ];
        let mut d = vec![
            Some(Vector2::new(0.0, 0.0)),
            None,
            Some(Vector2::new(10.0, 0.0)),
            None,
            None,
        ];
        interpolate_untouched(&o, &[4], &mut d);
        let d: Vec<Vector2> = d.into_iter().map(Option::unwrap).collect();
        // Halfway along x between deltas 0 and 10.
        assert!((d[1].x - 5.0).abs() < 1e-6);
        // Past the touched range on x: the nearer touched point's delta.
        assert!((d[3].x - 10.0).abs() < 1e-6);
        assert!((d[4].x - 0.0).abs() < 1e-6);
    }

    /// One touched point moves its whole contour; none leaves it alone.
    #[test]
    fn a_single_touched_point_shifts_its_contour() {
        let o = [Vector2::new(0.0, 0.0), Vector2::new(1.0, 0.0), Vector2::new(5.0, 5.0)];
        let mut d = vec![None, Some(Vector2::new(3.0, -2.0)), None];
        interpolate_untouched(&o, &[2], &mut d);
        assert!(d.iter().all(|v| *v == Some(Vector2::new(3.0, -2.0))));
        let mut d = vec![None, None, None];
        interpolate_untouched(&o, &[2], &mut d);
        assert!(d.iter().all(Option::is_none));
    }
}
