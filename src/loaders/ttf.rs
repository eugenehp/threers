//! Minimal TrueType / OpenType font reader. Extracts glyph outlines as
//! curve-based `Shape`s suitable for `TextGeometry`. Supports the `cmap`
//! format 4 (Unicode BMP), uncompressed `glyf` glyphs — simple ones, and
//! composite ones (accented letters, mostly) assembled from their components —
//! and `hmtx` advances. Hinting, OpenType `CFF` (PostScript) outlines, and
//! OpenType variations are out of scope.

use crate::curves::{Path, Shape};
use crate::geometries::Glyph;
use crate::math::Vector2;

#[derive(Debug)]
pub enum TtfError {
    Truncated,
    BadMagic,
    UnsupportedFormat,
}

pub struct TtfFont {
    pub units_per_em: u16,
    pub glyphs: Vec<TtfGlyph>,
    pub cmap: std::collections::HashMap<u32, u16>,
    /// `hhea.ascender` — height above the baseline, in font units.
    pub ascender: i16,
    /// `hhea.descender` — depth below the baseline, in font units (negative).
    pub descender: i16,
    /// `hhea.lineGap` — extra leading between lines, in font units.
    pub line_gap: i16,
    /// Kerning adjustments in font units, by `(left glyph, right glyph)`.
    ///
    /// A font's advance widths are what each glyph needs on its own; kerning is
    /// what a *pair* needs. Without it "AV" and "To" sit visibly loose, because
    /// the diagonal and the overhang leave a gap that the two advances know
    /// nothing about. Empty when the font has no `kern` table — which for a
    /// modern font often means its kerning is in `GPOS` instead, and that is not
    /// read here.
    pub kerning: std::collections::HashMap<(u16, u16), i16>,
}

pub struct TtfGlyph {
    pub shape: Shape,
    pub advance_width: u16,
}

impl TtfFont {
    /// Look up the glyph for a unicode codepoint and return a layout-ready
    /// `Glyph`. Falls back to glyph 0 (.notdef) if the codepoint isn't mapped.
    pub fn glyph_for(&self, c: u32) -> Option<Glyph> {
        let gid = self.cmap.get(&c).copied().unwrap_or(0);
        let g = self.glyphs.get(gid as usize)?;
        let scale = if self.units_per_em > 0 {
            1.0 / self.units_per_em as f32
        } else {
            1.0
        };
        let mut shape = Shape::new();
        for c in g.shape.outline.curve_path.curves.iter() {
            let _ = c;
        }
        // Clone-style copy: just reuse the same shape; integer-scale is left to
        // the caller (Glyph.advance is already in units).
        shape.outline = clone_path(&g.shape.outline);
        for hole in &g.shape.holes {
            shape.add_hole(clone_path(hole));
        }
        Some(Glyph {
            shape,
            advance: g.advance_width as f32 * scale,
        })
    }

    /// Parse a TTF byte slice into a usable font.
    pub fn parse(bytes: &[u8]) -> Result<Self, TtfError> {
        if bytes.len() < 12 {
            return Err(TtfError::Truncated);
        }
        let tag = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        // 0x00010000 = TrueType, 0x4F54544F = OTTO (OpenType/CFF — not supported).
        if tag != 0x00010000 && tag != 0x74727565 {
            return Err(TtfError::UnsupportedFormat);
        }
        let n_tables = u16::from_be_bytes([bytes[4], bytes[5]]) as usize;
        let mut tables: std::collections::HashMap<&'static [u8; 4], (usize, usize)> =
            std::collections::HashMap::new();
        let table_tags: [&'static [u8; 4]; 8] = [
            b"head", b"maxp", b"loca", b"glyf", b"cmap", b"hmtx", b"hhea", b"kern",
        ];
        for i in 0..n_tables {
            let off = 12 + i * 16;
            if off + 16 > bytes.len() {
                return Err(TtfError::Truncated);
            }
            let name = &bytes[off..off + 4];
            let table_off = u32::from_be_bytes([
                bytes[off + 8],
                bytes[off + 9],
                bytes[off + 10],
                bytes[off + 11],
            ]) as usize;
            let table_len = u32::from_be_bytes([
                bytes[off + 12],
                bytes[off + 13],
                bytes[off + 14],
                bytes[off + 15],
            ]) as usize;
            for tt in &table_tags {
                if *tt == name {
                    tables.insert(*tt, (table_off, table_len));
                }
            }
        }
        let (head_off, _) = *tables.get(b"head").ok_or(TtfError::UnsupportedFormat)?;
        if head_off + 54 > bytes.len() {
            return Err(TtfError::Truncated);
        }
        let units_per_em = u16::from_be_bytes([bytes[head_off + 18], bytes[head_off + 19]]);
        let index_to_loc_format = u16::from_be_bytes([bytes[head_off + 50], bytes[head_off + 51]]);

        let (maxp_off, _) = *tables.get(b"maxp").ok_or(TtfError::UnsupportedFormat)?;
        if maxp_off + 6 > bytes.len() {
            return Err(TtfError::Truncated);
        }
        let n_glyphs = u16::from_be_bytes([bytes[maxp_off + 4], bytes[maxp_off + 5]]) as usize;

        let (loca_off, _) = *tables.get(b"loca").ok_or(TtfError::UnsupportedFormat)?;
        let (glyf_off, _) = *tables.get(b"glyf").ok_or(TtfError::UnsupportedFormat)?;

        let glyph_offset = |idx: usize| -> usize {
            if index_to_loc_format == 0 {
                let o =
                    u16::from_be_bytes([bytes[loca_off + idx * 2], bytes[loca_off + idx * 2 + 1]])
                        as usize;
                glyf_off + o * 2
            } else {
                let o = u32::from_be_bytes([
                    bytes[loca_off + idx * 4],
                    bytes[loca_off + idx * 4 + 1],
                    bytes[loca_off + idx * 4 + 2],
                    bytes[loca_off + idx * 4 + 3],
                ]) as usize;
                glyf_off + o
            }
        };

        // For components named by id: an id past the table reads as empty
        // rather than past the end of `loca`.
        let bounded_offset = |idx: usize| -> usize {
            if idx <= n_glyphs {
                glyph_offset(idx)
            } else {
                0
            }
        };

        // hmtx advance widths.
        let (hhea_off, _) = *tables.get(b"hhea").ok_or(TtfError::UnsupportedFormat)?;
        if hhea_off + 36 > bytes.len() {
            return Err(TtfError::Truncated);
        }
        let ascender = i16::from_be_bytes([bytes[hhea_off + 4], bytes[hhea_off + 5]]);
        let descender = i16::from_be_bytes([bytes[hhea_off + 6], bytes[hhea_off + 7]]);
        let line_gap = i16::from_be_bytes([bytes[hhea_off + 8], bytes[hhea_off + 9]]);
        let n_h_metrics = u16::from_be_bytes([bytes[hhea_off + 34], bytes[hhea_off + 35]]) as usize;
        let (hmtx_off, _) = *tables.get(b"hmtx").ok_or(TtfError::UnsupportedFormat)?;
        let mut advance_widths = vec![0u16; n_glyphs];
        for i in 0..n_glyphs {
            if i < n_h_metrics {
                advance_widths[i] =
                    u16::from_be_bytes([bytes[hmtx_off + i * 4], bytes[hmtx_off + i * 4 + 1]]);
            } else {
                advance_widths[i] = advance_widths
                    .get(n_h_metrics.saturating_sub(1))
                    .copied()
                    .unwrap_or(0);
            }
        }

        // Parse cmap format 4.
        let (cmap_off, _) = *tables.get(b"cmap").ok_or(TtfError::UnsupportedFormat)?;
        let cmap = parse_cmap(bytes, cmap_off).unwrap_or_default();

        // Parse glyph outlines.
        let mut glyphs: Vec<TtfGlyph> = Vec::with_capacity(n_glyphs);
        // `g_idx` is a glyph id, passed to `glyph_offset` as well as indexing.
        #[allow(clippy::needless_range_loop)]
        for g_idx in 0..n_glyphs {
            let off = glyph_offset(g_idx);
            let next_off = glyph_offset(g_idx + 1);
            if next_off <= off {
                glyphs.push(TtfGlyph {
                    shape: Shape::new(),
                    advance_width: advance_widths[g_idx],
                });
                continue;
            }
            let shape = glyph_contours(bytes, &bounded_offset, g_idx, 0)
                .map(|c| shape_from_contours(&c))
                .unwrap_or_default();
            glyphs.push(TtfGlyph {
                shape,
                advance_width: advance_widths[g_idx],
            });
        }
        let kerning = tables
            .get(b"kern")
            .and_then(|&(off, len)| parse_kern(bytes, off, len))
            .unwrap_or_default();
        Ok(TtfFont {
            units_per_em,
            glyphs,
            cmap,
            ascender,
            descender,
            line_gap,
            kerning,
        })
    }

    /// The kerning between two glyphs, in font units. Zero when the pair has
    /// none, which is the overwhelming majority of pairs.
    pub fn kern(&self, left: u16, right: u16) -> i16 {
        self.kerning.get(&(left, right)).copied().unwrap_or(0)
    }
}

/// The `kern` table, format 0 — a sorted list of glyph pairs and adjustments.
///
/// Only horizontal, non-cross-stream format 0 subtables are read, which is what
/// a TrueType font with kerning actually ships. The header is two `uint16`s;
/// Apple's variant of this table starts with a `uint32` version instead, and is
/// left alone rather than guessed at — misreading the header would produce pairs
/// from whatever the bytes happened to say, and a wrong kern is worse than none.
fn parse_kern(
    bytes: &[u8],
    off: usize,
    len: usize,
) -> Option<std::collections::HashMap<(u16, u16), i16>> {
    let table = bytes.get(off..off + len)?;
    let be16 = |at: usize| -> Option<u16> {
        Some(u16::from_be_bytes([*table.get(at)?, *table.get(at + 1)?]))
    };
    if be16(0)? != 0 {
        return None;
    }
    let subtables = be16(2)? as usize;
    let mut out = std::collections::HashMap::new();
    let mut at = 4usize;
    for _ in 0..subtables {
        let length = be16(at + 2)? as usize;
        let coverage = be16(at + 4)?;
        // The format is the high byte; bit 0 of the low byte means horizontal,
        // and bit 2 means cross-stream, which adjusts the *other* axis.
        let format = coverage >> 8;
        let horizontal = coverage & 0x0001 != 0;
        let cross_stream = coverage & 0x0004 != 0;
        if format == 0 && horizontal && !cross_stream {
            let pairs = be16(at + 6)? as usize;
            // Four `uint16`s of binary-search hints sit between the count and
            // the pairs, and skipping them is what reads the first pair as a
            // search range.
            let mut pair_at = at + 14;
            for _ in 0..pairs {
                let left = be16(pair_at)?;
                let right = be16(pair_at + 2)?;
                let value = be16(pair_at + 4)? as i16;
                if value != 0 {
                    out.insert((left, right), value);
                }
                pair_at += 6;
            }
        }
        if length == 0 {
            break;
        }
        at += length;
    }
    Some(out)
}

fn clone_path(p: &Path) -> Path {
    let mut np = Path::new();
    np.current = p.current;
    for c in p.curve_path.curves.iter() {
        let samples = 12;
        let prev_first = c.get_point(0.0);
        np.move_to(prev_first);
        for i in 1..=samples {
            np.line_to(c.get_point(i as f32 / samples as f32));
        }
    }
    np
}

/// One contour's points in font units, each with whether it is on the curve.
type Contour = Vec<(Vector2, bool)>;

/// Glyph `gid`'s contours: its own for a simple glyph; for a composite one,
/// each component's, placed by its offset and scaled or transformed as its
/// flags say. `depth` guards against a font whose components refer back to
/// themselves.
fn glyph_contours(
    bytes: &[u8],
    glyph_offset: &dyn Fn(usize) -> usize,
    gid: usize,
    depth: u32,
) -> Option<Vec<Contour>> {
    let off = glyph_offset(gid);
    if glyph_offset(gid + 1) <= off || off + 10 > bytes.len() || depth > 8 {
        return None;
    }
    let n_contours = i16::from_be_bytes([bytes[off], bytes[off + 1]]);
    if n_contours >= 0 {
        return parse_simple_contours(bytes, off);
    }
    let rd16 = |p: usize| -> Option<i16> {
        Some(i16::from_be_bytes([*bytes.get(p)?, *bytes.get(p + 1)?]))
    };
    let f2dot14 = |p: usize| rd16(p).map(|v| v as f32 / 16384.0);
    let mut out = Vec::new();
    let mut p = off + 10; // numberOfContours and the bounding box
    loop {
        let flags = rd16(p)? as u16;
        let component = rd16(p + 2)? as u16 as usize;
        p += 4;
        // ARG_1_AND_2_ARE_WORDS, else bytes; signed when they are offsets.
        let (a1, a2) = if flags & 0x0001 != 0 {
            let v = (rd16(p)? as f32, rd16(p + 2)? as f32);
            p += 4;
            v
        } else {
            let v = (*bytes.get(p)? as i8 as f32, *bytes.get(p + 1)? as i8 as f32);
            p += 2;
            v
        };
        // ARGS_ARE_XY_VALUES: an offset. Otherwise they name points to
        // align, which fonts almost never use; placed unmoved.
        let (dx, dy) = if flags & 0x0002 != 0 {
            (a1, a2)
        } else {
            (0.0, 0.0)
        };
        // (xx, xy, yx, yy)
        let m = if flags & 0x0008 != 0 {
            let k = f2dot14(p)?;
            p += 2;
            (k, 0.0, 0.0, k)
        } else if flags & 0x0040 != 0 {
            let v = (f2dot14(p)?, 0.0, 0.0, f2dot14(p + 2)?);
            p += 4;
            v
        } else if flags & 0x0080 != 0 {
            let v = (
                f2dot14(p)?,
                f2dot14(p + 2)?,
                f2dot14(p + 4)?,
                f2dot14(p + 6)?,
            );
            p += 8;
            v
        } else {
            (1.0, 0.0, 0.0, 1.0)
        };
        if let Some(contours) = glyph_contours(bytes, glyph_offset, component, depth + 1) {
            for c in contours {
                out.push(
                    c.into_iter()
                        .map(|(v, on)| {
                            (
                                Vector2::new(
                                    m.0 * v.x + m.2 * v.y + dx,
                                    m.1 * v.x + m.3 * v.y + dy,
                                ),
                                on,
                            )
                        })
                        .collect(),
                );
            }
        }
        // MORE_COMPONENTS
        if flags & 0x0020 == 0 {
            break;
        }
    }
    Some(out)
}

fn parse_cmap(bytes: &[u8], cmap_off: usize) -> Option<std::collections::HashMap<u32, u16>> {
    if cmap_off + 4 > bytes.len() {
        return None;
    }
    let n = u16::from_be_bytes([bytes[cmap_off + 2], bytes[cmap_off + 3]]) as usize;
    let mut sub_off = 0usize;
    for i in 0..n {
        let rec = cmap_off + 4 + i * 8;
        if rec + 8 > bytes.len() {
            return None;
        }
        let platform = u16::from_be_bytes([bytes[rec], bytes[rec + 1]]);
        let encoding = u16::from_be_bytes([bytes[rec + 2], bytes[rec + 3]]);
        let offset = u32::from_be_bytes([
            bytes[rec + 4],
            bytes[rec + 5],
            bytes[rec + 6],
            bytes[rec + 7],
        ]) as usize;
        // platform=3 (Microsoft) encoding=1 (Unicode BMP) is the standard target.
        if platform == 3 && (encoding == 1 || encoding == 10) {
            sub_off = cmap_off + offset;
            break;
        }
    }
    if sub_off == 0 {
        return None;
    }
    let format = u16::from_be_bytes([bytes[sub_off], bytes[sub_off + 1]]);
    if format != 4 {
        return None;
    }
    let seg_count_x2 = u16::from_be_bytes([bytes[sub_off + 6], bytes[sub_off + 7]]) as usize;
    let seg_count = seg_count_x2 / 2;
    let end_off = sub_off + 14;
    let start_off = end_off + seg_count_x2 + 2;
    let id_delta_off = start_off + seg_count_x2;
    let id_range_off = id_delta_off + seg_count_x2;
    let mut map = std::collections::HashMap::new();
    for i in 0..seg_count {
        let end = u16::from_be_bytes([bytes[end_off + i * 2], bytes[end_off + i * 2 + 1]]);
        let start = u16::from_be_bytes([bytes[start_off + i * 2], bytes[start_off + i * 2 + 1]]);
        let id_delta =
            i16::from_be_bytes([bytes[id_delta_off + i * 2], bytes[id_delta_off + i * 2 + 1]]);
        let id_range_o =
            u16::from_be_bytes([bytes[id_range_off + i * 2], bytes[id_range_off + i * 2 + 1]])
                as usize;
        for c in start..=end {
            let gid = if id_range_o == 0 {
                (c as i32).wrapping_add(id_delta as i32) as u16
            } else {
                let off = id_range_off + i * 2 + id_range_o + (c - start) as usize * 2;
                if off + 2 > bytes.len() {
                    0
                } else {
                    let v = u16::from_be_bytes([bytes[off], bytes[off + 1]]);
                    if v == 0 {
                        0
                    } else {
                        (v as i32).wrapping_add(id_delta as i32) as u16
                    }
                }
            };
            map.insert(c as u32, gid);
        }
    }
    Some(map)
}

/// A simple glyph's contours.
fn parse_simple_contours(bytes: &[u8], off: usize) -> Option<Vec<Contour>> {
    if off + 10 > bytes.len() {
        return None;
    }
    let n_contours = i16::from_be_bytes([bytes[off], bytes[off + 1]]);
    if n_contours <= 0 {
        return None;
    }
    let n_contours = n_contours as usize;
    let mut p = off + 10;
    let mut end_pts = Vec::with_capacity(n_contours);
    for _ in 0..n_contours {
        if p + 2 > bytes.len() {
            return None;
        }
        end_pts.push(u16::from_be_bytes([bytes[p], bytes[p + 1]]) as usize);
        p += 2;
    }
    let n_points = end_pts.last().copied().unwrap_or(0) + 1;
    let instruction_len = u16::from_be_bytes([bytes[p], bytes[p + 1]]) as usize;
    p += 2 + instruction_len;

    // Flags.
    let mut flags = Vec::with_capacity(n_points);
    while flags.len() < n_points {
        if p >= bytes.len() {
            return None;
        }
        let f = bytes[p];
        p += 1;
        flags.push(f);
        if (f & 0x08) != 0 {
            let rep = bytes[p] as usize;
            p += 1;
            for _ in 0..rep {
                flags.push(f);
            }
        }
    }

    // x coords.
    let mut xs = Vec::with_capacity(n_points);
    let mut x = 0i32;
    for &f in &flags {
        let dx = if (f & 0x02) != 0 {
            let v = bytes[p] as i32;
            p += 1;
            if (f & 0x10) != 0 {
                v
            } else {
                -v
            }
        } else if (f & 0x10) != 0 {
            0
        } else {
            let v = i16::from_be_bytes([bytes[p], bytes[p + 1]]) as i32;
            p += 2;
            v
        };
        x += dx;
        xs.push(x);
    }
    // y coords.
    let mut ys = Vec::with_capacity(n_points);
    let mut y = 0i32;
    for &f in &flags {
        let dy = if (f & 0x04) != 0 {
            let v = bytes[p] as i32;
            p += 1;
            if (f & 0x20) != 0 {
                v
            } else {
                -v
            }
        } else if (f & 0x20) != 0 {
            0
        } else {
            let v = i16::from_be_bytes([bytes[p], bytes[p + 1]]) as i32;
            p += 2;
            v
        };
        y += dy;
        ys.push(y);
    }

    let mut contours = Vec::with_capacity(end_pts.len());
    let mut start = 0;
    for &end in &end_pts {
        if end >= n_points || start > end {
            return None;
        }
        contours.push(
            (start..=end)
                .map(|i| {
                    (
                        Vector2::new(xs[i] as f32, ys[i] as f32),
                        (flags[i] & 1) != 0,
                    )
                })
                .collect(),
        );
        start = end + 1;
    }
    Some(contours)
}

/// A path from contours. Off-curve points become quadratic bezier control
/// points; consecutive off-curves get an implied on-curve between them.
fn shape_from_contours(contours: &[Contour]) -> Shape {
    let mut path = Path::new();
    for pts in contours {
        if pts.is_empty() {
            continue;
        }
        // Start at the first on-curve point.
        let mut first_idx = 0;
        for (i, (_, on)) in pts.iter().enumerate() {
            if *on {
                first_idx = i;
                break;
            }
        }
        let first = pts[first_idx].0;
        path.move_to(first);
        let n = pts.len();
        let mut i = 1;
        while i < n + 1 {
            let cur = pts[(first_idx + i) % n];
            let next = pts[(first_idx + i + 1) % n];
            if cur.1 {
                path.line_to(cur.0);
            } else {
                // Quadratic — control = cur. End = next (or midpoint if off-off).
                let end_pt = if next.1 {
                    next.0
                } else {
                    Vector2::new((cur.0.x + next.0.x) * 0.5, (cur.0.y + next.0.y) * 0.5)
                };
                path.quadratic_curve_to(cur.0, end_pt);
                if next.1 {
                    i += 1;
                }
            }
            i += 1;
        }
    }
    Shape::from_path(path)
}

#[cfg(test)]
mod composite_tests {
    use super::*;

    /// DejaVu builds ü from u and a diaeresis: two more contours than u,
    /// above it, and the u's own contours where they are in the u.
    #[test]
    fn an_accented_letter_has_its_base_and_its_accent() {
        let font = TtfFont::parse(include_bytes!("../openscad/DejaVuSans.ttf")).unwrap();
        let points = |c: char| -> Vec<Vector2> {
            let g = &font.glyphs[font.cmap[&(c as u32)] as usize];
            g.shape
                .outline
                .curve_path
                .curves
                .iter()
                .map(|c| c.get_point(0.0))
                .collect()
        };
        let (u, uu) = (points('u'), points('ü'));
        assert!(!u.is_empty());
        assert!(
            uu.len() > u.len(),
            "ü {} curves against u {}",
            uu.len(),
            u.len()
        );
        let top = |p: &[Vector2]| p.iter().map(|v| v.y).fold(f32::MIN, f32::max);
        assert!(top(&uu) > top(&u) + 100.0, "the dots sit above the u");
        // Every point of the u is in the ü.
        for p in &u {
            assert!(
                uu.iter()
                    .any(|q| (q.x - p.x).abs() < 0.5 && (q.y - p.y).abs() < 0.5),
                "{p:?} missing"
            );
        }
    }
}

#[cfg(test)]
mod kern_tests {
    use super::*;

    /// A hand-built `kern` table, to pin the offsets.
    ///
    /// Everything here is a fixed offset into a packed header, and getting one
    /// wrong reads a search hint as a pair — which yields kerning for glyph ids
    /// that may well exist, so the font renders with adjustments it never asked
    /// for. The four binary-search fields between the count and the pairs are the
    /// usual casualty.
    #[test]
    fn a_format_zero_table_parses_at_the_right_offsets() {
        let mut table: Vec<u8> = Vec::new();
        table.extend_from_slice(&0u16.to_be_bytes()); // version
        table.extend_from_slice(&1u16.to_be_bytes()); // one subtable
        let subtable_len = 14 + 3 * 6;
        table.extend_from_slice(&0u16.to_be_bytes()); // subtable version
        table.extend_from_slice(&(subtable_len as u16).to_be_bytes());
        table.extend_from_slice(&0x0001u16.to_be_bytes()); // format 0, horizontal
        table.extend_from_slice(&3u16.to_be_bytes()); // nPairs
                                                      // searchRange, entrySelector, rangeShift — three fields that exist only
                                                      // to be skipped. Poisoned, so reading one as a pair is a visible failure
                                                      // rather than a plausible kern.
        for v in [0x0BADu16, 0x0BAD, 0x0BAD] {
            table.extend_from_slice(&v.to_be_bytes());
        }
        for (l, r, v) in [(10u16, 20u16, -40i16), (10, 21, 0), (11, 20, 15)] {
            table.extend_from_slice(&l.to_be_bytes());
            table.extend_from_slice(&r.to_be_bytes());
            table.extend_from_slice(&v.to_be_bytes());
        }

        let map = parse_kern(&table, 0, table.len()).expect("parses");
        assert_eq!(map.get(&(10, 20)).copied(), Some(-40));
        assert_eq!(map.get(&(11, 20)).copied(), Some(15));
        // A zero adjustment is not worth an entry.
        assert_eq!(map.get(&(10, 21)), None);
        // And nothing from the search hints leaked in.
        assert_eq!(map.len(), 2, "{map:?}");
    }

    /// A cross-stream or vertical subtable is skipped, not read as horizontal.
    ///
    /// Cross-stream kerning moves the *other* axis; applying it to advances would
    /// shift glyphs sideways by an amount meant for the baseline.
    #[test]
    fn only_horizontal_format_zero_subtables_are_read() {
        let build = |coverage: u16| {
            let mut t: Vec<u8> = Vec::new();
            t.extend_from_slice(&0u16.to_be_bytes());
            t.extend_from_slice(&1u16.to_be_bytes());
            t.extend_from_slice(&0u16.to_be_bytes());
            t.extend_from_slice(&((14 + 6) as u16).to_be_bytes());
            t.extend_from_slice(&coverage.to_be_bytes());
            t.extend_from_slice(&1u16.to_be_bytes()); // nPairs
            for _ in 0..3 {
                t.extend_from_slice(&0u16.to_be_bytes()); // the search hints
            }
            t.extend_from_slice(&1u16.to_be_bytes()); // left
            t.extend_from_slice(&2u16.to_be_bytes());
            t.extend_from_slice(&(-9i16).to_be_bytes());
            t
        };
        // Horizontal, format 0: read.
        let t = build(0x0001);
        assert_eq!(parse_kern(&t, 0, t.len()).unwrap().len(), 1);
        // Cross-stream: skipped.
        let t = build(0x0005);
        assert!(parse_kern(&t, 0, t.len()).unwrap().is_empty());
        // Vertical (horizontal bit clear): skipped.
        let t = build(0x0000);
        assert!(parse_kern(&t, 0, t.len()).unwrap().is_empty());
        // Format 2: skipped, since only format 0 is understood.
        let t = build(0x0201);
        assert!(parse_kern(&t, 0, t.len()).unwrap().is_empty());
    }

    /// A truncated or nonsense table yields nothing rather than panicking.
    #[test]
    fn a_broken_kern_table_is_not_fatal() {
        assert!(parse_kern(&[], 0, 0).is_none());
        // A version this does not know.
        let t = vec![0x00, 0x01, 0x00, 0x01];
        assert!(parse_kern(&t, 0, t.len()).is_none());
        // Truncated mid-subtable, at every length.
        let mut full: Vec<u8> = vec![0, 0, 0, 1, 0, 0, 0, 20, 0, 1, 0, 1];
        full.extend_from_slice(&[0u8; 20]);
        for cut in 0..full.len() {
            let _ = parse_kern(&full[..cut], 0, cut);
        }
    }

    /// Real fonts kern the pairs typography says they should.
    ///
    /// `AV`, `To` and `LT` all tighten — a diagonal beside a diagonal, and a
    /// crossbar overhanging a round letter. `oo` does not: two round letters need
    /// no adjustment, which is the control that says these numbers are the
    /// font's own and not something the parser invented.
    ///
    /// Skipped where the fonts are not installed.
    #[test]
    fn real_fonts_kern_the_expected_pairs() {
        let mut checked = 0;
        for path in [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",
            "/System/Library/Fonts/Supplemental/Trebuchet MS.ttf",
        ] {
            let Ok(bytes) = std::fs::read(path) else {
                continue;
            };
            let Ok(font) = TtfFont::parse(&bytes) else {
                continue;
            };
            if font.kerning.is_empty() {
                continue;
            }
            checked += 1;
            let gid = |c: char| font.cmap.get(&(c as u32)).copied().unwrap_or(0);
            for (a, b) in [('A', 'V'), ('T', 'o'), ('L', 'T')] {
                assert!(
                    font.kern(gid(a), gid(b)) < 0,
                    "{path}: {a}{b} should tighten, got {}",
                    font.kern(gid(a), gid(b))
                );
            }
            assert_eq!(
                font.kern(gid('o'), gid('o')),
                0,
                "{path}: two round letters need no kerning"
            );
        }
        if checked == 0 {
            eprintln!("skipping: no kerned system font found");
        }
    }
}
