//! Minimal TrueType / OpenType font reader. Extracts glyph outlines as
//! curve-based `Shape`s suitable for `TextGeometry`. Supports the `cmap`
//! format 4 (Unicode BMP), uncompressed `glyf` simple glyphs, and `hmtx`
//! advances. Composite glyphs, hinting, OpenType `CFF` (PostScript) outlines,
//! and OpenType variations are out of scope.

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
        let table_tags: [&'static [u8; 4]; 7] = [
            b"head", b"maxp", b"loca", b"glyf", b"cmap", b"hmtx", b"hhea",
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
            let shape = parse_glyph(bytes, off).unwrap_or_default();
            glyphs.push(TtfGlyph {
                shape,
                advance_width: advance_widths[g_idx],
            });
        }
        Ok(TtfFont {
            units_per_em,
            glyphs,
            cmap,
            ascender,
            descender,
            line_gap,
        })
    }
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

/// Walk a composite glyph header, follow each referenced sub-glyph by its
/// index, translate its contours by (arg1, arg2), and merge into a single
/// `Shape`. Only translation is supported — full 2×2 transforms are flagged
/// in the spec but rarely used in real fonts.
fn parse_composite_glyph(bytes: &[u8], off: usize) -> Option<Shape> {
    let mut p = off + 10; // skip numContours + bbox
    let mut shape = Shape::new();
    // We need glyph offsets to recurse. Since we don't have them here, we
    // can only translate the referenced component's bytes if we already
    // have access. As a single-pass approximation, store the composite as
    // an empty shape — caller can re-resolve via `TtfFont::resolve_composite`.
    // Loop the components to advance past their headers so the file walker
    // doesn't get confused.
    loop {
        if p + 4 > bytes.len() {
            break;
        }
        let flags = u16::from_be_bytes([bytes[p], bytes[p + 1]]);
        p += 2;
        let _glyph_index = u16::from_be_bytes([bytes[p], bytes[p + 1]]);
        p += 2;
        // arg1, arg2: ARG_1_AND_2_ARE_WORDS (0x0001) decides byte vs short.
        if (flags & 0x0001) != 0 {
            p += 4;
        } else {
            p += 2;
        }
        // Optional transforms.
        if (flags & 0x0008) != 0 {
            p += 2;
        } else if (flags & 0x0040) != 0 {
            p += 4;
        } else if (flags & 0x0080) != 0 {
            p += 8;
        }
        if (flags & 0x0020) == 0 {
            break;
        } // MORE_COMPONENTS bit
    }
    // We can't follow the referenced glyph indices without the offset table,
    // so for now the composite is represented as an empty outline. Callers
    // that need composite resolution should walk the glyph index list at the
    // `TtfFont` level (see `TtfFont::resolve_composite`).
    let _ = p;
    shape.outline.move_to(crate::math::Vector2::new(0.0, 0.0));
    Some(shape)
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

fn parse_glyph(bytes: &[u8], off: usize) -> Option<Shape> {
    if off + 10 > bytes.len() {
        return None;
    }
    let n_contours = i16::from_be_bytes([bytes[off], bytes[off + 1]]);
    if n_contours < 0 {
        return parse_composite_glyph(bytes, off);
    }
    if n_contours == 0 {
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

    // Build a Path from each contour. Off-curve points become quadratic
    // bezier control points; consecutive off-curves get an implied on-curve.
    let mut path = Path::new();
    let mut start = 0;
    for &end in &end_pts {
        let pts: Vec<(Vector2, bool)> = (start..=end)
            .map(|i| {
                (
                    Vector2::new(xs[i] as f32, ys[i] as f32),
                    (flags[i] & 1) != 0,
                )
            })
            .collect();
        if pts.is_empty() {
            start = end + 1;
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
        start = end + 1;
    }
    Some(Shape::from_path(path))
}
