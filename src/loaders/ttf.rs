//! Minimal TrueType / OpenType font reader. Extracts glyph outlines as
//! curve-based `Shape`s suitable for `TextGeometry`. Supports the `cmap`
//! formats **4 and 12** — the second is what reaches past the BMP, so emoji and
//! the CJK extensions resolve rather than drawing as boxes — uncompressed
//! `glyf` glyphs, simple ones and composite ones (accented letters, mostly)
//! assembled from their components, and `hmtx` advances, with pair kerning from
//! `kern` and from `GPOS`.
//!
//! Variable fonts are read at one point of their design space
//! ([`TtfFont::parse_variation`]): `gvar` moves the outlines, and `HVAR` (or
//! `gvar`'s phantom points) the advances. Hinting is out of scope.
//!
//! **`CFF` and `CFF2`** outlines are read too, by [`super::cff`] — PostScript
//! charstrings rather than point lists. Sixty of the 369 font files macOS ships
//! keep their outlines there and every one of them is a `.ttc` **collection**,
//! so both pieces are needed before any of them loads: Hiragino, Apple SD
//! Gothic Neo, Hiragino Sans GB and the Kohinoor family are all unreachable
//! without them. [`TtfFont::parse_face`] picks a face of a collection.
//!
//! This reads a font. It does not **shape** text: one codepoint becomes one
//! glyph, so a ligature stays two glyphs and Arabic letters do not join. What a
//! caption needs *around* that reading — composition, reordering, mirroring —
//! is in [`crate::captions::order`].

use crate::curves::{Path, Shape};
use crate::geometries::Glyph;
use crate::math::Vector2;

use super::gpos::GposKerning;
use super::ttf_var::{Gvar, ItemVariationStore, NamedInstance, VariationAxis};

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
    /// modern font usually means its kerning is in
    /// [`gpos_kerning`](Self::gpos_kerning) instead.
    pub kerning: std::collections::HashMap<(u16, u16), i16>,
    /// Pair kerning from `GPOS`'s `kern` feature, at this font's variation.
    pub gpos_kerning: GposKerning,
    /// `GPOS` and `GDEF`'s bytes, kept **only** when the font has mark positioning.
    ///
    /// Most Latin fonts have none — `SFNS` registers `kern` and one `ss09` lookup and
    /// nothing else — so they pay nothing for this, and `SFArabic` pays 75 kB. Resolving
    /// the anchors ahead of time instead would mean a table of every (mark, base) pair in
    /// the font, which is the product of two coverages.
    pub gpos: Vec<u8>,
    /// See [`Self::gpos`].
    pub gdef: Vec<u8>,
    /// `GSUB`'s bytes, unparsed, or empty where the font has no substitutions.
    ///
    /// This reader does not interpret them — it maps one codepoint to one glyph — but it
    /// is the thing holding the file, so it is the only place a shaper can get the table
    /// without walking the directory a second time. Owned because `TtfFont` borrows
    /// nothing; `GSUB` is 10 to 60 kB in the fonts here.
    pub gsub: Vec<u8>,
    /// The variation axes; empty for a static font.
    pub axes: Vec<VariationAxis>,
    /// A variable font's named instances ("Regular", "Bold", …).
    pub instances: Vec<NamedInstance>,
    /// The user-unit coordinate per axis the outlines were read at.
    pub location: Vec<f32>,
    /// The same location **normalized** to `-1..=1`, with `avar` applied.
    ///
    /// Kept because a `GPOS` anchor can vary with the axes and the item variation store
    /// it varies through needs this; deriving it again would mean retaining `fvar` and
    /// `avar` to do it with.
    pub coords: Vec<f32>,
    /// Apple's `trak` table, normal track: `(pixels per em, font units)`
    /// pairs, sorted by size. Empty for most fonts.
    pub tracking: Vec<(f32, f32)>,
    /// `OS/2.usWeightClass`: the face's own CSS weight (400 when absent).
    /// A variable font reports its default instance's.
    pub weight_class: u16,
    /// Whether a **run** of this face's glyphs composites by taking the *maximum* of two
    /// glyphs' coverage in a shared cell, rather than their union.
    ///
    /// True for any script whose letters join — Arabic, Syriac, N'Ko, Mongolian,
    /// Devanagari, Bengali, Gurmukhi — where two glyphs cover the same stroke and a union
    /// darkens the joint. Computed once here from four tables; see
    /// [`crate::loaders::overlap`] for the rule and where it was read.
    pub composites_run: bool,
}

pub struct TtfGlyph {
    pub shape: Shape,
    pub advance_width: u16,
}

impl TtfFont {
    /// Look up the glyph for a unicode codepoint and return a layout-ready
    /// `Glyph`, outline and advance both in ems. Falls back to glyph 0
    /// (.notdef) if the codepoint isn't mapped.
    ///
    /// One [`Shape`] cannot say which of a glyph's contours are holes or
    /// separate pieces — the dot of an `i`, the counter of an `o` — so this
    /// keeps every contour in its outline; [`generate_shapes`](Self::generate_shapes)
    /// sorts them out.
    #[deprecated(note = "use `generate_shapes`, which separates contours into shapes and holes")]
    pub fn glyph_for(&self, c: u32) -> Option<Glyph> {
        let gid = self.cmap.get(&c).copied().unwrap_or(0);
        let g = self.glyphs.get(gid as usize)?;
        let scale = 1.0 / self.units_per_em.max(1) as f32;
        let mut shape = Shape::new();
        shape.outline = scaled_path(&g.shape.outline, scale);
        for hole in &g.shape.holes {
            shape.add_hole(scaled_path(hole, scale));
        }
        Some(Glyph {
            shape,
            advance: g.advance_width as f32 * scale,
        })
    }

    /// One path per drawn character of `text`, `size` units to the em —
    /// three.js's `Font.createPaths` for a TrueType face. Lines break at
    /// `\n`, a line apart by the font's ascent, descent and gap; pairs are
    /// kerned. A character the font lacks draws as its `?`.
    pub fn create_paths(&self, text: &str, size: f64) -> Vec<crate::curves::shapes::ShapePath> {
        let scale = size / self.units_per_em.max(1) as f64;
        let line_height = (self.ascender as f64 - self.descender as f64 + self.line_gap as f64) * scale;
        let gid = |c: char| self.cmap.get(&(c as u32)).or_else(|| self.cmap.get(&('?' as u32))).copied();
        let mut paths = Vec::new();
        let (mut x, mut y) = (0.0f64, 0.0f64);
        let mut previous: Option<u16> = None;
        for c in text.chars() {
            if c == '\n' {
                (x, y) = (0.0, y - line_height);
                previous = None;
                continue;
            }
            let Some(g) = gid(c) else { continue };
            if let Some(p) = previous {
                x += self.kern(p, g) as f64 * scale;
            }
            let Some(glyph) = self.glyphs.get(g as usize) else { continue };
            paths.push(shape_path(&glyph.shape.outline, scale, x, y));
            x += glyph.advance_width as f64 * scale;
            previous = Some(g);
        }
        paths
    }

    /// `text`'s glyphs as shapes with holes, `size` units to the em —
    /// three.js's `Font.generateShapes` for a TrueType face, ready for
    /// [`ExtrudeGeometry::from_shapes`](crate::geometries::ExtrudeGeometry::from_shapes)
    /// or [`TextGeometry::from_ttf`](crate::geometries::TextGeometry::from_ttf).
    ///
    /// The curves stay curves — quadratic Béziers, as the font has them — so
    /// the extrusion's `curve_segments` decides how finely they are cut.
    pub fn generate_shapes(&self, text: &str, size: f64) -> Vec<crate::curves::shapes::Shape> {
        // TrueType draws outer contours clockwise and holes the other way.
        self.create_paths(text, size).iter().flat_map(|p| p.to_shapes(false)).collect()
    }

    /// Parse a TTF byte slice into a usable font — a variable font at its
    /// default instance, which is whatever its designer put in `glyf` (for
    /// SF Compact, Black).
    pub fn parse(bytes: &[u8]) -> Result<Self, TtfError> {
        Self::parse_variation(bytes, &[])
    }

    /// Parse a variable font at a point of its design space, in user units
    /// per axis tag — `(*b"wght", 400.0)`. Axes not named sit at their
    /// default, values clamp to the axis's range, and a static font ignores
    /// them all.
    pub fn parse_variation(bytes: &[u8], variation: &[([u8; 4], f32)]) -> Result<Self, TtfError> {
        Self::parse_face(bytes, 0, variation)
    }

    /// One face of a font **collection** (`.ttc`), at a point of its design space.
    ///
    /// A collection is several table directories in one file sharing their tables, and it
    /// is how every Asian system font ships: Hiragino, Apple SD Gothic Neo, Hiragino Sans
    /// GB and the Kohinoor family are all `.ttc`. Without this they do not load at all,
    /// which made the `CFF` support unreachable for the fonts that most need it.
    ///
    /// `face` out of range is [`TtfError::UnsupportedFormat`]. A plain `.ttf`/`.otf` has
    /// one face, numbered 0.
    pub fn parse_face(
        bytes: &[u8],
        face: u32,
        variation: &[([u8; 4], f32)],
    ) -> Result<Self, TtfError> {
        if bytes.len() < 12 {
            return Err(TtfError::Truncated);
        }
        // `ttcf`: a count and then one `Offset32` per face, each pointing at a directory
        // of exactly the shape a single-face file has at offset 0.
        let base = if &bytes[0..4] == b"ttcf" {
            let n = u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
            if face >= n {
                return Err(TtfError::UnsupportedFormat);
            }
            let at = 12 + 4 * face as usize;
            let b = bytes.get(at..at + 4).ok_or(TtfError::Truncated)?;
            u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize
        } else {
            0
        };
        if base + 12 > bytes.len() {
            return Err(TtfError::Truncated);
        }
        let tag = u32::from_be_bytes([
            bytes[base],
            bytes[base + 1],
            bytes[base + 2],
            bytes[base + 3],
        ]);
        // 0x00010000 = TrueType, 0x74727565 = `true` (Apple), 0x4F54544F = OTTO, whose
        // outlines are PostScript charstrings in `CFF ` or `CFF2` rather than `glyf`.
        // Sixty of the 369 font files macOS ships are the last kind, and they are the
        // ones that matter for Asian and Indic text: every Hiragino face, Apple SD Gothic
        // Neo, Hiragino Sans GB and the whole Kohinoor family.
        if tag != 0x0001_0000 && tag != 0x7472_7565 && tag != 0x4F54_544F {
            return Err(TtfError::UnsupportedFormat);
        }
        let n_tables = u16::from_be_bytes([bytes[base + 4], bytes[base + 5]]) as usize;
        let mut tables: std::collections::HashMap<&'static [u8; 4], (usize, usize)> =
            std::collections::HashMap::new();
        let table_tags: [&'static [u8; 4]; 22] = [
            b"head", b"maxp", b"loca", b"glyf", b"cmap", b"hmtx", b"hhea", b"kern", b"fvar",
            b"avar", b"gvar", b"HVAR", b"GPOS", b"GDEF", b"name", b"trak", b"OS/2",
            b"CFF ", b"CFF2", b"GSUB",
            // Both only for [`crate::loaders::overlap`]: which composite a run of this
            // face's glyphs gets. `MERG` decides alone where a font has one, and `meta`'s
            // `dlng` names the languages the face was designed for.
            b"MERG", b"meta",
        ];
        for i in 0..n_tables {
            let off = base + 12 + i * 16;
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

        // A `CFF` font has neither, and reaches its outlines through charstrings instead.
        let cff_table = tables.get(b"CFF ").or_else(|| tables.get(b"CFF2")).copied();
        let (loca_off, glyf_off) = match (tables.get(b"loca"), tables.get(b"glyf")) {
            (Some(&(l, _)), Some(&(g, _))) => (l, g),
            _ if cff_table.is_some() => (0, 0),
            _ => return Err(TtfError::UnsupportedFormat),
        };

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

        // The design-space location, normalized.
        let table = |t: &[u8; 4]| tables.get(t).map(|&(off, _)| off);
        let (axes, instances) = table(b"fvar")
            .and_then(|f| super::ttf_var::parse_fvar(bytes, f, table(b"name")))
            .unwrap_or_default();
        let location: Vec<f32> = axes
            .iter()
            .map(|a| {
                variation
                    .iter()
                    .rev()
                    .find(|(t, _)| *t == a.tag)
                    .map_or(a.default, |&(_, v)| v.clamp(a.min, a.max))
            })
            .collect();
        let coords = super::ttf_var::normalize(bytes, &axes, table(b"avar"), variation);
        let varied = coords.iter().any(|&c| c != 0.0);
        let gvar = table(b"gvar")
            .filter(|_| varied)
            .and_then(|g| Gvar::new(bytes, g, coords.clone()));
        let hvar = table(b"HVAR")
            .filter(|_| varied)
            .and_then(|h| super::ttf_var::hvar_advance_deltas(bytes, h, &coords, n_glyphs));

        // Parse glyph outlines.
        let mut glyphs: Vec<TtfGlyph> = Vec::with_capacity(n_glyphs);
        // `CFF`: the outlines are little stack programs, drawn rather than read. The
        // advance still comes from `hmtx` — a charstring can carry its own width, and for
        // every font here the two agree, so `hmtx` stays the one answer and `HVAR` keeps
        // varying it.
        if let Some((cff_off, cff_len)) = cff_table {
            let cff = bytes
                .get(cff_off..cff_off.saturating_add(cff_len))
                .and_then(|d| super::cff::Cff::parse(d, units_per_em, &coords));
            for g_idx in 0..n_glyphs {
                let advance = advance_widths[g_idx] as f32;
                let advance = match &hvar {
                    Some(h) => advance + h[g_idx],
                    None => advance,
                };
                let mut sink = super::cff::PathSink::default();
                let drawn = cff
                    .as_ref()
                    .and_then(|c| c.draw(super::cff::GlyphId(g_idx as u16), &mut sink));
                glyphs.push(TtfGlyph {
                    shape: Shape {
                        outline: if drawn.is_some() { sink.path } else { Path::new() },
                        holes: Vec::new(),
                    },
                    advance_width: advance.round().clamp(0.0, u16::MAX as f32) as u16,
                });
            }
        } else {
        // `g_idx` is a glyph id, passed to `glyph_offset` as well as indexing.
        #[allow(clippy::needless_range_loop)]
        for g_idx in 0..n_glyphs {
            let off = glyph_offset(g_idx);
            let next_off = glyph_offset(g_idx + 1);
            if next_off <= off {
                // No outline (a space), but its advance still varies.
                let advance = advance_widths[g_idx] as f32;
                let varied = match (&hvar, &gvar) {
                    (Some(h), _) => advance + h[g_idx],
                    (None, Some(g)) => {
                        let pp = [
                            Vector2::new(0.0, 0.0),
                            Vector2::new(advance, 0.0),
                            Vector2::new(0.0, 0.0),
                            Vector2::new(0.0, 0.0),
                        ];
                        g.deltas(g_idx, &pp, Some(&[]))
                            .map_or(advance, |d| advance + d[1].x - d[0].x)
                    }
                    (None, None) => advance,
                };
                glyphs.push(TtfGlyph {
                    shape: Shape::new(),
                    advance_width: varied.round().clamp(0.0, u16::MAX as f32) as u16,
                });
                continue;
            }
            let advance = advance_widths[g_idx];
            let Some((mut contours, phantom)) =
                glyph_contours(bytes, &bounded_offset, g_idx, 0, gvar.as_ref(), advance)
            else {
                glyphs.push(TtfGlyph { shape: Shape::new(), advance_width: advance });
                continue;
            };
            // The phantom points carry the varied metrics: the origin (pp1)
            // and the advance (pp2). The outline is moved so the origin stays
            // at the pen.
            let mut varied_advance = advance as f32;
            if let Some(pp) = phantom {
                for c in &mut contours {
                    for (v, _) in c.iter_mut() {
                        v.x -= pp[0].x;
                    }
                }
                varied_advance += pp[1].x - pp[0].x;
            }
            if let Some(h) = &hvar {
                varied_advance = advance as f32 + h[g_idx];
            }
            glyphs.push(TtfGlyph {
                shape: shape_from_contours(&contours),
                advance_width: varied_advance.round().clamp(0.0, u16::MAX as f32) as u16,
            });
        }
        }
        let kerning = tables
            .get(b"kern")
            .and_then(|&(off, len)| parse_kern(bytes, off, len))
            .unwrap_or_default();
        let store = if varied {
            table(b"GDEF")
                .and_then(|g| super::gpos::gdef_var_store(bytes, g))
                .and_then(|s| ItemVariationStore::new(bytes, s, &coords))
        } else {
            None
        };
        let gpos_kerning = table(b"GPOS")
            .and_then(|g| super::gpos::parse(bytes, g, store.as_ref()))
            .unwrap_or_default();
        // Kept only if there is mark positioning to read them for.
        let mark_bytes = (|| {
            let (go, gl) = *tables.get(b"GPOS")?;
            let gpos = bytes.get(go..go.saturating_add(gl))?;
            let gdef = tables
                .get(b"GDEF")
                .and_then(|&(o, l)| bytes.get(o..o.saturating_add(l)))
                .unwrap_or(&[]);
            super::mark::Marks::parse(gpos, gdef, None)?;
            Some((gpos.to_vec(), gdef.to_vec()))
        })()
        .unwrap_or_default();
        // Which composite this face's runs get — four tables, once per face. The `cmap`
        // is already a map here, so the range test is a lookup.
        let slice = |tag: &[u8; 4]| -> Option<&[u8]> {
            let &(o, l) = tables.get(tag)?;
            bytes.get(o..o.saturating_add(l))
        };
        let composites_run = super::overlap::composites_run(
            slice(b"MERG"),
            slice(b"meta"),
            slice(b"name"),
            slice(b"GPOS").unwrap_or(&[]),
            |cp| cmap.contains_key(&cp),
        );
        Ok(TtfFont {
            units_per_em,
            glyphs,
            cmap,
            ascender,
            descender,
            line_gap,
            kerning,
            gpos_kerning,
            coords: coords.clone(),
            gpos: mark_bytes.0,
            gdef: mark_bytes.1,
            gsub: table(b"GSUB")
                .and_then(|o| {
                    let len = tables.get(b"GSUB").map(|&(_, l)| l)?;
                    bytes.get(o..o.saturating_add(len)).map(<[u8]>::to_vec)
                })
                .unwrap_or_default(),
            axes,
            instances,
            location,
            tracking: table(b"trak").and_then(|t| parse_trak(bytes, t)).unwrap_or_default(),
            weight_class: table(b"OS/2")
                .and_then(|o| bytes.get(o + 4..o + 6))
                .map(|b| u16::from_be_bytes([b[0], b[1]]))
                .filter(|&w| (1..=1000).contains(&w))
                .unwrap_or(400),
            composites_run,
        })
    }

    /// Extra advance per glyph at `size` pixels per em, in font units: the
    /// `trak` table's normal track, between its sizes linearly and held at
    /// its ends. Apple's fonts are spaced to be read with it — SF opens up
    /// a little at text sizes and closes up at display ones — and CoreText
    /// applies it by default.
    pub fn tracking_at(&self, size: f32) -> f32 {
        let t = &self.tracking;
        match t.len() {
            0 => 0.0,
            1 => t[0].1,
            _ => {
                if size <= t[0].0 {
                    return t[0].1;
                }
                match t.windows(2).find(|w| size <= w[1].0) {
                    Some(w) => w[0].1 + (size - w[0].0) / (w[1].0 - w[0].0) * (w[1].1 - w[0].1),
                    None => t[t.len() - 1].1,
                }
            }
        }
    }

    /// Whether the font has variation axes.
    pub fn is_variable(&self) -> bool {
        !self.axes.is_empty()
    }

    /// The `wght` coordinate for a CSS font weight (100–900); `None` for a
    /// font without a weight axis. See [`wght_for_css_weight`](super::ttf_var::wght_for_css_weight).
    pub fn wght_for_css_weight(&self, weight: f32) -> Option<f32> {
        super::ttf_var::wght_for_css_weight(&self.axes, &self.instances, weight)
    }

    /// A font's variation axes and named instances without reading its
    /// glyphs — what choosing a location needs before
    /// [`parse_variation`](Self::parse_variation). Empty for a static font.
    pub fn variation_info(bytes: &[u8]) -> (Vec<VariationAxis>, Vec<NamedInstance>) {
        let rd16 = |p: usize| bytes.get(p..p + 2).map(|b| u16::from_be_bytes([b[0], b[1]]) as usize);
        let rd32 = |p: usize| bytes.get(p..p + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize);
        let find = |tag: &[u8; 4]| -> Option<usize> {
            (0..rd16(4)?).find_map(|i| {
                let r = 12 + 16 * i;
                (bytes.get(r..r + 4)? == tag).then(|| rd32(r + 8)).flatten()
            })
        };
        find(b"fvar")
            .and_then(|f| super::ttf_var::parse_fvar(bytes, f, find(b"name")))
            .unwrap_or_default()
    }

    /// The kerning between two glyphs, in font units. Zero when the pair has
    /// none, which is the overwhelming majority of pairs.
    ///
    /// From `kern` where the font has the pair there, else from `GPOS`.
    pub fn kern(&self, left: u16, right: u16) -> f32 {
        match self.kerning.get(&(left, right)) {
            Some(&k) => k as f32,
            None => self.gpos_kerning.kern(left, right),
        }
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

/// A glyph path's curves as they were built: lines, and quadratics whose
/// control point is recovered from the midpoint (`B(½) = ¼P₀ + ½C + ¼P₁`).
/// Each contour starts where the previous curve did not end.
fn glyph_segments(p: &Path) -> Vec<(bool, Vector2, Option<Vector2>, Vector2)> {
    let mut out = Vec::with_capacity(p.curve_path.curves.len());
    let mut last: Option<Vector2> = None;
    for c in p.curve_path.curves.iter() {
        let (a, m, b) = (c.get_point(0.0), c.get_point(0.5), c.get_point(1.0));
        let starts = last.is_none_or(|l| (l.x - a.x).abs() > 1e-4 || (l.y - a.y).abs() > 1e-4);
        let control = Vector2::new(2.0 * m.x - 0.5 * (a.x + b.x), 2.0 * m.y - 0.5 * (a.y + b.y));
        let straight = (m.x - 0.5 * (a.x + b.x)).abs() < 1e-3 && (m.y - 0.5 * (a.y + b.y)).abs() < 1e-3;
        out.push((starts, a, (!straight).then_some(control), b));
        last = Some(b);
    }
    out
}

/// `p` scaled, curve for curve.
fn scaled_path(p: &Path, scale: f32) -> Path {
    let mut np = Path::new();
    let k = |v: Vector2| Vector2::new(v.x * scale, v.y * scale);
    for (starts, a, control, b) in glyph_segments(p) {
        if starts {
            np.move_to(k(a));
        }
        match control {
            Some(c) => np.quadratic_curve_to(k(c), k(b)),
            None => np.line_to(k(b)),
        };
    }
    np
}

/// `p` as a three.js `ShapePath`, scaled and moved to `(x, y)`.
fn shape_path(p: &Path, scale: f64, x: f64, y: f64) -> crate::curves::shapes::ShapePath {
    let mut sp = crate::curves::shapes::ShapePath::new();
    let k = |v: Vector2| (v.x as f64 * scale + x, v.y as f64 * scale + y);
    for (starts, a, control, b) in glyph_segments(p) {
        if starts {
            let (ax, ay) = k(a);
            sp.move_to(ax, ay);
        }
        let (bx, by) = k(b);
        match control {
            Some(c) => {
                let (cx, cy) = k(c);
                sp.quadratic_curve_to(cx, cy, bx, by);
            }
            None => {
                sp.line_to(bx, by);
            }
        }
    }
    sp
}

/// `trak`'s horizontal normal track (track value 0) as `(size, value)`.
fn parse_trak(b: &[u8], off: usize) -> Option<Vec<(f32, f32)>> {
    let rd16 = |p: usize| Some(u16::from_be_bytes([*b.get(p)?, *b.get(p + 1)?]));
    let rd32 = |p: usize| Some(u32::from_be_bytes([*b.get(p)?, *b.get(p + 1)?, *b.get(p + 2)?, *b.get(p + 3)?]));
    let horiz = rd16(off + 6)? as usize;
    if horiz == 0 {
        return None;
    }
    let data = off + horiz;
    let n_tracks = rd16(data)? as usize;
    let n_sizes = rd16(data + 2)? as usize;
    let sizes = off + rd32(data + 4)? as usize;
    for t in 0..n_tracks {
        let e = data + 8 + 8 * t;
        if rd32(e)? != 0 {
            continue;
        }
        let values = off + rd16(e + 6)? as usize;
        return (0..n_sizes)
            .map(|k| {
                let size = rd32(sizes + 4 * k)? as i32 as f32 / 65536.0;
                Some((size, rd16(values + 2 * k)? as i16 as f32))
            })
            .collect();
    }
    None
}

/// One contour's points in font units, each with whether it is on the curve.
type Contour = Vec<(Vector2, bool)>;

/// The four phantom points' deltas, when a glyph was varied.
type Phantom = Option<[Vector2; 4]>;

/// Glyph `gid`'s contours: its own for a simple glyph; for a composite one,
/// each component's, placed by its offset and scaled or transformed as its
/// flags say. `depth` guards against a font whose components refer back to
/// themselves.
///
/// With `gvar`, the points (a composite's component offsets) are moved to
/// its location, and the deltas of the four phantom points come back too —
/// the origin, the advance, the top and the bottom — for the caller to vary
/// the metrics with. `advance` is the unvaried advance, the phantom points'
/// original position.
fn glyph_contours(
    bytes: &[u8],
    glyph_offset: &dyn Fn(usize) -> usize,
    gid: usize,
    depth: u32,
    gvar: Option<&Gvar>,
    advance: u16,
) -> Option<(Vec<Contour>, Phantom)> {
    let off = glyph_offset(gid);
    if glyph_offset(gid + 1) <= off || off + 10 > bytes.len() || depth > 8 {
        return None;
    }
    let phantoms = [
        Vector2::new(0.0, 0.0),
        Vector2::new(advance as f32, 0.0),
        Vector2::new(0.0, 0.0),
        Vector2::new(0.0, 0.0),
    ];
    let n_contours = i16::from_be_bytes([bytes[off], bytes[off + 1]]);
    if n_contours >= 0 {
        let mut contours = parse_simple_contours(bytes, off)?;
        let Some(gvar) = gvar else { return Some((contours, None)) };
        let mut points: Vec<Vector2> = contours.iter().flatten().map(|&(v, _)| v).collect();
        let mut ends = Vec::with_capacity(contours.len());
        let mut n = 0;
        for c in &contours {
            n += c.len();
            ends.push(n - 1);
        }
        points.extend(phantoms);
        let Some(d) = gvar.deltas(gid, &points, Some(&ends)) else {
            return Some((contours, None));
        };
        let mut k = 0;
        for c in &mut contours {
            for (v, _) in c.iter_mut() {
                v.x += d[k].x;
                v.y += d[k].y;
                k += 1;
            }
        }
        return Some((contours, Some([d[n], d[n + 1], d[n + 2], d[n + 3]])));
    }
    let rd16 = |p: usize| -> Option<i16> {
        Some(i16::from_be_bytes([*bytes.get(p)?, *bytes.get(p + 1)?]))
    };
    let f2dot14 = |p: usize| rd16(p).map(|v| v as f32 / 16384.0);
    // Each component's offset, flags and transform, then its contours.
    struct Component {
        offset: Vector2,
        xy: bool,
        m: (f32, f32, f32, f32),
        contours: Vec<Contour>,
    }
    let mut components: Vec<Component> = Vec::new();
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
        let contours = glyph_contours(bytes, glyph_offset, component, depth + 1, gvar, 0)
            .map(|(c, _)| c)
            .unwrap_or_default();
        components.push(Component {
            offset: Vector2::new(dx, dy),
            xy: flags & 0x0002 != 0,
            m,
            contours,
        });
        // MORE_COMPONENTS
        if flags & 0x0020 == 0 {
            break;
        }
    }
    // A composite's "points" are its components' offsets, then the phantoms.
    let mut phantom = None;
    if let Some(gvar) = gvar {
        let mut points: Vec<Vector2> = components.iter().map(|c| c.offset).collect();
        points.extend(phantoms);
        if let Some(d) = gvar.deltas(gid, &points, None) {
            let n = components.len();
            for (c, d) in components.iter_mut().zip(&d) {
                if c.xy {
                    c.offset.x += d.x;
                    c.offset.y += d.y;
                }
            }
            phantom = Some([d[n], d[n + 1], d[n + 2], d[n + 3]]);
        }
    }
    let mut out = Vec::new();
    for c in components {
        let (m, o) = (c.m, c.offset);
        for contour in c.contours {
            out.push(
                contour
                    .into_iter()
                    .map(|(v, on)| {
                        (Vector2::new(m.0 * v.x + m.2 * v.y + o.x, m.1 * v.x + m.3 * v.y + o.y), on)
                    })
                    .collect(),
            );
        }
    }
    Some((out, phantom))
}

/// The font's best Unicode `cmap` subtable, as fontTools' `getBestCmap`
/// picks one: full-range Unicode (format 12) before BMP-only (format 4),
/// Windows or Unicode platform alike. Many Apple fonts — Geneva, Monaco,
/// Zapfino — map only through the Unicode platform, and a reader of
/// Windows subtables alone found every character missing. A symbol font
/// (Windows encoding 0) maps its characters at U+F020–U+F0FF; they are
/// also mapped at the codes they stand for, as browsers do.
fn parse_cmap(bytes: &[u8], cmap_off: usize) -> Option<std::collections::HashMap<u32, u16>> {
    let rd16 = |p: usize| Some(u16::from_be_bytes([*bytes.get(p)?, *bytes.get(p + 1)?]));
    let rd32 = |p: usize| Some(u32::from_be_bytes([*bytes.get(p)?, *bytes.get(p + 1)?, *bytes.get(p + 2)?, *bytes.get(p + 3)?]));
    let n = rd16(cmap_off + 2)? as usize;
    let records: Vec<(u16, u16, usize)> = (0..n)
        .filter_map(|i| {
            let rec = cmap_off + 4 + i * 8;
            Some((rd16(rec)?, rd16(rec + 2)?, cmap_off + rd32(rec + 4)? as usize))
        })
        .collect();
    const PREFERENCE: [(u16, u16); 9] = [(3, 10), (0, 6), (0, 4), (3, 1), (0, 3), (0, 2), (0, 1), (0, 0), (3, 0)];
    let best = PREFERENCE.iter().find_map(|&(p, e)| {
        records
            .iter()
            .find(|r| r.0 == p && r.1 == e && matches!(rd16(r.2), Some(4) | Some(12)))
            .copied()
    });
    let Some((_platform, _encoding, sub_off)) = best else {
        // Nothing Unicode: a Macintosh Roman table (format 0 or 6), as
        // HarfBuzz falls back to — Hoefler Text Ornaments has only that.
        let (_, _, off) = records.iter().find(|r| r.0 == 1 && r.1 == 0 && matches!(rd16(r.2), Some(0) | Some(6)))?;
        return parse_cmap_mac_roman(bytes, *off);
    };
    let mut map = match rd16(sub_off)? {
        12 => parse_cmap_12(bytes, sub_off)?,
        _ => parse_cmap_4(bytes, sub_off)?,
    };
    // A symbol font encodes its glyphs at `0xF000 + byte`, so each one is reachable by
    // the character that byte stands for as well.
    //
    // This is **not** gated on the subtable being `(3, 0)`, though that is where the
    // convention comes from: `Hoefler Text Ornaments` puts the same `0xF0xx` content in a
    // `(0, 3)` table, which the preference order picks first — so gating on the platform
    // skipped the aliasing and left every letter unmapped. It was the one font of 178
    // that disagreed with HarfBuzz. `or_insert` never displaces a real mapping, so
    // applying it to any table that has such keys cannot cost anything.
    let symbols: Vec<(u32, u16)> = map
        .iter()
        .filter(|(&c, _)| (0xF000..=0xF0FF).contains(&c))
        .map(|(&c, &g)| (c - 0xF000, g))
        .collect();
    for (c, g) in symbols {
        map.entry(c).or_insert(g);
    }
    Some(map)
}

/// A Macintosh Roman subtable — format 0, a glyph per byte, or format 6, a
/// run of them — with the bytes read as the characters they encode.
fn parse_cmap_mac_roman(bytes: &[u8], off: usize) -> Option<std::collections::HashMap<u32, u16>> {
    let rd16 = |p: usize| Some(u16::from_be_bytes([*bytes.get(p)?, *bytes.get(p + 1)?]));
    let (first, glyphs): (u16, Vec<u16>) = match rd16(off)? {
        0 => (0, (0..256).map(|k| bytes.get(off + 6 + k).map(|&g| g as u16)).collect::<Option<_>>()?),
        _ => {
            let n = rd16(off + 8)? as usize;
            (rd16(off + 6)?, (0..n).map(|k| rd16(off + 10 + 2 * k)).collect::<Option<_>>()?)
        }
    };
    let mut map = std::collections::HashMap::new();
    for (k, g) in glyphs.into_iter().enumerate() {
        let byte = first as usize + k;
        if g == 0 || byte > 0xFF {
            continue;
        }
        let c = if byte < 0x80 { byte as u32 } else { MAC_ROMAN[byte - 0x80] as u32 };
        map.insert(c, g);
    }
    Some(map)
}

/// Macintosh Roman's upper half in Unicode, from byte 0x80.
const MAC_ROMAN: [u16; 128] = [
    0x00C4, 0x00C5, 0x00C7, 0x00C9, 0x00D1, 0x00D6, 0x00DC, 0x00E1,
    0x00E0, 0x00E2, 0x00E4, 0x00E3, 0x00E5, 0x00E7, 0x00E9, 0x00E8,
    0x00EA, 0x00EB, 0x00ED, 0x00EC, 0x00EE, 0x00EF, 0x00F1, 0x00F3,
    0x00F2, 0x00F4, 0x00F6, 0x00F5, 0x00FA, 0x00F9, 0x00FB, 0x00FC,
    0x2020, 0x00B0, 0x00A2, 0x00A3, 0x00A7, 0x2022, 0x00B6, 0x00DF,
    0x00AE, 0x00A9, 0x2122, 0x00B4, 0x00A8, 0x2260, 0x00C6, 0x00D8,
    0x221E, 0x00B1, 0x2264, 0x2265, 0x00A5, 0x00B5, 0x2202, 0x2211,
    0x220F, 0x03C0, 0x222B, 0x00AA, 0x00BA, 0x03A9, 0x00E6, 0x00F8,
    0x00BF, 0x00A1, 0x00AC, 0x221A, 0x0192, 0x2248, 0x2206, 0x00AB,
    0x00BB, 0x2026, 0x00A0, 0x00C0, 0x00C3, 0x00D5, 0x0152, 0x0153,
    0x2013, 0x2014, 0x201C, 0x201D, 0x2018, 0x2019, 0x00F7, 0x25CA,
    0x00FF, 0x0178, 0x2044, 0x20AC, 0x2039, 0x203A, 0xFB01, 0xFB02,
    0x2021, 0x00B7, 0x201A, 0x201E, 0x2030, 0x00C2, 0x00CA, 0x00C1,
    0x00CB, 0x00C8, 0x00CD, 0x00CE, 0x00CF, 0x00CC, 0x00D3, 0x00D4,
    0xF8FF, 0x00D2, 0x00DA, 0x00DB, 0x00D9, 0x0131, 0x02C6, 0x02DC,
    0x00AF, 0x02D8, 0x02D9, 0x02DA, 0x00B8, 0x02DD, 0x02DB, 0x02C7,
];

/// A format 12 subtable: groups of consecutive codes to consecutive glyphs.
fn parse_cmap_12(bytes: &[u8], off: usize) -> Option<std::collections::HashMap<u32, u16>> {
    let rd32 = |p: usize| Some(u32::from_be_bytes([*bytes.get(p)?, *bytes.get(p + 1)?, *bytes.get(p + 2)?, *bytes.get(p + 3)?]));
    let groups = rd32(off + 12)? as usize;
    let mut map = std::collections::HashMap::new();
    for g in 0..groups {
        let r = off + 16 + 12 * g;
        let (start, end, glyph) = (rd32(r)?, rd32(r + 4)?, rd32(r + 8)?);
        if end < start || end > 0x10FFFF {
            continue;
        }
        for c in start..=end {
            let gid = glyph + (c - start);
            if gid != 0 && gid <= u16::MAX as u32 {
                map.insert(c, gid as u16);
            }
        }
    }
    Some(map)
}

/// A format 4 subtable: BMP segments with deltas or a glyph array.
fn parse_cmap_4(bytes: &[u8], sub_off: usize) -> Option<std::collections::HashMap<u32, u16>> {
    if sub_off + 14 > bytes.len() {
        return None;
    }
    let seg_count_x2 = u16::from_be_bytes([bytes[sub_off + 6], bytes[sub_off + 7]]) as usize;
    let seg_count = seg_count_x2 / 2;
    let end_off = sub_off + 14;
    let start_off = end_off + seg_count_x2 + 2;
    let id_delta_off = start_off + seg_count_x2;
    let id_range_off = id_delta_off + seg_count_x2;
    if id_range_off + seg_count_x2 > bytes.len() {
        return None;
    }
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
            // Glyph 0 is "missing": recorded, a fallback face would never be
            // asked for the character.
            if gid != 0 {
                map.insert(c as u32, gid);
            }
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
        // A contour of control points only (a circle drawn as four
        // quadratics, say) starts at the implied on-curve point between its
        // last and first; starting at a control point instead bends every
        // one of its curves through the wrong place.
        let implied;
        let pts = if pts.iter().any(|p| p.1) {
            pts
        } else {
            let (a, b) = (pts[pts.len() - 1].0, pts[0].0);
            implied = std::iter::once((Vector2::new((a.x + b.x) * 0.5, (a.y + b.y) * 0.5), true))
                .chain(pts.iter().copied())
                .collect::<Vec<_>>();
            &implied
        };
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
                    font.kern(gid(a), gid(b)) < 0.0,
                    "{path}: {a}{b} should tighten, got {}",
                    font.kern(gid(a), gid(b))
                );
            }
            assert_eq!(
                font.kern(gid('o'), gid('o')),
                0.0,
                "{path}: two round letters need no kerning"
            );
        }
        if checked == 0 {
            eprintln!("skipping: no kerned system font found");
        }
    }
}

#[cfg(test)]
mod variation_tests {
    use super::*;

    /// The outline's bounds, sampled finely enough for a hundredth of a unit.
    fn bounds(g: &TtfGlyph) -> [f32; 4] {
        let mut b = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
        for c in g.shape.outline.curve_path.curves.iter() {
            for i in 0..=256 {
                let p = c.get_point(i as f32 / 256.0);
                b = [b[0].min(p.x), b[1].min(p.y), b[2].max(p.x), b[3].max(p.y)];
            }
        }
        b
    }

    fn system_font(name: &str) -> Option<Vec<u8>> {
        std::fs::read(format!("/System/Library/Fonts/{name}")).ok()
    }

    /// Varied outlines, advances and kerning against fontTools and HarfBuzz.
    ///
    /// The numbers are theirs, not this reader's: bounds and advances from
    /// fontTools' `getGlyphSet(location=…)` with a `BoundsPen`, kerning from
    /// `hb-shape --script=latn` (the pair's first advance minus the glyph's
    /// own). SF Compact's default instance is Black, so a reader that skips
    /// `gvar` fails every one of these by a wide margin. Skipped where the
    /// fonts are not installed.
    #[test]
    fn variable_fonts_match_fonttools_and_harfbuzz() {
        let gid = |f: &TtfFont, c: char| f.cmap.get(&(c as u32)).copied().unwrap_or(0);
        let close = |a: [f32; 4], b: [f32; 4]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 0.02);
        if let Some(bytes) = system_font("SFCompact.ttf") {
            let f = TtfFont::parse_variation(&bytes, &[(*b"wght", 457.9)]).unwrap();
            let a = &f.glyphs[gid(&f, 'A') as usize];
            // fontTools: width 1258.53, bounds (51.11, 0, 1207.43, 1365).
            assert_eq!(a.advance_width, 1259);
            assert!(close(bounds(a), [51.11, 0.0, 1207.43, 1365.0]), "{:?}", bounds(a));
            let six = &f.glyphs[gid(&f, '6') as usize];
            assert!(close(bounds(six), [136.33, -27.52, 1100.79, 1390.61]), "{:?}", bounds(six));
            // HarfBuzz: AV 1212 − 1299 = −87 (a class pair); 67 1264 − 1268
            // = −4, a glyph pair whose variation delta is found from its
            // pair set, not its subtable.
            assert!((f.kern(gid(&f, 'A'), gid(&f, 'V')) + 87.0).abs() < 0.5);
            assert!((f.kern(gid(&f, '6'), gid(&f, '7')) + 4.0).abs() < 0.5);

            let bold = TtfFont::parse_variation(&bytes, &[(*b"wght", 790.0), (*b"opsz", 20.0)]).unwrap();
            let a = &bold.glyphs[gid(&bold, 'A') as usize];
            assert_eq!(a.advance_width, 1298);
            assert!(close(bounds(a), [9.45, 0.0, 1288.45, 1365.0]), "{:?}", bounds(a));
        }
        if let Some(bytes) = system_font("SFNS.ttf") {
            // Three axes off their defaults, through `avar` and `HVAR`.
            let at = [(*b"wght", 300.0), (*b"wdth", 80.0), (*b"opsz", 60.0)];
            let f = TtfFont::parse_variation(&bytes, &at).unwrap();
            let g = &f.glyphs[gid(&f, 'g') as usize];
            assert_eq!(g.advance_width, 975);
            assert!(close(bounds(g), [62.88, -366.87, 871.2, 1055.0]), "{:?}", bounds(g));
        }
    }

    /// CSS weights land on the font's named instances, which for SF are not
    /// at the CSS numbers; and a static font has no axis to set.
    #[test]
    fn css_weights_go_through_the_named_instances() {
        let Some(bytes) = system_font("SFCompact.ttf") else { return };
        let (axes, instances) = TtfFont::variation_info(&bytes);
        let w = |css| super::super::ttf_var::wght_for_css_weight(&axes, &instances, css).unwrap();
        assert!((w(400.0) - 457.9).abs() < 0.01);
        assert!((w(700.0) - 790.0).abs() < 0.01);
        assert!((w(900.0) - 1000.0).abs() < 0.01);
        // The italic family's names carry a suffix.
        if let Some(italic) = system_font("SFCompactItalic.ttf") {
            let (axes, instances) = TtfFont::variation_info(&italic);
            let w = |css| super::super::ttf_var::wght_for_css_weight(&axes, &instances, css).unwrap();
            assert!((w(400.0) - 457.9).abs() < 0.01);
            assert!((w(700.0) - 790.0).abs() < 0.01);
        }
        // Between named weights, between their coordinates.
        assert!(w(550.0) > 556.0 && w(550.0) < 656.2);
        if let Some(arial) = system_font("Supplemental/Arial.ttf") {
            assert!(TtfFont::parse(&arial).unwrap().wght_for_css_weight(400.0).is_none());
        }
    }

    /// Apple's tracking: SF opens up at text sizes and closes up at display
    /// ones, by the table's values in between them.
    #[test]
    fn tracking_follows_the_trak_table() {
        let Some(bytes) = system_font("SFCompact.ttf") else { return };
        let f = TtfFont::parse(&bytes).unwrap();
        assert_eq!(f.tracking_at(14.0), 28.0);
        assert_eq!(f.tracking_at(16.0), 0.0);
        assert_eq!(f.tracking_at(14.5), 18.0);
        // Held at the ends.
        assert_eq!(f.tracking_at(2.0), 220.0);
        assert_eq!(f.tracking_at(400.0), -50.0);
    }
}

#[cfg(test)]
mod shape_tests {
    use super::*;

    /// A contour with no on-curve point is a closed run of quadratics
    /// through the midpoints of its control points: four control points at
    /// the corners of a square make a round shape through the side
    /// midpoints, inside the square and outside the diamond those make.
    #[test]
    fn an_all_control_point_contour_runs_through_the_midpoints() {
        let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
        let contour: Contour = corners.iter().map(|&(x, y)| (Vector2::new(x, y), false)).collect();
        let shape = shape_from_contours(&[contour]);
        let curves = &shape.outline.curve_path.curves;
        assert_eq!(curves.len(), 4);
        for c in curves.iter() {
            let (a, m) = (c.get_point(0.0), c.get_point(0.5));
            // Each curve starts on a side midpoint: one coordinate 0, the
            // other ±1.
            assert!((a.x.abs() + a.y.abs() - 1.0).abs() < 1e-6 && (a.x * a.y).abs() < 1e-6, "{a:?}");
            // And bulges halfway to its corner: |x| = |y| = 0.75.
            assert!((m.x.abs() - 0.75).abs() < 1e-6 && (m.y.abs() - 0.75).abs() < 1e-6, "{m:?}");
        }
    }

    /// Signed area of a closed polygon.
    fn area(points: &[crate::curves::shapes::Vec2d]) -> f64 {
        let n = points.len();
        (0..n)
            .map(|i| {
                let (p, q) = (points[i], points[(i + 1) % n]);
                p.x * q.y - q.x * p.y
            })
            .sum::<f64>()
            * 0.5
    }

    /// Glyphs come out as shapes with their holes — an `o` one shape with a
    /// hole, an `i` two shapes, a `B` one with two, `%` three (two rings and
    /// the bar) with two — and
    /// the ink between outlines and holes is the glyph's area as fontTools'
    /// `AreaPen` measures it at the same location. Skipped without SF
    /// Compact.
    #[test]
    fn text_shapes_separate_outlines_from_holes() {
        let Ok(bytes) = std::fs::read("/System/Library/Fonts/SFCompact.ttf") else { return };
        let f = TtfFont::parse_variation(&bytes, &[(*b"wght", 457.9)]).unwrap();
        let upem = f.units_per_em as f64;
        for (c, shapes, holes, fonttools) in [
            ('o', 1, 1, 467437.8),
            ('i', 2, 0, 241558.1),
            ('B', 1, 2, 720934.5),
            ('8', 1, 2, 644189.8),
            ('%', 3, 2, 706750.0),
        ] {
            let s = f.generate_shapes(&c.to_string(), upem);
            assert_eq!(s.len(), shapes, "{c}: shapes");
            assert_eq!(s.iter().map(|s| s.holes.len()).sum::<usize>(), holes, "{c}: holes");
            let ink: f64 = s
                .iter()
                .map(|s| {
                    let p = s.extract_points(64);
                    area(&p.shape).abs() - p.holes.iter().map(|h| area(h).abs()).sum::<f64>()
                })
                .sum();
            assert!((ink / fonttools - 1.0).abs() < 1e-4, "{c}: {ink} against {fonttools}");
        }
    }

    /// Kerned, in ems times the size, and a newline drops a line.
    #[test]
    fn text_shapes_are_laid_out_like_the_caption_face() {
        let Ok(bytes) = std::fs::read("/System/Library/Fonts/SFCompact.ttf") else { return };
        let f = TtfFont::parse_variation(&bytes, &[(*b"wght", 457.9)]).unwrap();
        let left = |text: &str| -> Vec<f64> {
            f.create_paths(text, 1.0)
                .iter()
                .map(|p| p.to_shapes(false)[0].extract_points(4).shape.iter().map(|v| v.x).fold(f64::MAX, f64::min))
                .collect()
        };
        let gid = |c: char| f.cmap[&(c as u32)];
        let upem = f.units_per_em as f64;
        let av = left("AV");
        let v_alone = left("V")[0];
        let expected =
            (f.glyphs[gid('A') as usize].advance_width as f64 + f.kern(gid('A'), gid('V')) as f64) / upem + v_alone;
        assert!((av[1] - expected).abs() < 1e-6, "{} against {expected}", av[1]);
        let two = f.create_paths("A\nA", 1.0);
        let top = |p: &crate::curves::shapes::ShapePath| {
            p.to_shapes(false)[0].extract_points(4).shape.iter().map(|v| v.y).fold(f64::MIN, f64::max)
        };
        let drop = top(&two[0]) - top(&two[1]);
        let line = (f.ascender as f64 - f.descender as f64 + f.line_gap as f64) / upem;
        assert!((drop - line).abs() < 1e-6);
    }
}

#[cfg(test)]
mod cmap_tests {
    use super::*;

    /// Characters past the BMP come from a format 12 subtable: DejaVu maps
    /// U+10300 (Old Italic A) to glyph 5370, as fontTools reads it.
    #[test]
    fn full_range_unicode_maps_past_the_bmp() {
        let font = TtfFont::parse(include_bytes!("../openscad/DejaVuSans.ttf")).unwrap();
        assert_eq!(font.cmap.get(&0x10300), Some(&5370));
        // Glyph 0 is "missing", not a mapping.
        assert!(font.cmap.values().all(|&g| g != 0));
    }

    /// Fonts that map only through the Unicode platform read as fonts, not
    /// as a box for every character: Geneva (format 12) and Monaco (format
    /// 4), both with `A` at glyph 36 by fontTools. Skipped where absent.
    #[test]
    fn unicode_platform_only_fonts_have_their_characters() {
        for name in ["Geneva.ttf", "Monaco.ttf"] {
            let Ok(bytes) = std::fs::read(format!("/System/Library/Fonts/{name}")) else { continue };
            let font = TtfFont::parse(&bytes).unwrap();
            assert_eq!(font.cmap.get(&('A' as u32)), Some(&36), "{name}");
        }
    }
}
