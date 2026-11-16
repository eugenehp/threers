//! An sfnt font file (TrueType outlines, or one face of a collection): its
//! table directory and the tables glyphs are loaded and laid out from.

use std::collections::BTreeMap;

/// Why a font cannot be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The data ends before what it says it holds.
    Truncated,
    /// Not an sfnt font, or a face index past the collection.
    NotAFont,
    /// A table glyphs need is missing.
    MissingTable(&'static str),
    /// Outlines that are not TrueType quadratics (CFF).
    Unsupported(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Truncated => write!(f, "truncated font data"),
            Error::NotAFont => write!(f, "not an sfnt font"),
            Error::MissingTable(t) => write!(f, "missing table '{t}'"),
            Error::Unsupported(what) => write!(f, "unsupported: {what}"),
        }
    }
}

impl std::error::Error for Error {}

/// Big-endian reads that fail on short data.
#[derive(Clone, Copy)]
pub(crate) struct Reader<'a> {
    pub(crate) data: &'a [u8],
}

impl<'a> Reader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Reader { data }
    }

    pub(crate) fn u8(&self, at: usize) -> Result<u8, Error> {
        self.data.get(at).copied().ok_or(Error::Truncated)
    }

    pub(crate) fn i8(&self, at: usize) -> Result<i8, Error> {
        Ok(self.u8(at)? as i8)
    }

    pub(crate) fn u16(&self, at: usize) -> Result<u16, Error> {
        let b = self.data.get(at..at + 2).ok_or(Error::Truncated)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    pub(crate) fn i16(&self, at: usize) -> Result<i16, Error> {
        Ok(self.u16(at)? as i16)
    }

    pub(crate) fn u32(&self, at: usize) -> Result<u32, Error> {
        let b = self.data.get(at..at + 4).ok_or(Error::Truncated)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub(crate) fn slice(&self, at: usize, len: usize) -> Result<&'a [u8], Error> {
        self.data.get(at..at + len).ok_or(Error::Truncated)
    }
}

/// One face of a font file.
pub struct Face {
    data: Vec<u8>,
    tables: BTreeMap<[u8; 4], (usize, usize)>,
    pub units_per_em: u16,
    /// `head.flags`
    pub head_flags: u16,
    /// `head.indexToLocFormat`: 1 for 32-bit offsets.
    pub index_to_loc_format: i16,
    pub num_glyphs: u16,
    /// `maxp`, of the TrueType interpreter.
    pub max_twilight_points: u16,
    pub max_storage: u16,
    pub max_function_defs: u16,
    pub max_instruction_defs: u16,
    pub max_stack_elements: u16,
    pub max_size_of_instructions: u16,
    pub max_points: u16,
    pub max_contours: u16,
    pub max_component_points: u16,
    pub max_component_contours: u16,
    pub max_component_depth: u16,
    /// `hhea`
    pub ascender: i16,
    pub descender: i16,
    pub line_gap: i16,
    pub advance_width_max: u16,
    pub number_of_h_metrics: u16,
    /// `OS/2` sTypo/usWin metrics, where the face has the table.
    pub os2: Option<Os2>,
    /// The Unicode subtable chosen for `glyph_index`.
    cmap: Option<(usize, u16)>,
    /// Whether the outlines are CFF (read for their tables only; their
    /// glyphs do not load).
    pub cff: bool,
}

/// The `OS/2` fields metrics and matching use.
#[derive(Clone, Copy, Debug)]
pub struct Os2 {
    pub version: u16,
    pub weight_class: u16,
    pub width_class: u16,
    pub fs_selection: u16,
    pub typo_ascender: i16,
    pub typo_descender: i16,
    pub typo_line_gap: i16,
    pub win_ascent: u16,
    pub win_descent: u16,
}

impl Face {
    /// The face `index` of a font file (0 for a single font).
    pub fn new(data: Vec<u8>, index: u32) -> Result<Face, Error> {
        let r = Reader::new(&data);
        let mut offset = 0usize;

        if r.slice(0, 4)? == b"ttcf" {
            let count = r.u32(8)?;

            if index >= count {
                return Err(Error::NotAFont);
            }

            offset = r.u32(12 + 4 * index as usize)? as usize;
        }

        let cff = match r.u32(offset)? {
            0x0001_0000 | 0x7472_7565 => false,
            0x4F54_544F => true,
            _ => return Err(Error::NotAFont),
        };

        let count = r.u16(offset + 4)? as usize;
        let mut tables = BTreeMap::new();

        for i in 0..count {
            let at = offset + 12 + 16 * i;
            let tag: [u8; 4] = r.slice(at, 4)?.try_into().expect("four bytes");
            let start = r.u32(at + 8)? as usize;
            let len = r.u32(at + 12)? as usize;

            if start.checked_add(len).is_some_and(|end| end <= data.len()) {
                tables.insert(tag, (start, len));
            }
        }

        let tab = |tag: &[u8; 4]| tables.get(tag).map(|&(start, len)| &data[start..start + len]);
        let mut f = Face {
            data: Vec::new(),
            tables: BTreeMap::new(),
            units_per_em: 0,
            head_flags: 0,
            index_to_loc_format: 0,
            num_glyphs: 0,
            max_twilight_points: 0,
            max_storage: 0,
            max_function_defs: 0,
            max_instruction_defs: 0,
            max_stack_elements: 0,
            max_size_of_instructions: 0,
            max_points: 0,
            max_contours: 0,
            max_component_points: 0,
            max_component_contours: 0,
            max_component_depth: 0,
            ascender: 0,
            descender: 0,
            line_gap: 0,
            advance_width_max: 0,
            number_of_h_metrics: 0,
            os2: None,
            cmap: None,
            cff,
        };

        let head = tab(b"head").ok_or(Error::MissingTable("head"))?;
        let h = Reader::new(head);
        f.head_flags = h.u16(16)?;
        f.units_per_em = h.u16(18)?;
        f.index_to_loc_format = h.i16(50)?;

        let maxp = tab(b"maxp").ok_or(Error::MissingTable("maxp"))?;
        let m = Reader::new(maxp);
        f.num_glyphs = m.u16(4)?;

        if m.u32(0)? >= 0x0001_0000 && maxp.len() >= 32 {
            f.max_points = m.u16(6)?;
            f.max_contours = m.u16(8)?;
            f.max_component_points = m.u16(10)?;
            f.max_component_contours = m.u16(12)?;
            f.max_twilight_points = m.u16(16)?;
            f.max_storage = m.u16(18)?;
            f.max_function_defs = m.u16(20)?;
            f.max_instruction_defs = m.u16(22)?;
            f.max_stack_elements = m.u16(24)?;
            f.max_size_of_instructions = m.u16(26)?;
            f.max_component_depth = m.u16(30)?;
        }

        let hhea = tab(b"hhea").ok_or(Error::MissingTable("hhea"))?;
        let hh = Reader::new(hhea);
        f.ascender = hh.i16(4)?;
        f.descender = hh.i16(6)?;
        f.line_gap = hh.i16(8)?;
        f.advance_width_max = hh.u16(10)?;
        f.number_of_h_metrics = hh.u16(34)?;

        if let Some(os2) = tab(b"OS/2") {
            let o = Reader::new(os2);

            if os2.len() >= 78 {
                f.os2 = Some(Os2 {
                    version: o.u16(0)?,
                    weight_class: o.u16(4)?,
                    width_class: o.u16(6)?,
                    fs_selection: o.u16(62)?,
                    typo_ascender: o.i16(68)?,
                    typo_descender: o.i16(70)?,
                    typo_line_gap: o.i16(72)?,
                    win_ascent: o.u16(74)?,
                    win_descent: o.u16(76)?,
                });
            }
        }

        f.data = data;
        f.tables = tables;
        let mut face = f;
        face.cmap = face.find_unicode_cmap();
        Ok(face)
    }

    /// A table's bytes.
    pub fn table(&self, tag: &[u8; 4]) -> Option<&[u8]> {
        self.tables.get(tag).map(|&(start, len)| &self.data[start..start + len])
    }

    /// The Unicode subtable `FT_Select_Charmap( FT_ENCODING_UNICODE )`
    /// takes: a UCS-4 one (format 12, platform 3 encoding 10, or platform 0)
    /// before a BMP one.
    fn find_unicode_cmap(&self) -> Option<(usize, u16)> {
        let cmap = self.table(b"cmap")?;
        let r = Reader::new(cmap);
        let count = r.u16(2).ok()? as usize;
        let mut bmp = None;
        let mut full = None;

        for i in 0..count {
            let at = 4 + 8 * i;
            let (platform, encoding) = (r.u16(at).ok()?, r.u16(at + 2).ok()?);
            let offset = r.u32(at + 4).ok()? as usize;
            let format = r.u16(offset).ok()?;
            let unicode = platform == 0 || (platform == 3 && (encoding == 1 || encoding == 10));

            if !unicode || !matches!(format, 4 | 12 | 13) {
                continue;
            }

            if matches!(format, 12 | 13) && full.is_none() {
                full = Some((offset, format));
            } else if format == 4 && bmp.is_none() {
                bmp = Some((offset, format));
            }
        }

        full.or(bmp)
    }

    /// Whether the face's character map is a symbol one (platform 3,
    /// encoding 0, and no Unicode one FreeType would take instead).
    pub fn has_symbol_cmap(&self) -> bool {
        if self.cmap.is_some() {
            return false;
        }

        let Some(cmap) = self.table(b"cmap") else { return false };
        let r = Reader::new(cmap);
        let count = r.u16(2).unwrap_or(0) as usize;
        (0..count).any(|i| r.u16(4 + 8 * i).ok() == Some(3) && r.u16(6 + 8 * i).ok() == Some(0))
    }

    /// `FT_Get_Char_Index`: the glyph of a character, 0 for none.
    pub fn glyph_index(&self, c: u32) -> u16 {
        let Some((offset, format)) = self.cmap else { return 0 };
        let Some(cmap) = self.table(b"cmap") else { return 0 };
        let r = Reader::new(cmap);

        let found = match format {
            4 => cmap4(r, offset, c),
            13 => cmap13(r, offset, c),
            _ => cmap12(r, offset, c),
        };

        found.ok().flatten().filter(|g| *g < self.num_glyphs).unwrap_or(0)
    }

    /// `hmtx`: the advance and left side bearing of a glyph.
    pub fn h_metrics(&self, gid: u16) -> (u16, i16) {
        let Some(hmtx) = self.table(b"hmtx") else { return (0, 0) };
        let r = Reader::new(hmtx);
        let n = self.number_of_h_metrics.max(1);

        if gid < n {
            let at = 4 * gid as usize;
            (r.u16(at).unwrap_or(0), r.i16(at + 2).unwrap_or(0))
        } else {
            let advance = r.u16(4 * (n as usize - 1)).unwrap_or(0);
            let at = 4 * n as usize + 2 * (gid - n) as usize;
            (advance, r.i16(at).unwrap_or(0))
        }
    }

    /// The `glyf` data of a glyph through `loca`; empty for none.
    pub fn glyph_data(&self, gid: u16) -> &[u8] {
        let (Some(loca), Some(glyf)) = (self.table(b"loca"), self.table(b"glyf")) else { return &[] };
        let r = Reader::new(loca);
        let g = gid as usize;

        let (start, end) = if self.index_to_loc_format == 0 {
            (r.u16(2 * g).map(|v| 2 * v as usize), r.u16(2 * g + 2).map(|v| 2 * v as usize))
        } else {
            (r.u32(4 * g).map(|v| v as usize), r.u32(4 * g + 4).map(|v| v as usize))
        };

        match (start, end) {
            (Ok(s), Ok(e)) if s < e && e <= glyf.len() => &glyf[s..e],
            // a broken end: as far as the table goes
            (Ok(s), _) if s < glyf.len() && gid + 1 == self.num_glyphs => &glyf[s..],
            _ => &[],
        }
    }
}

/// A format 4 subtable.
fn cmap4(r: Reader, offset: usize, c: u32) -> Result<Option<u16>, Error> {
    if c > 0xFFFF {
        return Ok(None);
    }

    let seg_x2 = r.u16(offset + 6)? as usize;
    let ends = offset + 14;
    let starts = ends + seg_x2 + 2;
    let deltas = starts + seg_x2;
    let ranges = deltas + seg_x2;

    for i in (0..seg_x2).step_by(2) {
        let end = r.u16(ends + i)? as u32;

        if c > end {
            continue;
        }

        let start = r.u16(starts + i)? as u32;

        if c < start {
            return Ok(None);
        }

        let delta = r.u16(deltas + i)?;
        let range = r.u16(ranges + i)? as usize;

        if range == 0 {
            return Ok(Some((c as u16).wrapping_add(delta)));
        }

        let at = ranges + i + range + 2 * (c - start) as usize;
        let g = r.u16(at)?;
        return Ok(if g == 0 { None } else { Some(g.wrapping_add(delta)) });
    }

    Ok(None)
}

/// A format 13 subtable: ranges of characters onto one glyph each.
fn cmap13(r: Reader, offset: usize, c: u32) -> Result<Option<u16>, Error> {
    let groups = r.u32(offset + 12)? as usize;

    for i in 0..groups {
        let at = offset + 16 + 12 * i;

        if (r.u32(at)?..=r.u32(at + 4)?).contains(&c) {
            return Ok(Some(r.u32(at + 8)? as u16));
        }
    }

    Ok(None)
}

/// A format 12 subtable.
fn cmap12(r: Reader, offset: usize, c: u32) -> Result<Option<u16>, Error> {
    let groups = r.u32(offset + 12)? as usize;

    for i in 0..groups {
        let at = offset + 16 + 12 * i;
        let (start, end) = (r.u32(at)?, r.u32(at + 4)?);

        if (start..=end).contains(&c) {
            return Ok(Some((r.u32(at + 8)? + (c - start)) as u16));
        }
    }

    Ok(None)
}
