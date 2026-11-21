//! HEIF still images (`.heic`, ISO/IEC 23008-12) to RGBA: the primary
//! image of the file, a single HEVC picture or a grid of HEVC tiles (as
//! iPhones and iPads write photos and screenshots), decoded with
//! [`HevcDecoder`](crate::decode::HevcDecoder), with its `colr` colour,
//! rotation (`irot`) and mirroring (`imir`) applied.
//!
//! Thumbnails, alpha planes, depth maps and other auxiliary images are left
//! out; the image is opaque.

use crate::decode::{ColorInfo, DecodedFrame, HevcDecoder};

/// A decoded image: `width * height` RGBA pixels, row by row.
#[derive(Debug, Clone)]
pub struct HeifImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Whether `bytes` start like a HEIF file (an `ftyp` box naming a HEIF brand).
pub fn is_heif(bytes: &[u8]) -> bool {
    if bytes.get(4..8) != Some(b"ftyp") {
        return false;
    }
    let size = be32(bytes, 0).unwrap_or(0) as usize;
    let end = size.min(bytes.len());
    // the major brand, then the compatible brands after the minor version
    let brands = std::iter::once(8).chain((16..end).step_by(4));
    brands.filter_map(|i| bytes.get(i..i + 4)).any(|b| {
        matches!(b, b"heic" | b"heix" | b"heim" | b"heis" | b"mif1" | b"msf1" | b"hevc" | b"hevx")
    })
}

/// Decode the primary image of a HEIF file.
pub fn decode(bytes: &[u8]) -> Result<HeifImage, String> {
    let file = File::parse(bytes)?;
    let primary = file.primary;
    let kind = file.item_type(primary).ok_or("the primary image isn't listed")?;
    let mut img = match &kind {
        b"hvc1" => {
            let f = file.decode_hvc1(primary)?;
            let color = file.color(primary).unwrap_or(f.color);
            Rgba::from_frame(&f, color)
        }
        b"grid" => file.decode_grid(primary)?,
        other => return Err(format!("the primary image is {}, not HEVC", String::from_utf8_lossy(other))),
    };
    for p in file.properties(primary) {
        match p {
            Property::Rot(turns) => img = img.rotate_ccw(*turns),
            Property::Mirror(axis) => img.mirror(*axis),
            _ => {}
        }
    }
    Ok(HeifImage { width: img.w, height: img.h, rgba: img.px })
}

// --- boxes -------------------------------------------------------------------

fn be16(b: &[u8], i: usize) -> Option<u16> {
    Some(u16::from_be_bytes(b.get(i..i + 2)?.try_into().ok()?))
}

fn be32(b: &[u8], i: usize) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(i..i + 4)?.try_into().ok()?))
}

/// A big-endian unsigned number of `n` bytes (0, 2, 4 or 8) at `*i`, advancing it.
fn take(b: &[u8], i: &mut usize, n: usize) -> Result<u64, String> {
    let s = b.get(*i..*i + n).ok_or("a box ends early")?;
    *i += n;
    Ok(s.iter().fold(0u64, |a, &x| (a << 8) | x as u64))
}

/// The boxes in `b`: (type, contents).
fn boxes(b: &[u8]) -> Result<Vec<([u8; 4], &[u8])>, String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 8 <= b.len() {
        let size = be32(b, i).unwrap() as u64;
        let kind: [u8; 4] = b[i + 4..i + 8].try_into().unwrap();
        let (head, size) = match size {
            1 => (16, u64::from_be_bytes(b.get(i + 8..i + 16).ok_or("a box ends early")?.try_into().unwrap())),
            0 => (8, (b.len() - i) as u64),
            n => (8, n),
        };
        let end = i.checked_add(size as usize).filter(|&e| e <= b.len() && size >= head as u64);
        let end = end.ok_or_else(|| format!("the {} box runs past its parent", String::from_utf8_lossy(&kind)))?;
        out.push((kind, &b[i + head..end]));
        i = end;
    }
    Ok(out)
}

fn find<'a>(list: &[([u8; 4], &'a [u8])], kind: &[u8; 4]) -> Option<&'a [u8]> {
    list.iter().find(|(k, _)| k == kind).map(|(_, b)| *b)
}

#[derive(Debug, Clone)]
enum Property {
    /// `hvcC`: the parameter sets and the NAL length size.
    Hvcc { nals: Vec<Vec<u8>>, length_size: usize },
    /// `colr` of type `nclx`.
    Color(ColorInfo),
    /// `irot`: quarter turns anticlockwise.
    Rot(u8),
    /// `imir`: about a vertical axis (0: left and right swap) or a
    /// horizontal one (1: top and bottom swap).
    Mirror(u8),
    Other,
}

struct Extent {
    offset: u64,
    length: u64,
}

struct Location {
    /// 0: offsets into the file; 1: into `idat`.
    method: u8,
    base: u64,
    extents: Vec<Extent>,
}

struct File<'a> {
    bytes: &'a [u8],
    idat: &'a [u8],
    primary: u32,
    items: Vec<(u32, [u8; 4])>,
    locations: Vec<(u32, Location)>,
    props: Vec<Property>,
    /// item → its properties, as indexes into `props`, in order.
    assoc: Vec<(u32, Vec<usize>)>,
    /// `dimg` references: a derived image → the images it's made of.
    dimg: Vec<(u32, Vec<u32>)>,
}

impl<'a> File<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Self, String> {
        if !is_heif(bytes) {
            return Err("not a HEIF file".into());
        }
        let top = boxes(bytes)?;
        let meta = find(&top, b"meta").ok_or("no meta box")?;
        let meta = boxes(meta.get(4..).ok_or("an empty meta box")?)?;

        let pitm = find(&meta, b"pitm").ok_or("no primary item")?;
        let primary = if pitm[0] == 0 { be16(pitm, 4).map(u32::from) } else { be32(pitm, 4) };
        let primary = primary.ok_or("a short pitm box")?;

        let mut items = Vec::new();
        if let Some(iinf) = find(&meta, b"iinf") {
            let skip = if iinf[0] == 0 { 6 } else { 8 };
            for (k, infe) in boxes(iinf.get(skip..).unwrap_or_default())? {
                if &k != b"infe" || infe[0] < 2 {
                    continue;
                }
                let (id, at) = if infe[0] == 2 { (be16(infe, 4).map(u32::from), 8) } else { (be32(infe, 4), 10) };
                if let (Some(id), Some(t)) = (id, infe.get(at..at + 4)) {
                    items.push((id, t.try_into().unwrap()));
                }
            }
        }

        let mut locations = Vec::new();
        if let Some(iloc) = find(&meta, b"iloc") {
            let v = iloc[0];
            let mut i = 4;
            let sizes = take(iloc, &mut i, 1)? as u8;
            let more = take(iloc, &mut i, 1)? as u8;
            let (off_n, len_n, base_n) = ((sizes >> 4) as usize, (sizes & 15) as usize, (more >> 4) as usize);
            let idx_n = if v >= 1 { (more & 15) as usize } else { 0 };
            let count = take(iloc, &mut i, if v < 2 { 2 } else { 4 })?;
            for _ in 0..count {
                let id = take(iloc, &mut i, if v < 2 { 2 } else { 4 })? as u32;
                let method = if v >= 1 { (take(iloc, &mut i, 2)? & 15) as u8 } else { 0 };
                let _data_ref = take(iloc, &mut i, 2)?;
                let base = take(iloc, &mut i, base_n)?;
                let n = take(iloc, &mut i, 2)?;
                let mut extents = Vec::new();
                for _ in 0..n {
                    take(iloc, &mut i, idx_n)?;
                    let offset = take(iloc, &mut i, off_n)?;
                    let length = take(iloc, &mut i, len_n)?;
                    extents.push(Extent { offset, length });
                }
                locations.push((id, Location { method, base, extents }));
            }
        }

        let mut props = Vec::new();
        let mut assoc = Vec::new();
        if let Some(iprp) = find(&meta, b"iprp") {
            let iprp = boxes(iprp)?;
            if let Some(ipco) = find(&iprp, b"ipco") {
                for (k, b) in boxes(ipco)? {
                    props.push(property(&k, b));
                }
            }
            for (k, ipma) in &iprp {
                if k != b"ipma" {
                    continue;
                }
                let (v, wide) = (ipma[0], ipma[3] & 1 == 1);
                let mut i = 4;
                for _ in 0..take(ipma, &mut i, 4)? {
                    let id = take(ipma, &mut i, if v < 1 { 2 } else { 4 })? as u32;
                    let mut list = Vec::new();
                    for _ in 0..take(ipma, &mut i, 1)? {
                        let a = take(ipma, &mut i, if wide { 2 } else { 1 })?;
                        let index = (a & if wide { 0x7fff } else { 0x7f }) as usize;
                        if index > 0 {
                            list.push(index - 1);
                        }
                    }
                    assoc.push((id, list));
                }
            }
        }

        let mut dimg = Vec::new();
        if let Some(iref) = find(&meta, b"iref") {
            let wide = iref[0] != 0;
            for (k, b) in boxes(iref.get(4..).unwrap_or_default())? {
                if &k != b"dimg" {
                    continue;
                }
                let n = if wide { 4 } else { 2 };
                let mut i = 0;
                let from = take(b, &mut i, n)? as u32;
                let count = take(b, &mut i, 2)?;
                let to = (0..count).map(|_| take(b, &mut i, n).map(|x| x as u32)).collect::<Result<_, _>>()?;
                dimg.push((from, to));
            }
        }

        Ok(File { bytes, idat: find(&meta, b"idat").unwrap_or_default(), primary, items, locations, props, assoc, dimg })
    }

    fn item_type(&self, id: u32) -> Option<[u8; 4]> {
        self.items.iter().find(|(i, _)| *i == id).map(|(_, t)| *t)
    }

    fn properties(&self, id: u32) -> impl Iterator<Item = &Property> {
        let list = self.assoc.iter().find(|(i, _)| *i == id).map(|(_, l)| l.as_slice()).unwrap_or_default();
        list.iter().filter_map(|&i| self.props.get(i))
    }

    fn color(&self, id: u32) -> Option<ColorInfo> {
        self.properties(id).find_map(|p| match p {
            Property::Color(c) => Some(*c),
            _ => None,
        })
    }

    /// An item's data, its extents joined.
    fn data(&self, id: u32) -> Result<Vec<u8>, String> {
        let loc = self.locations.iter().find(|(i, _)| *i == id).map(|(_, l)| l).ok_or("an item has no location")?;
        let src = match loc.method {
            0 => self.bytes,
            1 => self.idat,
            m => return Err(format!("item data by construction method {m} isn't supported")),
        };
        let mut out = Vec::new();
        for e in &loc.extents {
            let start = (loc.base + e.offset) as usize;
            let len = if e.length == 0 { src.len().saturating_sub(start) } else { e.length as usize };
            out.extend_from_slice(src.get(start..start + len).ok_or("item data runs past the file")?);
        }
        Ok(out)
    }

    fn decode_hvc1(&self, id: u32) -> Result<DecodedFrame, String> {
        let (nals, length_size) = self
            .properties(id)
            .find_map(|p| match p {
                Property::Hvcc { nals, length_size } => Some((nals, *length_size)),
                _ => None,
            })
            .ok_or("an HEVC image has no hvcC")?;
        let data = self.data(id)?;
        let mut d = HevcDecoder::new();
        for n in nals {
            d.decode_nal(n)?;
        }
        let mut i = 0;
        while i < data.len() {
            let n = take(&data, &mut i, length_size)? as usize;
            let nal = data.get(i..i + n).ok_or("a NAL unit runs past its image")?;
            d.decode_nal(nal)?;
            i += n;
        }
        d.flush_output();
        d.take_thread_error().map_or(Ok(()), Err)?;
        d.take_pictures().into_iter().next().map(|p| p.frame).ok_or_else(|| "an HEVC image decoded to nothing".into())
    }

    /// A grid: its tiles decoded, laid out row by row, cropped to its size.
    fn decode_grid(&self, id: u32) -> Result<Rgba, String> {
        let g = self.data(id)?;
        let mut i = 1;
        let flags = take(&g, &mut i, 1)?;
        let rows = take(&g, &mut i, 1)? as u32 + 1;
        let cols = take(&g, &mut i, 1)? as u32 + 1;
        let n = if flags & 1 == 1 { 4 } else { 2 };
        let w = take(&g, &mut i, n)? as u32;
        let h = take(&g, &mut i, n)? as u32;
        let tiles = self.dimg.iter().find(|(f, _)| *f == id).map(|(_, t)| t).ok_or("a grid without tiles")?;
        if tiles.len() != (rows * cols) as usize {
            return Err(format!("a {cols}×{rows} grid with {} tiles", tiles.len()));
        }
        let color = self.color(id);
        let mut out = Rgba { w, h, px: vec![255; (w * h * 4) as usize] };
        let (mut tw, mut th) = (0, 0);
        for (k, &t) in tiles.iter().enumerate() {
            let f = self.decode_hvc1(t)?;
            if k == 0 {
                (tw, th) = (f.width, f.height);
            }
            let tile = Rgba::from_frame(&f, color.or_else(|| self.color(t)).unwrap_or(f.color));
            let (x0, y0) = ((k as u32 % cols) * tw, (k as u32 / cols) * th);
            for y in 0..tile.h.min(h.saturating_sub(y0)) {
                let cw = tile.w.min(w.saturating_sub(x0)) as usize * 4;
                let src = (y * tile.w) as usize * 4;
                let dst = (((y0 + y) * w + x0) * 4) as usize;
                out.px[dst..dst + cw].copy_from_slice(&tile.px[src..src + cw]);
            }
        }
        Ok(out)
    }
}

fn property(kind: &[u8; 4], b: &[u8]) -> Property {
    match kind {
        b"hvcC" if b.len() >= 23 => {
            let length_size = (b[21] & 3) as usize + 1;
            let mut nals = Vec::new();
            let mut i = 23;
            for _ in 0..b[22] {
                let Some(count) = be16(b, i + 1) else { break };
                i += 3;
                for _ in 0..count {
                    let Some(n) = be16(b, i) else { break };
                    let Some(nal) = b.get(i + 2..i + 2 + n as usize) else { break };
                    nals.push(nal.to_vec());
                    i += 2 + n as usize;
                }
            }
            Property::Hvcc { nals, length_size }
        }
        b"colr" if b.get(0..4) == Some(b"nclx") && b.len() >= 11 => {
            let matrix = be16(b, 8).map(u32::from);
            Property::Color(ColorInfo::from_vui(matrix, b[10] & 0x80 != 0, 0))
        }
        b"irot" if !b.is_empty() => Property::Rot(b[0] & 3),
        b"imir" if !b.is_empty() => Property::Mirror(b[0] & 1),
        _ => Property::Other,
    }
}

// --- pixels ------------------------------------------------------------------

struct Rgba {
    w: u32,
    h: u32,
    px: Vec<u8>,
}

impl Rgba {
    fn from_frame(f: &DecodedFrame, color: ColorInfo) -> Self {
        let mut f = f.clone();
        f.color = color;
        Rgba { w: f.width, h: f.height, px: f.to_rgba() }
    }

    fn at(&self, x: u32, y: u32) -> usize {
        ((y * self.w + x) * 4) as usize
    }

    /// Turned anticlockwise `turns` quarter turns.
    fn rotate_ccw(self, turns: u8) -> Self {
        if turns == 0 {
            return self;
        }
        let (w, h) = if turns % 2 == 1 { (self.h, self.w) } else { (self.w, self.h) };
        let mut px = vec![0; self.px.len()];
        for y in 0..self.h {
            for x in 0..self.w {
                let (nx, ny) = match turns {
                    1 => (y, self.w - 1 - x),
                    2 => (self.w - 1 - x, self.h - 1 - y),
                    _ => (self.h - 1 - y, x),
                };
                let (s, d) = (self.at(x, y), ((ny * w + nx) * 4) as usize);
                px[d..d + 4].copy_from_slice(&self.px[s..s + 4]);
            }
        }
        Rgba { w, h, px }
    }

    fn mirror(&mut self, axis: u8) {
        let row = self.w as usize * 4;
        if axis == 1 {
            // top and bottom
            for y in 0..self.h as usize / 2 {
                let (a, b) = self.px.split_at_mut((self.h as usize - 1 - y) * row);
                a[y * row..(y + 1) * row].swap_with_slice(&mut b[..row]);
            }
        } else {
            for y in 0..self.h {
                for x in 0..self.w / 2 {
                    let (a, b) = (self.at(x, y), self.at(self.w - 1 - x, y));
                    for c in 0..4 {
                        self.px.swap(a + c, b + c);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: u32, h: u32) -> Rgba {
        Rgba { w, h, px: (0..w * h).flat_map(|i| [i as u8, 0, 0, 255]).collect() }
    }

    #[test]
    fn turns_and_mirrors() {
        // 0 1 2
        // 3 4 5
        let r = img(3, 2).rotate_ccw(1);
        assert_eq!((r.w, r.h), (2, 3));
        let red: Vec<u8> = r.px.chunks(4).map(|p| p[0]).collect();
        assert_eq!(red, [2, 5, 1, 4, 0, 3]);
        let r = img(3, 2).rotate_ccw(2);
        assert_eq!(r.px.chunks(4).map(|p| p[0]).collect::<Vec<_>>(), [5, 4, 3, 2, 1, 0]);
        let r = img(3, 2).rotate_ccw(3);
        assert_eq!(r.px.chunks(4).map(|p| p[0]).collect::<Vec<_>>(), [3, 0, 4, 1, 5, 2]);
        let mut m = img(3, 2);
        m.mirror(1);
        assert_eq!(m.px.chunks(4).map(|p| p[0]).collect::<Vec<_>>(), [3, 4, 5, 0, 1, 2]);
        let mut m = img(3, 2);
        m.mirror(0);
        assert_eq!(m.px.chunks(4).map(|p| p[0]).collect::<Vec<_>>(), [2, 1, 0, 5, 4, 3]);
    }

    /// A 96×64 gradient over a checkerboard, written by macOS (`sips -s
    /// format heic`): a single `hvc1` item with `colr` nclx.
    #[test]
    fn decodes_what_macos_writes() {
        let bytes = include_bytes!("../tests/support/heif/gradient.heic");
        assert!(is_heif(bytes));
        let img = decode(bytes).unwrap();
        assert_eq!((img.width, img.height), (96, 64));
        let (w, h) = (96usize, 64usize);
        let mut se = 0f64;
        for y in 0..h {
            for x in 0..w {
                let src = [x * 255 / (w - 1), y * 255 / (h - 1), if (x / 16 + y / 16) % 2 == 1 { 128 } else { 32 }];
                for c in 0..3 {
                    let e = img.rgba[(y * w + x) * 4 + c] as f64 - src[c] as f64;
                    se += e * e;
                }
            }
        }
        let psnr = 10.0 * (255.0f64 * 255.0 / (se / (w * h * 3) as f64)).log10();
        // what the encoder's own decode of this 774-byte file reaches is 30.8
        assert!(psnr > 29.0, "PSNR {psnr:.1} dB");
    }

    #[test]
    fn brands() {
        let mut f = vec![0, 0, 0, 24];
        f.extend_from_slice(b"ftypheic\0\0\0\0mif1heic");
        assert!(is_heif(&f));
        f[8..12].copy_from_slice(b"isom");
        f[16..24].copy_from_slice(b"isomavc1");
        assert!(!is_heif(&f));
        assert!(!is_heif(b"\x89PNG\r\n\x1a\n"));
    }
}

#[cfg(test)]
mod real_file {
    /// `HEIF_FILE=x.heic HEIF_OUT=x.rgba cargo test -p threers-codec --release heif::real_file -- --ignored`
    #[test]
    #[ignore]
    fn decode_a_real_file() {
        let path = std::env::var("HEIF_FILE").expect("HEIF_FILE");
        let t = std::time::Instant::now();
        let img = super::decode(&std::fs::read(path).unwrap()).unwrap();
        eprintln!("{}x{} in {:?}", img.width, img.height, t.elapsed());
        if let Ok(out) = std::env::var("HEIF_OUT") {
            std::fs::write(out, &img.rgba).unwrap();
        }
    }
}
