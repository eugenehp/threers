//! File-format importers and heightmaps for the OpenSCAD front end.
//!
//! Covers the mesh formats OpenSCAD's `import()` accepts beyond STL/OBJ (which
//! reuse the crate's shared loaders) — **OFF / DXF / SVG / 3MF / AMF** — plus the
//! `.dat` and `.png` heightfields for `surface()`. Kept as a child module of
//! `scad` so the parsers can use the parent's private `Shape`/`Solid` types and
//! 2D-arrangement helpers directly, while exposing only the format entry points
//! (`pub(super)`) back to the evaluator.

use super::{arrange_extract, chain_segments, pip_evenodd, polyhedron, ring_edges_into, Shape, Solid};

/// Parse a Geomview **OFF** mesh (`import("…off")`).
pub(super) fn parse_off(src: &str) -> Result<Solid, String> {
    let mut toks = src.split_whitespace();
    match toks.next() {
        Some("OFF") => {}
        _ => return Err("import: not an OFF file (missing 'OFF' header)".into()),
    }
    let next_usize = |t: &mut std::str::SplitWhitespace| -> Result<usize, String> {
        t.next().and_then(|s| s.parse().ok()).ok_or_else(|| "OFF: malformed counts".into())
    };
    let next_f32 = |t: &mut std::str::SplitWhitespace| -> Result<f32, String> {
        t.next().and_then(|s| s.parse().ok()).ok_or_else(|| "OFF: malformed vertex".into())
    };
    let nv = next_usize(&mut toks)?;
    let nf = next_usize(&mut toks)?;
    let _ne = toks.next(); // edge count (unused)
    let mut verts = Vec::with_capacity(nv);
    for _ in 0..nv {
        verts.push([next_f32(&mut toks)?, next_f32(&mut toks)?, next_f32(&mut toks)?]);
    }
    let mut faces: Vec<Vec<u32>> = Vec::with_capacity(nf);
    for _ in 0..nf {
        let k = next_usize(&mut toks)?;
        let mut f = Vec::with_capacity(k);
        for _ in 0..k {
            f.push(next_usize(&mut toks)? as u32);
        }
        faces.push(f);
    }
    Ok(polyhedron(&verts, &faces))
}

// --- 3MF / AMF import -------------------------------------------------------

/// Read the first ZIP entry whose name ends with `suffix` and return its
/// decompressed bytes. Parses the central directory (reliable sizes even when a
/// local header defers them to a data descriptor), supporting the two methods a
/// 3MF ever uses: store (0) and raw DEFLATE (8). Any malformed field → `None`.
fn zip_read_entry(z: &[u8], suffix: &str) -> Option<Vec<u8>> {
    let u16 = |o: usize| -> Option<usize> { Some(u16::from_le_bytes(z.get(o..o + 2)?.try_into().ok()?) as usize) };
    let u32 = |o: usize| -> Option<usize> { Some(u32::from_le_bytes(z.get(o..o + 4)?.try_into().ok()?) as usize) };
    // Locate the End Of Central Directory record (scan back for its signature).
    let eocd = (0..z.len().saturating_sub(21)).rev().find(|&i| z[i..].starts_with(b"PK\x05\x06"))?;
    let count = u16(eocd + 10)?;
    let mut p = u32(eocd + 16)?; // start of central directory
    for _ in 0..count {
        if !z.get(p..p + 4)?.starts_with(b"PK\x01\x02") {
            break;
        }
        let method = u16(p + 10)?;
        let comp_size = u32(p + 20)?;
        let name_len = u16(p + 28)?;
        let extra_len = u16(p + 30)?;
        let comment_len = u16(p + 32)?;
        let local_off = u32(p + 42)?;
        let name = std::str::from_utf8(z.get(p + 46..p + 46 + name_len)?).ok()?;
        if name.ends_with(suffix) {
            // Jump to the local header to find where the data actually starts.
            if !z.get(local_off..local_off + 4)?.starts_with(b"PK\x03\x04") {
                return None;
            }
            let l_name = u16(local_off + 26)?;
            let l_extra = u16(local_off + 28)?;
            let start = local_off + 30 + l_name + l_extra;
            let data = z.get(start..start + comp_size)?;
            return match method {
                0 => Some(data.to_vec()),
                8 => crate::loaders::deflate::inflate_raw(data).ok(),
                _ => None,
            };
        }
        p += 46 + name_len + extra_len + comment_len;
    }
    None
}

/// All `<elem …>` open-tag bodies (the text between `<elem` and the next `>`),
/// matching the element name exactly so `<vertex …>` never catches `<vertices>`.
fn xml_tags<'a>(xml: &'a str, elem: &str) -> Vec<&'a str> {
    let open = format!("<{elem}");
    let mut out = Vec::new();
    let mut rest = xml;
    while let Some(i) = rest.find(&open) {
        let after = &rest[i + open.len()..];
        let delim = after.chars().next().is_some_and(|c| c.is_whitespace() || c == '>' || c == '/');
        match after.find('>') {
            Some(end) => {
                if delim {
                    out.push(&after[..end]);
                }
                rest = &after[end + 1..];
            }
            None => break,
        }
    }
    out
}

/// Value of attribute `name` in an open-tag body, e.g. `x` in `x="1.5" y="0"`.
fn xml_attr(tag: &str, name: &str) -> Option<f64> {
    let key = format!("{name}=");
    let mut s = tag;
    loop {
        let i = s.find(&key)?;
        let whole = i == 0 || s.as_bytes()[i - 1].is_ascii_whitespace();
        let after = s[i + key.len()..].trim_start();
        if whole {
            let b = after.as_bytes();
            if let Some(&q) = b.first().filter(|&&c| c == b'"' || c == b'\'') {
                let v = &after[1..];
                if let Some(end) = v.find(q as char) {
                    return v[..end].trim().parse().ok();
                }
            }
        }
        s = &s[i + key.len()..];
    }
}

/// Text of the first `<tag>…</tag>` child element, parsed as a number.
fn xml_text(block: &str, tag: &str) -> Option<f64> {
    let open = format!("<{tag}>");
    let start = block.find(&open)? + open.len();
    let end = block[start..].find(&format!("</{tag}>"))? + start;
    block[start..end].trim().parse().ok()
}

/// Inner content of each `<elem …>…</elem>` (element-name boundary respected, so
/// `<mesh>` won't catch `<meshx>`). Used to walk objects/meshes so per-mesh
/// triangle indices can be offset correctly when several are merged.
fn xml_blocks<'a>(xml: &'a str, elem: &str) -> Vec<&'a str> {
    let (open, close) = (format!("<{elem}"), format!("</{elem}>"));
    let mut out = Vec::new();
    let mut rest = xml;
    while let Some(i) = rest.find(&open) {
        let after = &rest[i + open.len()..];
        let delim = after.chars().next().is_some_and(|c| c.is_whitespace() || c == '>' || c == '/');
        let Some(gt) = after.find('>') else { break };
        let content = &after[gt + 1..];
        match (delim, content.find(&close)) {
            (true, Some(e)) => {
                out.push(&content[..e]);
                rest = &content[e + close.len()..];
            }
            _ => rest = content,
        }
    }
    out
}

/// Parse a **3MF** model (its `3dmodel.model` XML): `<vertex x= y= z=/>` and
/// `<triangle v1= v2= v3=/>`, per `<mesh>` so multi-object files index correctly.
/// Winding is auto-oriented outward by [`polyhedron`].
pub(super) fn parse_3mf(bytes: &[u8]) -> Option<Solid> {
    let model = zip_read_entry(bytes, ".model")?;
    let xml = std::str::from_utf8(&model).ok()?;
    let (mut verts, mut faces) = (Vec::<[f32; 3]>::new(), Vec::<Vec<u32>>::new());
    for mesh in xml_blocks(xml, "mesh") {
        let base = verts.len() as u32;
        for t in xml_tags(mesh, "vertex") {
            verts.push([xml_attr(t, "x")? as f32, xml_attr(t, "y")? as f32, xml_attr(t, "z")? as f32]);
        }
        let n = verts.len() as u32 - base;
        for t in xml_tags(mesh, "triangle") {
            let (a, b, c) = (xml_attr(t, "v1")? as u32, xml_attr(t, "v2")? as u32, xml_attr(t, "v3")? as u32);
            if a < n && b < n && c < n {
                faces.push(vec![base + a, base + b, base + c]);
            }
        }
    }
    (verts.len() >= 3 && !faces.is_empty()).then(|| polyhedron(&verts, &faces))
}

/// Parse an **AMF** (uncompressed XML): `<vertex><coordinates><x/><y/><z/>` and
/// `<triangle><v1/><v2/><v3/></triangle>`, per `<mesh>` (a mesh's `<volume>`s share
/// its vertices) so multi-object files index correctly. Zip-wrapped AMF is out of
/// scope.
pub(super) fn parse_amf(src: &str) -> Option<Solid> {
    let (mut verts, mut faces) = (Vec::<[f32; 3]>::new(), Vec::<Vec<u32>>::new());
    for mesh in xml_blocks(src, "mesh") {
        let base = verts.len() as u32;
        for v in xml_blocks(mesh, "vertex") {
            // Height/position lives in <coordinates>; a sibling <normal> also has
            // <x>/<y>/<z>, so read from the coordinates sub-block specifically.
            let coords = xml_blocks(v, "coordinates").into_iter().next().unwrap_or(v);
            verts.push([xml_text(coords, "x")? as f32, xml_text(coords, "y")? as f32, xml_text(coords, "z")? as f32]);
        }
        let n = verts.len() as u32 - base;
        for t in xml_blocks(mesh, "triangle") {
            let (a, b, c) = (xml_text(t, "v1")? as u32, xml_text(t, "v2")? as u32, xml_text(t, "v3")? as u32);
            if a < n && b < n && c < n {
                faces.push(vec![base + a, base + b, base + c]);
            }
        }
    }
    (verts.len() >= 3 && !faces.is_empty()).then(|| polyhedron(&verts, &faces))
}

/// Paeth predictor (PNG filter type 4).
fn png_paeth(a: i32, b: i32, c: i32) -> i32 {
    let p = a + b - c;
    let (pa, pb, pc) = ((p - a).abs(), (p - b).abs(), (p - c).abs());
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Decode a PNG to a row-major grid of per-pixel **luminance** (0–255), for use as
/// a `surface()` heightmap. Handles 8- and 16-bit greyscale/truecolour (±alpha),
/// non-interlaced — the forms slicers and image tools emit. Returns `None` for
/// anything else (paletted, interlaced, sub-byte depth), so `surface()` can report
/// it cleanly. Reuses the in-tree zlib inflate.
pub(super) fn decode_png_luma(bytes: &[u8]) -> Option<Vec<Vec<f32>>> {
    if !bytes.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
        return None;
    }
    let (mut w, mut h, mut bd, mut ct) = (0usize, 0usize, 0u8, 0u8);
    let mut idat: Vec<u8> = Vec::new();
    let mut p = 8;
    while p + 12 <= bytes.len() {
        let len = u32::from_be_bytes(bytes.get(p..p + 4)?.try_into().ok()?) as usize;
        let typ = &bytes[p + 4..p + 8];
        let data = bytes.get(p + 8..p + 8 + len)?;
        match typ {
            b"IHDR" => {
                w = u32::from_be_bytes(data.get(0..4)?.try_into().ok()?) as usize;
                h = u32::from_be_bytes(data.get(4..8)?.try_into().ok()?) as usize;
                bd = *data.get(8)?;
                ct = *data.get(9)?;
                if *data.get(12)? != 0 {
                    return None; // interlaced (Adam7) unsupported
                }
            }
            b"IDAT" => idat.extend_from_slice(data),
            b"IEND" => break,
            _ => {}
        }
        p += 12 + len; // length(4) + type(4) + data + crc(4)
    }
    let channels = match ct {
        0 => 1, // greyscale
        2 => 3, // truecolour
        4 => 2, // greyscale + alpha
        6 => 4, // truecolour + alpha
        _ => return None,
    };
    let sample_bytes = match bd {
        8 => 1,
        16 => 2,
        _ => return None,
    };
    if w == 0 || h == 0 {
        return None;
    }
    let bpp = channels * sample_bytes;
    let stride = w * bpp;
    let raw = crate::loaders::deflate::inflate_zlib(&idat).ok()?;
    if raw.len() < (stride + 1) * h {
        return None;
    }
    // Reverse the per-scanline filter into `img` (h × stride, no filter bytes).
    let mut img = vec![0u8; stride * h];
    for y in 0..h {
        let filt = raw[y * (stride + 1)];
        let src = &raw[y * (stride + 1) + 1..y * (stride + 1) + 1 + stride];
        for x in 0..stride {
            let a = if x >= bpp { img[y * stride + x - bpp] as i32 } else { 0 };
            let b = if y > 0 { img[(y - 1) * stride + x] as i32 } else { 0 };
            let c = if x >= bpp && y > 0 { img[(y - 1) * stride + x - bpp] as i32 } else { 0 };
            let recon = match filt {
                0 => src[x] as i32,
                1 => src[x] as i32 + a,
                2 => src[x] as i32 + b,
                3 => src[x] as i32 + (a + b) / 2,
                4 => src[x] as i32 + png_paeth(a, b, c),
                _ => return None,
            };
            img[y * stride + x] = (recon & 0xff) as u8;
        }
    }
    // Per-pixel luminance (16-bit samples → high byte).
    let sample = |px: &[u8], ch: usize| -> f32 {
        if sample_bytes == 1 { px[ch] as f32 } else { px[ch * 2] as f32 }
    };
    let grid = (0..h)
        .map(|y| {
            (0..w)
                .map(|x| {
                    let px = &img[y * stride + x * bpp..];
                    match channels {
                        1 | 2 => sample(px, 0), // grey (drop alpha)
                        _ => 0.299 * sample(px, 0) + 0.587 * sample(px, 1) + 0.114 * sample(px, 2),
                    }
                })
                .collect()
        })
        .collect();
    Some(grid)
}

/// Fill a set of 2D rings into a `Shape` by the even-odd rule (proper hole
/// nesting), used by the 2D importers (`import("…dxf"/".svg")`).
fn rings_to_shape(rings: &[Vec<[f32; 2]>]) -> Shape {
    let mut edges = Vec::new();
    for r in rings {
        if r.len() >= 3 {
            ring_edges_into(&mut edges, r);
        }
    }
    if edges.is_empty() {
        return Shape::default();
    }
    let ec = edges.clone();
    let inside = move |p: [f64; 2]| pip_evenodd(p, &ec);
    arrange_extract(&edges, &inside)
}

/// Parse a 2D **DXF** into a `Shape` (`import("…dxf")`): reads `LWPOLYLINE`,
/// `POLYLINE`/`VERTEX`, `LINE` (chained into loops), and `CIRCLE` entities.
pub(super) fn parse_dxf(src: &str) -> Shape {
    // DXF is code/value pairs on alternating lines.
    let lines: Vec<&str> = src.lines().map(|l| l.trim()).collect();
    let mut pairs: Vec<(i32, String)> = Vec::new();
    let mut i = 0;
    while i + 1 < lines.len() {
        if let Ok(code) = lines[i].parse::<i32>() {
            pairs.push((code, lines[i + 1].to_string()));
            i += 2;
        } else {
            i += 1;
        }
    }
    let mut rings: Vec<Vec<[f32; 2]>> = Vec::new();
    let mut segs: Vec<([f32; 2], [f32; 2])> = Vec::new();
    let f = |s: &str| s.parse::<f32>().ok();
    let mut j = 0;
    while j < pairs.len() {
        if pairs[j].0 != 0 {
            j += 1;
            continue;
        }
        let ent = pairs[j].1.clone();
        // collect this entity's pairs up to the next code-0 marker
        let mut k = j + 1;
        while k < pairs.len() && pairs[k].0 != 0 {
            k += 1;
        }
        let body = &pairs[j + 1..k];
        match ent.as_str() {
            "LWPOLYLINE" | "POLYLINE" => {
                let mut verts = Vec::new();
                let mut cx = None;
                for (c, v) in body {
                    match c {
                        10 => cx = f(v),
                        20 => {
                            if let (Some(x), Some(y)) = (cx, f(v)) {
                                verts.push([x, y]);
                            }
                        }
                        _ => {}
                    }
                }
                if verts.len() >= 3 {
                    rings.push(verts);
                }
            }
            "LINE" => {
                let g = |code: i32| body.iter().find(|(c, _)| *c == code).and_then(|(_, v)| f(v));
                if let (Some(x1), Some(y1), Some(x2), Some(y2)) = (g(10), g(20), g(11), g(21)) {
                    segs.push(([x1, y1], [x2, y2]));
                }
            }
            "CIRCLE" => {
                let g = |code: i32| body.iter().find(|(c, _)| *c == code).and_then(|(_, v)| f(v));
                if let (Some(cx), Some(cy), Some(r)) = (g(10), g(20), g(40)) {
                    let n = 32;
                    rings.push(
                        (0..n)
                            .map(|i| {
                                let a = i as f32 / n as f32 * std::f32::consts::TAU;
                                [cx + r * a.cos(), cy + r * a.sin()]
                            })
                            .collect(),
                    );
                }
            }
            _ => {}
        }
        j = k;
    }
    for poly in chain_segments(&segs).polys {
        rings.push(poly.outer);
    }
    rings_to_shape(&rings)
}

/// Parse a minimal **SVG** into a `Shape` (`import("…svg")`): `rect`, `circle`,
/// `polygon`/`polyline`, and `path` (`M`/`L`/`H`/`V`/`Z`, with cubic/quadratic
/// segments flattened). The SVG y-axis (down) is flipped to OpenSCAD's y-up.
pub(super) fn parse_svg(src: &str) -> Shape {
    let mut rings: Vec<Vec<[f32; 2]>> = Vec::new();
    let num = |s: &str| s.parse::<f32>().ok();
    // crude attribute reader: `name="value"` within a tag substring
    let attr = |tag: &str, name: &str| -> Option<String> {
        let key = format!("{name}=\"");
        let p = tag.find(&key)? + key.len();
        let rest = &tag[p..];
        let end = rest.find('"')?;
        Some(rest[..end].to_string())
    };
    let coords = |s: &str| -> Vec<f32> {
        s.split(|c: char| c == ',' || c.is_whitespace()).filter_map(num).collect()
    };
    for tag in src.split('<') {
        let name = tag.split(|c: char| c.is_whitespace() || c == '>').next().unwrap_or("");
        match name {
            "rect" => {
                if let (Some(x), Some(y), Some(w), Some(h)) = (
                    attr(tag, "x").as_deref().and_then(num),
                    attr(tag, "y").as_deref().and_then(num),
                    attr(tag, "width").as_deref().and_then(num),
                    attr(tag, "height").as_deref().and_then(num),
                ) {
                    rings.push(vec![[x, -y], [x + w, -y], [x + w, -(y + h)], [x, -(y + h)]]);
                }
            }
            "circle" => {
                if let (Some(cx), Some(cy), Some(r)) = (
                    attr(tag, "cx").as_deref().and_then(num),
                    attr(tag, "cy").as_deref().and_then(num),
                    attr(tag, "r").as_deref().and_then(num),
                ) {
                    let n = 48;
                    rings.push(
                        (0..n)
                            .map(|i| {
                                let a = i as f32 / n as f32 * std::f32::consts::TAU;
                                [cx + r * a.cos(), -(cy + r * a.sin())]
                            })
                            .collect(),
                    );
                }
            }
            "polygon" | "polyline" => {
                if let Some(pts) = attr(tag, "points") {
                    let c = coords(&pts);
                    let ring: Vec<[f32; 2]> = c.chunks_exact(2).map(|p| [p[0], -p[1]]).collect();
                    if ring.len() >= 3 {
                        rings.push(ring);
                    }
                }
            }
            "path" => {
                if let Some(d) = attr(tag, "d") {
                    rings.extend(svg_path_rings(&d));
                }
            }
            _ => {}
        }
    }
    rings_to_shape(&rings)
}

/// Flatten SVG path data into rings (`M`/`L`/`H`/`V`/`Z`; cubic `C`/quadratic `Q`
/// flattened to line segments). Y is negated to OpenSCAD's y-up.
fn svg_path_rings(d: &str) -> Vec<Vec<[f32; 2]>> {
    // tokenize into commands and numbers
    let mut toks: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in d.chars() {
        if ch.is_alphabetic() {
            if !cur.is_empty() {
                toks.push(std::mem::take(&mut cur));
            }
            toks.push(ch.to_string());
        } else if ch == ',' || ch.is_whitespace() {
            if !cur.is_empty() {
                toks.push(std::mem::take(&mut cur));
            }
        } else if (ch == '-' || ch == '+') && !cur.is_empty() && !cur.ends_with(['e', 'E']) {
            toks.push(std::mem::take(&mut cur));
            cur.push(ch);
        } else {
            cur.push(ch);
        }
    }
    if !cur.is_empty() {
        toks.push(cur);
    }
    let mut rings: Vec<Vec<[f32; 2]>> = Vec::new();
    let mut ring: Vec<[f32; 2]> = Vec::new();
    let (mut px, mut py) = (0.0f32, 0.0f32);
    let mut ti = 0;
    let num = |t: &[String], i: &mut usize| -> f32 {
        let v = t.get(*i).and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0);
        *i += 1;
        v
    };
    while ti < toks.len() {
        let cmd = toks[ti].clone();
        if cmd.len() != 1 || !cmd.chars().next().unwrap().is_alphabetic() {
            ti += 1;
            continue;
        }
        ti += 1;
        let rel = cmd.chars().next().unwrap().is_lowercase();
        match cmd.to_ascii_uppercase().chars().next().unwrap() {
            'M' => {
                if !ring.is_empty() {
                    rings.push(std::mem::take(&mut ring));
                }
                let (x, y) = (num(&toks, &mut ti), num(&toks, &mut ti));
                px = if rel { px + x } else { x };
                py = if rel { py + y } else { y };
                ring.push([px, -py]);
            }
            'L' => {
                let (x, y) = (num(&toks, &mut ti), num(&toks, &mut ti));
                px = if rel { px + x } else { x };
                py = if rel { py + y } else { y };
                ring.push([px, -py]);
            }
            'H' => {
                let x = num(&toks, &mut ti);
                px = if rel { px + x } else { x };
                ring.push([px, -py]);
            }
            'V' => {
                let y = num(&toks, &mut ti);
                py = if rel { py + y } else { y };
                ring.push([px, -py]);
            }
            'C' => {
                let c = [num(&toks, &mut ti), num(&toks, &mut ti), num(&toks, &mut ti),
                         num(&toks, &mut ti), num(&toks, &mut ti), num(&toks, &mut ti)];
                let (x0, y0) = (px, py);
                let (c1x, c1y) = (if rel { x0 + c[0] } else { c[0] }, if rel { y0 + c[1] } else { c[1] });
                let (c2x, c2y) = (if rel { x0 + c[2] } else { c[2] }, if rel { y0 + c[3] } else { c[3] });
                let (ex, ey) = (if rel { x0 + c[4] } else { c[4] }, if rel { y0 + c[5] } else { c[5] });
                for s in 1..=12 {
                    let t = s as f32 / 12.0;
                    let mt = 1.0 - t;
                    let bx = mt * mt * mt * x0 + 3.0 * mt * mt * t * c1x + 3.0 * mt * t * t * c2x + t * t * t * ex;
                    let by = mt * mt * mt * y0 + 3.0 * mt * mt * t * c1y + 3.0 * mt * t * t * c2y + t * t * t * ey;
                    ring.push([bx, -by]);
                }
                px = ex;
                py = ey;
            }
            'Q' => {
                let c = [num(&toks, &mut ti), num(&toks, &mut ti), num(&toks, &mut ti), num(&toks, &mut ti)];
                let (x0, y0) = (px, py);
                let (cx, cy) = (if rel { x0 + c[0] } else { c[0] }, if rel { y0 + c[1] } else { c[1] });
                let (ex, ey) = (if rel { x0 + c[2] } else { c[2] }, if rel { y0 + c[3] } else { c[3] });
                for s in 1..=10 {
                    let t = s as f32 / 10.0;
                    let mt = 1.0 - t;
                    let bx = mt * mt * x0 + 2.0 * mt * t * cx + t * t * ex;
                    let by = mt * mt * y0 + 2.0 * mt * t * cy + t * t * ey;
                    ring.push([bx, -by]);
                }
                px = ex;
                py = ey;
            }
            'Z' => {
                if !ring.is_empty() {
                    rings.push(std::mem::take(&mut ring));
                }
            }
            _ => {
                ti += 1; // unknown command — skip a token to avoid stalling
            }
        }
    }
    if ring.len() >= 3 {
        rings.push(ring);
    }
    rings
}

/// Build a solid from a whitespace-separated heightmap matrix (OpenSCAD
/// `surface(file="…dat")`): each cell `(row, col)` becomes height `z` at
/// `(x=col, y=rows−1−row)`; a flat base at `z=0` closes it into a manifold.
pub(super) fn surface_dat(src: &str, center: bool) -> Result<Solid, String> {
    let grid: Vec<Vec<f32>> = src
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.split_whitespace().filter_map(|s| s.parse::<f32>().ok()).collect())
        .filter(|r: &Vec<f32>| !r.is_empty())
        .collect();
    surface_grid(grid, center)
}

/// Build the extruded, watertight heightmap solid from a row-major `grid` of Z
/// values (row 0 = top / max-Y, OpenSCAD convention). Shared by the text (`.dat`)
/// and PNG heightmap importers.
pub(super) fn surface_grid(grid: Vec<Vec<f32>>, center: bool) -> Result<Solid, String> {
    let rows = grid.len();
    let cols = grid.iter().map(|r| r.len()).min().unwrap_or(0);
    if rows < 2 || cols < 2 {
        return Err("surface: need at least a 2×2 heightmap".into());
    }
    let h = |r: usize, c: usize| grid[r][c];
    // OpenSCAD puts the base one unit below the minimum data value.
    let base_z = grid.iter().flatten().copied().fold(f32::INFINITY, f32::min) - 1.0;
    // OpenSCAD flips rows so the first line is at the top (max y).
    let xy = |r: usize, c: usize| [c as f32, (rows - 1 - r) as f32];
    let mut verts: Vec<[f32; 3]> = Vec::with_capacity(rows * cols * 2);
    let top = |r: usize, c: usize| (r * cols + c) as u32;
    let base = (rows * cols) as u32;
    let bot = |r: usize, c: usize| base + (r * cols + c) as u32;
    for r in 0..rows {
        for c in 0..cols {
            let p = xy(r, c);
            verts.push([p[0], p[1], h(r, c)]);
        }
    }
    for r in 0..rows {
        for c in 0..cols {
            let p = xy(r, c);
            verts.push([p[0], p[1], base_z]);
        }
    }
    let mut faces: Vec<Vec<u32>> = Vec::new();
    for r in 0..rows - 1 {
        for c in 0..cols - 1 {
            // top surface, CCW-from-above (+Z); base is the mirror (−Z)
            faces.push(vec![top(r, c), top(r + 1, c), top(r + 1, c + 1)]);
            faces.push(vec![top(r, c), top(r + 1, c + 1), top(r, c + 1)]);
            faces.push(vec![bot(r, c), bot(r + 1, c + 1), bot(r + 1, c)]);
            faces.push(vec![bot(r, c), bot(r, c + 1), bot(r + 1, c + 1)]);
        }
    }
    // CCW-from-above perimeter loop of top vertices.
    let mut boundary: Vec<usize> = Vec::new();
    for c in 0..cols {
        boundary.push((rows - 1) * cols + c); // bottom edge, left→right
    }
    for r in (0..rows - 1).rev() {
        boundary.push(r * cols + (cols - 1)); // right edge, up
    }
    for c in (0..cols - 1).rev() {
        boundary.push(c); // top edge (r=0), right→left
    }
    for r in 1..rows - 1 {
        boundary.push(r * cols); // left edge, down
    }
    // Each boundary edge a→b is closed by a wall b→a→(a+base)→(b+base). Where a
    // vertex sits at z=0 its top and base copies coincide, so the wall collapses
    // to a triangle (or nothing) — skip the degenerate triangles so the mesh stays
    // watertight (the top surface then meets the base directly at that edge).
    let coincident = |i: u32, j: u32| {
        let (p, q) = (verts[i as usize], verts[j as usize]);
        (p[0] - q[0]).abs() < 1e-9 && (p[1] - q[1]).abs() < 1e-9 && (p[2] - q[2]).abs() < 1e-9
    };
    let push_tri = |x: u32, y: u32, z: u32, faces: &mut Vec<Vec<u32>>| {
        if !coincident(x, y) && !coincident(y, z) && !coincident(z, x) {
            faces.push(vec![x, y, z]);
        }
    };
    for k in 0..boundary.len() {
        let a = boundary[k] as u32;
        let b = boundary[(k + 1) % boundary.len()] as u32;
        push_tri(b, a, a + base, &mut faces);
        push_tri(b, a + base, b + base, &mut faces);
    }
    let s = polyhedron(&verts, &faces);
    Ok(if center {
        s.translate([-((cols - 1) as f32) / 2.0, -((rows - 1) as f32) / 2.0, 0.0])
    } else {
        s
    })
}
