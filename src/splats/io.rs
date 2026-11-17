//! Reading and writing splats: the reference trainer's PLY and the `.splat`
//! format of the web viewers.

use super::{normalize_quat, sh_rest_coefficients, GaussianSplats, SH_C0};

/// Why a splat file could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplatError(String);

impl SplatError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for SplatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SplatError {}

#[derive(Clone, Copy, PartialEq)]
enum Format {
    Ascii,
    Little,
    Big,
}

#[derive(Clone, Copy)]
enum Scalar {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Scalar {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "char" | "int8" => Self::I8,
            "uchar" | "uint8" => Self::U8,
            "short" | "int16" => Self::I16,
            "ushort" | "uint16" => Self::U16,
            "int" | "int32" => Self::I32,
            "uint" | "uint32" => Self::U32,
            "float" | "float32" => Self::F32,
            "double" | "float64" => Self::F64,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// The value at the start of `b`, which holds at least `size()` bytes.
    fn read(self, b: &[u8], big: bool) -> f64 {
        macro_rules! get {
            ($t:ty, $n:expr) => {{
                let a: [u8; $n] = b[..$n].try_into().unwrap();
                (if big { <$t>::from_be_bytes(a) } else { <$t>::from_le_bytes(a) }) as f64
            }};
        }
        match self {
            Self::I8 => b[0] as i8 as f64,
            Self::U8 => b[0] as f64,
            Self::I16 => get!(i16, 2),
            Self::U16 => get!(u16, 2),
            Self::I32 => get!(i32, 4),
            Self::U32 => get!(u32, 4),
            Self::F32 => get!(f32, 4),
            Self::F64 => get!(f64, 8),
        }
    }
}

enum Property {
    Scalar(String, Scalar),
    /// A list: its count's type and its items'. Skipped.
    List(Scalar, Scalar),
}

struct Element {
    name: String,
    count: usize,
    props: Vec<Property>,
}

/// One element's rows, read in order: row `r`'s scalar `j` is
/// `values[r * scalars + j]`.
struct Rows<T = f32> {
    values: Vec<T>,
    scalars: usize,
}

impl GaussianSplats {
    /// Read the PLY the reference 3DGS trainer writes: a `vertex` element with
    /// `x y z`, `scale_0..2` (log standard deviations), and optionally `rot_0..3`
    /// (a quaternion, `w` first), `opacity` (a logit), `f_dc_0..2` and
    /// `f_rest_*` (spherical harmonics, channel-major). A file with
    /// `red green blue` in place of `f_dc_*` reads those as 0..255 colours.
    /// ASCII and binary of either endianness.
    ///
    /// Also the chunk-compressed PLY of PlayCanvas and SuperSplat (a `chunk`
    /// element of bounds, `packed_*` vertex properties and an optional `sh`
    /// element of bytes), decoded as their `splat-transform` decodes it.
    pub fn from_ply(bytes: &[u8]) -> Result<Self, SplatError> {
        let (format, elements, body) = ply_header(bytes)?;
        let mut at = 0;
        let mut text = (format == Format::Ascii)
            .then(|| std::str::from_utf8(body).map(str::split_ascii_whitespace))
            .transpose()
            .map_err(|_| SplatError::new("ASCII PLY body is not UTF-8"))?;
        let packed = elements
            .iter()
            .any(|e| e.name == "vertex" && e.column("packed_position").is_some());
        if packed {
            return from_compressed_ply(&elements, format, body, text.as_mut());
        }
        for e in &elements {
            if e.name != "vertex" {
                skip_element(e, format, body, &mut at, text.as_mut())?;
                continue;
            }
            let rows = read_rows(e, format, body, &mut at, text.as_mut(), |v| v as f32)?;
            return splats_from_vertices(e, &rows);
        }
        Err(SplatError::new("the PLY has no vertex element"))
    }

    /// Read any format this module does, told apart by its first bytes: PLY
    /// (plain or compressed, [`from_ply`](Self::from_ply)), `.spz`
    /// ([`from_spz`](Self::from_spz)), and otherwise `.splat`
    /// ([`from_splat`](Self::from_splat)), which has no header to recognise.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SplatError> {
        if bytes.starts_with(b"ply") {
            Self::from_ply(bytes)
        } else if bytes.starts_with(&[0x1f, 0x8b]) || bytes.starts_with(b"NGSP") {
            Self::from_spz(bytes)
        } else {
            Self::from_splat(bytes)
        }
    }

    /// Write the reference trainer's PLY (binary little-endian, every property
    /// a `float`): what [`from_ply`](Self::from_ply) and the tools that read
    /// the trainer's output read back.
    pub fn to_ply(&self) -> Result<Vec<u8>, SplatError> {
        self.validate()?;
        let n_rest = sh_rest_coefficients(self.sh_degree);
        let mut header = format!(
            "ply\nformat binary_little_endian 1.0\nelement vertex {}\n",
            self.len()
        );
        let mut names: Vec<String> = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
            .map(String::from)
            .into();
        names.extend((0..n_rest * 3).map(|k| format!("f_rest_{k}")));
        names.extend(
            ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
                .map(String::from),
        );
        for name in &names {
            header.push_str(&format!("property float {name}\n"));
        }
        header.push_str("end_header\n");
        let mut out = header.into_bytes();
        out.reserve(self.len() * names.len() * 4);
        let mut put = |v: f32| out.extend_from_slice(&v.to_le_bytes());
        for i in 0..self.len() {
            for v in self.positions[i].into_iter().chain([0.0; 3]) {
                put(v);
            }
            for c in self.colors[i] {
                put((c - 0.5) / SH_C0);
            }
            // On disk the bands are channel-major: every red coefficient, then
            // every green, then every blue.
            let rest = &self.sh[i * n_rest * 3..][..n_rest * 3];
            for ch in 0..3 {
                for k in 0..n_rest {
                    put(rest[k * 3 + ch]);
                }
            }
            let o = self.opacities[i].clamp(1e-7, 1.0 - 1e-7);
            put((o / (1.0 - o)).ln());
            for s in self.scales[i] {
                put(s.max(1e-30).ln());
            }
            let [x, y, z, w] = self.rotations[i];
            for v in [w, x, y, z] {
                put(v);
            }
        }
        Ok(out)
    }

    /// Read the `.splat` format: 32 bytes a splat, little-endian — the centre
    /// and the scales as three `f32`s each, then the colour and opacity as four
    /// bytes, then the rotation as four bytes `w x y z`, each `(q + 1) * 128`.
    /// It has no view-dependent colour, so the result is degree 0.
    pub fn from_splat(bytes: &[u8]) -> Result<Self, SplatError> {
        if !bytes.len().is_multiple_of(32) {
            return Err(SplatError::new(format!(
                "{} bytes is not a whole number of 32-byte splats",
                bytes.len()
            )));
        }
        let mut s = GaussianSplats::default();
        let n = bytes.len() / 32;
        s.positions.reserve(n);
        s.scales.reserve(n);
        s.rotations.reserve(n);
        s.opacities.reserve(n);
        s.colors.reserve(n);
        let f = |b: &[u8], k: usize| f32::from_le_bytes(b[k * 4..k * 4 + 4].try_into().unwrap());
        for b in bytes.as_chunks::<32>().0 {
            s.positions.push([f(b, 0), f(b, 1), f(b, 2)]);
            s.scales.push([f(b, 3), f(b, 4), f(b, 5)]);
            s.colors.push([b[24], b[25], b[26]].map(|c| c as f32 / 255.0));
            s.opacities.push(b[27] as f32 / 255.0);
            let q = |c: u8| (c as f32 - 128.0) / 128.0;
            s.rotations
                .push(normalize_quat([q(b[29]), q(b[30]), q(b[31]), q(b[28])]));
        }
        Ok(s)
    }

    /// Write the `.splat` format (see [`from_splat`](Self::from_splat)).
    /// Colours are clamped to 0..1 and quantised to bytes, as are opacities
    /// and rotations; view-dependent colour is dropped.
    pub fn to_splat(&self) -> Result<Vec<u8>, SplatError> {
        self.validate()?;
        let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        let rot = |v: f32| (v * 128.0 + 128.0).round().clamp(0.0, 255.0) as u8;
        let mut out = Vec::with_capacity(self.len() * 32);
        for i in 0..self.len() {
            for v in self.positions[i].iter().chain(&self.scales[i]) {
                out.extend_from_slice(&v.to_le_bytes());
            }
            let [r, g, b] = self.colors[i].map(byte);
            out.extend_from_slice(&[r, g, b, byte(self.opacities[i])]);
            let [x, y, z, w] = self.rotations[i];
            out.extend_from_slice(&[rot(w), rot(x), rot(y), rot(z)]);
        }
        Ok(out)
    }
}

fn ply_header(bytes: &[u8]) -> Result<(Format, Vec<Element>, &[u8]), SplatError> {
    const END: &[u8] = b"end_header";
    let end = bytes
        .windows(END.len())
        .position(|w| w == END)
        .ok_or_else(|| SplatError::new("no end_header: not a PLY"))?;
    let mut body = end + END.len();
    // The line ends with LF or CRLF.
    if bytes.get(body) == Some(&b'\r') {
        body += 1;
    }
    if bytes.get(body) == Some(&b'\n') {
        body += 1;
    }
    let header = std::str::from_utf8(&bytes[..end])
        .map_err(|_| SplatError::new("the PLY header is not text"))?;
    let mut lines = header.lines().map(str::trim);
    if lines.next() != Some("ply") {
        return Err(SplatError::new("does not start with `ply`"));
    }
    let mut format = None;
    let mut elements: Vec<Element> = Vec::new();
    for line in lines {
        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts.as_slice() {
            ["format", f, _] => {
                format = Some(match *f {
                    "ascii" => Format::Ascii,
                    "binary_little_endian" => Format::Little,
                    "binary_big_endian" => Format::Big,
                    other => return Err(SplatError::new(format!("unknown PLY format {other}"))),
                })
            }
            ["element", name, count] => elements.push(Element {
                name: (*name).into(),
                count: count
                    .parse()
                    .map_err(|_| SplatError::new(format!("bad element count {count}")))?,
                props: Vec::new(),
            }),
            ["property", "list", count, item, _] => {
                let ty = |t: &str| {
                    Scalar::parse(t).ok_or_else(|| SplatError::new(format!("unknown type {t}")))
                };
                let e = elements
                    .last_mut()
                    .ok_or_else(|| SplatError::new("a property before any element"))?;
                e.props.push(Property::List(ty(count)?, ty(item)?));
            }
            ["property", ty, name] => {
                let ty = Scalar::parse(ty)
                    .ok_or_else(|| SplatError::new(format!("unknown type {ty}")))?;
                let e = elements
                    .last_mut()
                    .ok_or_else(|| SplatError::new("a property before any element"))?;
                e.props.push(Property::Scalar((*name).into(), ty));
            }
            _ => {}
        }
    }
    let format = format.ok_or_else(|| SplatError::new("the PLY names no format"))?;
    Ok((format, elements, &bytes[body..]))
}

fn truncated() -> SplatError {
    SplatError::new("the PLY ends before its elements do")
}

/// One value: the next token of an ASCII body, or the next `ty` of a binary
/// one.
fn next_value<'a>(
    ty: Scalar,
    format: Format,
    body: &[u8],
    at: &mut usize,
    text: Option<&mut impl Iterator<Item = &'a str>>,
) -> Result<f64, SplatError> {
    match text {
        Some(tokens) => {
            let t = tokens.next().ok_or_else(truncated)?;
            t.parse()
                .map_err(|_| SplatError::new(format!("`{t}` is not a number")))
        }
        None => {
            let b = body.get(*at..*at + ty.size()).ok_or_else(truncated)?;
            *at += ty.size();
            Ok(ty.read(b, format == Format::Big))
        }
    }
}

fn skip_element<'a>(
    e: &Element,
    format: Format,
    body: &[u8],
    at: &mut usize,
    mut text: Option<&mut impl Iterator<Item = &'a str>>,
) -> Result<(), SplatError> {
    for _ in 0..e.count {
        for p in &e.props {
            match p {
                Property::Scalar(_, ty) => {
                    next_value(*ty, format, body, at, text.as_deref_mut())?;
                }
                Property::List(count, item) => {
                    let n = next_value(*count, format, body, at, text.as_deref_mut())?;
                    for _ in 0..n.max(0.0) as usize {
                        next_value(*item, format, body, at, text.as_deref_mut())?;
                    }
                }
            }
        }
    }
    Ok(())
}

impl Element {
    /// Where the scalar property `name` sits in a row.
    fn column(&self, name: &str) -> Option<usize> {
        self.props
            .iter()
            .position(|p| matches!(p, Property::Scalar(n, _) if n == name))
    }
}

fn read_rows<'a, T>(
    e: &Element,
    format: Format,
    body: &[u8],
    at: &mut usize,
    mut text: Option<&mut impl Iterator<Item = &'a str>>,
    convert: impl Fn(f64) -> T,
) -> Result<Rows<T>, SplatError> {
    let scalars = e
        .props
        .iter()
        .filter(|p| matches!(p, Property::Scalar(..)))
        .count();
    if e.props.len() != scalars {
        return Err(SplatError::new("a list property on the vertex element"));
    }
    // Binary rows of a fixed size: check the length once, up front, so a
    // truncated file fails before allocating for the count it claims.
    if format != Format::Ascii {
        let row: usize = e
            .props
            .iter()
            .map(|p| match p {
                Property::Scalar(_, ty) => ty.size(),
                Property::List(..) => 0,
            })
            .sum();
        let need = row.checked_mul(e.count).ok_or_else(truncated)?;
        if body.len().saturating_sub(*at) < need {
            return Err(truncated());
        }
    }
    let mut values = Vec::with_capacity(e.count.min(1 << 24) * scalars);
    for _ in 0..e.count {
        for p in &e.props {
            if let Property::Scalar(_, ty) = p {
                values.push(convert(next_value(*ty, format, body, at, text.as_deref_mut())?));
            }
        }
    }
    Ok(Rows { values, scalars })
}

fn splats_from_vertices(e: &Element, rows: &Rows) -> Result<GaussianSplats, SplatError> {
    let column = |name: &str| e.column(name);
    let need = |name: &str| {
        column(name).ok_or_else(|| {
            SplatError::new(format!("the vertex element has no `{name}`: not a splat PLY"))
        })
    };
    let pos = [need("x")?, need("y")?, need("z")?];
    let scale = [need("scale_0")?, need("scale_1")?, need("scale_2")?];
    let rot = [column("rot_0"), column("rot_1"), column("rot_2"), column("rot_3")];
    let opacity = column("opacity");
    let dc = [column("f_dc_0"), column("f_dc_1"), column("f_dc_2")];
    let rgb = [column("red"), column("green"), column("blue")];
    let mut rest = Vec::new();
    while let Some(c) = column(&format!("f_rest_{}", rest.len())) {
        rest.push(c);
    }
    let degree = match rest.len() {
        0 => 0,
        9 => 1,
        24 => 2,
        45 => 3,
        n => {
            return Err(SplatError::new(format!(
                "{n} f_rest coefficients is no spherical-harmonic degree"
            )))
        }
    };
    let n_rest = sh_rest_coefficients(degree);

    let mut s = GaussianSplats {
        sh_degree: degree,
        ..Default::default()
    };
    let n = e.count;
    s.positions.reserve(n);
    s.scales.reserve(n);
    s.rotations.reserve(n);
    s.opacities.reserve(n);
    s.colors.reserve(n);
    s.sh.reserve(n * n_rest * 3);
    for row in rows.values.chunks_exact(rows.scalars.max(1)).take(n) {
        s.positions.push(pos.map(|c| row[c]));
        s.scales.push(scale.map(|c| row[c].exp()));
        s.rotations.push(match rot {
            [Some(w), Some(x), Some(y), Some(z)] => normalize_quat([row[x], row[y], row[z], row[w]]),
            _ => [0.0, 0.0, 0.0, 1.0],
        });
        s.opacities
            .push(opacity.map_or(1.0, |c| 1.0 / (1.0 + (-row[c]).exp())));
        s.colors.push(match (dc, rgb) {
            ([Some(r), Some(g), Some(b)], _) => [r, g, b].map(|c| 0.5 + SH_C0 * row[c]),
            (_, [Some(r), Some(g), Some(b)]) => [r, g, b].map(|c| row[c] / 255.0),
            _ => [1.0; 3],
        });
        for k in 0..n_rest {
            for ch in 0..3 {
                s.sh.push(row[rest[ch * n_rest + k]]);
            }
        }
    }
    Ok(s)
}

/// Splats a chunk of the compressed PLY covers, and shares bounds across.
const CHUNK: usize = 256;

/// The compressed PLY: `chunk` rows of `f32` bounds (`min_x .. max_z`,
/// `min_scale_x .. max_scale_z`, optionally `min_r .. max_b`), one per 256
/// splats; `vertex` rows of four `u32`s, each a fraction of its chunk's
/// range; and an optional `sh` element of `f_rest_*` bytes.
fn from_compressed_ply<'a>(
    elements: &[Element],
    format: Format,
    body: &[u8],
    mut text: Option<&mut impl Iterator<Item = &'a str>>,
) -> Result<GaussianSplats, SplatError> {
    let mut at = 0;
    let (mut chunks, mut vertices, mut sh) = (None, None, None);
    for e in elements {
        let floats = |at: &mut usize, text: Option<&mut _>| {
            read_rows(e, format, body, at, text, |v| v as f32).map(|rows| Some((e, rows)))
        };
        match e.name.as_str() {
            "chunk" => chunks = floats(&mut at, text.as_deref_mut())?,
            "sh" => sh = floats(&mut at, text.as_deref_mut())?,
            // Exact: a packed `u32` does not survive `f32`.
            "vertex" => {
                vertices = Some((
                    e,
                    read_rows(e, format, body, &mut at, text.as_deref_mut(), |v| v as u32)?,
                ))
            }
            _ => skip_element(e, format, body, &mut at, text.as_deref_mut())?,
        }
    }
    let (Some((ce, chunks)), Some((ve, vertices))) = (chunks, vertices) else {
        return Err(SplatError::new("a compressed PLY needs a chunk and a vertex element"));
    };
    let need = |e: &Element, name: &str| {
        e.column(name).ok_or_else(|| {
            SplatError::new(format!("the compressed PLY's {} has no `{name}`", e.name))
        })
    };
    let n = ve.count;
    if ce.count != n.div_ceil(CHUNK) {
        return Err(SplatError::new(format!(
            "{} chunks for {n} splats; a chunk is {CHUNK}",
            ce.count
        )));
    }
    let bound = |name: &str| need(ce, name);
    let min_pos = [bound("min_x")?, bound("min_y")?, bound("min_z")?];
    let max_pos = [bound("max_x")?, bound("max_y")?, bound("max_z")?];
    let min_scale = [bound("min_scale_x")?, bound("min_scale_y")?, bound("min_scale_z")?];
    let max_scale = [bound("max_scale_x")?, bound("max_scale_y")?, bound("max_scale_z")?];
    let color_range = match [ce.column("min_r"), ce.column("min_g"), ce.column("min_b")] {
        [Some(r), Some(g), Some(b)] => Some(([r, g, b], [bound("max_r")?, bound("max_g")?, bound("max_b")?])),
        _ => None,
    };
    let packed = [
        need(ve, "packed_position")?,
        need(ve, "packed_rotation")?,
        need(ve, "packed_scale")?,
        need(ve, "packed_color")?,
    ];
    let (degree, rest) = match &sh {
        None => (0, Vec::new()),
        Some((e, _)) => {
            let mut rest = Vec::new();
            while let Some(c) = e.column(&format!("f_rest_{}", rest.len())) {
                rest.push(c);
            }
            let degree = match rest.len() {
                9 => 1,
                24 => 2,
                45 => 3,
                k => {
                    return Err(SplatError::new(format!(
                        "{k} f_rest bytes is no spherical-harmonic degree"
                    )))
                }
            };
            if e.count != n {
                return Err(SplatError::new("the compressed PLY's sh and vertex counts differ"));
            }
            (degree, rest)
        }
    };
    let n_rest = sh_rest_coefficients(degree);

    let unorm = |v: u32, bits: u32| (v & ((1 << bits) - 1)) as f32 / ((1 << bits) - 1) as f32;
    let lerp = |a: f32, b: f32, t: f32| a * (1.0 - t) + b * t;
    let unpack_11_10_11 = |v: u32| [unorm(v >> 21, 11), unorm(v >> 11, 10), unorm(v, 11)];

    let mut s = GaussianSplats {
        sh_degree: degree,
        ..Default::default()
    };
    s.positions.reserve(n);
    s.scales.reserve(n);
    s.rotations.reserve(n);
    s.opacities.reserve(n);
    s.colors.reserve(n);
    s.sh.reserve(n * n_rest * 3);
    for (i, v) in vertices.values.chunks_exact(vertices.scalars).take(n).enumerate() {
        let c = &chunks.values[(i / CHUNK) * chunks.scalars..][..chunks.scalars];
        let [position, rotation, scale, color] = packed.map(|k| v[k]);
        let p = unpack_11_10_11(position);
        s.positions.push([0, 1, 2].map(|k| lerp(c[min_pos[k]], c[max_pos[k]], p[k])));
        let t = unpack_11_10_11(scale);
        s.scales
            .push([0, 1, 2].map(|k| lerp(c[min_scale[k]], c[max_scale[k]], t[k]).exp()));
        // The largest of `rot_0..3` (`w x y z`) in the top two bits, the
        // other three in order below, ten bits each over ±√½.
        let norm = std::f32::consts::SQRT_2;
        let [a, b, d] = [20, 10, 0].map(|shift| (unorm(rotation >> shift, 10) - 0.5) * norm);
        let m = (1.0 - (a * a + b * b + d * d)).max(0.0).sqrt();
        let [w, x, y, z] = match rotation >> 30 {
            0 => [m, a, b, d],
            1 => [a, m, b, d],
            2 => [a, b, m, d],
            _ => [a, b, d, m],
        };
        s.rotations.push(normalize_quat([x, y, z, w]));
        // Colour and opacity as bytes; the stored opacity is the logit's
        // sigmoid, which is the opacity.
        let rgba = [24, 16, 8, 0].map(|shift| unorm(color >> shift, 8));
        s.colors.push(match color_range {
            Some((min, max)) => [0, 1, 2].map(|k| lerp(c[min[k]], c[max[k]], rgba[k])),
            None => [rgba[0], rgba[1], rgba[2]],
        });
        s.opacities.push(rgba[3]);
        if let Some((_, rows)) = &sh {
            let row = &rows.values[i * rows.scalars..][..rows.scalars];
            for k in 0..n_rest {
                for ch in 0..3 {
                    let byte = row[rest[ch * n_rest + k]];
                    let unit = match byte as u32 {
                        0 => 0.0,
                        255 => 1.0,
                        b => (b as f32 + 0.5) / 256.0,
                    };
                    s.sh.push((unit - 0.5) * 8.0);
                }
            }
        }
    }
    Ok(s)
}
