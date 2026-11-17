//! Niantic's `.spz`: splats quantised to a few bytes an attribute and
//! compressed, about a tenth of the trainer's PLY. Read here as the reference
//! library (`nianticlabs/spz`, `load-spz.cc`) reads it.
//!
//! Versions 1 to 3 are one gzip stream: a 16-byte header, then each
//! attribute for every splat in turn — positions, opacities, colours,
//! scales, rotations, spherical harmonics. Version 4 has a 32-byte header,
//! a table of contents, and the same six attributes as separate zstd frames.
//! Versions differ in the positions (1: half floats; later: 24-bit fixed
//! point) and the rotations (1 and 2: `x y z` as bytes, `w` rebuilt
//! positive; 3 on: the "smallest three", 32 bits).

use super::{io::SplatError, normalize_quat, sh_rest_coefficients, GaussianSplats, SH_C0};

const MAGIC: u32 = 0x5053_474e; // "NGSP"
const LATEST_VERSION: u32 = 4;
const FLAG_ANTIALIASED: u8 = 0x1;
/// `f_dc` is stored as `0.5 + 0.15 f_dc` in a byte, not with [`SH_C0`].
const COLOR_SCALE: f32 = 0.15;

/// The attribute bytes, one slice each, for `n` splats.
struct Packed<'a> {
    n: usize,
    sh_degree: u32,
    fractional_bits: u32,
    antialiased: bool,
    /// Version 1: half-float positions, six bytes a splat, not nine.
    half_positions: bool,
    /// Version 3 on: four-byte rotations, not three.
    smallest_three: bool,
    positions: &'a [u8],
    alphas: &'a [u8],
    colors: &'a [u8],
    scales: &'a [u8],
    rotations: &'a [u8],
    sh: &'a [u8],
}

/// Coefficients a channel has beyond the first at `degree` 0..=4, as the
/// file stores them: 0, 3, 8, 15 or 24.
fn sh_dim(degree: u32) -> usize {
    let k = degree as usize + 1;
    k * k - 1
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

/// Bytes each attribute takes for `n` splats, in file order. In `u64`: a
/// header's count times 72 overflows a 32-bit `usize`.
fn attribute_sizes(n: usize, sh_degree: u32, half: bool, smallest_three: bool) -> [u64; 6] {
    let n = n as u64;
    [
        n * if half { 6 } else { 9 },
        n,
        n * 3,
        n * 3,
        n * if smallest_three { 4 } else { 3 },
        n * sh_dim(sh_degree) as u64 * 3,
    ]
}

impl GaussianSplats {
    /// Read Niantic's `.spz`, any version (1 to 4).
    ///
    /// Spherical harmonics past degree 3 (version 4 allows 4) are dropped.
    /// Coordinates are as stored: `.spz` files are written right-handed with
    /// `y` up and `-z` forward (as three.js is), so a scene converted from a
    /// trainer's PLY, which has `y` down, comes out upright where the PLY
    /// did not. Extensions (georeferences, custom attributes) are skipped.
    pub fn from_spz(bytes: &[u8]) -> Result<Self, SplatError> {
        if bytes.len() >= 4 && u32_at(bytes, 0) == MAGIC {
            from_ngsp(bytes)
        } else if bytes.starts_with(&[0x1f, 0x8b]) {
            let raw = gunzip(bytes)?;
            unpack(&legacy(&raw)?)
        } else {
            Err(SplatError::new("not an .spz file: neither gzip nor NGSP"))
        }
    }
}

/// The 16-byte header and attributes of versions 1 to 3, already inflated.
fn legacy(raw: &[u8]) -> Result<Packed<'_>, SplatError> {
    if raw.len() < 16 || u32_at(raw, 0) != MAGIC {
        return Err(SplatError::new("the .spz has no NGSP header"));
    }
    let version = u32_at(raw, 4);
    if !(1..=LATEST_VERSION).contains(&version) {
        return Err(SplatError::new(format!(".spz version {version} is not supported")));
    }
    let n = u32_at(raw, 8) as usize;
    let (sh_degree, fractional_bits, flags) = (raw[12] as u32, raw[13] as u32, raw[14]);
    check_header(n, sh_degree)?;
    let half = version == 1;
    let smallest_three = version >= 3;
    let sizes = attribute_sizes(n, sh_degree, half, smallest_three);
    // Extensions may follow the attributes; they are not read.
    if ((raw.len() - 16) as u64) < sizes.iter().sum::<u64>() {
        return Err(SplatError::new("the .spz ends before its splats do"));
    }
    let mut at = 16;
    let mut next = |size: u64| {
        at += size as usize;
        &raw[at - size as usize..at]
    };
    Ok(Packed {
        n,
        sh_degree,
        fractional_bits,
        antialiased: flags & FLAG_ANTIALIASED != 0,
        half_positions: half,
        smallest_three,
        positions: next(sizes[0]),
        alphas: next(sizes[1]),
        colors: next(sizes[2]),
        scales: next(sizes[3]),
        rotations: next(sizes[4]),
        sh: next(sizes[5]),
    })
}

fn check_header(n: usize, sh_degree: u32) -> Result<(), SplatError> {
    if n == 0 || n > i32::MAX as usize {
        return Err(SplatError::new(format!("{n} splats is not a valid .spz count")));
    }
    if sh_degree > 4 {
        return Err(SplatError::new(format!("spherical-harmonic degree {sh_degree} in an .spz")));
    }
    Ok(())
}

/// Version 4: a 32-byte header, a table of contents (compressed and
/// uncompressed size, eight bytes each, per stream) at `tocByteOffset`, and
/// the streams back to back after it, filling the file.
fn from_ngsp(bytes: &[u8]) -> Result<GaussianSplats, SplatError> {
    if bytes.len() < 32 {
        return Err(SplatError::new("the .spz ends inside its header"));
    }
    let version = u32_at(bytes, 4);
    if version != 4 {
        return Err(SplatError::new(format!(
            "an uncompressed NGSP header with version {version}: only version 4 has one"
        )));
    }
    let n = u32_at(bytes, 8) as usize;
    let (sh_degree, fractional_bits, flags, streams) =
        (bytes[12] as u32, bytes[13] as u32, bytes[14], bytes[15] as usize);
    check_header(n, sh_degree)?;
    // Nothing decompresses past 1024:1; nine bytes a splat is the least.
    if n as u64 > bytes.len() as u64 * 1024 / 9 {
        return Err(SplatError::new(format!("{n} splats cannot fit in {} bytes", bytes.len())));
    }
    let toc = u32_at(bytes, 16) as usize;
    let toc_end = toc as u64 + streams as u64 * 16;
    if toc < 32 || toc_end > bytes.len() as u64 {
        return Err(SplatError::new("the .spz table of contents is outside the file"));
    }
    let sizes = attribute_sizes(n, sh_degree, false, true);
    // Empty attributes (the harmonics at degree 0) have no stream.
    let expected: Vec<u64> = sizes.iter().copied().filter(|&s| s > 0).collect();
    if streams != expected.len() {
        return Err(SplatError::new(format!(
            "the .spz has {streams} streams where {} were expected",
            expected.len()
        )));
    }
    let mut decoded = Vec::with_capacity(streams);
    let mut at = toc_end as usize;
    for (k, &want) in expected.iter().enumerate() {
        let entry = toc + k * 16;
        let (compressed, uncompressed) = (u64_at(bytes, entry), u64_at(bytes, entry + 8));
        if uncompressed != want {
            return Err(SplatError::new("an .spz stream has the wrong size"));
        }
        if compressed > (bytes.len() - at) as u64 {
            return Err(SplatError::new("an .spz stream runs past the end of the file"));
        }
        let frame = &bytes[at..at + compressed as usize];
        at += compressed as usize;
        decoded.push(unzstd(frame, want)?);
    }
    if at != bytes.len() {
        return Err(SplatError::new("the .spz streams do not fill the file"));
    }
    let mut streams = decoded.iter().map(Vec::as_slice);
    let mut next = |size: u64| if size == 0 { &[][..] } else { streams.next().unwrap() };
    unpack(&Packed {
        n,
        sh_degree,
        fractional_bits,
        antialiased: flags & FLAG_ANTIALIASED != 0,
        half_positions: false,
        smallest_three: true,
        positions: next(sizes[0]),
        alphas: next(sizes[1]),
        colors: next(sizes[2]),
        scales: next(sizes[3]),
        rotations: next(sizes[4]),
        sh: next(sizes[5]),
    })
}

/// The first zstd frame of `frame`, which must decode to exactly `size`
/// bytes. Extension frames may follow the last stream's; they are left.
fn unzstd(frame: &[u8], size: u64) -> Result<Vec<u8>, SplatError> {
    use std::io::Read;
    let bad = |e: &dyn std::fmt::Display| SplatError::new(format!("an .spz stream: {e}"));
    let decoder = ruzstd::decoding::StreamingDecoder::new(frame).map_err(|e| bad(&e))?;
    // The size is the header's word: reserve no more than the frame could
    // plausibly hold, and let a larger one grow as it really decodes.
    let mut out = Vec::with_capacity(size.min(frame.len() as u64 * 64) as usize);
    // One byte more than wanted, to notice a stream that is too long.
    decoder
        .take(size + 1)
        .read_to_end(&mut out)
        .map_err(|e| bad(&e))?;
    if out.len() as u64 != size {
        return Err(SplatError::new("an .spz stream decodes to the wrong size"));
    }
    Ok(out)
}

/// The deflate stream of a gzip member (RFC 1952), inflated.
fn gunzip(bytes: &[u8]) -> Result<Vec<u8>, SplatError> {
    let short = || SplatError::new("the .spz ends inside its gzip header");
    if bytes.len() < 18 || bytes[2] != 8 {
        return Err(SplatError::new("not a deflate gzip stream"));
    }
    let flags = bytes[3];
    let mut at = 10;
    if flags & 0x04 != 0 {
        let extra = u16::from_le_bytes(bytes.get(at..at + 2).ok_or_else(short)?.try_into().unwrap());
        at += 2 + extra as usize;
    }
    // A file name, then a comment, each zero-terminated.
    for bit in [0x08, 0x10] {
        if flags & bit != 0 {
            let rest = bytes.get(at..).ok_or_else(short)?;
            at += rest.iter().position(|&b| b == 0).ok_or_else(short)? + 1;
        }
    }
    if flags & 0x02 != 0 {
        at += 2;
    }
    let body = bytes.get(at..).ok_or_else(short)?;
    crate::loaders::deflate::inflate_raw(body)
        .map_err(|e| SplatError::new(format!("the .spz does not inflate: {e:?}")))
}

fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as f32;
    sign * match exp {
        0 => man * 2f32.powi(-24),
        31 if man == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        e => (1.0 + man / 1024.0) * 2f32.powi(e - 15),
    }
}

/// Bytes `x y z`, each `(q + 1) * 127.5`; `w` is rebuilt non-negative.
fn first_three(r: &[u8]) -> [f32; 4] {
    let [x, y, z] = [r[0], r[1], r[2]].map(|b| b as f32 / 127.5 - 1.0);
    let w = (1.0 - (x * x + y * y + z * z)).max(0.0).sqrt();
    [x, y, z, w]
}

/// 32 bits, little-endian: the index of the largest component (`x y z w`)
/// in the top two, then the other three from `w` down, ten bits each — a
/// sign over a nine-bit magnitude in units of `√½ / 511` — from the bottom.
fn smallest_three(r: &[u8]) -> [f32; 4] {
    const MASK: u32 = (1 << 9) - 1;
    let mut comp = u32::from_le_bytes([r[0], r[1], r[2], r[3]]);
    let largest = (comp >> 30) as usize;
    let mut q = [0.0f32; 4];
    let mut sum = 0.0;
    for i in (0..4).rev() {
        if i == largest {
            continue;
        }
        let magnitude = std::f32::consts::FRAC_1_SQRT_2 * (comp & MASK) as f32 / MASK as f32;
        q[i] = if (comp >> 9) & 1 == 1 { -magnitude } else { magnitude };
        sum += q[i] * q[i];
        comp >>= 10;
    }
    q[largest] = (1.0 - sum).max(0.0).sqrt();
    q
}

fn unpack(p: &Packed) -> Result<GaussianSplats, SplatError> {
    let degree = p.sh_degree.min(3);
    let (file_dim, n_rest) = (sh_dim(p.sh_degree), sh_rest_coefficients(degree));
    let mut s = GaussianSplats {
        sh_degree: degree,
        antialiased: p.antialiased,
        ..Default::default()
    };
    s.positions.reserve(p.n);
    s.scales.reserve(p.n);
    s.rotations.reserve(p.n);
    s.opacities.reserve(p.n);
    s.colors.reserve(p.n);
    s.sh.reserve(p.n * n_rest * 3);
    let unit = 1.0 / (1u64 << p.fractional_bits.min(31)) as f32;
    let rotation_bytes = if p.smallest_three { 4 } else { 3 };
    for i in 0..p.n {
        s.positions.push(if p.half_positions {
            let b = &p.positions[i * 6..][..6];
            [0, 1, 2].map(|k| half_to_f32(u16::from_le_bytes([b[k * 2], b[k * 2 + 1]])))
        } else {
            let b = &p.positions[i * 9..][..9];
            [0, 1, 2].map(|k| {
                let v = u32::from_le_bytes([b[k * 3], b[k * 3 + 1], b[k * 3 + 2], 0]);
                // Sign-extend the 24 bits.
                ((v << 8) as i32 >> 8) as f32 * unit
            })
        });
        s.scales
            .push([0, 1, 2].map(|k| (p.scales[i * 3 + k] as f32 / 16.0 - 10.0).exp()));
        let r = &p.rotations[i * rotation_bytes..][..rotation_bytes];
        s.rotations.push(normalize_quat(if p.smallest_three {
            smallest_three(r)
        } else {
            first_three(r)
        }));
        // The byte is the sigmoid of the trained logit: the opacity itself.
        s.opacities.push(p.alphas[i] as f32 / 255.0);
        s.colors.push([0, 1, 2].map(|k| {
            let dc = (p.colors[i * 3 + k] as f32 / 255.0 - 0.5) / COLOR_SCALE;
            0.5 + SH_C0 * dc
        }));
        // Coefficient-major with RGB inner, as `sh` is; degree 4's extra
        // coefficients come last and are dropped.
        let sh = &p.sh[i * file_dim * 3..][..n_rest * 3];
        s.sh.extend(sh.iter().map(|&b| (b as f32 - 128.0) / 128.0));
    }
    Ok(s)
}
