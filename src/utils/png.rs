//! Dependency-free PNG encode/decode, plus the DEFLATE codec underneath it.
//!
//! threers renders to RGBA and reads reference imagery back in for comparison,
//! so both directions need to exist without pulling in an image crate. The
//! encoder writes each image in the smallest layout that holds it exactly
//! (grey at 1 to 8 bits, a palette with transparency, RGB, RGBA, or 16 bits a
//! channel from [`encode_png16`]), picks each row's filter as libpng does,
//! and compresses with LZ77 and dynamic Huffman codes. The decoder takes every
//! colour type, bit depth and filter, interlaced (Adam7) or not, with `tRNS`
//! transparency; [`decode_png16`] keeps 16-bit precision.
//!
//! ```ignore
//! use threers::{decode_png, encode_png};
//!
//! let img = decode_png(&std::fs::read("ref.png")?)?;
//! let same = encode_png(img.width, img.height, &img.rgba);
//! ```

/// A decoded image, always 8-bit RGBA.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PngImage {
    pub width: u32,
    pub height: u32,
    /// `width * height * 4` bytes, row-major from the top-left.
    pub rgba: Vec<u8>,
}

impl PngImage {
    /// Pixel at `(x, y)` as `[r, g, b, a]`, or opaque black when out of range.
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        if x >= self.width || y >= self.height {
            return [0, 0, 0, 255];
        }
        let i = ((y * self.width + x) * 4) as usize;
        [
            self.rgba[i],
            self.rgba[i + 1],
            self.rgba[i + 2],
            self.rgba[i + 3],
        ]
    }

    /// Luminance at `(x, y)` in 0..=1, alpha-composited over white.
    pub fn luma(&self, x: u32, y: u32) -> f32 {
        let [r, g, b, a] = self.pixel(x, y);
        let a = a as f32 / 255.0;
        let l = (0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32) / 255.0;
        l * a + (1.0 - a)
    }
}

// ---------------------------------------------------------------- checksums

pub(crate) fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

// ------------------------------------------------------------ deflate encode

struct BitWriter {
    out: Vec<u8>,
    bit: u32,
    acc: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            bit: 0,
            acc: 0,
        }
    }
    /// DEFLATE writes Huffman codes MSB-first but everything else LSB-first;
    /// `push` takes already-reversed bits.
    fn push(&mut self, value: u32, bits: u32) {
        self.acc |= value << self.bit;
        self.bit += bits;
        while self.bit >= 8 {
            self.out.push((self.acc & 0xFF) as u8);
            self.acc >>= 8;
            self.bit -= 8;
        }
    }
    fn push_rev(&mut self, code: u32, bits: u32) {
        let mut v = 0u32;
        for i in 0..bits {
            v |= ((code >> (bits - 1 - i)) & 1) << i;
        }
        self.push(v, bits);
    }
    fn finish(mut self) -> Vec<u8> {
        if self.bit > 0 {
            self.out.push((self.acc & 0xFF) as u8);
        }
        self.out
    }
}

const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEN_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// What the LZ77 pass found, kept rather than emitted, because the Huffman
/// code cannot be chosen until the symbol frequencies are known.
enum Token {
    Lit(u8),
    Match { len: u16, dist: u16 },
}

/// Canonical Huffman code lengths for a set of frequencies, none longer than
/// `max_bits`.
///
/// Built by the usual repeated-merge, then — if anything came out too deep —
/// the frequencies are halved and it runs again. Crude next to package-merge,
/// but it converges quickly (flattening the distribution shortens the tree, and
/// in the limit a balanced tree over 286 symbols is nine deep) and it cannot
/// produce an invalid code, which package-merge implemented in a hurry can.
fn huffman_lengths(freq: &[u32], max_bits: u8) -> Vec<u8> {
    let n = freq.len();
    let mut scaled: Vec<u32> = freq.to_vec();
    loop {
        let used: Vec<usize> = (0..n).filter(|&i| scaled[i] > 0).collect();
        let mut lengths = vec![0u8; n];
        match used.len() {
            0 => return lengths,
            // A code needs at least two symbols; DEFLATE allows a one-symbol
            // alphabet only by giving it a length of 1.
            1 => {
                lengths[used[0]] = 1;
                return lengths;
            }
            _ => {}
        }
        // Merge the two lightest nodes until one remains, tracking depth by
        // incrementing every leaf under each merged node.
        let mut nodes: Vec<(u64, Vec<usize>)> =
            used.iter().map(|&i| (scaled[i] as u64, vec![i])).collect();
        while nodes.len() > 1 {
            nodes.sort_by_key(|n| std::cmp::Reverse(n.0));
            let a = nodes.pop().unwrap();
            let b = nodes.pop().unwrap();
            for &l in a.1.iter().chain(b.1.iter()) {
                lengths[l] += 1;
            }
            let mut merged = a.1;
            merged.extend(b.1);
            nodes.push((a.0 + b.0, merged));
        }
        if lengths.iter().all(|&l| l <= max_bits) {
            return lengths;
        }
        for f in scaled.iter_mut() {
            if *f > 0 {
                *f = (*f).div_ceil(2);
            }
        }
    }
}

/// Canonical codes from lengths, in DEFLATE's order: shorter codes first, and
/// within a length, ascending by symbol.
fn canonical_codes(lengths: &[u8]) -> Vec<u16> {
    let max = *lengths.iter().max().unwrap_or(&0) as usize;
    let mut count = vec![0u16; max + 1];
    for &l in lengths {
        if l > 0 {
            count[l as usize] += 1;
        }
    }
    let mut next = vec![0u16; max + 2];
    let mut code = 0u16;
    for bits in 1..=max {
        code = (code + count[bits - 1]) << 1;
        next[bits] = code;
    }
    let mut codes = vec![0u16; lengths.len()];
    for (sym, &l) in lengths.iter().enumerate() {
        if l > 0 {
            codes[sym] = next[l as usize];
            next[l as usize] += 1;
        }
    }
    codes
}

/// Run-length encode a table of code lengths using DEFLATE's 16/17/18 symbols.
fn rle_code_lengths(lengths: &[u8]) -> Vec<(u8, u8, u8)> {
    // (symbol, extra bits value, extra bit count)
    let mut out = Vec::new();
    let mut i = 0;
    while i < lengths.len() {
        let v = lengths[i];
        let mut run = 1;
        while i + run < lengths.len() && lengths[i + run] == v {
            run += 1;
        }
        if v == 0 {
            while run >= 11 {
                let n = run.min(138);
                out.push((18u8, (n - 11) as u8, 7));
                run -= n;
                i += n;
            }
            while run >= 3 {
                let n = run.min(10);
                out.push((17u8, (n - 3) as u8, 3));
                run -= n;
                i += n;
            }
            for _ in 0..run {
                out.push((0u8, 0, 0));
                i += 1;
            }
        } else {
            out.push((v, 0, 0));
            i += 1;
            run -= 1;
            // 16 repeats the *previous* length, so the first one is literal.
            while run >= 3 {
                let n = run.min(6);
                out.push((16u8, (n - 3) as u8, 2));
                run -= n;
                i += n;
            }
            for _ in 0..run {
                out.push((v, 0, 0));
                i += 1;
            }
        }
    }
    out
}

/// LZ77 + dynamic Huffman.
///
/// This used to emit fixed Huffman codes as it went, which is simple and about
/// a third bigger than it needs to be: measured over the baked planet maps,
/// recompressing the very same filtered bytes with a real encoder took 62.7 MB
/// to 45.0 MB. Fixed codes assume a distribution the data does not have — a
/// height map is mostly small deltas and a roughness map is three values, and
/// neither looks anything like the average of all files ever measured in 1990.
///
/// Choosing the code needs the frequencies, so the match pass now collects
/// tokens first and the writing happens after. That also makes it worth
/// searching harder for matches, since a longer match is fewer symbols.
fn deflate(data: &[u8]) -> Vec<u8> {
    const WINDOW: usize = 32768;
    const MIN_MATCH: usize = 3;
    const MAX_MATCH: usize = 258;
    const HASH_BITS: usize = 15;
    // Left where it was. Searching sixteen times harder is worth about five
    // per cent of the output — measured 21.1 MB against 20.0 MB over the baked
    // planet maps — because the win here was never the matching. It was the
    // code: fixed Huffman assumes a distribution real image data does not have.
    // `encode_png` writes every frame of an animation export, so the cheap
    // search stays and the saving comes for free.
    const MAX_CHAIN: usize = 32;

    let mut head = vec![usize::MAX; 1 << HASH_BITS];
    let mut prev = vec![usize::MAX; data.len().max(1)];
    let hash = |d: &[u8], i: usize| -> usize {
        ((d[i] as usize) << 10 ^ (d[i + 1] as usize) << 5 ^ d[i + 2] as usize)
            & ((1 << HASH_BITS) - 1)
    };

    let mut tokens: Vec<Token> = Vec::new();
    let mut lit_freq = vec![0u32; 286];
    let mut dist_freq = vec![0u32; 30];

    let mut i = 0usize;
    while i < data.len() {
        let (mut best_len, mut best_dist) = (0usize, 0usize);
        if i + MIN_MATCH <= data.len() {
            let h = hash(data, i);
            let mut cand = head[h];
            let mut chain = 0;
            while cand != usize::MAX && chain < MAX_CHAIN {
                if i - cand > WINDOW {
                    break;
                }
                let max = MAX_MATCH.min(data.len() - i);
                let mut l = 0;
                while l < max && data[cand + l] == data[i + l] {
                    l += 1;
                }
                if l > best_len {
                    best_len = l;
                    best_dist = i - cand;
                    if l >= MAX_MATCH {
                        break;
                    }
                }
                cand = prev[cand];
                chain += 1;
            }
            prev[i] = head[h];
            head[h] = i;
        }

        if best_len >= MIN_MATCH {
            let li = LEN_BASE
                .iter()
                .rposition(|&b| b as usize <= best_len)
                .unwrap();
            let di = DIST_BASE
                .iter()
                .rposition(|&b| b as usize <= best_dist)
                .unwrap();
            lit_freq[257 + li] += 1;
            dist_freq[di] += 1;
            tokens.push(Token::Match {
                len: best_len as u16,
                dist: best_dist as u16,
            });
            // `k` is a position in the window: hashed, stored, and chained.
            #[allow(clippy::needless_range_loop)]
            for k in (i + 1)..(i + best_len).min(data.len().saturating_sub(MIN_MATCH - 1)) {
                let h = hash(data, k);
                prev[k] = head[h];
                head[h] = k;
            }
            i += best_len;
        } else {
            lit_freq[data[i] as usize] += 1;
            tokens.push(Token::Lit(data[i]));
            i += 1;
        }
    }
    lit_freq[256] += 1; // end of block

    let lit_lengths = huffman_lengths(&lit_freq, 15);
    // A distance code is required even when nothing used one.
    if dist_freq.iter().all(|&f| f == 0) {
        dist_freq[0] = 1;
    }
    let dist_lengths = huffman_lengths(&dist_freq, 15);
    let lit_codes = canonical_codes(&lit_lengths);
    let dist_codes = canonical_codes(&dist_lengths);

    // HLIT / HDIST: the tables are written truncated to their last used symbol.
    let hlit = (lit_lengths.iter().rposition(|&l| l > 0).unwrap_or(256) + 1).max(257);
    let hdist = (dist_lengths.iter().rposition(|&l| l > 0).unwrap_or(0) + 1).max(1);

    // The two tables are RLE'd together, then that stream gets its own code.
    let mut table: Vec<u8> = lit_lengths[..hlit].to_vec();
    table.extend_from_slice(&dist_lengths[..hdist]);
    let rle = rle_code_lengths(&table);
    let mut cl_freq = vec![0u32; 19];
    for &(sym, _, _) in &rle {
        cl_freq[sym as usize] += 1;
    }
    let cl_lengths = huffman_lengths(&cl_freq, 7);
    let cl_codes = canonical_codes(&cl_lengths);
    // The code-length table is itself written in a fixed, frequency-ordered
    // permutation, so the common entries come first and trailing zeros drop.
    const CL_ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    let hclen = (CL_ORDER
        .iter()
        .rposition(|&i| cl_lengths[i] > 0)
        .unwrap_or(3)
        + 1)
    .max(4);

    let mut w = BitWriter::new();
    w.push(1, 1); // BFINAL
    w.push(2, 2); // BTYPE = 10, dynamic Huffman
    w.push((hlit - 257) as u32, 5);
    w.push((hdist - 1) as u32, 5);
    w.push((hclen - 4) as u32, 4);
    for &idx in CL_ORDER.iter().take(hclen) {
        w.push(cl_lengths[idx] as u32, 3);
    }
    for &(sym, extra, bits) in &rle {
        w.push_rev(
            cl_codes[sym as usize] as u32,
            cl_lengths[sym as usize] as u32,
        );
        if bits > 0 {
            w.push(extra as u32, bits as u32);
        }
    }
    for t in &tokens {
        match *t {
            Token::Lit(b) => {
                w.push_rev(lit_codes[b as usize] as u32, lit_lengths[b as usize] as u32);
            }
            Token::Match { len, dist } => {
                let li = LEN_BASE
                    .iter()
                    .rposition(|&b| b as usize <= len as usize)
                    .unwrap();
                let sym = 257 + li;
                w.push_rev(lit_codes[sym] as u32, lit_lengths[sym] as u32);
                if LEN_EXTRA[li] > 0 {
                    w.push((len as u32) - LEN_BASE[li] as u32, LEN_EXTRA[li] as u32);
                }
                let di = DIST_BASE
                    .iter()
                    .rposition(|&b| b as usize <= dist as usize)
                    .unwrap();
                w.push_rev(dist_codes[di] as u32, dist_lengths[di] as u32);
                if DIST_EXTRA[di] > 0 {
                    w.push((dist as u32) - DIST_BASE[di] as u32, DIST_EXTRA[di] as u32);
                }
            }
        }
    }
    w.push_rev(lit_codes[256] as u32, lit_lengths[256] as u32);
    w.finish()
}

/// `data` deflated in a zlib stream (RFC 1950), as PNG and PDF carry it.
pub(crate) fn zlib(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    out.extend_from_slice(&deflate(data));
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

// ------------------------------------------------------------ deflate decode

struct BitReader<'a> {
    d: &'a [u8],
    pos: usize,
    bit: u32,
}

impl<'a> BitReader<'a> {
    fn new(d: &'a [u8]) -> Self {
        Self { d, pos: 0, bit: 0 }
    }
    fn bits(&mut self, n: u32) -> Result<u32, String> {
        let mut v = 0u32;
        for i in 0..n {
            if self.pos >= self.d.len() {
                return Err("deflate: out of input".into());
            }
            let b = (self.d[self.pos] >> self.bit) & 1;
            v |= (b as u32) << i;
            self.bit += 1;
            if self.bit == 8 {
                self.bit = 0;
                self.pos += 1;
            }
        }
        Ok(v)
    }
    fn align(&mut self) {
        if self.bit > 0 {
            self.bit = 0;
            self.pos += 1;
        }
    }
}

/// Canonical Huffman decoding table built from code lengths.
struct Huffman {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Huffman {
        let mut counts = [0u16; 16];
        for &l in lengths {
            counts[l as usize] += 1;
        }
        counts[0] = 0;
        let mut offs = [0u16; 16];
        for i in 1..16 {
            offs[i] = offs[i - 1] + counts[i - 1];
        }
        let mut symbols = vec![0u16; lengths.len()];
        for (sym, &l) in lengths.iter().enumerate() {
            if l != 0 {
                symbols[offs[l as usize] as usize] = sym as u16;
                offs[l as usize] += 1;
            }
        }
        Huffman { counts, symbols }
    }

    fn decode(&self, r: &mut BitReader) -> Result<u16, String> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= r.bits(1)? as i32;
            let count = self.counts[len] as i32;
            if code - first < count {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err("deflate: bad Huffman code".into())
    }
}

fn inflate(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut r = BitReader::new(data);
    let mut out: Vec<u8> = Vec::new();
    loop {
        let final_block = r.bits(1)?;
        let btype = r.bits(2)?;
        match btype {
            0 => {
                r.align();
                if r.pos + 4 > r.d.len() {
                    return Err("deflate: truncated stored block".into());
                }
                let len = u16::from_le_bytes([r.d[r.pos], r.d[r.pos + 1]]) as usize;
                r.pos += 4;
                if r.pos + len > r.d.len() {
                    return Err("deflate: truncated stored data".into());
                }
                out.extend_from_slice(&r.d[r.pos..r.pos + len]);
                r.pos += len;
            }
            1 | 2 => {
                let (lit, dist) = if btype == 1 {
                    let mut l = [0u8; 288];
                    for (i, slot) in l.iter_mut().enumerate() {
                        *slot = match i {
                            0..=143 => 8,
                            144..=255 => 9,
                            256..=279 => 7,
                            _ => 8,
                        };
                    }
                    (Huffman::new(&l), Huffman::new(&[5u8; 30]))
                } else {
                    let hlit = r.bits(5)? as usize + 257;
                    let hdist = r.bits(5)? as usize + 1;
                    let hclen = r.bits(4)? as usize + 4;
                    const ORDER: [usize; 19] = [
                        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
                    ];
                    let mut cl = [0u8; 19];
                    for &o in ORDER.iter().take(hclen) {
                        cl[o] = r.bits(3)? as u8;
                    }
                    let clh = Huffman::new(&cl);
                    let mut lengths = vec![0u8; hlit + hdist];
                    let mut i = 0;
                    while i < lengths.len() {
                        let sym = clh.decode(&mut r)?;
                        match sym {
                            0..=15 => {
                                lengths[i] = sym as u8;
                                i += 1;
                            }
                            16 => {
                                if i == 0 {
                                    return Err("deflate: repeat with no previous length".into());
                                }
                                let prev = lengths[i - 1];
                                let n = 3 + r.bits(2)? as usize;
                                for _ in 0..n {
                                    if i < lengths.len() {
                                        lengths[i] = prev;
                                        i += 1;
                                    }
                                }
                            }
                            17 => {
                                let n = 3 + r.bits(3)? as usize;
                                i = (i + n).min(lengths.len());
                            }
                            18 => {
                                let n = 11 + r.bits(7)? as usize;
                                i = (i + n).min(lengths.len());
                            }
                            _ => return Err("deflate: bad code-length symbol".into()),
                        }
                    }
                    (
                        Huffman::new(&lengths[..hlit]),
                        Huffman::new(&lengths[hlit..]),
                    )
                };

                loop {
                    let sym = lit.decode(&mut r)?;
                    match sym {
                        0..=255 => out.push(sym as u8),
                        256 => break,
                        _ => {
                            let li = sym as usize - 257;
                            if li >= LEN_BASE.len() {
                                return Err("deflate: bad length symbol".into());
                            }
                            let len =
                                LEN_BASE[li] as usize + r.bits(LEN_EXTRA[li] as u32)? as usize;
                            let ds = dist.decode(&mut r)? as usize;
                            if ds >= DIST_BASE.len() {
                                return Err("deflate: bad distance symbol".into());
                            }
                            let d =
                                DIST_BASE[ds] as usize + r.bits(DIST_EXTRA[ds] as u32)? as usize;
                            if d > out.len() {
                                return Err("deflate: distance before start".into());
                            }
                            let start = out.len() - d;
                            for k in 0..len {
                                let b = out[start + k];
                                out.push(b);
                            }
                        }
                    }
                }
            }
            _ => return Err("deflate: reserved block type".into()),
        }
        if final_block == 1 {
            break;
        }
    }
    Ok(out)
}

// ------------------------------------------------------------------ PNG I/O

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_in = kind.to_vec();
    crc_in.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_in).to_be_bytes());
}

/// Encode 8-bit RGBA pixels as a PNG, in the smallest layout that holds
/// them exactly: grey (at 1, 2, 4 or 8 bits) when every pixel is grey and
/// opaque, grey and alpha when grey, a palette (with `tRNS` for its
/// transparent entries) when there are at most 256 colours and that comes
/// out smaller, else RGB when opaque or RGBA. [`decode_png`] gives back the same pixels.
///
/// Panics if `rgba.len() != width * height * 4`.
pub fn encode_png(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    encode(width, height, rgba, None)
}

/// Encode 8-bit RGBA pixels as a PNG that says its resolution: `dpi` dots
/// an inch (a `pHYs` chunk, in pixels a metre), as print and figure
/// workflows read it.
///
/// Panics if `rgba.len() != width * height * 4`.
pub fn encode_png_with_dpi(width: u32, height: u32, rgba: &[u8], dpi: f64) -> Vec<u8> {
    encode(width, height, rgba, Some((dpi / 0.0254).round().max(1.0) as u32))
}

/// Encode 16-bit RGBA pixels as a PNG: 16 bits a channel (grey, grey and
/// alpha, RGB or RGBA, whichever holds them exactly), or [`encode_png`]'s
/// 8-bit layouts when every sample is an 8-bit value widened (`v * 257`).
///
/// Panics if `rgba.len() != width * height * 4`.
pub fn encode_png16(width: u32, height: u32, rgba: &[u16]) -> Vec<u8> {
    assert_eq!(rgba.len(), (width as usize) * (height as usize) * 4, "rgba size mismatch");
    if rgba.iter().all(|&v| v % 257 == 0) {
        let narrow: Vec<u8> = rgba.iter().map(|&v| (v / 257) as u8).collect();
        return encode(width, height, &narrow, None);
    }
    let px = rgba.as_chunks::<4>().0;
    let grey = px.iter().all(|p| p[0] == p[1] && p[1] == p[2]);
    let opaque = px.iter().all(|p| p[3] == u16::MAX);
    let (colour, take): (u8, &[usize]) = match (grey, opaque) {
        (true, true) => (0, &[0]),
        (true, false) => (4, &[0, 3]),
        (false, true) => (2, &[0, 1, 2]),
        (false, false) => (6, &[0, 1, 2, 3]),
    };
    let w = width as usize;
    let rows: Vec<Vec<u8>> = (0..height as usize)
        .map(|y| px[y * w..(y + 1) * w].iter().flat_map(|p| take.iter().flat_map(move |&c| p[c].to_be_bytes())).collect())
        .collect();
    write_png(width, height, 16, colour, &[], &[], &rows, take.len() * 2, true, None)
}

/// The 8-bit layout [`encode_png`] picks for an image, and its rows packed.
struct Layout {
    depth: u8,
    colour: u8,
    palette: Vec<u8>,
    trns: Vec<u8>,
    rows: Vec<Vec<u8>>,
    /// Bytes a pixel, which the filters step back by (1 for packed pixels).
    bpp: usize,
}

fn layout(width: u32, height: u32, rgba: &[u8], allow_palette: bool) -> Layout {
    let w = width as usize;
    let px = rgba.as_chunks::<4>().0;
    let grey = px.iter().all(|p| p[0] == p[1] && p[1] == p[2]);
    let opaque = px.iter().all(|p| p[3] == 255);
    let rows_of = |f: &dyn Fn(&[[u8; 4]]) -> Vec<u8>| (0..height as usize).map(|y| f(&px[y * w..(y + 1) * w])).collect::<Vec<_>>();
    if grey && opaque {
        // the smallest depth whose levels hit every grey exactly
        let depth = [1u8, 2, 4].into_iter().find(|&d| {
            let step = 255 / ((1u16 << d) - 1) as u8;
            px.iter().all(|p| p[0] % step == 0)
        });
        return match depth {
            Some(d) => {
                let step = 255 / ((1u16 << d) - 1) as u8;
                let rows = rows_of(&|r| pack(r.iter().map(|p| p[0] / step), d));
                Layout { depth: d, colour: 0, palette: Vec::new(), trns: Vec::new(), rows, bpp: 1 }
            }
            None => Layout { depth: 8, colour: 0, palette: Vec::new(), trns: Vec::new(), rows: rows_of(&|r| r.iter().map(|p| p[0]).collect()), bpp: 1 },
        };
    }
    if let Some(colours) = palette_of(px).filter(|_| allow_palette) {
        let index: std::collections::HashMap<[u8; 4], u8> = colours.iter().enumerate().map(|(k, c)| (*c, k as u8)).collect();
        let depth = match colours.len() {
            0..=2 => 1,
            3..=4 => 2,
            5..=16 => 4,
            _ => 8,
        };
        let palette = colours.iter().flat_map(|c| [c[0], c[1], c[2]]).collect();
        // transparent entries come first, so tRNS stops at the last of them
        let trns: Vec<u8> = colours.iter().take_while(|c| c[3] < 255).map(|c| c[3]).collect();
        let rows = rows_of(&|r| pack(r.iter().map(|p| index[p]), depth));
        return Layout { depth, colour: 3, palette, trns, rows, bpp: 1 };
    }
    let (colour, take): (u8, &[usize]) = match (grey, opaque) {
        (true, _) => (4, &[0, 3]),
        (false, true) => (2, &[0, 1, 2]),
        (false, false) => (6, &[0, 1, 2, 3]),
    };
    let rows = rows_of(&|r| r.iter().flat_map(|p| take.iter().map(move |&c| p[c])).collect());
    Layout { depth: 8, colour, palette: Vec::new(), trns: Vec::new(), rows, bpp: take.len() }
}

/// The image's colours when there are at most 256: transparent ones first,
/// each group in order of first appearance.
fn palette_of(px: &[[u8; 4]]) -> Option<Vec<[u8; 4]>> {
    let mut seen = std::collections::HashSet::new();
    let mut order = Vec::new();
    for p in px {
        if seen.insert(*p) {
            if order.len() == 256 {
                return None;
            }
            order.push(*p);
        }
    }
    let (mut colours, opaque): (Vec<_>, Vec<_>) = order.into_iter().partition(|c| c[3] < 255);
    colours.extend(opaque);
    Some(colours)
}

/// Samples of `depth` bits (1, 2, 4 or 8) packed into bytes, most
/// significant first.
fn pack(samples: impl Iterator<Item = u8>, depth: u8) -> Vec<u8> {
    if depth == 8 {
        return samples.collect();
    }
    let per = 8 / depth as usize;
    let mut out = Vec::new();
    for (k, s) in samples.enumerate() {
        if k % per == 0 {
            out.push(0);
        }
        *out.last_mut().unwrap() |= s << (8 - depth as usize * (k % per + 1));
    }
    out
}

fn encode(width: u32, height: u32, rgba: &[u8], pixels_per_metre: Option<u32>) -> Vec<u8> {
    assert_eq!(rgba.len(), (width as usize) * (height as usize) * 4, "rgba size mismatch");
    let write = |l: &Layout| {
        // libpng's advice: palette and packed images compress best unfiltered
        let adaptive = l.colour != 3 && l.depth >= 8;
        write_png(width, height, l.depth, l.colour, &l.palette, &l.trns, &l.rows, l.bpp, adaptive, pixels_per_metre)
    };
    let l = layout(width, height, rgba, true);
    let png = write(&l);
    if l.colour != 3 {
        return png;
    }
    // A palette is not always smaller: the indices of a smooth gradient go
    // unfiltered, where its colours filter to almost nothing (580 bytes as
    // RGB against 1341 as a palette, for a 37x23 swirl). So both, and the
    // smaller; only images of 256 colours or fewer pay for the second try.
    let truecolour = write(&layout(width, height, rgba, false));
    if truecolour.len() < png.len() { truecolour } else { png }
}

#[allow(clippy::too_many_arguments)]
fn write_png(
    width: u32,
    height: u32,
    depth: u8,
    colour: u8,
    palette: &[u8],
    trns: &[u8],
    rows: &[Vec<u8>],
    bpp: usize,
    adaptive: bool,
    pixels_per_metre: Option<u32>,
) -> Vec<u8> {
    let raw = if adaptive { filtered(rows, bpp) } else { rows.iter().flat_map(|r| std::iter::once(0).chain(r.iter().copied())).collect() };

    let mut png = Vec::new();
    png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[depth, colour, 0, 0, 0]); // deflate, adaptive filtering, no interlace
    chunk(&mut png, b"IHDR", &ihdr);
    if let Some(ppm) = pixels_per_metre {
        let mut phys = Vec::with_capacity(9);
        phys.extend_from_slice(&ppm.to_be_bytes());
        phys.extend_from_slice(&ppm.to_be_bytes());
        phys.push(1); // the unit: the metre
        chunk(&mut png, b"pHYs", &phys);
    }
    if !palette.is_empty() {
        chunk(&mut png, b"PLTE", palette);
    }
    if !trns.is_empty() {
        chunk(&mut png, b"tRNS", trns);
    }
    chunk(&mut png, b"IDAT", &zlib(&raw));
    chunk(&mut png, b"IEND", &[]);
    png
}

/// The rows filtered, each after its filter type: the filter libpng's
/// heuristic picks, the one whose bytes, read as signed, sum smallest in
/// magnitude.
///
/// Measured against Sub on every row (the previous encoder's usual pick),
/// it wins on renders and photographs; on synthetic noise it can lose,
/// as libpng's does. An entropy estimate to choose between the two per
/// image picked wrong as often as right, so it is not here.
fn filtered(rows: &[Vec<u8>], bpp: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows.iter().map(|r| r.len() + 1).sum());
    let empty = Vec::new();
    for (y, row) in rows.iter().enumerate() {
        let prev = if y > 0 { &rows[y - 1] } else { &empty };
        let mut score = [0u64; 5];
        for x in 0..row.len() {
            let (a, b, c) = neighbours(row, prev, bpp, x);
            let v = row[x];
            for (f, p) in [0, a, b, ((a as u16 + b as u16) / 2) as u8, paeth(a, b, c)].into_iter().enumerate() {
                score[f] += (v.wrapping_sub(p) as i8).unsigned_abs() as u64;
            }
        }
        let best = (0..5).min_by_key(|&f| score[f]).unwrap();
        out.push(best as u8);
        for x in 0..row.len() {
            let (a, b, c) = neighbours(row, prev, bpp, x);
            let p = match best {
                0 => 0,
                1 => a,
                2 => b,
                3 => ((a as u16 + b as u16) / 2) as u8,
                _ => paeth(a, b, c),
            };
            out.push(row[x].wrapping_sub(p));
        }
    }
    out
}

/// The bytes left, above and above-left of byte `x`, zero past the edges.
fn neighbours(row: &[u8], prev: &[u8], bpp: usize, x: usize) -> (u8, u8, u8) {
    let a = if x >= bpp { row[x - bpp] } else { 0 };
    let b = prev.get(x).copied().unwrap_or(0);
    let c = if x >= bpp { prev.get(x - bpp).copied().unwrap_or(0) } else { 0 };
    (a, b, c)
}

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let (pa, pb, pc) = (a as i16, b as i16, c as i16);
    let p = pa + pb - pc;
    let (da, db, dc) = ((p - pa).abs(), (p - pb).abs(), (p - pc).abs());
    if da <= db && da <= dc {
        a
    } else if db <= dc {
        b
    } else {
        c
    }
}

/// A decoded image at 16 bits a channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PngImage16 {
    pub width: u32,
    pub height: u32,
    /// `width * height * 4` samples, row-major from the top-left; a narrower
    /// file's samples scaled to the full range (`v * 257` for 8 bits).
    pub rgba: Vec<u16>,
}

/// Decode a PNG into 8-bit RGBA; 16-bit samples keep their high byte.
pub fn decode_png(bytes: &[u8]) -> Result<PngImage, String> {
    let img = decode_png16(bytes)?;
    Ok(PngImage { width: img.width, height: img.height, rgba: img.rgba.iter().map(|&v| (v >> 8) as u8).collect() })
}

/// Decode a PNG into 16-bit RGBA, keeping a 16-bit file's precision.
///
/// Every colour type and bit depth the specification allows, interlaced
/// (Adam7) or not, with `tRNS` transparency for palettes and for the one
/// grey or RGB value it names.
pub fn decode_png16(bytes: &[u8]) -> Result<PngImage16, String> {
    if bytes.len() < 8 || bytes[..8] != [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A] {
        return Err("not a PNG".into());
    }
    let (mut w, mut h, mut depth, mut colour, mut interlace) = (0u32, 0u32, 0u8, 0u8, 0u8);
    let mut idat: Vec<u8> = Vec::new();
    let mut palette: Vec<[u8; 3]> = Vec::new();
    let mut trns: Vec<u8> = Vec::new();
    let mut p = 8usize;
    while p + 8 <= bytes.len() {
        let len = u32::from_be_bytes([bytes[p], bytes[p + 1], bytes[p + 2], bytes[p + 3]]) as usize;
        let kind = &bytes[p + 4..p + 8];
        let data = bytes.get(p + 8..p + 8 + len).ok_or("PNG: truncated chunk")?;
        match kind {
            b"IHDR" => {
                if data.len() < 13 {
                    return Err("PNG: short IHDR".into());
                }
                w = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
                h = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
                depth = data[8];
                colour = data[9];
                interlace = data[12];
            }
            b"PLTE" => palette = data.as_chunks::<3>().0.iter().map(|c| [c[0], c[1], c[2]]).collect(),
            b"tRNS" => trns = data.to_vec(),
            b"IDAT" => idat.extend_from_slice(data),
            b"IEND" => break,
            _ => {}
        }
        p += 12 + len;
    }
    let channels = match colour {
        0 => 1, // grey
        2 => 3, // rgb
        3 => 1, // palette index
        4 => 2, // grey + alpha
        6 => 4, // rgba
        _ => return Err(format!("PNG: unsupported colour type {colour}")),
    };
    // the specification's table of allowed depths
    let allowed = match colour {
        0 => matches!(depth, 1 | 2 | 4 | 8 | 16),
        3 => matches!(depth, 1 | 2 | 4 | 8),
        _ => matches!(depth, 8 | 16),
    };
    if !allowed {
        return Err(format!("PNG: unsupported bit depth {depth} for colour type {colour}"));
    }
    if interlace > 1 {
        return Err(format!("PNG: unknown interlace method {interlace}"));
    }
    if idat.len() < 2 {
        return Err("PNG: no image data".into());
    }
    let raw = inflate(&idat[2..])?; // skip the 2-byte zlib header
    let bits_per_px = channels * depth as usize;
    // The filters look one whole pixel back, or one byte for packed pixels.
    let bpp = (bits_per_px / 8).max(1);
    let (wu, hu) = (w as usize, h as usize);

    // tRNS for grey and RGB: the one sample value that is transparent
    let key: Option<[u32; 3]> = match (colour, trns.len()) {
        (0, 2..) => Some([u16::from_be_bytes([trns[0], trns[1]]) as u32; 3]),
        (2, 6..) => Some([
            u16::from_be_bytes([trns[0], trns[1]]) as u32,
            u16::from_be_bytes([trns[2], trns[3]]) as u32,
            u16::from_be_bytes([trns[4], trns[5]]) as u32,
        ]),
        _ => None,
    };
    let max = (1u32 << depth) - 1;
    // a sample widened to 16 bits: 1, 2, 4 and 8-bit levels land exactly
    let widen = |v: u32| (v * 65535 / max) as u16;

    let mut rgba = vec![u16::MAX; wu * hu * 4];
    // (x0, y0, dx, dy) per pass: Adam7's seven, or the whole image once
    let passes: &[(usize, usize, usize, usize)] = if interlace == 1 {
        &[(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
    } else {
        &[(0, 0, 1, 1)]
    };
    let mut at = 0usize;
    for &(x0, y0, dx, dy) in passes {
        let pw = if wu > x0 { (wu - x0).div_ceil(dx) } else { 0 };
        let ph = if hu > y0 { (hu - y0).div_ceil(dy) } else { 0 };
        if pw == 0 || ph == 0 {
            continue;
        }
        let stride = (pw * bits_per_px).div_ceil(8);
        let need = (stride + 1) * ph;
        let src = raw.get(at..at + need).ok_or("PNG: short image data")?;
        at += need;
        let lines = unfilter(src, stride, ph, bpp)?;
        // a sample: packed most significant first, or one or two bytes
        let sample = |line: &[u8], k: usize| -> u32 {
            match depth {
                16 => u16::from_be_bytes([line[2 * k], line[2 * k + 1]]) as u32,
                8 => line[k] as u32,
                d => {
                    let bit = k * d as usize;
                    ((line[bit / 8] >> (8 - d as usize - bit % 8)) as u32) & max
                }
            }
        };
        for py in 0..ph {
            let line = &lines[py * stride..(py + 1) * stride];
            let y = y0 + py * dy;
            for px in 0..pw {
                let o = (y * wu + x0 + px * dx) * 4;
                let s = px * channels;
                let out = &mut rgba[o..o + 4];
                match colour {
                    0 => {
                        let g = sample(line, s);
                        out[..3].fill(widen(g));
                        if key.is_some_and(|k| k[0] == g) {
                            out[3] = 0;
                        }
                    }
                    2 => {
                        let c = [sample(line, s), sample(line, s + 1), sample(line, s + 2)];
                        for (o, v) in out.iter_mut().zip(c) {
                            *o = widen(v);
                        }
                        if key == Some(c) {
                            out[3] = 0;
                        }
                    }
                    3 => {
                        let idx = sample(line, s) as usize;
                        let c = palette.get(idx).copied().unwrap_or([0, 0, 0]);
                        for k in 0..3 {
                            out[k] = c[k] as u16 * 257;
                        }
                        out[3] = trns.get(idx).copied().unwrap_or(255) as u16 * 257;
                    }
                    4 => {
                        out[..3].fill(widen(sample(line, s)));
                        out[3] = widen(sample(line, s + 1));
                    }
                    _ => {
                        for (k, o) in out.iter_mut().enumerate() {
                            *o = widen(sample(line, s + k));
                        }
                    }
                }
            }
        }
    }
    Ok(PngImage16 { width: w, height: h, rgba })
}

/// `rows` scanlines of `stride` bytes, each after its filter type byte,
/// with the filters undone.
fn unfilter(src: &[u8], stride: usize, rows: usize, bpp: usize) -> Result<Vec<u8>, String> {
    let mut lines: Vec<u8> = vec![0; stride * rows];
    for y in 0..rows {
        let ft = src[y * (stride + 1)];
        let row = &src[y * (stride + 1) + 1..y * (stride + 1) + 1 + stride];
        for x in 0..stride {
            let a = if x >= bpp { lines[y * stride + x - bpp] } else { 0 };
            let b = if y > 0 { lines[(y - 1) * stride + x] } else { 0 };
            let c = if x >= bpp && y > 0 { lines[(y - 1) * stride + x - bpp] } else { 0 };
            lines[y * stride + x] = match ft {
                0 => row[x],
                1 => row[x].wrapping_add(a),
                2 => row[x].wrapping_add(b),
                3 => row[x].wrapping_add(((a as u16 + b as u16) / 2) as u8),
                4 => row[x].wrapping_add(paeth(a, b, c)),
                _ => return Err(format!("PNG: bad filter type {ft}")),
            };
        }
    }
    Ok(lines)
}

#[cfg(test)]
mod tests {

    // 9x7 Adam7-interlaced PNGs and ImageMagick 7's own decode of each, as
    // RGBA: `magick -size 9x7 gradient:red-blue -swirl 120 -strip
    // -define png:color-type=2 -depth 8 -interlace PNG`, then that converted to
    // 2-bit grey (`-colorspace gray -depth 2 -define png:bit-depth=2
    // -define png:color-type=0`), and `magick f.png -depth 8 rgba:f.rgba`.
    const FX_RGB_IL: [u8; 221] = [
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 9, 0, 0, 0, 7,
        8, 2, 0, 0, 1, 34, 254, 192, 161, 0, 0, 0, 164, 73, 68, 65, 84, 8, 215, 53, 203, 177, 74, 66,
        97, 24, 128, 225, 231, 247, 252, 21, 168, 5, 97, 132, 83, 152, 67, 139, 53, 119, 21, 45, 193, 185, 148, 110,
        162, 75, 81, 112, 233, 38, 218, 203, 165, 65, 163, 73, 34, 9, 202, 19, 100, 231, 244, 181, 216, 250, 242, 188,
        41, 128, 32, 149, 198, 67, 243, 7, 23, 105, 27, 111, 220, 94, 186, 79, 99, 229, 129, 143, 119, 135, 75, 253,
        190, 37, 162, 165, 201, 126, 8, 226, 95, 131, 60, 117, 221, 181, 126, 114, 182, 231, 59, 223, 185, 26, 154, 63,
        58, 127, 113, 130, 216, 177, 217, 62, 51, 35, 36, 191, 133, 166, 81, 108, 236, 134, 148, 213, 45, 145, 39, 202,
        182, 170, 103, 149, 196, 179, 193, 194, 233, 151, 118, 215, 250, 216, 107, 26, 153, 21, 234, 125, 159, 29, 85, 45,
        191, 57, 90, 233, 37, 209, 81, 253, 1, 44, 52, 57, 1, 88, 112, 120, 155, 0, 0, 0, 0, 73, 69, 78,
        68, 174, 66, 96, 130,
    ];
    const FX_RGB_IL_RGBA: [u8; 252] = [
        255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255,
        255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 213, 0, 43, 255, 213, 0, 43, 255, 214, 0, 41, 255,
        217, 0, 38, 255, 214, 0, 41, 255, 208, 0, 47, 255, 207, 0, 48, 255, 211, 0, 44, 255, 213, 0, 43, 255,
        170, 0, 85, 255, 173, 0, 82, 255, 184, 0, 71, 255, 185, 0, 70, 255, 167, 0, 88, 255, 147, 0, 108, 255,
        143, 0, 112, 255, 154, 0, 101, 255, 167, 0, 88, 255, 128, 0, 128, 255, 138, 0, 117, 255, 157, 0, 98, 255,
        158, 0, 97, 255, 126, 0, 129, 255, 93, 0, 162, 255, 83, 0, 172, 255, 95, 0, 160, 255, 117, 0, 138, 255,
        85, 0, 170, 255, 96, 0, 159, 255, 117, 0, 138, 255, 130, 0, 125, 255, 119, 0, 136, 255, 86, 0, 169, 255,
        55, 0, 200, 255, 56, 0, 199, 255, 74, 0, 181, 255, 43, 0, 213, 255, 46, 0, 209, 255, 59, 0, 196, 255,
        70, 0, 185, 255, 66, 0, 189, 255, 46, 0, 209, 255, 27, 0, 228, 255, 28, 0, 227, 255, 39, 0, 216, 255,
        0, 0, 255, 255, 0, 0, 255, 255, 2, 0, 253, 255, 5, 0, 250, 255, 4, 0, 251, 255, 0, 0, 255, 255,
        0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255,
    ];

    const FX_G2_IL: [u8; 87] = [
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 9, 0, 0, 0, 7,
        2, 0, 0, 0, 1, 194, 71, 16, 139, 0, 0, 0, 30, 73, 68, 65, 84, 8, 215, 99, 8, 96, 112, 96,
        96, 96, 8, 0, 98, 1, 6, 16, 8, 5, 179, 129, 116, 168, 3, 3, 20, 0, 0, 50, 204, 2, 128, 98,
        150, 141, 20, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
    ];
    const FX_G2_IL_RGBA: [u8; 252] = [
        85, 85, 85, 255, 85, 85, 85, 255, 85, 85, 85, 255, 85, 85, 85, 255, 85, 85, 85, 255, 85, 85, 85, 255,
        85, 85, 85, 255, 85, 85, 85, 255, 85, 85, 85, 255, 85, 85, 85, 255, 85, 85, 85, 255, 85, 85, 85, 255,
        85, 85, 85, 255, 85, 85, 85, 255, 85, 85, 85, 255, 85, 85, 85, 255, 85, 85, 85, 255, 85, 85, 85, 255,
        0, 0, 0, 255, 85, 85, 85, 255, 85, 85, 85, 255, 85, 85, 85, 255, 0, 0, 0, 255, 0, 0, 0, 255,
        0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255,
        0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255,
        0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255,
        0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255,
        0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255,
        0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255,
        0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255,
    ];

    #[test]
    fn interlaced_images_decode_as_imagemagick_decodes_them() {
        for (png, want) in [(&FX_RGB_IL[..], &FX_RGB_IL_RGBA[..]), (&FX_G2_IL[..], &FX_G2_IL_RGBA[..])] {
            assert_eq!(png[28], 1, "the fixture is interlaced");
            let img = decode_png(png).unwrap();
            assert_eq!((img.width, img.height), (9, 7));
            assert_eq!(img.rgba, want);
        }
    }

    /// The colour type and bit depth an encoded PNG chose.
    fn header(png: &[u8]) -> (u8, u8) {
        (png[25], png[24])
    }

    #[test]
    fn each_image_takes_the_smallest_exact_layout() {
        let (w, h) = (16u32, 4u32);
        let px = |f: &dyn Fn(u32) -> [u8; 4]| (0..w * h).flat_map(f).collect::<Vec<u8>>();
        let cases: Vec<(&str, Vec<u8>, (u8, u8))> = vec![
            ("black and white", px(&|i| if i % 3 == 0 { [255; 4] } else { [0, 0, 0, 255] }), (0, 1)),
            ("four greys", px(&|i| { let g = [0, 85, 170, 255][(i % 4) as usize]; [g, g, g, 255] }), (0, 2)),
            ("sixteen greys", px(&|i| { let g = (i % 16) as u8 * 17; [g, g, g, 255] }), (0, 4)),
            ("grey", px(&|i| { let g = (i * 7) as u8 + 1; [g, g, g, 255] }), (0, 8)),
        ];
        for (name, rgba, want) in cases {
            let png = encode_png(w, h, &rgba);
            assert_eq!(header(&png), want, "{name}");
            assert_eq!(decode_png(&png).unwrap().rgba, rgba, "{name} round trip");
        }
        // two colours, one clear, scattered over enough pixels that the
        // palette's own chunks pay for themselves: 1-bit indices and tRNS
        let (pw, ph) = (64u32, 32u32);
        let two: Vec<u8> = (0..pw * ph).flat_map(|i| if (i * i / 3 + i / 5) % 7 < 3 { [1, 2, 3, 0] } else { [200, 0, 0, 255] }).collect();
        let png = encode_png(pw, ph, &two);
        assert_eq!(header(&png), (3, 1));
        assert!(png.windows(4).any(|c| c == b"tRNS"));
        assert_eq!(decode_png(&png).unwrap().rgba, two);
        // 64 colours: a palette or grey and alpha, whichever deflates smaller
        let few = px(&|i| [9, 9, 9, (i * 3) as u8]);
        let png = encode_png(w, h, &few);
        assert!(matches!(header(&png), (3, 8) | (4, 8)), "{:?}", header(&png));
        assert_eq!(decode_png(&png).unwrap().rgba, few);
        // more than 256 colours: RGB when opaque, RGBA when not, grey and alpha when grey
        let (w, h) = (32u32, 16u32);
        let many: Vec<u8> = (0..w * h).flat_map(|i| [(i % 251) as u8, (i / 7) as u8, (i % 13) as u8, 255]).collect();
        let png = encode_png(w, h, &many);
        assert_eq!(header(&png), (2, 8));
        assert_eq!(decode_png(&png).unwrap().rgba, many);
        let clear: Vec<u8> = many.chunks(4).enumerate().flat_map(|(k, p)| [p[0], p[1], p[2], (k % 200) as u8]).collect();
        let png = encode_png(w, h, &clear);
        assert_eq!(header(&png), (6, 8));
        assert_eq!(decode_png(&png).unwrap().rgba, clear);
        let grey_alpha: Vec<u8> = (0..w * h).flat_map(|i| { let g = (i % 256) as u8; [g, g, g, (i / 2 % 256) as u8] }).collect();
        let png = encode_png(w, h, &grey_alpha);
        assert_eq!(header(&png), (4, 8));
        assert_eq!(decode_png(&png).unwrap().rgba, grey_alpha);
    }

    #[test]
    fn sixteen_bits_survive_and_narrow_when_they_can() {
        let rgba16: Vec<u16> = (0..24u32).flat_map(|i| [(i * 2741) as u16, 1000, (i * 97) as u16, if i % 2 == 0 { 65535 } else { 1234 }]).collect();
        let png = encode_png16(6, 4, &rgba16);
        assert_eq!(header(&png), (6, 16));
        assert_eq!(decode_png16(&png).unwrap().rgba, rgba16);
        assert_eq!(decode_png(&png).unwrap().rgba, rgba16.iter().map(|v| (v >> 8) as u8).collect::<Vec<_>>());
        // 8-bit values widened go out as 8-bit
        let wide: Vec<u16> = [10u16, 20, 30, 255].iter().map(|v| v * 257).collect();
        assert_eq!(header(&encode_png16(1, 1, &wide)).1, 8);
    }

    #[test]
    fn a_grey_or_rgb_colour_key_is_transparent() {
        // 2x1 RGB, 8-bit, tRNS naming (1, 2, 3)
        let mut png = Vec::new();
        png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        chunk(&mut png, b"IHDR", &[0, 0, 0, 2, 0, 0, 0, 1, 8, 2, 0, 0, 0]);
        chunk(&mut png, b"tRNS", &[0, 1, 0, 2, 0, 3]);
        chunk(&mut png, b"IDAT", &zlib(&[0, 1, 2, 3, 4, 5, 6]));
        chunk(&mut png, b"IEND", &[]);
        assert_eq!(decode_png(&png).unwrap().rgba, vec![1, 2, 3, 0, 4, 5, 6, 255]);
    }

    #[test]
    fn a_png_says_its_dpi() {
        let png = encode_png_with_dpi(2, 1, &[255, 0, 0, 255, 0, 0, 255, 255], 300.0);
        let at = png.windows(4).position(|w| w == b"pHYs").expect("a pHYs chunk");
        let ppm = u32::from_be_bytes(png[at + 4..at + 8].try_into().unwrap());
        assert_eq!(ppm, (300.0f64 / 0.0254).round() as u32);
        assert_eq!(png[at + 12], 1, "per metre");
        let back = decode_png(&png).unwrap();
        assert_eq!((back.width, back.height, &back.rgba[..4]), (2, 1, &[255, 0, 0, 255][..]));
    }

    use super::*;

    /// A palette image packed `depth` bits a pixel, as OpenStreetMap serves
    /// its sea tiles: each row a filter byte and the packed indices.
    fn packed_palette_png(depth: u8, w: u32, indices: &[u8], palette: &[[u8; 3]]) -> Vec<u8> {
        let h = indices.len() as u32 / w;
        let stride = (w as usize * depth as usize).div_ceil(8);
        let mut raw = Vec::new();
        for row in indices.chunks(w as usize) {
            raw.push(0); // no filter
            let mut bytes = vec![0u8; stride];
            for (x, &v) in row.iter().enumerate() {
                let bit = x * depth as usize;
                bytes[bit / 8] |= v << (8 - depth as usize - bit % 8);
            }
            raw.extend(bytes);
        }
        let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&w.to_be_bytes());
        ihdr.extend_from_slice(&h.to_be_bytes());
        ihdr.extend_from_slice(&[depth, 3, 0, 0, 0]);
        chunk(&mut out, b"IHDR", &ihdr);
        chunk(&mut out, b"PLTE", &palette.concat());
        chunk(&mut out, b"IDAT", &zlib(&raw));
        chunk(&mut out, b"IEND", &[]);
        out
    }

    #[test]
    fn packed_palette_images_decode() {
        let pal = [[10, 20, 30], [200, 100, 50], [0, 255, 0], [9, 9, 9]];
        for (depth, colours) in [(1u8, 2u8), (2, 4), (4, 4)] {
            // Five pixels a row: the last byte of each row is part padding.
            let idx: Vec<u8> = (0..15).map(|i| (i % colours as u32) as u8).collect();
            let img = decode_png(&packed_palette_png(
                depth,
                5,
                &idx,
                &pal[..colours as usize],
            ))
            .unwrap_or_else(|e| panic!("{depth}-bit: {e}"));
            assert_eq!((img.width, img.height), (5, 3));
            for (i, &k) in idx.iter().enumerate() {
                let c = pal[k as usize];
                assert_eq!(
                    &img.rgba[i * 4..i * 4 + 4],
                    &[c[0], c[1], c[2], 255],
                    "{depth}-bit pixel {i}"
                );
            }
        }
    }

    fn sample(w: u32, h: u32) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                // flat runs plus a gradient, so both literals and matches appear
                let band = if (x / 7 + y / 5) % 3 == 0 { 200 } else { 32 };
                v.extend_from_slice(&[band, (x % 256) as u8, (y % 256) as u8, 255]);
            }
        }
        v
    }

    #[test]
    fn deflate_inflate_roundtrip() {
        for data in [
            vec![],
            vec![7u8; 1],
            vec![0u8; 5000],
            (0..40_000u32).map(|i| (i % 251) as u8).collect::<Vec<u8>>(),
            sample(64, 64),
        ] {
            let packed = deflate(&data);
            let back = inflate(&packed).expect("inflate");
            assert_eq!(back, data, "roundtrip failed for {} bytes", data.len());
        }
    }

    #[test]
    fn png_roundtrip() {
        let (w, h) = (57u32, 31u32);
        let rgba = sample(w, h);
        let png = encode_png(w, h, &rgba);
        let img = decode_png(&png).expect("decode");
        assert_eq!((img.width, img.height), (w, h));
        assert_eq!(img.rgba, rgba);
    }

    #[test]
    fn png_compresses_flat_art() {
        let (w, h) = (256u32, 256u32);
        let rgba = vec![255u8; (w * h * 4) as usize];
        let png = encode_png(w, h, &rgba);
        assert!(
            png.len() < (w * h * 4) as usize / 20,
            "flat white took {} bytes",
            png.len()
        );
        assert_eq!(decode_png(&png).unwrap().rgba, rgba);
    }

    #[test]
    fn rejects_non_png() {
        assert!(decode_png(b"not a png at all").is_err());
    }
}
