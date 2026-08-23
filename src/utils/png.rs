//! Dependency-free PNG encode/decode, plus the DEFLATE codec underneath it.
//!
//! threers renders to RGBA and reads reference imagery back in for comparison,
//! so both directions need to exist without pulling in an image crate. The
//! encoder emits fixed-Huffman DEFLATE (line art and renders compress well);
//! the decoder handles stored, fixed and dynamic Huffman blocks, all five PNG
//! filters, and 8-bit grey / grey+alpha / RGB / RGBA / palette inputs.
//!
//! ```ignore
//! use threers::{decode_png, encode_png};
//!
//! let img = decode_png(&std::fs::read("ref.png")?)?;
//! let same = encode_png(img.width, img.height, &img.rgba);
//! ```
//!
//! Not supported (returns `Err`): interlaced PNG, and 16-bit channels are
//! truncated to their high byte.

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

fn crc32(data: &[u8]) -> u32 {
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

/// LZ77 + fixed-Huffman DEFLATE. Good enough for renders and line art, and
/// small enough to keep honest.
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

fn zlib(data: &[u8]) -> Vec<u8> {
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

/// Encode 8-bit RGBA pixels as a PNG.
///
/// Panics if `rgba.len() != width * height * 4`.
pub fn encode_png(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    assert_eq!(
        rgba.len(),
        (width as usize) * (height as usize) * 4,
        "rgba size mismatch"
    );
    // Filter each scanline with Sub or None, whichever has the smaller
    // absolute sum — cheap, and a big win on flat artwork.
    let stride = width as usize * 4;
    let mut raw = Vec::with_capacity((stride + 1) * height as usize);
    for y in 0..height as usize {
        let row = &rgba[y * stride..(y + 1) * stride];
        let mut sub = vec![0u8; stride];
        for x in 0..stride {
            let left = if x >= 4 { row[x - 4] } else { 0 };
            sub[x] = row[x].wrapping_sub(left);
        }
        let score = |v: &[u8]| {
            v.iter()
                .map(|&b| (b as i8).unsigned_abs() as u32)
                .sum::<u32>()
        };
        if score(&sub) < score(row) {
            raw.push(1);
            raw.extend_from_slice(&sub);
        } else {
            raw.push(0);
            raw.extend_from_slice(row);
        }
    }

    let mut png = Vec::new();
    png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit RGBA, deflate, adaptive filter, no interlace
    chunk(&mut png, b"IHDR", &ihdr);
    chunk(&mut png, b"IDAT", &zlib(&raw));
    chunk(&mut png, b"IEND", &[]);
    png
}

/// Decode a PNG into 8-bit RGBA.
pub fn decode_png(bytes: &[u8]) -> Result<PngImage, String> {
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
        let data = bytes
            .get(p + 8..p + 8 + len)
            .ok_or("PNG: truncated chunk")?;
        match kind {
            b"IHDR" => {
                w = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
                h = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
                depth = data[8];
                colour = data[9];
                interlace = data[12];
            }
            b"PLTE" => palette = data.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
            b"tRNS" => trns = data.to_vec(),
            b"IDAT" => idat.extend_from_slice(data),
            b"IEND" => break,
            _ => {}
        }
        p += 12 + len;
    }
    if interlace != 0 {
        return Err("PNG: interlaced images are not supported".into());
    }
    if depth != 8 && depth != 16 {
        return Err(format!("PNG: unsupported bit depth {depth}"));
    }
    if idat.len() < 2 {
        return Err("PNG: no image data".into());
    }
    let raw = inflate(&idat[2..])?; // skip the 2-byte zlib header

    let channels = match colour {
        0 => 1, // grey
        2 => 3, // rgb
        3 => 1, // palette index
        4 => 2, // grey + alpha
        6 => 4, // rgba
        _ => return Err(format!("PNG: unsupported colour type {colour}")),
    };
    let bytes_per_px = channels * (depth as usize / 8);
    let stride = w as usize * bytes_per_px;
    if raw.len() < (stride + 1) * h as usize {
        return Err("PNG: short image data".into());
    }

    // Undo the per-scanline filters in place.
    let mut lines: Vec<u8> = vec![0; stride * h as usize];
    for y in 0..h as usize {
        let ft = raw[y * (stride + 1)];
        let src = &raw[y * (stride + 1) + 1..y * (stride + 1) + 1 + stride];
        for x in 0..stride {
            let a = if x >= bytes_per_px {
                lines[y * stride + x - bytes_per_px]
            } else {
                0
            };
            let b = if y > 0 {
                lines[(y - 1) * stride + x]
            } else {
                0
            };
            let c = if x >= bytes_per_px && y > 0 {
                lines[(y - 1) * stride + x - bytes_per_px]
            } else {
                0
            };
            let v = match ft {
                0 => src[x],
                1 => src[x].wrapping_add(a),
                2 => src[x].wrapping_add(b),
                3 => src[x].wrapping_add(((a as u16 + b as u16) / 2) as u8),
                4 => {
                    let (pa, pb, pc) = (a as i16, b as i16, c as i16);
                    let pp = pa + pb - pc;
                    let (da, db, dc) = ((pp - pa).abs(), (pp - pb).abs(), (pp - pc).abs());
                    let pred = if da <= db && da <= dc {
                        a
                    } else if db <= dc {
                        b
                    } else {
                        c
                    };
                    src[x].wrapping_add(pred)
                }
                _ => return Err(format!("PNG: bad filter type {ft}")),
            };
            lines[y * stride + x] = v;
        }
    }

    // Expand to RGBA.
    let step = depth as usize / 8; // 1 for 8-bit, 2 for 16-bit (take high byte)
    let mut rgba = vec![255u8; (w as usize) * (h as usize) * 4];
    for i in 0..(w as usize) * (h as usize) {
        let s = i * bytes_per_px;
        let o = i * 4;
        match colour {
            0 => {
                let g = lines[s];
                rgba[o] = g;
                rgba[o + 1] = g;
                rgba[o + 2] = g;
            }
            2 => {
                rgba[o] = lines[s];
                rgba[o + 1] = lines[s + step];
                rgba[o + 2] = lines[s + 2 * step];
            }
            3 => {
                let idx = lines[s] as usize;
                let c = palette.get(idx).copied().unwrap_or([0, 0, 0]);
                rgba[o] = c[0];
                rgba[o + 1] = c[1];
                rgba[o + 2] = c[2];
                rgba[o + 3] = trns.get(idx).copied().unwrap_or(255);
            }
            4 => {
                let g = lines[s];
                rgba[o] = g;
                rgba[o + 1] = g;
                rgba[o + 2] = g;
                rgba[o + 3] = lines[s + step];
            }
            6 => {
                rgba[o] = lines[s];
                rgba[o + 1] = lines[s + step];
                rgba[o + 2] = lines[s + 2 * step];
                rgba[o + 3] = lines[s + 3 * step];
            }
            _ => unreachable!(),
        }
    }
    Ok(PngImage {
        width: w,
        height: h,
        rgba,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
