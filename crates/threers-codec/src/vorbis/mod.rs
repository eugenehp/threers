//! Vorbis I decoding, ported from libvorbis 1.3.7 (the decoder inside
//! libsndfile / `soundfile`): codebooks (`_book_unquantize`, the packed
//! entry search), floor 1, residues 0/1/2, mapping 0 with polar coupling,
//! `mdct_backward`, and `vorbis_synthesis_blockin`'s lapping and granule
//! trimming, driven the way libsndfile's `vorbis_read_sample` drives it.
//! Floor 0 (pre-1.0 streams) is refused.
//!
//! The reference libvorbis was built with `-ffast-math`; the float order
//! here follows its machine code, which makes the output bit-identical to
//! `soundfile.read` (libsndfile 1.2.2 / libvorbis 1.3.7, arm64).

// Ported loop structure is kept as in the C reference.
#![allow(clippy::needless_range_loop, clippy::manual_memcpy, clippy::unnecessary_cast, clippy::manual_is_multiple_of)]

#[allow(clippy::all)]
/// Window shapes and the floor-1 dB table, generated from libvorbis 1.3.7.
pub mod tables;
/// Code generated from the reference build's machine code (it was compiled
/// with `-ffast-math`, so its float order is not the C source's).
#[allow(clippy::all)]
/// Kernels in the reference build's own floating-point order (it is compiled
/// with `-ffast-math`), so output matches it bit for bit.
pub mod built;

use super::ogg;

/// The reference build's contraction of `a*b ± c*d` (read from its machine
/// code: `fmadd` fuses the second product of a sum, `fnmsub` the first of a
/// difference). Everything else that differs from C order is in [`built`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fma {
    Mixed,
}

fn mode() -> Fma {
    Fma::Mixed
}

/// `fneg` of a computed value, kept a separate instruction: LLVM would
/// otherwise fold `-fma(a, b, c)` into `fnmadd`, which differs from the
/// reference's `fmadd` + `fneg` in the sign of an exactly-zero result.
#[inline(always)]
fn neg(x: f32) -> f32 {
    -std::hint::black_box(x)
}

/// `a*b + c*d`: `fmadd(c, d, a*b)`.
#[inline]
fn madd(_: Fma, a: f32, b: f32, c: f32, d: f32) -> f32 {
    c.mul_add(d, a * b)
}

/// `a*b - c*d`: `fnmsub(a, b, c*d)`.
#[inline]
fn msub(_: Fma, a: f32, b: f32, c: f32, d: f32) -> f32 {
    a.mul_add(b, -(c * d))
}

// ---- oggpack (LSb-first) -------------------------------------------------

struct Pack<'a> {
    d: &'a [u8],
    pos: usize,
    eop: bool,
}

impl<'a> Pack<'a> {
    fn new(d: &'a [u8]) -> Self {
        Self { d, pos: 0, eop: false }
    }
    fn left(&self) -> usize {
        if self.eop {
            0
        } else {
            self.d.len() * 8 - self.pos.min(self.d.len() * 8)
        }
    }
    /// `oggpack_look`: `-1` if fewer than `n` bits remain.
    fn look(&self, n: u32) -> i64 {
        if n == 0 {
            return 0;
        }
        if (self.left() as u32) < n {
            return -1;
        }
        let mut v = 0u64;
        for i in 0..n as usize {
            let p = self.pos + i;
            v |= (((self.d[p >> 3] >> (p & 7)) & 1) as u64) << i;
        }
        v as i64
    }
    fn adv(&mut self, n: u32) {
        if self.left() < n as usize {
            self.eop = true;
            self.pos = self.d.len() * 8;
        } else {
            self.pos += n as usize;
        }
    }
    /// `oggpack_read`: `-1` (and end of packet from then on) on overflow.
    fn read(&mut self, n: u32) -> i64 {
        let v = self.look(n);
        if v < 0 {
            self.eop = true;
            self.pos = self.d.len() * 8;
            return -1;
        }
        self.pos += n as usize;
        v
    }
}

fn ilog(v: u32) -> u32 {
    32 - v.leading_zeros()
}

// ---- codebooks --------------------------------------------------------------

struct Codebook {
    dim: usize,
    entries: usize,
    used: usize,
    /// Sorted bit-reversed codewords (used entries).
    codelist: Vec<u32>,
    lengths: Vec<u32>,
    dec_index: Vec<usize>,
    maxlen: u32,
    /// Value vectors in sorted order (`used * dim`), if mapped.
    values: Vec<f32>,
    maptype: u32,
}

fn float32_unpack(val: u32) -> f32 {
    let mut mant = (val & 0x1fffff) as f64;
    let sign = val & 0x8000_0000 != 0;
    let mut exp = ((val & 0x7fe0_0000) >> 21) as i64;
    if sign {
        mant = -mant;
    }
    exp = exp - 20 - 768;
    exp = exp.clamp(-63, 63);
    (mant * 2f64.powi(exp as i32)) as f32
}

fn maptype1_quantvals(entries: usize, dim: usize) -> usize {
    if entries < 1 {
        return 0;
    }
    let mut vals = ((entries as f32).powf(1.0 / dim as f32) as f64).floor() as i64;
    if vals < 1 {
        vals = 1;
    }
    loop {
        let (mut acc, mut acc1) = (1i64, 1i64);
        let mut i = 0;
        while i < dim {
            if entries as i64 / vals < acc {
                break;
            }
            acc *= vals;
            acc1 = acc1.saturating_mul(vals + 1);
            i += 1;
        }
        if i >= dim && acc <= entries as i64 && acc1 > entries as i64 {
            return vals as usize;
        }
        if i < dim || acc > entries as i64 {
            vals -= 1;
        } else {
            vals += 1;
        }
    }
}

/// `_make_words`: codewords for a length list (MSb-first), `None` if the
/// lengths over- or under-populate the tree.
fn make_words(l: &[u32], sparse: bool) -> Option<Vec<u32>> {
    let mut marker = [0u32; 33];
    let mut r = Vec::new();
    let mut count = 0usize;
    for &len in l {
        if len > 0 {
            let len = len as usize;
            let mut entry = marker[len];
            if len < 32 && (entry >> len) != 0 {
                return None;
            }
            r.push(entry);
            count += 1;
            for j in (1..=len).rev() {
                if marker[j] & 1 != 0 {
                    if j == 1 {
                        marker[1] += 1;
                    } else {
                        marker[j] = marker[j - 1] << 1;
                    }
                    break;
                }
                marker[j] += 1;
            }
            for j in len + 1..33 {
                if (marker[j] >> 1) == entry {
                    entry = marker[j];
                    marker[j] = marker[j - 1] << 1;
                } else {
                    break;
                }
            }
        } else if !sparse {
            r.push(0);
            count += 1;
        }
    }
    if !(count == 1 && marker[2] == 2) {
        for (i, &m) in marker.iter().enumerate().skip(1) {
            if m & (u32::MAX >> (32 - i)) != 0 {
                return None;
            }
        }
    }
    Some(r)
}

fn unpack_book(p: &mut Pack, fma: Fma) -> Result<Codebook, String> {
    if p.read(24) != 0x564342 {
        return Err("bad codebook sync".into());
    }
    let dim = p.read(16);
    let entries = p.read(24);
    if dim < 0 || entries < 0 {
        return Err("truncated codebook".into());
    }
    let (dim, entries) = (dim as usize, entries as usize);
    if ilog(dim as u32) + ilog(entries as u32) > 24 {
        return Err("codebook too large".into());
    }
    let mut lengths = vec![0u32; entries];
    match p.read(1) {
        0 => {
            let unused = p.read(1);
            if unused == 1 {
                for l in lengths.iter_mut() {
                    if p.read(1) == 1 {
                        let n = p.read(5);
                        if n < 0 {
                            return Err("truncated codebook".into());
                        }
                        *l = n as u32 + 1;
                    }
                }
            } else {
                for l in lengths.iter_mut() {
                    let n = p.read(5);
                    if n < 0 {
                        return Err("truncated codebook".into());
                    }
                    *l = n as u32 + 1;
                }
            }
        }
        1 => {
            let mut length = p.read(5) + 1;
            let mut i = 0;
            while i < entries {
                let num = p.read(ilog((entries - i) as u32));
                if num < 0 || length > 32 || num as usize > entries - i {
                    return Err("bad ordered codebook".into());
                }
                for _ in 0..num {
                    lengths[i] = length as u32;
                    i += 1;
                }
                length += 1;
            }
        }
        _ => return Err("truncated codebook".into()),
    }
    let maptype = p.read(4);
    let mut quant: Vec<i64> = Vec::new();
    let (mut q_min, mut q_delta, mut q_seq) = (0u32, 0u32, false);
    match maptype {
        0 => {}
        1 | 2 => {
            q_min = p.read(32) as u32;
            q_delta = p.read(32) as u32;
            let q_quant = p.read(4) + 1;
            q_seq = p.read(1) == 1;
            let n = if maptype == 1 { if dim == 0 { 0 } else { maptype1_quantvals(entries, dim) } } else { entries * dim };
            for _ in 0..n {
                quant.push(p.read(q_quant as u32));
            }
            if quant.last() == Some(&-1) {
                return Err("truncated codebook values".into());
            }
        }
        _ => return Err("bad codebook map type".into()),
    }
    // vorbis_book_init_decode
    let used = lengths.iter().filter(|&&l| l > 0).count();
    let mut cb = Codebook {
        dim,
        entries,
        used,
        codelist: Vec::new(),
        lengths: Vec::new(),
        dec_index: Vec::new(),
        maxlen: 0,
        values: Vec::new(),
        maptype: maptype as u32,
    };
    if used == 0 {
        return Ok(cb);
    }
    let codes = make_words(&lengths, true).ok_or("bad codebook lengths")?;
    // libvorbis stores bitreverse(LSb-first word) = the MSb-first codeword
    // left-aligned in 32 bits.
    let left: Vec<u32> = codes
        .iter()
        .zip(lengths.iter().filter(|&&l| l > 0))
        .map(|(&c, &l)| if l == 32 { c } else { c << (32 - l) })
        .collect();
    let mut order: Vec<usize> = (0..used).collect();
    order.sort_by_key(|&i| left[i]); // qsort(sort32a); codewords are unique
    let mut sortindex = vec![0usize; used];
    for (pos, &i) in order.iter().enumerate() {
        sortindex[i] = pos;
    }
    cb.codelist = vec![0; used];
    cb.lengths = vec![0; used];
    cb.dec_index = vec![0; used];
    let mut n = 0;
    for (e, &l) in lengths.iter().enumerate() {
        if l > 0 {
            cb.codelist[sortindex[n]] = left[n];
            cb.lengths[sortindex[n]] = l;
            cb.dec_index[sortindex[n]] = e;
            cb.maxlen = cb.maxlen.max(l);
            n += 1;
        }
    }
    // _book_unquantize (sparse)
    if maptype == 1 || maptype == 2 {
        let mindel = float32_unpack(q_min);
        let delta = float32_unpack(q_delta);
        cb.values = vec![0f32; used * dim];
        let quantvals = if maptype == 1 { maptype1_quantvals(entries, dim) } else { 0 };
        let mut count = 0;
        for (j, &l) in lengths.iter().enumerate() {
            if l == 0 {
                continue;
            }
            let mut last = 0f32;
            let mut indexdiv = 1usize;
            for k in 0..dim {
                let q = if maptype == 1 {
                    let index = (j / indexdiv) % quantvals;
                    indexdiv *= quantvals;
                    quant[index]
                } else {
                    quant[j * dim + k]
                };
                let v = (q as f32).abs();
                let _ = fma;
                let val = v.mul_add(delta, mindel) + last;
                if q_seq {
                    last = val;
                }
                cb.values[sortindex[count] * dim + k] = val;
            }
            count += 1;
        }
    }
    Ok(cb)
}

impl Codebook {
    /// `decode_packed_entry_number`: the sorted index, or `None` at EOP.
    fn decode_packed(&self, p: &mut Pack) -> Option<usize> {
        if self.used == 0 {
            return None;
        }
        let mut read = self.maxlen;
        let mut lok = p.look(read);
        while lok < 0 && read > 1 {
            read -= 1;
            lok = p.look(read);
        }
        if lok < 0 {
            return None;
        }
        let test = (lok as u32).reverse_bits();
        // Largest codeword <= test.
        let (mut lo, mut hi) = (0usize, self.used);
        while hi - lo > 1 {
            let m = (hi - lo) >> 1;
            if self.codelist[lo + m] > test {
                hi -= m;
            } else {
                lo += m;
            }
        }
        if self.lengths[lo] <= read {
            p.adv(self.lengths[lo]);
            return Some(lo);
        }
        p.adv(read);
        None
    }

    fn decode(&self, p: &mut Pack) -> i64 {
        match self.decode_packed(p) {
            Some(e) => self.dec_index[e] as i64,
            None => -1,
        }
    }

    fn vec(&self, e: usize) -> &[f32] {
        &self.values[e * self.dim..(e + 1) * self.dim]
    }
}

// ---- setup header -----------------------------------------------------------

struct Floor1 {
    partition_class: Vec<usize>,
    class_dim: Vec<usize>,
    class_subs: Vec<u32>,
    class_book: Vec<usize>,
    class_subbook: Vec<Vec<i64>>,
    mult: i32,
    postlist: Vec<i32>,
    // look
    forward_index: Vec<usize>,
    loneighbor: Vec<usize>,
    hineighbor: Vec<usize>,
    quant_q: i32,
    posts: usize,
}

struct Residue {
    kind: u32,
    begin: i64,
    end: i64,
    grouping: i64,
    groupbook: usize,
    secondstages: Vec<u32>,
    /// `partbooks[part][stage]`
    partbooks: Vec<Vec<Option<usize>>>,
    stages: usize,
    partvals: usize,
    decodemap: Vec<Vec<usize>>,
}

struct Mapping {
    submaps: usize,
    coupling: Vec<(usize, usize)>,
    chmux: Vec<usize>,
    floorsub: Vec<usize>,
    residuesub: Vec<usize>,
}

struct Mode {
    blockflag: bool,
    mapping: usize,
}

struct Setup {
    channels: usize,
    rate: u32,
    blocksizes: [usize; 2],
    books: Vec<Codebook>,
    floors: Vec<Floor1>,
    residues: Vec<Residue>,
    maps: Vec<Mapping>,
    modes: Vec<Mode>,
    modebits: u32,
}

fn floor1_unpack(p: &mut Pack, nbooks: usize) -> Result<Floor1, String> {
    let err = || Err::<Floor1, String>("bad floor 1 header".into());
    let partitions = p.read(5);
    if partitions < 0 {
        return err();
    }
    let mut partition_class = Vec::new();
    let mut maxclass: i64 = -1;
    for _ in 0..partitions {
        let c = p.read(4);
        if c < 0 {
            return err();
        }
        maxclass = maxclass.max(c);
        partition_class.push(c as usize);
    }
    let n_class = (maxclass + 1) as usize;
    let (mut class_dim, mut class_subs, mut class_book, mut class_subbook) = (vec![0; n_class], vec![0; n_class], vec![0; n_class], vec![Vec::new(); n_class]);
    for j in 0..n_class {
        class_dim[j] = (p.read(3) + 1) as usize;
        let subs = p.read(2);
        if subs < 0 {
            return err();
        }
        class_subs[j] = subs as u32;
        if subs > 0 {
            let b = p.read(8);
            if b < 0 || b as usize >= nbooks {
                return err();
            }
            class_book[j] = b as usize;
        }
        for _ in 0..(1 << subs) {
            let b = p.read(8) - 1;
            if b < -1 || b >= nbooks as i64 {
                return err();
            }
            class_subbook[j].push(b);
        }
    }
    let mult = (p.read(2) + 1) as i32;
    let rangebits = p.read(4);
    if rangebits < 0 {
        return err();
    }
    let mut postlist = vec![0i32, 1 << rangebits];
    let mut count = 0;
    for &c in &partition_class {
        count += class_dim[c];
        if count > 63 {
            return err();
        }
        while postlist.len() < count + 2 {
            let t = p.read(rangebits as u32);
            if t < 0 || t >= (1 << rangebits) {
                return err();
            }
            postlist.push(t as i32);
        }
    }
    let n = postlist.len();
    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by_key(|&i| postlist[i]);
    for w in sorted.windows(2) {
        if postlist[w[0]] == postlist[w[1]] {
            return err();
        }
    }
    let forward_index = sorted;
    let quant_q = match mult {
        1 => 256,
        2 => 128,
        3 => 86,
        _ => 64,
    };
    let (mut loneighbor, mut hineighbor) = (Vec::new(), Vec::new());
    for i in 0..n - 2 {
        let (mut lo, mut hi, mut lx, mut hx) = (0, 1, 0, postlist[1]);
        let cur = postlist[i + 2];
        for (j, &x) in postlist.iter().enumerate().take(i + 2) {
            if x > lx && x < cur {
                lo = j;
                lx = x;
            }
            if x < hx && x > cur {
                hi = j;
                hx = x;
            }
        }
        loneighbor.push(lo);
        hineighbor.push(hi);
    }
    Ok(Floor1 { partition_class, class_dim, class_subs, class_book, class_subbook, mult, postlist, forward_index, loneighbor, hineighbor, quant_q, posts: n })
}

fn residue_unpack(p: &mut Pack, kind: u32, books: &[Codebook]) -> Result<Residue, String> {
    let err = || Err::<Residue, String>("bad residue header".into());
    let begin = p.read(24);
    let end = p.read(24);
    let grouping = p.read(24) + 1;
    let partitions = (p.read(6) + 1) as usize;
    let groupbook = p.read(8);
    if groupbook < 0 {
        return err();
    }
    let mut secondstages = Vec::new();
    let mut acc = 0;
    for _ in 0..partitions {
        let mut cascade = p.read(3);
        let cflag = p.read(1);
        if cflag < 0 {
            return err();
        }
        if cflag == 1 {
            let c = p.read(5);
            if c < 0 {
                return err();
            }
            cascade |= c << 3;
        }
        acc += (cascade as u32).count_ones();
        secondstages.push(cascade as u32);
    }
    let mut booklist = Vec::new();
    for _ in 0..acc {
        let b = p.read(8);
        if b < 0 || b as usize >= books.len() || books[b as usize].maptype == 0 {
            return err();
        }
        booklist.push(b as usize);
    }
    let groupbook = groupbook as usize;
    if groupbook >= books.len() {
        return err();
    }
    let dim = books[groupbook].dim;
    let mut partvals = 1usize;
    for _ in 0..dim {
        partvals *= partitions;
        if partvals > books[groupbook].entries {
            return err();
        }
    }
    let mut partbooks = Vec::new();
    let mut maxstage = 0;
    let mut acc = 0;
    for &ss in &secondstages {
        let stages = ilog(ss) as usize;
        maxstage = maxstage.max(stages);
        let mut v = vec![None; stages];
        for (k, slot) in v.iter_mut().enumerate() {
            if ss & (1 << k) != 0 {
                *slot = Some(booklist[acc]);
                acc += 1;
            }
        }
        partbooks.push(v);
    }
    let mut decodemap = Vec::new();
    for j in 0..partvals {
        let mut val = j;
        let mut mult = partvals / partitions;
        let mut v = Vec::new();
        for _ in 0..dim {
            let deco = val / mult;
            val -= deco * mult;
            mult /= partitions;
            v.push(deco);
        }
        decodemap.push(v);
    }
    Ok(Residue { kind, begin, end, grouping, groupbook, secondstages, partbooks, stages: maxstage, partvals, decodemap })
}

fn parse_setup(id: &[u8], setup: &[u8], fma: Fma) -> Result<Setup, String> {
    if id.len() < 30 || &id[0..7] != b"\x01vorbis" {
        return Err("bad Vorbis identification header".into());
    }
    let mut p = Pack::new(&id[7..]);
    if p.read(32) != 0 {
        return Err("unsupported Vorbis version".into());
    }
    let channels = p.read(8) as usize;
    let rate = p.read(32) as u32;
    p.read(32);
    p.read(32);
    p.read(32);
    let b0 = 1usize << p.read(4);
    let b1 = 1usize << p.read(4);
    if rate < 1 || channels < 1 || b0 < 64 || b1 < b0 || b1 > 8192 || p.read(1) != 1 {
        return Err("bad Vorbis identification header".into());
    }
    if setup.len() < 7 || &setup[0..7] != b"\x05vorbis" {
        return Err("bad Vorbis setup header".into());
    }
    let mut p = Pack::new(&setup[7..]);
    let nbooks = (p.read(8) + 1) as usize;
    let mut books = Vec::with_capacity(nbooks);
    for _ in 0..nbooks {
        books.push(unpack_book(&mut p, fma)?);
    }
    let times = p.read(6) + 1;
    for _ in 0..times {
        if p.read(16) != 0 {
            return Err("bad Vorbis time backend".into());
        }
    }
    let nfloors = (p.read(6) + 1) as usize;
    let mut floors = Vec::new();
    for _ in 0..nfloors {
        match p.read(16) {
            1 => floors.push(floor1_unpack(&mut p, nbooks)?),
            0 => return Err("Vorbis floor 0 is not supported".into()),
            _ => return Err("bad Vorbis floor type".into()),
        }
    }
    let nres = (p.read(6) + 1) as usize;
    let mut residues = Vec::new();
    for _ in 0..nres {
        let k = p.read(16);
        if !(0..=2).contains(&k) {
            return Err("bad Vorbis residue type".into());
        }
        residues.push(residue_unpack(&mut p, k as u32, &books)?);
    }
    let nmaps = (p.read(6) + 1) as usize;
    let mut maps = Vec::new();
    for _ in 0..nmaps {
        if p.read(16) != 0 {
            return Err("bad Vorbis mapping type".into());
        }
        let submaps = if p.read(1) == 1 { (p.read(4) + 1) as usize } else { 1 };
        let mut coupling = Vec::new();
        if p.read(1) == 1 {
            let steps = p.read(8) + 1;
            let bits = ilog(channels as u32 - 1);
            for _ in 0..steps {
                let (m, a) = (p.read(bits), p.read(bits));
                if m < 0 || a < 0 || m == a || m as usize >= channels || a as usize >= channels {
                    return Err("bad Vorbis coupling".into());
                }
                coupling.push((m as usize, a as usize));
            }
        }
        if p.read(2) != 0 {
            return Err("bad Vorbis mapping reserved bits".into());
        }
        let mut chmux = vec![0usize; channels];
        if submaps > 1 {
            for c in chmux.iter_mut() {
                let v = p.read(4);
                if v < 0 || v as usize >= submaps {
                    return Err("bad Vorbis mux".into());
                }
                *c = v as usize;
            }
        }
        let (mut floorsub, mut residuesub) = (Vec::new(), Vec::new());
        for _ in 0..submaps {
            p.read(8);
            let f = p.read(8);
            let r = p.read(8);
            if f < 0 || f as usize >= nfloors || r < 0 || r as usize >= nres {
                return Err("bad Vorbis submap".into());
            }
            floorsub.push(f as usize);
            residuesub.push(r as usize);
        }
        maps.push(Mapping { submaps, coupling, chmux, floorsub, residuesub });
    }
    let nmodes = (p.read(6) + 1) as usize;
    let mut modes = Vec::new();
    for _ in 0..nmodes {
        let blockflag = p.read(1) == 1;
        let wt = p.read(16);
        let tt = p.read(16);
        let mapping = p.read(8);
        if wt != 0 || tt != 0 || mapping < 0 || mapping as usize >= nmaps {
            return Err("bad Vorbis mode".into());
        }
        modes.push(Mode { blockflag, mapping: mapping as usize });
    }
    if p.read(1) != 1 {
        return Err("bad Vorbis setup framing".into());
    }
    let modebits = ilog(nmodes as u32 - 1);
    Ok(Setup { channels, rate, blocksizes: [b0, b1], books, floors, residues, maps, modes, modebits })
}

// ---- floor 1 ---------------------------------------------------------------

fn render_point(x0: i32, x1: i32, y0: i32, y1: i32, x: i32) -> i32 {
    let (y0, y1) = (y0 & 0x7fff, y1 & 0x7fff);
    let dy = y1 - y0;
    let adx = x1 - x0;
    let ady = dy.abs();
    let err = ady * (x - x0);
    let off = err / adx;
    if dy < 0 {
        y0 - off
    } else {
        y0 + off
    }
}

fn floor1_inverse1(f: &Floor1, books: &[Codebook], p: &mut Pack) -> Option<Vec<i32>> {
    if p.read(1) != 1 {
        return None;
    }
    let mut fit = vec![0i32; f.posts];
    let qb = ilog(f.quant_q as u32 - 1);
    fit[0] = p.read(qb) as i32;
    fit[1] = p.read(qb) as i32;
    let mut j = 2;
    for &class in &f.partition_class {
        let cdim = f.class_dim[class];
        let csubbits = f.class_subs[class];
        let csub = 1i64 << csubbits;
        let mut cval = 0i64;
        if csubbits > 0 {
            cval = books[f.class_book[class]].decode(p);
            if cval == -1 {
                return None;
            }
        }
        for k in 0..cdim {
            let book = f.class_subbook[class][(cval & (csub - 1)) as usize];
            cval >>= csubbits;
            if book >= 0 {
                let v = books[book as usize].decode(p);
                if v == -1 {
                    return None;
                }
                fit[j + k] = v as i32;
            } else {
                fit[j + k] = 0;
            }
        }
        j += cdim;
    }
    for i in 2..f.posts {
        let (lo, hi) = (f.loneighbor[i - 2], f.hineighbor[i - 2]);
        let predicted = render_point(f.postlist[lo], f.postlist[hi], fit[lo], fit[hi], f.postlist[i]);
        let hiroom = f.quant_q - predicted;
        let loroom = predicted;
        let room = hiroom.min(loroom) << 1;
        let mut val = fit[i];
        if val != 0 {
            if val >= room {
                val = if hiroom > loroom { val - loroom } else { -1 - (val - hiroom) };
            } else if val & 1 != 0 {
                val = -((val + 1) >> 1);
            } else {
                val >>= 1;
            }
            fit[i] = (val + predicted) & 0x7fff;
            fit[lo] &= 0x7fff;
            fit[hi] &= 0x7fff;
        } else {
            fit[i] = predicted | 0x8000;
        }
    }
    Some(fit)
}

fn render_line(n: i32, x0: i32, x1: i32, y0: i32, y1: i32, d: &mut [f32]) {
    let dy = y1 - y0;
    let adx = x1 - x0;
    let mut ady = dy.abs();
    let base = dy / adx;
    let sy = if dy < 0 { base - 1 } else { base + 1 };
    let mut x = x0;
    let mut y = y0;
    let mut err = 0;
    ady -= (base * adx).abs();
    let n = n.min(x1);
    if x < n {
        d[x as usize] *= tables::FLOOR1_FROMDB[y as usize];
    }
    loop {
        x += 1;
        if x >= n {
            break;
        }
        err += ady;
        if err >= adx {
            err -= adx;
            y += sy;
        } else {
            y += base;
        }
        d[x as usize] *= tables::FLOOR1_FROMDB[y as usize];
    }
}

fn floor1_inverse2(f: &Floor1, fit: Option<&[i32]>, n: usize, out: &mut [f32]) {
    let Some(fit) = fit else {
        out[..n].fill(0.0);
        return;
    };
    let (mut hx, mut lx) = (0, 0);
    let mut ly = (fit[0] * f.mult).clamp(0, 255);
    for j in 1..f.posts {
        let current = f.forward_index[j];
        let hy = fit[current] & 0x7fff;
        if hy == fit[current] {
            hx = f.postlist[current];
            let hy = (hy * f.mult).clamp(0, 255);
            render_line(n as i32, lx, hx, ly, hy, out);
            lx = hx;
            ly = hy;
        }
    }
    for v in out.iter_mut().take(n).skip(hx as usize) {
        *v *= tables::FLOOR1_FROMDB[ly as usize];
    }
}

// ---- residues -------------------------------------------------------------

fn decodevs_add(b: &Codebook, a: &mut [f32], p: &mut Pack, n: usize) -> bool {
    if b.used == 0 {
        return true;
    }
    let step = n / b.dim;
    let mut entries = Vec::with_capacity(step);
    for _ in 0..step {
        match b.decode_packed(p) {
            Some(e) => entries.push(e),
            None => return false,
        }
    }
    let mut o = 0;
    for i in 0..b.dim {
        let mut j = 0;
        while o + j < n && j < step {
            a[o + j] += b.vec(entries[j])[i];
            j += 1;
        }
        o += step;
    }
    true
}

fn decodev_add(b: &Codebook, a: &mut [f32], p: &mut Pack, n: usize) -> bool {
    if b.used == 0 {
        return true;
    }
    let mut i = 0;
    while i < n {
        let Some(e) = b.decode_packed(p) else { return false };
        let t = b.vec(e);
        let mut j = 0;
        while i < n && j < b.dim {
            a[i] += t[j];
            i += 1;
            j += 1;
        }
    }
    true
}

fn decodevv_add(b: &Codebook, a: &mut [Vec<f32>], idx: &[usize], offset: usize, p: &mut Pack, n: usize) -> bool {
    if b.used == 0 {
        return true;
    }
    let ch = idx.len();
    let m = (offset + n) / ch;
    let mut i = offset / ch;
    let mut chptr = 0;
    while i < m {
        let Some(e) = b.decode_packed(p) else { return false };
        let t = b.vec(e);
        let mut j = 0;
        while i < m && j < b.dim {
            a[idx[chptr]][i] += t[j];
            chptr += 1;
            if chptr == ch {
                chptr = 0;
                i += 1;
            }
            j += 1;
        }
    }
    true
}

fn residue_inverse(r: &Residue, books: &[Codebook], p: &mut Pack, pcm: &mut [Vec<f32>], chans: &[usize], nonzero: &[bool], pcmend: usize) {
    let spp = r.grouping as usize;
    let ppw = books[r.groupbook].dim;
    if r.kind == 2 {
        let ch = chans.len();
        if !chans.iter().any(|&c| nonzero[c]) {
            return;
        }
        let max = (pcmend * ch) >> 1;
        let end = (r.end as usize).min(max);
        if end <= r.begin as usize {
            return;
        }
        let n = end - r.begin as usize;
        let partvals = n / spp;
        let partwords = partvals.div_ceil(ppw);
        let mut partword: Vec<usize> = vec![0; partwords];
        for s in 0..r.stages {
            let (mut i, mut l) = (0, 0);
            while i < partvals {
                if s == 0 {
                    let t = books[r.groupbook].decode(p);
                    if t == -1 || t as usize >= r.partvals {
                        return;
                    }
                    partword[l] = t as usize;
                }
                let mut k = 0;
                while k < ppw && i < partvals {
                    let part = r.decodemap[partword[l]][k];
                    if r.secondstages[part] & (1 << s) != 0 {
                        if let Some(Some(bk)) = r.partbooks[part].get(s) {
                            if !decodevv_add(&books[*bk], pcm, chans, i * spp + r.begin as usize, p, spp) {
                                return;
                            }
                        }
                    }
                    k += 1;
                    i += 1;
                }
                l += 1;
            }
        }
        return;
    }
    // Residue 0/1 over the nonzero channels only.
    let used: Vec<usize> = chans.iter().copied().filter(|&c| nonzero[c]).collect();
    if used.is_empty() {
        return;
    }
    let max = pcmend >> 1;
    let end = (r.end as usize).min(max);
    if end <= r.begin as usize {
        return;
    }
    let n = end - r.begin as usize;
    let partvals = n / spp;
    let partwords = partvals.div_ceil(ppw);
    let mut partword = vec![vec![0usize; partwords]; used.len()];
    for s in 0..r.stages {
        let (mut i, mut l) = (0, 0);
        while i < partvals {
            if s == 0 {
                for pw in partword.iter_mut() {
                    let t = books[r.groupbook].decode(p);
                    if t == -1 || t as usize >= r.partvals {
                        return;
                    }
                    pw[l] = t as usize;
                }
            }
            let mut k = 0;
            while k < ppw && i < partvals {
                for (j, &c) in used.iter().enumerate() {
                    let off = r.begin as usize + i * spp;
                    let part = r.decodemap[partword[j][l]][k];
                    if r.secondstages[part] & (1 << s) != 0 {
                        if let Some(Some(bk)) = r.partbooks[part].get(s) {
                            let ok = if r.kind == 0 {
                                decodevs_add(&books[*bk], &mut pcm[c][off..], p, spp)
                            } else {
                                decodev_add(&books[*bk], &mut pcm[c][off..], p, spp)
                            };
                            if !ok {
                                return;
                            }
                        }
                    }
                }
                k += 1;
                i += 1;
            }
            l += 1;
        }
    }
}

// ---- mdct (libvorbis mdct.c, float build) ---------------------------------

struct Mdct {
    n: usize,
    log2n: u32,
    trig: Vec<f32>,
    bitrev: Vec<usize>,
}


impl Mdct {
    fn new(n: usize) -> Self {
        let n2 = n >> 1;
        let log2n = ((n as f32).ln() / 2f32.ln()).round_ties_even() as u32;
        let mut t = vec![0f32; n + n / 4];
        let pi = std::f64::consts::PI;
        for i in 0..n / 4 {
            t[i * 2] = ((pi / n as f64) * (4 * i) as f64).cos() as f32;
            t[i * 2 + 1] = (-((pi / n as f64) * (4 * i) as f64).sin()) as f32;
            t[n2 + i * 2] = ((pi / (2 * n) as f64) * (2 * i + 1) as f64).cos() as f32;
            t[n2 + i * 2 + 1] = ((pi / (2 * n) as f64) * (2 * i + 1) as f64).sin() as f32;
        }
        for i in 0..n / 8 {
            t[n + i * 2] = (((pi / n as f64) * (4 * i + 2) as f64).cos() * 0.5) as f32;
            t[n + i * 2 + 1] = (-((pi / n as f64) * (4 * i + 2) as f64).sin() * 0.5) as f32;
        }
        let mask = (1usize << (log2n - 1)) - 1;
        let msb = 1usize << (log2n - 2);
        let mut bitrev = vec![0usize; n / 4];
        for i in 0..n / 8 {
            let mut acc = 0usize;
            let mut j = 0;
            while msb >> j != 0 {
                if (msb >> j) & i != 0 {
                    acc |= 1 << j;
                }
                j += 1;
            }
            bitrev[i * 2] = ((!acc) & mask).wrapping_sub(1);
            bitrev[i * 2 + 1] = acc;
        }
        Self { n, log2n, trig: t, bitrev }
    }

    /// One 8-element step of `mdct_butterfly_first/generic` at `x1`/`x2`
    /// (indices into `x`) with twiddles `t[a], t[a+1]`.
    fn bstep(m: Fma, x: &mut [f32], x1: usize, x2: usize, k: usize, t0: f32, t1: f32) {
        let r0 = x[x1 + k] - x[x2 + k];
        let r1 = x[x1 + k + 1] - x[x2 + k + 1];
        x[x1 + k] += x[x2 + k];
        x[x1 + k + 1] += x[x2 + k + 1];
        x[x2 + k] = madd(m, r1, t1, r0, t0);
        x[x2 + k + 1] = msub(m, r1, t0, r0, t1);
    }

    fn butterfly_first(&self, m: Fma, x: &mut [f32], base: usize, points: usize) {
        let t = &self.trig;
        let mut x1 = base + points - 8;
        let mut x2 = base + (points >> 1) - 8;
        let mut ti = 0;
        loop {
            Self::bstep(m, x, x1, x2, 6, t[ti], t[ti + 1]);
            Self::bstep(m, x, x1, x2, 4, t[ti + 4], t[ti + 5]);
            Self::bstep(m, x, x1, x2, 2, t[ti + 8], t[ti + 9]);
            Self::bstep(m, x, x1, x2, 0, t[ti + 12], t[ti + 13]);
            ti += 16;
            if x2 < base + 8 {
                break;
            }
            x1 -= 8;
            x2 -= 8;
        }
    }

    fn butterfly_generic(&self, m: Fma, x: &mut [f32], base: usize, points: usize, trigint: usize) {
        let t = &self.trig;
        let mut x1 = base + points - 8;
        let mut x2 = base + (points >> 1) - 8;
        let mut ti = 0;
        loop {
            Self::bstep(m, x, x1, x2, 6, t[ti], t[ti + 1]);
            ti += trigint;
            Self::bstep(m, x, x1, x2, 4, t[ti], t[ti + 1]);
            ti += trigint;
            Self::bstep(m, x, x1, x2, 2, t[ti], t[ti + 1]);
            ti += trigint;
            Self::bstep(m, x, x1, x2, 0, t[ti], t[ti + 1]);
            ti += trigint;
            if x2 < base + 8 {
                break;
            }
            x1 -= 8;
            x2 -= 8;
        }
    }

    fn butterflies(&self, m: Fma, x: &mut [f32], base: usize, points: usize) {
        let mut stages = self.log2n as i32 - 5;
        stages -= 1;
        if stages > 0 {
            self.butterfly_first(m, x, base, points);
        }
        let mut i = 1;
        loop {
            stages -= 1;
            if stages <= 0 {
                break;
            }
            for j in 0..(1usize << i) {
                self.butterfly_generic(m, x, base + (points >> i) * j, points >> i, 4 << i);
            }
            i += 1;
        }
        let mut j = 0;
        while j < points {
            // butterfly_32/16/8, inlined and reassociated by -ffast-math.
            built::butterfly_32_built(x, base + j);
            j += 32;
        }
    }

    fn bitreverse(&self, m: Fma, x: &mut [f32]) {
        let n = self.n;
        let bit = &self.bitrev;
        let t = &self.trig[n..];
        let xb = n >> 1; // x = w0 + n/2
        let mut w0 = 0usize;
        let mut w1 = n >> 1;
        let (mut bi, mut ti) = (0usize, 0usize);
        loop {
            let x0 = xb + bit[bi];
            let x1 = xb + bit[bi + 1];
            let r0 = x[x0 + 1] - x[x1 + 1];
            let r1 = x[x0] + x[x1];
            let r2 = madd(m, r1, t[ti], r0, t[ti + 1]);
            let r3 = msub(m, r1, t[ti + 1], r0, t[ti]);
            w1 -= 4;
            let r0 = (x[x0 + 1] + x[x1 + 1]) * 0.5;
            let r1 = (x[x0] - x[x1]) * 0.5;
            x[w0] = r0 + r2;
            x[w1 + 2] = r0 - r2;
            x[w0 + 1] = r1 + r3;
            x[w1 + 3] = r3 - r1;
            let x0 = xb + bit[bi + 2];
            let x1 = xb + bit[bi + 3];
            let r0 = x[x0 + 1] - x[x1 + 1];
            let r1 = x[x0] + x[x1];
            let r2 = madd(m, r1, t[ti + 2], r0, t[ti + 3]);
            let r3 = msub(m, r1, t[ti + 3], r0, t[ti + 2]);
            let r0 = (x[x0 + 1] + x[x1 + 1]) * 0.5;
            let r1 = (x[x0] - x[x1]) * 0.5;
            x[w0 + 2] = r0 + r2;
            x[w1] = r0 - r2;
            x[w0 + 3] = r1 + r3;
            x[w1 + 1] = r3 - r1;
            ti += 4;
            bi += 4;
            w0 += 4;
            if w0 >= w1 {
                break;
            }
        }
    }

    /// `mdct_backward(init, in, out)` with `in == out` (as mapping0 calls it).
    fn backward(&self, m: Fma, buf: &mut [f32]) {
        let n = self.n;
        let (n2, n4) = (n >> 1, n >> 2);
        let input: Vec<f32> = buf[..n2].to_vec();
        let out = buf;
        let t = &self.trig;
        // rotate
        let mut ix = n2 as isize - 7;
        let mut ox = n2 + n4;
        let mut ti = n4;
        loop {
            ox -= 4;
            let i = ix as usize;
            out[ox] = msub(m, -input[i + 2], t[ti + 3], input[i], t[ti + 2]);
            out[ox + 1] = msub(m, input[i], t[ti + 3], input[i + 2], t[ti + 2]);
            out[ox + 2] = msub(m, -input[i + 6], t[ti + 1], input[i + 4], t[ti]);
            out[ox + 3] = msub(m, input[i + 4], t[ti + 1], input[i + 6], t[ti]);
            ix -= 8;
            ti += 4;
            if ix < 0 {
                break;
            }
        }
        let mut ix = n2 as isize - 8;
        let mut ox = n2 + n4;
        let mut ti = n4;
        loop {
            ti -= 4;
            let i = ix as usize;
            out[ox] = madd(m, input[i + 4], t[ti + 3], input[i + 6], t[ti + 2]);
            out[ox + 1] = msub(m, input[i + 4], t[ti + 2], input[i + 6], t[ti + 3]);
            out[ox + 2] = madd(m, input[i], t[ti + 1], input[i + 2], t[ti]);
            out[ox + 3] = msub(m, input[i], t[ti], input[i + 2], t[ti + 1]);
            ix -= 8;
            ox += 4;
            if ix < 0 {
                break;
            }
        }
        self.butterflies(m, out, n2, n2);
        self.bitreverse(m, out);
        // rotate + window
        {
            let mut ox1 = n2 + n4;
            let mut ox2 = n2 + n4;
            let mut ix = 0usize;
            let mut ti = n2;
            loop {
                ox1 -= 4;
                out[ox1 + 3] = msub(m, out[ix], t[ti + 1], out[ix + 1], t[ti]);
                out[ox2] = neg(madd(m, out[ix], t[ti], out[ix + 1], t[ti + 1]));
                out[ox1 + 2] = msub(m, out[ix + 2], t[ti + 3], out[ix + 3], t[ti + 2]);
                out[ox2 + 1] = neg(madd(m, out[ix + 2], t[ti + 2], out[ix + 3], t[ti + 3]));
                out[ox1 + 1] = msub(m, out[ix + 4], t[ti + 5], out[ix + 5], t[ti + 4]);
                out[ox2 + 2] = neg(madd(m, out[ix + 4], t[ti + 4], out[ix + 5], t[ti + 5]));
                out[ox1] = msub(m, out[ix + 6], t[ti + 7], out[ix + 7], t[ti + 6]);
                out[ox2 + 3] = neg(madd(m, out[ix + 6], t[ti + 6], out[ix + 7], t[ti + 7]));
                ox2 += 4;
                ix += 8;
                ti += 8;
                if ix >= ox1 {
                    break;
                }
            }
            let mut ix = n2 + n4;
            let mut ox1 = n4;
            let mut ox2 = ox1;
            loop {
                ox1 -= 4;
                ix -= 4;
                out[ox1 + 3] = out[ix + 3];
                out[ox2] = -out[ox1 + 3];
                out[ox1 + 2] = out[ix + 2];
                out[ox2 + 1] = -out[ox1 + 2];
                out[ox1 + 1] = out[ix + 1];
                out[ox2 + 2] = -out[ox1 + 1];
                out[ox1] = out[ix];
                out[ox2 + 3] = -out[ox1];
                ox2 += 4;
                if ox2 >= ix {
                    break;
                }
            }
            let mut ix = n2 + n4;
            let mut ox1 = n2 + n4;
            let ox2 = n2;
            loop {
                ox1 -= 4;
                out[ox1] = out[ix + 3];
                out[ox1 + 1] = out[ix + 2];
                out[ox1 + 2] = out[ix + 1];
                out[ox1 + 3] = out[ix];
                ix += 4;
                if ox1 <= ox2 {
                    break;
                }
            }
        }
    }
}

// ---- decoding driver -------------------------------------------------------

/// Decoded Vorbis: planar float as libvorbis hands it to libsndfile.
pub struct Vorbis {
    /// Samples per second.
    pub sample_rate: u32,
    /// One `Vec` per channel.
    pub channels: Vec<Vec<f32>>,
}

/// `vorbis_packet_blocksize` for an audio packet (`None` if not audio).
fn packet_blocksize(s: &Setup, d: &[u8]) -> Option<usize> {
    let mut p = Pack::new(d);
    if p.read(1) != 0 {
        return None;
    }
    let mode = p.read(s.modebits);
    if mode < 0 || mode as usize >= s.modes.len() {
        return None;
    }
    Some(s.blocksizes[s.modes[mode as usize].blockflag as usize])
}

/// Decode an Ogg Vorbis file as libsndfile 1.2.2 reads it (frames
/// `pcm_end - pcm_start`).
pub fn decode_ogg(data: &[u8]) -> Result<Vorbis, String> {
    let fma = mode();
    let (pk, last_granule) = ogg::packets(data)?;
    if pk.len() < 3 || !pk[0].data.starts_with(b"\x01vorbis") {
        return Err("not Ogg Vorbis".into());
    }
    let s = parse_setup(&pk[0].data, &pk[2].data, fma)?;
    let ch = s.channels;
    let mdct = [Mdct::new(s.blocksizes[0]), Mdct::new(s.blocksizes[1])];
    let win = [
        tables::VWIN[(ilog(s.blocksizes[0] as u32) - 7) as usize],
        tables::VWIN[(ilog(s.blocksizes[1] as u32) - 7) as usize],
    ];
    // pcm_start: the first audio page's granule minus its decodable samples.
    let audio = &pk[3..];
    let first_page = audio.first().map(|p| p.page);
    let pcm_start: i64 = match first_page {
        Some(fp) => {
            let page: Vec<&ogg::Packet> = audio.iter().filter(|p| p.page == fp).collect();
            let mut lastblock: i64 = -1;
            let mut duration: i64 = 0;
            for p in &page {
                if let Some(b) = packet_blocksize(&s, &p.data) {
                    if lastblock != -1 {
                        duration += (lastblock + b as i64) >> 2;
                    }
                    lastblock = b as i64;
                }
            }
            let last = page.last().unwrap();
            if last.eos && last.granulepos <= duration {
                0
            } else {
                (last.granulepos - duration).max(0)
            }
        }
        None => 0,
    };
    let frames = if last_granule >= 0 { (last_granule - pcm_start).max(0) as usize } else { usize::MAX };

    // vorbis_dsp_state
    let n1 = s.blocksizes[1] >> 1;
    let mut vpcm = vec![vec![0f32; s.blocksizes[1]]; ch];
    let mut center_w = n1;
    #[allow(unused_assignments)]
    let (mut pcm_current, mut pcm_returned): (i64, i64) = (center_w as i64, -1);
    #[allow(unused_assignments)]
    let (mut lw, mut w) = (0usize, 0usize);
    let mut sequence: i64 = -1;
    let mut granulepos: i64 = -1;
    let mut sample_count: i64 = -1;
    let mut out: Vec<Vec<f32>> = vec![Vec::new(); ch];
    let mut emitted = 0usize;

    for packet in audio {
        // vorbis_synthesis
        let mut p = Pack::new(&packet.data);
        if p.read(1) != 0 {
            continue;
        }
        let mode = p.read(s.modebits);
        if mode < 0 || mode as usize >= s.modes.len() {
            continue;
        }
        let md = &s.modes[mode as usize];
        let vw = md.blockflag as usize;
        if vw == 1 {
            p.read(1);
            if p.read(1) == -1 {
                continue;
            }
        }
        let n = s.blocksizes[vw];
        let map = &s.maps[md.mapping];
        let mut pcm: Vec<Vec<f32>> = vec![vec![0f32; n]; ch];
        // mapping0_inverse
        let mut fits: Vec<Option<Vec<i32>>> = Vec::with_capacity(ch);
        let mut nonzero = vec![false; ch];
        for c in 0..ch {
            let f = &s.floors[map.floorsub[map.chmux[c]]];
            let fit = floor1_inverse1(f, &s.books, &mut p);
            nonzero[c] = fit.is_some();
            fits.push(fit);
        }
        for &(m, a) in &map.coupling {
            if nonzero[m] || nonzero[a] {
                nonzero[m] = true;
                nonzero[a] = true;
            }
        }
        for sm in 0..map.submaps {
            let chans: Vec<usize> = (0..ch).filter(|&c| map.chmux[c] == sm).collect();
            residue_inverse(&s.residues[map.residuesub[sm]], &s.books, &mut p, &mut pcm, &chans, &nonzero, n);
        }
        for &(m, a) in map.coupling.iter().rev() {
            for j in 0..n / 2 {
                let (mag, ang) = (pcm[m][j], pcm[a][j]);
                if mag > 0.0 {
                    if ang > 0.0 {
                        pcm[m][j] = mag;
                        pcm[a][j] = mag - ang;
                    } else {
                        pcm[a][j] = mag;
                        pcm[m][j] = mag + ang;
                    }
                } else if ang > 0.0 {
                    pcm[m][j] = mag;
                    pcm[a][j] = mag + ang;
                } else {
                    pcm[a][j] = mag;
                    pcm[m][j] = mag - ang;
                }
            }
        }
        for c in 0..ch {
            let f = &s.floors[map.floorsub[map.chmux[c]]];
            floor1_inverse2(f, fits[c].as_deref(), n / 2, &mut pcm[c]);
        }
        for c in pcm.iter_mut() {
            mdct[vw].backward(fma, c);
        }

        // vorbis_synthesis_blockin
        lw = w;
        w = vw;
        if sequence == -1 || sequence + 1 != packet.packetno {
            granulepos = -1;
            sample_count = -1;
        }
        sequence = packet.packetno;
        let nn = s.blocksizes[w] >> 1;
        let n0 = s.blocksizes[0] >> 1;
        let (this_center, prev_center) = if center_w != 0 { (n1, 0) } else { (0, n1) };
        for c in 0..ch {
            let v = &mut vpcm[c];
            let pb = &pcm[c];
            let lap = |dst: &mut [f32], src: &[f32], wv: &[f32], len: usize| {
                for i in 0..len {
                    dst[i] = madd(fma, dst[i], wv[len - i - 1], src[i], wv[i]);
                }
            };
            if lw == 1 {
                if w == 1 {
                    lap(&mut v[prev_center..], pb, win[1], n1);
                } else {
                    let o = prev_center + n1 / 2 - n0 / 2;
                    lap(&mut v[o..], pb, win[0], n0);
                }
            } else if w == 1 {
                let o = n1 / 2 - n0 / 2;
                lap(&mut v[prev_center..], &pb[o..], win[0], n0);
                for i in n0..n1 / 2 + n0 / 2 {
                    v[prev_center + i] = pb[o + i];
                }
            } else {
                lap(&mut v[prev_center..], pb, win[0], n0);
            }
            v[this_center..this_center + nn].copy_from_slice(&pb[nn..2 * nn]);
        }
        center_w = if center_w != 0 { 0 } else { n1 };
        if pcm_returned == -1 {
            pcm_returned = this_center as i64;
            pcm_current = this_center as i64;
        } else {
            pcm_returned = prev_center as i64;
            pcm_current = prev_center as i64 + ((s.blocksizes[lw] / 4 + s.blocksizes[w] / 4) as i64);
        }
        if sample_count == -1 {
            sample_count = 0;
        } else {
            sample_count += (s.blocksizes[lw] / 4 + s.blocksizes[w] / 4) as i64;
        }
        if granulepos == -1 {
            if packet.granulepos != -1 {
                granulepos = packet.granulepos;
                if sample_count > granulepos {
                    let mut extra = (sample_count - packet.granulepos).max(0);
                    if packet.eos {
                        extra = extra.min(pcm_current - pcm_returned);
                        pcm_current -= extra;
                    } else {
                        pcm_returned += extra;
                        pcm_returned = pcm_returned.min(pcm_current);
                    }
                }
            }
        } else {
            granulepos += (s.blocksizes[lw] / 4 + s.blocksizes[w] / 4) as i64;
            if packet.granulepos != -1 && granulepos != packet.granulepos {
                if granulepos > packet.granulepos {
                    let extra = granulepos - packet.granulepos;
                    if extra != 0 && packet.eos {
                        let extra = extra.min(pcm_current - pcm_returned).max(0);
                        pcm_current -= extra;
                    }
                }
                granulepos = packet.granulepos;
            }
        }
        // libsndfile drains pcmout after each block.
        if pcm_returned > -1 && pcm_returned < pcm_current {
            let avail = (pcm_current - pcm_returned) as usize;
            let take = avail.min(frames - emitted);
            for c in 0..ch {
                let st = pcm_returned as usize;
                out[c].extend_from_slice(&vpcm[c][st..st + take]);
            }
            emitted += take;
            pcm_returned += take as i64;
            if emitted == frames {
                break;
            }
        }
    }
    Ok(Vorbis { sample_rate: s.rate, channels: out })
}
