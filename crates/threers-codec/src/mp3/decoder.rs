//! mpg123 1.32.3's layer III decoder (`layer3.c`, `parse.c`, `frame.c`,
//! `synth_real.c`, `synth_mono.h`) as libsndfile drives it: float output,
//! `MPG123_GAPLESS | MPG123_NO_FRANKENSTEIN`, NEON64 synth.
//!
//! Integer work (headers, side info, scalefactors, Huffman, the bit
//! reservoir) is ported from the C. Float arrays live in the [`Cpu`] arena so
//! the kernels generated from the reference's machine code (`dct12`,
//! `dct36_neon64`, `dct64_real_neon64`, the synths, the vectorised antialias
//! loop) run on them directly.

#![allow(clippy::needless_range_loop, clippy::collapsible_if, clippy::explicit_counter_loop)]

use super::built;
use super::cpu::Cpu;
use super::tables as t;

// ---- arena layout (byte addresses) -----------------------------------------
const REAL_BUFFS: u64 = 0x1000; // [ch][2][0x110] f32
const DECWIN: u64 = 0x3000; // 512 + 32 + 512 f32 (+ slack)
const SAMPLES: u64 = 0x5000; // one frame of output, interleaved, 1152 * 2 f32
const MONO_TMP: u64 = 0x7800; // 64 f32
const HYBRID_IN: u64 = 0x8000; // [2][576] f32
const HYBRID_OUT: u64 = 0xA000; // [2][576] f32 ([ss][sb])
const HYBRID_BLOCK: u64 = 0xC000; // [2][2][576] f32
const WIN: u64 = 0xF000; // [4][36]
const WIN1: u64 = 0xF400; // [4][36]

const FREQS: [u32; 9] = [44100, 48000, 32000, 22050, 24000, 16000, 11025, 12000, 8000];
const TABSEL: [[[u32; 16]; 3]; 2] = [
    [
        [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 0],
        [0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 0],
        [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0],
    ],
    [
        [0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 0],
        [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0],
        [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0],
    ],
];

/// mpg123's `head_check` (layer III only here).
fn head_check(h: u32) -> bool {
    (h & 0xffe0_0000) == 0xffe0_0000 && (h >> 17) & 3 != 0 && (h >> 12) & 0xf != 0xf && (h >> 10) & 3 != 3
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Header {
    lay: u32,
    lsf: usize,
    mpeg25: bool,
    sfreq: usize,
    error_protection: bool,
    bitrate_index: usize,
    padding: usize,
    mode: u32,
    mode_ext: u32,
    stereo: usize,
    framesize: usize,
    ssize: usize,
    spf: usize,
}

/// `decode_header` for layer III; `free` is the known free-format frame
/// size (without padding), if any.
fn decode_header(h: u32, free: Option<usize>) -> Result<Header, String> {
    let lay = 4 - ((h >> 17) & 3);
    let ver = (h >> 19) & 3;
    let (lsf, mpeg25, sfreq) = if ver & 2 != 0 {
        let lsf = if ver & 1 != 0 { 0 } else { 1 };
        (lsf, false, ((h >> 10) & 3) as usize + lsf * 3)
    } else {
        (1, true, 6 + ((h >> 10) & 3) as usize)
    };
    let bitrate_index = ((h >> 12) & 0xf) as usize;
    if lay != 3 {
        return Err(format!("MPEG audio layer {lay} is not supported (layer III only)"));
    }
    let padding = ((h >> 9) & 1) as usize;
    let mode = (h >> 6) & 3;
    let stereo = if mode == 3 { 1 } else { 2 };
    let error_protection = (h >> 16) & 1 == 0;
    let mut ssize = if lsf == 1 { if stereo == 1 { 9 } else { 17 } } else if stereo == 1 { 17 } else { 32 };
    if error_protection {
        ssize += 2;
    }
    let framesize = if bitrate_index == 0 {
        free.ok_or("free-format frame size unknown")? + padding
    } else {
        let mut fs = TABSEL[lsf][2][bitrate_index] as i64 * 144000;
        fs /= (FREQS[sfreq] as i64) << lsf;
        fs += padding as i64 - 4;
        fs as usize
    };
    if framesize < ssize || framesize > MAXFRAMESIZE {
        return Err("bad MP3 frame size".into());
    }
    Ok(Header {
        lay,
        lsf,
        mpeg25,
        sfreq,
        error_protection,
        bitrate_index,
        padding,
        mode,
        mode_ext: (h >> 4) & 3,
        stereo,
        framesize,
        ssize,
        spf: if lsf == 1 { 576 } else { 1152 },
    })
}

/// mpg123's `head_compatible`: same sync/version/layer/sampling rate
/// (`HDR_CMPMASK`) and the same mono-ness.
fn head_compatible(a: u32, b: u32) -> bool {
    (a & 0xfffe_0c00) == (b & 0xfffe_0c00) && ((a >> 6) & 3 == 3) == ((b >> 6) & 3 == 3)
}

const MAXFRAMESIZE: usize = 3456;
const BSSPACE: usize = MAXFRAMESIZE + 512 + 4;
const GAPLESS_DELAY: i64 = 529;
const PREFRAMES: i64 = 4;

/// The parts of `mpg123_handle` the layer III path touches.
struct Dec<'a> {
    data: &'a [u8],
    filepos: usize,
    filelen: i64,
    cpu: Cpu,
    gainpow2: [f32; 378],
    long_limit: [[i32; 23]; 9],
    short_limit: [[i32; 14]; 9],
    // frame
    h: Header,
    /// `fr->freeformat_framesize`, once guessed.
    freeformat_framesize: Option<usize>,
    firsthead: u32,
    oldhead: u32,
    fsizeold: usize,
    bsspace: [Vec<u8>; 2],
    bsnum: usize,
    /// (buffer, offset) of `bsbuf` / `bsbufold`.
    bsbuf: (usize, usize),
    bsbufold: (usize, usize),
    wp: usize,
    bitindex: u32,
    bits_avail: i64,
    bitreservoir: u32,
    hybrid_blc: [usize; 2],
    bo: u32,
    // track
    num: i64,
    track_frames: i64,
    gapless_frames: i64,
    begin_s: i64,
    end_s: i64,
    begin_os: i64,
    end_os: i64,
    fullend_os: i64,
    firstframe: i64,
    firstoff: i64,
    lastframe: i64,
    lastoff: i64,
    ignoreframe: i64,
    mean_frames: i64,
    mean_framesize: f64,
    to_decode: bool,
    to_ignore: bool,
    fresh: bool,
    header_change: u32,
    /// The current frame's output (interleaved f32).
    out: Vec<f32>,
    /// `fr->index`: the header position of every frame read (step 1).
    index: Vec<usize>,
}

#[derive(Clone, Copy, Default)]
struct GrInfo {
    scfsi: i32,
    part2_3_length: u32,
    big_values: u32,
    scalefac_compress: u32,
    block_type: u32,
    mixed_block_flag: u32,
    table_select: [u32; 3],
    maxband: [i32; 3],
    maxbandl: i32,
    maxb: u32,
    region1start: u32,
    region2start: u32,
    preflag: u32,
    scalefac_scale: u32,
    count1table_select: u32,
    /// Offsets into `gainpow2` of `full_gain[3]` and `pow2gain`.
    full_gain: [i32; 3],
    pow2gain: i32,
}

#[derive(Default)]
struct SideInfo {
    main_data_begin: u32,
    ch: [[GrInfo; 2]; 2],
}

fn f(bits: u32) -> f32 {
    f32::from_bits(bits)
}

impl<'a> Dec<'a> {
    fn new(data: &'a [u8]) -> Self {
        // `INT123_init_layer3_gainpow2`: pow(2, -0.25 * (i + 210)), exact in
        // the binary exponent, so split it into 2^-k * 2^(-j/4).
        let quarter = [1.0f64, 0.840_896_415_253_714_5, std::f64::consts::FRAC_1_SQRT_2, 0.594_603_557_501_360_5];
        let mut gainpow2 = [0f32; 378];
        for (k, g) in gainpow2.iter_mut().enumerate() {
            let e = k as i64 - 256 + 210; // 2^(-e/4)
            let (q, r) = (e.div_euclid(4), e.rem_euclid(4));
            *g = (quarter[r as usize] * 2f64.powi(-q as i32)) as f32;
        }
        let mut long_limit = [[0i32; 23]; 9];
        let mut short_limit = [[0i32; 14]; 9];
        for j in 0..9 {
            for i in 0..23 {
                long_limit[j][i] = ((t::BAND_INFO[j].long_idx[i] as i32 - 1 + 8) / 18 + 1).min(32);
            }
            for i in 0..14 {
                short_limit[j][i] = ((t::BAND_INFO[j].short_idx[i] as i32 - 1) / 18 + 1).min(32);
            }
        }
        let mut cpu = Cpu::default();
        for (i, w) in t::WIN.iter().enumerate() {
            cpu.st32(WIN + 4 * i as u64, *w);
        }
        for (i, w) in t::WIN1.iter().enumerate() {
            cpu.st32(WIN1 + 4 * i as u64, *w);
        }
        // `INT123_make_decode_tables` with outscale 1, then the NEON layout.
        let mut decwin = [0f32; 512 + 32 + 512];
        let mut scaleval = -0.5f64;
        let (mut idx, mut j) = (0i64, 0i64);
        for i in 0..512i64 {
            if idx < 512 + 16 {
                let v = (t::INTWINBASE[j as usize] as f64 * scaleval) as f32;
                decwin[idx as usize + 16] = v;
                decwin[idx as usize] = v;
            }
            if i % 32 == 31 {
                idx -= 1023;
            }
            if i % 64 == 63 {
                scaleval = -scaleval;
            }
            idx += 32;
            j += if i < 256 { 1 } else { -1 };
        }
        for i in 512..544 {
            if i & 1 == 0 {
                decwin[i] = 0.0;
            }
        }
        for i in 0..512 {
            decwin[544 + i] = -decwin[511 - i];
        }
        for i in (0..512).step_by(2) {
            decwin[i] = -decwin[i];
        }
        cpu.put(DECWIN, &decwin);
        let mut d = Dec {
            data,
            filepos: 0,
            filelen: 0,
            cpu,
            gainpow2,
            long_limit,
            short_limit,
            h: decode_header(0xfffb_9000, None).unwrap(),
            freeformat_framesize: None,
            firsthead: 0,
            oldhead: 0,
            fsizeold: 0,
            bsspace: [vec![0; BSSPACE], vec![0; BSSPACE]],
            bsnum: 0,
            bsbuf: (1, 0),
            bsbufold: (1, 0),
            wp: 0,
            bitindex: 0,
            bits_avail: 0,
            bitreservoir: 0,
            hybrid_blc: [0, 0],
            bo: 1,
            num: -1,
            track_frames: 0,
            gapless_frames: -1,
            begin_s: 0,
            end_s: 0,
            begin_os: 0,
            end_os: 0,
            fullend_os: 0,
            firstframe: 0,
            firstoff: 0,
            lastframe: -1,
            lastoff: 0,
            ignoreframe: -PREFRAMES,
            mean_frames: 0,
            mean_framesize: 0.0,
            to_decode: false,
            to_ignore: false,
            fresh: true,
            header_change: 0,
            out: Vec::new(),
            index: Vec::new(),
        };
        d.h.framesize = 0;
        // `get_fileinfo`: the length without a trailing ID3v1 tag.
        let mut len = data.len() as i64;
        if data.len() >= 128 && &data[data.len() - 128..data.len() - 125] == b"TAG" {
            len -= 128;
        }
        d.filelen = len;
        d
    }

    // ---- getbits.h, on `wordpointer` inside the current `bsbuf` ------------

    fn byte(&self, i: usize) -> u32 {
        *self.bsspace[self.bsbuf.0].get(i).unwrap_or(&0) as u32
    }
    fn getbits(&mut self, n: u32) -> u32 {
        self.bits_avail -= n as i64;
        if self.bits_avail < 0 {
            return 0;
        }
        let mut r = (self.byte(self.wp) << 16) | (self.byte(self.wp + 1) << 8) | self.byte(self.wp + 2);
        r <<= self.bitindex;
        r &= 0xffffff;
        self.bitindex += n;
        r >>= 24 - n;
        self.wp += (self.bitindex >> 3) as usize;
        self.bitindex &= 7;
        r
    }
    fn skipbits(&mut self, n: u32) {
        self.bitindex += n;
        self.bits_avail -= n as i64;
        self.wp += (self.bitindex >> 3) as usize;
        self.bitindex &= 7;
    }
    fn getbits_fast(&mut self, n: u32) -> u32 {
        let mut u = (self.byte(self.wp) << self.bitindex) & 0xff;
        u |= (self.byte(self.wp + 1) << self.bitindex) >> 8 & 0xff;
        u <<= n;
        u >>= 8;
        self.bitindex += n;
        self.bits_avail -= n as i64;
        self.wp += (self.bitindex >> 3) as usize;
        self.bitindex &= 7;
        u
    }
    fn get1bit(&mut self) -> u32 {
        let u = (self.byte(self.wp) << self.bitindex) & 0xff;
        self.bitindex += 1;
        self.bits_avail -= 1;
        self.wp += (self.bitindex >> 3) as usize;
        self.bitindex &= 7;
        u >> 7
    }
    fn getbyte(&mut self) -> u32 {
        self.bits_avail -= 8;
        if self.bits_avail >= 0 {
            let b = self.byte(self.wp);
            self.wp += 1;
            b
        } else {
            0
        }
    }
    fn backbits(&mut self, n: i64) {
        self.bits_avail += n;
        let bi = self.bitindex as i64 - n;
        self.wp = (self.wp as i64 + (bi >> 3)) as usize;
        self.bitindex = (bi & 7) as u32;
    }
    fn getbitoffset(&self) -> u32 {
        self.bitindex.wrapping_neg() & 7
    }

    fn set_pointer(&mut self, part2: bool, backstep: usize) {
        self.bitindex = 0;
        let (b, off) = self.bsbuf;
        if part2 {
            self.wp = off + self.h.ssize - backstep;
            if backstep > 0 {
                let (ob, ooff) = self.bsbufold;
                let src = ooff + self.fsizeold - backstep;
                let tmp: Vec<u8> = self.bsspace[ob][src..src + backstep].to_vec();
                self.bsspace[b][self.wp..self.wp + backstep].copy_from_slice(&tmp);
            }
            self.bits_avail = (self.h.framesize as i64 - self.h.ssize as i64 + backstep as i64) * 8;
        } else {
            self.wp = off;
            self.bits_avail = self.h.ssize as i64 * 8;
        }
    }
}

const SINGLE_STEREO: i32 = -1;

impl Dec<'_> {
    /// `III_get_side_info`; `Err` is mpg123's "bad frame".
    fn get_side_info(&mut self, si: &mut SideInfo, stereo: usize, ms_stereo: bool, sfreq: usize) -> Result<(), ()> {
        let powdiff = 0; // single != SINGLE_MIX
        let lsf = self.h.lsf;
        let tab: [u32; 5] = if lsf == 0 { [2, 9, 5, 3, 4] } else { [1, 8, 1, 2, 9] };
        let mut needbits = tab[1] + if stereo == 1 { tab[2] } else { tab[3] };
        if lsf == 0 {
            needbits += stereo as u32 * 4;
        }
        needbits += tab[0] * stereo as u32 * (29 + tab[4] + 1 + 22 + if lsf == 0 { 1 } else { 0 } + 2);
        if self.bits_avail < needbits as i64 {
            return Err(());
        }
        si.main_data_begin = self.getbits(tab[1]);
        if si.main_data_begin > self.bitreservoir {
            self.backbits(tab[1] as i64);
            let b = self.bsbuf.0;
            let wp = self.wp;
            if lsf == 0 {
                self.bsspace[b][wp] = (self.bitreservoir >> 1) as u8;
                self.bsspace[b][wp + 1] = ((self.bitreservoir & 1) << 7) as u8;
            } else {
                self.bsspace[b][wp] = self.bitreservoir as u8;
            }
            for k in 2..self.h.ssize {
                self.bsspace[b][wp + k] = 0;
            }
            si.main_data_begin = self.getbits(tab[1]);
        }
        self.bitreservoir = self.bitreservoir + self.h.framesize as u32 - self.h.ssize as u32;
        let cap = if lsf == 0 { 511 } else { 255 };
        if self.bitreservoir > cap {
            self.bitreservoir = cap;
        }
        let _private_bits = self.getbits(if stereo == 1 { tab[2] } else { tab[3] });
        if lsf == 0 {
            for ch in 0..stereo {
                si.ch[ch][0].scfsi = -1;
                si.ch[ch][1].scfsi = self.getbits(4) as i32;
            }
        }
        let bi = &t::BAND_INFO[sfreq];
        for gr in 0..tab[0] as usize {
            for ch in 0..stereo {
                let mut g = si.ch[ch][gr];
                g.part2_3_length = self.getbits(12);
                g.big_values = self.getbits(9).min(288);
                let qss = self.getbits_fast(8) as i32;
                g.pow2gain = 256 - qss + powdiff + if ms_stereo { 2 } else { 0 };
                g.scalefac_compress = self.getbits(tab[4]);
                if g.part2_3_length == 0 {
                    g.scalefac_compress = 0;
                }
                if self.getbits(1) != 0 {
                    g.block_type = self.getbits_fast(2);
                    g.mixed_block_flag = self.get1bit();
                    g.table_select[0] = self.getbits_fast(5);
                    g.table_select[1] = self.getbits_fast(5);
                    g.table_select[2] = 0;
                    for i in 0..3 {
                        let sbg = (self.getbits_fast(3) << 3) as i32;
                        g.full_gain[i] = g.pow2gain + sbg;
                    }
                    if g.block_type == 0 {
                        return Err(());
                    }
                    if (lsf == 0 || g.block_type == 2) && !self.h.mpeg25 {
                        g.region1start = 36 >> 1;
                        g.region2start = 576 >> 1;
                    } else if self.h.mpeg25 {
                        let r0c = if g.block_type == 2 && g.mixed_block_flag == 0 { 5 } else { 7 };
                        let r1c = 20 - r0c;
                        g.region1start = bi.long_idx[r0c + 1] as u32 >> 1;
                        g.region2start = bi.long_idx[r0c + 1 + r1c + 1] as u32 >> 1;
                    } else {
                        g.region1start = 54 >> 1;
                        g.region2start = 576 >> 1;
                    }
                } else {
                    for i in 0..3 {
                        g.table_select[i] = self.getbits_fast(5);
                    }
                    let r0c = self.getbits_fast(4) as usize;
                    let r1c = self.getbits_fast(3) as usize;
                    g.region1start = bi.long_idx[r0c + 1] as u32 >> 1;
                    g.region2start = if r0c + 1 + r1c + 1 > 22 { 576 >> 1 } else { bi.long_idx[r0c + 1 + r1c + 1] as u32 >> 1 };
                    g.block_type = 0;
                    g.mixed_block_flag = 0;
                }
                if lsf == 0 {
                    g.preflag = self.get1bit();
                }
                g.scalefac_scale = self.get1bit();
                g.count1table_select = self.get1bit();
                si.ch[ch][gr] = g;
            }
        }
        Ok(())
    }

    /// `III_get_scale_factors_1`; `None` for "not enough bits".
    fn scale_factors_1(&mut self, scf: &mut [i32; 39], g: &GrInfo) -> Option<i32> {
        const SLEN: [[u32; 16]; 2] = [[0, 0, 0, 0, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4], [0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3]];
        let num0 = SLEN[0][g.scalefac_compress as usize];
        let num1 = SLEN[1][g.scalefac_compress as usize];
        let mut p = 0;
        let numbits;
        if g.block_type == 2 {
            numbits = ((num0 + num1) * 18 - if g.mixed_block_flag != 0 { num0 } else { 0 }) as i32;
            if self.bits_avail < numbits as i64 {
                return None;
            }
            let mut i = 18;
            if g.mixed_block_flag != 0 {
                for _ in 0..8 {
                    scf[p] = self.getbits_fast(num0) as i32;
                    p += 1;
                }
                i = 9;
            }
            for _ in 0..i {
                scf[p] = self.getbits_fast(num0) as i32;
                p += 1;
            }
            for _ in 0..18 {
                scf[p] = self.getbits_fast(num1) as i32;
                p += 1;
            }
            scf[p] = 0;
            scf[p + 1] = 0;
            scf[p + 2] = 0;
        } else {
            let scfsi = g.scfsi;
            if scfsi < 0 {
                numbits = ((num0 + num1) * 10 + num0) as i32;
                if self.bits_avail < numbits as i64 {
                    return None;
                }
                for _ in 0..11 {
                    scf[p] = self.getbits_fast(num0) as i32;
                    p += 1;
                }
                for _ in 0..10 {
                    scf[p] = self.getbits_fast(num1) as i32;
                    p += 1;
                }
                scf[p] = 0;
            } else {
                let nb = |bit: i32, n: u32, k: u32| if scfsi & bit == 0 { n * k } else { 0 };
                numbits = (nb(8, num0, 6) + nb(4, num0, 5) + nb(2, num1, 5) + nb(1, num1, 5)) as i32;
                if self.bits_avail < numbits as i64 {
                    return None;
                }
                for (bit, n, k) in [(8, num0, 6), (4, num0, 5), (2, num1, 5), (1, num1, 5)] {
                    if scfsi & bit == 0 {
                        for _ in 0..k {
                            scf[p] = self.getbits_fast(n) as i32;
                            p += 1;
                        }
                    } else {
                        p += k;
                    }
                }
                scf[p] = 0;
            }
        }
        Some(numbits)
    }

    /// `III_get_scale_factors_2`.
    fn scale_factors_2(&mut self, scf: &mut [i32; 39], g: &mut GrInfo, i_stereo: bool) -> Option<i32> {
        const STAB: [[[u32; 4]; 6]; 3] = [
            [[6, 5, 5, 5], [6, 5, 7, 3], [11, 10, 0, 0], [7, 7, 7, 0], [6, 6, 6, 3], [8, 8, 5, 0]],
            [[9, 9, 9, 9], [9, 9, 12, 6], [18, 18, 0, 0], [12, 12, 12, 0], [12, 9, 9, 6], [15, 12, 9, 0]],
            [[6, 9, 9, 9], [6, 9, 12, 6], [15, 18, 0, 0], [6, 15, 12, 0], [6, 12, 9, 6], [6, 18, 9, 0]],
        ];
        let mut slen = if i_stereo { t::I_SLEN2[(g.scalefac_compress >> 1) as usize] as u32 } else { t::N_SLEN2[g.scalefac_compress as usize] as u32 };
        g.preflag = (slen >> 15) & 1;
        let mut n = 0;
        if g.block_type == 2 {
            n += 1;
            if g.mixed_block_flag != 0 {
                n += 1;
            }
        }
        let pnt = &STAB[n][((slen >> 12) & 7) as usize];
        let mut numbits = 0u32;
        let mut slen2 = slen;
        for &pi in pnt {
            let num = slen2 & 7;
            slen2 >>= 3;
            if num != 0 {
                numbits += pi * num;
            }
        }
        if numbits > g.part2_3_length {
            return None;
        }
        let mut p = 0;
        for &pi in pnt {
            let num = slen & 7;
            slen >>= 3;
            for _ in 0..pi {
                scf[p] = if num != 0 { self.getbits_fast(num) as i32 } else { 0 };
                p += 1;
            }
        }
        for _ in 0..(n << 1) + 1 {
            scf[p] = 0;
            p += 1;
        }
        Some(numbits as i32)
    }
}

/// The `REFRESH_MASK` / `MSB_MASK` state of `III_dequantize_sample`.
struct Mask {
    mask: u32,
    num: i32,
    part2remain: i32,
}

impl Dec<'_> {
    #[inline]
    fn refresh(&mut self, m: &mut Mask) {
        while m.num < 24 {
            m.mask |= self.getbyte().wrapping_shl((24 - m.num) as u32);
            m.num += 8;
            m.part2remain -= 8;
        }
    }

    /// `III_dequantize_sample` (USE_NEW_HUFFTABLE, no CUT_SFB21); `Err` is
    /// mpg123's dequantisation failure.
    fn dequantize(&mut self, xr: &mut [f32; 576], scf: &[i32; 39], g: &mut GrInfo, sfreq: usize, part2bits: i32) -> Result<(), ()> {
        let shift = 1 + g.scalefac_scale;
        let mut xi: usize = 0; // xrpnt
        let mut m = Mask { mask: 0, num: 0, part2remain: g.part2_3_length as i32 - part2bits };
        let ispow = |x: i32| f(t::ISPOW[x as usize]);
        let mut sp = 0usize; // scf pointer
        if g.part2_3_length > 0 {
            let num = self.getbitoffset();
            m.mask = self.getbits(num) << 24;
            m.mask <<= 8 - num;
            m.num = num as i32;
            m.part2remain -= num as i32;
            let bv = g.big_values as i32;
            let region1 = g.region1start as i32;
            let region2 = g.region2start as i32;
            let mut l3 = ((576 >> 1) - bv) >> 1;
            let l: [i32; 3] = if bv <= region1 {
                [bv, 0, 0]
            } else if bv <= region2 {
                [region1, bv - region1, 0]
            } else {
                [region1, region2 - region1, bv - region2]
            };
            let msb = |mask: u32| mask & 0x8000_0000 != 0;

            if g.block_type == 2 {
                let mut max = [-1i32; 4];
                let (mut step, mut lwin, mut cb) = (0usize, 3usize, 0i32);
                let mut v = 0f32;
                let (map, base, end): (&[i16], usize, usize) = if g.mixed_block_flag != 0 {
                    max = [2, 2, 2, -1];
                    (&t::MAPBUF0, sfreq * 152, sfreq * 152 + if sfreq < 3 { 152 } else { 144 })
                } else {
                    (&t::MAPBUF1, sfreq * 156, sfreq * 156 + 156)
                };
                let mut mi = base;
                let mut mc = 0i32;
                macro_rules! next_band {
                    () => {{
                        mc = map[mi] as i32;
                        xi = map[mi + 1] as usize;
                        lwin = map[mi + 2] as usize;
                        cb = map[mi + 3] as i32;
                        mi += 4;
                        if lwin == 3 {
                            v = self.gainpow2[(g.pow2gain + (scf[sp] << shift)) as usize];
                            step = 1;
                        } else {
                            v = self.gainpow2[(g.full_gain[lwin] + (scf[sp] << shift)) as usize];
                            step = 3;
                        }
                        sp += 1;
                    }};
                }
                for i in 0..2 {
                    let mut lp = l[i];
                    let (linbits, table) = t::HT[g.table_select[i] as usize];
                    while lp > 0 {
                        if mc == 0 {
                            next_band!();
                        }
                        let (mut x, mut y) = self.huff_pair(&mut m, table);
                        for val in [&mut x, &mut y] {
                            if xi >= 576 {
                                return Err(());
                            }
                            if *val == 15 && linbits != 0 {
                                max[lwin] = cb;
                                self.refresh(&mut m);
                                *val += (m.mask >> (32 - linbits)) as i32;
                                m.num -= linbits as i32 + 1;
                                m.mask <<= linbits;
                                xr[xi] = if msb(m.mask) { -ispow(*val) * v } else { ispow(*val) * v };
                                m.mask <<= 1;
                            } else if *val != 0 {
                                max[lwin] = cb;
                                xr[xi] = if msb(m.mask) { -ispow(*val) * v } else { ispow(*val) * v };
                                m.num -= 1;
                                m.mask <<= 1;
                            } else {
                                xr[xi] = 0.0;
                            }
                            xi += step;
                        }
                        lp -= 1;
                        mc -= 1;
                    }
                }
                while l3 > 0 && m.part2remain + m.num > 0 {
                    let a = self.huff_count1(&mut m, t::HTC[g.count1table_select as usize].1);
                    if m.part2remain + m.num <= 0 {
                        m.num -= m.part2remain + m.num;
                        break;
                    }
                    for i in 0..4 {
                        if i & 1 == 0 {
                            if mc == 0 {
                                next_band!();
                            }
                            mc -= 1;
                        }
                        if xi >= 576 {
                            return Err(());
                        }
                        if a & (0x8 >> i) != 0 {
                            max[lwin] = cb;
                            if m.part2remain + m.num <= 0 {
                                break;
                            }
                            xr[xi] = if msb(m.mask) { -v } else { v };
                            m.num -= 1;
                            m.mask <<= 1;
                        } else {
                            xr[xi] = 0.0;
                        }
                        xi += step;
                    }
                    l3 -= 1;
                }
                if lwin < 3 {
                    loop {
                        while mc > 0 {
                            if xi >= 576 {
                                return Err(());
                            }
                            xr[xi] = 0.0;
                            xi += 3;
                            if xi < 576 {
                                xr[xi] = 0.0;
                            }
                            xi += 3;
                            mc -= 1;
                        }
                        if mi >= end {
                            break;
                        }
                        mc = map[mi] as i32;
                        xi = map[mi + 1] as usize;
                        let lw = map[mi + 2];
                        mi += 3;
                        if lw == 0 {
                            break;
                        }
                        mi += 1;
                    }
                }
                g.maxband = [max[0] + 1, max[1] + 1, max[2] + 1];
                g.maxbandl = max[3] + 1;
                let rmax = max[0].max(max[1]).max(max[2]) + 1;
                g.maxb = if rmax != 0 { self.short_limit[sfreq][rmax as usize] } else { self.long_limit[sfreq][(max[3] + 1) as usize] } as u32;
            } else {
                let pretab: &[u8] = if g.preflag != 0 { &t::PRETAB[22..] } else { &t::PRETAB[..22] };
                let mut pp = 0usize;
                let mut max = -1i32;
                let mut cb = 0i32;
                let map = &t::MAPBUF2;
                let mut mi = sfreq * 44;
                let mut v = 0f32;
                let mut mc = 0i32;
                macro_rules! next_band {
                    () => {{
                        mc = map[mi] as i32;
                        cb = map[mi + 1] as i32;
                        mi += 2;
                        v = self.gainpow2[(g.pow2gain + ((scf[sp] + pretab[pp] as i32) << shift)) as usize];
                        sp += 1;
                        pp += 1;
                    }};
                }
                for i in 0..3 {
                    let mut lp = l[i];
                    let (linbits, table) = t::HT[g.table_select[i] as usize];
                    while lp > 0 {
                        if mc == 0 {
                            next_band!();
                        }
                        let (mut x, mut y) = self.huff_pair(&mut m, table);
                        for val in [&mut x, &mut y] {
                            if xi >= 576 {
                                return Err(());
                            }
                            if *val == 15 && linbits != 0 {
                                max = cb;
                                self.refresh(&mut m);
                                *val += (m.mask >> (32 - linbits)) as i32;
                                m.num -= linbits as i32 + 1;
                                m.mask <<= linbits;
                                xr[xi] = if msb(m.mask) { -ispow(*val) * v } else { ispow(*val) * v };
                                m.mask <<= 1;
                            } else if *val != 0 {
                                max = cb;
                                xr[xi] = if msb(m.mask) { -ispow(*val) * v } else { ispow(*val) * v };
                                m.num -= 1;
                                m.mask <<= 1;
                            } else {
                                xr[xi] = 0.0;
                            }
                            xi += 1;
                        }
                        lp -= 1;
                        mc -= 1;
                    }
                }
                while l3 > 0 && m.part2remain + m.num > 0 {
                    let a = self.huff_count1(&mut m, t::HTC[g.count1table_select as usize].1);
                    if m.part2remain + m.num <= 0 {
                        m.num -= m.part2remain + m.num;
                        break;
                    }
                    for i in 0..4 {
                        if i & 1 == 0 {
                            if mc == 0 {
                                next_band!();
                            }
                            mc -= 1;
                        }
                        if xi >= 576 {
                            return Err(());
                        }
                        if a & (0x8 >> i) != 0 {
                            max = cb;
                            if m.part2remain + m.num <= 0 {
                                break;
                            }
                            xr[xi] = if msb(m.mask) { -v } else { v };
                            m.num -= 1;
                            m.mask <<= 1;
                        } else {
                            xr[xi] = 0.0;
                        }
                        xi += 1;
                    }
                    l3 -= 1;
                }
                g.maxbandl = max + 1;
                g.maxb = self.long_limit[sfreq][g.maxbandl as usize] as u32;
            }
            m.part2remain += m.num;
            self.backbits(m.num as i64);
        } else {
            m.part2remain = 0;
            g.maxband = [1, 1, 1];
            g.maxbandl = 1;
            g.maxb = 1;
        }
        while xi < 576 {
            xr[xi] = 0.0;
            xi += 1;
        }
        while m.part2remain > 16 {
            self.skipbits(16);
            m.part2remain -= 16;
        }
        if m.part2remain > 0 {
            self.skipbits(m.part2remain as u32);
        } else if m.part2remain < 0 {
            return Err(());
        }
        Ok(())
    }

    /// One big-value pair from a radix-4 `newhuff` table.
    #[inline]
    fn huff_pair(&mut self, m: &mut Mask, table: &[i16]) -> (i32, i32) {
        self.refresh(m);
        let mut vi = 0usize;
        let mut y;
        loop {
            y = table[vi + (m.mask >> 28) as usize] as i32;
            if y >= 0 {
                break;
            }
            vi = (vi as i32 - y) as usize;
            m.num -= 4;
            m.mask <<= 4;
        }
        m.num -= y >> 8;
        m.mask = m.mask.wrapping_shl((y >> 8) as u32);
        ((y >> 4) & 0xf, y & 0xf)
    }

    /// A count1 quadruple from a bit-serial table.
    #[inline]
    fn huff_count1(&mut self, m: &mut Mask, table: &[i16]) -> i32 {
        self.refresh(m);
        let mut vi = 0usize;
        loop {
            let a = table[vi] as i32;
            vi += 1;
            if a >= 0 {
                return a;
            }
            if m.mask & 0x8000_0000 != 0 {
                vi = (vi as i32 - a) as usize;
            }
            m.num -= 1;
            m.mask <<= 1;
        }
    }
}

impl Dec<'_> {
    /// `III_i_stereo`.
    fn i_stereo(xr: &mut [[f32; 576]; 2], scalefac: &[i32; 39], g: &GrInfo, sfreq: usize, ms_stereo: bool, lsf: usize) {
        let bi = &t::BAND_INFO[sfreq];
        let tab = lsf + (g.scalefac_compress as usize & lsf);
        let ms = ms_stereo as usize;
        let (tab1, tab2): (&[u32], &[u32]) = match (tab, ms) {
            (0, 0) => (&t::TAN1_1, &t::TAN2_1),
            (0, _) => (&t::TAN1_2, &t::TAN2_2),
            (1, 0) => (&t::POW1_1[..32], &t::POW2_1[..32]),
            (1, _) => (&t::POW1_2[..32], &t::POW2_2[..32]),
            (_, 0) => (&t::POW1_1[32..], &t::POW2_1[32..]),
            (_, _) => (&t::POW1_2[32..], &t::POW2_2[32..]),
        };
        let mut apply = |idx: &mut usize, sb: usize, step: usize, is_p: i32| {
            let (t1, t2) = (f(tab1[is_p as usize]), f(tab2[is_p as usize]));
            for _ in 0..sb {
                let v = xr[0][*idx];
                xr[0][*idx] = v * t1;
                xr[1][*idx] = v * t2;
                *idx += step;
            }
        };
        let mixed = g.mixed_block_flag as i32;
        if g.block_type == 2 {
            let mut do_l = g.mixed_block_flag != 0;
            for lwin in 0..3 {
                let mut sfb = g.maxband[lwin];
                if sfb > 3 {
                    do_l = false;
                }
                while sfb < 12 {
                    let is_p = scalefac[(sfb * 3 + lwin as i32 - mixed) as usize];
                    if is_p != 7 {
                        let mut idx = bi.short_idx[sfb as usize] as usize + lwin;
                        apply(&mut idx, bi.short_diff[sfb as usize] as usize, 3, is_p);
                    }
                    sfb += 1;
                }
                let is_p = scalefac[(11 * 3 + lwin as i32 - mixed) as usize];
                if is_p != 7 {
                    let mut idx = bi.short_idx[12] as usize + lwin;
                    apply(&mut idx, bi.short_diff[12] as usize, 3, is_p);
                }
            }
            if do_l {
                let mut sfb = g.maxbandl;
                if sfb > 21 {
                    return;
                }
                let mut idx = bi.long_idx[sfb as usize] as usize;
                while sfb < 8 {
                    let sb = bi.long_diff[sfb as usize] as usize;
                    let is_p = scalefac[sfb as usize];
                    if is_p != 7 {
                        apply(&mut idx, sb, 1, is_p);
                    } else {
                        idx += sb;
                    }
                    sfb += 1;
                }
            }
        } else {
            let mut sfb = g.maxbandl;
            if sfb > 21 {
                return;
            }
            let mut idx = bi.long_idx[sfb as usize] as usize;
            while sfb < 21 {
                let sb = bi.long_diff[sfb as usize] as usize;
                let is_p = scalefac[sfb as usize];
                if is_p != 7 {
                    apply(&mut idx, sb, 1, is_p);
                } else {
                    idx += sb;
                }
                sfb += 1;
            }
            let is_p = scalefac[20];
            if is_p != 7 {
                apply(&mut idx, bi.long_diff[21] as usize, 1, is_p);
            }
        }
    }

    /// `III_antialias` through the vectorised loop of the reference's
    /// `INT123_do_layer3`.
    fn antialias(&mut self, ch: usize, g: &GrInfo) {
        let sblim = if g.block_type == 2 {
            if g.mixed_block_flag == 0 {
                return;
            }
            1
        } else {
            g.maxb as i64 - 1
        };
        if sblim <= 0 {
            return;
        }
        self.cpu.x[8] = sblim as u64;
        self.cpu.x[9] = HYBRID_IN + (ch * 576 * 4) as u64 + 22 * 4;
        built::antialias_loop(&mut self.cpu);
    }

    /// `III_hybrid` with `dct36_neon64` and the C `dct12`.
    fn hybrid(&mut self, ch: usize, g: &GrInfo) {
        let b = self.hybrid_blc[ch];
        let block = |b: usize| HYBRID_BLOCK + ((b * 2 + ch) * 576 * 4) as u64;
        let mut raw1 = block(b);
        let mut raw2 = block(1 - b);
        self.hybrid_blc[ch] = 1 - b;
        let fs_in = HYBRID_IN + (ch * 576 * 4) as u64;
        let mut ts = HYBRID_OUT + (ch * 576 * 4) as u64;
        let mut sb = 0u32;
        let c = &mut self.cpu;
        let call = |c: &mut Cpu, k: fn(&mut Cpu), inp: u64, o1: u64, o2: u64, w: u64, ts: u64| {
            c.x[0] = inp;
            c.x[1] = o1;
            c.x[2] = o2;
            c.x[3] = w;
            c.x[4] = ts;
            k(c);
        };
        let row = |sb: u32| fs_in + (sb as u64) * 18 * 4;
        if g.mixed_block_flag != 0 {
            sb = 2;
            call(c, built::INT123_dct36_neon64, row(0), raw1, raw2, WIN, ts);
            call(c, built::INT123_dct36_neon64, row(1), raw1 + 72, raw2 + 72, WIN1, ts + 4);
            raw1 += 144;
            raw2 += 144;
            ts += 8;
        }
        let bt = g.block_type as u64;
        while sb < g.maxb {
            if bt == 2 {
                call(c, built::dct12, row(sb), raw1, raw2, WIN + 2 * 144, ts);
                call(c, built::dct12, row(sb + 1), raw1 + 72, raw2 + 72, WIN1 + 2 * 144, ts + 4);
            } else {
                call(c, built::INT123_dct36_neon64, row(sb), raw1, raw2, WIN + bt * 144, ts);
                call(c, built::INT123_dct36_neon64, row(sb + 1), raw1 + 72, raw2 + 72, WIN1 + bt * 144, ts + 4);
            }
            sb += 2;
            ts += 8;
            raw1 += 144;
            raw2 += 144;
        }
        while sb < 32 {
            for i in 0..18u64 {
                let v = c.ld32(raw1);
                c.st32(ts + i * 32 * 4, v);
                c.st32(raw2, 0);
                raw1 += 4;
                raw2 += 4;
            }
            sb += 1;
            ts += 4;
        }
    }

    /// `INT123_synth_1to1_real_neon64` through the `synth_mono` wrapper:
    /// 32 samples into `out`.
    fn synth_mono(&mut self, band: u64) {
        self.bo = self.bo.wrapping_sub(1) & 0xf;
        let buf = |j: u64| REAL_BUFFS + j * 0x110 * 4;
        let c = &mut self.cpu;
        let bo = self.bo as u64;
        let (b0, bo1) = if bo & 1 != 0 {
            c.x[0] = buf(1) + ((bo + 1) & 0xf) * 4;
            c.x[1] = buf(0) + bo * 4;
            (buf(0), bo)
        } else {
            c.x[0] = buf(0) + bo * 4;
            c.x[1] = buf(1) + (bo + 1) * 4;
            (buf(1), bo + 1)
        };
        c.x[2] = band;
        built::INT123_dct64_real_neon64(c);
        c.x[0] = DECWIN;
        c.x[1] = b0;
        c.x[2] = MONO_TMP;
        c.x[3] = bo1;
        built::INT123_synth_1to1_real_neon64_asm(c);
        for i in 0..32 {
            self.out.push(f(self.cpu.ld32(MONO_TMP + 8 * i)));
        }
    }

    /// `INT123_synth_1to1_fltst_neon64`: 32 interleaved stereo samples.
    fn synth_stereo(&mut self, band_l: u64, band_r: u64) {
        self.bo = self.bo.wrapping_sub(1) & 0xf;
        let buf = |ch: u64, j: u64| REAL_BUFFS + (ch * 2 + j) * 0x110 * 4;
        let c = &mut self.cpu;
        let bo = self.bo as u64;
        let (j0, j1, bo1, a, b) = if bo & 1 != 0 { (0, 1, bo, (bo + 1) & 0xf, bo) } else { (1, 0, bo + 1, bo, bo + 1) };
        for (ch, band) in [(0, band_l), (1, band_r)] {
            if bo & 1 != 0 {
                c.x[0] = buf(ch, 1) + a * 4;
                c.x[1] = buf(ch, 0) + b * 4;
            } else {
                c.x[0] = buf(ch, 0) + a * 4;
                c.x[1] = buf(ch, 1) + b * 4;
            }
            c.x[2] = band;
            built::INT123_dct64_real_neon64(c);
        }
        let _ = j1;
        c.x[0] = DECWIN;
        c.x[1] = buf(0, j0);
        c.x[2] = buf(1, j0);
        c.x[3] = SAMPLES;
        c.x[4] = bo1;
        built::INT123_synth_1to1_real_s_neon64_asm(c);
        for i in 0..64 {
            self.out.push(f(self.cpu.ld32(SAMPLES + 4 * i)));
        }
    }
}

impl Dec<'_> {
    /// `INT123_do_layer3` for libsndfile's setup (no forced mono); appends
    /// to `out` whatever the frame produced before any error.
    fn do_layer3(&mut self) {
        let stereo = self.h.stereo;
        let single = SINGLE_STEREO;
        let sfreq = self.h.sfreq;
        let (ms_stereo, i_stereo) = if self.h.mode == 1 { (self.h.mode_ext & 2 != 0, self.h.mode_ext & 1 != 0) } else { (false, false) };
        let granules = if self.h.lsf != 0 { 1 } else { 2 };
        let mut si = SideInfo::default();
        if self.get_side_info(&mut si, stereo, ms_stereo, sfreq).is_err() {
            return;
        }
        self.set_pointer(true, si.main_data_begin as usize);
        let mut scalefacs = [[0i32; 39]; 2];
        let mut xr = Box::new([[0f32; 576]; 2]);
        for gr in 0..granules {
            {
                let mut g = si.ch[0][gr];
                if g.part2_3_length as i64 > self.bits_avail {
                    return;
                }
                let part2bits = if self.h.lsf != 0 { self.scale_factors_2(&mut scalefacs[0], &mut g, false) } else { self.scale_factors_1(&mut scalefacs[0], &g) };
                let Some(part2bits) = part2bits else { return };
                let r = self.dequantize(&mut xr[0], &scalefacs[0], &mut g, sfreq, part2bits);
                si.ch[0][gr] = g;
                if r.is_err() || self.bits_avail < 0 {
                    return;
                }
            }
            if stereo == 2 {
                let mut g = si.ch[1][gr];
                let part2bits = if self.h.lsf != 0 { self.scale_factors_2(&mut scalefacs[1], &mut g, i_stereo) } else { self.scale_factors_1(&mut scalefacs[1], &g) };
                let Some(part2bits) = part2bits else { return };
                let r = self.dequantize(&mut xr[1], &scalefacs[1], &mut g, sfreq, part2bits);
                si.ch[1][gr] = g;
                if r.is_err() || self.bits_avail < 0 {
                    return;
                }
                if ms_stereo {
                    let maxb = si.ch[0][gr].maxb.max(si.ch[1][gr].maxb) as usize;
                    for i in 0..18 * maxb {
                        let (a, b) = (xr[0][i], xr[1][i]);
                        xr[0][i] = a + b;
                        xr[1][i] = a - b;
                    }
                }
                if i_stereo {
                    let g1 = si.ch[1][gr];
                    Self::i_stereo(&mut xr, &scalefacs[1], &g1, sfreq, ms_stereo, self.h.lsf);
                }
                if ms_stereo || i_stereo {
                    if si.ch[1][gr].maxb > si.ch[0][gr].maxb {
                        si.ch[0][gr].maxb = si.ch[1][gr].maxb;
                    } else {
                        si.ch[1][gr].maxb = si.ch[0][gr].maxb;
                    }
                }
            }
            let stereo1 = if stereo == 1 || single != SINGLE_STEREO { 1 } else { 2 };
            for ch in 0..stereo1 {
                self.cpu.put(HYBRID_IN + (ch * 576 * 4) as u64, &xr[ch]);
                let g = si.ch[ch][gr];
                self.antialias(ch, &g);
                self.hybrid(ch, &g);
            }
            for ss in 0..18u64 {
                let a = HYBRID_OUT + ss * 32 * 4;
                if stereo1 == 1 {
                    self.synth_mono(a);
                } else {
                    self.synth_stereo(a, a + 576 * 4);
                }
            }
        }
    }

    // ---- parse.c ------------------------------------------------------------

    fn head_at(&self, pos: usize) -> Option<u32> {
        let b = self.data.get(pos..pos + 4)?;
        Some(u32::from_be_bytes(b.try_into().unwrap()))
    }

    /// `decode_header` at `pos`, guessing a free-format frame size from the
    /// next matching header (`guess_freeformat_framesize`).
    fn header(&mut self, pos: usize, h: u32) -> Option<Header> {
        if !head_check(h) {
            return None;
        }
        if (h >> 12) & 0xf == 0 && self.freeformat_framesize.is_none() {
            const SAMEMASK: u32 = 0xfffe_fcc0;
            let i = (4..MAXFRAMESIZE + 4).find(|&i| self.head_at(pos + i + 1).is_some_and(|n| n & SAMEMASK == h & SAMEMASK))?;
            self.freeformat_framesize = Some(i - 3 - ((h >> 9) & 1) as usize);
        }
        decode_header(h, self.freeformat_framesize).ok()
    }

    /// The first header: skip ID3v2 tags and junk, accept a header whose
    /// successor (if any) is compatible (`skip_junk` + `do_readahead`).
    fn sync_first(&mut self) -> Option<(u32, Header)> {
        let mut pos = self.filepos;
        let mut junk = 0;
        loop {
            let d = self.data.get(pos..)?;
            if d.len() >= 10 && &d[..3] == b"ID3" && d[3] != 0xff && d[4] != 0xff && d[6..10].iter().all(|b| b & 0x80 == 0) {
                let size = d[6..10].iter().fold(0usize, |a, &b| (a << 7) | b as usize);
                pos += 10 + size + if d[5] & 0x10 != 0 { 10 } else { 0 };
                continue;
            }
            let h = self.head_at(pos)?;
            if head_check(h) {
                if let Some(hd) = self.header(pos, h) {
                    let next = pos + 4 + hd.framesize;
                    match self.head_at(next) {
                        Some(nh) if !(head_check(nh) && head_compatible(h, nh)) => {}
                        _ => {
                            self.filepos = pos + 4;
                            return Some((h, hd));
                        }
                    }
                }
            }
            pos += 1;
            junk += 1;
            if junk >= 65536 {
                return None;
            }
        }
    }

    /// A later header: resync forward to a compatible one.
    fn sync_next(&mut self) -> Option<(u32, Header)> {
        let mut pos = self.filepos;
        for _ in 0..=1024 {
            let h = self.head_at(pos)?;
            if head_check(h) {
                if let Some(hd) = self.header(pos, h) {
                    if pos == self.filepos || head_compatible(self.firsthead, h) {
                        self.filepos = pos + 4;
                        return Some((h, hd));
                    }
                }
            }
            pos += 1;
        }
        None
    }

    /// `INT123_read_frame`; false at the end of the stream.
    fn read_frame(&mut self) -> bool {
        self.fsizeold = self.h.framesize;
        self.to_decode = false;
        self.to_ignore = false;
        if (self.track_frames > 0 && self.num >= self.track_frames - 1) || (self.gapless_frames > 0 && self.num >= self.gapless_frames - 1) {
            return false;
        }
        let mut framepos;
        loop {
            let found = if self.firsthead == 0 { self.sync_first() } else { self.sync_next() };
            let Some((newhead, hd)) = found else { return false };
            if self.firsthead != 0 && !head_compatible(self.firsthead, newhead) {
                return false;
            }
            framepos = self.filepos - 4;
            let Some(body) = self.data.get(self.filepos..self.filepos + hd.framesize) else { return false };
            self.h = hd;
            let n = self.bsnum;
            self.bsspace[n][512..512 + hd.framesize].copy_from_slice(body);
            self.filepos += hd.framesize;
            self.bsbufold = self.bsbuf;
            self.bsbuf = (n, 512);
            self.bsnum = (self.bsnum + 1) & 1;
            if self.firsthead == 0 {
                self.firsthead = newhead;
                if self.num < 0 {
                    if self.check_lame_tag() {
                        self.oldhead = 0;
                        continue;
                    }
                }
            }
            break;
        }
        self.set_pointer(false, 0);
        self.mean_frames += 1;
        self.mean_framesize = ((self.mean_frames - 1) as f64 * self.mean_framesize + (self.h.framesize as f64 + 4.0)) / self.mean_frames as f64;
        self.num += 1;
        if self.num as usize == self.index.len() {
            self.index.push(framepos);
        }
        self.to_decode = true;
        self.to_ignore = true;
        if self.h.error_protection {
            let _crc = self.getbits(16);
        }
        if self.header_change < 2 {
            self.header_change = 2;
        }
        true
    }

    /// `check_lame_tag`: Xing/Info frame count and LAME delay/padding.
    fn check_lame_tag(&mut self) -> bool {
        let h = self.h;
        let mut lo = if h.stereo == 2 { if h.lsf != 0 { 17 } else { 32 } } else if h.lsf != 0 { 9 } else { 17 };
        if h.framesize < lo + 8 {
            return false;
        }
        let (b, off) = self.bsbuf;
        let buf = self.bsspace[b][off..off + h.framesize].to_vec();
        if buf[2..lo].iter().any(|&x| x != 0) {
            return false;
        }
        if &buf[lo..lo + 4] != b"Info" && &buf[lo..lo + 4] != b"Xing" {
            return false;
        }
        let rd = |o: &mut usize| {
            let v = u32::from_be_bytes(buf[*o..*o + 4].try_into().unwrap());
            *o += 4;
            v
        };
        lo += 4;
        let flags = rd(&mut lo);
        'tag: {
            let left = |o: usize, n: usize| h.framesize >= o + n;
            if flags & 1 != 0 {
                if !left(lo, 4) {
                    break 'tag;
                }
                let frames = rd(&mut lo) as i64;
                self.track_frames = if frames > (u32::MAX as i64) / 4 / 1152 { 0 } else { frames };
                self.gapless_init(self.track_frames, 0, 0);
            }
            if flags & 2 != 0 {
                if !left(lo, 4) {
                    break 'tag;
                }
                lo += 4;
            }
            if flags & 4 != 0 {
                if !left(lo, 100) {
                    break 'tag;
                }
                lo += 100;
            }
            if flags & 8 != 0 {
                if !left(lo, 4) {
                    break 'tag;
                }
                lo += 4;
            }
            if !left(lo, 24) {
                break 'tag;
            }
            if buf[lo] != 0 {
                let o = lo + 21;
                let pad_in = ((buf[o] as i64) << 4) | (buf[o + 1] as i64 >> 4);
                let pad_out = (((buf[o + 1] as i64) << 8) | buf[o + 2] as i64) & 0xfff;
                self.gapless_init(self.track_frames, pad_in, pad_out);
            }
        }
        self.bsbuf = (self.bsnum, 512);
        self.bsnum = (self.bsnum + 1) & 1;
        true
    }

    // ---- frame.c gapless ------------------------------------------------------

    fn gapless_init(&mut self, framecount: i64, bskip: i64, eskip: i64) {
        self.gapless_frames = framecount;
        if framecount > 0 && bskip >= 0 && eskip >= 0 {
            self.begin_s = bskip + GAPLESS_DELAY;
            self.end_s = framecount * self.h.spf as i64 - eskip + GAPLESS_DELAY;
        } else {
            self.begin_s = 0;
            self.end_s = 0;
        }
        self.begin_os = 0;
        self.end_os = 0;
        self.fullend_os = 0;
    }

    fn gapless_realinit(&mut self) {
        self.begin_os = self.begin_s;
        self.end_os = self.end_s;
        self.fullend_os = if self.gapless_frames > 0 { self.gapless_frames * self.h.spf as i64 } else { 0 };
    }

    fn set_frameseek(&mut self, fe: i64) {
        let spf = self.h.spf as i64;
        self.firstframe = fe;
        if self.gapless_frames > 0 {
            let beg_f = self.begin_os / spf;
            if fe <= beg_f {
                self.firstframe = beg_f;
                self.firstoff = self.begin_os - beg_f * spf;
            } else {
                self.firstoff = 0;
            }
            if self.end_os > 0 {
                self.lastframe = self.end_os / spf;
                self.lastoff = self.end_os - self.lastframe * spf;
            } else {
                self.lastframe = -1;
                self.lastoff = 0;
            }
        } else {
            self.firstoff = 0;
            self.lastoff = 0;
            self.lastframe = -1;
        }
        self.ignoreframe = self.firstframe - PREFRAMES;
    }

    /// Decode a frame and throw its output away (an ignored frame).
    fn decode_discard(&mut self) {
        let keep = std::mem::take(&mut self.out);
        self.do_layer3();
        self.out = keep;
    }

    /// `get_next_frame`; false at the end.
    fn get_next_frame(&mut self) -> bool {
        let mut change = false;
        if self.header_change > 1 && self.num >= 0 {
            change = true;
            self.header_change = 0;
        }
        loop {
            if self.to_ignore && self.num < self.firstframe && self.num >= self.ignoreframe {
                self.decode_discard();
                self.to_ignore = false;
                self.to_decode = false;
            }
            self.to_decode = false;
            if !self.read_frame() {
                self.track_frames = self.num + 1;
                return false;
            }
            if self.header_change > 1 {
                change = true;
                self.header_change = 0;
            }
            if self.num < self.firstframe {
                if !(self.to_ignore && self.num < self.firstframe && self.num >= self.ignoreframe) {
                    // `INT123_frame_skip`
                    self.set_pointer(true, 512);
                }
            } else {
                break;
            }
        }
        if change && self.fresh {
            self.gapless_realinit();
            self.set_frameseek(self.num);
            self.fresh = false;
            if self.num < self.firstframe {
                return self.get_next_frame();
            }
        }
        true
    }

    /// `mpg123_seek(0, SEEK_SET)`, which `soundfile.read` issues (via
    /// `sf_seek`) before reading: `INT123_frame_set_seek` + `do_the_seek`.
    fn seek_start(&mut self) {
        let spf = self.h.spf as i64;
        let mut sp = self.begin_os;
        if sp >= self.end_os {
            sp += self.fullend_os - self.end_os;
        }
        self.firstframe = sp / spf;
        self.ignoreframe = self.firstframe - PREFRAMES;
        self.firstoff = sp - self.firstframe * spf;
        let fnum = self.ignoreframe.max(0);
        if self.num < self.firstframe {
            self.to_decode = false;
            if self.num > fnum {
                return;
            }
        }
        if self.num == fnum && (self.to_decode || fnum < self.firstframe) {
            return;
        }
        if self.num == fnum - 1 {
            self.to_decode = false;
            return;
        }
        // `INT123_frame_buffers_reset` (`bo` is kept).
        self.bsnum = 0;
        self.bsbuf = (1, 0);
        self.bsbufold = (1, 0);
        self.bitreservoir = 0;
        self.cpu.put(REAL_BUFFS, &[0.0; 4 * 0x110]);
        for b in &mut self.bsspace {
            b[..MAXFRAMESIZE + 512].fill(0);
        }
        self.hybrid_blc = [0, 0];
        self.cpu.put(HYBRID_BLOCK, &[0.0; 4 * 576]);
        // `stream_seek_frame` from the frame index.
        let Some(&pos) = self.index.get(fnum as usize) else { return };
        // (`num >= newframe || num < preframe` always holds with step 1.)
        self.filepos = pos;
        self.num = fnum - 1;
        while self.num < fnum {
            if !self.read_frame() {
                break;
            }
        }
        if self.num < self.firstframe {
            self.to_decode = false;
        }
    }

    /// `decode_the_frame` + `frame_buffercheck`: the frame's kept samples.
    fn decode_the_frame(&mut self) -> Vec<f32> {
        let ch = self.h.stereo;
        let needed = self.h.spf * ch;
        self.out.clear();
        self.do_layer3();
        self.out.resize(needed, 0.0);
        let mut fill = needed;
        let mut start = 0;
        if !(self.gapless_frames > 0 && self.num >= self.gapless_frames) {
            if self.lastframe > -1 && self.num >= self.lastframe {
                let byteoff = if self.num == self.lastframe { self.lastoff as usize * ch } else { 0 };
                fill = fill.min(byteoff);
            }
            if self.firstoff != 0 && self.num == self.firstframe {
                let off = self.firstoff as usize * ch;
                if fill > off {
                    fill -= off;
                    start = off;
                } else {
                    fill = 0;
                }
                self.firstoff = 0;
            }
        }
        self.out[start..start + fill].to_vec()
    }
}

/// An MP3 stream decoded as `soundfile` (libsndfile 1.2.2 + mpg123 1.32.3,
/// arm64) returns it.
pub struct Mp3 {
    /// Samples per second.
    pub sample_rate: u32,
    /// Channel count.
    pub channels: usize,
    /// Interleaved samples.
    pub samples: Vec<f32>,
}

/// Whether `data` holds MPEG layer III frames: two consecutive compatible
/// headers within mpg123's 64 KiB junk window (after any ID3v2 tag).
pub fn probe(data: &[u8]) -> bool {
    let mut d = Dec::new(data);
    let Some((h0, h)) = d.sync_first() else { return false };
    d.head_at(d.filepos + h.framesize).is_some_and(|n| head_check(n) && head_compatible(h0, n))
}

/// Decode a whole MP3 (layer III) file.
pub fn decode(data: &[u8]) -> Result<Mp3, String> {
    let mut d = Dec::new(data);
    // `init_track` via `mpg123_length`.
    if !d.get_next_frame() {
        return Err("no MPEG audio frame found".into());
    }
    let spf = d.h.spf as i64;
    let x = if d.track_frames > 0 {
        d.track_frames * spf
    } else if d.filelen > 0 {
        let bpf = if d.mean_framesize != 0.0 { d.mean_framesize } else { d.h.framesize as f64 + 4.0 };
        (d.filelen as f64 / bpf * spf as f64) as i64
    } else {
        return Err("MP3 length unknown".into());
    };
    // `sample_adjust`
    let length = if x > d.end_os {
        if x < d.fullend_os {
            d.end_os - d.begin_os
        } else {
            x - (d.fullend_os - d.end_os + d.begin_os)
        }
    } else {
        x - d.begin_os
    };
    d.seek_start();
    let channels = d.h.stereo;
    let sample_rate = FREQS[d.h.sfreq];
    let want = length.max(0) as usize * channels;
    let mut samples = Vec::with_capacity(want);
    loop {
        if d.to_decode {
            let frame = d.decode_the_frame();
            d.to_decode = false;
            d.to_ignore = false;
            samples.extend_from_slice(&frame);
            if samples.len() >= want {
                break;
            }
        }
        if !d.get_next_frame() {
            break;
        }
    }
    samples.truncate(want);
    Ok(Mp3 { sample_rate, channels, samples })
}
