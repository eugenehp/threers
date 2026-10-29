//! AAC-LC raw_data_block decoding, after ffmpeg n7.1 `libavcodec/aac/aacdec.c`,
//! `aacdec_proc_template.c` and `aacdec_dsp_template.c` (float decoder).
//!
//! Supported: AAC-LC cores (object type 2, alone or under SBR/PS), 1024-
//! sample frames, one SCE or CPE (channel configuration 1 or 2, or a PCE
//! naming just that element); SBR and PS run through [`super::sbr`]. Main/LTP
//! prediction, coupling (CCE), larger layouts and 960-sample frames are
//! refused with [`AacError::Unsupported`].

use super::bits::BitReader;
use super::dsp::{scalarproduct, tables, vector_fmul_window};
use super::tables as t;
use super::vlc;

/// Why an AAC stream could not be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AacError {
    /// Valid AAC using something this decoder does not implement.
    Unsupported(String),
    /// A malformed stream.
    Invalid(String),
}

impl std::fmt::Display for AacError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AacError::Unsupported(s) => write!(f, "unsupported AAC: {s}"),
            AacError::Invalid(s) => write!(f, "invalid AAC: {s}"),
        }
    }
}

impl std::error::Error for AacError {}

fn invalid<T>(s: &str) -> Result<T, AacError> {
    Err(AacError::Invalid(s.into()))
}

fn unsupported<T>(s: impl Into<String>) -> Result<T, AacError> {
    Err(AacError::Unsupported(s.into()))
}

/// Sampling rates by `samplingFrequencyIndex` (ISO/IEC 14496-3).
pub const SAMPLE_RATES: [u32; 13] = [96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350];

/// The parts of an `AudioSpecificConfig` the decoder uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioSpecificConfig {
    /// The core object type (2, AAC-LC; an explicit SBR/PS type 5/29 is
    /// recorded in `sbr`/`ps`).
    pub object_type: u32,
    /// `samplingFrequencyIndex`.
    pub sampling_index: usize,
    /// The core sampling rate.
    pub sample_rate: u32,
    /// `channelConfiguration`.
    pub chan_config: u32,
    /// Output channels.
    pub channels: usize,
    /// `-1` unknown (implicit SBR possible), `0` absent, `1` present.
    pub sbr: i32,
    /// Parametric stereo: `-1` unknown, `0` absent, `1` present.
    pub ps: i32,
    /// The SBR output rate when signalled (0 when not).
    pub ext_sample_rate: u32,
    /// The element instance tag of the single SCE/CPE (from a PCE; 0 for
    /// the indexed configurations).
    pub elem_tag: u32,
}

impl AudioSpecificConfig {
    /// `ff_mpeg4audio_get_config_gb` with `sync_extension = 1`, then
    /// `decode_ga_specific_config`.
    pub fn parse(asc: &[u8]) -> Result<Self, AacError> {
        let mut br = BitReader::new(asc);
        let get_ot = |br: &mut BitReader| {
            let o = br.bits(5);
            if o == 31 {
                32 + br.bits(6)
            } else {
                o
            }
        };
        let get_sr = |br: &mut BitReader| -> (usize, u32) {
            let i = br.bits(4) as usize;
            if i == 15 {
                (i, br.bits(24))
            } else {
                (i, SAMPLE_RATES.get(i).copied().unwrap_or(0))
            }
        };
        let mut object_type = get_ot(&mut br);
        let (sampling_index, sample_rate) = get_sr(&mut br);
        let chan_config = br.bits(4);
        let channels = [0, 1, 2, 3, 4, 5, 6, 8, 0, 0, 0, 7, 8, 24, 8]
            .get(chan_config as usize)
            .copied()
            .ok_or_else(|| AacError::Invalid(format!("chan_config {chan_config}")))?;
        let (mut sbr, mut ps) = (-1i32, -1i32);
        let mut ext_sample_rate = 0;
        let mut explicit_sbr = false;
        if object_type == 5 || (object_type == 29 && !(br.show(3) & 3 != 0 && br.show(9) & 0x3f == 0)) {
            if object_type == 29 {
                ps = 1;
            }
            explicit_sbr = true;
            sbr = 1;
            ext_sample_rate = get_sr(&mut br).1;
            object_type = get_ot(&mut br);
        }
        let specific = br.pos();
        if !explicit_sbr {
            // Sync extension scan from the start of the specific config.
            let mut scan = BitReader::new(asc);
            scan.skip(specific as u32);
            while scan.left() > 15 {
                if scan.show(11) == 0x2b7 {
                    scan.skip(11);
                    let ext = get_ot(&mut scan);
                    if ext == 5 {
                        sbr = scan.bits(1) as i32;
                        if sbr == 1 {
                            ext_sample_rate = get_sr(&mut scan).1;
                            if ext_sample_rate == sample_rate {
                                sbr = -1;
                            }
                        }
                    }
                    if scan.left() > 11 && scan.bits(11) == 0x548 {
                        ps = scan.bits(1) as i32;
                    }
                    break;
                }
                scan.skip(1);
            }
        }
        if sbr == 0 {
            ps = 0;
        }
        if (ps == -1 && object_type != 2) || channels & !1 != 0 {
            ps = 0;
        }
        if object_type != 2 {
            return unsupported(format!("audio object type {object_type} (only AAC-LC cores)"));
        }
        if sampling_index >= 12 {
            return unsupported(format!("sampling index {sampling_index}"));
        }
        if br.bit() {
            return unsupported("960-sample frames");
        }
        if br.bit() {
            br.skip(14); // coreCoderDelay
        }
        let _extension_flag = br.bit();
        let (mut channels, mut elem_tag) = (channels, 0);
        if chan_config == 0 {
            // element_instance_tag, then decode_pce.
            br.skip(4);
            br.skip(2 + 4);
            let (front, side, back, lfe, assoc, cc) = (br.bits(4), br.bits(4), br.bits(4), br.bits(2), br.bits(3), br.bits(4));
            for skip in [4, 4] {
                if br.bit() {
                    br.skip(skip);
                }
            }
            if br.bit() {
                br.skip(3);
            }
            if front != 1 || side + back + lfe + cc != 0 {
                return unsupported("program config element layouts beyond one front SCE/CPE");
            }
            let is_cpe = br.bit();
            elem_tag = br.bits(4);
            br.skip(4 * assoc);
            channels = if is_cpe { 2 } else { 1 };
        } else if !(1..=2).contains(&chan_config) {
            return unsupported(format!("channel configuration {chan_config}"));
        }
        // decode_ga_specific_config's channel-count rule.
        if channels > 1 {
            ps = 0;
        } else if sbr == 1 && ps == -1 {
            ps = 1;
        }
        Ok(Self { object_type, sampling_index, sample_rate, chan_config, channels, sbr, ps, ext_sample_rate, elem_tag })
    }
}

const ONLY_LONG: u8 = 0;
const LONG_START: u8 = 1;
const EIGHT_SHORT: u8 = 2;
const LONG_STOP: u8 = 3;

const ZERO_BT: u8 = 0;
const NOISE_BT: u8 = 13;
const INTENSITY_BT2: u8 = 14;
const INTENSITY_BT: u8 = 15;

#[derive(Clone, Default)]
struct Ics {
    max_sfb: usize,
    window_sequence: [u8; 2],
    use_kb_window: [bool; 2],
    num_window_groups: usize,
    group_len: [usize; 8],
    num_windows: usize,
    swb_offset: &'static [u16],
    num_swb: usize,
    tns_max_bands: usize,
    predictor_present: bool,
}

#[derive(Clone, Default)]
struct Tns {
    present: bool,
    n_filt: [usize; 8],
    length: [[usize; 4]; 8],
    order: [[usize; 4]; 8],
    direction: [[bool; 4]; 8],
    coef: [[[f32; 20]; 4]; 8],
}

#[derive(Clone)]
struct Sce {
    ics: Ics,
    tns: Tns,
    band_type: [u8; 128],
    sfo: [i32; 128],
    sf: [f32; 128],
    coeffs: [f32; 1024],
    saved: [f32; 1536],
    output: [f32; 1024],
}

impl Default for Sce {
    fn default() -> Self {
        Self {
            ics: Ics::default(),
            tns: Tns::default(),
            band_type: [0; 128],
            sfo: [0; 128],
            sf: [0.0; 128],
            coeffs: [0.0; 1024],
            saved: [0.0; 1536],
            output: [0.0; 1024],
        }
    }
}

/// How often each coding tool was used (frames, or bands for the
/// band-level tools) — a coverage record for parity tests.
#[derive(Debug, Clone, Default)]
pub struct ToolStats {
    /// Raw data blocks decoded.
    pub frames: u64,
    /// Frames with an `EIGHT_SHORT_SEQUENCE` window.
    pub short_windows: u64,
    /// Frames with a start or stop window.
    pub start_stop_windows: u64,
    /// Frames with the Kaiser-Bessel-derived window shape.
    pub kbd_windows: u64,
    /// Bands coded mid/side.
    pub ms_bands: u64,
    /// Bands coded with intensity stereo.
    pub intensity_bands: u64,
    /// Bands coded with perceptual noise substitution.
    pub noise_bands: u64,
    /// Bands using the escape codebook.
    pub escape_bands: u64,
    /// Pulse data blocks.
    pub pulses: u64,
    /// Temporal noise shaping filters.
    pub tns_filters: u64,
    /// Dynamic range control payloads.
    pub drc: u64,
    /// Fill and data stream elements.
    pub fill_or_dse: u64,
    /// Frames with SBR active (a header received), and of those coupled
    /// stereo SBR / downsampled (single-rate) SBR.
    pub sbr_frames: u64,
    /// Frames with coupled-stereo SBR.
    pub sbr_coupled: u64,
    /// Frames with downsampled (single-rate) SBR.
    pub sbr_downsampled: u64,
    /// Frames with parametric stereo active, in 34-band mode, with IPD/OPD.
    pub ps_frames: u64,
    /// Frames with 34-band parametric stereo.
    pub ps_34_bands: u64,
    /// Frames with IPD/OPD parametric stereo.
    pub ps_ipdopd: u64,
}

struct Pulse {
    num_pulse: usize,
    pos: [usize; 4],
    amp: [i32; 4],
}

/// An AAC decoder, as ffmpeg n7.1 decodes AAC-LC and HE-AAC (v1 and v2).
pub struct AacDecoder {
    /// The parsed `AudioSpecificConfig`.
    pub config: AudioSpecificConfig,
    ch: Vec<Sce>,
    ms_mask: [bool; 128],
    random_state: u32,
    frames: u64,
    buf: [f32; 1024],
    temp: [f32; 128],
    /// Set when a FIL element names libfaac: ffmpeg then skips 1024 samples.
    pub libfaac_skip: bool,
    /// Which coding tools the stream has used so far.
    pub stats: ToolStats,
    /// The element's SBR context (`ExtChannelElement`), allocated with it.
    sbr: Box<super::sbr::Sbr>,
    /// `oc[1].status == OC_LOCKED`: a frame with audio was decoded.
    locked: bool,
    /// Right-channel output of a mono element with parametric stereo.
    ps_right: Vec<f32>,
}

impl AacDecoder {
    /// A decoder for the `AudioSpecificConfig` `asc` (from `esds`, or
    /// [`AacTrack::asc`](super::AacTrack::asc)).
    pub fn new(asc: &[u8]) -> Result<Self, AacError> {
        let config = AudioSpecificConfig::parse(asc)?;
        let n = config.channels;
        Ok(Self {
            config,
            ch: vec![Sce::default(); n],
            ms_mask: [false; 128],
            random_state: 0x1f2e3d4c,
            frames: 0,
            buf: [0.0; 1024],
            temp: [0.0; 128],
            libfaac_skip: false,
            stats: ToolStats::default(),
            sbr: Box::new(super::sbr::Sbr::new(if n == 1 { super::sbr::TYPE_SCE } else { super::sbr::TYPE_CPE })),
            locked: false,
            ps_right: vec![0.0; 2048],
        })
    }

    /// Output channels (a mono stream with parametric stereo decodes to 2).
    pub fn channels(&self) -> usize {
        if self.config.ps == 1 && self.config.channels == 1 {
            2
        } else {
            self.config.channels
        }
    }

    /// `multiplier`: 1 when SBR doubles the output rate.
    fn multiplier(&self) -> u32 {
        (self.config.sbr == 1 && self.config.ext_sample_rate > self.config.sample_rate) as u32
    }

    /// Samples per channel per frame (1024, or 2048 with SBR).
    pub fn frame_len(&self) -> usize {
        1024 << self.multiplier()
    }

    /// The output sampling rate.
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate << self.multiplier()
    }

    /// Decode one raw_data_block into 1024 planar samples per channel
    /// (`AV_SAMPLE_FMT_FLTP`), appended to `out[c]`.
    pub fn decode_frame(&mut self, packet: &[u8], out: &mut Vec<Vec<f32>>) -> Result<(), AacError> {
        let mut br = BitReader::new(packet);
        let mut present = [false; 2];
        let mut have_prev = false;
        let mut prev_type = 0u32;
        loop {
            let elem_type = br.bits(3);
            if elem_type == 7 {
                break;
            }
            let elem_id = br.bits(4) as usize;
            match elem_type {
                0 | 3 => {
                    // SCE / LFE
                    if self.config.channels != 1 || elem_id as u32 != self.config.elem_tag || elem_type == 3 {
                        return unsupported("element layout beyond mono SCE / stereo CPE");
                    }
                    if present[0] {
                        return invalid("duplicate channel element");
                    }
                    present[0] = true;
                    let mut ch = std::mem::take(&mut self.ch[0]);
                    let r = self.decode_ics(&mut br, &mut ch, false);
                    self.ch[0] = ch;
                    r?;
                    have_prev = true;
                    prev_type = super::sbr::TYPE_SCE;
                }
                1 => {
                    if self.config.channels != 2 || elem_id as u32 != self.config.elem_tag {
                        return unsupported("element layout beyond mono SCE / stereo CPE");
                    }
                    if present[1] {
                        return invalid("duplicate channel element");
                    }
                    present[1] = true;
                    self.decode_cpe(&mut br)?;
                    have_prev = true;
                    prev_type = super::sbr::TYPE_CPE;
                }
                2 => return unsupported("coupling channel element"),
                4 => {
                    self.stats.fill_or_dse += 1;
                    // DSE
                    let byte_align = br.bit();
                    let mut count = br.bits(8) as isize;
                    if count == 255 {
                        count += br.bits(8) as isize;
                    }
                    if byte_align {
                        br.align();
                    }
                    if br.left() < 8 * count {
                        return invalid("DSE overread");
                    }
                    br.skip_long(8 * count);
                }
                5 => return unsupported("program config element in the stream"),
                6 => {
                    self.stats.fill_or_dse += 1;
                    let mut cnt = elem_id as isize;
                    if cnt == 15 {
                        cnt += br.bits(8) as isize - 1;
                    }
                    if br.left() < 8 * cnt {
                        return invalid("FIL overread");
                    }
                    while cnt > 0 {
                        cnt -= self.extension_payload(&mut br, cnt, have_prev.then_some(prev_type))?;
                    }
                }
                _ => unreachable!(),
            }
            if br.left() < 3 {
                return invalid("overread");
            }
        }
        if !present[0] && !present[1] {
            return invalid("frame without audio");
        }
        let nout = self.channels();
        if out.len() < nout {
            out.resize(nout, Vec::new());
        }
        let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(self.ch.len());
        for c in 0..self.ch.len() {
            let mut ch = std::mem::take(&mut self.ch[c]);
            let st = &mut self.stats;
            match ch.ics.window_sequence[0] {
                EIGHT_SHORT => st.short_windows += 1,
                LONG_START | LONG_STOP => st.start_stop_windows += 1,
                _ => {}
            }
            st.kbd_windows += ch.ics.use_kb_window[0] as u64;
            let nb = ch.ics.num_window_groups * ch.ics.max_sfb;
            for &bt in &ch.band_type[..nb] {
                match bt {
                    NOISE_BT => st.noise_bands += 1,
                    INTENSITY_BT | INTENSITY_BT2 => st.intensity_bands += 1,
                    11 => st.escape_bands += 1,
                    _ => {}
                }
            }
            if ch.tns.present {
                st.tns_filters += ch.tns.n_filt[..ch.ics.num_windows].iter().sum::<usize>() as u64;
                apply_tns(&mut ch.coeffs, &ch.tns, &ch.ics);
            }
            self.imdct_and_windowing(&mut ch);
            outputs.push(ch.output.to_vec());
            self.ch[c] = ch;
        }
        if self.config.sbr > 0 {
            // ff_aac_sbr_apply on che->ch[0].output / ch[1].output (2048 each).
            let id = if self.ch.len() == 2 { super::sbr::TYPE_CPE } else { super::sbr::TYPE_SCE };
            let downsampled = self.config.ext_sample_rate < self.sbr.sample_rate as u32;
            let mut l = vec![0f32; 2048];
            l[..1024].copy_from_slice(&outputs[0]);
            let mut r = std::mem::take(&mut self.ps_right);
            if outputs.len() > 1 {
                r[..1024].copy_from_slice(&outputs[1]);
            }
            self.sbr.apply(id, &mut l, &mut r, downsampled, self.config.ps == 1);
            let (sbr_on, coupled, ps_on, ps34, ipd) = self.sbr.coverage();
            let st = &mut self.stats;
            st.sbr_frames += sbr_on as u64;
            st.sbr_coupled += (sbr_on && coupled) as u64;
            st.sbr_downsampled += (sbr_on && downsampled) as u64;
            let ps_on = ps_on && self.config.ps == 1;
            st.ps_frames += ps_on as u64;
            st.ps_34_bands += (ps_on && ps34) as u64;
            st.ps_ipdopd += (ps_on && ipd) as u64;
            let n = self.frame_len();
            outputs.clear();
            outputs.push(l[..n].to_vec());
            if nout == 2 {
                outputs.push(r[..n].to_vec());
            }
            self.ps_right = r;
        }
        for (plane, o) in out.iter_mut().zip(&outputs) {
            plane.extend_from_slice(o);
        }
        self.locked = true;
        self.frames += 1;
        self.stats.frames += 1;
        Ok(())
    }

    fn extension_payload(&mut self, br: &mut BitReader, cnt: isize, prev: Option<u32>) -> Result<isize, AacError> {
        let ty = br.bits(4);
        match ty {
            0xd | 0xe => {
                // decode_extension_payload's EXT_SBR_DATA(_CRC) cases.
                let Some(id) = prev else {
                    return Ok(cnt);
                };
                if self.config.sbr == 0 || (self.config.sbr == -1 && self.locked) {
                    br.skip_long(8 * cnt - 4);
                    return Ok(cnt);
                }
                if self.config.ps == -1 && !self.locked && self.config.channels == 1 {
                    self.config.sbr = 1;
                    self.config.ps = 1;
                } else {
                    self.config.sbr = 1;
                }
                let core = self.config.sample_rate;
                let ps_allowed = self.config.ps;
                let mut ext = self.config.ext_sample_rate;
                self.sbr.decode_extension(br, ty == 0xe, cnt, id, core, &mut ext, ps_allowed);
                self.config.ext_sample_rate = ext;
                Ok(cnt)
            }
            0xb => {
                self.stats.drc += 1;
                Ok(decode_dynamic_range(br))
            }
            0x1 => {
                // EXT_FILL: decode_fill.
                let mut len = 8 * cnt - 4;
                if len >= 13 + 7 * 8 {
                    br.skip(13);
                    len -= 13;
                    let mut s = Vec::new();
                    while s.len() + 1 < 256 && len >= 8 {
                        s.push(br.bits(8) as u8);
                        len -= 8;
                    }
                    if let Some(rest) = s.strip_prefix(b"libfaac ") {
                        let txt = String::from_utf8_lossy(rest);
                        let mut it = txt.splitn(2, '.');
                        let ok = |x: Option<&str>| {
                            x.map(|v| v.trim_start().chars().take_while(|c| c.is_ascii_digit() || *c == '-').count() > 0)
                                .unwrap_or(false)
                        };
                        if ok(it.next()) && ok(it.next()) {
                            self.libfaac_skip = true;
                        }
                    }
                }
                br.skip_long(len);
                Ok(cnt)
            }
            _ => {
                br.skip_long(8 * cnt - 4);
                Ok(cnt)
            }
        }
    }

    fn decode_cpe(&mut self, br: &mut BitReader) -> Result<(), AacError> {
        let common_window = br.bit();
        let mut ms_present = 0;
        let (mut c0, mut c1) = (std::mem::take(&mut self.ch[0]), std::mem::take(&mut self.ch[1]));
        let r = (|| {
            if common_window {
                self.decode_ics_info(br, &mut c0.ics)?;
                let kb = c1.ics.use_kb_window[0];
                c1.ics = c0.ics.clone();
                c1.ics.use_kb_window[1] = kb;
                ms_present = br.bits(2);
                if ms_present == 3 {
                    return invalid("ms_present = 3");
                }
                let max_idx = c0.ics.num_window_groups * c0.ics.max_sfb;
                if ms_present == 1 {
                    for i in 0..max_idx {
                        self.ms_mask[i] = br.bit();
                    }
                } else if ms_present == 2 {
                    self.ms_mask[..max_idx].fill(true);
                }
            }
            self.decode_ics(br, &mut c0, common_window)?;
            self.decode_ics(br, &mut c1, common_window)?;
            if common_window && ms_present != 0 {
                self.apply_mid_side(&mut c0, &mut c1);
            }
            self.apply_intensity(&c0, &mut c1, ms_present != 0);
            Ok(())
        })();
        self.ch[0] = c0;
        self.ch[1] = c1;
        r
    }

    fn decode_ics_info(&self, br: &mut BitReader, ics: &mut Ics) -> Result<(), AacError> {
        let si = self.config.sampling_index;
        br.skip(1); // reserved
        ics.window_sequence[1] = ics.window_sequence[0];
        ics.window_sequence[0] = br.bits(2) as u8;
        ics.use_kb_window[1] = ics.use_kb_window[0];
        ics.use_kb_window[0] = br.bit();
        ics.num_window_groups = 1;
        ics.group_len[0] = 1;
        if ics.window_sequence[0] == EIGHT_SHORT {
            ics.max_sfb = br.bits(4) as usize;
            for _ in 0..7 {
                if br.bit() {
                    ics.group_len[ics.num_window_groups - 1] += 1;
                } else {
                    ics.num_window_groups += 1;
                    ics.group_len[ics.num_window_groups - 1] = 1;
                }
            }
            ics.num_windows = 8;
            ics.swb_offset = t::SWB_OFFSET_128[si];
            ics.num_swb = t::NUM_SWB_128[si] as usize;
            ics.tns_max_bands = t::TNS_MAX_BANDS_128[si] as usize;
            ics.predictor_present = false;
        } else {
            ics.max_sfb = br.bits(6) as usize;
            ics.num_windows = 1;
            ics.swb_offset = t::SWB_OFFSET_1024[si];
            ics.num_swb = t::NUM_SWB_1024[si] as usize;
            ics.tns_max_bands = t::TNS_MAX_BANDS_1024[si] as usize;
            ics.predictor_present = br.bit();
            if ics.predictor_present {
                ics.max_sfb = 0;
                return invalid("Prediction is not allowed in AAC-LC");
            }
        }
        if ics.max_sfb > ics.num_swb {
            ics.max_sfb = 0;
            return invalid("max_sfb exceeds num_swb");
        }
        Ok(())
    }

    fn decode_ics(&mut self, br: &mut BitReader, sce: &mut Sce, common_window: bool) -> Result<(), AacError> {
        let r = self.decode_ics_inner(br, sce, common_window);
        if r.is_err() {
            sce.tns.present = false;
        }
        r
    }

    fn decode_ics_inner(&mut self, br: &mut BitReader, sce: &mut Sce, common_window: bool) -> Result<(), AacError> {
        let global_gain = br.bits(8) as i32;
        if !common_window {
            self.decode_ics_info(br, &mut sce.ics)?;
        }
        decode_band_types(br, sce)?;
        decode_scalefactors(br, sce, global_gain)?;
        dequant_scalefactors(sce);
        let mut pulse = None;
        if br.bit() {
            if sce.ics.window_sequence[0] == EIGHT_SHORT {
                return invalid("Pulse tool not allowed in eight short sequence");
            }
            pulse = Some(decode_pulses(br, sce.ics.swb_offset, sce.ics.num_swb).ok_or(AacError::Invalid("pulse".into()))?);
            self.stats.pulses += 1;
        }
        sce.tns.present = br.bit();
        if sce.tns.present {
            decode_tns(br, &mut sce.tns, &sce.ics)?;
        }
        if br.bit() {
            return unsupported("SSR gain control");
        }
        self.decode_spectrum_and_dequant(br, pulse.as_ref(), sce)
    }

    fn decode_spectrum_and_dequant(&mut self, br: &mut BitReader, pulse: Option<&Pulse>, sce: &mut Sce) -> Result<(), AacError> {
        let tb = tables();
        let ics = &sce.ics;
        let c = 1024 / ics.num_windows;
        let offsets = ics.swb_offset;
        let coef = &mut sce.coeffs;
        for g in 0..ics.num_windows {
            let from = g * 128 + offsets[ics.max_sfb] as usize;
            coef[from..g * 128 + c].fill(0.0);
        }
        let mut idx = 0;
        let mut base = 0usize;
        for g in 0..ics.num_window_groups {
            let g_len = ics.group_len[g];
            for i in 0..ics.max_sfb {
                let bt = sce.band_type[idx];
                let off = offsets[i] as usize;
                let off_len = offsets[i + 1] as usize - off;
                let sfv = sce.sf[idx];
                if bt == ZERO_BT || bt >= INTENSITY_BT2 {
                    for grp in 0..g_len {
                        let s = base + grp * 128 + off;
                        coef[s..s + off_len].fill(0.0);
                    }
                } else if bt == NOISE_BT {
                    for grp in 0..g_len {
                        let s = base + grp * 128 + off;
                        let cf = &mut coef[s..s + off_len];
                        for v in cf.iter_mut() {
                            self.random_state = self.random_state.wrapping_mul(1664525).wrapping_add(1013904223);
                            *v = self.random_state as i32 as f32;
                        }
                        let band_energy = scalarproduct(cf);
                        let scale = sfv / band_energy.sqrt();
                        for v in cf.iter_mut() {
                            *v *= scale;
                        }
                    }
                } else {
                    let cbt_m1 = (bt - 1) as usize;
                    let vq = t::VEC_VALS[cbt_m1];
                    let vl = vlc::spectral(cbt_m1);
                    for grp in 0..g_len {
                        let s = base + grp * 128 + off;
                        let cf = &mut coef[s..s + off_len];
                        let read = |br: &mut BitReader| vl.read(br).ok_or(AacError::Invalid("spectral VLC".into()));
                        match cbt_m1 >> 1 {
                            0 => {
                                for q in cf.chunks_exact_mut(4) {
                                    let ix = read(br)? as usize;
                                    q[0] = vq[ix & 3] * sfv;
                                    q[1] = vq[ix >> 2 & 3] * sfv;
                                    q[2] = vq[ix >> 4 & 3] * sfv;
                                    q[3] = vq[ix >> 6 & 3] * sfv;
                                }
                            }
                            1 => {
                                for q in cf.chunks_exact_mut(4) {
                                    let ix = read(br)?;
                                    let nnz = ix >> 8 & 15;
                                    let mut sign = if nnz != 0 { br.peek32() } else { 0 };
                                    br.skip(nnz);
                                    // VMUL4S
                                    let mut nz = ix >> 12;
                                    let sb = sfv.to_bits();
                                    let ix = ix as usize;
                                    q[0] = vq[ix & 3] * f32::from_bits(sb ^ (sign & 1 << 31));
                                    sign <<= nz & 1;
                                    nz >>= 1;
                                    q[1] = vq[ix >> 2 & 3] * f32::from_bits(sb ^ (sign & 1 << 31));
                                    sign <<= nz & 1;
                                    nz >>= 1;
                                    q[2] = vq[ix >> 4 & 3] * f32::from_bits(sb ^ (sign & 1 << 31));
                                    sign <<= nz & 1;
                                    q[3] = vq[ix >> 6 & 3] * f32::from_bits(sb ^ (sign & 1 << 31));
                                }
                            }
                            2 => {
                                for q in cf.chunks_exact_mut(2) {
                                    let ix = read(br)? as usize;
                                    q[0] = vq[ix & 15] * sfv;
                                    q[1] = vq[ix >> 4 & 15] * sfv;
                                }
                            }
                            3 | 4 => {
                                for q in cf.chunks_exact_mut(2) {
                                    let ix = read(br)?;
                                    let nnz = ix >> 8 & 15;
                                    let sign = if nnz != 0 { br.show(nnz) << (ix >> 12) } else { 0 };
                                    br.skip(nnz);
                                    // VMUL2S
                                    let sb = sfv.to_bits();
                                    let ix = ix as usize;
                                    q[0] = vq[ix & 15] * f32::from_bits(sb ^ ((sign >> 1) << 31));
                                    q[1] = vq[ix >> 4 & 15] * f32::from_bits(sb ^ (sign << 31));
                                }
                            }
                            _ => {
                                for q in cf.chunks_exact_mut(2) {
                                    let mut ix = read(br)?;
                                    if ix == 0 {
                                        q[0] = 0.0;
                                        q[1] = 0.0;
                                        continue;
                                    }
                                    let nnz = ix >> 12;
                                    let nzt = ix >> 8;
                                    let mut bits = if nnz != 0 { br.show(nnz) << (32 - nnz) } else { 0 };
                                    br.skip(nnz);
                                    for (j, slot) in q.iter_mut().enumerate() {
                                        if nzt & (1 << j) != 0 {
                                            let b = br.peek32();
                                            let b = (!b).leading_zeros();
                                            if b > 8 {
                                                return invalid("ESC overflow");
                                            }
                                            br.skip(b + 1);
                                            let b = b + 4;
                                            let n = (1u32 << b) + br.bits(b);
                                            *slot = f32::from_bits(tb.cbrt[n as usize] | (bits & 1 << 31));
                                            bits <<= 1;
                                        } else {
                                            let v = vq[(ix & 15) as usize].to_bits();
                                            *slot = f32::from_bits((bits & 1 << 31) | v);
                                            bits <<= (v != 0) as u32;
                                        }
                                        ix >>= 4;
                                    }
                                }
                                for v in cf.iter_mut() {
                                    *v *= sfv;
                                }
                            }
                        }
                    }
                }
                idx += 1;
            }
            base += g_len << 7;
        }
        if let Some(p) = pulse {
            let mut idx = 0;
            for i in 0..p.num_pulse {
                let pos = p.pos[i];
                let co = coef[pos];
                while offsets[idx + 1] as usize <= pos {
                    idx += 1;
                }
                if sce.band_type[idx] != NOISE_BT && sce.sf[idx] != 0.0 {
                    let mut ico = -(p.amp[i] as f32);
                    if co != 0.0 {
                        let co = co / sce.sf[idx];
                        ico = co / co.abs().sqrt().sqrt() + if co > 0.0 { -ico } else { ico };
                    }
                    coef[pos] = ico.abs().cbrt() * ico * sce.sf[idx];
                }
            }
        }
        Ok(())
    }

    fn apply_mid_side(&mut self, c0: &mut Sce, c1: &mut Sce) {
        let mut ms_bands = 0;
        let ics = &c0.ics;
        let offsets = ics.swb_offset;
        let max_sfb = ics.max_sfb;
        let mut base = 0;
        for g in 0..ics.num_window_groups {
            for sfb in 0..max_sfb {
                let idx = g * max_sfb + sfb;
                if self.ms_mask[idx] && c0.band_type[idx] < NOISE_BT && c1.band_type[idx] < NOISE_BT {
                    ms_bands += 1;
                    for grp in 0..ics.group_len[g] {
                        let s = base + grp * 128 + offsets[sfb] as usize;
                        for k in s..base + grp * 128 + offsets[sfb + 1] as usize {
                            let (a, b) = (c0.coeffs[k], c1.coeffs[k]);
                            c0.coeffs[k] = a + b;
                            c1.coeffs[k] = a - b;
                        }
                    }
                }
            }
            base += ics.group_len[g] * 128;
        }
        self.stats.ms_bands += ms_bands;
    }

    fn apply_intensity(&self, c0: &Sce, c1: &mut Sce, ms_present: bool) {
        let ics = c1.ics.clone();
        let offsets = ics.swb_offset;
        let mut base = 0;
        for g in 0..ics.num_window_groups {
            for sfb in 0..ics.max_sfb {
                let idx = g * ics.max_sfb + sfb;
                let bt = c1.band_type[idx];
                if bt == INTENSITY_BT || bt == INTENSITY_BT2 {
                    let mut c = -1 + 2 * (bt as i32 - 14);
                    if ms_present {
                        c *= 1 - 2 * self.ms_mask[idx] as i32;
                    }
                    let scale = c as f32 * c1.sf[idx];
                    for grp in 0..ics.group_len[g] {
                        let s = base + grp * 128;
                        for k in s + offsets[sfb] as usize..s + offsets[sfb + 1] as usize {
                            c1.coeffs[k] = c0.coeffs[k] * scale;
                        }
                    }
                }
            }
            base += ics.group_len[g] * 128;
        }
    }

    fn imdct_and_windowing(&mut self, sce: &mut Sce) {
        let tb = tables();
        let ics = &sce.ics;
        let swindow: &[f32] = if ics.use_kb_window[0] { &tb.kbd_short } else { &tb.sine_short };
        let lwindow_prev: &[f32] = if ics.use_kb_window[1] { &tb.kbd_long } else { &tb.sine_long };
        let swindow_prev: &[f32] = if ics.use_kb_window[1] { &tb.kbd_short } else { &tb.sine_short };
        let buf = &mut self.buf;
        let temp = &mut self.temp;
        let (ws0, ws1) = (ics.window_sequence[0], ics.window_sequence[1]);
        if ws0 == EIGHT_SHORT {
            for i in (0..1024).step_by(128) {
                tb.imdct_short.run(&sce.coeffs[i..i + 128], &mut buf[i..i + 128]);
            }
        } else {
            tb.imdct_long.run(&sce.coeffs, buf);
        }
        let out = &mut sce.output;
        let saved = &mut sce.saved;
        if (ws1 == ONLY_LONG || ws1 == LONG_STOP) && (ws0 == ONLY_LONG || ws0 == LONG_START) {
            vector_fmul_window(out, saved, buf, lwindow_prev, 512);
        } else {
            out[..448].copy_from_slice(&saved[..448]);
            if ws0 == EIGHT_SHORT {
                vector_fmul_window(&mut out[448..], &saved[448..], &buf[0..], swindow_prev, 64);
                vector_fmul_window(&mut out[448 + 128..], &buf[64..], &buf[128..], swindow, 64);
                vector_fmul_window(&mut out[448 + 256..], &buf[128 + 64..], &buf[256..], swindow, 64);
                vector_fmul_window(&mut out[448 + 384..], &buf[256 + 64..], &buf[384..], swindow, 64);
                vector_fmul_window(temp, &buf[384 + 64..], &buf[512..], swindow, 64);
                out[448 + 512..1024].copy_from_slice(&temp[..64]);
            } else {
                vector_fmul_window(&mut out[448..], &saved[448..], buf, swindow_prev, 64);
                out[576..1024].copy_from_slice(&buf[64..512]);
            }
        }
        if ws0 == EIGHT_SHORT {
            saved[..64].copy_from_slice(&temp[64..128]);
            vector_fmul_window(&mut saved[64..], &buf[512 + 64..], &buf[640..], swindow, 64);
            vector_fmul_window(&mut saved[192..], &buf[640 + 64..], &buf[768..], swindow, 64);
            vector_fmul_window(&mut saved[320..], &buf[768 + 64..], &buf[896..], swindow, 64);
            saved[448..512].copy_from_slice(&buf[896 + 64..1024]);
        } else if ws0 == LONG_START {
            saved[..448].copy_from_slice(&buf[512..960]);
            saved[448..512].copy_from_slice(&buf[896 + 64..1024]);
        } else {
            saved[..512].copy_from_slice(&buf[512..1024]);
        }
    }
}

fn decode_band_types(br: &mut BitReader, sce: &mut Sce) -> Result<(), AacError> {
    let ics = &sce.ics;
    let bits = if ics.window_sequence[0] == EIGHT_SHORT { 3 } else { 5 };
    for g in 0..ics.num_window_groups {
        let mut k = 0;
        while k < ics.max_sfb {
            let mut sect_end = k;
            let bt = br.bits(4) as u8;
            if bt == 12 {
                return invalid("invalid band type");
            }
            loop {
                let inc = br.bits(bits) as usize;
                sect_end += inc;
                if br.left() < 0 {
                    return invalid("band types overread");
                }
                if sect_end > ics.max_sfb {
                    return invalid("band count exceeds max_sfb");
                }
                if inc != (1 << bits) - 1 {
                    break;
                }
            }
            while k < sect_end {
                sce.band_type[g * ics.max_sfb + k] = bt;
                k += 1;
            }
        }
    }
    Ok(())
}

fn decode_scalefactors(br: &mut BitReader, sce: &mut Sce, global_gain: i32) -> Result<(), AacError> {
    let ics = &sce.ics;
    let mut offset = [global_gain, global_gain - 90, 0];
    let mut noise_flag = 1;
    let vl = vlc::scalefactor();
    let read = |br: &mut BitReader| vl.read(br).map(|v| v as i32 - 60).ok_or(AacError::Invalid("scalefactor VLC".into()));
    for g in 0..ics.num_window_groups {
        for sfb in 0..ics.max_sfb {
            let i = g * ics.max_sfb + sfb;
            match sce.band_type[i] {
                ZERO_BT => sce.sfo[i] = 0,
                INTENSITY_BT | INTENSITY_BT2 => {
                    offset[2] += read(br)?;
                    sce.sfo[i] = offset[2].clamp(-155, 100) - 100;
                }
                NOISE_BT => {
                    let first = noise_flag > 0;
                    noise_flag -= 1;
                    if first {
                        offset[1] += br.bits(9) as i32 - 256;
                    } else {
                        offset[1] += read(br)?;
                    }
                    sce.sfo[i] = offset[1].clamp(-100, 155);
                }
                _ => {
                    offset[0] += read(br)?;
                    if offset[0] as u32 > 255 {
                        return invalid("scalefactor out of range");
                    }
                    sce.sfo[i] = offset[0] - 100;
                }
            }
        }
    }
    Ok(())
}

fn dequant_scalefactors(sce: &mut Sce) {
    let p = &tables().pow2sf;
    let ics = &sce.ics;
    for i in 0..ics.num_window_groups * ics.max_sfb {
        let sfo = sce.sfo[i];
        sce.sf[i] = match sce.band_type[i] {
            ZERO_BT => 0.0,
            INTENSITY_BT | INTENSITY_BT2 => p[(-sfo - 100 + 200) as usize],
            _ => -p[(sfo + 200) as usize],
        };
    }
}

fn decode_pulses(br: &mut BitReader, swb_offset: &[u16], num_swb: usize) -> Option<Pulse> {
    let mut p = Pulse { num_pulse: br.bits(2) as usize + 1, pos: [0; 4], amp: [0; 4] };
    let swb = br.bits(6) as usize;
    if swb >= num_swb {
        return None;
    }
    let end = swb_offset[num_swb] as usize;
    p.pos[0] = swb_offset[swb] as usize + br.bits(5) as usize;
    if p.pos[0] >= end {
        return None;
    }
    p.amp[0] = br.bits(4) as i32;
    for i in 1..p.num_pulse {
        p.pos[i] = br.bits(5) as usize + p.pos[i - 1];
        if p.pos[i] >= end {
            return None;
        }
        p.amp[i] = br.bits(4) as i32;
    }
    Some(p)
}

fn decode_tns(br: &mut BitReader, tns: &mut Tns, ics: &Ics) -> Result<(), AacError> {
    let is8 = (ics.window_sequence[0] == EIGHT_SHORT) as u32;
    let max_order = if is8 == 1 { 7 } else { 12 };
    for w in 0..ics.num_windows {
        tns.n_filt[w] = br.bits(2 - is8) as usize;
        if tns.n_filt[w] == 0 {
            continue;
        }
        let coef_res = br.bits(1);
        for f in 0..tns.n_filt[w] {
            tns.length[w][f] = br.bits(6 - 2 * is8) as usize;
            tns.order[w][f] = br.bits(5 - 2 * is8) as usize;
            if tns.order[w][f] > max_order {
                tns.order[w][f] = 0;
                return invalid("TNS filter order");
            }
            if tns.order[w][f] != 0 {
                tns.direction[w][f] = br.bit();
                let compress = br.bits(1);
                let coef_len = coef_res + 3 - compress;
                let map = t::TNS_TMP2_MAP[(2 * compress + coef_res) as usize];
                for i in 0..tns.order[w][f] {
                    tns.coef[w][f][i] = map[br.bits(coef_len) as usize];
                }
            }
        }
    }
    Ok(())
}

/// `compute_lpc_coefs(coef, order, lpc, 0, 0, 0)` in float: the in-place
/// step-up recursion. The reference build does not contract C arithmetic
/// (measured on av_tx's `CMUL`), so every product rounds on its own.
fn compute_lpc(coef: &[f32], order: usize, lpc: &mut [f32; 20]) {
    for i in 0..order {
        let r = -coef[i];
        lpc[i] = r;
        for j in 0..(i + 1) >> 1 {
            let f = lpc[j];
            let b = lpc[i - 1 - j];
            lpc[j] = f + r * b;
            lpc[i - 1 - j] = b + r * f;
        }
    }
}

fn apply_tns(coef: &mut [f32; 1024], tns: &Tns, ics: &Ics) {
    let mmm = ics.tns_max_bands.min(ics.max_sfb);
    if mmm == 0 {
        return;
    }
    let mut lpc = [0f32; 20];
    for w in 0..ics.num_windows {
        let mut bottom = ics.num_swb;
        for f in 0..tns.n_filt[w] {
            let top = bottom;
            bottom = top.saturating_sub(tns.length[w][f]);
            let order = tns.order[w][f];
            if order == 0 {
                continue;
            }
            compute_lpc(&tns.coef[w][f], order, &mut lpc);
            let mut start = ics.swb_offset[bottom.min(mmm)] as isize;
            let end = ics.swb_offset[top.min(mmm)] as isize;
            let size = end - start;
            if size <= 0 {
                continue;
            }
            let inc: isize = if tns.direction[w][f] {
                start = end - 1;
                -1
            } else {
                1
            };
            start += (w * 128) as isize;
            for m in 0..size as usize {
                let s = start as usize;
                for i in 1..=m.min(order) {
                    let prev = coef[(start - i as isize * inc) as usize];
                    coef[s] -= prev * lpc[i - 1];
                }
                start += inc;
            }
        }
    }
}

/// `decode_dynamic_range`: parsed for its length only (the float decoder
/// does not apply DRC).
fn decode_dynamic_range(br: &mut BitReader) -> isize {
    let mut n = 1;
    let mut num_bands = 1;
    if br.bit() {
        br.skip(8);
        n += 1;
    }
    if br.bit() {
        let mut num_excl = 0;
        loop {
            br.skip(7);
            num_excl += 7;
            if !(num_excl < 64 - 7 && br.bit()) {
                break;
            }
        }
        n += num_excl / 7;
    }
    if br.bit() {
        let band_incr = br.bits(4) as isize;
        br.skip(4);
        n += 1;
        num_bands += band_incr;
        for _ in 0..num_bands {
            br.skip(8);
            n += 1;
        }
    }
    if br.bit() {
        br.skip(8);
        n += 1;
    }
    for _ in 0..num_bands {
        br.skip(8);
        n += 1;
    }
    n
}

