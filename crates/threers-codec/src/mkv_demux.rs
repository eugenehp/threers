//! Matroska / WebM demuxer (EBML), and the IVF container, for reading.
//!
//! Pure Rust and wasm-safe. It reads what a decoder needs: the tracks
//! (codec id, `CodecPrivate`, video size, audio rate/channels, default
//! duration, codec delay) and every frame in file order with its timestamp,
//! after undoing Xiph, EBML and fixed-size lacing. Elements of unknown size
//! (live WebM) are closed by the next element that cannot be their child, as
//! ffmpeg's `matroskadec` does. Timestamps are in `TimestampScale` units
//! (`timestamp_scale` nanoseconds each — 1 ms by default), the time base
//! ffmpeg gives Matroska streams.
//!
//! [`Mkv::parse`] is the entry point; [`parse_ivf`] reads IVF (VP8/VP9/AV1
//! test vectors).

/// The kind of a track (`TrackType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackKind {
    /// `TrackType` 1.
    Video,
    /// `TrackType` 2.
    Audio,
    /// `TrackType` 17.
    Subtitle,
    /// Anything else (buttons, logos, metadata, complex).
    Other,
}

/// One `TrackEntry`.
#[derive(Clone, Debug)]
pub struct MkvTrack {
    /// `TrackNumber`, which frames refer to.
    pub number: u64,
    /// `TrackType`.
    pub kind: TrackKind,
    /// `CodecID`, e.g. `V_VP9`, `V_AV1`, `A_OPUS`, `A_VORBIS`.
    pub codec_id: String,
    /// `CodecPrivate`, verbatim (an `OpusHead`, Vorbis headers, `av1C`, ...).
    pub codec_private: Vec<u8>,
    /// `DefaultDuration` in nanoseconds.
    pub default_duration: Option<u64>,
    /// `CodecDelay` and `SeekPreRoll` in nanoseconds.
    pub codec_delay: u64,
    /// `SeekPreRoll` in nanoseconds.
    pub seek_preroll: u64,
    /// `FlagDefault`.
    pub flag_default: bool,
    /// `PixelWidth`.
    pub width: u32,
    /// `PixelHeight`.
    pub height: u32,
    /// `DisplayWidth`, when it differs from the pixel width.
    pub display_width: Option<u32>,
    /// `DisplayHeight`, when it differs from the pixel height.
    pub display_height: Option<u32>,
    /// `AlphaMode` 1: a second (alpha) bitstream rides in `BlockAdditional`.
    pub alpha_mode: bool,
    /// The deprecated video `FrameRate` (frames per second).
    pub frame_rate: Option<f64>,
    /// `SamplingFrequency`.
    pub sample_rate: f64,
    /// `Channels`.
    pub channels: u32,
    /// `BitDepth`, 0 when absent.
    pub bit_depth: u32,
    /// `Language`, the ISO 639-2 code.
    pub language: Option<String>,
    /// Header-stripping compression (`ContentCompAlgo` 3): bytes to prepend
    /// to every frame.
    pub strip_prefix: Vec<u8>,
}

/// One frame of one track, laces split.
#[derive(Clone, Debug)]
pub struct MkvFrame {
    /// The `TrackNumber` of the track this frame belongs to.
    pub track: u64,
    /// In `timestamp_scale` units. Laced frames after the first get the
    /// block's timestamp plus their share of the block duration, when known.
    pub pts: i64,
    /// `BlockDuration` (or the lace's share of it), same units.
    pub duration: Option<i64>,
    /// Whether a decoder can start here.
    pub keyframe: bool,
    /// `DiscardPadding`, nanoseconds.
    pub discard_padding: i64,
    /// The coded frame, with any header-stripping prefix restored.
    pub data: Vec<u8>,
    /// `BlockAdditional` with `BlockAddID` 1 (the WebM alpha bitstream).
    pub alpha: Option<Vec<u8>>,
}

/// A parsed Matroska / WebM file.
#[derive(Clone, Debug, Default)]
pub struct Mkv {
    /// `DocType` (`webm` or `matroska`).
    pub doc_type: String,
    /// Nanoseconds per timestamp unit (`TimestampScale`).
    pub timestamp_scale: u64,
    /// `Duration`, in timestamp units.
    pub duration: Option<f64>,
    /// Every `TrackEntry`, in file order.
    pub tracks: Vec<MkvTrack>,
    /// Every frame of every track, in file order.
    pub frames: Vec<MkvFrame>,
}

// Element ids.
const EBML: u32 = 0x1A45_DFA3;
const DOC_TYPE: u32 = 0x4282;
const SEGMENT: u32 = 0x1853_8067;
const INFO: u32 = 0x1549_A966;
const TIMESTAMP_SCALE: u32 = 0x2A_D7B1;
const DURATION: u32 = 0x4489;
const TRACKS: u32 = 0x1654_AE6B;
const TRACK_ENTRY: u32 = 0xAE;
const TRACK_NUMBER: u32 = 0xD7;
const TRACK_TYPE: u32 = 0x83;
const FLAG_DEFAULT: u32 = 0x88;
const CODEC_ID: u32 = 0x86;
const CODEC_PRIVATE: u32 = 0x63A2;
const DEFAULT_DURATION: u32 = 0x23_E383;
const CODEC_DELAY: u32 = 0x56AA;
const SEEK_PREROLL: u32 = 0x56BB;
const LANGUAGE: u32 = 0x22_B59C;
const VIDEO: u32 = 0xE0;
const PIXEL_WIDTH: u32 = 0xB0;
const PIXEL_HEIGHT: u32 = 0xBA;
const DISPLAY_WIDTH: u32 = 0x54B0;
const DISPLAY_HEIGHT: u32 = 0x54BA;
const ALPHA_MODE: u32 = 0x53C0;
const FRAME_RATE: u32 = 0x23_83E3;
const AUDIO: u32 = 0xE1;
const SAMPLING_FREQUENCY: u32 = 0xB5;
const CHANNELS: u32 = 0x9F;
const BIT_DEPTH: u32 = 0x6264;
const CONTENT_ENCODINGS: u32 = 0x6D80;
const CONTENT_ENCODING: u32 = 0x6240;
const CONTENT_COMPRESSION: u32 = 0x5034;
const CONTENT_COMP_ALGO: u32 = 0x4254;
const CONTENT_COMP_SETTINGS: u32 = 0x4255;
const CLUSTER: u32 = 0x1F43_B675;
const TIMESTAMP: u32 = 0xE7;
const SIMPLE_BLOCK: u32 = 0xA3;
const BLOCK_GROUP: u32 = 0xA0;
const BLOCK: u32 = 0xA1;
const BLOCK_DURATION: u32 = 0x9B;
const REFERENCE_BLOCK: u32 = 0xFB;
const DISCARD_PADDING: u32 = 0x75A2;
const BLOCK_ADDITIONS: u32 = 0x75A1;
const BLOCK_MORE: u32 = 0xA6;
const BLOCK_ADD_ID: u32 = 0xEE;
const BLOCK_ADDITIONAL: u32 = 0xA5;

/// Top-level children of a `Segment`: an unknown-size `Cluster` ends at the
/// next of these.
const SEGMENT_CHILDREN: [u32; 9] = [
    0x114D_9B74, // SeekHead
    INFO,
    TRACKS,
    CLUSTER,
    0x1C53_BB6B, // Cues
    0x1043_A770, // Chapters
    0x1254_C367, // Tags
    0x1941_A469, // Attachments
    SEGMENT,
];

struct Reader<'a> {
    d: &'a [u8],
}

impl<'a> Reader<'a> {
    /// An element id (1-4 bytes, marker bit kept) at `pos`.
    fn id(&self, pos: usize) -> Option<(u32, usize)> {
        let b = *self.d.get(pos)?;
        let len = b.leading_zeros() as usize + 1;
        if len > 4 || pos + len > self.d.len() {
            return None;
        }
        let mut v = 0u32;
        for i in 0..len {
            v = (v << 8) | self.d[pos + i] as u32;
        }
        Some((v, len))
    }

    /// A data size (VINT, marker removed); `None` size = unknown.
    fn size(&self, pos: usize) -> Option<(Option<u64>, usize)> {
        let b = *self.d.get(pos)?;
        let len = b.leading_zeros() as usize + 1;
        if len > 8 || pos + len > self.d.len() {
            return None;
        }
        let mut v = (b as u64) & (0xFF >> len);
        let mut all_ones = v == (0xFF >> len) as u64;
        for i in 1..len {
            let x = self.d[pos + i];
            all_ones &= x == 0xFF;
            v = (v << 8) | x as u64;
        }
        Some((if all_ones { None } else { Some(v) }, len))
    }

    /// The element at `pos`: `(id, data start, data end)`, `end` clipped to
    /// `limit` (an unknown size runs to `limit` for now).
    fn element(&self, pos: usize, limit: usize) -> Option<(u32, usize, usize, bool)> {
        let (id, il) = self.id(pos)?;
        let (sz, sl) = self.size(pos + il)?;
        let start = pos + il + sl;
        if start > limit {
            return None;
        }
        Some(match sz {
            Some(s) => (id, start, (start as u64).saturating_add(s).min(limit as u64) as usize, false),
            None => (id, start, limit, true),
        })
    }
}

fn uint(b: &[u8]) -> u64 {
    b.iter().take(8).fold(0u64, |v, &x| (v << 8) | x as u64)
}

fn sint(b: &[u8]) -> i64 {
    if b.is_empty() {
        return 0;
    }
    let u = uint(b);
    let bits = 8 * b.len().min(8);
    if bits == 64 {
        u as i64
    } else {
        ((u << (64 - bits)) as i64) >> (64 - bits)
    }
}

fn float(b: &[u8]) -> f64 {
    match b.len() {
        4 => f32::from_be_bytes([b[0], b[1], b[2], b[3]]) as f64,
        8 => f64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
        _ => 0.0,
    }
}

fn string(b: &[u8]) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).into_owned()
}

/// Iterate the children of `[start, end)`.
fn children(r: &Reader, start: usize, end: usize, mut f: impl FnMut(u32, usize, usize)) {
    let mut pos = start;
    while pos < end {
        let Some((id, s, e, _)) = r.element(pos, end) else {
            break;
        };
        f(id, s, e);
        if e <= pos {
            break;
        }
        pos = e;
    }
}

impl Mkv {
    /// Parse a whole file.
    pub fn parse(data: &[u8]) -> Result<Mkv, String> {
        let r = Reader { d: data };
        let mut mkv = Mkv { timestamp_scale: 1_000_000, ..Default::default() };
        let mut pos = 0usize;
        let mut saw_ebml = false;
        while pos < data.len() {
            let Some((id, s, e, _)) = r.element(pos, data.len()) else {
                break;
            };
            match id {
                EBML => {
                    saw_ebml = true;
                    children(&r, s, e, |id, s, e| {
                        if id == DOC_TYPE {
                            mkv.doc_type = string(&data[s..e]);
                        }
                    });
                }
                SEGMENT => mkv.segment(&r, s, e)?,
                _ => {}
            }
            pos = e.max(pos + 1);
        }
        if !saw_ebml {
            return Err("not a Matroska/WebM file (no EBML header)".into());
        }
        Ok(mkv)
    }

    /// The first track of `kind`, preferring one flagged default.
    pub fn track(&self, kind: TrackKind) -> Option<&MkvTrack> {
        let mut it = self.tracks.iter().filter(|t| t.kind == kind);
        let first = it.clone().next();
        it.find(|t| t.flag_default).or(first)
    }

    /// The frames of track `number`, in file order.
    pub fn frames_of(&self, number: u64) -> impl Iterator<Item = &MkvFrame> {
        self.frames.iter().filter(move |f| f.track == number)
    }

    fn segment(&mut self, r: &Reader, start: usize, end: usize) -> Result<(), String> {
        let d = r.d;
        let mut pos = start;
        while pos < end {
            let Some((id, s, mut e, unknown)) = r.element(pos, end) else {
                break;
            };
            if unknown {
                // Close an unknown-size element at the next segment child.
                e = self.unknown_end(r, s, end);
            }
            match id {
                INFO => children(r, s, e, |id, s, e| match id {
                    TIMESTAMP_SCALE => self.timestamp_scale = uint(&d[s..e]).max(1),
                    DURATION => self.duration = Some(float(&d[s..e])),
                    _ => {}
                }),
                TRACKS => children(r, s, e, |id, s, e| {
                    if id == TRACK_ENTRY {
                        self.tracks.push(track_entry(r, s, e));
                    }
                }),
                CLUSTER => self.cluster(r, s, e),
                _ => {}
            }
            if e <= pos {
                break;
            }
            pos = e;
        }
        Ok(())
    }

    /// Where an unknown-size element starting its data at `s` ends: at the
    /// first element that is a segment child.
    fn unknown_end(&self, r: &Reader, s: usize, end: usize) -> usize {
        let mut pos = s;
        while pos < end {
            let Some((id, _, e, unknown)) = r.element(pos, end) else {
                return end;
            };
            if SEGMENT_CHILDREN.contains(&id) {
                return pos;
            }
            if unknown || e <= pos {
                return end;
            }
            pos = e;
        }
        end
    }

    fn cluster(&mut self, r: &Reader, start: usize, end: usize) {
        let d = r.d;
        let mut cluster_ts = 0i64;
        let mut pos = start;
        while pos < end {
            let Some((id, s, e, _)) = r.element(pos, end) else {
                break;
            };
            match id {
                TIMESTAMP => cluster_ts = uint(&d[s..e]) as i64,
                SIMPLE_BLOCK => self.block(&d[s..e], cluster_ts, None, None, 0, None),
                BLOCK_GROUP => {
                    let mut block = None;
                    let mut duration = None;
                    let mut reference = false;
                    let mut padding = 0;
                    let mut alpha = None;
                    children(r, s, e, |id, s, e| match id {
                        BLOCK => block = Some((s, e)),
                        BLOCK_DURATION => duration = Some(uint(&d[s..e]) as i64),
                        REFERENCE_BLOCK => reference = true,
                        DISCARD_PADDING => padding = sint(&d[s..e]),
                        BLOCK_ADDITIONS => children(r, s, e, |id, s, e| {
                            if id == BLOCK_MORE {
                                let mut add_id = 1;
                                let mut payload = None;
                                children(r, s, e, |id, s, e| match id {
                                    BLOCK_ADD_ID => add_id = uint(&d[s..e]),
                                    BLOCK_ADDITIONAL => payload = Some(d[s..e].to_vec()),
                                    _ => {}
                                });
                                if add_id == 1 {
                                    alpha = payload;
                                }
                            }
                        }),
                        _ => {}
                    });
                    if let Some((bs, be)) = block {
                        self.block(&d[bs..be], cluster_ts, duration, Some(!reference), padding, alpha);
                    }
                }
                _ => {}
            }
            if e <= pos {
                break;
            }
            pos = e;
        }
    }

    /// One `SimpleBlock` / `Block`. `keyframe` is `Some` for a `Block`
    /// (no `ReferenceBlock` = key); a `SimpleBlock` carries its own flag.
    fn block(
        &mut self,
        b: &[u8],
        cluster_ts: i64,
        duration: Option<i64>,
        keyframe: Option<bool>,
        discard_padding: i64,
        alpha: Option<Vec<u8>>,
    ) {
        let r = Reader { d: b };
        let Some((Some(track), tl)) = r.size(0) else {
            return;
        };
        if b.len() < tl + 3 {
            return;
        }
        let rel = i16::from_be_bytes([b[tl], b[tl + 1]]) as i64;
        let flags = b[tl + 2];
        let key = keyframe.unwrap_or(flags & 0x80 != 0);
        let lacing = (flags >> 1) & 3;
        let mut p = tl + 3;
        let mut sizes: Vec<usize> = Vec::new();
        if lacing != 0 {
            let Some(&n) = b.get(p) else {
                return;
            };
            p += 1;
            let count = n as usize + 1;
            match lacing {
                1 => {
                    // Xiph.
                    for _ in 0..count - 1 {
                        let mut s = 0usize;
                        loop {
                            let Some(&x) = b.get(p) else {
                                return;
                            };
                            p += 1;
                            s += x as usize;
                            if x != 255 {
                                break;
                            }
                        }
                        sizes.push(s);
                    }
                }
                3 => {
                    // EBML: first size a VINT, then signed differences.
                    let Some((Some(first), l)) = r.size(p) else {
                        return;
                    };
                    p += l;
                    let mut cur = first as i64;
                    sizes.push(cur as usize);
                    for _ in 1..count - 1 {
                        let Some((Some(v), l)) = r.size(p) else {
                            return;
                        };
                        let bias = (1i64 << (7 * l - 1)) - 1;
                        cur += v as i64 - bias;
                        p += l;
                        if cur < 0 {
                            return;
                        }
                        sizes.push(cur as usize);
                    }
                }
                _ => {
                    // Fixed size.
                    let total = b.len().saturating_sub(p);
                    sizes = vec![total / count; count - 1];
                }
            }
            let used: usize = sizes.iter().sum();
            if p + used > b.len() {
                return;
            }
            sizes.push(b.len() - p - used);
        } else {
            sizes.push(b.len() - p);
        }
        let prefix = self.tracks.iter().find(|t| t.number == track).map(|t| t.strip_prefix.clone()).unwrap_or_default();
        let n = sizes.len() as i64;
        let lace_dur = duration.map(|d| d / n);
        let pts0 = cluster_ts + rel;
        for (i, &s) in sizes.iter().enumerate() {
            let mut data = prefix.clone();
            data.extend_from_slice(&b[p..p + s]);
            p += s;
            let last = i + 1 == sizes.len();
            self.frames.push(MkvFrame {
                track,
                pts: pts0 + lace_dur.map_or(0, |d| d * i as i64),
                duration: lace_dur,
                keyframe: key,
                discard_padding: if last { discard_padding } else { 0 },
                data,
                alpha: if i == 0 { alpha.clone() } else { None },
            });
        }
    }
}

fn track_entry(r: &Reader, s: usize, e: usize) -> MkvTrack {
    let d = r.d;
    let mut t = MkvTrack {
        number: 0,
        kind: TrackKind::Other,
        codec_id: String::new(),
        codec_private: Vec::new(),
        default_duration: None,
        codec_delay: 0,
        seek_preroll: 0,
        flag_default: true,
        width: 0,
        height: 0,
        display_width: None,
        display_height: None,
        alpha_mode: false,
        frame_rate: None,
        sample_rate: 8000.0,
        channels: 1,
        bit_depth: 0,
        language: None,
        strip_prefix: Vec::new(),
    };
    children(r, s, e, |id, s, e| {
        let b = &d[s..e];
        match id {
            TRACK_NUMBER => t.number = uint(b),
            TRACK_TYPE => {
                t.kind = match uint(b) {
                    1 => TrackKind::Video,
                    2 => TrackKind::Audio,
                    0x11 => TrackKind::Subtitle,
                    _ => TrackKind::Other,
                }
            }
            FLAG_DEFAULT => t.flag_default = uint(b) != 0,
            CODEC_ID => t.codec_id = string(b),
            CODEC_PRIVATE => t.codec_private = b.to_vec(),
            DEFAULT_DURATION => t.default_duration = Some(uint(b)),
            CODEC_DELAY => t.codec_delay = uint(b),
            SEEK_PREROLL => t.seek_preroll = uint(b),
            LANGUAGE => t.language = Some(string(b)),
            VIDEO => children(r, s, e, |id, s, e| {
                let b = &d[s..e];
                match id {
                    PIXEL_WIDTH => t.width = uint(b) as u32,
                    PIXEL_HEIGHT => t.height = uint(b) as u32,
                    DISPLAY_WIDTH => t.display_width = Some(uint(b) as u32),
                    DISPLAY_HEIGHT => t.display_height = Some(uint(b) as u32),
                    ALPHA_MODE => t.alpha_mode = uint(b) == 1,
                    FRAME_RATE => t.frame_rate = Some(float(b)),
                    _ => {}
                }
            }),
            AUDIO => children(r, s, e, |id, s, e| {
                let b = &d[s..e];
                match id {
                    SAMPLING_FREQUENCY => t.sample_rate = float(b),
                    CHANNELS => t.channels = uint(b) as u32,
                    BIT_DEPTH => t.bit_depth = uint(b) as u32,
                    _ => {}
                }
            }),
            CONTENT_ENCODINGS => children(r, s, e, |id, s, e| {
                if id != CONTENT_ENCODING {
                    return;
                }
                children(r, s, e, |id, s, e| {
                    if id != CONTENT_COMPRESSION {
                        return;
                    }
                    let mut algo = 0;
                    let mut settings = Vec::new();
                    children(r, s, e, |id, s, e| match id {
                        CONTENT_COMP_ALGO => algo = uint(&d[s..e]),
                        CONTENT_COMP_SETTINGS => settings = d[s..e].to_vec(),
                        _ => {}
                    });
                    if algo == 3 {
                        t.strip_prefix = settings;
                    }
                });
            }),
            _ => {}
        }
    });
    t
}

/// An IVF file: `(fourcc, width, height, timebase (num, den), frames as
/// (pts, data))`.
#[allow(clippy::type_complexity)]
pub fn parse_ivf(data: &[u8]) -> Result<([u8; 4], u32, u32, (u32, u32), Vec<(i64, Vec<u8>)>), String> {
    if data.len() < 32 || &data[0..4] != b"DKIF" {
        return Err("not an IVF file".into());
    }
    let le16 = |o: usize| u16::from_le_bytes([data[o], data[o + 1]]) as u32;
    let le32 = |o: usize| u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
    let hdr = le16(6) as usize;
    let fourcc = [data[8], data[9], data[10], data[11]];
    let (w, h) = (le16(12), le16(14));
    let (den, num) = (le32(16), le32(20));
    let mut pos = hdr.max(32);
    let mut frames = Vec::new();
    while pos + 12 <= data.len() {
        let size = le32(pos) as usize;
        let pts = u64::from_le_bytes(data[pos + 4..pos + 12].try_into().unwrap()) as i64;
        pos += 12;
        if pos + size > data.len() {
            break;
        }
        frames.push((pts, data[pos..pos + size].to_vec()));
        pos += size;
    }
    Ok((fourcc, w, h, (num, den), frames))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_back_our_own_webm() {
        // A frame muxed by `webm`, read back.
        use crate::webm::{mux_webm, WebmCodec, WebmFrame, WebmParams};
        let frames = [WebmFrame { data: &[1, 2, 3, 4], alpha: None, timecode: 0, keyframe: true }];
        let bytes = mux_webm(&WebmParams { width: 16, height: 16, codec: WebmCodec::Vp9, timecode_scale_ns: 0, frames: &frames });
        let mkv = Mkv::parse(&bytes).unwrap();
        assert_eq!(mkv.doc_type, "webm");
        let t = mkv.track(TrackKind::Video).unwrap();
        assert_eq!(t.codec_id, "V_VP9");
        assert_eq!((t.width, t.height), (16, 16));
        let f: Vec<_> = mkv.frames_of(t.number).collect();
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].data, vec![1, 2, 3, 4]);
        assert!(f[0].keyframe);
    }

    #[test]
    fn ebml_lacing_sizes() {
        // Track 1, ts 0, flags: keyframe + EBML lacing, 3 frames of 2, 3, 1 bytes.
        // First size VINT 0x82 (=2); difference +1 as a 1-byte signed VINT:
        // 0x80 | (1 + 63) = 0xC0.
        let mut b = vec![0x81, 0x00, 0x00, 0x80 | 0x06, 2, 0x82, 0xC0];
        b.extend_from_slice(&[10, 11, 20, 21, 22, 30]);
        let mut mkv = Mkv { timestamp_scale: 1_000_000, ..Default::default() };
        mkv.block(&b, 100, Some(30), None, 0, None);
        let sizes: Vec<_> = mkv.frames.iter().map(|f| f.data.len()).collect();
        assert_eq!(sizes, vec![2, 3, 1]);
        assert_eq!(mkv.frames[2].data, vec![30]);
        assert_eq!(mkv.frames.iter().map(|f| f.pts).collect::<Vec<_>>(), vec![100, 110, 120]);
    }
}
