//! Minimal ISOBMFF / MP4 muxer for H.264 and HEVC video tracks, optionally with
//! a `tx3g` timed-text (subtitle) track. Pure Rust, wasm-safe.
//!
//! Enough of the box hierarchy to produce a `.mp4`/`.mov` that AVFoundation and
//! ffmpeg open: `ftyp` + `mdat` + `moov`, with an `avc1`/`hvc1` sample entry
//! carrying an `avcC`/`hvcC` config record (and, for transparent HEVC, Apple's
//! `almo` alpha-mode box). NAL units inside samples are 4-byte length-prefixed
//! (not Annex-B).
//!
//! [`mux_hevc_with_captions`] adds a second track holding 3GPP timed text
//! (`tx3g`, what ffmpeg calls `mov_text`) — the soft-subtitle format MP4
//! players understand, so the viewer can toggle captions off.
//!
//! Layout is `ftyp, mdat, moov` — `mdat` before `moov` so chunk offsets (`stco`)
//! are fixed before `moov`'s size is known.

use crate::captions::CaptionTrack;

/// Wrap `payload` in a box with `type_` (a 4-byte tag).
fn bx(type_: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(8 + payload.len());
    v.extend_from_slice(&((8 + payload.len()) as u32).to_be_bytes());
    v.extend_from_slice(type_);
    v.extend_from_slice(payload);
    v
}

/// A full-box (`version` + 24-bit `flags`) wrapping `payload`.
fn fullbox(type_: &[u8; 4], version: u8, flags: u32, payload: &[u8]) -> Vec<u8> {
    let mut p = Vec::with_capacity(4 + payload.len());
    p.push(version);
    p.extend_from_slice(&flags.to_be_bytes()[1..]);
    p.extend_from_slice(payload);
    bx(type_, &p)
}

fn concat(parts: &[Vec<u8>]) -> Vec<u8> {
    let mut v = Vec::new();
    for p in parts {
        v.extend_from_slice(p);
    }
    v
}

/// Shared timing/layout fields for both H.264 and HEVC muxers.
struct CommonMp4Params<'a> {
    pub width: u32,
    pub height: u32,
    pub timescale: u32,
    pub frame_duration: u32,
    pub samples: &'a [Vec<u8>],
}

/// Parameters for [`mux_hevc`].
pub struct Mp4Params<'a> {
    pub width: u32,
    pub height: u32,
    /// Media timescale (ticks per second). Apple uses 600.
    pub timescale: u32,
    /// Duration of each sample in `timescale` ticks.
    pub frame_duration: u32,
    /// The `hvcC` box payload (without its size/type header) — see
    /// [`crate::codec::hevc::hvcc`].
    pub hvcc_payload: &'a [u8],
    /// Optional Apple `almo` alpha-mode box payload (4 bytes) to append to the
    /// sample entry, signaling a transparent track.
    pub almo_payload: Option<&'a [u8]>,
    /// Per-frame sample data: each entry is that frame's NAL units, already
    /// 4-byte length-prefixed and concatenated.
    pub samples: &'a [Vec<u8>],
}

/// Parameters for [`mux_h264`].
pub struct H264Mp4Params<'a> {
    pub width: u32,
    pub height: u32,
    pub timescale: u32,
    pub frame_duration: u32,
    /// The `avcC` box payload (without its size/type header) — see
    /// [`crate::codec::h264::avcc`].
    pub avcc_payload: &'a [u8],
    pub samples: &'a [Vec<u8>],
}

impl<'a> From<&Mp4Params<'a>> for CommonMp4Params<'a> {
    fn from(p: &Mp4Params<'a>) -> Self {
        Self {
            width: p.width,
            height: p.height,
            timescale: p.timescale,
            frame_duration: p.frame_duration,
            samples: p.samples,
        }
    }
}

impl<'a> From<&H264Mp4Params<'a>> for CommonMp4Params<'a> {
    fn from(p: &H264Mp4Params<'a>) -> Self {
        Self {
            width: p.width,
            height: p.height,
            timescale: p.timescale,
            frame_duration: p.frame_duration,
            samples: p.samples,
        }
    }
}

/// Mux one HEVC video track into a `.mp4`/`.mov` byte stream.
pub fn mux_hevc(p: &Mp4Params) -> Vec<u8> {
    mux_hevc_internal(p, None)
}

/// Mux an HEVC video track plus a `tx3g` soft-subtitle track.
///
/// The caption track becomes a second, selectable text track — the player can
/// show or hide it, unlike burned-in captions. `tx3g` shows one sample at a
/// time, so cues that overlap are serialized: a cue starting before the
/// previous one ends is pushed back to that end time. Gaps between cues are
/// filled with empty samples, which is how a `tx3g` track blanks the display.
///
/// ```no_run
/// use threers::codec::mp4::{mux_hevc_with_captions, Mp4Params};
/// use threers::captions::CaptionTrack;
/// let track = CaptionTrack::new().language("en").cue(0.0, 2.0, "Hello");
/// let bytes = mux_hevc_with_captions(
///     &Mp4Params {
///         width: 640, height: 360, timescale: 600, frame_duration: 20,
///         hvcc_payload: &[], almo_payload: None, samples: &[],
///     },
///     &track,
/// );
/// # let _ = bytes;
/// ```
pub fn mux_hevc_with_captions(p: &Mp4Params, captions: &CaptionTrack) -> Vec<u8> {
    mux_hevc_internal(p, Some(captions))
}

/// Mux one H.264 video track into a `.mp4`/`.mov` byte stream.
pub fn mux_h264(p: &H264Mp4Params) -> Vec<u8> {
    mux_h264_internal(p, None)
}

/// Mux an H.264 video track plus a `tx3g` soft-subtitle track.
pub fn mux_h264_with_captions(p: &H264Mp4Params, captions: &CaptionTrack) -> Vec<u8> {
    mux_h264_internal(p, Some(captions))
}

fn mux_hevc_internal(p: &Mp4Params, captions: Option<&CaptionTrack>) -> Vec<u8> {
    let common = CommonMp4Params::from(p);
    let stsd = build_hevc_stsd(p);
    mux_common(
        &common,
        &[b"isom", b"iso2", b"mp41", b"hvc1"],
        stsd,
        captions,
    )
}

fn mux_h264_internal(p: &H264Mp4Params, captions: Option<&CaptionTrack>) -> Vec<u8> {
    let common = CommonMp4Params::from(p);
    let stsd = build_h264_stsd(p);
    mux_common(
        &common,
        &[b"isom", b"iso2", b"mp41", b"avc1"],
        stsd,
        captions,
    )
}

fn build_hevc_stsd(p: &Mp4Params) -> Vec<u8> {
    let hvcc = bx(b"hvcC", p.hvcc_payload);
    let mut sample_entry = Vec::new();
    sample_entry.extend_from_slice(&[0u8; 6]); // reserved
    sample_entry.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
    sample_entry.extend_from_slice(&[0u8; 16]); // pre_defined + reserved (2+2+12)
    sample_entry.extend_from_slice(&(p.width as u16).to_be_bytes());
    sample_entry.extend_from_slice(&(p.height as u16).to_be_bytes());
    sample_entry.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // horizresolution 72dpi
    sample_entry.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // vertresolution
    sample_entry.extend_from_slice(&0u32.to_be_bytes()); // reserved
    sample_entry.extend_from_slice(&1u16.to_be_bytes()); // frame_count
    sample_entry.extend_from_slice(&[0u8; 32]); // compressorname
    sample_entry.extend_from_slice(&0x0018u16.to_be_bytes()); // depth
    sample_entry.extend_from_slice(&0xFFFFu16.to_be_bytes()); // pre_defined -1
    sample_entry.extend_from_slice(&hvcc);
    if let Some(almo) = p.almo_payload {
        sample_entry.extend_from_slice(&bx(b"almo", almo));
    }
    let hvc1 = bx(b"hvc1", &sample_entry);
    let mut payload = Vec::new();
    payload.extend_from_slice(&1u32.to_be_bytes()); // entry_count
    payload.extend_from_slice(&hvc1);
    fullbox(b"stsd", 0, 0, &payload)
}

fn build_h264_stsd(p: &H264Mp4Params) -> Vec<u8> {
    let avcc = bx(b"avcC", p.avcc_payload);
    let mut sample_entry = Vec::new();
    sample_entry.extend_from_slice(&[0u8; 6]); // reserved
    sample_entry.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
    sample_entry.extend_from_slice(&[0u8; 16]); // pre_defined + reserved
    sample_entry.extend_from_slice(&(p.width as u16).to_be_bytes());
    sample_entry.extend_from_slice(&(p.height as u16).to_be_bytes());
    sample_entry.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // horizresolution
    sample_entry.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // vertresolution
    sample_entry.extend_from_slice(&0u32.to_be_bytes()); // reserved
    sample_entry.extend_from_slice(&1u16.to_be_bytes()); // frame_count
    sample_entry.extend_from_slice(&[0u8; 32]); // compressorname
    sample_entry.extend_from_slice(&0x0018u16.to_be_bytes()); // depth
    sample_entry.extend_from_slice(&0xFFFFu16.to_be_bytes()); // pre_defined -1
    sample_entry.extend_from_slice(&avcc);
    let avc1 = bx(b"avc1", &sample_entry);
    let mut payload = Vec::new();
    payload.extend_from_slice(&1u32.to_be_bytes()); // entry_count
    payload.extend_from_slice(&avc1);
    fullbox(b"stsd", 0, 0, &payload)
}

fn mux_common(
    p: &CommonMp4Params,
    brands: &[&[u8; 4]],
    stsd: Vec<u8>,
    captions: Option<&CaptionTrack>,
) -> Vec<u8> {
    let num_samples = p.samples.len().max(1) as u32;
    let total_duration = p.frame_duration * num_samples;

    // ---- ftyp ----
    let ftyp = bx(b"ftyp", &{
        let mut v = Vec::new();
        v.extend_from_slice(b"isom"); // major_brand
        v.extend_from_slice(&0u32.to_be_bytes()); // minor_version
        for brand in brands {
            v.extend_from_slice(*brand);
        }
        v
    });

    // Timed-text samples share the video's timescale and run the full movie
    // duration so the text track never ends early.
    let text_samples = captions
        .map(|t| build_text_samples(t, p.timescale, total_duration))
        .unwrap_or_default();

    // ---- mdat (video samples, then text samples) ----
    let mut mdat_payload = concat(p.samples);
    let video_bytes = mdat_payload.len() as u32;
    for s in &text_samples {
        mdat_payload.extend_from_slice(&s.data);
    }
    let mdat = bx(b"mdat", &mdat_payload);
    let mdat_data_offset = (ftyp.len() + 8) as u32; // sample data starts after mdat header
    let text_data_offset = mdat_data_offset + video_bytes;

    // ---- sample tables ----
    let stts = fullbox(b"stts", 0, 0, &{
        let mut v = Vec::new();
        v.extend_from_slice(&1u32.to_be_bytes()); // entry_count
        v.extend_from_slice(&num_samples.to_be_bytes()); // sample_count
        v.extend_from_slice(&p.frame_duration.to_be_bytes()); // sample_delta
        v
    });

    // All samples are IDR → all sync samples: stss lists them all.
    let stss = fullbox(b"stss", 0, 0, &{
        let mut v = Vec::new();
        v.extend_from_slice(&num_samples.to_be_bytes());
        for i in 1..=num_samples {
            v.extend_from_slice(&i.to_be_bytes());
        }
        v
    });

    let stsc = fullbox(b"stsc", 0, 0, &{
        let mut v = Vec::new();
        v.extend_from_slice(&1u32.to_be_bytes()); // entry_count
        v.extend_from_slice(&1u32.to_be_bytes()); // first_chunk
        v.extend_from_slice(&num_samples.to_be_bytes()); // samples_per_chunk
        v.extend_from_slice(&1u32.to_be_bytes()); // sample_description_index
        v
    });

    let stsz = fullbox(b"stsz", 0, 0, &{
        let mut v = Vec::new();
        v.extend_from_slice(&0u32.to_be_bytes()); // sample_size 0 = per-sample
        v.extend_from_slice(&num_samples.to_be_bytes()); // sample_count
        for s in p.samples {
            v.extend_from_slice(&(s.len() as u32).to_be_bytes());
        }
        v
    });

    // Single chunk holding all samples, located at the start of mdat's payload.
    let stco = fullbox(b"stco", 0, 0, &{
        let mut v = Vec::new();
        v.extend_from_slice(&1u32.to_be_bytes()); // entry_count
        v.extend_from_slice(&mdat_data_offset.to_be_bytes());
        v
    });

    let stbl = bx(b"stbl", &concat(&[stsd, stts, stss, stsc, stsz, stco]));

    // ---- minf ----
    let vmhd = fullbox(b"vmhd", 0, 1, &[0u8; 8]); // graphicsmode + opcolor
    let dref = fullbox(b"dref", 0, 0, &{
        let mut v = Vec::new();
        v.extend_from_slice(&1u32.to_be_bytes()); // entry_count
        v.extend_from_slice(&fullbox(b"url ", 0, 1, &[])); // self-contained
        v
    });
    let dinf = bx(b"dinf", &dref);
    let vid_hdlr = fullbox(b"hdlr", 0, 0, &{
        let mut v = Vec::new();
        v.extend_from_slice(&0u32.to_be_bytes()); // pre_defined
        v.extend_from_slice(b"vide"); // handler_type
        v.extend_from_slice(&[0u8; 12]); // reserved
        v.extend_from_slice(b"threers\0"); // name
        v
    });
    let minf = bx(b"minf", &concat(&[vmhd, vid_hdlr, dinf, stbl]));

    // ---- mdia ----
    let mdhd = fullbox(b"mdhd", 0, 0, &{
        let mut v = Vec::new();
        v.extend_from_slice(&0u32.to_be_bytes()); // creation_time
        v.extend_from_slice(&0u32.to_be_bytes()); // modification_time
        v.extend_from_slice(&p.timescale.to_be_bytes());
        v.extend_from_slice(&total_duration.to_be_bytes());
        v.extend_from_slice(&0x55c4u16.to_be_bytes()); // language 'und'
        v.extend_from_slice(&0u16.to_be_bytes()); // pre_defined
        v
    });
    let mdia_hdlr = fullbox(b"hdlr", 0, 0, &{
        let mut v = Vec::new();
        v.extend_from_slice(&0u32.to_be_bytes());
        v.extend_from_slice(b"vide");
        v.extend_from_slice(&[0u8; 12]);
        v.extend_from_slice(b"threers\0");
        v
    });
    let mdia = bx(b"mdia", &concat(&[mdhd, mdia_hdlr, minf]));

    // ---- trak ----
    let tkhd = fullbox(b"tkhd", 0, 7 /* enabled|in_movie|in_preview */, &{
        let mut v = Vec::new();
        v.extend_from_slice(&0u32.to_be_bytes()); // creation_time
        v.extend_from_slice(&0u32.to_be_bytes()); // modification_time
        v.extend_from_slice(&1u32.to_be_bytes()); // track_ID
        v.extend_from_slice(&0u32.to_be_bytes()); // reserved
        v.extend_from_slice(&total_duration.to_be_bytes());
        v.extend_from_slice(&[0u8; 8]); // reserved
        v.extend_from_slice(&0u16.to_be_bytes()); // layer
        v.extend_from_slice(&0u16.to_be_bytes()); // alternate_group
        v.extend_from_slice(&0u16.to_be_bytes()); // volume
        v.extend_from_slice(&0u16.to_be_bytes()); // reserved
        v.extend_from_slice(&UNITY_MATRIX);
        v.extend_from_slice(&(p.width << 16).to_be_bytes()); // width 16.16
        v.extend_from_slice(&(p.height << 16).to_be_bytes()); // height 16.16
        v
    });
    let trak = bx(b"trak", &concat(&[tkhd, mdia]));

    let text_trak = (!text_samples.is_empty()).then(|| {
        text_track(
            p.width,
            p.height,
            captions.map(|c| c.language.as_str()).unwrap_or(""),
            &text_samples,
            p.timescale,
            total_duration,
            text_data_offset,
        )
    });
    let next_track_id = if text_trak.is_some() { 3u32 } else { 2 };

    // ---- mvhd ----
    let mvhd = fullbox(b"mvhd", 0, 0, &{
        let mut v = Vec::new();
        v.extend_from_slice(&0u32.to_be_bytes()); // creation_time
        v.extend_from_slice(&0u32.to_be_bytes()); // modification_time
        v.extend_from_slice(&p.timescale.to_be_bytes());
        v.extend_from_slice(&total_duration.to_be_bytes());
        v.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // rate 1.0
        v.extend_from_slice(&0x0100u16.to_be_bytes()); // volume 1.0
        v.extend_from_slice(&0u16.to_be_bytes()); // reserved
        v.extend_from_slice(&[0u8; 8]); // reserved
        v.extend_from_slice(&UNITY_MATRIX);
        v.extend_from_slice(&[0u8; 24]); // pre_defined
        v.extend_from_slice(&next_track_id.to_be_bytes());
        v
    });
    let mut moov_children = vec![mvhd, trak];
    moov_children.extend(text_trak);
    let moov = bx(b"moov", &concat(&moov_children));

    concat(&[ftyp, mdat, moov])
}

/// One `tx3g` sample: a duration in media ticks and its serialized payload.
struct TextSample {
    duration: u32,
    data: Vec<u8>,
}

/// Turn cues into the gap-free sample run a `tx3g` track requires.
///
/// A `tx3g` track has no notion of "nothing showing" other than a sample whose
/// text is empty, so every gap — including the one before the first cue and
/// after the last — becomes a zero-length text sample.
fn build_text_samples(
    track: &CaptionTrack,
    timescale: u32,
    total_duration: u32,
) -> Vec<TextSample> {
    let ticks = |seconds: f64| -> u32 {
        (seconds.max(0.0) * timescale as f64)
            .round()
            .min(u32::MAX as f64) as u32
    };
    let mut samples: Vec<TextSample> = Vec::new();
    let mut cursor = 0u32;

    for cue in &track.cues {
        let text = cue.plain_text();
        if text.trim().is_empty() {
            continue;
        }
        // Overlapping cues cannot display at once — serialize them.
        let start = ticks(cue.start).max(cursor);
        let end = ticks(cue.end).max(start);
        if end == start {
            continue;
        }
        if start > cursor {
            samples.push(TextSample {
                duration: start - cursor,
                data: encode_text_sample(""),
            });
        }
        samples.push(TextSample {
            duration: end - start,
            data: encode_text_sample(&text),
        });
        cursor = end;
    }

    if samples.is_empty() {
        return samples;
    }
    // Run the track to the end of the movie so players do not stop early.
    if cursor < total_duration {
        samples.push(TextSample {
            duration: total_duration - cursor,
            data: encode_text_sample(""),
        });
    }
    samples
}

/// A `tx3g` sample body: 16-bit UTF-8 byte count, then the text itself.
fn encode_text_sample(text: &str) -> Vec<u8> {
    // Players expect LF line breaks inside a sample; CR would show as a box.
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let bytes = text.as_bytes();
    let len = bytes.len().min(u16::MAX as usize);
    let mut v = Vec::with_capacity(2 + len);
    v.extend_from_slice(&(len as u16).to_be_bytes());
    v.extend_from_slice(&bytes[..len]);
    v
}

/// Build the `trak` for the timed-text track.
fn text_track(
    width: u32,
    height: u32,
    language: &str,
    samples: &[TextSample],
    timescale: u32,
    total_duration: u32,
    data_offset: u32,
) -> Vec<u8> {
    let count = samples.len() as u32;

    // ---- stsd > tx3g ----
    let stsd = {
        let mut entry = Vec::new();
        entry.extend_from_slice(&[0u8; 6]); // reserved
        entry.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
        entry.extend_from_slice(&0u32.to_be_bytes()); // displayFlags
        entry.push(1); // horizontal-justification: center
        entry.push(0xFF); // vertical-justification: -1 = bottom
        entry.extend_from_slice(&[0, 0, 0, 0]); // background-color-rgba (transparent)
                                                // BoxRecord default-text-box — the full frame.
        entry.extend_from_slice(&0i16.to_be_bytes()); // top
        entry.extend_from_slice(&0i16.to_be_bytes()); // left
        entry.extend_from_slice(&(height.min(i16::MAX as u32) as i16).to_be_bytes()); // bottom
        entry.extend_from_slice(&(width.min(i16::MAX as u32) as i16).to_be_bytes()); // right
                                                                                     // StyleRecord default-style — white text in the first font table entry.
        entry.extend_from_slice(&0u16.to_be_bytes()); // startChar
        entry.extend_from_slice(&0u16.to_be_bytes()); // endChar
        entry.extend_from_slice(&1u16.to_be_bytes()); // font-ID
        entry.push(0); // face-style-flags
        entry.push(((height / 16).clamp(12, 255)) as u8); // font-size
        entry.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]); // text-color-rgba
                                                            // FontTableBox.
        let ftab = bx(b"ftab", &{
            let name = b"Sans-Serif";
            let mut v = Vec::new();
            v.extend_from_slice(&1u16.to_be_bytes()); // entry-count
            v.extend_from_slice(&1u16.to_be_bytes()); // font-ID
            v.push(name.len() as u8);
            v.extend_from_slice(name);
            v
        });
        entry.extend_from_slice(&ftab);

        let mut payload = Vec::new();
        payload.extend_from_slice(&1u32.to_be_bytes()); // entry_count
        payload.extend_from_slice(&bx(b"tx3g", &entry));
        fullbox(b"stsd", 0, 0, &payload)
    };

    // ---- stts: run-length encoded per-sample durations ----
    let stts = {
        let mut runs: Vec<(u32, u32)> = Vec::new(); // (count, delta)
        for s in samples {
            match runs.last_mut() {
                Some((n, delta)) if *delta == s.duration => *n += 1,
                _ => runs.push((1, s.duration)),
            }
        }
        let mut v = Vec::new();
        v.extend_from_slice(&(runs.len() as u32).to_be_bytes());
        for (n, delta) in runs {
            v.extend_from_slice(&n.to_be_bytes());
            v.extend_from_slice(&delta.to_be_bytes());
        }
        fullbox(b"stts", 0, 0, &v)
    };

    let stsc = fullbox(b"stsc", 0, 0, &{
        let mut v = Vec::new();
        v.extend_from_slice(&1u32.to_be_bytes()); // entry_count
        v.extend_from_slice(&1u32.to_be_bytes()); // first_chunk
        v.extend_from_slice(&count.to_be_bytes()); // samples_per_chunk
        v.extend_from_slice(&1u32.to_be_bytes()); // sample_description_index
        v
    });

    let stsz = fullbox(b"stsz", 0, 0, &{
        let mut v = Vec::new();
        v.extend_from_slice(&0u32.to_be_bytes()); // per-sample sizes follow
        v.extend_from_slice(&count.to_be_bytes());
        for s in samples {
            v.extend_from_slice(&(s.data.len() as u32).to_be_bytes());
        }
        v
    });

    let stco = fullbox(b"stco", 0, 0, &{
        let mut v = Vec::new();
        v.extend_from_slice(&1u32.to_be_bytes()); // entry_count
        v.extend_from_slice(&data_offset.to_be_bytes());
        v
    });

    let stbl = bx(b"stbl", &concat(&[stsd, stts, stsc, stsz, stco]));

    // ---- minf: text media has a null media header ----
    let nmhd = fullbox(b"nmhd", 0, 0, &[]);
    let dref = fullbox(b"dref", 0, 0, &{
        let mut v = Vec::new();
        v.extend_from_slice(&1u32.to_be_bytes());
        v.extend_from_slice(&fullbox(b"url ", 0, 1, &[]));
        v
    });
    let dinf = bx(b"dinf", &dref);
    let minf = bx(b"minf", &concat(&[nmhd, dinf, stbl]));

    // ---- mdia ----
    let mdhd = fullbox(b"mdhd", 0, 0, &{
        let mut v = Vec::new();
        v.extend_from_slice(&0u32.to_be_bytes()); // creation_time
        v.extend_from_slice(&0u32.to_be_bytes()); // modification_time
        v.extend_from_slice(&timescale.to_be_bytes());
        v.extend_from_slice(&total_duration.to_be_bytes());
        v.extend_from_slice(&pack_language(language).to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes()); // pre_defined
        v
    });
    let hdlr = fullbox(b"hdlr", 0, 0, &{
        let mut v = Vec::new();
        v.extend_from_slice(&0u32.to_be_bytes()); // pre_defined
        v.extend_from_slice(b"sbtl"); // subtitle handler
        v.extend_from_slice(&[0u8; 12]); // reserved
        v.extend_from_slice(b"threers captions\0");
        v
    });
    let mdia = bx(b"mdia", &concat(&[mdhd, hdlr, minf]));

    // ---- tkhd: same frame box as the video, drawn in front (layer -1) ----
    let tkhd = fullbox(b"tkhd", 0, 3 /* enabled | in_movie */, &{
        let mut v = Vec::new();
        v.extend_from_slice(&0u32.to_be_bytes()); // creation_time
        v.extend_from_slice(&0u32.to_be_bytes()); // modification_time
        v.extend_from_slice(&2u32.to_be_bytes()); // track_ID
        v.extend_from_slice(&0u32.to_be_bytes()); // reserved
        v.extend_from_slice(&total_duration.to_be_bytes());
        v.extend_from_slice(&[0u8; 8]); // reserved
        v.extend_from_slice(&(-1i16).to_be_bytes()); // layer: in front of video
        v.extend_from_slice(&0u16.to_be_bytes()); // alternate_group
        v.extend_from_slice(&0u16.to_be_bytes()); // volume
        v.extend_from_slice(&0u16.to_be_bytes()); // reserved
        v.extend_from_slice(&UNITY_MATRIX);
        v.extend_from_slice(&(width << 16).to_be_bytes());
        v.extend_from_slice(&(height << 16).to_be_bytes());
        v
    });

    bx(b"trak", &concat(&[tkhd, mdia]))
}

/// Pack a BCP-47 tag into `mdhd`'s 15-bit ISO-639-2 field (5 bits per letter,
/// each offset by `0x60`).
fn pack_language(language: &str) -> u16 {
    let code = crate::captions::iso639_2(language);
    let bits = |b: u8| ((b - 0x60) & 0x1F) as u16;
    (bits(code[0]) << 10) | (bits(code[1]) << 5) | bits(code[2])
}

/// The identity transformation matrix used in `tkhd`/`mvhd` (16.16 / 2.30 fixed).
const UNITY_MATRIX: [u8; 36] = [
    0x00, 0x01, 0x00, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0x00, 0x01, 0x00, 0x00, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0x40, 0x00, 0x00, 0x00,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boxes_are_well_formed() {
        let sample = vec![0u8; 40];
        let out = mux_hevc(&Mp4Params {
            width: 64,
            height: 64,
            timescale: 600,
            frame_duration: 20,
            hvcc_payload: &[0u8; 30],
            almo_payload: Some(&[0, 0, 1, 2]),
            samples: &[sample],
        });
        assert_eq!(&out[4..8], b"ftyp");
        // Top-level boxes must tile exactly.
        let mut o = 0;
        let mut tags = Vec::new();
        while o + 8 <= out.len() {
            let size = u32::from_be_bytes([out[o], out[o + 1], out[o + 2], out[o + 3]]) as usize;
            tags.push(String::from_utf8_lossy(&out[o + 4..o + 8]).to_string());
            assert!(size >= 8 && o + size <= out.len(), "box {size} at {o}");
            o += size;
        }
        assert_eq!(o, out.len(), "top-level boxes tile the file");
        assert_eq!(tags, vec!["ftyp", "mdat", "moov"]);
    }

    #[test]
    fn h264_boxes_are_well_formed() {
        let sample = vec![0u8; 40];
        let out = mux_h264(&H264Mp4Params {
            width: 64,
            height: 64,
            timescale: 600,
            frame_duration: 20,
            avcc_payload: &[0u8; 20],
            samples: &[sample],
        });
        assert_eq!(&out[4..8], b"ftyp");
        let mut o = 0;
        let mut tags = Vec::new();
        while o + 8 <= out.len() {
            let size = u32::from_be_bytes([out[o], out[o + 1], out[o + 2], out[o + 3]]) as usize;
            tags.push(String::from_utf8_lossy(&out[o + 4..o + 8]).to_string());
            assert!(size >= 8 && o + size <= out.len(), "box {size} at {o}");
            o += size;
        }
        assert_eq!(o, out.len(), "top-level boxes tile the file");
        assert_eq!(tags, vec!["ftyp", "mdat", "moov"]);
        assert!(out.windows(4).any(|w| w == b"avc1"));
    }
}
