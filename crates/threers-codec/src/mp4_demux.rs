//! Reading an MP4 back apart — the mirror of [`mp4`](super::mp4).
//!
//! The crate could already write an MP4 and not read one, which meant any
//! pipeline that wanted a clip *in* had to shell out to ffmpeg for it. That is
//! the same trade the `videotoolbox` module exists to avoid on the
//! way out: a process boundary and a raw pipe, 8 MB a frame at 1080p and 99 MB
//! at 8K, copied twice, to move pictures the system decoder could have handed
//! over directly.
//!
//! This finds the video track's sample table and reports where every coded
//! frame lives in the file, along with the parameter sets a decoder needs to
//! start. It does not decode anything — see
//! `videotoolbox_decode::VideoToolboxDecoder` for that (macOS, `videotoolbox`
//! feature).
//!
//! Only what a video track needs: `stsd` for the codec and its configuration,
//! `stts` for durations, `stsz`/`stsc`/`stco` for where the samples are, and
//! `stss` for which of them a decoder can start on. Edit lists, fragmented MP4
//! (`moof`) and multiple sample descriptions are refused rather than guessed
//! at.

/// Which codec a track's samples are in. The two VideoToolbox will decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackCodec {
    /// `avc1` / `avc3`, configured by an `avcC` box.
    H264,
    /// `hvc1` / `hev1`, configured by an `hvcC` box.
    Hevc,
    /// VP9 (Matroska `V_VP9`; no decoder configuration).
    Vp9,
    /// AV1 (`av01` with an `av1C` box, Matroska `V_AV1`, or IVF).
    Av1,
}

/// One coded frame: where it is in the file, when it decodes and when it shows.
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    /// Byte offset from the start of the file.
    pub offset: u64,
    /// Size in bytes.
    pub size: u32,
    /// In the track's timescale.
    pub duration: u32,
    /// Decode timestamp: the running sum of the durations before this one.
    /// Samples are stored in decode order, so this only ever goes up.
    pub dts: i64,
    /// Presentation timestamp — `dts` plus the `ctts` composition offset.
    ///
    /// **Not the same order as `dts` when the stream has B-frames**, which is
    /// most streams: a B-frame is coded after the picture it is displayed
    /// before. Anything showing these frames has to sort by this and not by
    /// their position in the file.
    pub pts: i64,
    /// A sync sample — a decoder can begin here. Everything is one for
    /// all-intra footage; for anything else this is what `stss` lists.
    pub sync: bool,
}

/// A demuxed video track.
#[derive(Debug, Clone)]
pub struct VideoTrack {
    /// Which decoder this track needs.
    pub codec: TrackCodec,
    /// Picture width in pixels.
    pub width: u32,
    /// Picture height in pixels.
    pub height: u32,
    /// The `avcC` or `hvcC` payload, verbatim — what
    /// `CMVideoFormatDescriptionCreateFromH264ParameterSets` and its HEVC
    /// twin want, and the only part of the file a decoder needs before the
    /// first frame.
    pub config: Vec<u8>,
    /// Ticks per second for [`Sample::duration`].
    pub timescale: u32,
    /// In decode order — the order they sit in the file, which is the order a
    /// decoder must be fed them in.
    pub samples: Vec<Sample>,
}

impl VideoTrack {
    /// Indices into [`samples`](Self::samples), in the order the frames are
    /// meant to be *shown*.
    ///
    /// The same as decode order for an all-intra or IPPP stream, and not for
    /// anything with B-frames. A renderer walking `samples` directly gets the
    /// pictures shuffled — which looks like the clip stuttering in a short
    /// repeating cycle, one cycle per mini-GOP.
    pub fn presentation_order(&self) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.samples.len()).collect();
        idx.sort_by_key(|&i| self.samples[i].pts);
        idx
    }

    /// How far a decoder has to read ahead before the next picture to show is
    /// available — the most any frame is displayed after something coded later.
    pub fn max_reorder(&self) -> usize {
        let order = self.presentation_order();
        order
            .iter()
            .enumerate()
            .map(|(shown, &decoded)| decoded.saturating_sub(shown))
            .max()
            .unwrap_or(0)
    }

    /// Frames per second, averaged over the track. `None` if it has no
    /// duration to divide by.
    pub fn fps(&self) -> Option<f64> {
        let ticks: u64 = self.samples.iter().map(|s| s.duration as u64).sum();
        if ticks == 0 || self.samples.is_empty() {
            return None;
        }
        Some(self.samples.len() as f64 * self.timescale as f64 / ticks as f64)
    }
}

/// Walk the boxes at `data`, calling `f` with the type and body of each.
///
/// Handles the two size escapes: `0` means "to the end of the parent", and `1`
/// means a 64-bit size follows the type. A box that claims to run past its
/// parent stops the walk rather than panicking on the slice.
pub(crate) fn boxes<'a>(data: &'a [u8], mut f: impl FnMut(&[u8; 4], &'a [u8])) {
    let mut off = 0usize;
    while off + 8 <= data.len() {
        let size32 = u32::from_be_bytes(data[off..off + 4].try_into().unwrap()) as usize;
        let ty: [u8; 4] = data[off + 4..off + 8].try_into().unwrap();
        let (header, size) = match size32 {
            0 => (8, data.len() - off),
            1 => {
                if off + 16 > data.len() {
                    return;
                }
                let s = u64::from_be_bytes(data[off + 8..off + 16].try_into().unwrap()) as usize;
                (16, s)
            }
            n => (8, n),
        };
        if size < header || off + size > data.len() {
            return;
        }
        f(&ty, &data[off + header..off + size]);
        off += size;
    }
}

pub(crate) fn find<'a>(data: &'a [u8], want: &[u8; 4]) -> Option<&'a [u8]> {
    let mut hit = None;
    boxes(data, |ty, body| {
        if ty == want && hit.is_none() {
            hit = Some(body);
        }
    });
    hit
}

pub(crate) fn u32at(b: &[u8], i: usize) -> u32 {
    u32::from_be_bytes(b[i..i + 4].try_into().unwrap())
}

/// Read the first video track out of an MP4.
///
/// `data` is the whole file; sample offsets are relative to it.
pub fn demux_video(data: &[u8]) -> Result<VideoTrack, String> {
    if find(data, b"moof").is_some() {
        return Err("fragmented MP4 (moof) is not supported".into());
    }
    let moov = find(data, b"moov").ok_or("no moov box — not an MP4?")?;

    // The first trak whose handler says 'vide'.
    let mut video_stbl: Option<&[u8]> = None;
    let mut timescale = 0u32;
    boxes(moov, |ty, trak| {
        if ty != b"trak" || video_stbl.is_some() {
            return;
        }
        let Some(mdia) = find(trak, b"mdia") else { return };
        let is_video = find(mdia, b"hdlr")
            .map(|h| h.len() >= 12 && &h[8..12] == b"vide")
            .unwrap_or(false);
        if !is_video {
            return;
        }
        if let Some(mdhd) = find(mdia, b"mdhd") {
            // version 0: creation(4) modification(4) timescale(4); version 1
            // widens the first two to 8 bytes each.
            timescale = match mdhd.first() {
                Some(1) if mdhd.len() >= 24 => u32at(mdhd, 20),
                _ if mdhd.len() >= 16 => u32at(mdhd, 12),
                _ => 0,
            };
        }
        video_stbl = find(mdia, b"minf").and_then(|m| find(m, b"stbl"));
    });
    let stbl = video_stbl.ok_or("no video track")?;
    if timescale == 0 {
        return Err("video track has no timescale".into());
    }

    // --- stsd: codec, size, and the decoder configuration ------------------
    let stsd = find(stbl, b"stsd").ok_or("no stsd")?;
    if stsd.len() < 8 {
        return Err("stsd too short".into());
    }
    if u32at(stsd, 4) != 1 {
        return Err("more than one sample description is not supported".into());
    }
    let entry = &stsd[8..];
    if entry.len() < 86 {
        return Err("sample entry too short".into());
    }
    let format: [u8; 4] = entry[4..8].try_into().unwrap();
    let codec = match &format {
        b"avc1" | b"avc3" => TrackCodec::H264,
        b"hvc1" | b"hev1" => TrackCodec::Hevc,
        b"vp09" => TrackCodec::Vp9,
        b"av01" => TrackCodec::Av1,
        other => {
            return Err(format!(
                "unsupported sample format {:?} — only H.264, HEVC, VP9 and AV1",
                String::from_utf8_lossy(other)
            ))
        }
    };
    let width = u16::from_be_bytes(entry[32..34].try_into().unwrap()) as u32;
    let height = u16::from_be_bytes(entry[34..36].try_into().unwrap()) as u32;
    // Extensions follow the 78-byte visual sample entry body.
    let want_cfg: &[u8; 4] = match codec {
        TrackCodec::H264 => b"avcC",
        TrackCodec::Hevc => b"hvcC",
        TrackCodec::Vp9 => b"vpcC",
        TrackCodec::Av1 => b"av1C",
    };
    let config = find(&entry[86..], want_cfg)
        .ok_or_else(|| format!("no {} in the sample entry", String::from_utf8_lossy(want_cfg)))?
        .to_vec();

    // --- stts: one duration per sample -------------------------------------
    let stts = find(stbl, b"stts").ok_or("no stts")?;
    let mut durations: Vec<u32> = Vec::new();
    if stts.len() >= 8 {
        let n = u32at(stts, 4) as usize;
        for i in 0..n {
            let o = 8 + i * 8;
            if o + 8 > stts.len() {
                break;
            }
            let count = u32at(stts, o);
            let delta = u32at(stts, o + 4);
            durations.extend(std::iter::repeat_n(delta, count as usize));
        }
    }

    // --- ctts: how far each sample's display time is from its decode time ---
    //
    // Optional, and its absence means they are the same. Version 1 makes the
    // offset signed, which is how a stream expresses a picture shown *before*
    // the frame it was coded after.
    let mut offsets: Vec<i32> = Vec::new();
    if let Some(ctts) = find(stbl, b"ctts") {
        if ctts.len() >= 8 {
            let n = u32at(ctts, 4) as usize;
            for i in 0..n {
                let o = 8 + i * 8;
                if o + 8 > ctts.len() {
                    break;
                }
                let count = u32at(ctts, o) as usize;
                let off = u32at(ctts, o + 4) as i32;
                offsets.extend(std::iter::repeat_n(off, count));
            }
        }
    }

    // --- stsz: one size per sample -----------------------------------------
    let stsz = find(stbl, b"stsz").ok_or("no stsz")?;
    if stsz.len() < 12 {
        return Err("stsz too short".into());
    }
    let uniform = u32at(stsz, 4);
    let count = u32at(stsz, 8) as usize;
    let sizes: Vec<u32> = if uniform != 0 {
        vec![uniform; count]
    } else {
        (0..count)
            .map(|i| {
                let o = 12 + i * 4;
                if o + 4 <= stsz.len() {
                    u32at(stsz, o)
                } else {
                    0
                }
            })
            .collect()
    };

    // --- stsc + stco: where each sample sits --------------------------------
    let stsc = find(stbl, b"stsc").ok_or("no stsc")?;
    let mut runs: Vec<(u32, u32)> = Vec::new(); // (first_chunk, samples_per_chunk)
    if stsc.len() >= 8 {
        let n = u32at(stsc, 4) as usize;
        for i in 0..n {
            let o = 8 + i * 12;
            if o + 12 > stsc.len() {
                break;
            }
            runs.push((u32at(stsc, o), u32at(stsc, o + 4)));
        }
    }
    let chunk_offsets: Vec<u64> = if let Some(co64) = find(stbl, b"co64") {
        let n = u32at(co64, 4) as usize;
        (0..n)
            .map(|i| {
                let o = 8 + i * 8;
                if o + 8 <= co64.len() {
                    u64::from_be_bytes(co64[o..o + 8].try_into().unwrap())
                } else {
                    0
                }
            })
            .collect()
    } else {
        let stco = find(stbl, b"stco").ok_or("no stco or co64")?;
        let n = u32at(stco, 4) as usize;
        (0..n)
            .map(|i| {
                let o = 8 + i * 4;
                if o + 4 <= stco.len() {
                    u32at(stco, o) as u64
                } else {
                    0
                }
            })
            .collect()
    };

    // --- stss: which samples a decoder can start on -------------------------
    let sync: Option<std::collections::HashSet<u32>> = find(stbl, b"stss").map(|stss| {
        let n = if stss.len() >= 8 { u32at(stss, 4) as usize } else { 0 };
        (0..n)
            .filter_map(|i| {
                let o = 8 + i * 4;
                (o + 4 <= stss.len()).then(|| u32at(stss, o))
            })
            .collect()
    });

    // Walk the chunks, laying samples out consecutively within each.
    let mut samples = Vec::with_capacity(sizes.len());
    let mut idx = 0usize;
    let mut dts = 0i64;
    for (ci, &chunk_off) in chunk_offsets.iter().enumerate() {
        let chunk = ci as u32 + 1;
        // The last run whose first_chunk has been reached governs this chunk.
        let per = runs
            .iter()
            .rev()
            .find(|(first, _)| *first <= chunk)
            .map(|(_, n)| *n)
            .unwrap_or(0);
        let mut at = chunk_off;
        for _ in 0..per {
            if idx >= sizes.len() {
                break;
            }
            let duration = durations.get(idx).copied().unwrap_or(0);
            samples.push(Sample {
                offset: at,
                size: sizes[idx],
                duration,
                dts,
                pts: dts + offsets.get(idx).copied().unwrap_or(0) as i64,
                // No stss means every sample is a sync sample.
                sync: sync
                    .as_ref()
                    .map(|s| s.contains(&(idx as u32 + 1)))
                    .unwrap_or(true),
            });
            at += sizes[idx] as u64;
            dts += duration as i64;
            idx += 1;
        }
    }
    if samples.is_empty() {
        return Err("the video track has no samples".into());
    }

    Ok(VideoTrack {
        codec,
        width,
        height,
        config,
        timescale,
        samples,
    })
}
