//! The AAC track of an MP4/MOV, and the samples ffmpeg n7.1's mov demuxer
//! (`libavformat/mov.c`) hands the decoder for it.
//!
//! With one edit (`elst`: media time `M`, duration `D`) `mov_fix_index`
//! keeps the packets from the last one at or before `max(M - timescale, 0)`
//! (every audio packet is a keyframe) through the first whose end reaches
//! `M + D`, and asks the decoder to drop the first `M - start` output samples
//! (discarded packets and `skip_samples`). An `iTunSMPB` tag read after the
//! track replaces the skip with its priming count. No samples are trimmed at
//! the end.

use crate::mp4_demux::{boxes, find, u32at};

/// One AAC access unit in the file.
#[derive(Debug, Clone, Copy)]
pub struct AudioSample {
    /// Byte offset from the start of the file.
    pub offset: u64,
    /// Size in bytes.
    pub size: u32,
    /// Decode timestamp, in the track's timescale.
    pub dts: i64,
    /// Duration, in the track's timescale.
    pub duration: u32,
}

/// The AAC track of an MP4/M4A file.
#[derive(Debug, Clone)]
pub struct AacTrack {
    /// `AudioSpecificConfig` from `esds`.
    pub asc: Vec<u8>,
    /// Ticks per second for sample timestamps (usually the sample rate).
    pub timescale: u32,
    /// Every sample of the track, in file order.
    pub samples: Vec<AudioSample>,
    /// `[first, last)` of [`samples`](Self::samples) the demuxer delivers.
    pub keep: (usize, usize),
    /// Decoded samples ffmpeg drops from the start (`skip_samples`).
    pub skip_samples: u64,
}

fn u64at(b: &[u8], i: usize) -> u64 {
    u64::from_be_bytes(b[i..i + 8].try_into().unwrap())
}

/// MPEG-4 descriptor: `(tag, body, rest)`.
fn descriptor(b: &[u8]) -> Option<(u8, &[u8], &[u8])> {
    let tag = *b.first()?;
    let mut len = 0usize;
    let mut i = 1;
    for _ in 0..4 {
        let c = *b.get(i)?;
        i += 1;
        len = (len << 7) | (c & 0x7f) as usize;
        if c & 0x80 == 0 {
            break;
        }
    }
    let body = b.get(i..i + len)?;
    Some((tag, body, &b[i + len..]))
}

/// AudioSpecificConfig inside an `esds` box body.
fn esds_asc(esds: &[u8]) -> Option<Vec<u8>> {
    let (tag, es, _) = descriptor(esds.get(4..)?)?;
    if tag != 3 {
        return None;
    }
    let flags = *es.get(2)?;
    let mut p = 3;
    if flags & 0x80 != 0 {
        p += 2;
    }
    if flags & 0x40 != 0 {
        p += 1 + *es.get(p)? as usize;
    }
    if flags & 0x20 != 0 {
        p += 2;
    }
    let (tag, dc, _) = descriptor(es.get(p..)?)?;
    if tag != 4 || *dc.first()? != 0x40 {
        return None;
    }
    let (tag, dsi, _) = descriptor(dc.get(13..)?)?;
    (tag == 5).then(|| dsi.to_vec())
}

fn itunsmpb_priming(udta: &[u8]) -> Option<i64> {
    let meta = find(udta, b"meta")?;
    let ilst = find(meta.get(4..)?, b"ilst").or_else(|| find(meta, b"ilst"))?;
    let mut out = None;
    boxes(ilst, |ty, item| {
        if ty != b"----" || out.is_some() {
            return;
        }
        let name = find(item, b"name").and_then(|n| n.get(4..));
        if name != Some(b"iTunSMPB".as_slice()) {
            return;
        }
        let Some(data) = find(item, b"data").and_then(|d| d.get(8..)) else { return };
        let s = String::from_utf8_lossy(data);
        let f: Vec<i64> = s.split_whitespace().take(4).filter_map(|x| i64::from_str_radix(x, 16).ok()).collect();
        if f.len() >= 4 {
            out = Some(f[1]);
        }
    });
    out.filter(|&p| p > 0 && p < 16384)
}

/// The first `soun` track, which must carry `mp4a` with an AAC `esds`.
pub fn demux_aac(data: &[u8]) -> Result<AacTrack, String> {
    if find(data, b"moof").is_some() {
        return Err("fragmented MP4 (moof) is not supported".into());
    }
    let moov = find(data, b"moov").ok_or("no moov box")?;
    let mvhd = find(moov, b"mvhd").ok_or("no mvhd")?;
    let movie_ts = if mvhd.first() == Some(&1) { u32at(mvhd, 20) } else { u32at(mvhd, 12) } as i64;

    let mut found: Option<(&[u8], &[u8])> = None; // (trak, mdia)
    let mut last_trak_is_ours = false;
    let mut priming: Option<i64> = None;
    boxes(moov, |ty, body| match ty {
        b"trak" => {
            last_trak_is_ours = false;
            if found.is_some() {
                return;
            }
            let Some(mdia) = find(body, b"mdia") else { return };
            let is_audio = find(mdia, b"hdlr").map(|h| h.len() >= 12 && &h[8..12] == b"soun").unwrap_or(false);
            if is_audio {
                found = Some((body, mdia));
                last_trak_is_ours = true;
                if let Some(p) = find(body, b"udta").and_then(itunsmpb_priming) {
                    priming = Some(p);
                }
            }
        }
        b"udta" if last_trak_is_ours => {
            if let Some(p) = itunsmpb_priming(body) {
                priming = Some(p);
            }
        }
        _ => {}
    });
    let (trak, mdia) = found.ok_or("no audio track")?;
    let mdhd = find(mdia, b"mdhd").ok_or("no mdhd")?;
    let timescale = if mdhd.first() == Some(&1) { u32at(mdhd, 20) } else { u32at(mdhd, 12) };
    if timescale == 0 {
        return Err("audio track has no timescale".into());
    }
    let stbl = find(mdia, b"minf").and_then(|m| find(m, b"stbl")).ok_or("no stbl")?;

    let stsd = find(stbl, b"stsd").ok_or("no stsd")?;
    if stsd.len() < 16 || u32at(stsd, 4) != 1 {
        return Err("expected one sample description".into());
    }
    let entry = &stsd[8..];
    if entry.len() < 36 || &entry[4..8] != b"mp4a" {
        return Err("audio sample entry is not mp4a".into());
    }
    // QuickTime sound description v1 / v2 extend the 28-byte body.
    let ext = match u16::from_be_bytes([entry[16], entry[17]]) {
        1 => 16,
        2 => 36,
        _ => 0,
    };
    let children = entry.get(36 + ext..).ok_or("short mp4a entry")?;
    let esds = find(children, b"esds")
        .or_else(|| find(children, b"wave").and_then(|w| find(w, b"esds")))
        .ok_or("no esds")?;
    let asc = esds_asc(esds).ok_or("esds is not MPEG-4 AAC")?;

    let stts = find(stbl, b"stts").ok_or("no stts")?;
    let mut durations = Vec::new();
    for i in 0..u32at(stts, 4) as usize {
        let o = 8 + i * 8;
        if o + 8 > stts.len() {
            break;
        }
        durations.extend(std::iter::repeat_n(u32at(stts, o + 4), u32at(stts, o) as usize));
    }
    let stsz = find(stbl, b"stsz").ok_or("no stsz")?;
    let uniform = u32at(stsz, 4);
    let count = u32at(stsz, 8) as usize;
    let sizes: Vec<u32> = (0..count)
        .map(|i| if uniform != 0 { uniform } else { u32at(stsz, 12 + i * 4) })
        .collect();
    let stsc = find(stbl, b"stsc").ok_or("no stsc")?;
    let runs: Vec<(u32, u32)> = (0..u32at(stsc, 4) as usize)
        .map(|i| (u32at(stsc, 8 + i * 12), u32at(stsc, 12 + i * 12)))
        .collect();
    let chunks: Vec<u64> = if let Some(co64) = find(stbl, b"co64") {
        (0..u32at(co64, 4) as usize).map(|i| u64at(co64, 8 + i * 8)).collect()
    } else {
        let stco = find(stbl, b"stco").ok_or("no stco")?;
        (0..u32at(stco, 4) as usize).map(|i| u32at(stco, 8 + i * 4) as u64).collect()
    };
    let mut samples = Vec::with_capacity(count);
    let mut dts = 0i64;
    for (ci, &off) in chunks.iter().enumerate() {
        let per = runs.iter().rev().find(|r| r.0 <= ci as u32 + 1).map(|r| r.1).unwrap_or(0);
        let mut at = off;
        for _ in 0..per {
            let i = samples.len();
            if i >= count {
                break;
            }
            let duration = durations.get(i).copied().unwrap_or(0);
            samples.push(AudioSample { offset: at, size: sizes[i], dts, duration });
            at += sizes[i] as u64;
            dts += duration as i64;
        }
    }
    if samples.is_empty() {
        return Err("the audio track has no samples".into());
    }

    // Edit list.
    let mut edits = Vec::new();
    if let Some(elst) = find(trak, b"edts").and_then(|e| find(e, b"elst")) {
        let v1 = elst.first() == Some(&1);
        let n = u32at(elst, 4) as usize;
        for i in 0..n {
            let (d, m) = if v1 {
                let o = 8 + i * 20;
                (u64at(elst, o) as i64, u64at(elst, o + 8) as i64)
            } else {
                let o = 8 + i * 12;
                (u32at(elst, o) as i64, u32at(elst, o + 4) as i32 as i64)
            };
            edits.push((m, d));
        }
    }
    let mut keep = (0, samples.len());
    let mut skip: i64 = 0;
    let non_empty: Vec<_> = edits.iter().skip_while(|e| e.0 == -1).collect();
    if non_empty.len() > 1 {
        return Err("multiple edit list entries are not supported".into());
    }
    if let Some(&&(m, d)) = non_empty.first() {
        if movie_ts == 0 {
            return Err("mvhd timescale 0 with an edit list".into());
        }
        // av_rescale: round half away from zero.
        let ts = timescale as i64;
        let d = (d as i128 * ts as i128 + movie_ts as i128 / 2) / movie_ts as i128;
        let d = d as i64;
        let search = (m - ts).max(samples[0].dts);
        let first = samples.iter().rposition(|s| s.dts <= search).unwrap_or(0);
        let mut last = samples.len();
        for (i, s) in samples.iter().enumerate().skip(first) {
            let dur = samples.get(i + 1).map(|n| n.dts - s.dts).unwrap_or(d);
            if s.dts + dur >= d + m {
                last = i + 1;
                break;
            }
        }
        keep = (first, last);
        skip = (m - samples[first].dts).max(0);
    }
    if let Some(p) = priming {
        skip = p;
    }
    Ok(AacTrack { asc, timescale, samples, keep, skip_samples: skip as u64 })
}
