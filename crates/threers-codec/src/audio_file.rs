//! One entry point for audio files, decoded the way libsndfile
//! (`soundfile.read`) returns them: planar float, signed integers scaled by
//! `1 / 2^(bits-1)`. Formats: WAV/RF64, AIFF/AIFF-C, FLAC (native and in
//! Ogg), Vorbis in Ogg and MP3, plus AAC in MP4/M4A and Opus in
//! WebM/Matroska (which libsndfile cannot read; decoded as ffmpeg 7.1 does).
//! Recognised by content, not extension.
//!
//! ```no_run
//! let pcm = threers_codec::audio_file::read(&std::fs::read("take.m4a").unwrap()).unwrap();
//! println!("{:.2}s of {}-channel audio", pcm.duration(), pcm.channels.len());
//! ```

use super::pcm::Pcm;

/// What kind of file `data` holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// RIFF WAVE or RF64.
    Wav,
    /// AIFF or AIFF-C.
    Aiff,
    /// Native FLAC (`fLaC`).
    Flac,
    /// An Ogg stream: FLAC or Vorbis.
    Ogg,
    /// MPEG audio Layer III, with or without an ID3v2 tag.
    Mp3,
    /// ISO base media (MP4/M4A) carrying AAC.
    Mp4,
    /// Matroska or WebM carrying Opus.
    Matroska,
    /// None of the above.
    Unknown,
}

/// What kind of file `data` holds, judged from its first bytes.
pub fn sniff(data: &[u8]) -> Kind {
    let h = |s: &[u8]| data.starts_with(s);
    if h(b"RIFF") || h(b"RF64") {
        Kind::Wav
    } else if h(b"FORM") {
        Kind::Aiff
    } else if h(b"fLaC") {
        Kind::Flac
    } else if h(b"OggS") {
        Kind::Ogg
    } else if h(b"ID3") {
        // ID3v2 in front of FLAC or MPEG audio.
        let sz = data.get(6..10).map_or(0, |b| b.iter().fold(0usize, |a, &x| (a << 7) | (x & 0x7f) as usize));
        if data.get(10 + sz..14 + sz) == Some(b"fLaC".as_slice()) {
            Kind::Flac
        } else {
            Kind::Mp3
        }
    } else if h(&[0x1a, 0x45, 0xdf, 0xa3]) {
        Kind::Matroska
    } else if matches!(data.get(4..8), Some(b"ftyp" | b"moov" | b"mdat" | b"wide" | b"free")) {
        Kind::Mp4
    } else if (data.len() > 2 && data[0] == 0xff && data[1] & 0xe0 == 0xe0) || super::mp3::probe(data) {
        Kind::Mp3
    } else {
        Kind::Unknown
    }
}

/// Decode an audio file to planar float.
pub fn read(data: &[u8]) -> Result<Pcm, String> {
    match sniff(data) {
        Kind::Wav => super::pcm::read_wav(data),
        Kind::Aiff => super::pcm::read_aiff(data),
        Kind::Flac => {
            let f = super::flac::decode(data)?;
            Ok(Pcm { sample_rate: f.info.sample_rate, channels: f.to_f32() })
        }
        Kind::Mp4 => {
            let a = super::aac::decode_mp4(data).map_err(|e| e.to_string())?;
            Ok(Pcm { sample_rate: a.sample_rate, channels: a.planes })
        }
        Kind::Matroska => {
            let d = super::opus::decode_matroska(data)?.ok_or("no audio track")?;
            Ok(Pcm { sample_rate: d.sample_rate, channels: d.planes })
        }
        Kind::Ogg => {
            if super::ogg::packets(data)?.0.first().is_some_and(|p| p.data.starts_with(b"\x7fFLAC")) {
                let f = super::ogg::decode_flac(data)?;
                Ok(Pcm { sample_rate: f.info.sample_rate, channels: f.to_f32() })
            } else {
                let v = super::vorbis::decode_ogg(data)?;
                Ok(Pcm { sample_rate: v.sample_rate, channels: v.channels })
            }
        }
        Kind::Mp3 => {
            let m = super::mp3::decode(data)?;
            let n = m.channels;
            let channels = (0..n).map(|c| m.samples.iter().skip(c).step_by(n).copied().collect()).collect();
            Ok(Pcm { sample_rate: m.sample_rate, channels })
        }
        Kind::Unknown => Err("unrecognised audio format".into()),
    }
}
