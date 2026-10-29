//! Incremental MP4 writing for **live** capture, as opposed to the batch muxers
//! in [`crate::mp4`].
//!
//! [`mux_h264`](crate::mp4::mux_h264) is built for an export: it takes
//! every sample up front, gives them all one fixed duration, and marks them all
//! as sync samples, which is exactly right for a render that produced a constant
//! frame rate of all-intra frames. A camera or a screen feed breaks all three
//! assumptions:
//!
//! * **Variable frame rate.** A screen feed emits frames when the screen
//!   changes. Forcing a constant duration makes playback drift from wall clock.
//! * **Mostly inter frames.** Marking every frame as a sync sample tells the
//!   player it can seek anywhere, and seeking then lands on a P-frame and shows
//!   garbage until the next real keyframe.
//! * **Unbounded length.** An hour of 1080p is gigabytes; it cannot sit in a
//!   `Vec` waiting for the mux.
//!
//! So this module keeps per-sample durations, records which samples really are
//! keyframes, and streams sample data to the sink as it arrives — only the
//! sample *tables* stay in memory, about 16 bytes per frame (an hour at 60 fps
//! is a few megabytes). It also carries an optional uncompressed audio track, so
//! a capture with sound lands in one file instead of a video file plus a
//! sidecar WAV.
//!
//! The sink must be seekable: `mdat` is written before its length is known and
//! patched at the end.
//!
//! ```no_run
//! use std::fs::File;
//! use threers_codec::mp4_stream::{AudioConfig, Mp4StreamWriter, VideoConfig};
//!
//! # let avcc_bytes: Vec<u8> = Vec::new();
//! # let avcc_frame: Vec<u8> = Vec::new();
//! # let pcm: Vec<u8> = Vec::new();
//! let video = VideoConfig {
//!     width: 1170,
//!     height: 2532,
//!     timescale: 1_000_000, // microseconds; fine-grained enough for real PTS
//!     avcc: avcc_bytes,     // from the stream's SPS/PPS
//! };
//! let mut w = Mp4StreamWriter::new(
//!     File::create("capture.mp4")?,
//!     video,
//!     Some(AudioConfig::pcm_s16le(48_000, 2)),
//! )?;
//! w.write_video_sample(&avcc_frame, 16_666, true)?;
//! w.write_audio_sample(&pcm)?;
//! let stats = w.finish()?;
//! # Ok::<(), std::io::Error>(())
//! ```

use std::io::{self, Seek, SeekFrom, Write};

use super::mp4::{bx, concat, fullbox, UNITY_MATRIX};

/// Movie-level timescale. Track timescales are independent of this.
const MOVIE_TIMESCALE: u32 = 1000;

/// The video track of an [`Mp4StreamWriter`]: H.264 configured by an `avcC`.
#[derive(Clone)]
pub struct VideoConfig {
    /// Picture width in pixels.
    pub width: u32,
    /// Picture height in pixels.
    pub height: u32,
    /// Ticks per second for this track's durations. Use something fine-grained
    /// (1_000_000 for microseconds) so real presentation deltas survive.
    pub timescale: u32,
    /// `avcC` box payload without its size/type header — see
    /// [`crate::h264::avcc::build_avcc`].
    pub avcc: Vec<u8>,
}

/// How the audio samples are coded.
///
/// The two cases are not variations on a theme: PCM frames are all the same
/// size and arrive as a byte stream to be cut up, while AAC frames are already
/// packets of their own length and carry a configuration the decoder cannot
/// work without. They differ in the sample entry, the sample-size table, and
/// even the file's brand.
#[derive(Clone, PartialEq, Eq)]
pub enum AudioCoding {
    /// Signed 16-bit little-endian interleaved PCM, stored as `sowt`. Makes the
    /// file a QuickTime movie — see [`Mp4StreamWriter::new`].
    PcmS16le,
    /// AAC access units, stored as `mp4a` with an `esds`.
    ///
    /// `asc` is the AudioSpecificConfig — two bytes for plain AAC-LC, five with
    /// explicit SBR signalling. It is not something to synthesise: every
    /// encoder hands it over, WebCodecs as `decoderConfig.description` on the
    /// first output and the ISO reference as the `DecoderSpecificInfo`. Without
    /// it the track says it is AAC and declines to say which kind.
    Aac {
        /// The AudioSpecificConfig.
        asc: Vec<u8>,
    },
}

/// The optional audio track of an [`Mp4StreamWriter`].
#[derive(Clone)]
pub struct AudioConfig {
    /// Samples per second.
    pub sample_rate: u32,
    /// Channel count.
    pub channels: u16,
    /// Bits per sample (16 for PCM; what the encoder reports for AAC).
    pub bits_per_sample: u16,
    /// How the samples are coded.
    pub coding: AudioCoding,
    /// Samples the encoder emitted before the first real one, to be skipped on
    /// playback via an edit list.
    ///
    /// A transform codec cannot start cold: AAC's first packets carry the
    /// window it needed to get going, and the content proper begins after them
    /// — 2048 samples for the usual two-frame priming, which is 43 ms and
    /// plainly visible against a picture. The samples stay in the file, because
    /// the decoder needs them; an `elst` is how the format says "begin here".
    pub encoder_delay: u32,
}

impl AudioConfig {
    /// Signed 16-bit little-endian interleaved PCM, stored as `sowt`.
    pub fn pcm_s16le(sample_rate: u32, channels: u16) -> Self {
        AudioConfig {
            sample_rate,
            channels,
            bits_per_sample: 16,
            coding: AudioCoding::PcmS16le,
            encoder_delay: 0, // PCM starts where it starts
        }
    }

    /// AAC, with the AudioSpecificConfig its decoder needs.
    ///
    /// `encoder_delay` is in samples; see the field. Zero if the encoder has
    /// already accounted for it, or you would rather not say.
    pub fn aac(sample_rate: u32, channels: u16, asc: Vec<u8>, encoder_delay: u32) -> Self {
        AudioConfig {
            sample_rate,
            channels,
            bits_per_sample: 16,
            coding: AudioCoding::Aac { asc },
            encoder_delay,
        }
    }

    /// Bytes in one audio frame across all channels. PCM only — an AAC packet
    /// is its own size.
    pub fn bytes_per_frame(&self) -> usize {
        usize::from(self.channels) * usize::from(self.bits_per_sample / 8)
    }

    fn is_pcm(&self) -> bool {
        self.coding == AudioCoding::PcmS16le
    }
}

/// One MPEG-4 descriptor: a tag, a length in base-128 continuation bytes, and
/// the payload. `esds` is built out of nothing else.
fn descriptor(tag: u8, payload: &[u8]) -> Vec<u8> {
    let mut v = vec![tag];
    let mut len = payload.len();
    // Base-128, most significant group first, every byte but the last carrying
    // the continuation bit.
    let mut groups = Vec::new();
    loop {
        groups.push((len & 0x7f) as u8);
        len >>= 7;
        if len == 0 {
            break;
        }
    }
    for (i, g) in groups.iter().enumerate().rev() {
        v.push(if i == 0 { *g } else { *g | 0x80 });
    }
    v.extend_from_slice(payload);
    v
}

/// What an [`Mp4StreamWriter`] has written so far.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Mp4StreamStats {
    /// Video samples written.
    pub video_samples: u32,
    /// Audio frames written.
    pub audio_frames: u32,
    /// Video samples marked as sync samples.
    pub keyframes: u32,
    /// Bytes of sample data written (the `mdat` payload).
    pub media_bytes: u64,
    /// Total video duration, in the video timescale.
    pub video_duration_ticks: u64,
    /// Total audio duration, in samples.
    pub audio_duration_ticks: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Video,
    Audio,
}

/// Per-track sample tables. Everything here is run-length encoded or index-only,
/// which is what keeps a long capture's memory flat.
#[derive(Default)]
struct Track {
    /// Per-sample byte sizes. Left empty for constant-size tracks (PCM).
    sizes: Vec<u32>,
    /// `stts` as (consecutive sample count, duration) runs.
    durations: Vec<(u32, u32)>,
    /// 1-based sample numbers that are sync samples. Only populated once a
    /// non-sync sample has been seen; see [`Track::sync_sample_table`].
    sync: Vec<u32>,
    all_sync: bool,
    /// `(file offset, samples in chunk)`.
    chunks: Vec<(u64, u32)>,
    samples: u32,
    duration: u64,
}

impl Track {
    fn new() -> Self {
        Track {
            all_sync: true,
            ..Default::default()
        }
    }

    fn add_duration(&mut self, d: u32) {
        match self.durations.last_mut() {
            Some((count, delta)) if *delta == d => *count += 1,
            _ => self.durations.push((1, d)),
        }
        self.duration += u64::from(d);
    }

    /// Record `n` samples starting at `offset`, extending the current chunk when
    /// this track also wrote the previous sample.
    fn add_chunk(&mut self, offset: u64, n: u32, contiguous: bool) {
        if contiguous {
            if let Some(last) = self.chunks.last_mut() {
                last.1 += n;
                return;
            }
        }
        self.chunks.push((offset, n));
    }

    fn mark_sync(&mut self, sample: u32, sync: bool) {
        if sync {
            self.sync.push(sample);
        } else {
            self.all_sync = false;
        }
    }

    /// `None` when every sample is a sync sample: an absent `stss` means exactly
    /// that, and omitting the box is both smaller and unambiguous.
    fn sync_sample_table(&self) -> Option<Vec<u8>> {
        if self.all_sync {
            return None;
        }
        Some(fullbox(b"stss", 0, 0, &{
            let mut v = Vec::with_capacity(4 + self.sync.len() * 4);
            v.extend_from_slice(&(self.sync.len() as u32).to_be_bytes());
            for s in &self.sync {
                v.extend_from_slice(&s.to_be_bytes());
            }
            v
        }))
    }

    fn time_to_sample(&self) -> Vec<u8> {
        fullbox(b"stts", 0, 0, &{
            let mut v = Vec::with_capacity(4 + self.durations.len() * 8);
            v.extend_from_slice(&(self.durations.len() as u32).to_be_bytes());
            for (count, delta) in &self.durations {
                v.extend_from_slice(&count.to_be_bytes());
                v.extend_from_slice(&delta.to_be_bytes());
            }
            v
        })
    }

    /// `stsz`, using the constant-size form when `constant_size` is given.
    fn sample_size(&self, constant_size: Option<u32>) -> Vec<u8> {
        fullbox(b"stsz", 0, 0, &{
            let mut v = Vec::new();
            match constant_size {
                Some(size) => {
                    v.extend_from_slice(&size.to_be_bytes());
                    v.extend_from_slice(&self.samples.to_be_bytes());
                }
                None => {
                    v.extend_from_slice(&0u32.to_be_bytes());
                    v.extend_from_slice(&(self.sizes.len() as u32).to_be_bytes());
                    for s in &self.sizes {
                        v.extend_from_slice(&s.to_be_bytes());
                    }
                }
            }
            v
        })
    }

    /// `stsc`, collapsing consecutive chunks that hold the same sample count.
    fn sample_to_chunk(&self) -> Vec<u8> {
        let mut entries: Vec<(u32, u32)> = Vec::new(); // (first_chunk, samples_per_chunk)
        for (i, (_, n)) in self.chunks.iter().enumerate() {
            let first_chunk = (i + 1) as u32;
            match entries.last() {
                Some((_, prev)) if prev == n => {}
                _ => entries.push((first_chunk, *n)),
            }
        }
        fullbox(b"stsc", 0, 0, &{
            let mut v = Vec::with_capacity(4 + entries.len() * 12);
            v.extend_from_slice(&(entries.len() as u32).to_be_bytes());
            for (first, per) in &entries {
                v.extend_from_slice(&first.to_be_bytes());
                v.extend_from_slice(&per.to_be_bytes());
                v.extend_from_slice(&1u32.to_be_bytes()); // sample_description_index
            }
            v
        })
    }

    /// `stco`, or `co64` when any offset needs more than 32 bits.
    fn chunk_offsets(&self) -> Vec<u8> {
        let needs_64 = self
            .chunks
            .iter()
            .any(|(off, _)| *off > u64::from(u32::MAX));
        if needs_64 {
            fullbox(b"co64", 0, 0, &{
                let mut v = Vec::with_capacity(4 + self.chunks.len() * 8);
                v.extend_from_slice(&(self.chunks.len() as u32).to_be_bytes());
                for (off, _) in &self.chunks {
                    v.extend_from_slice(&off.to_be_bytes());
                }
                v
            })
        } else {
            fullbox(b"stco", 0, 0, &{
                let mut v = Vec::with_capacity(4 + self.chunks.len() * 4);
                v.extend_from_slice(&(self.chunks.len() as u32).to_be_bytes());
                for (off, _) in &self.chunks {
                    v.extend_from_slice(&(*off as u32).to_be_bytes());
                }
                v
            })
        }
    }
}

/// Streams an MP4 to a seekable sink as samples arrive.
pub struct Mp4StreamWriter<W: Write + Seek> {
    out: W,
    video_cfg: VideoConfig,
    audio_cfg: Option<AudioConfig>,
    video: Track,
    audio: Track,
    /// File offset of the `mdat` box header.
    mdat_header_offset: u64,
    /// File offset of the first byte of `mdat` payload.
    payload_start: u64,
    payload_len: u64,
    last_kind: Option<Kind>,
    keyframes: u32,
    /// Audio bytes that did not fill a whole frame, carried to the next write.
    pcm_remainder: Vec<u8>,
    finished: bool,
}

impl<W: Write + Seek> Mp4StreamWriter<W> {
    /// Start a file on `out`: the `ftyp` and an open `mdat` are written now, the
    /// `moov` on [`finish`](Self::finish). `PcmS16le` audio makes the file a
    /// QuickTime movie (`qt` brand), because that is the brand `sowt` belongs to.
    pub fn new(mut out: W, video: VideoConfig, audio: Option<AudioConfig>) -> io::Result<Self> {
        assert!(video.timescale > 0, "video timescale must be non-zero");

        // A file with an uncompressed `sowt` audio track is a QuickTime movie
        // rather than plain ISO base media, so it is branded accordingly. AAC
        // is not: `mp4a` is ISO base media's own audio and the file stays an
        // MP4, which is the difference between something a browser will play
        // and something only QuickTime and VLC will. Video only keeps the same
        // brands the batch muxer uses.
        let ftyp = if audio.as_ref().is_some_and(AudioConfig::is_pcm) {
            bx(b"ftyp", &{
                let mut v = Vec::new();
                v.extend_from_slice(b"qt  ");
                v.extend_from_slice(&0x0000_0200u32.to_be_bytes()); // minor_version
                v.extend_from_slice(b"qt  ");
                v
            })
        } else {
            bx(b"ftyp", &{
                let mut v = Vec::new();
                v.extend_from_slice(b"isom");
                v.extend_from_slice(&0u32.to_be_bytes());
                for brand in [b"isom", b"iso2", b"avc1", b"mp41"] {
                    v.extend_from_slice(brand);
                }
                v
            })
        };
        out.write_all(&ftyp)?;

        // Always use the 64-bit `mdat` header. The size is unknown until the end,
        // and reserving the extended form means patching it later never has to
        // move the payload.
        let mdat_header_offset = ftyp.len() as u64;
        out.write_all(&1u32.to_be_bytes())?; // size == 1 → largesize follows
        out.write_all(b"mdat")?;
        out.write_all(&0u64.to_be_bytes())?; // largesize placeholder
        let payload_start = mdat_header_offset + 16;

        Ok(Mp4StreamWriter {
            out,
            video_cfg: video,
            audio_cfg: audio,
            video: Track::new(),
            audio: Track::new(),
            mdat_header_offset,
            payload_start,
            payload_len: 0,
            last_kind: None,
            keyframes: 0,
            pcm_remainder: Vec::new(),
            finished: false,
        })
    }

    /// Append one coded video frame.
    ///
    /// `sample` holds that frame's NAL units in AVCC framing (each prefixed with
    /// a 4-byte big-endian length), which is what an `avc1` track stores and what
    /// hardware encoders and Apple's capture protocol already hand over — no
    /// conversion needed. `duration` is in the video track's timescale, and
    /// `sync` must be true only for genuine keyframes.
    pub fn write_video_sample(
        &mut self,
        sample: &[u8],
        duration: u32,
        sync: bool,
    ) -> io::Result<()> {
        if sample.is_empty() {
            return Ok(());
        }
        let offset = self.payload_start + self.payload_len;
        self.out.write_all(sample)?;
        self.payload_len += sample.len() as u64;

        let contiguous = self.last_kind == Some(Kind::Video);
        self.video.sizes.push(sample.len() as u32);
        self.video.samples += 1;
        self.video.add_duration(duration);
        self.video.add_chunk(offset, 1, contiguous);
        self.video.mark_sync(self.video.samples, sync);
        if sync {
            self.keyframes += 1;
        }
        self.last_kind = Some(Kind::Video);
        Ok(())
    }

    /// Append interleaved PCM. Any trailing partial frame is held back and
    /// prepended to the next call, so callers can pass whatever sized buffers
    /// their source produces.
    pub fn write_audio_sample(&mut self, pcm: &[u8]) -> io::Result<()> {
        let Some(cfg) = self.audio_cfg.clone() else {
            return Ok(()); // no audio track configured; drop silently
        };
        if !cfg.is_pcm() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "this track is AAC — use write_audio_packet",
            ));
        }
        let frame = cfg.bytes_per_frame();
        if frame == 0 || pcm.is_empty() && self.pcm_remainder.is_empty() {
            return Ok(());
        }

        // Splice the carried remainder onto the front, only allocating when
        // there actually is one.
        let joined: &[u8] = if self.pcm_remainder.is_empty() {
            pcm
        } else {
            self.pcm_remainder.extend_from_slice(pcm);
            &self.pcm_remainder
        };

        let frames = joined.len() / frame;
        let usable = frames * frame;
        if frames == 0 {
            if self.pcm_remainder.is_empty() {
                self.pcm_remainder.extend_from_slice(pcm);
            }
            return Ok(());
        }

        let offset = self.payload_start + self.payload_len;
        self.out.write_all(&joined[..usable])?;
        self.payload_len += usable as u64;
        let leftover = joined[usable..].to_vec();

        let contiguous = self.last_kind == Some(Kind::Audio);
        self.audio.samples += frames as u32;
        // Every PCM frame lasts exactly one tick at a sample-rate timescale, so
        // the whole track collapses to a single `stts` run.
        match self.audio.durations.last_mut() {
            Some((count, delta)) if *delta == 1 => *count += frames as u32,
            _ => self.audio.durations.push((frames as u32, 1)),
        }
        self.audio.duration += frames as u64;
        self.audio.add_chunk(offset, frames as u32, contiguous);

        self.pcm_remainder = leftover;
        self.last_kind = Some(Kind::Audio);
        Ok(())
    }

    /// Append one coded audio packet.
    ///
    /// The AAC counterpart of [`write_audio_sample`](Self::write_audio_sample),
    /// and it is a different shape because the data is: a packet is already its
    /// own unit and its own length, so nothing is cut up and nothing is carried
    /// over. `duration` is in the audio track's timescale, which is the sample
    /// rate — 1024 for AAC-LC, 2048 for HE-AAC's doubled frame.
    ///
    /// Every AAC access unit is independently decodable, so the track keeps its
    /// `all_sync` default and no `stss` is written.
    pub fn write_audio_packet(&mut self, packet: &[u8], duration: u32) -> io::Result<()> {
        let Some(cfg) = self.audio_cfg.as_ref() else {
            return Ok(()); // no audio track configured; drop silently
        };
        if cfg.is_pcm() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "this track is PCM — use write_audio_sample",
            ));
        }
        if packet.is_empty() {
            return Ok(());
        }

        let offset = self.payload_start + self.payload_len;
        self.out.write_all(packet)?;
        self.payload_len += packet.len() as u64;

        let contiguous = self.last_kind == Some(Kind::Audio);
        self.audio.sizes.push(packet.len() as u32);
        self.audio.samples += 1;
        self.audio.add_duration(duration);
        self.audio.add_chunk(offset, 1, contiguous);
        self.last_kind = Some(Kind::Audio);
        Ok(())
    }

    /// Patch the `mdat` length, append `moov`, and report what was written.
    pub fn finish(self) -> io::Result<Mp4StreamStats> {
        self.finish_with_sink().map(|(stats, _)| stats)
    }

    /// [`finish`](Self::finish), also handing back the sink — for callers that
    /// need the handle afterwards (to `sync_all`, or to read a test buffer).
    pub fn finish_with_sink(mut self) -> io::Result<(Mp4StreamStats, W)> {
        self.finished = true;

        // Patch the mdat largesize now that the payload is complete.
        let mdat_total = 16 + self.payload_len;
        self.out
            .seek(SeekFrom::Start(self.mdat_header_offset + 8))?;
        self.out.write_all(&mdat_total.to_be_bytes())?;
        self.out.seek(SeekFrom::End(0))?;

        let moov = self.build_moov();
        self.out.write_all(&moov)?;
        self.out.flush()?;

        let stats = Mp4StreamStats {
            video_samples: self.video.samples,
            audio_frames: self.audio.samples,
            keyframes: self.keyframes,
            media_bytes: self.payload_len,
            video_duration_ticks: self.video.duration,
            audio_duration_ticks: self.audio.duration,
        };
        Ok((stats, self.out))
    }

    /// Track duration converted into the movie timescale.
    fn movie_ticks(duration: u64, timescale: u32) -> u32 {
        if timescale == 0 {
            return 0;
        }
        (duration.saturating_mul(u64::from(MOVIE_TIMESCALE)) / u64::from(timescale)) as u32
    }

    fn build_moov(&self) -> Vec<u8> {
        let video_movie_ticks =
            Self::movie_ticks(self.video.duration, self.video_cfg.timescale);
        let audio_movie_ticks = self
            .audio_cfg
            .as_ref()
            .map(|c| Self::movie_ticks(self.audio.duration, c.sample_rate))
            .unwrap_or(0);
        let movie_duration = video_movie_ticks.max(audio_movie_ticks);

        let video_trak = self.build_video_trak(video_movie_ticks);
        let audio_trak = self
            .audio_cfg
            .as_ref()
            .map(|c| self.build_audio_trak(c, audio_movie_ticks));
        let next_track_id = if audio_trak.is_some() { 3u32 } else { 2 };

        let mvhd = fullbox(b"mvhd", 0, 0, &{
            let mut v = Vec::new();
            v.extend_from_slice(&0u32.to_be_bytes()); // creation_time
            v.extend_from_slice(&0u32.to_be_bytes()); // modification_time
            v.extend_from_slice(&MOVIE_TIMESCALE.to_be_bytes());
            v.extend_from_slice(&movie_duration.to_be_bytes());
            v.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // rate 1.0
            v.extend_from_slice(&0x0100u16.to_be_bytes()); // volume 1.0
            v.extend_from_slice(&0u16.to_be_bytes()); // reserved
            v.extend_from_slice(&[0u8; 8]); // reserved
            v.extend_from_slice(&UNITY_MATRIX);
            v.extend_from_slice(&[0u8; 24]); // pre_defined
            v.extend_from_slice(&next_track_id.to_be_bytes());
            v
        });

        let mut children = vec![mvhd, video_trak];
        children.extend(audio_trak);
        bx(b"moov", &concat(&children))
    }

    fn build_video_trak(&self, movie_ticks: u32) -> Vec<u8> {
        let cfg = &self.video_cfg;

        let avc1 = bx(b"avc1", &{
            let mut v = Vec::new();
            v.extend_from_slice(&[0u8; 6]); // reserved
            v.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
            v.extend_from_slice(&[0u8; 16]); // pre_defined + reserved
            v.extend_from_slice(&(cfg.width as u16).to_be_bytes());
            v.extend_from_slice(&(cfg.height as u16).to_be_bytes());
            v.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // horizresolution 72
            v.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // vertresolution 72
            v.extend_from_slice(&0u32.to_be_bytes()); // reserved
            v.extend_from_slice(&1u16.to_be_bytes()); // frame_count
            v.extend_from_slice(&[0u8; 32]); // compressorname
            v.extend_from_slice(&0x0018u16.to_be_bytes()); // depth 24
            v.extend_from_slice(&(-1i16).to_be_bytes()); // pre_defined
            v.extend_from_slice(&bx(b"avcC", &cfg.avcc));
            v
        });
        let stsd = fullbox(b"stsd", 0, 0, &{
            let mut v = Vec::new();
            v.extend_from_slice(&1u32.to_be_bytes()); // entry_count
            v.extend_from_slice(&avc1);
            v
        });

        let mut stbl_children = vec![stsd, self.video.time_to_sample()];
        stbl_children.extend(self.video.sync_sample_table());
        stbl_children.push(self.video.sample_to_chunk());
        stbl_children.push(self.video.sample_size(None));
        stbl_children.push(self.video.chunk_offsets());
        let stbl = bx(b"stbl", &concat(&stbl_children));

        let vmhd = fullbox(b"vmhd", 0, 1, &[0u8; 8]);
        let minf = bx(
            b"minf",
            &concat(&[vmhd, handler(b"vide"), data_information(), stbl]),
        );
        let mdia = bx(
            b"mdia",
            &concat(&[
                media_header(cfg.timescale, self.video.duration),
                handler(b"vide"),
                minf,
            ]),
        );

        let tkhd = fullbox(b"tkhd", 0, 7, &{
            let mut v = Vec::new();
            v.extend_from_slice(&0u32.to_be_bytes());
            v.extend_from_slice(&0u32.to_be_bytes());
            v.extend_from_slice(&1u32.to_be_bytes()); // track_ID
            v.extend_from_slice(&0u32.to_be_bytes()); // reserved
            v.extend_from_slice(&movie_ticks.to_be_bytes());
            v.extend_from_slice(&[0u8; 8]); // reserved
            v.extend_from_slice(&0u16.to_be_bytes()); // layer
            v.extend_from_slice(&0u16.to_be_bytes()); // alternate_group
            v.extend_from_slice(&0u16.to_be_bytes()); // volume
            v.extend_from_slice(&0u16.to_be_bytes()); // reserved
            v.extend_from_slice(&UNITY_MATRIX);
            v.extend_from_slice(&(cfg.width << 16).to_be_bytes());
            v.extend_from_slice(&(cfg.height << 16).to_be_bytes());
            v
        });
        bx(b"trak", &concat(&[tkhd, mdia]))
    }

    fn build_audio_trak(&self, cfg: &AudioConfig, movie_ticks: u32) -> Vec<u8> {
        // Both codings use a version-0 sound sample description — the same
        // sixteen bytes of channels, depth and rate — and differ only in the
        // four-character type and what hangs off the end of it.
        let mut entry = Vec::new();
        entry.extend_from_slice(&[0u8; 6]); // reserved
        entry.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
        entry.extend_from_slice(&0u16.to_be_bytes()); // version
        entry.extend_from_slice(&0u16.to_be_bytes()); // revision_level
        entry.extend_from_slice(&0u32.to_be_bytes()); // vendor
        entry.extend_from_slice(&cfg.channels.to_be_bytes());
        entry.extend_from_slice(&cfg.bits_per_sample.to_be_bytes());
        entry.extend_from_slice(&0u16.to_be_bytes()); // compression_id
        entry.extend_from_slice(&0u16.to_be_bytes()); // packet_size
        // sample_rate as 16.16 fixed point
        entry.extend_from_slice(&(cfg.sample_rate << 16).to_be_bytes());

        let sample_entry = match &cfg.coding {
            // `sowt` is signed 16-bit little-endian PCM, and a version-0
            // description is enough for it — what QuickTime, ffmpeg and VLC all
            // read back.
            AudioCoding::PcmS16le => bx(b"sowt", &entry),
            // `mp4a` says only "MPEG-4 audio"; which kind is in the `esds`
            // underneath it, as a nest of descriptors ending in the encoder's
            // own AudioSpecificConfig. The two constants are the ones that make
            // it AAC: object type 0x40 is MPEG-4 Audio, and 0x15 is stream type
            // 5 (audio) shifted up two with the reserved bit set.
            AudioCoding::Aac { asc } => {
                let dec_specific = descriptor(0x05, asc);
                let dec_config = descriptor(0x04, &{
                    let mut v = Vec::new();
                    v.push(0x40); // objectTypeIndication: MPEG-4 Audio
                    v.push(0x15); // streamType 5 << 2 | upStream 0 << 1 | reserved 1
                    v.extend_from_slice(&[0, 0, 0]); // bufferSizeDB, unstated
                    v.extend_from_slice(&0u32.to_be_bytes()); // maxBitrate, unstated
                    v.extend_from_slice(&0u32.to_be_bytes()); // avgBitrate, unstated
                    v.extend_from_slice(&dec_specific);
                    v
                });
                // SLConfig 0x02 is "MP4 defaults", which is what a file-backed
                // track uses: the timing is in the sample tables, not in-band.
                let sl_config = descriptor(0x06, &[0x02]);
                let es = descriptor(0x03, &{
                    let mut v = Vec::new();
                    v.extend_from_slice(&1u16.to_be_bytes()); // ES_ID
                    v.push(0x00); // no dependency, no URL, no OCR
                    v.extend_from_slice(&dec_config);
                    v.extend_from_slice(&sl_config);
                    v
                });
                entry.extend_from_slice(&fullbox(b"esds", 0, 0, &es));
                bx(b"mp4a", &entry)
            }
        };
        let stsd = fullbox(b"stsd", 0, 0, &{
            let mut v = Vec::new();
            v.extend_from_slice(&1u32.to_be_bytes());
            v.extend_from_slice(&sample_entry);
            v
        });

        // PCM frames are all one size and the table says so once; AAC packets
        // are each their own and have to be listed.
        let constant_size = cfg.is_pcm().then(|| cfg.bytes_per_frame() as u32);
        let stbl = bx(
            b"stbl",
            &concat(&[
                stsd,
                self.audio.time_to_sample(),
                self.audio.sample_to_chunk(),
                self.audio.sample_size(constant_size),
                self.audio.chunk_offsets(),
            ]),
        );

        let smhd = fullbox(b"smhd", 0, 0, &[0u8; 4]); // balance + reserved
        let minf = bx(
            b"minf",
            &concat(&[smhd, handler(b"soun"), data_information(), stbl]),
        );
        let mdia = bx(
            b"mdia",
            &concat(&[
                media_header(cfg.sample_rate, self.audio.duration),
                handler(b"soun"),
                minf,
            ]),
        );

        // Skip the priming: present this track from `encoder_delay` into its
        // media rather than from the top, which is the only way a file can say
        // "these samples are here for the decoder, not for the listener".
        let edts = if cfg.encoder_delay > 0 && self.audio.duration > u64::from(cfg.encoder_delay) {
            let media = self.audio.duration - u64::from(cfg.encoder_delay);
            let elst = fullbox(b"elst", 0, 0, &{
                let mut v = Vec::new();
                v.extend_from_slice(&1u32.to_be_bytes()); // entry_count
                v.extend_from_slice(&Self::movie_ticks(media, cfg.sample_rate).to_be_bytes());
                v.extend_from_slice(&cfg.encoder_delay.to_be_bytes()); // media_time
                v.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // media_rate 1.0
                v
            });
            Some(bx(b"edts", &elst))
        } else {
            None
        };

        let tkhd = fullbox(b"tkhd", 0, 7, &{
            let mut v = Vec::new();
            v.extend_from_slice(&0u32.to_be_bytes());
            v.extend_from_slice(&0u32.to_be_bytes());
            v.extend_from_slice(&2u32.to_be_bytes()); // track_ID
            v.extend_from_slice(&0u32.to_be_bytes());
            v.extend_from_slice(&movie_ticks.to_be_bytes());
            v.extend_from_slice(&[0u8; 8]);
            v.extend_from_slice(&0u16.to_be_bytes()); // layer
            v.extend_from_slice(&0u16.to_be_bytes()); // alternate_group
            v.extend_from_slice(&0x0100u16.to_be_bytes()); // volume 1.0
            v.extend_from_slice(&0u16.to_be_bytes()); // reserved
            v.extend_from_slice(&UNITY_MATRIX);
            v.extend_from_slice(&0u32.to_be_bytes()); // width
            v.extend_from_slice(&0u32.to_be_bytes()); // height
            v
        });
        let mut parts = vec![tkhd];
        parts.extend(edts);
        parts.push(mdia);
        bx(b"trak", &concat(&parts))
    }
}

fn media_header(timescale: u32, duration: u64) -> Vec<u8> {
    fullbox(b"mdhd", 0, 0, &{
        let mut v = Vec::new();
        v.extend_from_slice(&0u32.to_be_bytes()); // creation_time
        v.extend_from_slice(&0u32.to_be_bytes()); // modification_time
        v.extend_from_slice(&timescale.to_be_bytes());
        v.extend_from_slice(&(duration.min(u64::from(u32::MAX)) as u32).to_be_bytes());
        v.extend_from_slice(&0x55c4u16.to_be_bytes()); // language 'und'
        v.extend_from_slice(&0u16.to_be_bytes()); // pre_defined
        v
    })
}

fn handler(kind: &[u8; 4]) -> Vec<u8> {
    fullbox(b"hdlr", 0, 0, &{
        let mut v = Vec::new();
        v.extend_from_slice(&0u32.to_be_bytes()); // pre_defined
        v.extend_from_slice(kind);
        v.extend_from_slice(&[0u8; 12]); // reserved
        v.extend_from_slice(b"threers\0");
        v
    })
}

fn data_information() -> Vec<u8> {
    let dref = fullbox(b"dref", 0, 0, &{
        let mut v = Vec::new();
        v.extend_from_slice(&1u32.to_be_bytes()); // entry_count
        v.extend_from_slice(&fullbox(b"url ", 0, 1, &[])); // self-contained
        v
    });
    bx(b"dinf", &dref)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mp4_demux::demux_video;
    use std::io::Cursor;

    /// Real H.264 parameter sets, captured from an iPad screen stream.
    const SPS: &[u8] = &[
        0x27, 0x64, 0x00, 0x33, 0xac, 0x56, 0x80, 0x47, 0x01, 0x33, 0xe6, 0x9e, 0x6e, 0x04, 0x04,
        0x04, 0x04,
    ];
    const PPS: &[u8] = &[0x28, 0xee, 0x3c, 0xb0];

    fn video_config() -> VideoConfig {
        VideoConfig {
            width: 1126,
            height: 2436,
            timescale: 1_000_000,
            avcc: crate::h264::avcc::build_avcc(SPS, PPS),
        }
    }

    /// One AVCC-framed NAL of `n` payload bytes, so the total sample is `n + 4`.
    fn avcc_frame(n: usize, fill: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(4 + n);
        v.extend_from_slice(&(n as u32).to_be_bytes());
        v.extend(std::iter::repeat_n(fill, n));
        v
    }

    fn contains_tag(data: &[u8], tag: &[u8; 4]) -> bool {
        data.windows(4).any(|w| w == tag)
    }

    type Frame = (Vec<u8>, u32, bool);

    fn write_video_only(frames: &[Frame]) -> Vec<u8> {
        let mut w = Mp4StreamWriter::new(Cursor::new(Vec::new()), video_config(), None).unwrap();
        for (data, dur, sync) in frames {
            w.write_video_sample(data, *dur, *sync).unwrap();
        }
        w.finish_with_sink().unwrap().1.into_inner()
    }

    #[test]
    fn roundtrips_through_the_demuxer() {
        let frames: Vec<Frame> = vec![
            (avcc_frame(120, 0xa1), 16_666, true),
            (avcc_frame(40, 0xa2), 16_666, false),
            (avcc_frame(55, 0xa3), 33_333, false),
            (avcc_frame(200, 0xa4), 16_666, true),
            (avcc_frame(30, 0xa5), 16_666, false),
        ];
        let bytes = write_video_only(&frames);

        let track = demux_video(&bytes).expect("demux");
        assert_eq!((track.width, track.height), (1126, 2436));
        assert_eq!(track.timescale, 1_000_000);
        assert_eq!(track.config, crate::h264::avcc::build_avcc(SPS, PPS));
        assert_eq!(track.samples.len(), frames.len());

        for (i, (data, dur, sync)) in frames.iter().enumerate() {
            let s = &track.samples[i];
            assert_eq!(s.size as usize, data.len(), "sample {i} size");
            assert_eq!(s.duration, *dur, "sample {i} duration");
            assert_eq!(s.sync, *sync, "sample {i} sync flag");
            let start = s.offset as usize;
            assert_eq!(
                &bytes[start..start + s.size as usize],
                data.as_slice(),
                "sample {i} bytes at its recorded offset"
            );
        }
    }

    #[test]
    fn variable_durations_survive_the_round_trip() {
        // The shape a screen feed actually produces: a burst at frame rate, then
        // a long gap while nothing on screen changes. A fixed-duration muxer
        // would silently rewrite this to a constant rate.
        let durations = [16_666u32, 16_666, 16_666, 500_000, 16_666];
        let frames: Vec<Frame> = durations
            .iter()
            .enumerate()
            .map(|(i, d)| (avcc_frame(32 + i, 0xb0 + i as u8), *d, i == 0))
            .collect();
        let bytes = write_video_only(&frames);
        let track = demux_video(&bytes).unwrap();
        let got: Vec<u32> = track.samples.iter().map(|s| s.duration).collect();
        assert_eq!(got, durations);
    }

    #[test]
    fn stss_is_omitted_when_every_sample_is_a_keyframe() {
        let frames: Vec<Frame> = (0..3u8)
            .map(|i| (avcc_frame(16, 0xc0 + i), 16_666, true))
            .collect();
        let bytes = write_video_only(&frames);
        assert!(
            !contains_tag(&bytes, b"stss"),
            "an all-keyframe track should omit stss; its absence already means \
             every sample is a sync sample"
        );
        let track = demux_video(&bytes).unwrap();
        assert!(track.samples.iter().all(|s| s.sync));
    }

    #[test]
    fn stss_lists_only_real_keyframes() {
        // Guards the bug this module exists to fix: claiming every frame is a
        // sync sample makes a player seek onto a P-frame and show garbage.
        let sync_flags = [true, false, false, false, true, false];
        let frames: Vec<Frame> = sync_flags
            .iter()
            .enumerate()
            .map(|(i, s)| (avcc_frame(16 + i, 0xd0 + i as u8), 16_666, *s))
            .collect();
        let bytes = write_video_only(&frames);
        assert!(contains_tag(&bytes, b"stss"), "a mixed track needs an stss");
        let track = demux_video(&bytes).unwrap();
        let got: Vec<bool> = track.samples.iter().map(|s| s.sync).collect();
        assert_eq!(got, sync_flags);
    }

    #[test]
    fn mdat_length_is_patched_after_the_fact() {
        let frames: Vec<Frame> = vec![(avcc_frame(100, 0xe1), 16_666, true)];
        let bytes = write_video_only(&frames);

        let ftyp_len = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
        assert_eq!(&bytes[ftyp_len + 4..ftyp_len + 8], b"mdat");
        assert_eq!(
            u32::from_be_bytes(bytes[ftyp_len..ftyp_len + 4].try_into().unwrap()),
            1,
            "mdat should use the 64-bit extended size form so it can be patched \
             without moving the payload"
        );
        let largesize = u64::from_be_bytes(bytes[ftyp_len + 8..ftyp_len + 16].try_into().unwrap());
        assert_eq!(largesize, 16 + 104, "mdat size must include its own header");
    }

    #[test]
    fn audio_track_is_written_and_interleaved_with_video() {
        let cfg = AudioConfig::pcm_s16le(48_000, 2);
        assert_eq!(cfg.bytes_per_frame(), 4);

        let mut w =
            Mp4StreamWriter::new(Cursor::new(Vec::new()), video_config(), Some(cfg)).unwrap();
        for i in 0..4u8 {
            w.write_video_sample(&avcc_frame(64, 0xf0 + i), 16_666, i == 0)
                .unwrap();
            // 480 frames is 10 ms of 48 kHz stereo.
            w.write_audio_sample(&vec![i; 480 * 4]).unwrap();
        }
        let (stats, cursor) = w.finish_with_sink().unwrap();
        let bytes = cursor.into_inner();

        assert_eq!(stats.video_samples, 4);
        assert_eq!(stats.audio_frames, 4 * 480);
        assert_eq!(stats.keyframes, 1);

        assert!(contains_tag(&bytes, b"sowt"), "missing pcm sample entry");
        assert!(contains_tag(&bytes, b"smhd"), "missing sound media header");
        assert!(contains_tag(&bytes, b"qt  "), "a pcm file brands as qt");

        // Interleaving puts each video sample in its own chunk; the offsets must
        // still land exactly on the AVCC length prefix we wrote.
        let track = demux_video(&bytes).unwrap();
        assert_eq!(track.samples.len(), 4);
        for (i, s) in track.samples.iter().enumerate() {
            assert_eq!(s.size, 68, "sample {i} size");
            let start = s.offset as usize;
            assert_eq!(
                u32::from_be_bytes(bytes[start..start + 4].try_into().unwrap()),
                64,
                "sample {i} offset should point at its length prefix"
            );
        }
    }

    /// AAC-LC, 48 kHz, stereo — the two bytes every encoder emits for it.
    const ASC: &[u8] = &[0x11, 0x90];

    #[test]
    fn aac_writes_an_mp4a_entry_with_the_config_inside_it() {
        let cfg = AudioConfig::aac(48_000, 2, ASC.to_vec(), 0);
        let mut w =
            Mp4StreamWriter::new(Cursor::new(Vec::new()), video_config(), Some(cfg)).unwrap();
        for i in 0..4u8 {
            w.write_video_sample(&avcc_frame(64, 0xf0 + i), 16_666, i == 0)
                .unwrap();
            // AAC-LC codes 1024 samples a packet, whatever the packet weighs.
            w.write_audio_packet(&vec![i; 180 + i as usize], 1024).unwrap();
        }
        let (stats, cursor) = w.finish_with_sink().unwrap();
        let bytes = cursor.into_inner();

        assert_eq!(stats.audio_frames, 4);
        assert_eq!(stats.audio_duration_ticks, 4 * 1024);

        assert!(contains_tag(&bytes, b"mp4a"), "missing the aac sample entry");
        assert!(contains_tag(&bytes, b"esds"), "missing the decoder config");
        assert!(!contains_tag(&bytes, b"sowt"), "this is not a pcm track");
        // The whole point of AAC over PCM here: the file stays an MP4 rather
        // than becoming a QuickTime movie, which is what a browser will play.
        assert!(!contains_tag(&bytes, b"qt  "), "an aac file must stay isom");

        // The AudioSpecificConfig has to survive into the file verbatim, inside
        // a DecoderSpecificInfo of its own length — a decoder that cannot find
        // it will not play the track at all.
        let want = [0x05u8, ASC.len() as u8, ASC[0], ASC[1]];
        assert!(
            bytes.windows(want.len()).any(|w| w == want),
            "the DecoderSpecificInfo should carry the ASC as written"
        );

        // Video must still be readable with an audio track beside it.
        let track = demux_video(&bytes).unwrap();
        assert_eq!(track.samples.len(), 4);
    }

    #[test]
    fn an_edit_list_hides_the_encoder_priming() {
        // 2048 samples of priming is what a two-frame AAC encoder emits before
        // the content starts, and 43 ms of silence against a picture is plainly
        // visible. The samples stay — the decoder needs them — and the edit
        // list says where to begin.
        let cfg = AudioConfig::aac(48_000, 2, ASC.to_vec(), 2048);
        let mut w =
            Mp4StreamWriter::new(Cursor::new(Vec::new()), video_config(), Some(cfg)).unwrap();
        w.write_video_sample(&avcc_frame(16, 1), 16_666, true).unwrap();
        for _ in 0..8 {
            w.write_audio_packet(&[7u8; 128], 1024).unwrap();
        }
        let bytes = w.finish_with_sink().unwrap().1.into_inner();

        assert!(contains_tag(&bytes, b"elst"), "no edit list");
        // One entry, running from media time 2048 at normal rate. 8192 samples
        // of track less 2048 of priming is 6144, which is 128 ms of movie.
        let want: Vec<u8> = {
            let mut v = Vec::new();
            v.extend_from_slice(&1u32.to_be_bytes()); // entry_count
            v.extend_from_slice(&128u32.to_be_bytes()); // segment_duration, movie ticks
            v.extend_from_slice(&2048u32.to_be_bytes()); // media_time
            v.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // rate 1.0
            v
        };
        assert!(
            bytes.windows(want.len()).any(|w| w == want),
            "the edit list should start the track at the first real sample"
        );
    }

    #[test]
    fn no_priming_means_no_edit_list() {
        // An `elst` that skips nothing is a box every player has to parse to
        // learn it was not needed.
        let cfg = AudioConfig::aac(48_000, 2, ASC.to_vec(), 0);
        let mut w =
            Mp4StreamWriter::new(Cursor::new(Vec::new()), video_config(), Some(cfg)).unwrap();
        w.write_video_sample(&avcc_frame(16, 1), 16_666, true).unwrap();
        w.write_audio_packet(&[7u8; 128], 1024).unwrap();
        let bytes = w.finish_with_sink().unwrap().1.into_inner();
        assert!(!contains_tag(&bytes, b"elst"), "nothing to skip, nothing to say");
    }

    #[test]
    fn aac_packets_keep_their_own_sizes() {
        // PCM can say "every sample is four bytes" once. AAC cannot: a packet
        // is as long as the coder made it, so the sizes have to be listed, and
        // a constant-size `stsz` would hand the decoder the wrong bytes.
        let cfg = AudioConfig::aac(44_100, 1, ASC.to_vec(), 0);
        let mut w =
            Mp4StreamWriter::new(Cursor::new(Vec::new()), video_config(), Some(cfg)).unwrap();
        w.write_video_sample(&avcc_frame(16, 1), 16_666, true).unwrap();
        let packets: [&[u8]; 3] = [&[1u8; 97], &[2u8; 211], &[3u8; 64]];
        for p in packets {
            w.write_audio_packet(p, 1024).unwrap();
        }
        let (stats, cursor) = w.finish_with_sink().unwrap();
        let bytes = cursor.into_inner();
        assert_eq!(stats.audio_frames, 3);

        // Find the audio `stsz`: the video one lists one sample, this one three.
        let want: Vec<u8> = {
            let mut v = vec![0, 0, 0, 0]; // sample_size 0 → sizes follow
            v.extend_from_slice(&3u32.to_be_bytes());
            for p in packets {
                v.extend_from_slice(&(p.len() as u32).to_be_bytes());
            }
            v
        };
        assert!(
            bytes.windows(want.len()).any(|w| w == want),
            "each packet's own length must reach the sample size table"
        );
    }

    #[test]
    fn the_two_codings_refuse_each_other() {
        let mut aac =
            Mp4StreamWriter::new(Cursor::new(Vec::new()), video_config(),
                                 Some(AudioConfig::aac(48_000, 2, ASC.to_vec(), 0))).unwrap();
        assert!(aac.write_audio_sample(&[0; 16]).is_err(), "pcm into an aac track");

        let mut pcm =
            Mp4StreamWriter::new(Cursor::new(Vec::new()), video_config(),
                                 Some(AudioConfig::pcm_s16le(48_000, 2))).unwrap();
        assert!(pcm.write_audio_packet(&[0; 16], 1024).is_err(), "a packet into pcm");
    }

    #[test]
    fn descriptor_lengths_use_the_base_128_form() {
        // Under 128 bytes it is one byte, which is every ordinary ASC. Over it,
        // the length continues across bytes with the top bit set — getting this
        // wrong truncates the config and the track goes silent.
        assert_eq!(descriptor(0x05, &[0xaa, 0xbb]), vec![0x05, 0x02, 0xaa, 0xbb]);
        let long = vec![0u8; 200];
        let d = descriptor(0x04, &long);
        assert_eq!(&d[..3], &[0x04, 0x81, 0x48], "200 is 0x81 0x48 in base 128");
        assert_eq!(d.len(), 3 + 200);
    }

    #[test]
    fn partial_pcm_frames_are_carried_rather_than_dropped() {
        let cfg = AudioConfig::pcm_s16le(48_000, 2);
        let mut w =
            Mp4StreamWriter::new(Cursor::new(Vec::new()), video_config(), Some(cfg)).unwrap();
        w.write_video_sample(&avcc_frame(16, 1), 16_666, true)
            .unwrap();
        // Six bytes is one whole 4-byte frame plus half of the next.
        w.write_audio_sample(&[1, 2, 3, 4, 5, 6]).unwrap();
        // These two complete it, so two frames should be stored in total.
        w.write_audio_sample(&[7, 8]).unwrap();
        let stats = w.finish().unwrap();
        assert_eq!(stats.audio_frames, 2, "the split frame must not be lost");
        assert_eq!(stats.audio_duration_ticks, 2);
    }

    #[test]
    fn a_capture_with_no_samples_still_produces_a_parsable_file() {
        let w = Mp4StreamWriter::new(Cursor::new(Vec::new()), video_config(), None).unwrap();
        let bytes = w.finish_with_sink().unwrap().1.into_inner();
        assert!(contains_tag(&bytes, b"moov"));
        assert!(contains_tag(&bytes, b"avcC"));
    }
}
