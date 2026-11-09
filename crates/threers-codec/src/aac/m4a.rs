//! An audio-only MP4 (`.m4a`) around AAC access units: what a phone, a
//! browser or a media library expects an AAC file to be.
//!
//! The file has one `mp4a` track and an edit list that skips the priming
//! samples and stops at the last input sample, so a player starts and ends
//! where the audio did. The priming is [`PRIMING`], 2112 samples: the 1024 the
//! encoder's overlap needs plus 1088 of leading silence this writer adds.
//! Apple's decoders drop 2112 from the start of an AAC file whatever its edit
//! list says (it is what Apple's own encoder leaves), so declaring the same
//! number keeps every player in step. Boxes are written in the
//! order `ftyp`, `moov`, `mdat` (the index first, so it plays while it
//! downloads).

use super::encoder::{AacEncoder, EncodeError, EncoderConfig, DELAY, FRAME};

/// Samples at the start of the file's audio that a player skips.
pub const PRIMING: usize = 2_112;

fn put(out: &mut Vec<u8>, kind: &[u8; 4], body: &[u8]) {
    out.extend_from_slice(&(body.len() as u32 + 8).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
}

fn boxed(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() + 8);
    put(&mut out, kind, body);
    out
}

/// A box with a version and flags word.
fn full(kind: &[u8; 4], flags: u32, body: &[u8]) -> Vec<u8> {
    let mut b = flags.to_be_bytes().to_vec();
    b.extend_from_slice(body);
    boxed(kind, &b)
}

fn u32s(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_be_bytes()).collect()
}

/// An MPEG-4 descriptor with a four-byte length, which every reader accepts.
fn descriptor(tag: u8, body: &[u8]) -> Vec<u8> {
    let n = body.len();
    let mut out = vec![
        tag,
        0x80 | ((n >> 21) & 0x7F) as u8,
        0x80 | ((n >> 14) & 0x7F) as u8,
        0x80 | ((n >> 7) & 0x7F) as u8,
        (n & 0x7F) as u8,
    ];
    out.extend_from_slice(body);
    out
}

const UNITY_MATRIX: [u32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000];

/// Writes the file. `input_samples` is how many samples per channel the
/// listener should hear (the edit list ends there); `priming` is how many
/// the units hold before them; `units` are the access units in order.
pub fn write_m4a(
    cfg: &EncoderConfig,
    asc: [u8; 2],
    units: &[Vec<u8>],
    input_samples: u64,
    priming: usize,
) -> Vec<u8> {
    let rate = cfg.sample_rate;
    let media_duration = units.len() as u64 * FRAME as u64;
    let bytes: u64 = units.iter().map(|u| u.len() as u64).sum();
    let seconds = (input_samples as f64 / f64::from(rate)).max(1e-9);
    let average = (bytes as f64 * 8.0 / seconds) as u32;
    let peak = units
        .iter()
        .map(|u| u.len() as u64 * 8 * u64::from(rate) / FRAME as u64)
        .max()
        .unwrap_or(0) as u32;

    let esds = {
        let decoder_specific = descriptor(0x05, &asc);
        let mut config = vec![0x40, 0x15]; // MPEG-4 audio, audio stream
        config.extend_from_slice(&[
            0,
            ((cfg.channels * 6144 / 8) >> 8) as u8,
            (cfg.channels * 6144 / 8) as u8,
        ]); // buffer size in bytes
        config.extend_from_slice(&peak.to_be_bytes());
        config.extend_from_slice(&average.to_be_bytes());
        config.extend_from_slice(&decoder_specific);
        let mut es = vec![0, 0, 0]; // ES_ID, no flags
        es.extend_from_slice(&descriptor(0x04, &config));
        es.extend_from_slice(&descriptor(0x06, &[0x02])); // SL config: MP4
        full(b"esds", 0, &descriptor(0x03, &es))
    };

    let mp4a = {
        let mut b = vec![0u8; 6];
        b.extend_from_slice(&1u16.to_be_bytes()); // data reference index
        b.extend_from_slice(&[0u8; 8]); // version, revision, vendor
        b.extend_from_slice(&(cfg.channels as u16).to_be_bytes());
        b.extend_from_slice(&16u16.to_be_bytes()); // sample size
        b.extend_from_slice(&[0u8; 4]); // pre-defined, reserved
                                        // 16.16 fixed point holds rates up to 65535 Hz; the config has the real one.
        b.extend_from_slice(&(if rate <= 65_535 { rate << 16 } else { 0 }).to_be_bytes());
        b.extend_from_slice(&esds);
        boxed(b"mp4a", &b)
    };

    let mut sizes = u32s(&[0, units.len() as u32]);
    for u in units {
        sizes.extend_from_slice(&(u.len() as u32).to_be_bytes());
    }
    let stbl = |chunk_offset: u32| {
        let mut b = Vec::new();
        b.extend(full(b"stsd", 0, &[&1u32.to_be_bytes()[..], &mp4a].concat()));
        b.extend(full(
            b"stts",
            0,
            &u32s(&[1, units.len() as u32, FRAME as u32]),
        ));
        b.extend(full(b"stsc", 0, &u32s(&[1, 1, units.len() as u32, 1])));
        b.extend(full(b"stsz", 0, &sizes));
        b.extend(full(b"stco", 0, &u32s(&[1, chunk_offset])));
        boxed(b"stbl", &b)
    };

    let moov = |chunk_offset: u32| {
        let mut mvhd = u32s(&[0, 0, rate, input_samples as u32, 0x0001_0000]);
        mvhd.extend_from_slice(&0x0100u16.to_be_bytes());
        mvhd.extend_from_slice(&[0u8; 10]);
        mvhd.extend_from_slice(&u32s(&UNITY_MATRIX));
        mvhd.extend_from_slice(&[0u8; 24]);
        mvhd.extend_from_slice(&2u32.to_be_bytes());

        let mut tkhd = u32s(&[0, 0, 1, 0, input_samples as u32, 0, 0]);
        tkhd.extend_from_slice(&[0u8; 4]); // layer, alternate group
        tkhd.extend_from_slice(&0x0100u16.to_be_bytes());
        tkhd.extend_from_slice(&[0u8; 2]);
        tkhd.extend_from_slice(&u32s(&UNITY_MATRIX));
        tkhd.extend_from_slice(&[0u8; 8]);

        // One edit: skip the priming, play the input's length.
        let elst = full(
            b"elst",
            0,
            &u32s(&[1, input_samples as u32, priming as u32, 0x0001_0000]),
        );
        let edts = boxed(b"edts", &elst);

        let mut mdhd = u32s(&[0, 0, rate, media_duration as u32]);
        mdhd.extend_from_slice(&0x55C4u16.to_be_bytes()); // "und"
        mdhd.extend_from_slice(&[0u8; 2]);

        let mut hdlr = u32s(&[0]);
        hdlr.extend_from_slice(b"soun");
        hdlr.extend_from_slice(&[0u8; 12]);
        hdlr.extend_from_slice(b"SoundHandler\0");

        let dinf = boxed(
            b"dinf",
            &full(
                b"dref",
                0,
                &[&1u32.to_be_bytes()[..], &full(b"url ", 1, &[])].concat(),
            ),
        );
        let minf = boxed(
            b"minf",
            &[full(b"smhd", 0, &[0u8; 4]), dinf, stbl(chunk_offset)].concat(),
        );
        let mdia = boxed(
            b"mdia",
            &[full(b"mdhd", 0, &mdhd), full(b"hdlr", 0, &hdlr), minf].concat(),
        );
        let trak = boxed(b"trak", &[full(b"tkhd", 3, &tkhd), edts, mdia].concat());
        boxed(b"moov", &[full(b"mvhd", 0, &mvhd), trak].concat())
    };

    let mut ftyp = b"M4A ".to_vec();
    ftyp.extend_from_slice(&0u32.to_be_bytes());
    ftyp.extend_from_slice(b"M4A mp42isom");
    let ftyp = boxed(b"ftyp", &ftyp);
    // The index size does not depend on the offset it holds.
    let moov_size = moov(0).len();
    let data_start = (ftyp.len() + moov_size + 8) as u32;
    let mut out = ftyp;
    out.extend(moov(data_start));
    let mut mdat = Vec::with_capacity(bytes as usize);
    for u in units {
        mdat.extend_from_slice(u);
    }
    put(&mut out, b"mdat", &mdat);
    out
}

/// Encodes whole channels to an `.m4a` file.
pub fn encode_m4a(planar: &[Vec<f32>], cfg: &EncoderConfig) -> Result<Vec<u8>, EncodeError> {
    let mut enc = AacEncoder::new(*cfg)?;
    // The silence that brings the encoder's own delay up to the priming.
    let lead = vec![0.0f32; PRIMING - DELAY];
    let leads: Vec<&[f32]> = planar.iter().map(|_| lead.as_slice()).collect();
    let mut units = enc.push(&leads)?;
    let views: Vec<&[f32]> = planar.iter().map(Vec::as_slice).collect();
    units.extend(enc.push(&views)?);
    units.extend(enc.finish());
    Ok(write_m4a(
        cfg,
        enc.audio_specific_config(),
        &units,
        planar.first().map_or(0, Vec::len) as u64,
        PRIMING,
    ))
}
