//! Reading an MP4 back apart, checked against the muxer that wrote it.
//!
//! The crate could write an MP4 and not read one, so a pipeline that wanted a
//! clip *in* had to shell out to ffmpeg for it. These pin the demuxer against
//! `codec::mp4`, which is the one reader/writer pair where both halves are
//! here and a round trip means something.

#![cfg(feature = "native-codec")]

use threers::codec::mp4_demux::{demux_video, TrackCodec};

/// Four samples of made-up "coded" bytes, muxed and read back.
fn round_trip(sizes: &[usize]) -> (Vec<u8>, Vec<Vec<u8>>) {
    let samples: Vec<Vec<u8>> = sizes
        .iter()
        .enumerate()
        .map(|(i, n)| (0..*n).map(|k| (i * 7 + k) as u8).collect())
        .collect();
    // A minimal but real avcC: one SPS and one PPS, 4-byte NAL lengths.
    let sps = [0x67u8, 0x64, 0x00, 0x1f, 0xac, 0xd9];
    let pps = [0x68u8, 0xeb, 0xe3, 0xcb];
    let mut avcc = vec![1, sps[1], sps[2], sps[3], 0xff, 0xe1];
    avcc.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    avcc.extend_from_slice(&sps);
    avcc.push(1);
    avcc.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    avcc.extend_from_slice(&pps);

    let mp4 = threers::codec::mp4::mux_h264(&threers::codec::mp4::H264Mp4Params {
        width: 640,
        height: 360,
        timescale: 30_000,
        frame_duration: 1_000,
        avcc_payload: &avcc,
        samples: &samples,
    });
    (mp4, samples)
}

#[test]
fn a_muxed_mp4_reads_back_sample_for_sample() {
    let sizes = [1200usize, 340, 512, 96];
    let (mp4, samples) = round_trip(&sizes);
    let t = demux_video(&mp4).expect("demux");

    assert_eq!(t.codec, TrackCodec::H264);
    assert_eq!((t.width, t.height), (640, 360));
    assert_eq!(t.samples.len(), samples.len(), "sample count");
    assert!(!t.config.is_empty(), "no avcC came back");

    for (i, (s, want)) in t.samples.iter().zip(&samples).enumerate() {
        assert_eq!(s.size as usize, want.len(), "sample {i} size");
        let got = &mp4[s.offset as usize..][..s.size as usize];
        assert_eq!(got, want.as_slice(), "sample {i} bytes");
    }
    // No `stss` means every sample is a sync sample, which is what this
    // muxer writes and what the reader has to assume.
    assert!(t.samples.iter().all(|s| s.sync));

    let fps = t.fps().expect("fps");
    assert!((fps - 30.0).abs() < 0.01, "fps came back {fps}");
}

/// The last sample has to end exactly at the end of the media — an off-by-one
/// in the chunk walk shows up here and nowhere else.
#[test]
fn the_sample_table_covers_the_file_exactly() {
    let (mp4, samples) = round_trip(&[900usize, 128, 64]);
    let t = demux_video(&mp4).unwrap();
    let total: u64 = t.samples.iter().map(|s| s.size as u64).sum();
    assert_eq!(total, samples.iter().map(|s| s.len() as u64).sum::<u64>());
    let last = t.samples.last().unwrap();
    assert!(
        last.offset + last.size as u64 <= mp4.len() as u64,
        "the last sample runs past the file"
    );
}

#[test]
fn it_refuses_what_it_cannot_read() {
    assert!(demux_video(b"not an mp4 at all").is_err());
    // A file with a moov but no video track.
    let mut fake = Vec::new();
    fake.extend_from_slice(&8u32.to_be_bytes());
    fake.extend_from_slice(b"moov");
    assert!(demux_video(&fake).is_err());
}

/// Decode order is not display order, and the demuxer has to say so.
///
/// A B-frame is coded after the picture it is shown *before*, which MP4
/// expresses with a `ctts` composition offset per sample. Reading the samples
/// in file order and calling that the clip plays it shuffled — in a short
/// repeating cycle, one per mini-GOP. Caught on a real render: the screen's
/// frame-to-frame difference ran 4.8, 2.3, 1.6, 0.8, 1.6, 0.9 and round again
/// while the source's was a flat 1.23.
#[test]
fn composition_offsets_reorder_the_frames() {
    // Four samples, coded I P B B, shown I B B P — the smallest arrangement
    // that has a frame displayed before one coded earlier.
    let mut mp4 = Vec::new();
    // Build it by hand: the crate's muxer writes no ctts, so this is a
    // hand-rolled stbl with one. Header first.
    fn bx(ty: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut v = ((body.len() + 8) as u32).to_be_bytes().to_vec();
        v.extend_from_slice(ty);
        v.extend_from_slice(body);
        v
    }
    let durations = [512u32; 4];
    let offsets = [0i32, 1536, 512, 1024];
    let sizes = [10u32, 20, 30, 40];

    let mut stts = vec![0, 0, 0, 0, 0, 0, 0, 1];
    stts.extend_from_slice(&4u32.to_be_bytes());
    stts.extend_from_slice(&durations[0].to_be_bytes());
    let mut ctts = vec![0u8, 0, 0, 0];
    ctts.extend_from_slice(&4u32.to_be_bytes());
    for o in offsets {
        ctts.extend_from_slice(&1u32.to_be_bytes());
        ctts.extend_from_slice(&(o as u32).to_be_bytes());
    }
    let mut stsz = vec![0u8, 0, 0, 0, 0, 0, 0, 0];
    stsz.extend_from_slice(&4u32.to_be_bytes());
    for s in sizes {
        stsz.extend_from_slice(&s.to_be_bytes());
    }
    let mut stsc = vec![0u8, 0, 0, 0];
    stsc.extend_from_slice(&1u32.to_be_bytes());
    stsc.extend_from_slice(&1u32.to_be_bytes());
    stsc.extend_from_slice(&4u32.to_be_bytes());
    stsc.extend_from_slice(&1u32.to_be_bytes());
    let mut stco = vec![0u8, 0, 0, 0];
    stco.extend_from_slice(&1u32.to_be_bytes());
    stco.extend_from_slice(&4096u32.to_be_bytes());

    // A minimal avc1 sample entry with an avcC.
    let avcc = bx(b"avcC", &[1, 0x64, 0, 0x1f, 0xff, 0xe1, 0, 1, 0x67, 1, 0, 1, 0x68]);
    let mut entry = vec![0u8; 78];
    entry[24..26].copy_from_slice(&640u16.to_be_bytes());
    entry[26..28].copy_from_slice(&360u16.to_be_bytes());
    let mut avc1 = Vec::new();
    avc1.extend_from_slice(&entry);
    avc1.extend_from_slice(&avcc);
    let avc1 = bx(b"avc1", &avc1);
    let mut stsd = vec![0u8, 0, 0, 0];
    stsd.extend_from_slice(&1u32.to_be_bytes());
    stsd.extend_from_slice(&avc1);

    let mut stbl = Vec::new();
    for (ty, body) in [
        (b"stsd", &stsd),
        (b"stts", &stts),
        (b"ctts", &ctts),
        (b"stsz", &stsz),
        (b"stsc", &stsc),
        (b"stco", &stco),
    ] {
        stbl.extend_from_slice(&bx(ty, body));
    }
    let mut mdhd = vec![0u8; 12];
    mdhd.extend_from_slice(&15360u32.to_be_bytes());
    mdhd.extend_from_slice(&[0u8; 8]);
    let mut hdlr = vec![0u8; 8];
    hdlr.extend_from_slice(b"vide");
    hdlr.extend_from_slice(&[0u8; 12]);
    let minf = bx(b"minf", &bx(b"stbl", &stbl));
    let mut mdia = Vec::new();
    mdia.extend_from_slice(&bx(b"mdhd", &mdhd));
    mdia.extend_from_slice(&bx(b"hdlr", &hdlr));
    mdia.extend_from_slice(&minf);
    let trak = bx(b"trak", &bx(b"mdia", &mdia));
    mp4.extend_from_slice(&bx(b"moov", &trak));
    mp4.resize(4096 + 100, 0);

    let t = demux_video(&mp4).expect("demux");
    assert_eq!(t.samples.len(), 4);

    // Decode timestamps march; presentation ones do not.
    let dts: Vec<i64> = t.samples.iter().map(|s| s.dts).collect();
    let pts: Vec<i64> = t.samples.iter().map(|s| s.pts).collect();
    assert_eq!(dts, vec![0, 512, 1024, 1536], "decode times");
    assert_eq!(pts, vec![0, 2048, 1536, 2560], "presentation times");

    // Which means the display order is not 0,1,2,3.
    let order = t.presentation_order();
    assert_eq!(order, vec![0, 2, 1, 3], "presentation order");
    assert_ne!(order, vec![0, 1, 2, 3], "the reorder is the whole point");

    // And a decoder has to read one sample ahead of what it can show.
    assert_eq!(t.max_reorder(), 1);
}
