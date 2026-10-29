# threers-codec

Pure-Rust media codecs. Video and audio encoders, decoders, muxers and
demuxers, with hardware acceleration where the platform has it and no C
dependencies anywhere else.

- **No dependencies** beyond `std` (`rayon` with `parallel`), no build script,
  no system libraries.
- **Builds for `wasm32`**: outside the opt-in hardware modules
  (`videotoolbox`, `nvidia`, `vaapi`) there are no OS, process or filesystem
  calls. Encoders return `Vec<u8>`; the caller
  owns I/O.
- **Held to the reference implementations**: the decoders are ports of HM,
  libavcodec, dav1d, libvorbis and mpg123 tested against them for exact
  output, and the H.264 and HEVC encoders' streams decode in ffmpeg to the
  reconstruction they report.
- **Hardware first**: `hw::Mp4Encoder` and `decode::Decoder` use VideoToolbox
  on macOS, NVENC/NVDEC on NVIDIA GPUs (Linux, Windows) and VAAPI on AMD and
  Intel GPUs (Linux), and fall back to the native codecs everywhere else. The
  NVIDIA and VA libraries are loaded at run time, not linked.

It is the media half of [threers](https://crates.io/crates/threers), which
re-exports it as `threers::codec`.

## Usage

```toml
[dependencies]
threers-codec = "0.0.9"
```

```rust,no_run
use threers_codec::prelude::*;

// Encode RGBA frames to MP4, on the hardware encoder where there is one.
let mut enc = Mp4Encoder::new(640, 360, EncodeOptions::default())?;
println!("encoding with {:?}", enc.backend());
for _ in 0..30 {
    enc.push_rgba(&vec![128; 640 * 360 * 4])?;
}
std::fs::write("out.mp4", enc.finish()?)?;

// Decode an audio file to planar f32: WAV, AIFF, FLAC, Vorbis, MP3, AAC, Opus.
let pcm = audio_file::read(&std::fs::read("song.flac")?)?;
println!("{:.1}s at {} Hz", pcm.duration(), pcm.sample_rate);

// Decode a video track: H.264, HEVC, VP9 or AV1.
let file = std::fs::read("clip.mp4")?;
let track = demux_video(&file)?;
let mut dec = Decoder::new(&track)?;
for s in &track.samples {
    dec.push(&file[s.offset as usize..][..s.size as usize], s.pts, s.dts, s.duration as i64)?;
    if let Some(bgra) = dec.take(s.pts) {
        // width * height * 4 bytes; `take_frame` on `NativeDecoder` returns the YUV planes
        let _ = bgra;
    }
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

## What is in it

| Kind | Formats |
|------|---------|
| Video encoders | H.264 (CAVLC and CABAC, intra and inter), HEVC (intra, with alpha), VP9 (intra and inter, with alpha) |
| Video decoders | H.264, HEVC, VP9, AV1 |
| Audio decoders | AAC (LC, HE-AAC with SBR and PS), MP3, Opus (SILK, CELT, hybrid), Vorbis, FLAC, WAV/RF64, AIFF |
| Animated images | GIF (encode and decode), APNG |
| Containers | MP4 (mux, streaming mux, demux), WebM (mux), Matroska/WebM and IVF (demux), Ogg (demux) |
| Hardware | VideoToolbox HEVC encode, H.264/HEVC decode (macOS); NVENC HEVC encode, NVDEC H.264/HEVC/VP9/AV1 decode (NVIDIA on Linux and Windows); VAAPI HEVC encode, H.264/HEVC/VP9/AV1 decode (AMD, Intel on Linux) |
| Other | Bitrate and quantizer rate control, swresample-compatible resampling, swscale-compatible YUV→RGB, SRT/WebVTT captions muxed as soft subtitles |

Every stage of every codec is a public module (entropy coders, transforms,
loop filters, parameter-set parsers, MDCTs), so the building blocks are
available to tools and research as well as the high-level calls. The
[crate docs](https://docs.rs/threers-codec) map them.

## Features

| Feature | Default | What it adds |
|---------|---------|--------------|
| `decode` | on | The video decoders (H.264, HEVC, VP9, AV1) |
| `gif` | on | GIF89a encode and decode |
| `parallel` | | Frame-parallel encoding on `rayon` |
| `videotoolbox` | | VideoToolbox encode and decode on macOS (inert elsewhere) |
| `nvidia` | | NVENC encode and NVDEC decode on Linux and Windows, driver loaded at run time (inert elsewhere) |
| `vaapi` | | VAAPI HEVC encode and H.264/HEVC/VP9/AV1 decode on Linux, libva loaded at run time (inert elsewhere) |

## Tests

```sh
cargo test -p threers-codec --all-features --release
```

Some suites compare against `ffmpeg` when it is on `PATH`, and skip that
comparison when it is not. The Opus parity fixtures live in the repository
rather than the published crate, so run the suite from a checkout.

The hardware suites skip without their hardware; require it to make a run
count, one test thread at a time:

```sh
THREERS_REQUIRE_NVIDIA=1 RUST_TEST_THREADS=1 cargo test -p threers-codec --release --features nvidia,parallel
THREERS_REQUIRE_VAAPI=1 RUST_TEST_THREADS=1 cargo test -p threers-codec --release --features vaapi,parallel
```

Every hardware decoder is checked against the native one, picture for
picture; the encoders' files are decoded back and compared with what went
in. They have run on an NVIDIA RTX 3080 Ti (Linux), an AMD Radeon 780M
(Mesa) and an Intel Alder Lake iGPU (`iHD`), and VideoToolbox on Apple
silicon. The NVIDIA backend builds for Windows but has not been run there.

## License

MIT
