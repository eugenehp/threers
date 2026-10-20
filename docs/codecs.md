# Native codecs

Part of the [threers](../README.md) documentation.

# Native codecs (GIF, APNG, VP9, H.264, HEVC)

Enable `native-codec` for dependency-free encoders that also build on `wasm32`; GIF (encode and decode) is the `gif` feature, which implies `native-codec`. `media` turns on everything below.

```rust
// Cargo.toml: threers = { version = "0.0.7", features = ["gif"] }
use threers::{encode_gif, decode_gif, GifEncoder, GifOptions, PaletteMode};

let mut enc = GifEncoder::new(64, 64, 0)
    .colors(64)
    .dither(true)
    .diff_rects(true)
    .palette_mode(PaletteMode::Local);
// enc.add_frame(&rgba, 1, 10);
let bytes = enc.finish();
let (_info, frames) = decode_gif(&bytes).unwrap();
```

See `threers::codec` module docs and `tests/{gif,apng,webm,h264_*,hevc_*}.rs` for VP9/WebM, APNG, H.264/MP4, and HEVC usage.

## Decoding H.264 and HEVC

`threers::codec::decode` holds from-scratch H.264 and HEVC decoders: the software counterpart of `videotoolbox_decode::VideoToolboxDecoder`, with the same interface, but pure Rust. It runs on every platform, including `wasm32`. The decoders need the `video` feature as well as `native-codec` (`media` turns on both), so an app that only encodes does not compile them. Open it from a demuxed track, push samples in decode order, and take pictures back by presentation time:

```rust
use threers::codec::decode::NativeDecoder;
use threers::codec::mp4_demux::demux_video;

let file = std::fs::read("clip.mp4").unwrap();
let track = demux_video(&file).unwrap();
let mut dec = NativeDecoder::new(&track).unwrap();
for s in &track.samples {
    let bytes = &file[s.offset as usize..][..s.size as usize];
    dec.push(bytes, s.pts, s.dts, s.duration as i64).unwrap();
    if let Some(bgra) = dec.take(s.pts) {
        // width * height * 4, like VideoToolbox's output
    }
}
```

`take_frame` returns the planes (`DecodedFrame`, at the stream's own bit depth) instead of BGRA. `decode_annexb` decodes a raw `.264` / `.265` elementary stream into display order.

| | Supported | Refused with an error |
|---|---|---|
| H.264 | Baseline, Main and High (progressive); CAVLC and CABAC; I/P/B slices, multiple slices, 8×8 transform, scaling matrices; explicit and implicit weighted prediction; spatial and temporal direct; long-term references and every MMCO; 4:2:0 and monochrome, 8–14 bit | interlaced (field pictures, MBAFF), 4:2:2/4:4:4, FMO, data partitioning, SP/SI, lossless transform bypass |
| HEVC | Main, Main 10, Main Still Picture, 4:2:0 Main 12; AMP, tiles, WPP, dependent slice segments, PCM, transform skip, transquant bypass, sign data hiding, scaling lists, weighted prediction, long-term references, TMVP, deblocking, SAO; base layer of multi-layer streams (Apple's alpha layer is skipped) | 4:2:2/4:4:4 and the other range-extension tools |

Output is bit-exact. `tests/native_decode.rs` compares every decoded sample against an independent truth: the crate's own encoders (PCM must come back lossless), x264 streams against ffmpeg's decode, and x265 streams against x265's own reconstruction. The external-tool checks skip themselves when the tools are not installed. During development the decoders were also matched against VideoToolbox-encoded H.264 and HEVC (8- and 10-bit), and against libde265.

Some tools are implemented but appear in none of those streams, because no encoder available here produces them: H.264 long-term references, MMCO operations 2–6, POC type 1 and CABAC `I_PCM`; HEVC long-term references and dependent slice segments. Treat those paths as less proven than the rest.

For HEVC the reference is HM (`TAppDecoder`), the JCT-VC reference decoder. Every stream above decodes identically to it, including 10- and 12-bit and VideoToolbox output. HM isn't packaged, so `tests/native_decode.rs` uses it only when `THREERS_HM_DECODER` points at the binary. To build it on macOS, clone `https://vcgit.hhi.fraunhofer.de/jvet/HM.git`, remove `-Werror` from `cmake/CMakeBuild/cmake/modules/BBuildEnv.cmake` (newer clang trips it), then `cmake -DCMAKE_BUILD_TYPE=Release .. && make TAppDecoder`. Without HM, the test compares against x265's `--recon`. With `THREERS_HEVC_CONFORMANCE` also set to a copy of the JCT-VC conformance bitstreams (`rsync -rltLW rsync://fate-suite.ffmpeg.org/fate-suite/hevc-conformance/ dir`), the whole suite is checked too: 172 of 188 streams match HM, the 10 using 4:2:2/4:4:4 or range extensions are refused, and HM itself cannot decode the other 6. Including the decode order of pictures, which follows the spec's DPB output process (§C.5.2). Don't use ffmpeg or libde265 as the HEVC reference: ffmpeg 8 deblocks chroma too late for SAO with 16×16 CTBs (off by two chroma samples; a fix is prepared for upstream), and libde265 wrong on multi-slice x265 streams.

Speed is single-threaded scalar Rust, about 15–20 ms per 1080p frame on an M-series Mac: fast enough for real-time 1080p30, but several times slower than ffmpeg's hand-written SIMD. Where VideoToolbox is available, it remains the faster choice.
