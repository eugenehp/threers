# Native codecs

Part of the [threers](../README.md) documentation.

# Native codecs (GIF, APNG, VP9, H.264, HEVC, AV1, audio)

The codecs live in their own crate, [`threers-codec`](../crates/threers-codec), which `threers` re-exports as `threers::codec`. Through `threers`, enable `native-codec` for dependency-free encoders, muxers and audio decoders that also build on `wasm32`; GIF (encode and decode) is the `gif` feature and the video decoders are `native-decode` (or `video`), both of which imply `native-codec`. `media` turns on everything below.

A program that only encodes or decodes media can depend on `threers-codec` directly and skip the renderer. Its features are `decode` and `gif` (both on by default), `parallel` and `videotoolbox`, and `threers_codec::prelude` imports the common types:

```rust
// Cargo.toml: threers-codec = { version = "0.0.9", features = ["parallel", "videotoolbox"] }
use threers_codec::prelude::*;

let mut enc = Mp4Encoder::new(1280, 720, EncodeOptions::default()).unwrap();
// enc.push_rgba(&rgba) per frame
let mp4: Vec<u8> = enc.finish().unwrap();
```

```rust
// Cargo.toml: threers = { version = "0.0.9", features = ["gif"] }
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

See the `threers_codec` crate docs and `crates/threers-codec/tests/{gif,apng,webm,h264_*,hevc_*}.rs` for VP9/WebM, APNG, H.264/MP4, and HEVC usage.

## Hardware acceleration

`codec::hw::Mp4Encoder` and `codec::decode::Decoder` put the hardware codecs and the native ones behind one interface each. With the `videotoolbox` feature on macOS they open a VideoToolbox session; with the `nvidia` feature on Linux or Windows they use the GPU's NVENC and NVDEC engines (HEVC encode; H.264, HEVC 8/10-bit, VP9 and AV1 decode). Without the feature, without the hardware, or when the hardware refuses a stream (VideoToolbox and VP9/AV1, for one), they fall back to the pure-Rust codec instead of failing.

The NVIDIA libraries (`libcuda`, `libnvcuvid`, `libnvidia-encode`; the `nv*.dll`s on Windows) are opened at run time rather than linked, so a binary built with `nvidia` starts on any machine and simply decodes natively where there is no driver. Driver 520 (Linux) or 522 (Windows) or newer is needed.

With the `vaapi` feature on Linux, `Decoder` also decodes H.264, HEVC (Main, Main 10), VP9 (profiles 0 and 2) and AV1 (Main, 8- and 10-bit) on any GPU with a VA driver, AMD through Mesa and Intel through `iHD` or `i965`, and `Mp4Encoder` encodes HEVC there. VAAPI does not parse the bitstream, so the native decoders still do: they read the headers, number and mark references, build the reference lists and order output, and hand each slice, frame or tile group to the GPU through accelerator hooks (`decode::h264::H264Accel`, `decode::hevc::HevcAccel`, `decode::vp9::Vp9Accel`, `decode::av1::Av1Accel`) instead of reconstructing it. AV1 film grain stays with the native decoder, applied to what the GPU returns, so grain streams match too. For encoding, threers writes the VPS, SPS and PPS (with the colour description) and slice headers itself as packed headers, and follows what the driver reports it requires: Intel's low-power encoder (the only one on recent Intel GPUs) gets asymmetric partitions, per-CU QP and its transform depth turned on, and P frames coded as B slices whose two lists hold the same picture. Both drivers have been run: Mesa on a Radeon 780M and Intel's `iHD` on an Alder Lake iGPU, which decodes H.264, HEVC and VP9 (its free build has no AV1). On a Radeon 780M (Ryzen 9 8945HS) that takes 1080p decoding from about 11–12 ms a frame to about 4 ms, including the copy back to memory, and halves the CPU time. `crates/threers-codec/tests/vaapi.rs` checks the VAAPI pictures against the native decoder's, byte for byte, over H.264 streams with B-frames, CAVLC, 8×8 transforms and scaling matrices, weighted prediction, several slices and MBAFF, and HEVC streams with B-frames, 10-bit samples, WPP, tiles, several slices, scaling lists, weighted prediction and an odd picture size, VP9 streams with alt-ref superframes, tiles, error resilience, segmentation, 10-bit and lossless coding, and AV1 streams from SVT-AV1 and libaom (10-bit, film grain, tiles, screen content) plus the libaom and FATE conformance vectors; and that the encoder's MP4s decode cleanly at sizes that are not whole CTBs. `crates/threers-codec/tests/nvidia.rs` checks that NVDEC's pictures are identical to the native decoder's on H.264 and HEVC streams with B-frames, 10-bit HEVC, VP9 and AV1, and that NVENC's files decode cleanly; it ran on an RTX 3080 Ti under Linux. The Windows build compiles but has not been run. `backend()` says which one you got, and `with_backend` pins one:

```rust
use threers::codec::hw::{EncodeBackend, EncodeOptions, Mp4Encoder};

let opts = EncodeOptions { fps: 30, quality: 0.8, ..Default::default() };
let mut enc = Mp4Encoder::new(1920, 1080, opts).unwrap();
println!("{:?}", enc.backend()); // VideoToolbox on a Mac, Native elsewhere
```

`EncodeOptions::quality` runs from 0 to 1: VideoToolbox takes it as its own quality property, and the native encoder maps it onto quantizers 40 to 18. `bitrate` sets a target instead. The native encoder is all-intra, so it is slower and its files are larger. With `parallel` it encodes batches of frames on every core, with output byte-identical to a sequential encode. `crates/threers-codec/tests/hw.rs` encodes on every backend this build has, decodes each result with every decoder, and checks that they agree.

### Testing the hardware backends

Each hardware suite skips itself on a machine without the hardware, so a green run proves nothing unless the hardware is required: `THREERS_REQUIRE_NVIDIA=1` makes `tests/nvidia.rs` fail without an NVIDIA GPU and driver, and `THREERS_REQUIRE_VAAPI=1` makes `tests/vaapi.rs` fail without a VA driver (a driver may still lack a codec — Intel's free `iHD` has no AV1 — and that case is reported and skipped). Run GPU suites with `RUST_TEST_THREADS=1`: on Linux, NVIDIA's driver hangs at thread exit when several test threads hold devices. AMD (Mesa) and Intel (`iHD`) read different parts of the VA parameters — Mesa re-parses headers that Intel takes from the buffers — so a change to the VAAPI backend wants both. `tests/av1_conformance.rs` reads AV1 vectors from `THREERS_AV1_CONFORMANCE`; `tests/vaapi.rs` checks them too, from `/var/tmp/av1conf` by default.

```sh
THREERS_REQUIRE_NVIDIA=1 RUST_TEST_THREADS=1 cargo test -p threers-codec --release --features nvidia,parallel
THREERS_REQUIRE_VAAPI=1 RUST_TEST_THREADS=1 cargo test -p threers-codec --release --features vaapi,parallel
```

## Decoding H.264 and HEVC

`threers::codec::decode` holds from-scratch H.264 and HEVC decoders: the software counterpart of `videotoolbox_decode::VideoToolboxDecoder`, with the same interface, but pure Rust. It runs on every platform, including `wasm32`. The decoders need `native-decode` or `video` (`media` turns them on), so an app that only encodes does not compile them. VP9 and AV1 decoders sit alongside them in `decode::vp9` and `decode::av1`. Open it from a demuxed track, push samples in decode order, and take pictures back by presentation time:

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

Output is bit-exact. `crates/threers-codec/tests/native_decode.rs` compares every decoded sample against an independent truth: the crate's own encoders (PCM must come back lossless), x264 streams against ffmpeg's decode, and x265 streams against x265's own reconstruction. The external-tool checks skip themselves when the tools are not installed. During development the decoders were also matched against VideoToolbox-encoded H.264 and HEVC (8- and 10-bit), and against libde265.

Some tools are implemented but appear in none of those streams, because no encoder available here produces them: H.264 long-term references, MMCO operations 2–6, POC type 1 and CABAC `I_PCM`; HEVC long-term references and dependent slice segments. Treat those paths as less proven than the rest.

For HEVC the reference is HM (`TAppDecoder`), the JCT-VC reference decoder. Every stream above decodes identically to it, including 10- and 12-bit and VideoToolbox output. HM isn't packaged, so `crates/threers-codec/tests/native_decode.rs` uses it only when `THREERS_HM_DECODER` points at the binary. To build it on macOS, clone `https://vcgit.hhi.fraunhofer.de/jvet/HM.git`, remove `-Werror` from `cmake/CMakeBuild/cmake/modules/BBuildEnv.cmake` (newer clang trips it), then `cmake -DCMAKE_BUILD_TYPE=Release .. && make TAppDecoder`. Without HM, the test compares against x265's `--recon`. With `THREERS_HEVC_CONFORMANCE` also set to a copy of the JCT-VC conformance bitstreams (`rsync -rltLW rsync://fate-suite.ffmpeg.org/fate-suite/hevc-conformance/ dir`), the whole suite is checked too: 172 of 188 streams match HM, the 10 using 4:2:2/4:4:4 or range extensions are refused, and HM itself cannot decode the other 6. Including the decode order of pictures, which follows the spec's DPB output process (§C.5.2). Don't use ffmpeg or libde265 as the HEVC reference: ffmpeg 8 deblocks chroma too late for SAO with 16×16 CTBs (off by two chroma samples; a fix is prepared for upstream), and libde265 wrong on multi-slice x265 streams.

Speed is single-threaded Rust written so the compiler vectorises the hot loops. On one performance core of an i9-12900H, 1080p decodes at about 16 ms a frame for H.264 and HEVC, 13 ms for VP9 and 22 ms for AV1 (x264, x265, libvpx and SVT-AV1 streams at ordinary settings): real-time at 1080p30 for all but AV1, but still three to five times slower than ffmpeg's hand-written SIMD. Where VideoToolbox or NVDEC is available, `Decoder` uses it.
