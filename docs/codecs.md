# Native codecs

Part of the [threers](../README.md) documentation.

# Native codecs (GIF, APNG, VP9, H.264, HEVC, AV1, audio)

The codecs live in their own crate, [`threers-codec`](../crates/threers-codec), which `threers` re-exports as `threers::codec`. Through `threers`, enable `native-codec` for dependency-free encoders, muxers and audio decoders that also build on `wasm32`; GIF (encode and decode) is the `gif` feature and the video decoders are `native-decode` (or `video`), both of which imply `native-codec`. `media` turns on everything below.

A program that only encodes or decodes media can depend on `threers-codec` directly and skip the renderer. Its features are `decode` and `gif` (both on by default), `parallel` and `videotoolbox`, and `threers_codec::prelude` imports the common types:

```rust
// Cargo.toml: threers-codec = { version = "0.0.12", features = ["parallel", "videotoolbox"] }
use threers_codec::prelude::*;

let mut enc = Mp4Encoder::new(1280, 720, EncodeOptions::default()).unwrap();
// enc.push_rgba(&rgba) per frame
let mp4: Vec<u8> = enc.finish().unwrap();
```

```rust
// Cargo.toml: threers = { version = "0.0.12", features = ["gif"] }
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

## Encoding AV1

`threers_codec::av1::Av1Encoder` writes AV1 from scratch: 8-bit 4:2:0, sides a multiple of 8, a `qidx` from 1 to 255. Every picture is a key frame, so a clip compresses like Motion JPEG rather than like video.

```rust
use threers_codec::av1::Av1Encoder;

let mut enc = Av1Encoder::new(1280, 720, 80).unwrap();
// let unit: Vec<u8> = enc.encode(&y, &u, &v).unwrap();
```

Each call returns one temporal unit for `mux_webm` with `WebmCodec::Av1`. It chooses blocks of 8, 16 or 32 by activity and uses DC prediction and the DCT, with default probabilities and no loop filters, so it needs more bytes than a full encoder at the same quality. `tests/av1_encode.rs` holds it to dav1d: both decoders must produce the same samples.

## Hardware acceleration

`codec::hw::Mp4Encoder` and `codec::decode::Decoder` put the hardware codecs and the native ones behind one interface each. With the `videotoolbox` feature on macOS they open a VideoToolbox session; with the `nvidia` feature on Linux or Windows they use the GPU's NVENC and NVDEC engines (HEVC encode; H.264, HEVC 8/10-bit, VP9 and AV1 decode). Without the feature, without the hardware, or when the hardware refuses a stream, they fall back to the pure-Rust codec instead of failing. VideoToolbox also decodes AV1 (M3 and later) and VP9. `native-decode` turns all the hardware decoders on; see [Video on a mesh](#video-on-a-mesh).

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
| H.264 | Baseline, Main and High, field pictures and MBAFF included; CAVLC and CABAC; I/P/B slices, multiple slices, 8×8 transform, scaling matrices; explicit and implicit weighted prediction; spatial and temporal direct; long-term references and every MMCO; 4:2:0, 4:2:2 and monochrome, 8–14 bit | 4:4:4, FMO, data partitioning, SP/SI, lossless transform bypass |
| HEVC | Main, Main 10, Main Still Picture, 4:2:0 Main 12; AMP, tiles, WPP, dependent slice segments, PCM, transform skip, transquant bypass, sign data hiding, scaling lists, weighted prediction, long-term references, TMVP, deblocking, SAO; base layer of multi-layer streams (Apple's alpha layer is skipped) | 4:2:2/4:4:4 and the other range-extension tools |

Output is bit-exact. `crates/threers-codec/tests/native_decode.rs` compares every decoded sample against an independent truth: the crate's own encoders (PCM must come back lossless), x264 streams against ffmpeg's decode, and x265 streams against x265's own reconstruction. The external-tool checks skip themselves when the tools are not installed. During development the decoders were also matched against VideoToolbox-encoded H.264 and HEVC (8- and 10-bit), and against libde265.

The JVT H.264 and JCT-VC HEVC conformance sets cover what no encoder here produces: long-term references, MMCO 2–6, POC type 1, CABAC `I_PCM`, field pictures, MBAFF and dependent slice segments. `h264_conformance_suite_matches_ffmpeg` runs the first against ffmpeg with `THREERS_H264_CONFORMANCE` set to a copy of `rsync://fate-suite.ffmpeg.org/fate-suite/h264-conformance/`: 189 of 195 streams match, and the other six use FMO or SP/SI, which are refused.

For HEVC the reference is HM (`TAppDecoder`), the JCT-VC reference decoder. Every stream above decodes identically to it, including 10- and 12-bit and VideoToolbox output. HM isn't packaged, so `crates/threers-codec/tests/native_decode.rs` uses it only when `THREERS_HM_DECODER` points at the binary. To build it on macOS, clone `https://vcgit.hhi.fraunhofer.de/jvet/HM.git`, remove `-Werror` from `cmake/CMakeBuild/cmake/modules/BBuildEnv.cmake` (newer clang trips it), then `cmake -DCMAKE_BUILD_TYPE=Release .. && make TAppDecoder`. Without HM, the test compares against x265's `--recon`. With `THREERS_HEVC_CONFORMANCE` also set to a copy of the JCT-VC conformance bitstreams (`rsync -rltLW rsync://fate-suite.ffmpeg.org/fate-suite/hevc-conformance/ dir`), the whole suite is checked too: 172 of 188 streams match HM, the 10 using 4:2:2/4:4:4 or range extensions are refused, and HM itself cannot decode the other 6. Including the decode order of pictures, which follows the spec's DPB output process (§C.5.2). Don't use ffmpeg or libde265 as the HEVC reference: ffmpeg 8 deblocks chroma too late for SAO with 16×16 CTBs (off by two chroma samples; a fix is prepared for upstream), and libde265 wrong on multi-slice x265 streams.

With `parallel`, `decode_annexb` and `decode_video` reconstruct H.264 and HEVC pictures concurrently: headers stay in order on the calling thread, and a picture starts once its references have finished. On an i9-12900H a 1080p x265 stream goes from 12.1 to 6.4 ms a frame and an x264 one from 12.6 to 7.7, reached by four threads. More do not help, because a picture waits for its reference to finish completely, so the chain of P-frames sets the pace. Field and MBAFF H.264 stays on the calling thread, and `NativeDecoder::push` is unthreaded.

Speed on one thread is Rust written so the compiler vectorises the hot loops. On one performance core of an i9-12900H, 1080p decodes at about 16 ms a frame for H.264 and HEVC, 13 ms for VP9 and 22 ms for AV1 (x264, x265, libvpx and SVT-AV1 streams at ordinary settings): real-time at 1080p30 for all but AV1, but still three to five times slower than ffmpeg's hand-written SIMD. Where VideoToolbox or NVDEC is available, `Decoder` uses it.

## Video on a mesh

`VideoPlayer`, in the [`threers-video`](../crates/threers-video) crate, decodes a file and shows it in a texture, for any material slot: a TV screen's `emissive_map`, a cube's `map`, a billboard in a game level. It plays 1080p, 4K and 8K.

```rust
use threers_video::VideoPlayer;
use threers::{Color, Mesh, Object3D, PlaneGeometry, StandardMaterial};

let mut player = VideoPlayer::open(std::fs::read("clip.mp4").unwrap()).unwrap();
player.set_looping(true);
let mut screen = StandardMaterial::new(Color::from_hex(0x111111)).with_emissive_map(player.texture());
screen.emissive = Color::WHITE; // glows with the picture, whatever the lighting
scene.add(Object3D::mesh(Mesh::new(PlaneGeometry::new(1.6, 0.9), screen.into())));
// Each frame, with the time that passed:
player.update(dt);
```

It reads MP4/MOV (H.264, HEVC, AV1), WebM/Matroska and IVF (VP9, AV1). Natively a thread decodes a few pictures ahead, so a frame never waits on the decoder; in a browser (wasm) it decodes inline as each picture comes due. A late `update` skips to the picture due rather than showing the ones in between. `pause`, `play`, `seek`, `set_speed` and `set_looping` do what they say.

### How large pictures stay fast

- **Hardware decoding, by default.** `native-decode` turns on every hardware decoder the platform has, each loaded at run time and compiled only where it exists: VideoToolbox on macOS (H.264, HEVC, VP9, and AV1 on M3 and later), NVDEC on NVIDIA under Linux and Windows (H.264, HEVC, VP9, AV1), VAAPI on Linux (AMD, Intel). `VideoReader` tries them in that order and falls back to the native decoders for anything they refuse, including a stream one accepts but then gives no picture for (NVDEC with VP9 below its smallest size). Every hardware path here matches the native decoders bit for bit, checked frame by frame on all four codecs. `VideoReader::backend()` says which is in use; `open_software` keeps to the native decoders.
- **The hardware's own format, to the GPU.** VideoToolbox, NVDEC and VAAPI hand back NV12 (P010 above 8 bits) as `decode::PackedFrame`, through `VideoReader::next_picture`, and it goes to the texture as it is: no pass over the pixels on the CPU. VAAPI's rows are copied straight out of the surface's image (`VaapiDecoder::take_picture`); AV1 with film grain comes as planes, since the grain is added on the CPU. On NVDEC that took 8K from 66–82 ms a picture to 20–29 ms; on a Radeon 780M, 8K AV1 from 135 ms to 39 ms, 8K VP9 from 72 to 39 and 8K HEVC from 72 to 53.
- **The GPU converts the colour.** The texture takes the decoder's YUV (`textures::YuvFrame`, planar or interleaved), uploaded as `R8Uint`/`R16Uint`/`Rg8Uint`/`Rg16Uint` textures, and one shader pass on wgpu or Metal converts, turns the picture over and rebuilds the mip chain. 8-bit 4:2:0 is 1.5 bytes a pixel to upload against RGBA's 4. Depths from 8 to 16 bits, 4:2:0, 4:2:2, 4:4:0, 4:4:4, monochrome, BT.601/709/2020 and both ranges; the shader and `YuvFrame::to_rgba_into` (its CPU twin) agree within one level.
- **Every core for the native decoders.** H.264 and HEVC reconstruct several pictures at once on the thread pool (`native-decode` turns on the codec's threads; single-threaded wasm is unaffected), with identical output: 2–2.5x faster at 4K and 8K.
- **Files mapped, not read.** `VideoPlayer::open_file` and `VideoReader::open_file` map the file (`codec::bytes::Bytes::map`, `mmap` on Unix with sequential read-ahead advice): the first picture is up after reading the index and one sample, and a film of many gigabytes is never all in memory. MP4, Matroska/WebM and IVF are all read in place: the demuxer records where each sample lies in the file (`mkv_demux::Mkv::index`, `mkv_demux::index_ivf`) instead of copying it. Only a Matroska track with header stripping has its packets copied, to put the stripped bytes back.
- **Mipmaps.** A `VideoTexture` has a mip chain, rebuilt on the GPU after each frame, so a 4K picture on a small screen does not shimmer. `without_mipmaps()` turns it off.
- **Too big for the GPU.** A frame larger than the device's largest texture is shrunk by a whole factor in the same pass, averaging in linear light.

Decoding speed, a picture at a time through `VideoReader::next_picture`:

| Clip | NVIDIA RTX 3080 Ti Laptop, NVDEC (machine idle) | M4 Pro, VideoToolbox (machine heavily loaded) | Radeon 780M, VAAPI | Native, 20-core Linux, all cores |
|---|---|---|---|---|
| 4K H.264 | 9.2 ms (108 fps) | 4–6 ms | 19 ms | 21 ms |
| 4K HEVC | 7.0 ms (142 fps) | 3–8 ms | 22–25 ms | 27 ms |
| 4K AV1 | 5.1 ms (195 fps) | 5 ms | 20 ms | 44 ms |
| 4K VP9 | 6.9 ms (144 fps) | 9 ms | 22 ms | 42 ms |
| 8K HEVC | 29 ms (35 fps) | 12–22 ms | 53 ms | 115 ms |
| 8K AV1 | 20 ms (50 fps) | 17 ms | 39 ms | 160 ms |
| 8K VP9 | 24 ms (41 fps) | 27 ms | 39 ms | 195 ms |
| 8K H.264 | past NVDEC's H.264 limit: native | ~110 ms, past the hardware limit | past the hardware limit | 204 ms |

On the Radeon, 8K HEVC's 53 ms was about 17 ms decoding, 19 ms in the driver's `vaGetImage` copy out of the decoder's surface, 2 ms copying the rows, and most of the rest the kernel handing a fresh 50 MB buffer to each picture. Those buffers are now reused: the hardware decoders copy into buffers from `VideoReader::frame_pool` (`codec::bytes::FramePool`), and `VideoPlayer` gives each picture's buffers back once the texture has replaced it (`TextureStream::swap_yuv`), or skipped it as late; the 8-bit narrowing of the native decoders' pictures draws from the same pool. That took 8K HEVC to 35–39 ms on the Radeon and 17–21 ms on NVDEC, and 4K HEVC on the Radeon from 12–22 ms to 9 ms; AV1 and VP9, whose buffers the allocator was already reusing, are unchanged. A caller of `VideoReader` gets the same with `recycle`.

In a window on the M4 Pro (loaded, load average about 40), 4K clips showed 29 of 30 fps with the scene at 55–58 fps on Metal and wgpu, and 8K HEVC about 25 of 30 fps with the scene at 36 fps. H.264 stops at about 4K in hardware everywhere; HEVC, AV1 and VP9 are the 8K codecs.

### In a browser

- **The browser's decoder.** `VideoPlayer::open` in wasm (and `WebVideoPlayer.open(bytes)` for JavaScript) demuxes the file itself and feeds the browser's WebCodecs `VideoDecoder`, asking for hardware, with the codec string and description `codec::decode::webcodecs` reads from the stream (`avc1.640028`, `hvc1.1.6.L120.90`, `av01.0.08M.08`, `vp09.00.41.08`) and each chunk marked key or delta. A browser without WebCodecs, or one that refuses the codec, gets the native decoders.
- **Frames never come back to the CPU.** A decoded `VideoFrame`, or a playing `<video>` element, goes into the texture with `copyExternalImageToTexture`, GPU to GPU (`TextureStream::write_browser`, `WebDataTexture.writeVideo`). The three.js shim's `VideoTexture` uses this, copying only when `requestVideoFrameCallback` says there is a new frame; before, it drew each frame to a 2D canvas and read it back with `getImageData`, 33 MB a frame at 4K.
- `tests/web_video` checks both in real Chrome, on WebGPU and on WebGL2.

### Sound

The soundtrack (AAC in MP4/MOV, Opus in WebM/Matroska) decodes on a thread of its own; `VideoPlayer::soundtrack()` hands it over once ready as an interleaved `audio::Audio`. threers has no audio output of its own, so playing it is up to the app: start it at `time()`, and start it again there whenever `jumps()` moves on (each seek, and each time a looping clip goes round) or playback resumes. In a `threers-ecs` game with the `audio` feature and an `AudioPlugin`, `Spawned::video(player)` does all of that: the sound starts where the picture is, follows seeks, loops, pauses, `set_speed` and the game's time scale, and holds with everything else while the game is paused. `Video::muted(player)` leaves it silent. The mixer's `Command::Seek` (`AudioOut::seek`) moves a playing voice.

### The pieces

| Piece | What it does |
|---|---|
| `codec::decode::VideoReader` | A file's pictures one at a time in display order, with `rewind`. B-frames are reordered from the container's timestamps, holding back only the pictures that need it. `open_shared` takes the bytes in an `Arc`, for a caller that reads them again. |
| `codec::decode::demux`, `webcodecs` | The track as a decoder is fed it (`Demuxed`: codec, configuration record, samples with times), and what a platform decoder needs on top: the WebCodecs codec string, the description, which samples are key frames. |
| `codec::bytes::Bytes` | Shared bytes, in memory or a mapped file (`Bytes::map`); everything that reads a whole file takes them. |
| `textures::TextureStream` | A frame that changes while the texture stays the same: `write_yuv` (planes for the GPU to convert), `write_rgba` or `write_with`, from any thread. A renderer that misses frames shows the newest. |
| `textures::VideoTexture` | A `Texture` carrying a stream: sRGB, linear filtering, a mip chain. |
| `Texture::pixels()` | The pixels now: a streamed texture's newest frame, converted on the CPU if it is YUV. The path tracer and the USD exporter read this, so they see the frame on show. |

The wgpu and Metal renderers give a streamed texture one GPU texture, made once, and put each new frame into it. The material, the texture's identity and every bind group around it stay the same. Changing pixels through `Texture::replace_rgba_bytes` builds a new GPU texture each time and is meant for occasional updates.

`crates/threers-video/examples/video_texture.rs` plays a clip on a screen and a spinning cube (`cargo run -p threers-video --example video_texture`); `--features videotoolbox` for hardware decoding, `--seconds N` to print the frame rate and how many video frames were shown. Without a file, it makes its own clip with the H.264 encoder. `tests/video_texture.rs` checks the reader, the player's timing, looping and jumps, the soundtrack in both containers, that both renderers show the codec's own colours, and that VideoToolbox matches the native decoder; `src/renderer/yuv.rs` checks the GPU conversion against the CPU's across depths, subsamplings, matrices and ranges.
