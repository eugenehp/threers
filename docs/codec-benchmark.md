# Codec benchmark: native Rust vs ffmpeg vs VideoToolbox

Why exports balloon, measured rather than guessed.

## Running it

```sh
cargo build --release --example codec_bench --features native-codec
python3 scripts/codec_bench.py --clip all --out out/codec-bench
```

`examples/codec_bench.rs` is the native arm — it encodes a raw `yuv420p` clip
with each pure-Rust encoder and reports bytes and encode time.
`scripts/codec_bench.py` prepares the clips, runs the ffmpeg and VideoToolbox
arms, decodes every result back to `yuv420p`, and scores it.

## Method

Every arm is fed the **same planar `yuv420p` bytes** and every output is decoded
back to `yuv420p` and compared against that same reference, so what is measured
is the codec and not somebody's colour conversion. The native encoders read the
planes directly rather than converting from RGBA, for the same reason.

Two ffmpeg configurations are reported, because only both together are honest:

- **all-intra** (`-g 1`) — the like-for-like number, since the native encoders
  code every frame independently;
- **default GOP** — what you would actually ship, and so the gap that matters to
  the file on disk.

The encode-rate column is not quite like-for-like: the native arm reports the
encoder's own timer, while the ffmpeg arms are wall clock including process
startup and reading the raw clip. That bias favours the native encoders, and
they are still several times slower — so the ordering holds regardless.

### A note on VMAF

VMAF is a fitted regression, not a distance. On moving content it predicts under
100 even for a bit-identical decode — 98.89 on the `frog` clip, 99.92 on
`static`. The harness scores the reference against itself first and reports that
ceiling, so the arms below are read against what is actually reachable rather
than against a 100 that nothing can score.

## Results

### `frog` — 1280×1440, 17 frames @ 24fps, real render (raw 47.0 MB, ceiling 98.89)

| Arm | Size | PSNR | VMAF | Encode |
|---|---|---|---|---|
| native h264 `I_PCM` | **47.25 MB** | ∞ | 98.89 | 178 fps |
| native hevc `I_PCM` | **47.25 MB** | ∞ | 98.89 | 126 fps |
| native hevc intra, prediction-only | 0.03 MB | 16.6 | 0.31 | 8 fps |
| **native hevc intra + residual qp22** | **0.14 MB** | **59.9** | **97.67** | 3.5 fps |
| **native hevc intra + residual qp27** | **0.11 MB** | **52.4** | **96.48** | 3.5 fps |
| **native hevc intra + residual qp32** | **0.08 MB** | **48.3** | **93.57** | 3.6 fps |
| **native hevc intra + residual qp37** | **0.06 MB** | **45.0** | **89.50** | 2.5 fps |
| x264 all-intra crf23 | 0.12 MB | 59.0 | 97.86 | 193 fps |
| x265 all-intra crf23 | 0.18 MB | 66.9 | 98.49 | 19 fps |
| x264 crf23 | 0.06 MB | 54.4 | 96.22 | 177 fps |
| x265 crf23 | 0.09 MB | 56.7 | 96.70 | 67 fps |
| hevc_videotoolbox 4M | 0.08 MB | 59.6 | 97.07 | 75 fps |

### `static` — 1280×720, 48 frames @ 24fps (raw 66.4 MB, ceiling 99.92)

| Arm | Size | PSNR | VMAF | Encode |
|---|---|---|---|---|
| native h264 `I_PCM` | **66.70 MB** | ∞ | 99.92 | 365 fps |
| native hevc intra, prediction-only | 0.04 MB | 13.7 | **0.00** | 11 fps |
| native hevc intra + residual (qp27) | 0.44 MB | 5.9 | **0.00** | 11 fps |
| x264 all-intra crf23 | 0.60 MB | 51.2 | 97.51 | 525 fps |
| x265 all-intra crf23 | 0.67 MB | 55.0 | 98.44 | 33 fps |
| x265 all-intra crf28 | 0.46 MB | 51.6 | 96.78 | 38 fps |
| x264 crf23 | 0.13 MB | 50.7 | 97.32 | 366 fps |
| x265 crf23 | 0.11 MB | 51.8 | 96.35 | 135 fps |
| hevc_videotoolbox 4M | 0.31 MB | 53.9 | 98.42 | 155 fps |

## Findings

**1. The native H.264 path writes uncompressed video.** Both `I_PCM` arms are
bit-exact losslessly (PSNR ∞) at ~1.5 bytes/pixel/frame — 47 MB for 17 frames,
67 MB for 48. `src/video.rs` routes `VideoCodec::H264` here whenever
`native-codec` is enabled (which `media`, `full`, `wasm-full` and `videotoolbox`
all imply), *before* `append_codec_args` runs, so `.crf()` and `.bitrate()` are
silently discarded. That is the single largest cause of a ballooning export, and
it is invisible at the call site.

**2. The native residual coder was non-conformant. Fixed — two wrong bytes in
the CABAC tables.** `transIdxLps[28]` was 23 (spec: 22) and `rangeTabLps[31][0]`
was 28 (spec: 29), in `src/codec/hevc/tables.rs`. Before the fix the residual
path scored PSNR ~5 dB and VMAF ~2 while producing files several times larger
than the useless prediction-only path; after it, VMAF 94.9 at 0.16 MB.

Two things hid this. The encode↔decode round-trip test in `cabac.rs` uses the
**same tables on both sides**, so it agrees with itself while disagreeing with
the standard — a wrong constant is invisible to a symmetric test. And a CABAC
context only reaches state 28 or 31 after a long one-sided run of bins, so the
smooth images in `tests/hevc_compress.rs` never drove it there. It took an
external decoder *and* detailed content, which is what
`tests/hevc_residual_conformance.rs` now does. `tables.rs` also carries a
structural-invariant test: the spec's `transIdxLps` is monotonically
non-decreasing, and `..., 21, 21, 23, 22, 23, ...` is not — that one was
findable with no reference at all.

**3. Inter prediction is the dominant lever, not intra tuning.** On `static`, at
matched quality (VMAF ~96.8 vs ~96.4), x265 all-intra crf28 is 0.46 MB and x265
with a normal GOP is 0.11 MB — **~4× from inter prediction alone**. No amount of
intra work in the native encoder reaches that, because it codes every frame
independently by construction.

**4. The fixed native encoder is now within ~1.1–1.2× of x264/x265 all-intra**,
at 3.5 fps against their 14–30. It is still all-intra, so point 3 applies on top.

Measured as size at matched VMAF, interpolated across a six-point CRF/QP sweep on
each encoder:

| matched VMAF | native vs x265 all-intra | native vs x264 all-intra |
|---|---|---|
| 92 | 1.07× | 1.08× |
| 94 | 1.15× | 1.10× |
| 96 | 1.19× | 1.15× |
| 97 | 1.18× | 1.17× |

**Do not quote this from one clip.** The same sweep on the `frog` render — a
mostly-flat origami model on a plain background — puts native at 0.81–1.05× of
x265, i.e. apparently *beating* it below VMAF 96. That is a property of the clip,
not the encoder: flat content suits an SSD-driven encoder with no
psycho-visual tuning, and it is exactly the kind of result that looks like a
breakthrough until you run a second clip. The table above is from the dense
`testsrc2` content, and is the number to trust of the two.

Two changes got it from the 2.3× it started at:

*Transform-tree splitting* (`max_transform_hierarchy_depth_intra = 1`, so a
16×16 CU may split into four 8×8). This is worth more than "the transform fits
the detail better", because HEVC predicts **per transform block** from already
reconstructed neighbours — splitting also halves how far a block sits from the
samples it predicts from. Measured at fixed QP it was smaller *and* better at
every point (e.g. qp22: 0.25 → 0.19 MB and VMAF 96.74 → 97.36).

*A real rate-distortion split decision*, applied recursively at every level.
Both shapes are built, the reconstruction rolled back between them, and the
cheaper by `SSD + λ·bits` kept — with the bits counted **exactly** by cloning the
CABAC engine and contexts and coding the candidate into the copy, rather than
modelled. Versus forcing the split it bought +2 to +5 dB PSNR at equal or smaller
size, though **VMAF was flat** at that stage (97.36 → 97.16 at qp22). Worth
stating plainly: the cost function minimises SSD, which *is* PSNR, and
SSD-guided decisions tend to favour smoothing that a perceptual metric does not
reward. Judge further work down this road on both columns.

*Splitting all the way to 4x4* (`max_transform_hierarchy_depth_intra = 2`), which
also brings in DST-VII for 4x4 intra luma and the 4:2:0 case where four 4x4 luma
blocks share one 4x4 chroma block carried in the fourth child's transform unit.
This one was unambiguous — smaller *and* better on both metrics at every QP
(qp22: 0.19 → 0.14 MB, PSNR 58.4 → 59.9, VMAF 97.16 → 97.67).

**5. Nothing could reach the compressed encoder.** `CompressedEncoder` was
exported and used by no export path: both `src/video.rs` and the browser path in
`src/codec/animation.rs` went through the `I_PCM` H.264 encoder. So the wasm MP4
download — where ffmpeg genuinely cannot run and this matters most — was
uncompressed. There is now `hevc::encode_compressed_mp4` and a
`BrowserCodec::Mp4Hevc` that uses it.

### Keyframe interval — 20s 1280×720 clip

The `static` and `frog` clips above are too short to measure this: at a 2s
interval a 2s clip has barely two keyframes. On a 20s clip:

| x265 crf23 | Size | VMAF |
|---|---|---|
| keyint 24 (1s) | 1.23 MB | 97.67 |
| keyint 48 (2s) | 1.15 MB | 97.65 |
| keyint 120 (5s) | 1.10 MB | 97.44 |
| keyint 480 (20s) | 1.09 MB | 97.42 |

| hevc_videotoolbox @ 4 Mb/s | Size | VMAF |
|---|---|---|
| live-capture settings, g=2s | 7.04 MB | 99.76 |
| export settings, g=5s | **7.93 MB** | 99.78 |

**These point opposite ways, and the reason matters.** A longer GOP frees up
bits; what happens next depends on what the encoder was told to hit.

- Under a **quality** target (CRF), nothing is obliged to spend the savings, so
  the file shrinks — 12% here going from 1s to 5s, for 0.2 VMAF.
- Under a **bitrate** target, rate control spends whatever the budget allows, so
  the file does not shrink; the bits go into quality instead. The export profile
  above is *larger* than the live-capture one at the same `-b:v` because it
  adheres to the requested bitrate more closely.

So: to make a bitrate-targeted export smaller, lower the bitrate. Tuning the GOP
only helps when you are targeting quality.

## What to use

For file size at quality, on this evidence:

- `VideoCodec::Hevc` (libx265) or `VideoCodec::H264` **without** `native-codec` —
  CRF 18–23 is visually transparent for rendered content.
- `hevc_videotoolbox` when encode wall-clock matters: 75–155 fps and within ~1
  VMAF of x265 on both clips, at 2–3× the size.
- `BrowserCodec::Mp4Hevc` (or `hevc::encode_compressed_mp4`) in the browser and
  anywhere else ffmpeg is unavailable — ~300× smaller than the `I_PCM` MP4 it
  replaces, at the cost of HEVC's narrower playback support versus H.264.

## Changes made off the back of this

- `VideoOptions::keyframe_interval` and `VideoOptions::preset` — neither had any
  control before, so `-g` and `-preset` were never emitted at all.
- `videotoolbox::VtTuning` — the in-process encoder was hardcoded to live-capture
  settings (`RealTime`, `PrioritizeEncodingSpeedOverQuality`, a keyframe every 2
  seconds). Those now default to an export profile, and `VtTuning::realtime()`
  restores the old behaviour for live capture. Measured honestly, this buys
  bitrate adherence and a fraction of a VMAF point — **not** a smaller file. It
  is the right default for an export, but it is not the size fix.
- A warning when the native H.264 path drops a requested quality setting,
  including the size to expect.
- `tables.rs` fixes plus a structural-invariant test; `residual(true)` is now the
  default on `CompressedEncoder`, since it is conformant.
- `hevc::encode_compressed_mp4` / `BrowserCodec::Mp4Hevc`, so the compressed
  encoder is reachable at all.
- Transform-tree splitting, per-transform-block intra prediction with
  decoding-order (z-order) reference availability, mode-dependent coefficient
  scans, and an SSD+λ·bits split decision using exact trial-encoded bit counts.
- Two further conformance fixes the splitting exposed, both scan-related:
  `sig_coeff_flag`'s 8×8 context offset is `(scanIdx == 0) ? 9 : 15` and not a
  constant 9, and a vertical scan transposes the signalled last-significant
  position (§9.3.4.2.5, §7.4.9.11). Neither can be hit without 8×8 luma or 4×4
  chroma blocks, so they were unreachable before the split existed.

### 8K (7680×4320)

3 frames of the same dense `testsrc2` content, raw 149 MB.

| Arm | Size | PSNR | VMAF | Encode (3 frames) |
|---|---|---|---|---|
| native hevc intra qp22 | 2.93 MB | 52.7 | 95.89 | 32.3 s |
| native hevc intra qp30 | 1.79 MB | 45.1 | 92.03 | 26.7 s |
| native hevc intra qp38 | 0.83 MB | 39.0 | 82.37 | 20.9 s |
| x265 all-intra crf28 | 2.12 MB | 51.4 | 95.96 | 2.2 s |
| x265 all-intra crf36 | 1.20 MB | 44.2 | 91.08 | 2.1 s |
| x264 all-intra crf28 | 1.57 MB | 46.8 | 92.99 | 0.3 s |
| hevc_videotoolbox 200M | 2.68 MB | 54.7 | 96.61 | 0.5 s |
| hevc_videotoolbox 50M | 1.08 MB | 42.5 | 90.66 | 0.5 s |

It is **correct** at 8K — the decode matches the encoder's reconstruction across
all 149,299,200 bytes at every QP — but the compression gap widens from the
1.07–1.19× measured at 720p to roughly **1.3–1.4× of x265 all-intra**.

That widening is structural and points at the next thing to fix. The CTB is
pinned to 16×16 (`CTB_SIZE`) with transforms no larger than 16×16, while x265
uses CTBs up to 64×64 and transforms up to 32×32. At 720p a 16×16 block is a
reasonable unit; at 8K the same block covers a ninth of the picture area it used
to, so large smooth regions that one 32×32 transform would carry cheaply are
paid for many times over. Raising `CTB_SIZE` is a bigger change than anything
above — it makes `split_cu_flag` real, since the quadtree currently never splits
— but it is where the remaining gap lives at high resolution.

Speed is the other 8K story: ~7–11 s/frame against x265's 0.7 and
VideoToolbox's 0.23. At this resolution the hardware encoder is the only
sensible choice unless you specifically cannot use it.

> **Superseded.** The coding quadtree that paragraph asks for was built — see
> *Closing the gap* below. Every number in the table above predates it: the
> compression improved, and the speed got about 4× worse. Corrected figures
> follow.

#### Speed, measured after the quadtree

Timing on this machine is unreliable by default: it is shared, and a load
average of 50 on 14 cores turns any wall-clock reading into a reading of what
else was running. The tell is that ffmpeg's own numbers moved 5× between two
runs minutes apart. So these are **minimum (user + sys) CPU seconds over
repeated runs**, which is what survives contention — and the comparison is
therefore total CPU work, not elapsed time.

| | 720p, 10 frames | 8K, 1 frame |
|---|---|---|
| native H.264 | 0.51 s | 1.17 s |
| x264 all-intra | 0.20 s | 0.36 s |
| native HEVC (before optimisation) | 7.93 s | 21.52 s |
| native HEVC (after) | **5.11 s** | **12.1 s** |
| x265 all-intra | 2.65 s | 7.39 s |

The ratio is stable across a 36× change in picture area: **native H.264 costs
2.6–3.3× the CPU of x264, native HEVC 2.9–3.0× the CPU of x265.** For a
from-scratch encoder against two mature ones, that is the honest shape of the
gap, and it is smaller than the wall-clock impression suggests.

Wall clock is worse than that, and for a separate reason. x264 and x265 use
every core; both native encoders use exactly one — there is no `rayon`, no
thread, nowhere in `src/codec/`. At 8K on an unloaded machine x265 finishes its
7.39 s of CPU in about 2.7 s of elapsed time, while the native HEVC encoder
takes its full 21.5 s.

So the elapsed-time gap is two independent factors of about three: **~3×
algorithmic**, which is search pruning this encoder does not do, and **~2.7×
threading**. The second is nearly free to close. Every frame is an IDR and
`compressed_slice_rbsp` takes no `&mut self` — there is no cross-frame state at
all — so frame-level parallelism is not merely safe, it is bit-exact.

A 300-frame 8K export — ten seconds at 30fps — is 1 h 48 min of single-threaded
work today.

### Memory and IO: stream, do not collect

The 8K test surfaced a shape problem that ran through the whole export path.
Three separate places each held the entire clip or the entire file:

1. the encoders took `&[Yuv420Frame]` — a 4:2:0 8K frame is ~50 MB, so ten
   seconds at 24fps is ~12 GB of sources;
2. `export_native_h264` collected every frame before encoding any;
3. the MP4 muxer built the file through **three** full copies of the sample data
   (`concat`, then the `mdat` box, then the final assembly) and returned it as a
   `Vec<u8>` that the caller then wrote.

None of it was necessary. The muxer only needs the *encoded samples*, which are
orders of magnitude smaller than the sources, and the file layout is
`ftyp, mdat, moov` — so chunk offsets are known before the payload is written and
`mdat` can go straight to the sink.

| Measurement | Before | After |
|---|---|---|
| 8K HEVC, 3 frames | 393 MB | 164 MB |
| 8K HEVC, 12 frames | 1242 MB | 177 MB |
| 4K native H.264 export, 24 frames | 1518 MB | **385 MB** |
| 4K native HEVC, 24 frames | — | **88 MB** |

The streaming forms are flat in clip length where the old ones grew ~50 MB per
frame. Output is byte-identical, which the golden-checksum tests in
`tests/h264_animation_parity.rs` already pinned and
`tests/hevc_native_mp4.rs` now pins directly.

The remaining 385 MB on the H.264 path is `I_PCM`'s own doing: its samples are
as large as the raw frames, so they dominate once the sources are gone. The HEVC
path, whose samples are ~1000× smaller, sits at 88 MB and stays there. That is
the same conclusion as finding 1, arrived at from the memory side.

Two smaller ones found along the way: `Yuv420Frame::from_rgba` allocated and
filled an alpha plane for every frame and then discarded it whenever the input
was opaque (33 MB per frame at 8K), and the browser path in
`src/codec/animation.rs` collected every **RGBA** frame — 4 bytes per pixel,
133 MB each at 8K — before encoding. The MP4 codecs there now consume frames
lazily; GIF, APNG and WebM still collect, because they genuinely need the whole
set.

## H.264: the native path now compresses

Finding 1 above — the native H.264 encoder codes `I_PCM`, i.e. raw samples — was
the largest single cause of a ballooning export. `src/codec/h264/` now carries a
real intra encoder: the 4x4 integer transform and quantizer (`transform.rs`),
nine luma and four chroma prediction modes (`intra.rs`), CAVLC residual coding
(`cavlc.rs`), and `I_NxN` macroblock assembly (`compress.rs`). It is conformant —
ffmpeg decodes to the encoder's own reconstruction bit-for-bit — across sizes,
QPs and content, including a randomised sweep (`tests/h264_compress.rs`).

Both export paths use it now: `VideoCodec::H264` and `BrowserCodec::Mp4`. Same
codec, same universal playback reach, about **a hundredth the size**:

| 17 frames of 1280x1440 | Size | PSNR | VMAF | Encode |
|---|---|---|---|---|
| native `I_PCM` (what these used to emit) | 47.25 MB | ∞ | 98.89 | 157 fps |
| native compressed qp22 | **0.48 MB** | 52.3 | 97.71 | 33 fps |
| native compressed qp27 | **0.45 MB** | 47.9 | 96.48 | 33 fps |
| native compressed qp37 | **0.40 MB** | 41.7 | 90.96 | 33 fps |
| x264 all-intra crf28 | 0.09 MB | 55.1 | 96.28 | 347 fps |
| native **HEVC** intra qp27 (for scale) | 0.11 MB | 52.4 | 96.48 | 3.7 fps |

`VideoQuality::Crf(n)` now maps to the encoder's QP instead of being discarded;
`I_PCM` is still reachable as `h264::encode_mp4` for anyone who wants lossless.

**It is not efficient yet — about 5x of x264 at matched VMAF**, and 4x the
project's own HEVC encoder, which reaches the same VMAF 96.48 in 0.11 MB against
this encoder's 0.45 MB. The size barely moves with QP, which says where the bits
are going: `I_NxN` pays sixteen mode signals and sixteen coefficient tokens per
macroblock whether or not there is anything to code, and a rendered frame is
mostly flat. The missing piece is `I_16x16`, whose whole point is to code a flat
macroblock in a handful of bits, followed by the transform-splitting and
rate-distortion work the HEVC path already has. It is, however, ~9x faster than
the HEVC encoder, precisely because it does none of that searching.

### What the work turned up

Six bugs. The last three are the interesting ones, because each was invisible to
everything except an external decoder:

1. **`total_zeros` counted the wrong zeros** — it counts zeros *before* the
   highest-frequency coefficient; I included the ones beyond it, so a DC-only
   block claimed 15 instead of 0.
2. **The in-loop deblocking filter was left on.** H.264 deblocks by default and
   the filter is in-loop, so the encoder's reconstruction drifted from the
   decoder's at every block edge, and more at high QP. The signature was
   unmistakable once a block was printed: the interior matched exactly and only
   the right column and bottom row differed.
3. **`CT1`'s Kraft sum was 1.031** — above 1, impossible for a prefix code.
   Caught by a structural test, no reference needed.
4. **Two `CT1` `coeff_token` entries were transposed.**
5. **Levels were bounded by the wrong thing.** A dequantized coefficient is
   representable well past the point where the *inverse transform's* first-stage
   sums are, and decoders hold those in 16 bits. `|d| = 20800` is a fine
   coefficient whose row sum is not.
6. **Two `total_zeros` entries at `tzVlcIndex = 12` were transposed.** This one
   caused both of the gaps that had been left open — "chroma AC across
   macroblocks" and "steep high-frequency luma" turned out to be two ways of
   reaching a 12-coefficient block.
7. **The chroma-DC bound from fix 5 was itself far too tight.** Bounding each
   level by the case where all four are maximal and aligned clamped a solid
   colour's DC to a twentieth of what it needs — a visible error on saturated
   content. The four levels pass through an inverse Hadamard, so what has to stay
   in range is the *output*; checking that and only scaling when it genuinely
   overflows costs nothing on real content.

### How they were found

Transcribed constants cannot be checked by round-tripping through themselves,
and CAVLC is almost entirely transcribed constants. Two things worked:

* **Structural invariants that hold whatever the values are** — prefix-freeness,
  Kraft, entry counts, a permutation check on the `coded_block_pattern` map.
  That found bug 3 outright and would have found bug 4's table.
* **Probing individual table entries against a real decoder**, by forcing exact
  coefficient patterns through the encoder and sweeping. Every
  `(TrailingOnes, TotalCoeff)` cell in all four `coeff_token` tables, every
  `total_zeros` entry, every `coded_block_pattern` value and the chroma DC table
  are each verified against ffmpeg this way.

Bug 6 is the cautionary one. Every hand-written probe passed; it needed a
12-coefficient block whose levels varied enough to drive the `suffixLength`
adaptation, and no hand-written case happened to produce that. A randomised
sweep over real content found it in seconds. Uniform test patterns are a blind
spot — they were the reason two gaps stayed open through several rounds of
targeted probing.

## Closing the gap: `I_16x16` and a real coding quadtree

The two items at the top of the list below have been implemented. Both are
conformant against ffmpeg, and both were measured twice — once on the `frog`
clip used everywhere else in this document, and once on a clip built from the
ten reference *photographs* in `out/*-ref.jpg`, scaled and cropped to 1280x720.

The second clip exists because the first one lied. `frog` is a rendered origami
model on a white background: overwhelmingly flat, and flat content flatters an
encoder that has just learned to code flat areas cheaply. Measured on `frog`
alone, the native HEVC encoder appears to beat x265 all-intra by 1.4-2.9x, which
is not a result anyone should believe about a from-scratch encoder against a
mature one. On photographs it lands where a careful reading would expect. Both
numbers are below; the photographic ones are the honest ones.

### H.264: `I_16x16`

`I_NxN` pays sixteen mode signals and sixteen coefficient tokens per macroblock
whether or not there is anything to code. `I_16x16` predicts the whole macroblock
at once and sends the sixteen block DCs through a second 4x4 Hadamard, so a flat
macroblock costs a couple of dozen bits. `src/codec/h264/compress.rs` now builds
both candidates per macroblock and picks by exact rate-distortion — CAVLC carries
no adaptive state between macroblocks, so a trial encode is not an estimate, it
is the bit count.

On `frog`, 34 frames of 1280x720, `I_NxN` only against both candidates:

| QP | `I_NxN` only | with `I_16x16` |
|---|---|---|
| 22 | 504 KB @ 51.8 dB | **237 KB @ 59.8 dB** |
| 27 | 469 KB @ 46.0 dB | **200 KB @ 53.3 dB** |
| 32 | 440 KB @ 47.4 dB | **170 KB @ 49.1 dB** |
| 37 | 417 KB @ 36.7 dB | **140 KB @ 46.2 dB** |

Smaller *and* better at every point, which is not the usual shape of a codec
improvement. It happens here because the DC path is also a quality change: the
Hadamard concentrates sixteen block DCs into one coefficient and the quantizer
runs two bits coarser on it, so the effective DC resolution is four times finer.
On content that is mostly flat, DC error is most of the error.

Against x264 all-intra at matched PSNR, the gap went from **5.7-8.6x to
1.4-2.5x** on `frog`, and is **1.10-1.40x** on the photographs.

### HEVC: 64x64 CTBs and a CU quadtree

The CTB was pinned to 16x16 with `log2_diff_max_min_luma_coding_block_size = 0`,
so `split_cu_flag` was never coded and every CTB was exactly one coding unit. It
is now 64x64 with a quadtree down to 8x8 and transforms up to 32x32
(`with_coding_tree(6, 3, 5)`), which needed:

* a 32-point DCT (`src/codec/hevc/transform.rs`);
* `last_sig_coeff` group tables extended from 16 positions to 32;
* `split_cu_flag` with its three contexts, and the `CtDepth` grid they are
  derived from;
* the forced, unsignalled split of coding blocks that hang off the picture edge,
  which is what lets the coded picture stay padded to 8 rather than to 64;
* per-minimum-block mode tracking, because the above-neighbour mode candidate is
  now often another coding unit inside the same CTB rather than always DC.

Photographs, 1280x720:

| QP | CTB 16, no quadtree | CTB 64, quadtree to 8 |
|---|---|---|
| 22 | 529 KB @ 47.89 dB | **495 KB @ 48.10 dB** |
| 27 | 342 KB @ 44.11 dB | **313 KB @ 44.48 dB** |
| 32 | 221 KB @ 40.92 dB | **198 KB @ 41.21 dB** |
| 37 | 146 KB @ 37.72 dB | **127 KB @ 38.11 dB** |

That is **8-19% off the bitrate at matched quality**, more at low rates. At 4K
(the same photographs tiled 3x3, so the detail density is unchanged rather than
smoothed by upscaling) it is **12-22%**. The gain did *not* grow much with
resolution, which is worth recording because the prediction was that it would:
the earlier reasoning assumed higher resolution brings larger flat regions, and
a tiled clip deliberately has none of that. Content that genuinely gets flatter
as it gets bigger would gain more.

Stopping the quadtree at 16x16 instead of 8x8 costs 5-6% of the bitrate and
saves 35% of the time — so the smallest coding unit earns its keep.

### H.264: CABAC

The compressed H.264 path now codes Main profile with CABAC instead of Baseline
with CAVLC. The arithmetic engine did not have to be written — H.264 §9.3.4.3
and H.265 §9.3.4.3 specify the same coder, down to `rangeTabLPS` and the
transition tables — so `src/codec/h264/cabac.rs` reuses the HEVC engine and adds
what actually differs: the context *initialisation*, and the binarisation and
context derivation of every I-slice syntax element.

Photographs, 1280x720, same encoder decisions either side:

| QP | CAVLC | CABAC | saved |
|---|---|---|---|
| 22 | 619 KB @ 47.63 dB | **587 KB** | 5.2% |
| 27 | 419 KB @ 44.01 dB | **383 KB** | 8.6% |
| 32 | 288 KB @ 40.70 dB | **253 KB** | 12.2% |
| 37 | 206 KB @ 37.63 dB | **175 KB** | 15.3% |

More at low rates, which is the right shape: at high QP most bins are "no
coefficient here", and CAVLC cannot spend less than a bit to say it.

Against x264 all-intra at matched PSNR the native H.264 encoder is now
**1.07-1.13x**, from 1.10-1.40x with CAVLC and about 5x before `I_16x16`. (The
final figures are in the summary below and include deblocking, which moves the
distortion without moving the rate.)

#### The initialisation tables, and how to get 440 constants right

CABAC needs an `(m, n)` pair per context — about 220 of them for I-slice syntax,
none derivable, none checkable against each other. This is the transcription
problem that produced most of the bugs recorded above, at four times the scale,
and a wrong pair does not fail loudly: the arithmetic coder is perfectly happy
with a wrong probability and simply decodes to something else.

So they were not transcribed. OpenH264's upstream source is vendored in the
local cargo registry (`openh264-sys2`, BSD-2-Clause), and its table is indexed by
the specification's own `ctxIdx` with the I-slice column first — the same
numbering §9.3.1.1 uses. `src/codec/h264/cabac_tables.rs` is generated from it,
and every value in it is a constant of the ITU-T H.264 specification rather than
anything OpenH264 invented.

What is checked in this repository is the *use* of them: compile-time assertions
that no derivation can index into `ctxIdx` 11..=59, which are P/B-only and
undefined for an I slice, and that the per-category context blocks tile end to
end without overlapping.

#### The bug, and what it says about fuzzing

One, and it was found by the opposite of the usual blind spot. The chroma DC
`coded_block_flag` context asked whether the neighbouring macroblock had a
chroma block at all (`CodedBlockPatternChroma != 0`) rather than whether that
block carried any coefficients. The two differ only when chroma AC is coded over
a zero DC.

Noise never does that: it drives every coded block pattern to 15, so every
neighbour has coefficients everywhere and the distinction never arises. The
existing randomised sweep fills all three planes with noise and passed
cleanly — while a plain gradient failed at the third macroblock.

The recorded lesson from the CAVLC work was "uniform test patterns are a blind
spot". The truer statement is that *any* fixed texture is: noise hides the
partially-coded macroblock exactly as flat fields hide the fully-coded one.
`fuzz_smooth_content_is_conformant` now randomises the smoothness — slopes and
coarse ripples at random amplitudes, so some planes go flat while others do
not — alongside the noise sweep.

### Per-block QP, and an adaptive quantizer that did not pay

Both encoders can now vary the quantizer within a picture: H.264 per macroblock
via `mb_qp_delta`, HEVC per coding tree block via `cu_qp_delta` with
`diff_cu_qp_delta_depth = 0`. Both are conformant with the quantizer swinging
hard and often (`varying_quantizer_is_conformant`,
`per_ctb_quantizer_is_conformant`).

The element is more delicate than it looks in both standards, and for the same
reason: it is *prediction-coded and conditional*. A block that codes no residual
sends no delta, and then the decoder's QP is the predictor rather than whatever
the encoder used — so the predictor must not advance past it. In HEVC the delta
does not even live in the coding unit's header; it rides in the first transform
unit anywhere below that has something to code, which at 4x4 means testing the
*parent's* chroma flags, for all four children rather than only the one carrying
the shared chroma residual. Getting that wrong desynchronises the quantizer
without desynchronising the bitstream — the stream still parses, it just
dequantizes with the wrong step from that point on.

On top of that sits adaptive quantization: lower QP where the picture is flat
and quantization error is visible, higher where texture masks it, centred on the
picture's own mean log-variance so the average QP does not move.

**It measured worse.** Ten reference photographs at 1280x720, VMAF at matched
size (the native encoders have no psychovisual tooling, so VMAF is the only
perceptual metric here):

| | H.264 | HEVC |
|---|---|---|
| AQ strength 0.5 | −0.14 to −0.33 VMAF | — |
| AQ strength 1.0 | −0.2 to −0.5 VMAF | −0.20 to −0.33 VMAF |
| Inverted (−0.5, −1.0) | −0.03 to +0.07 VMAF | — |

The inverted control matters: running the same machinery *backwards* — more bits
on detail, fewer on flat areas — lands within noise of a uniform quantizer,
while the forward direction is consistently worse. So this is not a broken
implementation pushing quality around at random. The masking model simply does
not buy anything on this content and this metric, and it costs something.

That is a believable result rather than a surprising one. VMAF's detail-loss
term penalises exactly what adaptive quantization does — coarsen texture — and
x264's own AQ is well known to lower VMAF against `--aq-mode 0`. It is entirely
possible that AQ looks better to a human here and the metric cannot see it. But
a metric that says "worse" is not evidence for shipping it on.

So **adaptive quantization is off by default in both encoders** and reachable
via `CompressedEncoder::aq`, with the measurement recorded here rather than the
usual assertion that AQ is a win. The per-block quantizer underneath it stays,
because it is also the thing a rate controller needs, and neither encoder has
one yet.

### Deblocking

Both encoders disabled the in-loop filter — H.264 with
`disable_deblocking_filter_idc = 1`, HEVC with
`pps_deblocking_filter_disabled_flag = 1`. Neither was a considered choice; the
comment in `pps.rs` said as much. Both now implement it and leave it on.

The filter costs nothing in bits: it changes what the decoder shows, not what
the encoder sends. And it is not actually in-loop here — intra prediction reads
samples from *before* the filter (H.264 §8.3, H.265 §8.4.4.2.2), and neither
encoder has inter prediction, so the whole picture can be filtered once at the
end rather than interleaved with reconstruction. That is what makes this
tractable at all: the usual hazard with an in-loop filter is that any deviation
compounds through the reference chain, and there is no reference chain yet.

Ten reference photographs at 1280x720. Identical bitstreams either side — only
the flag differs:

| | ΔPSNR | ΔSSIM | ΔVMAF |
|---|---|---|---|
| H.264, QP 36 → 48 | +0.26 to +0.28 dB | +0.0027 to +0.0075 | −0.11 to −0.95 |
| HEVC, QP 30 → 44 | +0.17 to +0.18 dB | +0.0008 to +0.0024 | −0.01 to −0.05 |

Two metrics up, one down, and the SSIM gain grows as the rate falls — which is
where blocking artefacts live. VMAF disagreeing is expected: its detail-loss
term treats a low-pass filter as damage, and deblocking is a low-pass filter.
The same disagreement showed up on adaptive quantization above, but with the
opposite conclusion, because there PSNR and SSIM did not go the other way.

Worth noting how much gentler HEVC's filter is: an order of magnitude less VMAF
cost for a comparable PSNR gain. It works on an 8×8 grid rather than 4×4, so
half as many edges are candidates; it decides per four lines rather than per
line; and it judges each segment from the second differences either side, which
distinguishes a coding step from a real one better than H.264's three
first-difference thresholds.

#### Two details that only an external decoder catches

**The 8×8 grid is not every eighth sample.** HEVC filters transform *boundaries*
that fall on the 8×8 grid, not every 8×8 line. Inside a 32×32 transform there is
nothing to smooth at sample 8, 16 or 24, and filtering there would blur real
content. The decided transform tree therefore has to be walked afterwards to
record where leaves actually end.

**`QpY` is not constant across a quantization group.** `CuQpDeltaVal` is zero
until the delta is parsed, so coding units decoded *before* the first one with
anything to code sit at the predictor while the rest sit at the target. The
deblocking thresholds are derived per coding unit, so the difference is visible —
it showed up as ten samples out of forty thousand, at one QP, in one adaptive
quantization setting: a `tC` of 5 where the decoder used 6, because the chroma
QP was one step low. Tracking `QpY` per coding unit rather than per coding tree
block fixed it.

#### The threshold tables

H.264's α, β and t'C0 came out of OpenH264's vendored source, the same way as
the CABAC initialisation values. HEVC's β' and t'C were recalled and then
checked byte-for-byte against three independent decoders installed on this
machine — libde265, x265 and ffmpeg's libavcodec all carry the pair adjacent in
their data segments, and all three agree with each other and with the
recollection. Two of the three had to be searched by byte pattern rather than by
symbol; that is a perfectly good check, and a far better one than reading the
numbers twice.

### Benchmarked properly: BD-rate on three metrics, two contents

Everything above compared encoders by interpolating between two rate points.
That answers a different question at every pair you pick. `scripts/codec_bench.py`
now computes **BD-rate** instead — fit `log(rate)` as a cubic in the quality
metric, integrate both curves over the quality range they share, divide — on six
rate points per curve, on PSNR, SSIM and VMAF, against ffmpeg at both its
default tuning and `-tune psnr`.

Negative means the native encoder needs fewer bits for the same quality.

| | | PSNR | SSIM | VMAF |
|---|---|---|---|---|
| **photographs** | native H.264 vs x264 | +9.2% | +7.6% | **+23.3%** |
| | native HEVC vs x265 | −4.4% | −0.4% | −1.5% |
| **rendered (`frog`)** | native H.264 vs x264 | −2.4% | −0.9% | +6.1% |
| | native HEVC vs x265 | **−32.9%** | **−29.5%** | **−45.7%** |

(ffmpeg anchors are `-tune psnr` all-intra; see below for why.)

Three things fall out of this that the old two-point tables could not show.

**Content moves the answer by 30 to 45 points.** Same encoder, same anchors,
same metrics: native HEVC is at parity with x265 on photographs and needs half
the bits on a rendered clip. All three metrics agree on both sides, so this is
not metric noise. It is not the encoder being twice as good either — it is x265
declining to exploit very easy content, its CRF scale bottoming out at 78 KB
where this encoder reaches the same quality in 37 KB. Both numbers are real and
neither is *the* answer. Since this project renders 3D scenes, the rendered row
is the one its users will feel; the photographic row is the one that says
whether the codec is actually good.

**Which ffmpeg tuning you compare against decides the sign.** Against x264's
*defaults*, native H.264 wins on PSNR by 10%. Against `-tune psnr` it loses by
9%. The 19-point swing is x264's psychovisual rate-distortion and adaptive
quantization trading PSNR away deliberately, and scoring against it measures
that trade rather than codec efficiency.

The surprise is that `-tune psnr` also beats x264's defaults on **VMAF**, by
about 1.5 points at equal size — because it turns adaptive quantization off.
That is the same effect measured on this project's own adaptive quantization a
few sections up, reproduced independently in a mature encoder. It is the
strongest evidence available that shipping AQ off was right, and it arrived from
an anchor chosen for an unrelated reason.

**H.264 is much further behind on VMAF than on PSNR.** +9.2% on PSNR, +7.6% on
SSIM, +23.3% on VMAF. A PSNR-only comparison — which is what this document had
until now — hid that entirely. HEVC shows no such spread, and the difference
between the two is where the remaining work is: the HEVC path does full
rate-distortion over a coding quadtree and a transform tree, while the H.264
path still picks its 4×4 intra modes by SAD and has no trellis quantization.

### Timing on a shared machine

Wall clock here is unmeasurable: at a load average of 50 on 14 cores, ffmpeg's
own numbers moved 5× between runs minutes apart. The harness now reports
**minimum (user + sys) CPU seconds over repeated runs** (`--repeats N`) and warns
when the load average would make anything else meaningless. That measures total
work rather than elapsed time, which does not flatter a threaded encoder for
having cores available — the right comparison for a single-threaded one.

| 720p, 10 frames, matched quality | CPU s | per frame |
|---|---|---|
| native H.264 | 0.43 | 0.043 |
| x264 all-intra | 0.17 | 0.017 |
| native HEVC | 8.51 | 0.851 |
| x265 all-intra | 2.38 | 0.238 |

**2.5× the CPU of x264, 3.6× the CPU of x265** — and the ratio holds at 8K, so
it is a property of the encoders rather than of the machine's mood. The HEVC
figure is now 2.1× and 1.6× after the optimisation work below.

### Making it faster without changing a byte

Profiling one 8K HEVC frame with `sample(1)` said the encoder was spending most
of its time on things that are not encoding:

| before | after | where |
|---|---|---|
| 22.9% | 16.4% | `transform::forward` |
| 16.7% | 8.4% | allocator (`malloc`/`free`) |
| 15.3% | 16.4% | `intra::angular` |
| 7.1% | — | `compress::decode_order` |
| 4.9% | 10.5% | `memcpy` / `memset` |

Roughly a quarter of the encode was memory management. Nothing was leaking or
thrashing; it was simply that every helper returned a fresh `Vec`, and
`code_leaf` — which runs once per candidate per level of two nested trees —
made nine of them per call.

The whole exercise was run against a bit-exactness oracle: eighteen
`(clip × encoder × QP)` combinations hashed before and after every change, on
top of the conformance suites. An optimisation that changes one byte of output
has changed the bitstream, not its speed, and none of these did.

**Four changes.**

*`decode_order` divided by the coding-tree size.* That size is always a power of
two, but nothing in the type says so, and the compiler was emitting real integer
divisions — two divisions and two remainders per call, on the hottest path in
the encoder. A shift and a mask. The Morton interleave under it was a
sixteen-round loop; the doubling-and-masking ladder does it in five steps. And
because `available()` compares the neighbour's position against the *current
block's*, which is the same for all `4n+1` reference samples, half the work was
recomputing a constant.

*The transform read its intermediate column-wise.* Stage two consumed columns of
what stage one produced, a stride-`n` gather that at 32×32 touches thirty-two
cache lines per dot product. Writing stage one transposed makes both passes walk
forwards. The intermediate also narrowed from `i64` to `i32` — the residual is a
difference of 8-bit samples, so a row sum cannot exceed 90 · 255 · 32 before the
shift, comfortably inside 32 bits.

*The mode search rebuilt its reference samples thirty-five times.* `extract`
depends on the mode only through `filter_flag`, which is a boolean — so across
all thirty-five modes there are exactly *two* reference sets. That turned a
hundred and five allocations and thirty-five smoothing passes into three and
one.

*The per-block buffers became a reusable `Scratch`.* The first attempt used
fixed-size stack arrays and came out **slower**: zeroing four kilobytes to use
sixteen of them on a 4×4 block costs more than the allocation it saved. Sizing
the buffers to the largest block once and slicing them per call is what actually
worked.

**Result: 1.53× at 720p, 1.78× at 8K** — 21.5 s to 12.1 s of CPU per 8K frame —
with byte-identical output.

Two things did not work, and are worth recording as much as the four that did.

**The even/odd butterfly measured as noise.** The DCT matrix rows are
alternately symmetric and antisymmetric about their centre, so half the
multiplies are redundant, and the regrouping is an exact identity over the
integers — the bit-exactness check confirms it. It should have been a clear 2×
on the transform. Interleaved A/B runs put it at about 4% faster with a spread
wider than the effect. LLVM vectorises the full-length contiguous dot product
well, and splitting it into two half-length passes with a staging array gives
back most of what the halved multiply count wins. It is kept — strictly fewer
operations, provably identical output — but it is not the win the operation
count promises.

**Reusing the prediction buffer did nothing measurable.** The profile attributed
15.3% to `intra::predict`, which read as an allocation problem; it was the
arithmetic. Removing 105 allocations per block from the mode search moved
nothing until the *other* buffers were fixed too. Profiles attribute cost to
where it is spent, not to what causes it.

The remaining profile has no dominant term — the largest is 16% — which is the
shape that says the cheap structural wins are gone. What is left is algorithmic:
the mode search still evaluates all thirty-five modes at full block size, and
that is a decision-quality change rather than an optimisation, so it belongs
with the rate-distortion work rather than here.

### Frame parallelism

All-intra means every frame is an IDR that carries nothing forward.
`compressed_slice_rbsp` takes the config and the pixels and no `&mut self`;
the only per-encoder state was a one-shot "have the parameter sets been emitted"
flag. So frames can be encoded on separate threads, and the result is not merely
equivalent to encoding them in order — it is the same bytes.

Both export paths now pull frames in batches, encode the batch with
`rayon` (behind the existing `parallel` feature), and emit the results in order.
`CompressedEncoder::encode_slice_au` is the `&self` entry point that makes it
possible; without the feature the same code runs the batch serially, so the
feature changes the schedule and nothing else.

Thirty frames of 1280x720 through the HEVC export path:

| threads | wall | CPU | output |
|---|---|---|---|
| 1 | 40.6 s | 18.4 s | `39d6de5abffb` |
| 2 | 15.8 s | 19.1 s | `39d6de5abffb` |
| 4 | 6.8 s | 19.4 s | `39d6de5abffb` |
| 6 | 5.6 s | 19.6 s | `39d6de5abffb` |
| 10 | **3.7 s** | 20.2 s | `39d6de5abffb` |
| 14 | 5.3 s | 20.1 s | `39d6de5abffb` |

Two things to read there, and only one of them is the speedup.

**CPU time is flat.** 18.4 s to 20.2 s across a fourteen-fold change in thread
count: the parallelism costs about 9% in total work and no more. That number is
load-independent, which is what makes it the trustworthy one on this machine.

**Wall clock is not a clean measurement here.** The single-threaded run reports
18.4 s of CPU in 40.6 s of wall — it was getting less than half a core. So the
ratios in that column are a floor, not the speedup; the honest reading is
CPU ÷ wall, the parallelism actually achieved, which peaks at 5.5× on ten
threads and *falls* at fourteen, where the four efficiency cores and the
contention start costing more than they add.

**The hash is identical at every thread count.** That is the property that
matters, and `tests/frame_parallel.rs` pins it by varying the thread count at run
time rather than trusting the argument above.

#### Why it batches instead of encoding everything at once

Memory. An 8K frame is 50 MB of input, and the encoder allocates a
reconstruction and padded plane copies alongside it — about 4.5 bytes per pixel
in flight, or 149 MB per frame. Encoding a whole clip concurrently would undo
the streaming work that got export memory down in the first place, so the batch
is the smaller of the thread count and what fits in a 256 MB working set:

| | in flight per frame | batch |
|---|---|---|
| 720p | 3 MB | thread count |
| 4K | 31 MB | 8 |
| 8K | 124 MB | 2 |

So the parallelism is full-strength up to about 1080p, tapers through 4K, and is
worth roughly 2× at 8K — where a frame is expensive enough that two at a time
still helps. Callers who would rather spend the memory can raise the budget with
[`set_frame_memory_budget`], or drive `encode_slice_au` from their own pool.
Better still, see *Tiles* below: they parallelise *within* a picture, which is
what actually fixes the 8K case and the single-frame case with it.

### Tiles

Frame parallelism has two blind spots: a single-frame encode has nothing to
parallelise, and at 8K the batch is memory-bound to two frames however many
cores are free. Tiles fix both, by splitting one picture into regions that are
independent of each other.

They are independent in the strong sense — intra prediction stops at a tile edge
and the entropy coder restarts — which means a tile is nothing more than a
smaller picture. So each one is encoded by the same `encode_ctbs` that codes a
whole frame, on its own plane region, and the substreams are concatenated.
Nothing in the coding path knows tiles exist. Two things stay picture-level: the
reconstruction is stitched back together, and the deblocking filter runs over the
result, so tile edges are smoothed like any other block edge — which is what
`loop_filter_across_tiles_enabled_flag` promises, and it costs nothing here
because the filter was already a separate pass.

#### What they cost

| | 2x2 | 4x4 |
|---|---|---|
| 8K | +0.05% | +0.43% |
| 4K | +0.22% | +0.75% |
| 1080p | +0.43% | +0.85% |
| 360p | +1.12% | +5.06% |

The cost tracks internal boundary per unit area, so what matters is tile *size*,
not count — tiles of four megapixels or more are essentially free.

#### What they buy

One 8K frame, fourteen threads:

| tiles | wall | speedup | bitrate |
|---|---|---|---|
| 1x1 | 23.9 s | — | — |
| 2x1 | 13.3 s | 1.80x | +0.04% |
| 2x2 | 8.8 s | 2.71x | +0.05% |
| 4x2 | 5.1 s | 4.73x | +0.08% |
| 4x4 | 3.6 s | **6.73x** | +0.43% |

And they compose with frame parallelism. A four-frame 8K export, which the
memory budget caps at two frames at a time, goes from 54.9 s to **17.9 s** with
`4x2` tiles — sixteen concurrent units instead of two — for 0.08% more bits and
2.7% more CPU. Against single-threaded and untiled, 76.5 s to 17.9 s.

#### The default

`with_auto_tiles` picks the grid from the picture size: halve the longer side
until a tile is at most four megapixels. That is `1x1` up to 1440p, `2x1` at 4K,
`4x2` at 8K — so it engages exactly where it is free and changes nothing below.

Deliberately *not* derived from the core count, which would be the obvious way to
size it: the same input has to encode to the same bytes on every machine.

#### An estimate that was wrong by a factor of ten

Before implementing this, the cost was estimated by encoding sub-rectangles as
separate files and summing — which said 4x4 at 8K would cost **+4.1%**, enough
to make tiles look like a poor trade against simply raising the frame memory
budget. That was the recommendation given.

The real figure is **+0.43%**. The difference is 907 bytes per extra file of MP4
container and repeated parameter sets — 13.6 KB across sixteen files, against a
316 KB frame. The estimate was measuring its own scaffolding.

The lesson is narrower than "estimates are unreliable": the estimate was
measuring the right *quantity* on the wrong *artefact*. Simulating a bitstream
feature with N independent bitstreams charges you N times for everything a real
bitstream would carry once, and at these ratios that overhead swamped the effect
under test. Where the simulation can't share what the real thing shares, measure
the payload, not the file.

### Rate control

The original question that started this document was why exports balloon. Until
now the only answer available was "pick a smaller quantizer and see what
happens" — `.bitrate("5M")` was parsed, warned about, and thrown away.

Both native encoders now take a bitrate. `VideoQuality::Bitrate` reaches the
encoder, and `Quality::Bitrate` is available directly on both export paths.

#### The rate law, and how wrong the obvious one is

Intra pictures follow a law simple enough to steer by:

```text
log2(bytes) ≈ k − slope · qp
```

`k` describes the *content* and `slope` how fast the encoder gives it up. The
obvious value for `slope` is **1/6**: the quantizer step doubles every six QP,
so the rate should halve. Measured on this encoder it is **0.116** — nearer
1/8.6. Coefficients that quantize to zero stop costing anything at all, and the
residual's entropy does not fall as fast as its magnitude.

That is not a rounding error. Assuming 1/6 makes the controller under-correct,
and it undershot a 20 Mbit/s target by **51%** before the slope was measured
rather than assumed. The slope is now learned from pairs of frames coded at
different quantizers, and 1/6 appears nowhere in the code.

#### Accuracy

Asking for a bitrate and weighing the file. Noise-like content, which is the
hard case — its rate curve is steep enough that one quantizer step changes the
size by a third.

| frames | 0.3 Mb/s | 0.8 Mb/s | 2.0 Mb/s |
|---|---|---|---|
| 96 | +2.2% | −0.2% | +1.0% |
| 48 | +0.9% | +0.2% | +8.5% |
| 30 | +2.5% | +2.6% | +7.8% |
| 24 | +18.7% | +8.1% | +0.2% |

Longer is tighter, and that is structural rather than incidental: the controller
learns once per group of frames, so the number of groups is the number of
chances it gets to correct itself. Eight rounds is the floor, which a
ninety-six-frame clip spends comfortably and a twenty-four-frame clip does not.

The group size is derived from the clip length and the memory budget and from
nothing else. It was briefly derived from how many frames fit on the available
cores, which made the same clip encode to a different size on a different
machine — one thread missed a target by 19% where fourteen missed it by 1%.
`thread_count_does_not_change_a_rate_controlled_encode` pins that shut.

#### Four things that had to be measured rather than reasoned

**The slope is not 1/6.** The quantizer step doubles every six QP, so the rate
should halve. It does not: measured, the slope is 0.116, nearer 1/8.6.
Assuming 1/6 undershot a 20 Mbit/s target by 51%.

**The slope is not one number either.** On the same content it is 0.075 between
QP 20 and 30 and **0.32** between 40 and 51 — once most coefficients quantize to
zero, each further step costs far more of the rate. A clamp that allowed only
0.06–0.22 pinned the estimate below the truth and left the controller unable to
reach high quantizers at all.

**Weighting the newest measurement at 0.4 was worth a 30% miss; 0.7 leaves 8%.**
An estimate that averages across regions of the curve describes none of them,
and an underestimated slope makes every correction overshoot, which in a
feedback loop is an oscillation — the trace showed 48 → 33 → 37 → 51 → 43.

**Deriving the step limit from the slope measured worse than a constant.**
Making one correction always change the rate by the same ratio is the elegant
version: at a slope of 0.44 that is a single quantizer step. It is also useless,
because the controller then cannot cross the distance to its target before the
clip ends. A flat limit of eight steps beat it everywhere.

None of these were guessable, and three of them were things I had already
assumed and written down before measuring.

#### Both ways a target can be impossible

A bitrate can be out of reach in either direction, and both look identical from
outside — a file that is not the size you asked for.

*Too complex*: past QP 51 the encoder has nothing coarser to offer. *Too
simple*: past QP 0 it has nothing finer, and a flat rendered clip is as large as
it will ever get. On the `frog` render, 20 Mbit/s produces a 410 KB file however
hard you ask, because that is the whole of what the content is:

```text
threers: this content cannot fill that bitrate — even losslessly quantized
it is about 20.8x smaller, so the file is smaller than asked for.
That is the content, not a failure.
```

Both are reported from what was *actually coded* rather than from the model.
Extrapolating the rate law out to QP 0 or 51 is precisely where it stops
holding: doing that reported the rendered clip as "2.9x too simple" for a target
it in fact came within 3% of.

#### A bug worth recording

Threading rate control through the export path broke fixed-quantizer encoding,
silently, by dropping every clip's first frame:

```rust
if let (Some(rc), Some(first)) = (rc.as_mut(), frames.next())
```

The tuple is built before the pattern is tried, so `frames.next()` runs whether
or not there is a controller to feed it to. With no controller the pattern fails
and the frame is discarded.

The test written to prevent exactly this — "a fixed quantizer is unaffected" —
passed. It compared `write_compressed_mp4` against `write_mp4_quality(Qp)`, and
`write_compressed_mp4` had just been changed to delegate to `write_mp4_quality`.
It was comparing the path with itself. What caught it was the bit-exactness
oracle from the optimisation work, which hashes real encoder output against
hashes taken before any of this began.

The replacement counts entries in the MP4 sample-size box — a reference the
encoder cannot supply, because it is the file's own record of how many frames it
holds.

### Where both encoders now sit

Ten reference photographs, 1280x720, all-intra everywhere, ffmpeg tuned for PSNR
so the comparison is not decided by psychovisual settings the native encoders do
not have. Native columns are the encoders as they now ship — `I_16x16` and
CABAC on the H.264 side, 64x64 coding tree blocks and a quadtree on the HEVC
side, deblocking on both, adaptive quantization off.

| Encoder | ~48 dB | ~44 dB | ~41 dB | ~38 dB |
|---|---|---|---|---|
| native H.264 | 573 KB | 374 KB | 247 KB | 171 KB |
| x264 all-intra | 538 KB | 350 KB | 232 KB | 151 KB |
| native HEVC | 483 KB | 306 KB | 193 KB | 124 KB |
| x265 all-intra | 480 KB | 314 KB | 208 KB | 131 KB |

(ffmpeg rows interpolated onto the native encoders' own PSNR points.)

**Native H.264 is 1.07-1.13x of x264 all-intra**, from about 5x before this
work. **Native HEVC is 0.93-1.01x of x265 all-intra** — at parity or slightly
ahead on this clip, from 1.15x. Neither has inter prediction, so neither is
comparable to what ffmpeg ships by default; this is the like-for-like all-intra
comparison, and inter is the whole of what is still missing.

Two caveats on reading that. The native HEVC encoder being *ahead* of x265 at
the lower rates is a narrow result, not a general one: x265 at `--tune psnr` is
still making decisions this encoder does not make at all — and on the `frog`
clip, which is flat rendered content, the same comparison flatters the native
encoder by 1.4-2.9x, which nobody should believe. And the HEVC encoder pays for
its position in time — about 3× x265's CPU for the same picture, and
single-threaded on top of that. See *Speed, measured after the quadtree*.

### The bug that got through

One, and it is the same shape as the ones in the CAVLC work above. The 4x4
Hadamard for `I_16x16`'s luma DC was written with its Walsh rows in DCT order —
`(sum, alternating, low pair, high pair)` instead of the spec's
`(sum, low pair, alternating, high pair)`.

A permuted Walsh matrix is still symmetric and still orthogonal, so `H·H = 4I`
held, the round-trip test passed, and "a flat field stays flat" passed too —
position 0 is a pure sum and is the one entry a row permutation cannot move.
Every other block's DC was shifted by a constant. Only ffmpeg saw it.

The test that now pins it checks the transform of all sixteen impulses against
`o[i][j] = H[i][r]·H[c][j]`, which admits exactly one matrix. The same lesson
produced the approach taken for the 32-point DCT: rather than transcribe 1024
constants, it is *generated* from `DCT16` plus the sixteen numbers of its own
first odd row, and the generator is proved by feeding it `DCT4` and checking it
emits `DCT8`, then feeding it `DCT8` and checking it emits `DCT16` — two tables
an external decoder has already agreed with.

## What would move the native encoder next

Five of the six items that used to be on this list have been done, and each is
measured above: `I_16x16` for H.264, larger CTBs and a coding quadtree for HEVC,
CABAC for H.264, per-block QP with adaptive quantization, and deblocking for
both. One remains, and it is worth more than all five together.

**Inter prediction.** Still absent, still the largest single factor — the
measurements above put it at roughly 4x, and everything in this document is
all-intra on both sides of every comparison. It is also not a feature so much as
a second encoder. A minimal version — P slices, one reference frame, 16x16
partitions, integer-pel search, skip mode — would already need: reference
picture management and its signalling; motion estimation; the median motion
vector predictor and `mvd` coding; quarter-pel interpolation; a second set of
macroblock types and coded-block-pattern mappings; the P-slice CABAC context
columns, which the current initialisation table deliberately does not carry; the
inter boundary-strength derivation, which compares motion vectors and reference
indices where the intra one is a constant; and — the structural one — deblocking
becoming genuinely in-loop, because the reference is the *filtered* picture,
so any deviation compounds down the whole chain instead of being confined to the
output. HEVC's version is larger again: advanced motion vector prediction, merge
list construction, 8-tap luma interpolation, reference picture sets.

Smaller things still open, in rough order of gain per unit of work:

1. **SAO (HEVC).** Off in the SPS. Band and edge offsets, signalled per CTB, and
   applied after deblocking. Worth a few percent and, unlike adaptive
   quantization, it is rate-distortion decided per block against real
   distortion rather than against a masking model.
2. **8x8 transform (H.264 High profile).** `transform_8x8_flag` is off; x264
   uses it by default, and it is part of why x264 stays ahead on smooth content
   even after `I_16x16`.
3. **Two-pass rate control.** Single-pass lands within a percent or two on a
   clip of any length, and within about 10% on one too short to give it four
   feedback rounds. A first pass over the clip would fix the short case and
   tighten the rest, at double the encode time.
4. **Faster search.** The HEVC coding quadtree costs 4x the encode time for
   8-19% of the bitrate. Neighbour-depth-bounded search and an early exit on a
   coding unit that codes nothing are the standard answers, neither implemented.
5. **Sign data hiding, transform skip, scaling lists.** All off in both. Each is
   worth a small percentage and none is hard.

A caution carried over from earlier work, now with evidence behind it: the
adaptive-quantization result above and the deblocking result above disagreed
with VMAF in the same direction, and only one of them was worth shipping. Judge
anything that trades sharpness for smoothness on more than one metric, and say
which ones disagreed.

`AllowFrameReordering` is deliberately still off: the comment at
`src/videotoolbox.rs` records that the elementary stream carries no timestamps,
so the remux reconstructs them from frame order and B-frames make the last two
frames collide on DTS. The real fix is carrying PTS/DTS through the muxer; until
then B-frames stay off and cost ~15% compression.
