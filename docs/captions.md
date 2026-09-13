# Subtitles and captions

Part of the [threers](../README.md) documentation.

# Subtitles and captions

`threers::captions` is always available — pure Rust, no dependencies, no font
files, and it builds for `wasm32`. It parses and writes SubRip and WebVTT,
rasterizes cue text, and hands the pixels to whichever path you need.

```bash
cargo run --example captions_render                                          # PNGs: captions over a 3D render
cargo run --release --example captions_video --features "video,native-codec" # every delivery mode
```

**On screen** — the renderer blends the active cue over the frame it just drew:

```rust
use threers::captions::{CaptionOverlay, CaptionTrack};

let track = CaptionTrack::parse(&std::fs::read_to_string("dialogue.vtt")?)?;
let mut overlay = CaptionOverlay::new(track).auto_scale(true);

renderer.render(&mut scene, &camera, &view, false);
renderer.draw_caption_overlay(&mut overlay, time_seconds, width, height, &view, format);
```

The text is rasterized only when the active cue changes, so a frame in the
middle of a cue costs one texture upload — and a frame with no cue costs
nothing.

**In an exported video** — pick how the cues ship:

```rust
use threers::{CaptionMode, CaptionTrack, VideoCodec, VideoExporter};

VideoExporter::new("out.mp4")
    .size(1280, 720).frames(300).fps(30).codec(VideoCodec::H264)
    .captions(CaptionTrack::parse_srt(&srt)?)
    .caption_mode(CaptionMode::BurnAndSidecar)   // pixels *and* out.srt
    .export(render)?;
```

| `CaptionMode` | Result |
|------|--------|
| `Burn` (default) | text composited into the frames — works with every codec, including GIF and APNG |
| `Sidecar` | a separate `out.srt` / `out.vtt` beside the video |
| `Embed` | a soft subtitle track in the container: MP4 `mov_text`, WebM WebVTT |
| `BurnAndSidecar` | both of the first two |

Styling is a `CaptionStyle` — type size, fill, outline, shadow, background box,
alignment, frame anchor, margins, and wrap width — and `for_height` rescales a
1080p-authored style for any output size. Text draws with a built-in 8×8 bitmap
face by default (zero assets, Latin-1 accents fold to their base letter); pass a
`.ttf` to `CaptionFont::from_ttf_bytes` / `VideoOptions::caption_font` for real
typography.

The muxers can also write soft subtitles without ffmpeg:
`codec::mp4::mux_hevc_with_captions` (a `tx3g` timed-text track) and
`codec::webm::mux_webm_with_captions` (a `D_WEBVTT/SUBTITLES` track). Both are
round-trip verified against ffmpeg in `tests/captions_mux.rs`.

In the browser, `renderer.setCaptions(overlay)` makes every `render()` draw the
current cue — see `web/examples/captions.html`:

```js
import THREE, { CaptionTrack, CaptionOverlay } from './web/threejs-shim.js';

const overlay = new CaptionOverlay(CaptionTrack.parse(vttText), { fontSize: 34 });
overlay.setAutoScale(true);
renderer.setCaptions(overlay);
renderer.captionTime = video.currentTime;   // then render() as usual
```
