# threers-captions

The caption data model for [threers](https://github.com/eugenehp/threers):
timed-text tracks and cues with their style, SubRip (`.srt`) and WebVTT
(`.vtt`) parsing and writing, and WCAG contrast and legibility checks.

It has no dependencies, so it builds anywhere, wasm included.
`threers-codec` re-exports it as `threers_codec::captions` and muxes the
tracks into MP4 and WebM; `threers::captions` re-exports it and draws them.

```rust
use threers_captions::CaptionTrack;

let track = CaptionTrack::parse_srt(
    "1\n00:00:00,000 --> 00:00:02,500\nHello, world!\n",
).unwrap();
assert_eq!(track.active_at(1.0).len(), 1);
```
