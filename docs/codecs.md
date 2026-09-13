# Native codecs

Part of the [threers](../README.md) documentation.

# Native codecs (GIF, APNG, VP9, H.264, HEVC)

Enable `native-codec` for dependency-free encoders (and a GIF decoder) that also build on `wasm32`:

```rust
// Cargo.toml: threers = { version = "0.0.5", features = ["native-codec"] }
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
