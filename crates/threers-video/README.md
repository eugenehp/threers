# threers-video

Video for [threers](https://github.com/eugenehp/threers):

- **Export** rendered frames to MP4 (H.264), MOV (HEVC), WebM (VP9), GIF and
  APNG, with soft or burned-in captions — through threers' pure-Rust encoders,
  or ffmpeg where it is installed.
- **Sequences**: render stills, a video, or both from one pass.
- **`VideoPlayer`**: video on a mesh, with its sound, decoded in pure Rust on
  every core, in hardware (VideoToolbox, NVDEC, VAAPI), or by WebCodecs in a
  browser.
- **`ScadVideo`** (`openscad` feature): `export_video` on an OpenSCAD animation.

It sits on top of `threers`, which does not depend on it.
