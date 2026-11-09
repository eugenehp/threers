//! Video on a mesh: a file decoded into a `VideoTexture`, frame by frame, and
//! drawn by the GPU renderers.
//!
//! The clips are made here with the crate's own H.264 encoder, so what each
//! frame holds is known: flat colours, one per frame, and a frame split red
//! over blue to show which way up the picture lands.

use std::sync::Arc;
use threers::cameras::{Camera, OrthographicCamera};
use threers::codec::decode::VideoReader;
use threers::codec::h264::{encode_mp4, Yuv420Frame};
use threers::core::{Mesh, Object3D};
use threers::geometries::PlaneGeometry;
use threers::materials::{BasicMaterial, Material};
use threers::math::Color;
use threers::scene::Scene;
use threers::textures::{Texture, VideoTexture};
use threers_video::player::VideoPlayer;

const W: u32 = 64;
const H: u32 = 48;
const FPS: u32 = 10;

/// An RGBA frame of one colour, or two: `top` over `bottom`.
fn frame(top: [u8; 3], bottom: [u8; 3]) -> Yuv420Frame {
    let mut rgba = Vec::with_capacity((W * H * 4) as usize);
    for y in 0..H {
        let c = if y < H / 2 { top } else { bottom };
        for _ in 0..W {
            rgba.extend_from_slice(&[c[0], c[1], c[2], 255]);
        }
    }
    Yuv420Frame::from_rgba(W, H, &rgba)
}

const RED: [u8; 3] = [220, 30, 30];
const GREEN: [u8; 3] = [30, 200, 40];
const BLUE: [u8; 3] = [30, 40, 220];

/// Three frames at 10 fps: red over blue, then all green, then all blue.
fn clip() -> Vec<u8> {
    encode_mp4(
        W,
        H,
        FPS,
        &[frame(RED, BLUE), frame(GREEN, GREEN), frame(BLUE, BLUE)],
    )
}

/// Close enough for a colour through YUV 4:2:0 and back.
fn near(got: &[u8], want: [u8; 3]) -> bool {
    (0..3).all(|c| got[c].abs_diff(want[c]) <= 24)
}

/// The pixel at (x, y), top row first.
fn pixel(rgba: &[u8], width: u32, x: u32, y: u32) -> [u8; 3] {
    let i = ((y * width + x) * 4) as usize;
    [rgba[i], rgba[i + 1], rgba[i + 2]]
}

/// The colour the player's texture holds now, sampled in its top and bottom
/// halves.
fn showing(player: &VideoPlayer) -> ([u8; 3], [u8; 3]) {
    player.video_texture().stream().read(|rgba, _| {
        (
            pixel(rgba, W, W / 2, H / 4),
            pixel(rgba, W, W / 2, 3 * H / 4),
        )
    })
}

#[test]
fn the_reader_gives_every_picture_in_order_and_rewinds() {
    let mut reader = VideoReader::open(clip()).unwrap();
    assert_eq!(reader.size(), (W, H));
    assert_eq!(reader.frame_count(), 3);
    assert!(
        (reader.duration() - 0.3).abs() < 1e-6,
        "{}",
        reader.duration()
    );
    for round in 0..2 {
        let mut seen = Vec::new();
        while let Some((at, f)) = reader.next_frame().unwrap() {
            let rgba = f.to_rgba();
            seen.push((at, pixel(&rgba, W, W / 2, 3 * H / 4)));
        }
        let times: Vec<f64> = seen.iter().map(|s| s.0).collect();
        assert_eq!(times.len(), 3, "round {round}");
        for (got, want) in times.iter().zip([0.0, 0.1, 0.2]) {
            assert!((got - want).abs() < 1e-6, "round {round}: {times:?}");
        }
        assert!(near(&seen[0].1, BLUE) && near(&seen[1].1, GREEN) && near(&seen[2].1, BLUE));
        reader.rewind().unwrap();
    }
}

#[test]
fn the_player_shows_each_picture_when_it_is_due() {
    let mut player = VideoPlayer::open_inline(clip()).unwrap();
    assert_eq!(player.size(), (W, H));
    // The first picture shows at once, the right way up in the frame.
    let (top, bottom) = showing(&player);
    assert!(near(&top, RED) && near(&bottom, BLUE), "{top:?} {bottom:?}");

    player.update(0.05);
    assert!(near(&showing(&player).0, RED), "not due yet");
    player.update(0.05);
    assert!(near(&showing(&player).0, GREEN));
    player.update(0.1);
    assert!(near(&showing(&player).0, BLUE));

    // Past the end it stops on the last picture.
    player.update(0.5);
    assert!(player.is_finished());
    assert!(near(&showing(&player).0, BLUE));
    assert!((player.time() - 0.3).abs() < 1e-6, "{}", player.time());

    // Paused, time stands still.
    player.seek(0.0);
    player.pause();
    player.update(1.0);
    assert_eq!(player.time(), 0.0);
}

#[test]
fn a_late_update_skips_to_the_picture_due_and_looping_goes_round() {
    let mut player = VideoPlayer::open_inline(clip()).unwrap();
    player.set_looping(true);
    let before = player.video_texture().stream().frames();
    // From 0 to 0.25 in one step: the last picture, written once, not three times.
    player.update(0.25);
    assert!(near(&showing(&player).0, BLUE));
    assert_eq!(player.video_texture().stream().frames(), before + 1);
    // 0.25 + 0.1 = 0.35, which is 0.05 into the second time round: red over blue.
    player.update(0.1);
    assert!(!player.is_finished());
    assert!((player.time() - 0.05).abs() < 1e-6, "{}", player.time());
    let (top, bottom) = showing(&player);
    assert!(near(&top, RED) && near(&bottom, BLUE), "{top:?} {bottom:?}");
}

#[test]
fn the_threaded_player_keeps_up() {
    let mut player = VideoPlayer::open(clip()).unwrap();
    player.update(0.1);
    // The thread decodes ahead; give it a moment to have the picture.
    let start = std::time::Instant::now();
    while !near(&showing(&player).0, GREEN) {
        assert!(
            start.elapsed().as_secs() < 10,
            "the green picture never came"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
        player.update(0.0);
    }
    player.set_looping(true);
    player.update(0.25);
    let start = std::time::Instant::now();
    while !near(&showing(&player).1, BLUE) || !near(&showing(&player).0, RED) {
        assert!(
            start.elapsed().as_secs() < 10,
            "it never looped back to the start"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
        player.update(0.0);
    }
}

#[test]
fn an_unsupported_file_says_so() {
    assert!(VideoPlayer::open(b"not a video at all".to_vec()).is_err());
    assert!(VideoPlayer::open(Vec::new()).is_err());
}

/// A plane filling an orthographic view, showing `texture` unlit.
fn screen(texture: Arc<Texture>) -> (Scene, OrthographicCamera) {
    let mut scene = Scene::new();
    let mut material = BasicMaterial::new(Color::WHITE);
    material.map = Some(texture);
    scene.add(Object3D::mesh(Mesh::new(
        PlaneGeometry::new(2.0, 2.0),
        Material::Basic(material),
    )));
    let mut camera = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    camera.position.z = 2.0;
    (scene, camera)
}

/// Render the player's texture three times, moving the video on between, and
/// check each frame shows the picture due.
fn follows_the_video(name: &str, render: &mut dyn FnMut(&mut Scene, &dyn Camera) -> Vec<u8>) {
    const N: u32 = 32;
    let mut player = VideoPlayer::open_inline(clip()).unwrap();
    let (mut scene, camera) = screen(player.texture());
    let out = render(&mut scene, &camera);
    let (top, bottom) = (
        pixel(&out, N, N / 2, N / 4),
        pixel(&out, N, N / 2, 3 * N / 4),
    );
    assert!(
        near(&top, RED) && near(&bottom, BLUE),
        "{name}: first frame {top:?} over {bottom:?}"
    );
    player.update(0.1);
    let out = render(&mut scene, &camera);
    assert!(
        near(&pixel(&out, N, N / 2, N / 2), GREEN),
        "{name}: second frame"
    );
    player.update(0.1);
    let out = render(&mut scene, &camera);
    assert!(
        near(&pixel(&out, N, N / 2, N / 2), BLUE),
        "{name}: third frame"
    );
}

#[test]
fn the_renderer_draws_each_new_frame() {
    let Ok(mut r) = threers::HeadlessRenderer::builder().size(32, 32).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    follows_the_video("wgpu", &mut |s, c| r.render_to_rgba(s, c));
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn the_metal_renderer_draws_each_new_frame() {
    let mut r = threers::metal::MetalHeadlessRenderer::builder()
        .size(32, 32)
        .build()
        .expect("a Metal device");
    follows_the_video("metal", &mut |s, c| r.render_to_rgba(s, c).unwrap());
}

#[test]
fn a_video_texture_is_one_texture_whatever_is_written() {
    let video = VideoTexture::new(4, 2);
    let texture = video.texture();
    assert!(!video.write_rgba(&[0; 3]), "wrong length is refused");
    assert!(video.write_rgba(&[200; 4 * 2 * 4]));
    // The same texture, so materials holding it see the new frame.
    assert!(Arc::ptr_eq(&texture, &video.texture()));
    assert_eq!(video.stream().frames(), 1);
    // A chain, rebuilt on the GPU after each frame, unless asked not to.
    assert!(texture.mipmap_filter.is_some());
    assert!(video
        .clone()
        .without_mipmaps()
        .texture()
        .mipmap_filter
        .is_none());
    // No pixels of its own: a reader gets the stream's frame.
    assert!(texture.data.is_empty());
    assert_eq!(&texture.pixels()[..4], &[200; 4]);
}

/// The player hands the texture the decoder's planes, and the CPU conversion
/// that readers get agrees with the codec's own.
#[test]
fn planes_convert_as_the_codec_converts() {
    let mut reader = VideoReader::open(clip()).unwrap();
    let (_, first) = reader.next_frame().unwrap().unwrap();
    let want = first.to_rgba();
    let player = VideoPlayer::open_inline(clip()).unwrap();
    let got = player.texture().pixels().into_owned();
    assert_eq!(got.len(), want.len());
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    assert!(worst <= 1, "off by {worst}");
}

/// Each output pixel is one texel, sampled at its centre, so the frame on
/// screen is the conversion's output: compared with the codec's own.
fn shows_the_codec_colours(name: &str, render: &mut dyn FnMut(&mut Scene, &dyn Camera) -> Vec<u8>) {
    let mut reader = VideoReader::open(clip()).unwrap();
    let (_, first) = reader.next_frame().unwrap().unwrap();
    let want = first.to_rgba();
    let player = VideoPlayer::open_inline(clip()).unwrap();
    let mut texture = (*player.texture()).clone();
    texture.mag_filter = threers::textures::TextureFilter::Nearest;
    texture.min_filter = threers::textures::TextureFilter::Nearest;
    texture.mipmap_filter = None;
    let (mut scene, camera) = screen(Arc::new(texture));
    let out = render(&mut scene, &camera);
    assert_eq!(out.len(), want.len(), "{name}");
    let worst = out
        .iter()
        .zip(&want)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    assert!(worst <= 3, "{name}: off by {worst}");
}

#[test]
fn the_renderer_shows_the_codec_colours() {
    let Ok(mut r) = threers::HeadlessRenderer::builder().size(W, H).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    shows_the_codec_colours("wgpu", &mut |s, c| r.render_to_rgba(s, c));
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn the_metal_renderer_shows_the_codec_colours() {
    let mut r = threers::metal::MetalHeadlessRenderer::builder()
        .size(W, H)
        .build()
        .expect("a Metal device");
    shows_the_codec_colours("metal", &mut |s, c| r.render_to_rgba(s, c).unwrap());
}

/// A clip with sound: 64x48 test pattern and a 440 Hz tone, two seconds.
fn with_sound(ext: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/../../tests/support/video/tone.{ext}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

/// Rising zero crossings a second: the tone's frequency.
fn pitch(samples: &[f32], rate: u32) -> f64 {
    let crossings = samples
        .windows(2)
        .filter(|w| w[0] < 0.0 && w[1] >= 0.0)
        .count();
    crossings as f64 * rate as f64 / samples.len() as f64
}

#[test]
fn the_soundtrack_comes_with_the_picture() {
    for (ext, rate) in [("mp4", 44_100), ("webm", 48_000)] {
        let mut player = VideoPlayer::open_inline(with_sound(ext)).unwrap();
        let track = player.soundtrack().expect("a soundtrack").clone();
        assert_eq!((track.sample_rate, track.channels), (rate, 1), "{ext}");
        let samples = track.buffer.unwrap();
        let seconds = samples.len() as f64 / rate as f64;
        assert!((seconds - 2.0).abs() < 0.1, "{ext}: {seconds} s");
        let mid = &samples[samples.len() / 4..samples.len() * 3 / 4];
        let f = pitch(mid, rate);
        assert!((f - 440.0).abs() < 5.0, "{ext}: {f} Hz");

        // Decoded on its own thread when the player has one.
        let mut player = VideoPlayer::open(with_sound(ext)).unwrap();
        let start = std::time::Instant::now();
        while player.soundtrack().is_none() {
            assert!(
                start.elapsed().as_secs() < 20,
                "{ext}: the soundtrack never came"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }
    // A clip with no sound has none.
    assert!(VideoPlayer::open_inline(clip())
        .unwrap()
        .soundtrack()
        .is_none());
}

#[test]
fn seeks_and_loops_are_jumps() {
    let mut player = VideoPlayer::open_inline(clip()).unwrap();
    player.set_looping(true);
    assert_eq!(player.jumps(), 0);
    player.update(0.1);
    assert_eq!(player.jumps(), 0, "running on is not a jump");
    player.seek(0.2);
    assert_eq!(player.jumps(), 1);
    player.update(0.15);
    assert_eq!(player.jumps(), 2, "going round is one");
    assert!((player.time() - 0.05).abs() < 1e-6);
}

/// The hardware decoder (VideoToolbox, NVDEC, VAAPI) decodes what the native
/// decoder decodes, sample for sample, and hands the pictures on as it made
/// them (NV12) for the GPU to convert. Skips on a machine with none.
#[test]
fn hardware_decoding_matches_software() {
    use threers::codec::decode::{Backend, Picture};
    // 256 × 144 of a moving gradient: the 64 × 48 clips are below what
    // AMD's decoder takes (64 lines), and it would skip there.
    let (w, h) = (256, 144);
    let frames: Vec<Yuv420Frame> = (0..20u32)
        .map(|t| {
            let rgba: Vec<u8> = (0..w * h)
                .flat_map(|i| {
                    let (x, y) = (i % w, i / w);
                    [
                        (x + 4 * t) as u8,
                        (y * 255 / h) as u8,
                        ((x ^ y) + 8 * t) as u8,
                        255,
                    ]
                })
                .collect();
            Yuv420Frame::from_rgba(w, h, &rgba)
        })
        .collect();
    let data = encode_mp4(w, h, 25, &frames);
    let mut hw = VideoReader::open(data.clone()).unwrap();
    let mut sw = VideoReader::open_software(data).unwrap();
    if hw.backend() == Backend::Native {
        // Every Mac decodes H.264 in hardware, so there it is a failure.
        if cfg!(target_os = "macos") {
            panic!("VideoToolbox refused H.264");
        }
        eprintln!("skipping: no hardware decoder here");
        return;
    }
    eprintln!("hardware decoder: {:?}", hw.backend());
    assert_eq!(sw.backend(), Backend::Native);
    let mut n = 0;
    while let Some((at, picture)) = hw.next_picture().unwrap() {
        assert!(
            matches!(picture, Picture::Packed(_)),
            "picture {n}: planes, not the hardware's NV12"
        );
        let a = picture.clone().into_decoded();
        // Its buffers back to be filled again: the pictures after this one
        // are decoded into them, and still have to match.
        hw.recycle(picture);
        let (want_at, b) = sw.next_frame().unwrap().expect("as many pictures");
        assert_eq!(at, want_at);
        assert!(
            a.y == b.y && a.u == b.u && a.v == b.v,
            "picture {n} differs"
        );
        n += 1;
    }
    assert!(sw.next_frame().unwrap().is_none());
    assert_eq!(n, 20);
    // Still the hardware: it gave up on nothing along the way.
    assert_ne!(hw.backend(), Backend::Native);
    assert!(!hw.frame_pool().is_empty(), "buffers waiting to be reused");
}
