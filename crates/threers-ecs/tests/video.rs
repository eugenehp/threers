//! A video on an entity's mesh moves on with game time: a frame of the app
//! shows the picture due, and a paused game holds it.

use std::sync::Arc;
use threers::codec::h264::{encode_mp4, Yuv420Frame};
use threers_ecs::prelude::*;

const W: u32 = 32;
const H: u32 = 32;

/// Two pictures at 10 fps: red, then green.
fn clip() -> Vec<u8> {
    let solid = |c: [u8; 3]| {
        let rgba: Vec<u8> = (0..W * H).flat_map(|_| [c[0], c[1], c[2], 255]).collect();
        Yuv420Frame::from_rgba(W, H, &rgba)
    };
    encode_mp4(W, H, 10, &[solid([220, 30, 30]), solid([30, 200, 40])])
}

/// The red channel of the picture on show.
fn red(app: &App, screen: Entity) -> u8 {
    app.world
        .get::<Video>(screen)
        .unwrap()
        .player
        .video_texture()
        .stream()
        .read(|rgba, _| rgba[0])
}

#[test]
fn a_video_follows_game_time() {
    let mut app = App::new();
    let player = VideoPlayer::open_inline(clip()).unwrap();
    let texture = player.texture();
    let screen = app
        .world
        .mesh(
            PlaneGeometry::new(1.0, 1.0),
            BasicMaterial::new(Color::WHITE),
        )
        .video(player)
        .id();

    // The mesh shows the video's texture.
    let node = app.world.node(screen).unwrap();
    let ObjectKind::Mesh(mesh) = &node.kind else {
        panic!("a mesh")
    };
    let Material::Basic(material) = &*mesh.material else {
        panic!("basic")
    };
    assert!(Arc::ptr_eq(material.map.as_ref().unwrap(), &texture));
    assert!(red(&app, screen) > 150, "red first");

    // Paused, the picture stays.
    app.world.resource_mut::<Time>().paused = true;
    app.frame(0.2);
    assert!(red(&app, screen) > 150, "paused holds the first picture");

    app.world.resource_mut::<Time>().paused = false;
    app.frame(0.1);
    assert!(
        red(&app, screen) < 80,
        "green once a tenth of game time has passed"
    );
}

/// The soundtrack plays with the picture, from where the picture is.
#[cfg(feature = "audio")]
#[test]
fn a_video_plays_its_sound_in_step() {
    use threers_ecs::audio::{AudioOut, AudioPlugin};
    let file = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/support/video/tone.mp4"
    ))
    .unwrap();
    let mut app = App::new();
    app.plugin(AudioPlugin::Offline);
    let mut player = VideoPlayer::open_inline(file.clone()).unwrap();
    let track = player.soundtrack().unwrap().buffer.clone().unwrap();
    let screen = app
        .world
        .mesh(
            PlaneGeometry::new(1.0, 1.0),
            BasicMaterial::new(Color::WHITE),
        )
        .video(player)
        .id();
    app.frame(0.0);
    let sounding = |app: &App| app.world.get::<Video>(screen).unwrap().is_sounding();
    assert!(sounding(&app));
    assert_eq!(app.world.resource::<AudioOut>().offline_voices(), Some(1));

    // Seek a second in: the next samples out are the track's from there,
    // at the equal-power centre gain.
    app.world.get_mut::<Video>(screen).unwrap().player.seek(1.0);
    app.frame(0.0);
    let mut out = vec![0.0f32; 256];
    app.world.resource_mut::<AudioOut>().render(&mut out, 1);
    let gain = std::f32::consts::FRAC_1_SQRT_2;
    for (i, got) in out.iter().enumerate() {
        let want = track[44_100 + i] * gain;
        assert!((got - want).abs() < 1e-4, "sample {i}: {got} vs {want}");
    }
    assert_eq!(
        app.world.resource::<AudioOut>().offline_voices(),
        Some(1),
        "the old voice stopped"
    );

    // Paused, the sound stops; playing again, it comes back.
    app.world.get_mut::<Video>(screen).unwrap().player.pause();
    app.frame(0.1);
    assert!(!sounding(&app));
    assert_eq!(app.world.resource::<AudioOut>().offline_voices(), Some(0));
    app.world.get_mut::<Video>(screen).unwrap().player.play();
    app.frame(0.1);
    assert!(sounding(&app));

    // Removed, it goes quiet; a muted one never sounds.
    app.world.despawn(screen);
    assert_eq!(app.world.resource::<AudioOut>().offline_voices(), Some(0));
    let quiet = app
        .world
        .mesh(
            PlaneGeometry::new(1.0, 1.0),
            BasicMaterial::new(Color::WHITE),
        )
        .id();
    app.world
        .insert(quiet, Video::muted(VideoPlayer::open_inline(file).unwrap()));
    app.frame(0.1);
    assert!(!app.world.get::<Video>(quiet).unwrap().is_sounding());
}
