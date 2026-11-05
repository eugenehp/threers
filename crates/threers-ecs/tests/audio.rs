//! The mixer, measured from its samples, and files made by ffmpeg.

use std::sync::Arc;
use threers::audio::Audio;
use threers::core::Object3D;
use threers::math::Vector3;
use threers_ecs::assets::{Asset, Assets, AssetsPlugin};
use threers_ecs::audio::*;
use threers_ecs::prelude::*;
use threers_ecs::testing::TestGame;

const RATE: f32 = 44_100.0;

/// A constant level, so gains can be read straight off the output.
fn level(value: f32, seconds: f32) -> Audio {
    Audio {
        buffer: Some(Arc::new(vec![value; (RATE * seconds) as usize])),
        sample_rate: RATE as u32,
        channels: 1,
        ..Audio::default()
    }
}

fn out() -> AudioOut {
    AudioOut::offline(RATE)
}

fn render(out: &mut AudioOut, frames: usize) -> Vec<f32> {
    let mut buffer = vec![0.0; frames * 2];
    assert!(out.render(&mut buffer, 2));
    buffer
}

fn crossings(signal: &[f32]) -> usize {
    signal
        .windows(2)
        .filter(|w| w[0] <= 0.0 && w[1] > 0.0)
        .count()
}

#[test]
fn silence_until_something_plays() {
    let mut o = out();
    assert!(render(&mut o, 64).iter().all(|&s| s == 0.0));
}

#[test]
fn a_centred_sound_is_equal_on_both_sides_at_equal_power() {
    let mut o = out();
    o.play(&level(1.0, 0.1));
    let s = render(&mut o, 64);
    assert!((s[0] - s[1]).abs() < 1e-6);
    assert!((s[0] * s[0] + s[1] * s[1] - 1.0).abs() < 1e-4, "{}", s[0]);
}

#[test]
fn panning_moves_a_voice_across_with_no_click() {
    let mut o = out();
    let voice = o.play(&level(1.0, 1.0)).unwrap();
    // Keep the last sample before the change: a jump lands right at the boundary.
    let before = render(&mut o, 16);
    o.send(Command::Mix {
        id: voice,
        gain: 1.0,
        pan: 1.0,
    });

    let mut lefts = vec![before[before.len() - 2]];
    let after = render(&mut o, 2000);
    lefts.extend(after.iter().step_by(2));
    let worst = lefts
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs())
        .fold(0.0_f32, f32::max);
    assert!(worst < 0.01, "jump of {worst}");
    let end = &after[after.len() - 2..];
    assert!(
        end[0].abs() < 1e-3 && (end[1] - 1.0).abs() < 1e-3,
        "{end:?}"
    );
}

#[test]
fn voices_add_up_and_the_output_never_exceeds_one() {
    let mut o = out();
    let quiet = level(0.25, 0.1);
    o.play(&quiet);
    o.play(&quiet);
    let two = render(&mut o, 8)[0];
    assert!(
        (two - 0.5 * std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5 * 2.0 + 0.2,
        "{two}"
    );

    let mut o = out();
    for _ in 0..10 {
        o.play(&level(1.0, 0.1));
    }
    assert!(render(&mut o, 64).iter().all(|s| s.abs() <= 1.0));
}

#[test]
fn a_sound_ends_and_its_voice_goes() {
    let mut o = out();
    o.play(&level(1.0, 0.01));
    assert_eq!(o.offline_voices(), Some(1));
    let s = render(&mut o, 1000);
    assert!(s[2 * 400] != 0.0);
    assert_eq!(s[2 * 999], 0.0, "past the end");
    assert_eq!(o.offline_voices(), Some(0));
}

#[test]
fn a_loop_keeps_going_with_no_gap() {
    let mut o = out();
    let mut sound = level(1.0, 0.01);
    sound.loop_ = true;
    o.play(&sound);
    let s = render(&mut o, 5000);
    assert!(s.iter().step_by(2).all(|&v| v > 0.5), "dropped out");
    assert_eq!(o.offline_voices(), Some(1));
}

#[test]
fn stop_silences_and_a_stale_id_does_nothing() {
    let mut o = out();
    let voice = o.play(&level(1.0, 1.0)).unwrap();
    o.stop(voice);
    o.stop(voice);
    o.send(Command::Mix {
        id: voice,
        gain: 1.0,
        pan: 0.0,
    });
    assert!(render(&mut o, 16).iter().all(|&s| s == 0.0));
}

#[test]
fn a_different_source_rate_keeps_its_pitch() {
    // 440 Hz recorded at 22.05 kHz, played into a 44.1 kHz output.
    let source_rate = 22_050.0;
    let samples: Vec<f32> = (0..22_050)
        .map(|i| (i as f32 / source_rate * 440.0 * std::f32::consts::TAU).sin())
        .collect();
    let sound = Audio {
        buffer: Some(Arc::new(samples)),
        sample_rate: source_rate as u32,
        channels: 1,
        ..Audio::default()
    };
    let mut o = out();
    o.play(&sound);
    let s = render(&mut o, 44_100);
    let left: Vec<f32> = s.iter().step_by(2).copied().collect();
    let count = crossings(&left);
    assert!((438..=442).contains(&count), "{count} Hz");
}

#[test]
fn playback_rate_shifts_pitch() {
    let mut sound = tone(440.0, 1.0, 1.0);
    sound.playback_rate = 2.0;
    let mut o = out();
    o.play(&sound);
    let s = render(&mut o, 22_050);
    let left: Vec<f32> = s.iter().step_by(2).copied().collect();
    let count = crossings(&left);
    assert!((438..=442).contains(&count), "{count} in half a second");
}

#[test]
fn stereo_sources_keep_their_sides() {
    let sound = Audio {
        buffer: Some(Arc::new([0.8_f32, -0.8].repeat(1000))),
        sample_rate: RATE as u32,
        channels: 2,
        ..Audio::default()
    };
    let mut o = out();
    o.play(&sound);
    let s = render(&mut o, 16);
    assert!(s[0] > 0.0 && s[1] < 0.0, "{:?}", &s[..2]);
}

#[test]
fn master_volume_and_mono_output() {
    let mut o = out();
    o.set_master(0.5);
    o.play(&level(1.0, 0.1));
    let s = render(&mut o, 8);
    assert!((s[0] - 0.5 * std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-4);

    let mut mono = vec![0.0; 8];
    o.render(&mut mono, 1);
    assert!((mono[0] - 0.5 * std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-4);
}

#[test]
fn a_tone_is_the_frequency_it_says() {
    let sound = tone(1000.0, 1.0, 1.0);
    let count = crossings(sound.buffer.as_ref().unwrap());
    assert!((998..=1001).contains(&count), "{count}");
}

fn check_file(bytes: &[u8]) {
    let sound = decode(bytes).expect("decode");
    assert_eq!(sound.sample_rate, 44_100);
    let samples = sound.buffer.as_ref().unwrap();
    let frames = samples.len() / sound.channels as usize;
    let seconds = frames as f32 / 44_100.0;
    assert!((0.2..0.35).contains(&seconds), "{seconds}");
    let first: Vec<f32> = samples
        .iter()
        .step_by(sound.channels as usize)
        .copied()
        .collect();
    // Encoders pad with silence, which is not part of the tone: count only the
    // stretch that has signal in it.
    let start = first.iter().position(|s| s.abs() > 0.01).unwrap();
    let end = first.iter().rposition(|s| s.abs() > 0.01).unwrap();
    let sounding = (end - start) as f32 / 44_100.0;
    let hz = crossings(&first[start..=end]) as f32 / sounding;
    assert!((hz - 440.0).abs() < 15.0, "{hz} Hz");
}

#[test]
fn real_files_decode_to_the_right_pitch_and_length() {
    check_file(include_bytes!("data/tone.wav"));
    check_file(include_bytes!("data/tone.ogg"));
    check_file(include_bytes!("data/tone.mp3"));
}

#[test]
fn garbage_is_an_error_not_a_panic() {
    assert!(decode(b"definitely not audio").is_err());
    assert!(decode(&[]).is_err());
    let mut cut = include_bytes!("data/tone.ogg").to_vec();
    cut.truncate(200);
    let _ = decode(&cut);
}

#[test]
fn sounds_load_as_assets() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let mut game = TestGame::new(|app| {
        app.plugin(AssetsPlugin::new(dir));
    });
    let handle = game
        .world()
        .resource_mut::<Assets>()
        .load::<SoundAsset>("tone.ogg");
    let start = std::time::Instant::now();
    while game.resource::<Assets>().get(&handle).is_none() {
        assert!(start.elapsed().as_secs() < 10);
        game.frames(1);
    }
    assert!(SoundAsset::parse(b"nope").is_err());
}

fn game() -> (TestGame, Entity) {
    let mut id = None;
    let game = TestGame::new(|app| {
        app.plugin(AudioPlugin::Offline);
        let node = app.world.spawn(Object3D::group());
        app.world.scene.get_mut(node).unwrap().position = Vector3::new(2.0, 0.0, 0.0);
        id = Some(node);
    });
    (game, id.unwrap())
}

fn sides(game: &mut TestGame) -> (f32, f32) {
    let mut buffer = vec![0.0; 4000];
    game.world()
        .resource_mut::<AudioOut>()
        .render(&mut buffer, 2);
    (buffer[3998], buffer[3999])
}

fn settle(game: &mut TestGame) {
    game.world().scene.update_world();
    game.frames(2);
}

#[test]
fn a_speaker_to_the_right_is_louder_on_the_right_and_fades_with_distance() {
    let (mut g, id) = game();
    let mut sound = level(1.0, 5.0);
    sound.loop_ = true;
    g.world().insert(id, Speaker::new(sound));
    settle(&mut g);
    let (left, right) = sides(&mut g);
    assert!(right > left * 2.0, "{left} {right}");

    g.world().scene.get_mut(id).unwrap().position = Vector3::new(20.0, 0.0, 0.0);
    settle(&mut g);
    let (_, far) = sides(&mut g);
    assert!(far < right * 0.5, "{far} {right}");

    g.world().scene.get_mut(id).unwrap().position = Vector3::new(-2.0, 0.0, 0.0);
    settle(&mut g);
    let (left, right) = sides(&mut g);
    assert!(left > right * 2.0, "{left} {right}");
}

#[test]
fn despawning_a_speaker_stops_its_sound() {
    let (mut g, id) = game();
    let mut sound = level(1.0, 5.0);
    sound.loop_ = true;
    g.world().insert(id, Speaker::new(sound));
    settle(&mut g);
    assert_eq!(g.resource::<AudioOut>().offline_voices(), Some(1));
    g.world().despawn(id);
    assert_eq!(g.resource::<AudioOut>().offline_voices(), Some(0));
}

#[test]
fn clearing_playing_stops_and_setting_it_starts_again() {
    let (mut g, id) = game();
    let mut sound = level(1.0, 5.0);
    sound.loop_ = true;
    g.world().insert(id, Speaker::new(sound));
    settle(&mut g);
    g.world()
        .get_mut::<Speaker>(id)
        .unwrap()
        .source
        .audio
        .playing = false;
    g.frames(1);
    assert_eq!(g.resource::<AudioOut>().offline_voices(), Some(0));
    g.world()
        .get_mut::<Speaker>(id)
        .unwrap()
        .source
        .audio
        .playing = true;
    g.frames(1);
    assert_eq!(g.resource::<AudioOut>().offline_voices(), Some(1));
}

#[test]
fn a_short_sound_reports_when_it_ends() {
    let (mut g, id) = game();
    g.world().insert(id, Speaker::new(level(1.0, 0.1)));
    g.frames(1);
    assert!(g.world().get::<Speaker>(id).unwrap().is_playing());
    g.seconds(0.2);
    assert_eq!(g.world().events::<SoundFinished>(), [SoundFinished(id)]);
    assert!(!g.world().get::<Speaker>(id).unwrap().is_playing());
}

#[test]
fn pausing_the_game_pauses_sound() {
    let (mut g, _) = game();
    g.world().resource_mut::<AudioOut>().play(&level(1.0, 5.0));
    g.world().resource_mut::<Time>().paused = true;
    g.frames(1);
    let mut buffer = vec![0.0; 64];
    g.world().resource_mut::<AudioOut>().render(&mut buffer, 2);
    assert!(buffer.iter().all(|&s| s == 0.0));
    g.world().resource_mut::<Time>().paused = false;
    g.frames(1);
    g.world().resource_mut::<AudioOut>().render(&mut buffer, 2);
    assert!(buffer.iter().any(|&s| s != 0.0));
}

/// Through the real device. Fails on a machine with no output, which is the point
/// of running it: this is the one test of the hardware path.
#[test]
fn the_real_device_opens_and_takes_commands() {
    let mut out = match AudioOut::open() {
        Ok(out) => out,
        // A machine with no sound card, as a hosted runner has.
        Err(e) => return eprintln!("SKIP: no audio output device: {e}"),
    };
    let voice = out.play(&tone(440.0, 0.05, 0.0)).expect("voice");
    std::thread::sleep(std::time::Duration::from_millis(100));
    out.stop(voice);
    assert!(!out.render(&mut [0.0; 4], 2), "a device is not offline");
}

fn peak(signal: &[f32]) -> f32 {
    signal.iter().fold(0.0, |top, s| top.max(s.abs()))
}

#[test]
fn a_waveform_loops_with_no_seam() {
    let sound = waveform(80.0, 0.3, 0.5, &[0.5, 0.3]);
    let samples = sound.buffer.as_ref().unwrap();
    // The step across the join is no bigger than any step inside.
    let seam = (samples[0] - samples[samples.len() - 1]).abs();
    let largest = samples
        .windows(2)
        .map(|p| (p[1] - p[0]).abs())
        .fold(0.0, f32::max);
    assert!(seam <= largest * 1.01, "{seam} against {largest}");
    assert!(sound.loop_);
    assert!(peak(samples) <= 0.5);
}

#[test]
fn pitch_changes_a_playing_sound_without_restarting_it() {
    let mut out = out();
    let voice = out.play(&waveform(200.0, 0.5, 0.5, &[])).unwrap();
    let low = crossings(&render(&mut out, 22_050));
    out.set_pitch(voice, 2.0);
    render(&mut out, 22_050); // through the glide
    let high = crossings(&render(&mut out, 22_050));
    let ratio = high as f32 / low as f32;
    assert!((1.95..2.05).contains(&ratio), "{low} to {high}");
    assert_eq!(out.offline_voices(), Some(1));
}

#[test]
fn a_bus_turns_its_own_sounds_down_and_no_others() {
    let mut out = out();
    out.play(&level(0.5, 2.0));
    out.set_bus_volume(Bus::Music, 0.0);
    for _ in 0..20 {
        render(&mut out, 1024);
    }
    assert!(peak(&render(&mut out, 1024)) > 0.2, "effects kept going");
    out.set_bus_volume(Bus::Effects, 0.25);
    for _ in 0..40 {
        render(&mut out, 1024);
    }
    let left = peak(&render(&mut out, 1024));
    assert!((0.05..0.12).contains(&left), "{left}");
}

#[test]
fn a_fade_out_falls_away_and_ends() {
    let mut out = out();
    let voice = out.play(&looped(0.5)).unwrap();
    render(&mut out, 1024);
    out.fade_out(voice, 0.1);
    let middle = peak(&render(&mut out, 2205)[3000..]);
    assert!(middle < 0.35, "{middle}");
    render(&mut out, 44_100 / 5);
    assert_eq!(out.offline_voices(), Some(0));
}

fn looped(value: f32) -> Audio {
    let mut sound = level(value, 0.5);
    sound.loop_ = true;
    sound
}

/// Equal parts of each mid-way, and only the new tune once it is done.
#[test]
fn new_music_crossfades_from_the_old() {
    let mut out = out();
    out.play_music(&looped(0.4), 0.0);
    render(&mut out, 4410);
    out.play_music(&looped(0.2), 0.2);
    assert_eq!(out.offline_voices(), Some(2));
    let mid = render(&mut out, 4410);
    let at_middle = peak(&mid[mid.len() - 2..]);
    assert!(at_middle > 0.15 && at_middle < 0.4, "{at_middle}");
    render(&mut out, 44_100 / 2);
    assert_eq!(out.offline_voices(), Some(1));
    let end = peak(&render(&mut out, 1024));
    assert!(
        (end - 0.2 * std::f32::consts::FRAC_1_SQRT_2).abs() < 0.03,
        "{end}"
    );
}

#[test]
fn music_answers_to_its_bus() {
    let mut out = out();
    out.play_music(&looped(0.4), 0.0);
    out.set_bus_volume(Bus::Music, 0.0);
    for _ in 0..40 {
        render(&mut out, 1024);
    }
    assert!(peak(&render(&mut out, 1024)) < 0.01);
}

#[test]
fn a_speakers_pitch_can_move_while_it_plays() {
    let (mut game, id) = game();
    game.world()
        .insert(id, Speaker::looping(waveform(200.0, 0.5, 0.5, &[])));
    settle(&mut game);
    let listen = |game: &mut TestGame| {
        let mut buffer = vec![0.0; 44_100];
        game.world()
            .resource_mut::<AudioOut>()
            .render(&mut buffer, 2);
        let left: Vec<f32> = buffer.iter().step_by(2).copied().collect();
        crossings(&left)
    };
    let before = listen(&mut game);
    game.world().get_mut::<Speaker>(id).unwrap().pitch = 1.5;
    game.frames(2);
    let mut settle_buffer = vec![0.0; 22_050];
    game.world()
        .resource_mut::<AudioOut>()
        .render(&mut settle_buffer, 2);
    let after = listen(&mut game);
    let ratio = after as f32 / before as f32;
    assert!((1.45..1.55).contains(&ratio), "{before} to {after}");
    assert!(game.world().get::<Speaker>(id).unwrap().is_playing());
}

#[test]
fn speakers_play_several_sounds_from_one_entity_and_go_with_it() {
    let (mut game, id) = game();
    game.world().insert(
        id,
        Speakers(vec![
            Speaker::looping(waveform(200.0, 0.5, 0.4, &[])),
            Speaker::looping(waveform(500.0, 0.5, 0.4, &[])),
        ]),
    );
    settle(&mut game);
    assert_eq!(
        game.world().resource::<AudioOut>().offline_voices(),
        Some(2)
    );
    game.world().get_mut::<Speakers>(id).unwrap().0[1]
        .source
        .audio
        .playing = false;
    game.frames(1);
    let mut buffer = vec![0.0; 4410];
    game.world()
        .resource_mut::<AudioOut>()
        .render(&mut buffer, 2);
    assert_eq!(
        game.world().resource::<AudioOut>().offline_voices(),
        Some(1)
    );
    game.world().despawn(id);
    game.frames(1);
    game.world()
        .resource_mut::<AudioOut>()
        .render(&mut buffer, 2);
    assert_eq!(
        game.world().resource::<AudioOut>().offline_voices(),
        Some(0)
    );
}
