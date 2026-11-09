//! Video on an entity's mesh, with its sound.
//!
//! [`Video`] holds a [`VideoPlayer`]; the app moves every one on by game time
//! each frame, so pausing the game or slowing it down does the same to the
//! video. [`Spawned::video`](crate::spawn::Spawned::video) puts the player's
//! texture in the mesh's `map` and attaches it:
//!
//! ```no_run
//! # use threers_ecs::prelude::*;
//! use threers_ecs::video::VideoPlayer;
//!
//! let mut app = App::new();
//! let player = VideoPlayer::open(std::fs::read("clip.mp4").unwrap()).unwrap();
//! app.world
//!     .mesh(PlaneGeometry::new(1.6, 0.9), BasicMaterial::new(Color::WHITE))
//!     .video(player);
//! ```
//!
//! With the `audio` feature and an [`AudioPlugin`](crate::audio::AudioPlugin),
//! the clip's soundtrack plays too, kept to the picture: it starts where the
//! video is once decoded, starts again at the right place after a seek or a
//! loop, stops while the video is paused, follows its speed and the game's
//! time scale, and holds with everything else while the game is paused.
//!
//! Needs the `video` feature, which turns on threers's decoders.

use crate::app::Time;
use crate::world::{Component, World};
pub use threers::textures::VideoTexture;
pub use threers_video::VideoPlayer;

/// A video playing on this entity. Its texture is wherever it was put; the
/// app keeps it moving on game time, and plays its sound.
pub struct Video {
    /// The video.
    pub player: VideoPlayer,
    /// Play the soundtrack, when there is one and the app has sound. On
    /// unless made [`muted`](Self::muted).
    pub sound: bool,
    /// The soundtrack's volume: 1 as recorded.
    pub volume: f32,
    /// The volume group it answers to.
    #[cfg(feature = "audio")]
    pub bus: crate::audio::Bus,
    #[cfg(feature = "audio")]
    voice: Option<crate::audio::VoiceId>,
    /// The player's jump count the voice was started for.
    #[cfg(feature = "audio")]
    heard_jumps: u64,
    #[cfg(feature = "audio")]
    sent: (f32, f32),
}
impl Component for Video {}

impl Video {
    /// A video with its sound.
    pub fn new(player: VideoPlayer) -> Self {
        Self {
            player,
            sound: true,
            volume: 1.0,
            #[cfg(feature = "audio")]
            bus: crate::audio::Bus::default(),
            #[cfg(feature = "audio")]
            voice: None,
            #[cfg(feature = "audio")]
            heard_jumps: 0,
            #[cfg(feature = "audio")]
            sent: (1.0, 1.0),
        }
    }

    /// A video without its sound.
    pub fn muted(player: VideoPlayer) -> Self {
        Self {
            sound: false,
            ..Self::new(player)
        }
    }

    /// Whether its soundtrack is playing now.
    pub fn is_sounding(&self) -> bool {
        #[cfg(feature = "audio")]
        return self.voice.is_some();
        #[cfg(not(feature = "audio"))]
        false
    }
}

/// Move every [`Video`] on by this frame's game time, and keep the sound of
/// each with its picture.
pub(crate) fn run_videos(w: &mut World) {
    let time = *w.resource::<Time>();
    w.each::<&mut Video>(|video| video.player.update(time.delta as f64));
    #[cfg(feature = "audio")]
    if w.get_resource::<crate::audio::AudioOut>().is_some() {
        w.with_resource::<crate::audio::AudioOut, _>(|w, out| {
            w.each::<&mut Video>(|video| follow(out, video, time.scale));
        });
    }
}

/// Start, stop, move or re-pitch a video's soundtrack to match its picture.
#[cfg(feature = "audio")]
fn follow(out: &mut crate::audio::AudioOut, video: &mut Video, scale: f64) {
    let wanted = video.sound && video.player.is_playing() && !video.player.is_finished();
    let pitch = (video.player.speed() * scale) as f32;
    let (volume, bus) = (video.volume, video.bus);
    let track = if wanted && pitch > 0.0 {
        video.player.soundtrack().cloned()
    } else {
        None
    };
    let Some(track) = track else {
        if let Some(voice) = video.voice.take() {
            out.stop(voice);
        }
        return;
    };
    let jumps = video.player.jumps();
    if video.voice.is_none() || jumps != video.heard_jumps {
        if let Some(voice) = video.voice.take() {
            out.stop(voice);
        }
        let voice = out.play_mixed(&track, volume, 0.0, bus, 0.0);
        if let Some(voice) = voice {
            out.seek(voice, video.player.time());
            if pitch != 1.0 {
                out.set_pitch(voice, pitch);
            }
        }
        video.voice = voice;
        video.heard_jumps = jumps;
        video.sent = (pitch, volume);
        return;
    }
    if let Some(voice) = video.voice {
        if video.sent.0 != pitch {
            out.set_pitch(voice, pitch);
        }
        if video.sent.1 != volume {
            out.send(crate::audio::Command::Mix {
                id: voice,
                gain: volume,
                pan: 0.0,
            });
        }
        video.sent = (pitch, volume);
    }
}

/// Stop a removed video's sound.
#[cfg(feature = "audio")]
pub(crate) fn on_remove(w: &mut World, video: &Video) {
    if let (Some(voice), true) = (
        video.voice,
        w.get_resource::<crate::audio::AudioOut>().is_some(),
    ) {
        w.resource_mut::<crate::audio::AudioOut>().stop(voice);
    }
}
