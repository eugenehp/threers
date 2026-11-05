//! Sound: a mixer, spatial sources and the output device.
//!
//! The [`Mixer`] is plain Rust that turns voices into samples, the same on every
//! platform, and can run with no device at all. On a native build it lives on the
//! audio thread and takes [`Command`]s over a channel, so the thread that must
//! never wait never takes a lock.
//!
//! ```no_run
//! use threers_ecs::audio::*;
//! use threers_ecs::prelude::*;
//!
//! fn setup(app: &mut App) {
//!     app.plugin(AudioPlugin::default());
//!     let beep = tone(880.0, 0.1, 0.5);
//!     app.world.resource_mut::<AudioOut>().play(&beep);
//! }
//! ```

use crate::app::{App, MainCamera, Plugin, Time};
use crate::assets::Asset;
use crate::query::NodeRef;
use crate::world::{Component, Entity, World};
use std::sync::Arc;
use threers::audio::{Audio, AudioListener, PositionalAudio};

/// One playing sound's id. Ids are not reused, so a stale one does nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoiceId(u64);

/// A group of sounds with one volume, for a settings menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Bus {
    Music,
    #[default]
    Effects,
    Voice,
}

/// What the game asks of the mixer.
#[derive(Debug, Clone)]
pub enum Command {
    Play {
        id: VoiceId,
        samples: Arc<Vec<f32>>,
        channels: u16,
        rate: f32,
        looping: bool,
        gain: f32,
        pan: f32,
        bus: Bus,
        /// Seconds to rise from silence. Zero starts at full level.
        fade_in: f32,
    },
    Stop(VoiceId),
    Mix {
        id: VoiceId,
        gain: f32,
        pan: f32,
    },
    /// Play faster or slower, and so higher or lower: 1 is as recorded. Glides
    /// there rather than steps, so a rising engine has no zipper noise.
    Pitch {
        id: VoiceId,
        factor: f32,
    },
    /// Fall to silence over `seconds` and stop.
    FadeOut {
        id: VoiceId,
        seconds: f32,
    },
    BusVolume(Bus, f32),
    Master(f32),
    Pause(bool),
    /// Silence everything at once.
    StopAll,
}

struct Voice {
    id: VoiceId,
    samples: Arc<Vec<f32>>,
    channels: usize,
    /// Source frames per output frame, at the source's own pitch.
    step: f64,
    /// The pitch now, and where it is heading.
    pitch: f64,
    pitch_target: f64,
    bus: Bus,
    /// A fade's level, and how much it moves per frame.
    level: f32,
    level_step: f32,
    /// Whether it stops on reaching silence.
    fading_out: bool,
    position: f64,
    looping: bool,
    /// Left and right gain now, and where they are heading.
    now: [f32; 2],
    target: [f32; 2],
}

/// Equal power, so a sound crossing the middle keeps its loudness.
fn gains(gain: f32, pan: f32) -> [f32; 2] {
    let angle = (pan.clamp(-1.0, 1.0) + 1.0) * std::f32::consts::FRAC_PI_4;
    [gain * angle.cos(), gain * angle.sin()]
}

impl Voice {
    fn frames(&self) -> usize {
        self.samples.len() / self.channels
    }

    /// Stereo at the current position, between source frames. A loop interpolates
    /// across its seam, which a one-shot has no reason to.
    fn sample(&self) -> [f32; 2] {
        let frames = self.frames();
        let at = self.position.floor();
        let t = (self.position - at) as f32;
        let first = at as usize;
        let next = if first + 1 < frames {
            first + 1
        } else if self.looping {
            0
        } else {
            first
        };
        let read = |frame: usize, channel: usize| {
            self.samples[frame * self.channels + channel.min(self.channels - 1)]
        };
        [0, 1].map(|channel| {
            let (a, b) = (read(first, channel), read(next, channel));
            a + (b - a) * t
        })
    }
}

/// Turns voices into samples.
pub struct Mixer {
    rate: f32,
    voices: Vec<Voice>,
    master: f32,
    /// Each [`Bus`]'s volume, and where its slide has got to.
    buses: [f32; 3],
    bus_now: [f32; 3],
    paused: bool,
    /// Gain moved per output frame: a 5 ms slide, so a change never clicks.
    ramp: f32,
}

impl Mixer {
    /// `rate` is the output's sample rate.
    pub fn new(rate: f32) -> Self {
        Self {
            rate,
            voices: Vec::new(),
            master: 1.0,
            buses: [1.0; 3],
            bus_now: [1.0; 3],
            paused: false,
            ramp: 1.0 / (0.005 * rate),
        }
    }

    pub fn voices(&self) -> usize {
        self.voices.len()
    }

    pub fn apply(&mut self, command: Command) {
        match command {
            Command::Play {
                id,
                samples,
                channels,
                rate,
                looping,
                gain,
                pan,
                bus,
                fade_in,
            } => {
                if samples.is_empty() || channels == 0 {
                    return;
                }
                let start = gains(gain, pan);
                let (level, level_step) = if fade_in > 0.0 {
                    (0.0, 1.0 / (fade_in * self.rate))
                } else {
                    (1.0, 0.0)
                };
                self.voices.push(Voice {
                    id,
                    samples,
                    channels: channels as usize,
                    step: (rate / self.rate) as f64,
                    pitch: 1.0,
                    pitch_target: 1.0,
                    bus,
                    level,
                    level_step,
                    fading_out: false,
                    position: 0.0,
                    looping,
                    now: start,
                    target: start,
                });
            }
            Command::Stop(id) => self.voices.retain(|voice| voice.id != id),
            Command::Mix { id, gain, pan } => {
                if let Some(voice) = self.voices.iter_mut().find(|voice| voice.id == id) {
                    voice.target = gains(gain, pan);
                }
            }
            Command::Pitch { id, factor } => {
                if let Some(voice) = self.voices.iter_mut().find(|voice| voice.id == id) {
                    voice.pitch_target = factor.max(0.01) as f64;
                }
            }
            Command::FadeOut { id, seconds } => {
                let rate = self.rate;
                if let Some(voice) = self.voices.iter_mut().find(|voice| voice.id == id) {
                    voice.fading_out = true;
                    voice.level_step = -1.0 / (seconds.max(0.001) * rate);
                }
            }
            Command::BusVolume(bus, volume) => self.buses[bus as usize] = volume.max(0.0),
            Command::Master(volume) => self.master = volume,
            Command::Pause(paused) => self.paused = paused,
            Command::StopAll => self.voices.clear(),
        }
    }

    /// Fill `out`, interleaved with `channels` per frame. Mono output gets the
    /// average, and channels past the second are left silent.
    pub fn render(&mut self, out: &mut [f32], channels: usize) {
        out.fill(0.0);
        if self.paused || channels == 0 {
            return;
        }
        let ramp = self.ramp;
        // About 50 ms to any pitch, whatever the output rate.
        let glide = 1.0 - (-1.0 / (0.05 * self.rate as f64)).exp();
        for bus in 0..3 {
            let gap = self.buses[bus] - self.bus_now[bus];
            // A block at a time is smooth enough for a volume slider.
            self.bus_now[bus] += gap * 0.25;
        }
        let bus_now = self.bus_now;
        for voice in &mut self.voices {
            let bus = bus_now[voice.bus as usize];
            for frame in out.chunks_exact_mut(channels) {
                if voice.position >= voice.frames() as f64 {
                    break;
                }
                if voice.fading_out && voice.level <= 0.0 {
                    voice.position = voice.frames() as f64;
                    voice.looping = false;
                    break;
                }
                voice.level = (voice.level + voice.level_step).clamp(0.0, 1.0);
                voice.pitch += (voice.pitch_target - voice.pitch) * glide;
                let [left, right] = voice.sample();
                for side in 0..2 {
                    let gap = voice.target[side] - voice.now[side];
                    voice.now[side] += gap.clamp(-ramp, ramp);
                }
                let scale = voice.level * bus;
                let (left, right) = (left * voice.now[0] * scale, right * voice.now[1] * scale);
                if channels == 1 {
                    frame[0] += (left + right) * 0.5;
                } else {
                    frame[0] += left;
                    frame[1] += right;
                }
                voice.position += voice.step * voice.pitch;
                if voice.looping && voice.position >= voice.frames() as f64 {
                    voice.position -= voice.frames() as f64;
                }
            }
        }
        self.voices
            .retain(|voice| voice.position < voice.frames() as f64);
        for sample in out.iter_mut() {
            *sample = (*sample * self.master).clamp(-1.0, 1.0);
        }
    }
}

/// A note with overtones that loops with no seam: the length is rounded to whole
/// cycles, so the end meets the start. `harmonics` are the strengths of the first
/// overtones and up, the fundamental being 1. Good for an engine or a hum, with
/// [`Speaker::pitch`] moving it.
///
/// ```
/// use threers_ecs::audio::{waveform, Speaker};
/// let mut engine = Speaker::looping(waveform(80.0, 0.5, 0.4, &[0.6, 0.4, 0.3]));
/// engine.pitch = 1.8;
/// ```
pub fn waveform(frequency: f32, seconds: f32, volume: f32, harmonics: &[f32]) -> Audio {
    let rate = 44_100u32;
    let cycles = (frequency * seconds).round().max(1.0);
    let count = (cycles / frequency * rate as f32).round() as usize;
    let total: f32 = 1.0 + harmonics.iter().sum::<f32>();
    let samples: Vec<f32> = (0..count)
        .map(|i| {
            let phase = i as f32 / count as f32 * cycles * std::f32::consts::TAU;
            let mut value = phase.sin();
            for (n, strength) in harmonics.iter().enumerate() {
                value += strength * (phase * (n + 2) as f32).sin();
            }
            value / total * volume
        })
        .collect();
    Audio {
        buffer: Some(Arc::new(samples)),
        sample_rate: rate,
        channels: 1,
        loop_: true,
        ..Audio::default()
    }
}

/// A pure tone, for sound effects with no file behind them.
pub fn tone(frequency: f32, seconds: f32, volume: f32) -> Audio {
    let rate = 44_100;
    let count = (seconds * rate as f32) as usize;
    let samples = (0..count)
        .map(|i| {
            let t = i as f32 / rate as f32;
            // A short fade at each end, or the edges click.
            let edge = (t / 0.005).min((seconds - t) / 0.005).clamp(0.0, 1.0);
            (t * frequency * std::f32::consts::TAU).sin() * volume * edge
        })
        .collect();
    Audio {
        buffer: Some(Arc::new(samples)),
        sample_rate: rate,
        channels: 1,
        ..Audio::default()
    }
}

/// Decode a WAV, Ogg Vorbis or MP3 file.
///
/// # Errors
///
/// If the bytes are not one of those.
pub fn decode(bytes: &[u8]) -> Result<Audio, String> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let source = std::io::Cursor::new(bytes.to_vec());
    let stream = MediaSourceStream::new(Box::new(source), Default::default());
    let probed = symphonia::default::get_probe()
        .format(
            &Hint::new(),
            stream,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| format!("not a sound file: {e}"))?;
    let mut format = probed.format;
    let track = format.default_track().ok_or("no audio track")?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("unsupported sound: {e}"))?;

    let mut samples = Vec::new();
    let (mut rate, mut channels) = (0, 0);
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(_)) => break,
            Err(e) => return Err(format!("bad sound file: {e}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            // One damaged frame is a blip, not a reason to lose the sound.
            Err(Error::DecodeError(_)) => continue,
            Err(e) => return Err(format!("bad sound file: {e}")),
        };
        let spec = *decoded.spec();
        rate = spec.rate;
        channels = spec.channels.count() as u16;
        let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buffer.copy_interleaved_ref(decoded);
        samples.extend_from_slice(buffer.samples());
    }
    if samples.is_empty() {
        return Err("sound file is empty".to_owned());
    }
    Ok(Audio {
        buffer: Some(Arc::new(samples)),
        sample_rate: rate,
        channels,
        ..Audio::default()
    })
}

/// A decoded sound file, for [`Assets`](crate::assets::Assets).
pub struct SoundAsset(pub Audio);

impl Asset for SoundAsset {
    fn parse(bytes: &[u8]) -> Result<Self, String> {
        decode(bytes).map(SoundAsset)
    }
}

enum Sink {
    /// Rendered on request, for tests and tools.
    Offline(Mixer),
    #[cfg(not(target_arch = "wasm32"))]
    Device {
        commands: std::sync::mpsc::Sender<Command>,
        _stream: cpal::Stream,
    },
    #[cfg(all(target_arch = "wasm32", feature = "web"))]
    Web(std::rc::Rc<std::cell::RefCell<Mixer>>),
}

/// Where sound goes, as a resource.
pub struct AudioOut {
    sink: Sink,
    next: u64,
    music: Option<VoiceId>,
}

impl AudioOut {
    /// A mixer with no device, read with [`render`](Self::render).
    pub fn offline(rate: f32) -> Self {
        Self {
            sink: Sink::Offline(Mixer::new(rate)),
            next: 0,
            music: None,
        }
    }

    /// The default output device.
    ///
    /// # Errors
    ///
    /// If there is none, or it cannot be opened.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open() -> Result<Self, String> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
        use cpal::SampleFormat;

        let device = cpal::default_host()
            .default_output_device()
            .ok_or("no audio output device")?;
        let config = device.default_output_config().map_err(|e| e.to_string())?;
        let format = config.sample_format();
        let config: cpal::StreamConfig = config.into();
        let channels = config.channels as usize;
        let mut mixer = Mixer::new(config.sample_rate.0 as f32);
        let (commands, inbox) = std::sync::mpsc::channel();
        let mut scratch: Vec<f32> = Vec::new();
        let mut fill = move |len: usize| -> *const f32 {
            while let Ok(command) = inbox.try_recv() {
                mixer.apply(command);
            }
            scratch.resize(len, 0.0);
            mixer.render(&mut scratch, channels);
            scratch.as_ptr()
        };
        let report = |error| eprintln!("audio output: {error}");
        let stream = match format {
            SampleFormat::F32 => device.build_output_stream(
                &config,
                move |out: &mut [f32], _| {
                    let len = out.len();
                    // SAFETY: `fill` returns a buffer of `len` samples it keeps alive.
                    out.copy_from_slice(unsafe { std::slice::from_raw_parts(fill(len), len) });
                },
                report,
                None,
            ),
            SampleFormat::I16 => device.build_output_stream(
                &config,
                move |out: &mut [i16], _| {
                    let len = out.len();
                    // SAFETY: as above.
                    let mixed = unsafe { std::slice::from_raw_parts(fill(len), len) };
                    for (out, sample) in out.iter_mut().zip(mixed) {
                        *out = (*sample * i16::MAX as f32) as i16;
                    }
                },
                report,
                None,
            ),
            other => return Err(format!("unsupported output format {other:?}")),
        }
        .map_err(|e| e.to_string())?;
        stream.play().map_err(|e| e.to_string())?;
        Ok(Self {
            sink: Sink::Device {
                commands,
                _stream: stream,
            },
            next: 0,
            music: None,
        })
    }

    #[cfg(all(target_arch = "wasm32", feature = "web"))]
    pub(crate) fn web(mixer: std::rc::Rc<std::cell::RefCell<Mixer>>) -> Self {
        Self {
            sink: Sink::Web(mixer),
            next: 0,
            music: None,
        }
    }

    /// Send a command to the mixer.
    pub fn send(&mut self, command: Command) {
        match &mut self.sink {
            Sink::Offline(mixer) => mixer.apply(command),
            #[cfg(not(target_arch = "wasm32"))]
            Sink::Device { commands, .. } => {
                let _ = commands.send(command);
            }
            #[cfg(all(target_arch = "wasm32", feature = "web"))]
            Sink::Web(mixer) => mixer.borrow_mut().apply(command),
        }
    }

    /// Play a sound once at its own volume, not placed anywhere.
    pub fn play(&mut self, sound: &Audio) -> Option<VoiceId> {
        self.play_mixed(sound, sound.volume, 0.0, Bus::Effects, 0.0)
    }

    /// As [`play`](Self::play), on a [`Bus`] of your choosing.
    pub fn play_on(&mut self, bus: Bus, sound: &Audio) -> Option<VoiceId> {
        self.play_mixed(sound, sound.volume, 0.0, bus, 0.0)
    }

    /// Make `sound` the music, looping, crossfading from what was playing over
    /// `seconds`.
    pub fn play_music(&mut self, sound: &Audio, seconds: f32) -> Option<VoiceId> {
        self.stop_music(seconds);
        let mut sound = sound.clone();
        sound.loop_ = true;
        self.music = self.play_mixed(&sound, sound.volume, 0.0, Bus::Music, seconds);
        self.music
    }

    /// Fade the music out over `seconds`.
    pub fn stop_music(&mut self, seconds: f32) {
        if let Some(id) = self.music.take() {
            self.send(Command::FadeOut { id, seconds });
        }
    }

    /// One [`Bus`]'s volume: 1 is as mixed.
    pub fn set_bus_volume(&mut self, bus: Bus, volume: f32) {
        self.send(Command::BusVolume(bus, volume));
    }

    /// Change a playing sound's pitch: 1 is as recorded, 2 an octave up.
    pub fn set_pitch(&mut self, voice: VoiceId, factor: f32) {
        self.send(Command::Pitch { id: voice, factor });
    }

    /// Fade a playing sound to silence and stop it.
    pub fn fade_out(&mut self, voice: VoiceId, seconds: f32) {
        self.send(Command::FadeOut { id: voice, seconds });
    }

    pub(crate) fn play_mixed(
        &mut self,
        sound: &Audio,
        gain: f32,
        pan: f32,
        bus: Bus,
        fade_in: f32,
    ) -> Option<VoiceId> {
        let samples = sound.buffer.clone()?;
        let id = VoiceId(self.next);
        self.next += 1;
        self.send(Command::Play {
            id,
            samples,
            channels: sound.channels,
            rate: sound.sample_rate as f32 * sound.playback_rate,
            looping: sound.loop_,
            gain,
            pan,
            bus,
            fade_in,
        });
        Some(id)
    }

    /// Silence every voice, music included.
    pub fn stop_all(&mut self) {
        self.music = None;
        self.send(Command::StopAll);
    }

    pub fn stop(&mut self, voice: VoiceId) {
        self.send(Command::Stop(voice));
    }

    /// Overall volume.
    pub fn set_master(&mut self, volume: f32) {
        self.send(Command::Master(volume));
    }

    /// Render from an [`offline`](Self::offline) mixer, returning whether it is one.
    pub fn render(&mut self, out: &mut [f32], channels: usize) -> bool {
        match &mut self.sink {
            Sink::Offline(mixer) => {
                mixer.render(out, channels);
                true
            }
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    /// Voices playing in an [`offline`](Self::offline) mixer.
    pub fn offline_voices(&self) -> Option<usize> {
        match &self.sink {
            Sink::Offline(mixer) => Some(mixer.voices()),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }
}

/// A sound coming from an entity, heard from where the camera is.
pub struct Speaker {
    /// The sound and how it falls off. Set `audio.playing` to start it.
    pub source: PositionalAudio,
    /// How fast it plays, and so how high: 1 is as recorded. Change it at any time.
    pub pitch: f32,
    /// Which volume it answers to.
    pub bus: Bus,
    sent_pitch: f32,
    voice: Option<VoiceId>,
    /// When a sound that does not loop will have ended, in real time.
    ends: f64,
}

impl Component for Speaker {}

impl Speaker {
    /// A speaker that starts playing at once.
    pub fn new(sound: Audio) -> Self {
        let mut source = PositionalAudio {
            audio: sound,
            ..PositionalAudio::default()
        };
        source.audio.playing = true;
        Self {
            source,
            pitch: 1.0,
            bus: Bus::default(),
            sent_pitch: 1.0,
            voice: None,
            ends: 0.0,
        }
    }

    /// A speaker that plays round and round until told otherwise.
    pub fn looping(mut sound: Audio) -> Self {
        sound.loop_ = true;
        Self::new(sound)
    }

    pub fn is_playing(&self) -> bool {
        self.voice.is_some()
    }
}

/// Several sounds from one entity, each a [`Speaker`] of its own: an engine and
/// its tyres, a character's steps and voice. An entity holds one component of a
/// type, so this is how it holds more than one sound.
///
/// ```
/// # use threers_ecs::prelude::*;
/// let engine = Speaker::looping(waveform(70.0, 0.5, 0.5, &[0.5]));
/// let tyres = Speaker::looping(waveform(820.0, 0.4, 0.3, &[]));
/// let car = Speakers(vec![engine, tyres]);
/// assert_eq!(car.0.len(), 2);
/// ```
#[derive(Default)]
pub struct Speakers(pub Vec<Speaker>);
impl Component for Speakers {}

/// A [`Speaker`]'s sound played to its end.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoundFinished(pub Entity);

/// Where sound is heard from, as a resource. Follows [`MainCamera`] if there is one.
#[derive(Debug, Clone, Copy, Default)]
pub struct Listener(pub AudioListener);

/// How [`AudioPlugin`] finds its output.
#[derive(Default)]
pub enum AudioPlugin {
    /// The default device. If there is none the game runs silent, and says so.
    #[default]
    Device,
    /// No device, for tests.
    Offline,
}

impl Plugin for AudioPlugin {
    fn build(self, app: &mut App) {
        let out = match self {
            AudioPlugin::Offline => AudioOut::offline(44_100.0),
            #[cfg(not(target_arch = "wasm32"))]
            AudioPlugin::Device => AudioOut::open().unwrap_or_else(|error| {
                eprintln!("no sound: {error}");
                AudioOut::offline(44_100.0)
            }),
            #[cfg(target_arch = "wasm32")]
            AudioPlugin::Device => AudioOut::offline(44_100.0),
        };
        app.resource(out)
            .resource(Listener::default())
            .resource(WasPaused(false))
            .system(audio_step);
        app.world.on_remove::<Speaker>(|w, _, speaker| {
            if let Some(voice) = speaker.voice {
                w.resource_mut::<AudioOut>().stop(voice);
            }
        });
        app.world.on_remove::<Speakers>(|w, _, speakers| {
            for voice in speakers.0.iter().filter_map(|s| s.voice) {
                w.resource_mut::<AudioOut>().stop(voice);
            }
        });
    }
}

struct WasPaused(bool);

/// Start, stop and place sounds for this frame.
pub fn audio_step(w: &mut World) {
    let time = *w.resource::<Time>();
    if w.resource::<WasPaused>().0 != time.paused {
        w.resource_mut::<WasPaused>().0 = time.paused;
        w.resource_mut::<AudioOut>()
            .send(Command::Pause(time.paused));
    }
    if let Some(camera) = w.get_resource::<MainCamera>() {
        let (position, target, up) = (camera.0.position, camera.0.target, camera.0.up);
        let listener = &mut w.resource_mut::<Listener>().0;
        listener.position = position;
        listener.forward = (target - position).normalize();
        listener.up = up;
    }
    let listener = w.resource::<Listener>().0;

    let mut finished = Vec::new();
    w.with_resource::<AudioOut, _>(|w, out| {
        w.each::<(Entity, NodeRef, &mut Speaker)>(|(id, node, speaker)| {
            let mix = speaker.source.mix(&node.matrix_world, &listener);
            if step_speaker(out, speaker, mix, time.real_elapsed) {
                finished.push(SoundFinished(id));
            }
        });
        w.each::<(Entity, NodeRef, &mut Speakers)>(|(id, node, speakers)| {
            for speaker in &mut speakers.0 {
                let mix = speaker.source.mix(&node.matrix_world, &listener);
                if step_speaker(out, speaker, mix, time.real_elapsed) {
                    finished.push(SoundFinished(id));
                }
            }
        });
    });
    for event in finished {
        w.send(event);
    }
}

/// Start, stop, place or re-pitch one speaker's voice. Returns whether a sound
/// that does not loop came to its end.
fn step_speaker(
    out: &mut AudioOut,
    speaker: &mut Speaker,
    mix: threers::audio::SpatialMix,
    now: f64,
) -> bool {
    let wants = speaker.source.audio.playing;
    let looping = speaker.source.audio.loop_;
    match (speaker.voice, wants) {
        (None, true) => {
            let sound = &speaker.source.audio;
            let voice = out.play_mixed(sound, mix.gain, mix.pan, speaker.bus, 0.0);
            if let Some(voice) = voice.filter(|_| speaker.pitch != 1.0) {
                out.set_pitch(voice, speaker.pitch);
            }
            let length = sound.buffer.as_ref().map_or(0.0, |samples| {
                let frames = samples.len() / sound.channels.max(1) as usize;
                frames as f64 / (sound.sample_rate as f64 * sound.playback_rate as f64)
            });
            speaker.voice = voice;
            speaker.ends = now + length / speaker.pitch.max(0.01) as f64;
            speaker.sent_pitch = speaker.pitch;
            false
        }
        (Some(voice), false) => {
            out.stop(voice);
            speaker.voice = None;
            false
        }
        (Some(_), true) if !looping && now >= speaker.ends => {
            speaker.voice = None;
            speaker.source.audio.playing = false;
            true
        }
        (Some(voice), true) => {
            out.send(Command::Mix {
                id: voice,
                gain: mix.gain,
                pan: mix.pan,
            });
            if speaker.pitch != speaker.sent_pitch {
                out.set_pitch(voice, speaker.pitch);
                speaker.sent_pitch = speaker.pitch;
            }
            false
        }
        (None, false) => false,
    }
}
