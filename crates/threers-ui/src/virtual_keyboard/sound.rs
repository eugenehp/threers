//! Key clicks out of the default audio device (the `sound` feature).

use std::sync::mpsc::{channel, Receiver, Sender};

use super::feedback::KeySound;

/// Plays [`KeySound`]s on the default output device, mixed so fast typing
/// overlaps its clicks.
pub struct ClickPlayer {
    clicks: Sender<(KeySound, f32)>,
    _stream: cpal::Stream,
}

/// The device's side: clicks made once at its rate, and those sounding.
struct Mix {
    inbox: Receiver<(KeySound, f32)>,
    sounds: [Vec<f32>; 3],
    /// Which sound, how far into it, how loud.
    playing: Vec<(usize, usize, f32)>,
    channels: usize,
}

impl Mix {
    fn fill(&mut self, out: &mut [f32]) {
        while let Ok((s, volume)) = self.inbox.try_recv() {
            let i = match s {
                KeySound::Letter => 0,
                KeySound::Delete => 1,
                KeySound::Modifier => 2,
            };
            self.playing.push((i, 0, volume.clamp(0.0, 1.0)));
        }
        for frame in out.chunks_mut(self.channels.max(1)) {
            let mut v = 0.0;
            for (i, at, volume) in &mut self.playing {
                if let Some(s) = self.sounds[*i].get(*at) {
                    v += s * *volume;
                }
                *at += 1;
            }
            frame.fill(v.clamp(-1.0, 1.0));
        }
        let sounds = &self.sounds;
        self.playing.retain(|(i, at, _)| *at < sounds[*i].len());
    }
}

impl ClickPlayer {
    /// On the default output device.
    ///
    /// # Errors
    ///
    /// If there is none, or it cannot be opened.
    pub fn open() -> Result<Self, String> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
        use cpal::SampleFormat;

        let device = cpal::default_host()
            .default_output_device()
            .ok_or("no audio output device")?;
        let config = device.default_output_config().map_err(|e| e.to_string())?;
        let format = config.sample_format();
        let config: cpal::StreamConfig = config.into();
        let rate = config.sample_rate;
        let (clicks, inbox) = channel();
        let mut mix = Mix {
            inbox,
            sounds: [KeySound::Letter, KeySound::Delete, KeySound::Modifier]
                .map(|s| s.samples(rate)),
            playing: vec![],
            channels: config.channels as usize,
        };
        let report = |error| eprintln!("key clicks: {error}");
        let stream = match format {
            SampleFormat::F32 => device.build_output_stream(
                config,
                move |out: &mut [f32], _| mix.fill(out),
                report,
                None,
            ),
            SampleFormat::I16 => {
                let mut scratch = vec![];
                device.build_output_stream(
                    config,
                    move |out: &mut [i16], _| {
                        scratch.resize(out.len(), 0.0);
                        mix.fill(&mut scratch);
                        for (o, s) in out.iter_mut().zip(&scratch) {
                            *o = (s * i16::MAX as f32) as i16;
                        }
                    },
                    report,
                    None,
                )
            }
            other => return Err(format!("unsupported output format {other:?}")),
        }
        .map_err(|e| e.to_string())?;
        stream.play().map_err(|e| e.to_string())?;
        Ok(Self {
            clicks,
            _stream: stream,
        })
    }

    /// Plays `sound` at `volume` (0 to 1).
    pub fn play(&self, sound: KeySound, volume: f32) {
        let _ = self.clicks.send((sound, volume));
    }
}
