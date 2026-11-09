use std::sync::Arc;

/// Non-positional audio source. Holds shared PCM data and playback parameters.
#[derive(Debug, Clone)]
pub struct Audio {
    pub buffer: Option<Arc<Vec<f32>>>,
    pub sample_rate: u32,
    pub channels: u16,
    pub loop_: bool,
    pub volume: f32,
    pub playback_rate: f32,
    pub playing: bool,
}

impl Default for Audio {
    fn default() -> Self {
        Self {
            buffer: None,
            sample_rate: 44_100,
            channels: 2,
            loop_: false,
            volume: 1.0,
            playback_rate: 1.0,
            playing: false,
        }
    }
}

impl Audio {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_buffer(&mut self, buffer: Arc<Vec<f32>>) -> &mut Self {
        self.buffer = Some(buffer);
        self
    }
}
