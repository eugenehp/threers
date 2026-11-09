/// Convenience helper computing FFT-band magnitudes from raw samples. Mirrors
/// three.js's `AudioAnalyser` for the things downstream code uses (visualizer
/// bar heights, beat detection, etc.). The actual FFT is left to the caller —
/// this struct just holds the result vector and provides convenience accessors.
#[derive(Debug, Clone, Default)]
pub struct AudioAnalyser {
    pub fft_size: u32,
    pub frequency_data: Vec<f32>,
}

impl AudioAnalyser {
    pub fn new(fft_size: u32) -> Self {
        Self {
            fft_size,
            frequency_data: vec![0.0; (fft_size / 2) as usize],
        }
    }

    pub fn average_frequency(&self) -> f32 {
        if self.frequency_data.is_empty() {
            return 0.0;
        }
        self.frequency_data.iter().sum::<f32>() / self.frequency_data.len() as f32
    }
}
