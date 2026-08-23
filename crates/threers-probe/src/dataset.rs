//! Training tiles on disk.
//!
//! ```text
//! magic   "THRSP001"   8 bytes
//! tile    u32          side length, a multiple of 8
//! count   u32          number of tiles
//! inputs  u32          input planes per tile
//! outputs u32          target planes per tile
//! tiles   f32 × count × (inputs + outputs) × tile × tile   LE, planar
//! ```

use anyhow::{bail, ensure, Result};
use std::path::Path;

const MAGIC: &[u8; 8] = b"THRSP001";

/// A set of training tiles, held in memory.
pub struct Dataset {
    tile: usize,
    count: usize,
    inputs: usize,
    outputs: usize,
    data: Vec<f32>,
}

impl Dataset {
    pub fn new(tile: usize, inputs: usize, outputs: usize, data: Vec<f32>) -> Result<Self> {
        ensure!(
            tile > 0 && tile.is_multiple_of(8),
            "tile {tile} must be a positive multiple of 8"
        );
        ensure!(inputs > 0 && outputs > 0, "a tile cannot have {inputs}+{outputs} planes");
        let per = (inputs + outputs) * tile * tile;
        ensure!(
            data.len().is_multiple_of(per),
            "data length {} is not a multiple of {per}",
            data.len()
        );
        let count = data.len() / per;
        if let Some(bad) = data.iter().position(|v| !v.is_finite()) {
            bail!(
                "dataset holds {} at tile {}, plane {}",
                data[bad],
                bad / per,
                (bad % per) / (tile * tile)
            );
        }
        Ok(Self {
            tile,
            count,
            inputs,
            outputs,
            data,
        })
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|e| anyhow::anyhow!("probe: cannot read {}: {e}", path.display()))?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        ensure!(bytes.len() >= 24, "dataset is {} bytes, too short", bytes.len());
        if &bytes[..8] != MAGIC {
            bail!(
                "dataset does not start with {}",
                String::from_utf8_lossy(MAGIC)
            );
        }
        let tile = u32::from_le_bytes(bytes[8..12].try_into()?) as usize;
        let count = u32::from_le_bytes(bytes[12..16].try_into()?) as usize;
        let inputs = u32::from_le_bytes(bytes[16..20].try_into()?) as usize;
        let outputs = u32::from_le_bytes(bytes[20..24].try_into()?) as usize;
        let per = (inputs + outputs) * tile * tile;
        let want = count * per;
        let floats = (bytes.len() - 24) / 4;
        ensure!(
            floats == want,
            "dataset holds {floats} floats, {count} tiles of {tile} with \
             {inputs}+{outputs} planes need {want}"
        );
        let mut data = Vec::with_capacity(want);
        for chunk in bytes[24..].chunks_exact(4) {
            data.push(f32::from_le_bytes(chunk.try_into()?));
        }
        Self::new(tile, inputs, outputs, data)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut bytes = Vec::with_capacity(24 + self.data.len() * 4);
        bytes.extend_from_slice(MAGIC);
        for v in [self.tile, self.count, self.inputs, self.outputs] {
            bytes.extend_from_slice(&(v as u32).to_le_bytes());
        }
        for v in &self.data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(path.as_ref(), bytes)
            .map_err(|e| anyhow::anyhow!("probe: cannot write {}: {e}", path.as_ref().display()))?;
        Ok(())
    }

    pub fn check_shape(&self, inputs: usize, outputs: usize) -> Result<()> {
        ensure!(
            self.inputs == inputs && self.outputs == outputs,
            "dataset holds {}+{} planes, this network wants {inputs}+{outputs}",
            self.inputs,
            self.outputs
        );
        Ok(())
    }

    pub fn tile(&self) -> usize {
        self.tile
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn inputs(&self) -> usize {
        self.inputs
    }

    pub fn outputs(&self) -> usize {
        self.outputs
    }

    /// Gather `indices` into contiguous input and target buffers.
    pub fn gather(&self, indices: &[usize]) -> Result<(Vec<f32>, Vec<f32>)> {
        let per = (self.inputs + self.outputs) * self.tile * self.tile;
        let in_plane = self.inputs * self.tile * self.tile;
        let mut input = Vec::with_capacity(indices.len() * in_plane);
        let mut target = Vec::with_capacity(indices.len() * (per - in_plane));
        for &idx in indices {
            ensure!(idx < self.count, "tile {idx} is past the set ({})", self.count);
            let base = idx * per;
            input.extend_from_slice(&self.data[base..base + in_plane]);
            target.extend_from_slice(&self.data[base + in_plane..base + per]);
        }
        Ok((input, target))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IN_CHANNELS, OUT_CHANNELS};

    #[test]
    fn a_round_trip_preserves_tiles() {
        let tile = 8;
        let per = (IN_CHANNELS + OUT_CHANNELS) * tile * tile;
        let mut data = vec![0.0f32; 2 * per];
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f32 * 0.01;
        }
        let set = Dataset::new(tile, IN_CHANNELS, OUT_CHANNELS, data.clone()).unwrap();
        let dir = std::env::temp_dir().join("threers-probe-dataset.bin");
        set.save(&dir).unwrap();
        let loaded = Dataset::open(&dir).unwrap();
        let _ = std::fs::remove_file(&dir);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.tile(), 8);
        let (input, target) = loaded.gather(&[1]).unwrap();
        assert_eq!(input.len(), IN_CHANNELS * tile * tile);
        assert_eq!(target.len(), OUT_CHANNELS * tile * tile);
        assert_eq!(input[0], data[per]);
    }

    #[test]
    fn a_nan_is_refused() {
        let tile = 8;
        let per = (IN_CHANNELS + OUT_CHANNELS) * tile * tile;
        let mut data = vec![0.0f32; per];
        data[3] = f32::NAN;
        assert!(Dataset::new(tile, IN_CHANNELS, OUT_CHANNELS, data).is_err());
    }
}
