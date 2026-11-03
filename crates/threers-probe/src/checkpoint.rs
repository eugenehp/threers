//! Trained weights on disk.
//!
//! ```text
//! magic    "THRSW003"   8 bytes
//! inputs   u32
//! widths   u32 × 4
//! head     u32          0 = Direct, 1 = Kernel, 2 = Hybrid, 3 = Hops
//! radius   u32          0 if Direct, kernel radius, or hop count
//! cap      f32          hybrid irradiance cap (0 → default)
//! count    u32          number of parameter tensors
//! per tensor: elems u32, then that many f32
//! ```
//!
//! `THRSW002` is still readable (no cap field). `THRSW001` → Direct head.

use anyhow::{bail, ensure, Result};
use std::path::Path;

use crate::model::{Head, ProbeArch, Widths, HYBRID_IRRADIANCE_CAP, IN_CHANNELS};

const MAGIC_V1: &[u8; 8] = b"THRSW001";
const MAGIC_V2: &[u8; 8] = b"THRSW002";
const MAGIC_V3: &[u8; 8] = b"THRSW003";

pub fn save(net: &ProbeArch, params: &[Vec<f32>], path: impl AsRef<Path>) -> Result<()> {
    ensure!(
        params.len() == net.params().len(),
        "{} tensors for a network with {}",
        params.len(),
        net.params().len()
    );
    let w = net.widths();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC_V3);
    bytes.extend_from_slice(&(IN_CHANNELS as u32).to_le_bytes());
    for v in [w.level0, w.level1, w.level2, w.level3] {
        bytes.extend_from_slice(&(v as u32).to_le_bytes());
    }
    bytes.extend_from_slice(&net.head().code().to_le_bytes());
    bytes.extend_from_slice(&net.head().radius().to_le_bytes());
    bytes.extend_from_slice(&net.head().cap().to_le_bytes());
    bytes.extend_from_slice(&(params.len() as u32).to_le_bytes());
    for (spec, values) in net.params().iter().zip(params) {
        ensure!(
            values.len() == spec.elems(),
            "{}: {} values for a {:?} tensor",
            spec.name,
            values.len(),
            spec.shape
        );
        bytes.extend_from_slice(&(values.len() as u32).to_le_bytes());
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path.as_ref(), bytes)
        .map_err(|e| anyhow::anyhow!("probe: cannot write {}: {e}", path.as_ref().display()))?;
    Ok(())
}

pub fn load(path: impl AsRef<Path>) -> Result<(ProbeArch, Vec<Vec<f32>>)> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|e| anyhow::anyhow!("probe: cannot read {}: {e}", path.as_ref().display()))?;
    load_bytes(&bytes)
}

pub fn load_bytes(bytes: &[u8]) -> Result<(ProbeArch, Vec<Vec<f32>>)> {
    ensure!(bytes.len() >= 32, "weights file is {} bytes, too short", bytes.len());
    let (head, widths, count, mut off) = match &bytes[..8] {
        m if m == MAGIC_V1 => {
            let inputs = u32::from_le_bytes(bytes[8..12].try_into()?) as usize;
            ensure!(
                inputs == IN_CHANNELS,
                "weights take {inputs} planes, this crate expects {IN_CHANNELS}"
            );
            let widths = read_widths(&bytes[12..28])?;
            let count = u32::from_le_bytes(bytes[28..32].try_into()?) as usize;
            (Head::Direct, widths, count, 32usize)
        }
        m if m == MAGIC_V2 => {
            ensure!(bytes.len() >= 40, "THRSW002 is {} bytes, too short", bytes.len());
            let inputs = u32::from_le_bytes(bytes[8..12].try_into()?) as usize;
            ensure!(
                inputs == IN_CHANNELS,
                "weights take {inputs} planes, this crate expects {IN_CHANNELS}"
            );
            let widths = read_widths(&bytes[12..28])?;
            let code = u32::from_le_bytes(bytes[28..32].try_into()?);
            let radius = u32::from_le_bytes(bytes[32..36].try_into()?);
            let head = Head::from_code(code, radius, HYBRID_IRRADIANCE_CAP).ok_or_else(|| {
                anyhow::anyhow!("unknown probe head code {code} radius {radius}")
            })?;
            let count = u32::from_le_bytes(bytes[36..40].try_into()?) as usize;
            (head, widths, count, 40usize)
        }
        m if m == MAGIC_V3 => {
            ensure!(bytes.len() >= 44, "THRSW003 is {} bytes, too short", bytes.len());
            let inputs = u32::from_le_bytes(bytes[8..12].try_into()?) as usize;
            ensure!(
                inputs == IN_CHANNELS,
                "weights take {inputs} planes, this crate expects {IN_CHANNELS}"
            );
            let widths = read_widths(&bytes[12..28])?;
            let code = u32::from_le_bytes(bytes[28..32].try_into()?);
            let radius = u32::from_le_bytes(bytes[32..36].try_into()?);
            let cap = f32::from_le_bytes(bytes[36..40].try_into()?);
            let head = Head::from_code(code, radius, cap).ok_or_else(|| {
                anyhow::anyhow!("unknown probe head code {code} radius {radius} cap {cap}")
            })?;
            let count = u32::from_le_bytes(bytes[40..44].try_into()?) as usize;
            (head, widths, count, 44usize)
        }
        _ => bail!(
            "weights do not start with {}, {}, or {}",
            String::from_utf8_lossy(MAGIC_V3),
            String::from_utf8_lossy(MAGIC_V2),
            String::from_utf8_lossy(MAGIC_V1)
        ),
    };
    let net = ProbeArch::with_head(widths, head);
    ensure!(
        count == net.params().len(),
        "file has {count} tensors, network has {}",
        net.params().len()
    );
    let mut params = Vec::with_capacity(count);
    for spec in net.params() {
        ensure!(off + 4 <= bytes.len(), "truncated weights at {}", spec.name);
        let elems = u32::from_le_bytes(bytes[off..off + 4].try_into()?) as usize;
        off += 4;
        ensure!(
            elems == spec.elems(),
            "{}: file has {elems} values, shape {:?} wants {}",
            spec.name,
            spec.shape,
            spec.elems()
        );
        ensure!(off + elems * 4 <= bytes.len(), "truncated values at {}", spec.name);
        let mut values = Vec::with_capacity(elems);
        for _ in 0..elems {
            values.push(f32::from_le_bytes(bytes[off..off + 4].try_into()?));
            off += 4;
        }
        params.push(values);
    }
    Ok((net, params))
}

fn read_widths(bytes: &[u8]) -> Result<Widths> {
    ensure!(bytes.len() == 16, "widths header is {} bytes", bytes.len());
    Ok(Widths {
        level0: u32::from_le_bytes(bytes[0..4].try_into()?) as usize,
        level1: u32::from_le_bytes(bytes[4..8].try_into()?) as usize,
        level2: u32::from_le_bytes(bytes[8..12].try_into()?) as usize,
        level3: u32::from_le_bytes(bytes[12..16].try_into()?) as usize,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{init_params, Widths};

    #[test]
    fn weights_round_trip() {
        for net in [
            ProbeArch::new(Widths::tiny()),
            ProbeArch::gathering(Widths::tiny()),
            ProbeArch::hybrid(Widths::tiny()),
            ProbeArch::hops(),
        ] {
            let params = init_params(&net, 42);
            let path = std::env::temp_dir().join(format!(
                "threers-probe-weights-{:?}.bin",
                net.head()
            ));
            save(&net, &params, &path).unwrap();
            let (loaded, values) = load(&path).unwrap();
            let _ = std::fs::remove_file(&path);
            assert_eq!(loaded.widths(), net.widths());
            assert_eq!(loaded.head(), net.head());
            assert_eq!(values, params);
        }
    }
}
