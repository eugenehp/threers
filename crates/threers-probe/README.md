# threers-probe

Screen-space neural GI from a G-buffer and a cheap lighting probe. The path tracer
is the teacher; RLX runs the same graph on Metal, CUDA, or CPU (browser-ready).

**Stack (22 input planes):**

- probe RGB, albedo, normal, depth, demodulated irradiance
- quarter-res probe, DDGI-lite world bins, encoded hit position
- **Hybrid head** (default): 7×7 kernel gather + shadow-gated irradiance residual
- **Hops head** (`ARCH=hops`): five à-trous light jumps in linear light (dilations 1–16), normal-gated cardinals/diagonals, global DDGI/quarter-res/irradiance taps, probe-gated residual, optional NRC — ~1.7k parameters
- **NRC** (optional): tiny world-space MLP fused on gated pixels

## Proof of concept

```sh
cargo test -p threers-probe
GPU=1 cargo run --release -p threers-probe --example probe --features generate,metal
GPU=1 ARCH=hops cargo run --release -p threers-probe --example probe --features generate,metal
```

Trains on a mixed family (enclosed, Cornell-like, open), scores a hostile
hold-out (canonical Cornell, glass Cornell, open floor, furnace), and writes:

| output | |
|---|---|
| `out/probe.bin` | U-Net / hybrid weights (`THRSW003`) |
| `out/probe_hops.bin` | hops-head weights (`ARCH=hops`) |
| `out/nrc_hops.bin` | NRC on hops (`ARCH=hops`) |
| `out/nrc.bin` | NRC MLP weights (`THRSN001`) |
| `out/probe/*.png` | probe / U-Net / fused / reference previews |

Load at runtime:

```rust
let mut gi = ProbeGi::load("out/probe.bin", Some("out/nrc.bin"), h, w, device)?;
let beauty = gi.reconstruct(&packed_input)?;
```

## Environment

| env | default | |
|---|---|---|
| `SCENES` | 40 (60 for `ARCH=hops`) | family scenes (last quarter = val) |
| `FRAME` | 64 | square resolution |
| `TILE` | 32 | training tile, multiple of 8 |
| `ARCH` | hybrid | `hybrid`, `gathering`, `direct`, or `hops` |
| `PROBE` | 8 | probe samples per pixel |
| `PROBE_BOUNCES` | 3 | probe path depth |
| `REF` | 64 | reference spp |
| `EPOCHS` | 50 (100 for `ARCH=hops`) | U-Net / hops epochs |
| `NRC_EPOCHS` | 30 (40 for `ARCH=hops`) | NRC epochs (enclosed/Cornell only) |
| `BATCH` | 2 | |
| `GPU` | 0 | `1` → path-tracer GPU backend |
| `LOAD` | 0 | `1` scores saved weights without retraining |

NRC trains on `(reference − U-Net) / albedo` and fuses only where: finite depth,
albedo ≥ 0.12, compressed probe < 0.72, and probe differs from the U-Net output
(relative L² ≥ 0.04). The U-Net residual is multiplied by `(1 − probe)` so
bright surfaces stay near the gather.

## Hostile hold-out (hops head, fused +NRC)

Lower % = closer to reference vs copying the probe alone.

| scene | probe | hops | +NRC |
|---|---:|---:|---:|
| Cornell | 100% | 64% | **63%** |
| Cornell glass | 100% | 77% | **77%** |
| Open | 100% | 100% | 100% |
| Furnace | 100% | 100% | 100% |
| **All hostile** | 100% | 72% | **72%** |

Room-context DDGI tint (`GRAPH_ROOM_TINT = 0.22`) runs **inside the hop graph** on neutral albedo so training and inference match.

Hybrid baseline (~28k params) is ~74% all hostile. The hops stack is ~1.7k params + NRC.

MIT, like threers.
