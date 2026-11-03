//! Train hybrid KPN + DDGI + NRC on a mixed family; score hostile hold-outs.
//!
//! ```sh
//! GPU=1 cargo run --release -p threers-probe --example probe --features generate,metal
//! GPU=1 ARCH=hops cargo run --release -p threers-probe --example probe --features generate,metal
//! ```

use std::time::Instant;

use threers::raytrace::{RaytraceSettings, RtCamera};
use threers::{encode_png, ToneMapping};
use threers_probe::checkpoint;
use threers_probe::generate::{self, Built};
use threers_probe::model::{Head, ProbeArch};
use threers_probe::pack::{self, PlaneExtras};
use threers_probe::{
    colour_planes, init_nrc_params, pack_nrc_batch_with, preferred_device, save_nrc, train_nrc_dataset, Batch, Dataset,
    HOPS_GI_LOSS_BOOST, NrcArch, NrcNet, ProbeGi, ProbeNet, TrainConfig, Trainer, Widths, GI_LOSS_BOOST,
    IN_CHANNELS, LOSS_EPSILON, NRC_MIN_ALBEDO, NRC_OUT, NRC_PROBE_CONFIDENCE, OUT_CHANNELS,
    HOPS_NRC_PROBE_CONFIDENCE, HOPS_NRC_CAP, nrc_input_channels,
};

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_arch() -> ProbeArch {
    match std::env::var("ARCH")
        .ok()
        .unwrap_or_else(|| "hybrid".into())
        .to_ascii_lowercase()
        .as_str()
    {
        "hops" | "jump" | "propagate" => ProbeArch::hops(),
        "direct" => ProbeArch::new(Widths::tiny()),
        "gathering" | "kernel" => ProbeArch::gathering(Widths::tiny()),
        _ => ProbeArch::hybrid(Widths::tiny()),
    }
}

fn nrc_path(hops: bool) -> Option<&'static str> {
    if hops {
        let p = "out/nrc_hops.bin";
        if std::path::Path::new(p).exists() {
            Some(p)
        } else {
            None
        }
    } else {
        Some("out/nrc.bin")
    }
}

fn weights_path(hops: bool) -> String {
    if let Ok(p) = std::env::var("WEIGHTS") {
        if !p.is_empty() {
            return p;
        }
    }
    if hops {
        "out/probe_hops.bin".into()
    } else {
        "out/probe.bin".into()
    }
}

fn nrc_save_path(hops: bool) -> String {
    if let Ok(p) = std::env::var("NRC_OUT") {
        if !p.is_empty() {
            return p;
        }
    }
    if hops {
        "out/nrc_hops.bin".into()
    } else {
        "out/nrc.bin".into()
    }
}

fn main() {
    let frame = env_u32("FRAME", 64);
    let tile = env_u32("TILE", 32) as usize;
    let batch_n = env_u32("BATCH", 2) as usize;
    assert!(
        (frame as usize).is_multiple_of(tile) && tile.is_multiple_of(8),
        "FRAME={frame} must be a multiple of TILE={tile}, and TILE a multiple of 8"
    );
    let net = env_arch();
    let hops = net.head().is_hops();
    let scenes = env_u32("SCENES", if hops { 60 } else { 40 });
    let probe_spp = env_u32("PROBE", 8);
    let probe_bounces = env_u32("PROBE_BOUNCES", 3);
    let reference_spp = env_u32("REF", 64);
    let epochs = env_u32("EPOCHS", if hops { 100 } else { 50 }) as usize;
    let nrc_epochs = env_u32("NRC_EPOCHS", if hops { 40 } else { 30 }) as usize;
    let head_label = match net.head() {
        Head::Direct => "direct",
        Head::Kernel { .. } => "gathering 7×7",
        Head::Hybrid { .. } => "hybrid 7×7+cap",
        Head::Hops { .. } => "hops 1-2-4-8-16+cap",
    };
    let device = preferred_device();
    let mut rt = make_renderer(frame, frame);
    println!(
        "threers-probe   {head_label}   {IN_CHANNELS} planes   {frame}²  tile {tile}  {scenes} family scenes  probe {probe_spp} spp × {probe_bounces} bounce  ref {reference_spp} spp  NRC {nrc_epochs} ep"
    );
    println!(
        "device {device:?}  {} params  backend {}",
        net.parameter_count(),
        rt.backend_name()
    );

    let nrc_gap = env_f32(
        "NRC_GAP",
        if hops {
            HOPS_NRC_PROBE_CONFIDENCE
        } else {
            NRC_PROBE_CONFIDENCE
        },
    );
    let nrc_only = env_u32("NRC_ONLY", 0) == 1;
    let side = frame as usize;
    if env_u32("LOAD", 0) == 1 && !nrc_only {
        let mut stack = ProbeGi::load_with_nrc_fuse(
            weights_path(hops),
            nrc_path(hops),
            side,
            side,
            device,
            nrc_gap,
            if hops { HOPS_NRC_CAP } else { threers_probe::NRC_CAP },
        )
        .expect("load");
        score_hostile(&mut rt, &mut stack, frame, probe_spp, probe_bounces, reference_spp, side);
        return;
    }

    let t0 = Instant::now();
    let mut tiles = Vec::new();
    let mut frames = Vec::new();
    for index in 0..scenes {
        let packed = render_pair(
            &mut rt,
            generate::family(index),
            frame,
            probe_spp,
            probe_bounces,
            reference_spp,
            index as u64,
        );
        println!("  family {:>3}/{scenes}  {}", index + 1, packed.kind);
        frames.push(packed.clone());
        tiles.extend(
            pack::tiles(
                &packed.input,
                &packed.target,
                frame as usize,
                frame as usize,
                tile,
            )
            .expect("tile"),
        );
    }
    println!("family rendered in {:.1}s", t0.elapsed().as_secs_f32());

    let per_scene = ((frame as usize) / tile) * ((frame as usize) / tile);
    let val_scenes = (scenes / 4).max(1);
    let train_scenes = scenes - val_scenes;

    if nrc_only {
        let (net_loaded, params) = checkpoint::load(weights_path(hops)).expect("load hops for nrc-only");
        let infer_frame =
            ProbeNet::new(net_loaded, params, 1, side, side, device).expect("infer");
        let mut stack = build_nrc_stack(
            infer_frame,
            &frames,
            train_scenes,
            side,
            tile,
            batch_n,
            device,
            nrc_epochs,
            nrc_gap,
            hops,
        );
        score_hostile(
            &mut rt,
            &mut stack,
            frame,
            probe_spp,
            probe_bounces,
            reference_spp,
            side,
        );
        return;
    }

    let train_count = train_scenes as usize * per_scene;
    let split = train_count * (IN_CHANNELS + OUT_CHANNELS) * tile * tile;
    let train =
        Dataset::new(tile, IN_CHANNELS, OUT_CHANNELS, tiles[..split].to_vec()).expect("train");
    let val = Dataset::new(tile, IN_CHANNELS, OUT_CHANNELS, tiles[split..].to_vec()).expect("val");
    let _ = std::fs::create_dir_all("out");
    train.save("out/probe_train.bin").expect("write train");
    val.save("out/probe_val.bin").expect("write val");
    println!(
        "tiles  train {}  val {}  ({train_scenes} / {val_scenes} family scenes)",
        train.len(),
        val.len()
    );

    let steps = (train.len() / batch_n) * epochs;
    let mut trainer = Trainer::new(
        net,
        batch_n,
        tile,
        tile,
        device,
        TrainConfig {
            learning_rate: if hops { 8e-4 } else { 5e-4 },
            gi_loss_boost: if hops { HOPS_GI_LOSS_BOOST } else { GI_LOSS_BOOST },
            chroma_loss_boost: 0.0,
            neutral_loss_boost: if hops {
                env_f32("NEUTRAL_LOSS", 0.0)
            } else {
                0.0
            },
            total_steps: steps as u32,
            ..Default::default()
        },
    )
    .expect("compile trainer");

    let probe_err = eval_probe_baseline(&val);
    println!("\nfamily val, copy probe   {probe_err:.5}");

    let mut indices: Vec<usize> = (0..train.len()).collect();
    let mut rng = Shuffle(0x9E37_79B9_7F4A_7C15);
    let t1 = Instant::now();
    for epoch in 1..=epochs {
        shuffle(&mut indices, &mut rng);
        let mut sum = 0.0f64;
        let mut n = 0usize;
        for chunk in indices.chunks(batch_n) {
            if chunk.len() != batch_n {
                break;
            }
            let (input, target) = train.gather(chunk).expect("gather");
            let loss = trainer
                .step(&Batch {
                    n: batch_n,
                    h: tile,
                    w: tile,
                    input: &input,
                    target: &target,
                })
                .expect("step");
            sum += loss as f64;
            n += 1;
        }
        if epoch == 1 || epoch == epochs || epoch % 10 == 0 {
            let val_err = eval_net(trainer.net(), trainer.params(), tile, device, &val);
            println!(
                "  epoch {epoch:>3}/{epochs}  train {:.5}  val {val_err:.5}",
                (sum / n.max(1) as f64).sqrt()
            );
        }
    }
    println!("network trained in {:.1}s", t1.elapsed().as_secs_f32());
    let _ = std::fs::create_dir_all("out");
    checkpoint::save(trainer.net(), trainer.params(), weights_path(hops)).expect("save weights");

    let side = frame as usize;
    let infer_frame = ProbeNet::new(
        trainer.net().clone(),
        trainer.params().to_vec(),
        1,
        side,
        side,
        device,
    )
    .expect("infer frame");

    let trained_err = eval_net(trainer.net(), trainer.params(), tile, device, &val);
    println!("\nfamily validation relative-L2");
    report("copy probe", probe_err, probe_err);
    report(head_label, trained_err, probe_err);

    let mut stack = if nrc_epochs == 0 {
        ProbeGi::new(infer_frame, None, side, side)
    } else {
        build_nrc_stack(
            infer_frame,
            &frames,
            train_scenes,
            side,
            tile,
            batch_n,
            device,
            nrc_epochs,
            nrc_gap,
            hops,
        )
    };

    score_hostile(
        &mut rt,
        &mut stack,
        frame,
        probe_spp,
        probe_bounces,
        reference_spp,
        side,
    );

    let val_start = train_scenes as usize;
    for (k, packed) in frames[val_start..].iter().enumerate() {
        let pred = stack
            .unet_mut()
            .reconstruct(&packed.input, side, side)
            .expect("reconstruct");
        write_preview(
            packed.frame,
            &format!("out/probe/val_{k}_probe.png"),
            &packed.input,
        );
        write_preview(packed.frame, &format!("out/probe/val_{k}_pred.png"), &pred);
        write_preview(
            packed.frame,
            &format!("out/probe/val_{k}_ref.png"),
            &packed.target,
        );
    }
    if nrc_epochs == 0 {
        println!("\nwrote {} and out/probe/*.png", weights_path(hops));
    } else {
        println!(
            "\nwrote {}, {}, and out/probe/*.png",
            weights_path(hops),
            nrc_save_path(hops)
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn build_nrc_stack(
    mut infer_frame: ProbeNet,
    frames: &[Packed],
    train_scenes: u32,
    side: usize,
    tile: usize,
    batch_n: usize,
    device: rlx::Device,
    nrc_epochs: usize,
    nrc_gap: f32,
    hops: bool,
) -> ProbeGi {
    let nrc_in = nrc_input_channels();
    let mut nrc_tiles = Vec::new();
    let mut nrc_scenes = 0u32;
    for packed in &frames[..train_scenes as usize] {
        if !generate::nrc_trainable_kind(&packed.kind) {
            continue;
        }
        nrc_scenes += 1;
        let pixels = packed.pixels();
        let unet_pred = infer_frame
            .reconstruct(&packed.input, side, side)
            .expect("unet for nrc pack");
        let (nrc_in_buf, nrc_tgt) =
            pack_nrc_batch_with(&packed.input, &unet_pred, &packed.target, pixels, nrc_gap);
        nrc_tiles.extend(
            pack::tiles_with(&nrc_in_buf, &nrc_tgt, nrc_in, NRC_OUT, side, side, tile)
                .expect("nrc tile"),
        );
    }
    println!(
        "NRC train  {nrc_scenes} enclosed/cornell scenes  in={nrc_in}  residual=(ref−unet)  albedo>={NRC_MIN_ALBEDO}  probe-gap>={nrc_gap}",
    );
    let nrc_arch = NrcArch::new();
    let nrc_train = Dataset::new(tile, nrc_in, NRC_OUT, nrc_tiles).expect("nrc train");
    let nrc_steps = (nrc_train.len().max(1) / batch_n.max(1)) * nrc_epochs;
    let nrc_params = if nrc_train.is_empty() {
        eprintln!("warning: no NRC training tiles (all open in train split); using zero init");
        init_nrc_params(&nrc_arch, 0x4E52_4301)
    } else {
        train_nrc_dataset(
            &nrc_arch,
            &nrc_train,
            batch_n,
            device,
            TrainConfig {
                learning_rate: 1e-3,
                total_steps: nrc_steps as u32,
                ..Default::default()
            },
            nrc_epochs,
            0x4E52_4301,
        )
        .expect("train nrc")
    };
    save_nrc(&nrc_arch, &nrc_params, nrc_save_path(hops)).expect("save nrc");
    let mut nrc_net =
        NrcNet::new(nrc_arch.clone(), nrc_params, 1, side, side, device).expect("nrc infer");
    let online_epochs = env_u32("ONLINE_NRC", if hops { 12 } else { 0 }) as usize;
    if online_epochs > 0 {
        let mut online_scenes = Vec::new();
        for packed in &frames[..train_scenes as usize] {
            if !generate::nrc_trainable_kind(&packed.kind) {
                continue;
            }
            let unet_pred = infer_frame
                .reconstruct(&packed.input, side, side)
                .expect("unet for online nrc");
            online_scenes.push((packed.input.clone(), unet_pred, packed.target.clone()));
        }
        println!(
            "NRC online  {online_epochs} ep/scene × {} enclosed scenes",
            online_scenes.len()
        );
        nrc_net
            .finetune_scenes(
                device,
                side,
                online_epochs,
                TrainConfig {
                    learning_rate: 5e-4,
                    total_steps: (online_scenes.len() * online_epochs) as u32,
                    ..Default::default()
                },
                nrc_gap,
                &online_scenes
                    .iter()
                    .map(|(a, b, c)| (a.as_slice(), b.as_slice(), c.as_slice()))
                    .collect::<Vec<_>>(),
            )
            .expect("online nrc");
        save_nrc(&nrc_arch, nrc_net.params(), nrc_save_path(hops)).expect("save online nrc");
    }
    ProbeGi::with_nrc_fuse(
        infer_frame,
        Some(nrc_net),
        side,
        side,
        nrc_gap,
        if hops {
            HOPS_NRC_CAP
        } else {
            threers_probe::NRC_CAP
        },
    )
}

fn score_hostile(
    rt: &mut threers::raytrace::RaytraceRenderer,
    stack: &mut ProbeGi,
    frame: u32,
    probe_spp: u32,
    probe_bounces: u32,
    reference_spp: u32,
    side: usize,
) {
    println!("\nhostile hold-out (never trained on)");
    let t2 = Instant::now();
    let _ = std::fs::create_dir_all("out/probe");
    let mut hostile_probe = (0.0f64, 0usize);
    let mut hostile_unet = (0.0f64, 0usize);
    let mut hostile_fused = (0.0f64, 0usize);
    for index in 0..generate::hostile_count() {
        let packed = render_pair(
            rt,
            generate::hostile(index),
            frame,
            probe_spp,
            probe_bounces,
            reference_spp,
            0xC0FF | index as u64,
        );
        let name = generate::hostile_name(index);
        let (p, u, f) = score_ablation(stack, &packed, name);
        if name == "cornell" {
            println!(
                "ABLATION cornell_probe={p:.5} cornell_unet={u:.5} cornell_fused={f:.5} cornell_pct={:.1}",
                100.0 * f / p
            );
        }
        println!(
            "  {name:<14}  probe {p:.5}  unet {u:.5}  +nrc {f:.5}  ({:.0}% / {:.0}%)",
            100.0 * u / p,
            100.0 * f / p
        );
        accumulate(&mut hostile_probe, p, packed.pixels());
        accumulate(&mut hostile_unet, u, packed.pixels());
        accumulate(&mut hostile_fused, f, packed.pixels());
        write_preview(
            packed.frame,
            &format!("out/probe/{name}_probe.png"),
            &packed.input,
        );
        let pred = stack
            .unet_mut()
            .reconstruct(&packed.input, side, side)
            .expect("reconstruct");
        write_preview(packed.frame, &format!("out/probe/{name}_pred.png"), &pred);
        let fused = stack.reconstruct(&packed.input).expect("fused");
        write_preview(
            packed.frame,
            &format!("out/probe/{name}_fused.png"),
            &fused,
        );
        write_preview(
            packed.frame,
            &format!("out/probe/{name}_ref.png"),
            &packed.target,
        );
    }
    println!("hostile rendered in {:.1}s", t2.elapsed().as_secs_f32());
    let hp = root(hostile_probe);
    let hu = root(hostile_unet);
    let hf = root(hostile_fused);
    println!(
        "  {:<14}  probe {hp:.5}  unet {hu:.5}  +nrc {hf:.5}  ({:.0}% / {:.0}%)",
        "all hostile",
        100.0 * hu / hp,
        100.0 * hf / hp
    );
}

fn report(label: &str, err: f32, probe: f32) {
    println!(
        "  {label:<14}  {err:.5}   ({:.0}% of the probe)",
        100.0 * err / probe
    );
}

fn eval_probe_baseline(set: &Dataset) -> f32 {
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for i in 0..set.len() {
        let (input, target) = set.gather(&[i]).unwrap();
        let colour = colour_planes(&input, set.tile() * set.tile());
        let (s, c) = ProbeNet::relative_error_sum(colour, &target, LOSS_EPSILON);
        sum += s;
        n += c;
    }
    (sum / n.max(1) as f64).sqrt() as f32
}

fn eval_net(
    net: &ProbeArch,
    params: &[Vec<f32>],
    tile: usize,
    device: rlx::Device,
    set: &Dataset,
) -> f32 {
    let mut infer =
        ProbeNet::new(net.clone(), params.to_vec(), 1, tile, tile, device).expect("infer");
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for i in 0..set.len() {
        let (input, target) = set.gather(&[i]).unwrap();
        let pred = infer.run(&input).expect("run");
        let (s, c) = ProbeNet::relative_error_sum(&pred, &target, LOSS_EPSILON);
        sum += s;
        n += c;
    }
    (sum / n.max(1) as f64).sqrt() as f32
}

fn score_ablation(stack: &mut ProbeGi, packed: &Packed, name_hint: &str) -> (f32, f32, f32) {
    let side = packed.frame as usize;
    let pixels = side * side;
    let probe = ProbeNet::relative_error(
        colour_planes(&packed.input, pixels),
        &packed.target,
        LOSS_EPSILON,
    );
    let pred = stack
        .unet_mut()
        .reconstruct(&packed.input, side, side)
        .expect("reconstruct");
    let unet = ProbeNet::relative_error(&pred, &packed.target, LOSS_EPSILON);
    let fused = stack.reconstruct(&packed.input).expect("fused");
    let both = ProbeNet::relative_error(&fused, &packed.target, LOSS_EPSILON);
    if name_hint == "cornell" {
        print_regions("cornell regions  probe", colour_planes(&packed.input, pixels), &packed.target, &packed.input, pixels);
        print_regions("cornell regions  hops ", &pred, &packed.target, &packed.input, pixels);
        print_regions("cornell regions  fused", &fused, &packed.target, &packed.input, pixels);
    }
    (probe, unet, both)
}

fn print_regions(label: &str, pred: &[f32], target: &[f32], input: &[f32], pixels: usize) {
    let mut buckets: [(&str, f64, usize); 5] = [
        ("floor", 0.0, 0),
        ("ceil", 0.0, 0),
        ("wallX", 0.0, 0),
        ("wallZ", 0.0, 0),
        ("other", 0.0, 0),
    ];
    let mut floor_bias = [0.0f64; 3];
    let mut floor_n = 0usize;
    let eps = LOSS_EPSILON as f64;
    for i in 0..pixels {
        let nx = input[6 * pixels + i];
        let ny = input[7 * pixels + i];
        let nz = input[8 * pixels + i];
        let slot = if ny > 0.7 {
            0
        } else if ny < -0.7 {
            1
        } else if nx.abs() > 0.7 {
            2
        } else if nz.abs() > 0.7 {
            3
        } else {
            4
        };
        for c in 0..OUT_CHANNELS {
            let y = pred[c * pixels + i] as f64;
            let t = target[c * pixels + i] as f64;
            let d = y - t;
            buckets[slot].1 += d * d / (t * t + eps);
            buckets[slot].2 += 1;
            if slot == 0 {
                floor_bias[c] += d;
                if c == 0 {
                    floor_n += 1;
                }
            }
        }
    }
    print!("  {label}");
    for (name, sum, n) in buckets {
        if n == 0 {
            continue;
        }
        print!("  {name} {:.4}", (sum / n as f64).sqrt());
    }
    if floor_n > 0 {
        print!(
            "  floorΔ r{:.4} g{:.4} b{:.4}",
            floor_bias[0] / floor_n as f64,
            floor_bias[1] / floor_n as f64,
            floor_bias[2] / floor_n as f64
        );
    }
    println!();
}

fn accumulate(acc: &mut (f64, usize), rms: f32, pixels: usize) {
    let n = pixels * OUT_CHANNELS;
    acc.0 += (rms as f64) * (rms as f64) * n as f64;
    acc.1 += n;
}

fn root(acc: (f64, usize)) -> f32 {
    if acc.1 == 0 {
        return f32::NAN;
    }
    (acc.0 / acc.1 as f64).sqrt() as f32
}

fn write_preview(frame: u32, path: &str, planar: &[f32]) {
    let side = frame as usize;
    let mut rgba = vec![0u8; side * side * 4];
    for y in 0..side {
        for x in 0..side {
            let i = y * side + x;
            let o = i * 4;
            rgba[o] = srgb8(planar[i]);
            rgba[o + 1] = srgb8(planar[side * side + i]);
            rgba[o + 2] = srgb8(planar[2 * side * side + i]);
            rgba[o + 3] = 255;
        }
    }
    std::fs::write(path, encode_png(frame, frame, &rgba)).expect("png");
}

fn srgb8(compressed: f32) -> u8 {
    let v = compressed.clamp(0.0, 1.0);
    let s = if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

#[derive(Clone)]
struct Packed {
    frame: u32,
    kind: String,
    input: Vec<f32>,
    target: Vec<f32>,
}

impl Packed {
    fn pixels(&self) -> usize {
        (self.frame * self.frame) as usize
    }
}

fn world_positions(
    camera: &threers::PerspectiveCamera,
    depth: &[f32],
    width: usize,
    height: usize,
) -> Vec<[f32; 3]> {
    let rt = RtCamera::new(camera, &RaytraceSettings::default());
    let mut out = vec![[f32::NAN; 3]; width * height];
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let d = depth[i];
            if !d.is_finite() {
                continue;
            }
            let (origin, dir) = rt.pixel_ray(
                x as u32,
                y as u32,
                width as u32,
                height as u32,
                (0.5, 0.5),
                (0.5, 0.5),
                // Mid-shutter: this reprojects a static depth buffer, so the
                // instant only matters if a motion blur is configured, and the
                // middle of the exposure is the frame this depth belongs to.
                0.5,
            );
            let hit = origin + dir * d;
            out[i] = [hit.x, hit.y, hit.z];
        }
    }
    out
}

fn render_pair(
    rt: &mut threers::raytrace::RaytraceRenderer,
    built: Built,
    frame: u32,
    probe_spp: u32,
    probe_bounces: u32,
    reference_spp: u32,
    seed: u64,
) -> Packed {
    let Built {
        mut scene,
        camera,
        kind,
    } = built;
    rt.set_settings(RaytraceSettings {
        samples_per_pixel: probe_spp.max(1),
        max_bounces: probe_bounces.max(1),
        min_bounces: 1,
        clamp_indirect: 100.0,
        adaptive_threshold: 0.0,
        denoise: false,
        tone_mapping: ToneMapping::None,
        seed: 0x5eed_0000_0000 | seed,
        ..Default::default()
    });
    rt.render(&mut scene, &camera).expect("probe");
    let probe = rt.film().resolve_hdr();
    let albedo = rt.film().resolve_albedo();
    let normal = rt.film().resolve_normal();
    let depth = rt.film().resolve_depth();
    let scale = rt.traced_scene().map(|s| s.scale()).unwrap_or(1.0);
    let world = world_positions(&camera, &depth, frame as usize, frame as usize);

    rt.set_settings(RaytraceSettings {
        samples_per_pixel: reference_spp,
        max_bounces: 5,
        min_bounces: 3,
        clamp_indirect: 100.0,
        adaptive_threshold: 0.0,
        denoise: false,
        tone_mapping: ToneMapping::None,
        seed: 0x5eed_0000_0000 | seed,
        ..Default::default()
    });
    rt.render(&mut scene, &camera).expect("reference");
    let reference = rt.film().resolve_hdr();

    let (input, target) = pack::planes(
        frame as usize,
        frame as usize,
        &probe,
        &albedo,
        &normal,
        &depth,
        &reference,
        scale,
        Some(PlaneExtras { world: &world }),
    )
    .expect("pack");
    Packed {
        frame,
        kind,
        input,
        target,
    }
}

fn make_renderer(width: u32, height: u32) -> threers::raytrace::RaytraceRenderer {
    if env_u32("GPU", 0) == 0 {
        return threers::raytrace::RaytraceRenderer::new(width, height);
    }
    match threers::raytrace::gpu::GpuBackend::headless() {
        Ok(b) => threers::raytrace::RaytraceRenderer::with_backend(width, height, Box::new(b)),
        Err(e) => {
            eprintln!("gpu backend unavailable ({e}); staying on the cpu");
            threers::raytrace::RaytraceRenderer::new(width, height)
        }
    }
}

struct Shuffle(u64);

fn shuffle(items: &mut [usize], rng: &mut Shuffle) {
    for i in (1..items.len()).rev() {
        rng.0 ^= rng.0 << 13;
        rng.0 ^= rng.0 >> 7;
        rng.0 ^= rng.0 << 17;
        let j = ((rng.0 >> 40) as usize) % (i + 1);
        items.swap(i, j);
    }
}
