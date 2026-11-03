//! Run Cornell ablation matrix and print a summary table.
//!
//! ```sh
//! GPU=1 cargo run --release -p threers-probe --example probe_ablation --features generate,metal
//! ABLATIONS=baseline,G cargo run --release -p threers-probe --example probe_ablation --features generate,metal
//! ```

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::Instant;

struct Ablation {
    id: &'static str,
    label: &'static str,
    env: &'static [(&'static str, &'static str)],
}

fn ablations() -> &'static [Ablation] {
    static LIST: &[Ablation] = &[
        Ablation {
            id: "baseline",
            label: "current weights (LOAD)",
            env: &[("LOAD", "1")],
        },
        Ablation {
            id: "G",
            label: "NRC fuse gap 0.018",
            env: &[("LOAD", "1"), ("NRC_GAP", "0.018")],
        },
        Ablation {
            id: "C",
            label: "NRC-only + ONLINE_NRC=20",
            env: &[
                ("NRC_ONLY", "1"),
                ("ONLINE_NRC", "20"),
                ("NRC_EPOCHS", "40"),
                ("NRC_OUT", "out/ablation_C_nrc.bin"),
            ],
        },
        Ablation {
            id: "F",
            label: "NRC-only + NRC_DDGI=1",
            env: &[
                ("NRC_ONLY", "1"),
                ("NRC_DDGI", "1"),
                ("NRC_EPOCHS", "40"),
                ("ONLINE_NRC", "12"),
                ("NRC_OUT", "out/ablation_F_nrc.bin"),
            ],
        },
        Ablation {
            id: "A",
            label: "PROBE=16 PROBE_BOUNCES=4 full retrain",
            env: &[
                ("PROBE", "16"),
                ("PROBE_BOUNCES", "4"),
                ("EPOCHS", "60"),
                ("WEIGHTS", "out/ablation_A_probe.bin"),
                ("NRC_OUT", "out/ablation_A_nrc.bin"),
            ],
        },
        Ablation {
            id: "B",
            label: "NEUTRAL_LOSS=1.15 full retrain",
            env: &[
                ("NEUTRAL_LOSS", "1.15"),
                ("EPOCHS", "60"),
                ("WEIGHTS", "out/ablation_B_probe.bin"),
                ("NRC_OUT", "out/ablation_B_nrc.bin"),
            ],
        },
        Ablation {
            id: "D",
            label: "CORNELL_EMPTY=0.35 full retrain",
            env: &[
                ("CORNELL_EMPTY", "0.35"),
                ("EPOCHS", "60"),
                ("WEIGHTS", "out/ablation_D_probe.bin"),
                ("NRC_OUT", "out/ablation_D_nrc.bin"),
            ],
        },
        Ablation {
            id: "E",
            label: "DDGI_VIS_BLEND=1 full retrain",
            env: &[
                ("DDGI_VIS_BLEND", "1"),
                ("EPOCHS", "60"),
                ("WEIGHTS", "out/ablation_E_probe.bin"),
                ("NRC_OUT", "out/ablation_E_nrc.bin"),
            ],
        },
    ];
    LIST
}

#[derive(Default, Clone, Copy)]
struct Score {
    probe: f32,
    unet: f32,
    fused: f32,
    pct: f32,
    all_pct: f32,
}

fn main() {
    let filter = std::env::var("ABLATIONS")
        .ok()
        .map(|s| {
            s.split(',')
                .map(|x| x.trim().to_ascii_lowercase())
                .filter(|x| !x.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let runs: Vec<&Ablation> = if filter.is_empty() {
        ablations().iter().collect()
    } else {
        ablations()
            .iter()
            .filter(|a| filter.iter().any(|f| f == a.id.to_ascii_lowercase().as_str()))
            .collect()
    };
    assert!(!runs.is_empty(), "no ablations matched ABLATIONS filter");

    let gpu = std::env::var("GPU").unwrap_or_else(|_| "1".into());
    let mut results: Vec<(&str, Score)> = Vec::new();
    let t0 = Instant::now();

    for ab in runs {
        println!("\n=== ablation {} — {} ===", ab.id, ab.label);
        match run_probe(&gpu, ab) {
            Ok(score) => results.push((ab.id, score)),
            Err(e) => {
                eprintln!("ablation {} failed: {e}", ab.id);
                results.push((ab.id, Score::default()));
            }
        }
    }

    println!(
        "\n=== Cornell ablation summary ({:.0}s) ===",
        t0.elapsed().as_secs_f32()
    );
    println!(
        "{:<10} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "id", "probe", "unet", "+nrc", "corn%", "all%"
    );
    for (id, s) in &results {
        if s.pct.is_finite() {
            println!(
                "{:<10} {:>10.5} {:>10.5} {:>10.5} {:>7.1}% {:>7.1}%",
                id, s.probe, s.unet, s.fused, s.pct, s.all_pct
            );
        } else {
            println!("{:<10}  (failed)", id);
        }
    }
}

fn run_probe(gpu: &str, ab: &Ablation) -> std::io::Result<Score> {
    let mut cmd = Command::new("cargo");
    cmd.current_dir(workspace_root())
        .args([
            "run",
            "--release",
            "-p",
            "threers-probe",
            "--example",
            "probe",
            "--features",
            "generate,metal",
        ])
        .env("GPU", gpu)
        .env("ARCH", "hops")
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());

    for (k, v) in ab.env {
        cmd.env(k, v);
    }

    let mut child = cmd.spawn()?;
    let stdout = child.stdout.take().expect("stdout");
    let reader = BufReader::new(stdout);
    let mut score = Score::default();
    for line in reader.lines() {
        let line = line?;
        print!("{line}");
        if let Some(rest) = line.strip_prefix("ABLATION cornell_probe=") {
            parse_cornell_line(rest, &mut score);
        }
        if line.contains("all hostile") && line.contains("+nrc") {
            if let Some(idx) = line.rfind('/') {
                let tail = &line[idx + 1..];
                if let Some(end) = tail.find('%') {
                    score.all_pct = tail[..end].trim().parse().unwrap_or(f32::NAN);
                }
            }
        }
    }
    let status = child.wait()?;
    if !status.success() {
        return Err(std::io::Error::other(format!(
            "probe exited with {status}"
        )));
    }
    Ok(score)
}

fn parse_cornell_line(rest: &str, score: &mut Score) {
    let mut parts = rest.split_whitespace();
    if let Some(v) = parts.next() {
        score.probe = v.parse().unwrap_or(f32::NAN);
    }
    for part in parts {
        if let Some(v) = part.strip_prefix("cornell_unet=") {
            score.unet = v.parse().unwrap_or(f32::NAN);
        } else if let Some(v) = part.strip_prefix("cornell_fused=") {
            score.fused = v.parse().unwrap_or(f32::NAN);
        } else if let Some(v) = part.strip_prefix("cornell_pct=") {
            score.pct = v.trim_end_matches('%').parse().unwrap_or(f32::NAN);
        }
    }
}

fn workspace_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}
