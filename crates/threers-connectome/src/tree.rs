//! Reading a connectome text tree.
//!
//! Both releases expose the same shape from the outside — an `index.tsv`, a
//! `neurons/<last two digits>/<id>/` directory per neuron — but not the same
//! columns, so every field is looked up by header name rather than by position.

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::Placed;

/// Every 16th skeleton node is plenty for a centroid: a 2,000-node arbour still
/// contributes 125 samples, which lands the mean well inside a micron.
const CENTROID_STRIDE: usize = 16;

/// One of the two releases, and the column each publishes a position in.
pub struct Release {
    pub key: &'static str,
    pub title: &'static str,
    pub subtitle: &'static str,
    /// `index.tsv` column names holding x, y, z — in nanometres, like the rest
    /// of both trees.
    pub pos_cols: [&'static str; 3],
    /// What that column actually is, for the viewer's coverage note.
    pub pos_source: &'static str,
}

impl Release {
    pub const ALL: [Release; 2] = [
        Release {
            key: "flywire-783",
            title: "FlyWire FAFB v783",
            subtitle: "adult female Drosophila brain",
            pos_cols: ["soma_x_nm", "soma_y_nm", "soma_z_nm"],
            pos_source: "soma",
        },
        Release {
            key: "malecns-v1.0",
            title: "Male CNS v1.0",
            subtitle: "adult male Drosophila brain + optic lobes + VNC",
            pos_cols: ["centroid_x_nm", "centroid_y_nm", "centroid_z_nm"],
            pos_source: "skeleton centroid",
        },
    ];

    pub fn find(key: &str) -> Option<&'static Release> {
        Release::ALL.iter().find(|r| r.key == key)
    }
}

/// One neuron, reduced to what the viewer draws and filters on.
pub struct Node {
    pub id: u64,
    pub pos: [f32; 3],
    pub role: String,
    pub nt: String,
    pub class: String,
    /// `cell_type \t cell_class \t side`, the hover line, kept pre-joined.
    pub label: String,
    pub synapses: u32,
    pub placed: Placed,
}

/// A release, read.
pub struct Tree {
    pub key: String,
    pub title: String,
    pub subtitle: String,
    pub pos_source: String,
    pub nodes: Vec<Node>,
    /// Rows in `index.tsv` — always ≥ `nodes.len()`.
    pub indexed: usize,
    pub from_index: usize,
    pub from_skeleton: usize,
    /// Neurons with neither a published position nor a skeleton to derive one
    /// from. Reported rather than dropped silently.
    pub unplaceable: Vec<u64>,
}

/// `neurons/<last two digits of id>/<id>/`.
pub fn neuron_dir(root: &Path, tree_key: &str, id: &str) -> PathBuf {
    let bucket = if id.len() >= 2 { &id[id.len() - 2..] } else { id };
    root.join(tree_key).join("neurons").join(bucket).join(id)
}

/// Mean of every `CENTROID_STRIDE`-th node of an SWC file.
///
/// Returns `None` only if the file is missing or holds no usable node; a
/// skeleton short enough for the stride to skip every node falls back to its
/// first node rather than to nothing.
pub fn skeleton_centroid(path: &Path) -> Option<[f64; 3]> {
    let raw = fs::read(path).ok()?;
    let mut sum = [0.0f64; 3];
    let mut n = 0u64;
    let mut seen = 0usize;
    let mut first = None;
    for line in raw.split(|&b| b == b'\n') {
        if line.is_empty() || line[0] == b'#' {
            continue;
        }
        seen += 1;
        let xyz = match swc_xyz(line) {
            Some(v) => v,
            None => continue,
        };
        if first.is_none() {
            first = Some(xyz);
        }
        if !seen.is_multiple_of(CENTROID_STRIDE) {
            continue;
        }
        for d in 0..3 {
            sum[d] += xyz[d];
        }
        n += 1;
    }
    if n > 0 {
        Some([sum[0] / n as f64, sum[1] / n as f64, sum[2] / n as f64])
    } else {
        first
    }
}

/// Columns 2..5 of an SWC row: `n type x y z radius parent`.
fn swc_xyz(line: &[u8]) -> Option<[f64; 3]> {
    let mut f = split_ws(line);
    f.next()?;
    f.next()?;
    let x = parse_f64(f.next()?)?;
    let y = parse_f64(f.next()?)?;
    let z = parse_f64(f.next()?)?;
    Some([x, y, z])
}

/// A neuron's arbour as line-segment endpoints: six floats (child, then parent)
/// per edge, ready to become a `LineSegments` position attribute.
///
/// Parents are not guaranteed to appear before their children — FlyWire's healed
/// skeletons root wherever the soma landed — so the node table is built in full
/// before any edge is emitted.
pub struct Skeleton {
    pub segments: Vec<f32>,
    pub nodes: usize,
    pub mean_radius_nm: f64,
    pub soma: Option<[f32; 3]>,
}

pub fn read_skeleton(path: &Path) -> io::Result<Skeleton> {
    let raw = fs::read(path)?;
    let mut index: HashMap<i64, usize> = HashMap::new();
    let mut xyz: Vec<[f32; 3]> = Vec::new();
    let mut parent: Vec<i64> = Vec::new();
    let mut radius_sum = 0.0f64;
    let mut radius_n = 0u64;
    let mut soma = None;

    for line in raw.split(|&b| b == b'\n') {
        if line.is_empty() || line[0] == b'#' {
            continue;
        }
        let mut f = split_ws(line);
        let (n, t, x, y, z, r, p) = match (
            f.next().and_then(parse_i64),
            f.next().and_then(parse_i64),
            f.next().and_then(parse_f64),
            f.next().and_then(parse_f64),
            f.next().and_then(parse_f64),
            f.next().and_then(parse_f64),
            f.next().and_then(parse_i64),
        ) {
            (Some(n), Some(t), Some(x), Some(y), Some(z), Some(r), Some(p)) => (n, t, x, y, z, r, p),
            _ => continue,
        };
        index.insert(n, xyz.len());
        xyz.push([x as f32, y as f32, z as f32]);
        parent.push(p);
        if r > 0.0 {
            radius_sum += r;
            radius_n += 1;
        }
        // FlyWire marks the soma node `type 1`; Male CNS cannot, so this stays
        // `None` there and the viewer falls back to the node's own position.
        if t == 1 && soma.is_none() {
            soma = Some([x as f32, y as f32, z as f32]);
        }
    }

    let mut segments = Vec::with_capacity(xyz.len() * 6);
    for (i, &p) in parent.iter().enumerate() {
        let pi = match index.get(&p) {
            Some(&pi) if pi != i => pi,   // skip roots and self-loops
            _ => continue,
        };
        segments.extend_from_slice(&xyz[i]);
        segments.extend_from_slice(&xyz[pi]);
    }

    Ok(Skeleton {
        segments,
        nodes: xyz.len(),
        mean_radius_nm: if radius_n > 0 { radius_sum / radius_n as f64 } else { 0.0 },
        soma,
    })
}

impl Tree {
    /// Read a release: parse `index.tsv`, then walk the skeletons of every
    /// neuron the index left without a position.
    pub fn read(root: &Path, rel: &Release) -> io::Result<Tree> {
        let path = root.join(rel.key).join("index.tsv");
        let mut text = String::new();
        fs::File::open(&path)?.read_to_string(&mut text)?;

        let mut lines = text.split('\n');
        let header: Vec<&str> = lines.next().unwrap_or("").trim_end_matches('\r').split('\t').collect();
        let col = |name: &str| header.iter().position(|h| *h == name);

        let c_id = col("root_id").ok_or_else(|| bad(&format!("{}: no root_id column", path.display())))?;
        let (c_x, c_y, c_z) = (col(rel.pos_cols[0]), col(rel.pos_cols[1]), col(rel.pos_cols[2]));
        let c_role = col("role");
        let c_nt = col("top_nt");
        let c_class = col("super_class");
        let c_type = col("cell_type");
        let c_cellclass = col("cell_class");
        let c_side = col("side");
        let c_in = col("in_synapses");
        let c_out = col("out_synapses");

        let mut nodes: Vec<Node> = Vec::new();
        let mut pending: Vec<(usize, u64)> = Vec::new();   // slot in `nodes`, id
        let mut indexed = 0usize;
        let mut from_index = 0usize;

        for line in lines {
            let line = line.trim_end_matches('\r');
            if line.is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            let id = match f.get(c_id).and_then(|s| s.parse::<u64>().ok()) {
                Some(id) => id,
                None => continue,
            };
            indexed += 1;

            let get = |c: Option<usize>| c.and_then(|i| f.get(i)).copied().unwrap_or("");
            let num = |c: Option<usize>| get(c).parse::<u32>().unwrap_or(0);

            let pos = match (c_x, c_y, c_z) {
                (Some(x), Some(y), Some(z)) => match (
                    f.get(x).and_then(|s| s.parse::<f32>().ok()),
                    f.get(y).and_then(|s| s.parse::<f32>().ok()),
                    f.get(z).and_then(|s| s.parse::<f32>().ok()),
                ) {
                    (Some(x), Some(y), Some(z)) => Some([x, y, z]),
                    _ => None,
                },
                _ => None,
            };

            let placed = if pos.is_some() { Placed::Index } else { Placed::Skeleton };
            if pos.is_some() {
                from_index += 1;
            } else {
                pending.push((nodes.len(), id));
            }

            nodes.push(Node {
                id,
                pos: pos.unwrap_or([f32::NAN; 3]),
                role: non_empty(get(c_role), "unknown"),
                nt: non_empty(get(c_nt), "unknown"),
                class: non_empty(get(c_class), "unknown"),
                label: format!("{}\t{}\t{}", get(c_type), get(c_cellclass), get(c_side)),
                synapses: num(c_in).saturating_add(num(c_out)),
                placed,
            });
        }

        // Reading tens of thousands of small files off an external volume is
        // latency-bound, not CPU-bound, so a handful of threads pays for itself
        // where more cores would not.
        let found = resolve_missing(root, rel.key, &pending);
        for (slot, centroid) in &found {
            let n = &mut nodes[*slot];
            n.pos = [centroid[0] as f32, centroid[1] as f32, centroid[2] as f32];
        }
        let from_skeleton = found.len();

        let mut unplaceable = Vec::new();
        let mut kept = Vec::with_capacity(nodes.len());
        for n in nodes {
            if n.pos[0].is_nan() {
                unplaceable.push(n.id);
            } else {
                kept.push(n);
            }
        }

        Ok(Tree {
            key: rel.key.to_string(),
            title: rel.title.to_string(),
            subtitle: rel.subtitle.to_string(),
            pos_source: rel.pos_source.to_string(),
            nodes: kept,
            indexed,
            from_index,
            from_skeleton,
            unplaceable,
        })
    }
}

/// Walk `pending` neurons' skeletons on a small thread pool, returning the
/// (slot, centroid) pairs that resolved.
fn resolve_missing(root: &Path, key: &str, pending: &[(usize, u64)]) -> Vec<(usize, [f64; 3])> {
    if pending.is_empty() {
        return Vec::new();
    }
    let next = AtomicUsize::new(0);
    let out: Mutex<Vec<(usize, [f64; 3])>> = Mutex::new(Vec::new());
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);

    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                let mut mine = Vec::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= pending.len() {
                        break;
                    }
                    let (slot, id) = pending[i];
                    let path = neuron_dir(root, key, &id.to_string()).join("skeleton.swc");
                    if let Some(c) = skeleton_centroid(&path) {
                        mine.push((slot, c));
                    }
                }
                out.lock().unwrap().extend(mine);
            });
        }
    });
    out.into_inner().unwrap()
}

fn non_empty(s: &str, fallback: &str) -> String {
    if s.is_empty() { fallback.to_string() } else { s.to_string() }
}

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

// --- tiny byte-slice parsing, so a 9 GB skeleton corpus never allocates a String

fn split_ws(line: &[u8]) -> impl Iterator<Item = &[u8]> {
    line.split(|b| b.is_ascii_whitespace()).filter(|s| !s.is_empty())
}

fn parse_f64(b: &[u8]) -> Option<f64> {
    std::str::from_utf8(b).ok()?.parse().ok()
}

fn parse_i64(b: &[u8]) -> Option<i64> {
    std::str::from_utf8(b).ok()?.parse().ok()
}
