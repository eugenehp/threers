//! Host the viewer, and read the connectome tree on demand behind two routes.
//!
//! ```text
//! cargo run --release -p threers-connectome --bin connectome-serve
//! ```
//!
//! | URL | Serves |
//! |---|---|
//! | `/` | `web/connectome/` — the page and `viewer.js` |
//! | `/threers/…` | `web/` — the shim and the wasm build |
//! | `/data/…` | the tables from `connectome-build` |
//! | `/api/skeleton/<tree>/<id>` | that neuron's `skeleton.swc` as line-segment endpoints |
//! | `/api/neuron/<tree>/<id>` | its `meta.tsv`, top partners and per-region tallies |

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use threers_connectome::http::{self, Request, Response};
use threers_connectome::tree::{neuron_dir, read_skeleton, Release};

const DEFAULT_TREE: &str = "/Volumes/C4TB/connectome-fs";
const MAX_PARTNERS: usize = 25;

struct Config {
    tree: PathBuf,
    data: PathBuf,
    web: PathBuf,
}

fn main() -> ExitCode {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo = crate_dir.join("..").join("..");
    let mut cfg = Config {
        tree: PathBuf::from(
            std::env::var("CONNECTOME_FS").unwrap_or_else(|_| DEFAULT_TREE.to_string()),
        ),
        data: crate_dir.join("data"),
        web: repo.join("web"),
    };
    let mut port: u16 = std::env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8787);

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--tree" => cfg.tree = PathBuf::from(args.next().unwrap_or_default()),
            "--data" => cfg.data = PathBuf::from(args.next().unwrap_or_default()),
            "--web" => cfg.web = PathBuf::from(args.next().unwrap_or_default()),
            "--port" => port = args.next().and_then(|p| p.parse().ok()).unwrap_or(port),
            "-h" | "--help" => {
                println!("usage: connectome-serve [--port N] [--tree <connectome-fs>] [--data <dir>] [--web <threers web dir>]");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument: {other}");
                return ExitCode::FAILURE;
            }
        }
    }

    if !cfg.data.join("meta.json").is_file() {
        eprintln!("no node tables at {}", cfg.data.display());
        eprintln!("run: cargo run --release -p threers-connectome --bin connectome-build");
        return ExitCode::FAILURE;
    }

    let mounts = vec![
        ("/data".to_string(), cfg.data.clone()),
        ("/threers".to_string(), cfg.web.clone()),
        (String::new(), cfg.web.join("connectome")),
    ];

    println!("connectome viewer  http://127.0.0.1:{port}/");
    println!("  tree     {}", cfg.tree.display());
    println!("  tables   {}", cfg.data.display());
    println!("  threers  {}", cfg.web.display());

    let tree_root = cfg.tree.clone();
    let addr = format!("127.0.0.1:{port}");
    match http::serve(&addr, mounts, move |req| api(&tree_root, req)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("listening on {addr}: {e}");
            ExitCode::FAILURE
        }
    }
}

/// The two tree-reading routes. Anything else falls through to a static file.
fn api(root: &Path, req: &Request) -> Option<Response> {
    let rest = req.path.strip_prefix("/api/")?;
    let mut parts = rest.split('/');
    let kind = parts.next()?;
    let tree = parts.next()?;
    let id = parts.next()?;
    if parts.next().is_some() {
        return Some(Response::json(400, r#"{"error":"bad path"}"#.into()));
    }
    // Ids are digits and trees are one of two known keys — that is the whole
    // validation surface for a filesystem path built out of client input.
    if Release::find(tree).is_none() || id.is_empty() || id.len() > 20
        || !id.bytes().all(|b| b.is_ascii_digit())
    {
        return Some(Response::json(400, r#"{"error":"bad tree or id"}"#.into()));
    }
    let dir = neuron_dir(root, tree, id);
    match kind {
        "skeleton" => Some(skeleton(&dir, tree, id)),
        "neuron" => Some(neuron(&dir, tree, id)),
        _ => None,
    }
}

fn skeleton(dir: &Path, tree: &str, id: &str) -> Response {
    let sk = match read_skeleton(&dir.join("skeleton.swc")) {
        Ok(s) => s,
        Err(_) => {
            return Response::json(404, format!(
                r#"{{"error":"no skeleton.swc","tree":"{tree}","id":"{id}"}}"#
            ))
        }
    };
    let mut body = Vec::with_capacity(sk.segments.len() * 4);
    for v in &sk.segments {
        body.extend_from_slice(&v.to_le_bytes());
    }
    let soma = sk
        .soma
        .map(|s| format!("{},{},{}", s[0], s[1], s[2]))
        .unwrap_or_default();
    Response::new(200, "application/octet-stream", body)
        .header("X-Nodes", &sk.nodes.to_string())
        .header("X-Segments", &(sk.segments.len() / 6).to_string())
        .header("X-Mean-Radius-Nm", &format!("{:.1}", sk.mean_radius_nm))
        .header("X-Soma", &soma)
        .header(
            "Access-Control-Expose-Headers",
            "X-Nodes, X-Segments, X-Mean-Radius-Nm, X-Soma",
        )
}

fn neuron(dir: &Path, tree: &str, id: &str) -> Response {
    let meta = match http::read_kv_tsv(&dir.join("meta.tsv")) {
        Ok(m) => m,
        Err(_) => {
            return Response::json(404, format!(
                r#"{{"error":"no such neuron","tree":"{tree}","id":"{id}"}}"#
            ))
        }
    };

    let partners = http::read_tsv(&dir.join("connections.tsv"))
        .map(|(h, r)| http::top_partners(&h, &r, MAX_PARTNERS))
        .unwrap_or_default();

    let layers: Vec<String> = http::read_tsv(&dir.join("layers.tsv"))
        .map(|(_, rows)| rows.iter().take(40).map(|r| r.join(" · ")).collect())
        .unwrap_or_default();

    let meta_json = meta
        .iter()
        .map(|(k, v)| format!("{}:{}", esc(k), esc(v)))
        .collect::<Vec<_>>()
        .join(",");
    let dir_json = |d: &str| {
        partners
            .get(d)
            .map(|v| {
                v.iter()
                    .map(|(p, w)| format!(r#"{{"id":{},"syn":{}}}"#, esc(p), w))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .unwrap_or_default()
    };
    let layers_json = layers.iter().map(|l| esc(l)).collect::<Vec<_>>().join(",");

    Response::json(
        200,
        format!(
            r#"{{"tree":{},"id":{},"meta":{{{}}},"partners":{{"in":[{}],"out":[{}]}},"layers":[{}]}}"#,
            esc(tree), esc(id), meta_json, dir_json("in"), dir_json("out"), layers_json
        ),
    )
}

/// JSON string literal. Values come out of the tree, so control characters and
/// quotes are possible even if unlikely.
fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
