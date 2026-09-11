//! A small HTTP/1.1 server, because that is all the viewer needs.
//!
//! Static files out of a few mounted roots, plus a routing hook for the two
//! endpoints that read the connectome tree. `Connection: close` throughout: at
//! a dozen requests per page load, connection reuse would buy nothing and cost
//! a state machine.

use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};

pub struct Response {
    pub status: u16,
    pub content_type: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Response {
    pub fn new(status: u16, content_type: &str, body: Vec<u8>) -> Response {
        Response { status, content_type: content_type.to_string(), headers: Vec::new(), body }
    }
    pub fn json(status: u16, body: String) -> Response {
        Response::new(status, "application/json; charset=utf-8", body.into_bytes())
    }
    pub fn text(status: u16, body: &str) -> Response {
        Response::new(status, "text/plain; charset=utf-8", body.as_bytes().to_vec())
    }
    pub fn header(mut self, k: &str, v: &str) -> Response {
        self.headers.push((k.to_string(), v.to_string()));
        self
    }
}

pub struct Request {
    pub method: String,
    /// Percent-decoded, query stripped.
    pub path: String,
}

/// URL prefix -> directory. Longest prefix wins, and `""` is the fallback root.
pub type Mounts = Vec<(String, PathBuf)>;

pub fn serve<F>(addr: &str, mounts: Mounts, route: F) -> io::Result<()>
where
    F: Fn(&Request) -> Option<Response> + Send + Sync + 'static,
{
    let listener = TcpListener::bind(addr)?;
    let shared = std::sync::Arc::new((mounts, route));
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        let shared = shared.clone();
        std::thread::spawn(move || {
            let _ = handle(stream, &shared.0, &shared.1);
        });
    }
    Ok(())
}

fn handle<F>(mut stream: TcpStream, mounts: &Mounts, route: &F) -> io::Result<()>
where
    F: Fn(&Request) -> Option<Response>,
{
    let req = match read_request(&mut stream)? {
        Some(r) => r,
        None => return write_response(&mut stream, "GET", Response::text(400, "bad request\n")),
    };
    if req.method != "GET" && req.method != "HEAD" {
        return write_response(&mut stream, &req.method, Response::text(405, "method not allowed\n"));
    }
    let res = route(&req).unwrap_or_else(|| static_file(mounts, &req.path));
    write_response(&mut stream, &req.method, res)
}

fn read_request(stream: &mut TcpStream) -> io::Result<Option<Request>> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let method = match parts.next() {
        Some(m) => m.to_string(),
        None => return Ok(None),
    };
    let target = parts.next().unwrap_or("/");
    // Drain the headers so the client's write completes before we reply.
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 || h == "\r\n" || h == "\n" {
            break;
        }
    }
    let path = percent_decode(target.split(['?', '#']).next().unwrap_or("/"));
    Ok(Some(Request { method, path }))
}

fn write_response(stream: &mut TcpStream, method: &str, res: Response) -> io::Result<()> {
    let reason = match res.status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        _ => "OK",
    };
    let mut head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n",
        res.status, reason, res.content_type, res.body.len()
    );
    for (k, v) in &res.headers {
        head.push_str(&format!("{}: {}\r\n", k, v));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    if method != "HEAD" {
        stream.write_all(&res.body)?;
    }
    stream.flush()
}

/// Resolve a URL path inside its mount and read the file.
///
/// The path is rebuilt component by component with `..` and absolute roots
/// dropped, so nothing a client sends can climb out of the mount — the check
/// does not depend on the file existing, which a `canonicalize`-based one would.
fn static_file(mounts: &Mounts, url_path: &str) -> Response {
    let mut best: Option<(&str, &PathBuf)> = None;
    for (prefix, root) in mounts {
        let matches = prefix.is_empty()
            || url_path == prefix
            || url_path.starts_with(&format!("{}/", prefix));
        if matches && best.map(|(p, _)| prefix.len() > p.len()).unwrap_or(true) {
            best = Some((prefix, root));
        }
    }
    let (prefix, root) = match best {
        Some(b) => b,
        None => return Response::text(404, "not found\n"),
    };

    let mut rel = &url_path[prefix.len()..];
    if rel.is_empty() || rel == "/" {
        rel = "/index.html";
    }
    let mut path = root.clone();
    for comp in Path::new(rel).components() {
        if let Component::Normal(c) = comp {
            path.push(c);
        }
    }
    match fs::read(&path) {
        Ok(bytes) => {
            let ct = mime_of(&path);
            let mut r = Response::new(200, ct, bytes);
            if ct == "application/wasm" || ct.starts_with("application/octet") {
                r = r.header("Cache-Control", "no-cache");
            }
            r
        }
        Err(_) => Response::text(404, &format!("not found: {}\n", url_path)),
    }
}

fn mime_of(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" => "application/json; charset=utf-8",
        "wasm" => "application/wasm",
        "txt" | "ts" => "text/plain; charset=utf-8",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "svg" => "image/svg+xml",
        "glsl" => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2])) {
                out.push(h * 16 + l);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// `field <tab> value` files (`meta.tsv`), in order.
pub fn read_kv_tsv(path: &Path) -> io::Result<Vec<(String, String)>> {
    let mut text = String::new();
    fs::File::open(path)?.read_to_string(&mut text)?;
    Ok(text
        .lines()
        .skip(1)
        .filter_map(|l| {
            let mut f = l.splitn(2, '\t');
            let k = f.next()?.to_string();
            if k.is_empty() {
                return None;
            }
            Some((k, f.next().unwrap_or("").to_string()))
        })
        .collect())
}

/// A header row plus rows, for `connections.tsv` and friends.
pub fn read_tsv(path: &Path) -> io::Result<(Vec<String>, Vec<Vec<String>>)> {
    let mut text = String::new();
    fs::File::open(path)?.read_to_string(&mut text)?;
    let mut lines = text.lines();
    let header: Vec<String> = lines.next().unwrap_or("").split('\t').map(str::to_string).collect();
    let rows = lines
        .filter(|l| !l.is_empty())
        .map(|l| l.split('\t').map(str::to_string).collect())
        .collect();
    Ok((header, rows))
}

/// Used by the neuron endpoint to fold a connection table down to top partners.
pub fn top_partners(
    header: &[String],
    rows: &[Vec<String>],
    limit: usize,
) -> HashMap<String, Vec<(String, u64)>> {
    let c_partner = header.iter().position(|h| h == "partner");
    let c_dir = header.iter().position(|h| h == "dir");
    // FlyWire spells it `syn_count`, Male CNS `weight`; take whichever is there.
    let c_n = header
        .iter()
        .position(|h| h.contains("syn_count") || h.contains("weight") || h.contains("count"));

    let mut acc: HashMap<(String, String), u64> = HashMap::new();
    if let (Some(cp), Some(cd)) = (c_partner, c_dir) {
        for r in rows {
            let (p, d) = match (r.get(cp), r.get(cd)) {
                (Some(p), Some(d)) if d == "in" || d == "out" => (p.clone(), d.clone()),
                _ => continue,
            };
            let w = c_n.and_then(|c| r.get(c)).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
            *acc.entry((d, p)).or_insert(0) += w;
        }
    }
    let mut out: HashMap<String, Vec<(String, u64)>> = HashMap::new();
    for ((d, p), w) in acc {
        out.entry(d).or_default().push((p, w));
    }
    for v in out.values_mut() {
        v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v.truncate(limit);
    }
    out.entry("in".into()).or_default();
    out.entry("out".into()).or_default();
    out
}
