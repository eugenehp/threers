//! Get the Earth imagery, or serve it — because 6.2 GB does not belong in a repo.
//!
//! ```text
//! cargo run --release --example earth_assets -- check
//! cargo run --release --example earth_assets -- fetch
//! cargo run --release --example earth_assets -- fetch --only blackmarble
//! cargo run --release --example earth_assets -- serve --port 8080
//! ```
//!
//! | Flag | Effect |
//! |------|--------|
//! | `--assets DIR` | where the imagery lives (default `web/assets/earth`) |
//! | `--only NAME` | fetch one group: bluemarble, blackmarble, maps, sky |
//! | `--port N` | which port `serve` listens on (default 8080) |
//!
//! WHAT IS DOWNLOADED AND WHAT IS BUILT.
//!
//! `fetch` gets the SOURCES, under `<assets>/source`, with the names NASA gives
//! them. It does not write the files the loaders open: `day.16k.png`, the BC1
//! tiles under `globe/`, the baked maps. Those are derived, by
//! `bake_planet_maps` and by the tile cutters, and conflating the two would
//! leave a tree where it is impossible to tell what came from NASA and what
//! this code invented. `check` reports both, separately.
//!
//! EVERY DOWNLOAD IS VERIFIED AGAINST CONTENT-LENGTH.
//!
//! This is not belt and braces. These servers answer with chunked encoding, and
//! a chunked transfer that stops early is indistinguishable from a complete one
//! as far as the exit code goes: curl returns 0 and leaves a truncated file.
//! Six of eight tiles arrived that way on the first run that fetched them, and
//! the failure surfaced later as a decoder complaining about a corrupt image.
//! So the size is read from the server first, and the transfer is resumed until
//! the file on disk matches it.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// One file: where it goes, where it comes from, and how big it should be.
///
/// The size is here so a truncated transfer is caught at the point it happens
/// rather than by whatever tries to decode the file next week. All of them were
/// read off the servers rather than copied from a page.
struct Source {
    group: &'static str,
    name: &'static str,
    url: &'static str,
    bytes: u64,
}

const SOURCES: &[Source] = &[
    // Blue Marble: Next Generation, December 2004, topography and bathymetry.
    // Eight 21600x21600 tiles, A1..D2 west-to-east then north-to-south.
    // https://visibleearth.nasa.gov/images/73909
    Source { group: "bluemarble", name: "world.topo.bathy.200412.3x21600x21600.A1.jpg",
             url: "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x21600x21600.A1.jpg", bytes: 56_557_236 },
    Source { group: "bluemarble", name: "world.topo.bathy.200412.3x21600x21600.B1.jpg",
             url: "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x21600x21600.B1.jpg", bytes: 0 },
    Source { group: "bluemarble", name: "world.topo.bathy.200412.3x21600x21600.C1.jpg",
             url: "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x21600x21600.C1.jpg", bytes: 0 },
    Source { group: "bluemarble", name: "world.topo.bathy.200412.3x21600x21600.D1.jpg",
             url: "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x21600x21600.D1.jpg", bytes: 0 },
    Source { group: "bluemarble", name: "world.topo.bathy.200412.3x21600x21600.A2.jpg",
             url: "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x21600x21600.A2.jpg", bytes: 0 },
    Source { group: "bluemarble", name: "world.topo.bathy.200412.3x21600x21600.B2.jpg",
             url: "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x21600x21600.B2.jpg", bytes: 0 },
    Source { group: "bluemarble", name: "world.topo.bathy.200412.3x21600x21600.C2.jpg",
             url: "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x21600x21600.C2.jpg", bytes: 0 },
    Source { group: "bluemarble", name: "world.topo.bathy.200412.3x21600x21600.D2.jpg",
             url: "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x21600x21600.D2.jpg", bytes: 0 },

    // Black Marble 2016 at 500 m, the same eight-tile layout. This is the one
    // the globe's night maps are cut from; the 3 km version below is the whole
    // world in one file and is 6.4x coarser.
    // https://earthobservatory.nasa.gov/features/NightLights
    Source { group: "blackmarble", name: "BlackMarble_2016_A1_geo.tif",
             url: "https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144898/BlackMarble_2016_A1_geo.tif", bytes: 321_170_093 },
    Source { group: "blackmarble", name: "BlackMarble_2016_B1_geo.tif",
             url: "https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144898/BlackMarble_2016_B1_geo.tif", bytes: 310_401_888 },
    Source { group: "blackmarble", name: "BlackMarble_2016_C1_geo.tif",
             url: "https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144898/BlackMarble_2016_C1_geo.tif", bytes: 503_603_258 },
    Source { group: "blackmarble", name: "BlackMarble_2016_D1_geo.tif",
             url: "https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144898/BlackMarble_2016_D1_geo.tif", bytes: 387_774_688 },
    Source { group: "blackmarble", name: "BlackMarble_2016_A2_geo.tif",
             url: "https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144898/BlackMarble_2016_A2_geo.tif", bytes: 171_407_332 },
    Source { group: "blackmarble", name: "BlackMarble_2016_B2_geo.tif",
             url: "https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144898/BlackMarble_2016_B2_geo.tif", bytes: 270_956_469 },
    Source { group: "blackmarble", name: "BlackMarble_2016_C2_geo.tif",
             url: "https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144898/BlackMarble_2016_C2_geo.tif", bytes: 230_172_417 },
    Source { group: "blackmarble", name: "BlackMarble_2016_D2_geo.tif",
             url: "https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144898/BlackMarble_2016_D2_geo.tif", bytes: 237_917_468 },

    // The single-file maps.
    Source { group: "maps", name: "land_ocean_ice_8192.png",
             url: "https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57730/land_ocean_ice_8192.png", bytes: 24_147_269 },
    Source { group: "maps", name: "gebco_08_rev_elev_21600x10800.png",
             url: "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/gebco_08_rev_elev_21600x10800.png", bytes: 18_414_843 },
    Source { group: "maps", name: "dnb_land_ocean_ice.2012.13500x6750.jpg",
             url: "https://eoimages.gsfc.nasa.gov/images/imagerecords/79000/79765/dnb_land_ocean_ice.2012.13500x6750.jpg", bytes: 7_813_537 },
    Source { group: "maps", name: "cloud_combined_2048.jpg",
             url: "https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57747/cloud_combined_2048.jpg", bytes: 829_367 },

    // Sky and Moon, from Goddard's Scientific Visualization Studio.
    Source { group: "sky", name: "starmap_2020_4k.exr",
             url: "https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/starmap_2020_4k.exr", bytes: 35_997_085 },
    Source { group: "sky", name: "starmap_2020_8k.exr",
             url: "https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/starmap_2020_8k.exr", bytes: 130_530_278 },
    Source { group: "sky", name: "lroc_color_poles_8k.tif",
             url: "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/lroc_color_poles_8k.tif", bytes: 50_641_970 },
    Source { group: "sky", name: "ldem_16_uint.tif",
             url: "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/ldem_16_uint.tif", bytes: 33_201_026 },
];

/// What the loaders open, as opposed to what NASA ships. Derived, not fetched.
const DERIVED: &[(&str, &str)] = &[
    (
        "day.16k.png",
        "bake_planet_maps, from the eight Blue Marble tiles",
    ),
    ("night.png", "from dnb_land_ocean_ice"),
    ("clouds.png", "from cloud_combined"),
    ("elevation.png", "from gebco_08_rev_elev"),
    ("ice_cap.png", "from land_ocean_ice_8192"),
    (
        "globe",
        "18 BC1 day tiles + 18 night tiles at 14400 square, from the 500 m sources",
    ),
    ("baked", "bake_planet_maps output"),
    (
        "sky",
        "star and Moon maps, converted from the SVS originals",
    ),
];

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let arg = |k: &str| {
        a.iter()
            .position(|x| x == k)
            .and_then(|i| a.get(i + 1))
            .cloned()
    };
    let assets = PathBuf::from(arg("--assets").unwrap_or_else(|| "web/assets/earth".into()));
    match a.first().map(String::as_str) {
        Some("check") => check(&assets),
        Some("fetch") => fetch(&assets, arg("--only")),
        Some("serve") => serve(
            &assets,
            arg("--port").and_then(|p| p.parse().ok()).unwrap_or(8080),
        ),
        _ => {
            eprintln!("usage: earth_assets check | fetch [--only GROUP] | serve [--port N]");
            eprintln!("       [--assets DIR]   groups: bluemarble blackmarble maps sky");
            std::process::exit(2);
        }
    }
}

fn human(n: u64) -> String {
    match n {
        n if n >= 1 << 30 => format!("{:.1} GB", n as f64 / (1u64 << 30) as f64),
        n if n >= 1 << 20 => format!("{:.0} MB", n as f64 / (1u64 << 20) as f64),
        n => format!("{n} B"),
    }
}

fn dir_size(p: &Path) -> u64 {
    let Ok(rd) = std::fs::read_dir(p) else {
        return 0;
    };
    rd.flatten()
        .map(|e| match e.metadata() {
            Ok(m) if m.is_dir() => dir_size(&e.path()),
            Ok(m) => m.len(),
            Err(_) => 0,
        })
        .sum()
}

fn check(assets: &Path) {
    println!("sources — {}/source\n", assets.display());
    let src = assets.join("source");
    let (mut have, mut want) = (0u64, 0u64);
    for s in SOURCES {
        let p = src.join(s.name);
        let on_disk = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
        want += s.bytes;
        have += on_disk;
        let state = if on_disk == 0 {
            "missing".to_string()
        } else if s.bytes != 0 && on_disk != s.bytes {
            format!("TRUNCATED {} of {}", human(on_disk), human(s.bytes))
        } else {
            human(on_disk)
        };
        println!("  {:<12} {:<46} {}", s.group, s.name, state);
    }
    println!(
        "\n  {} of {} (sizes shown as 0 are read from the server at fetch time)",
        human(have),
        human(want)
    );

    println!("\nderived — built, not downloaded\n");
    for (name, how) in DERIVED {
        let p = assets.join(name);
        let n = if p.is_dir() {
            dir_size(&p)
        } else {
            std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0)
        };
        let state = if n == 0 { "missing".into() } else { human(n) };
        println!("  {:<14} {:<10} {}", name, state, how);
    }
    println!("\ntotal on disk: {}", human(dir_size(assets)));
}

/// What the server says this file is, which is the only size worth trusting.
fn remote_len(url: &str) -> Option<u64> {
    let out = std::process::Command::new("curl")
        .args(["-sIL", "--http1.1", "--max-time", "30", url])
        .output()
        .ok()?;
    let head = String::from_utf8_lossy(&out.stdout);
    // Last one wins: a redirect chain reports the length of each hop.
    head.lines()
        .filter(|l| l.to_ascii_lowercase().starts_with("content-length:"))
        .filter_map(|l| l.split(':').nth(1)?.trim().parse::<u64>().ok())
        .next_back()
}

fn fetch(assets: &Path, only: Option<String>) {
    let dir = assets.join("source");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("{}: {e}", dir.display());
        std::process::exit(1);
    }
    let mut failed = 0;
    for s in SOURCES {
        if only.as_deref().is_some_and(|g| g != s.group) {
            continue;
        }
        let path = dir.join(s.name);
        let want = match remote_len(s.url) {
            Some(n) => n,
            None => {
                // Fall back to the recorded size rather than refuse: a server
                // that will not answer HEAD may still answer GET.
                if s.bytes == 0 {
                    eprintln!("  {} — no size from server, skipping", s.name);
                    failed += 1;
                    continue;
                }
                s.bytes
            }
        };
        if std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0) == want {
            println!("  have {:<46} {}", s.name, human(want));
            continue;
        }
        println!("  get  {:<46} {}", s.name, human(want));
        // Resume until the file matches. `-C -` continues a partial transfer;
        // the loop is what makes a silent truncation self-correcting.
        let mut ok = false;
        for _ in 0..12 {
            let _ = std::process::Command::new("curl")
                .args([
                    "-s",
                    "--http1.1",
                    "-C",
                    "-",
                    "--retry",
                    "3",
                    "--retry-all-errors",
                    "--speed-limit",
                    "10000",
                    "--speed-time",
                    "90",
                    "-o",
                    path.to_str().unwrap_or_default(),
                    s.url,
                ])
                .status();
            if std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0) == want {
                ok = true;
                break;
            }
        }
        if !ok {
            let got = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            eprintln!("  FAIL {} — {} of {}", s.name, human(got), human(want));
            failed += 1;
        }
    }
    if failed > 0 {
        eprintln!("\n{failed} incomplete; run again to resume");
        std::process::exit(1);
    }
    println!("\nall sources present. Derived maps are built separately:");
    println!("  cargo run --release --features planet --example bake_planet_maps");
}

fn mime(p: &Path) -> &'static str {
    match p.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "tif" | "tiff" => "image/tiff",
        "json" => "application/json",
        "html" => "text/html; charset=utf-8",
        "txt" | "" => "text/plain; charset=utf-8",
        // .tex is this crate's own texture blob; nothing standard claims it.
        _ => "application/octet-stream",
    }
}

/// Serve the assets over HTTP, so a browser or another machine can read them
/// without a 6 GB copy.
///
/// Range requests are supported because the point of this is large files: a
/// tile is a slice of a blob, and a client that has to take the whole thing to
/// read one mip is no better off than copying the directory.
fn serve(assets: &Path, port: u16) {
    let root = match assets.canonicalize() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{}: {e}", assets.display());
            std::process::exit(1);
        }
    };
    let listener = match std::net::TcpListener::bind(("127.0.0.1", port)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("bind :{port}: {e}");
            std::process::exit(1);
        }
    };
    println!("serving {} on http://127.0.0.1:{port}/", root.display());
    for stream in listener.incoming().flatten() {
        let root = root.clone();
        std::thread::spawn(move || {
            let _ = handle(stream, &root);
        });
    }
}

fn handle(mut s: std::net::TcpStream, root: &Path) -> std::io::Result<()> {
    let mut buf = [0u8; 8192];
    let n = s.read(&mut buf)?;
    let req = String::from_utf8_lossy(&buf[..n]).to_string();
    let mut lines = req.lines();
    let start = lines.next().unwrap_or_default();
    let mut parts = start.split_whitespace();
    let (method, target) = (parts.next().unwrap_or(""), parts.next().unwrap_or("/"));
    if method != "GET" && method != "HEAD" {
        return reply(&mut s, 405, "text/plain", b"method not allowed", None);
    }
    let range = lines
        .find(|l| l.to_ascii_lowercase().starts_with("range:"))
        .and_then(|l| l.split('=').nth(1).map(str::to_string));

    // No traversal: resolve, then require the result to stay under the root.
    let rel = target
        .split('?')
        .next()
        .unwrap_or("/")
        .trim_start_matches('/');
    let path = root.join(rel);
    let Ok(path) = path.canonicalize() else {
        return reply(&mut s, 404, "text/plain", b"not found", None);
    };
    if !path.starts_with(root) {
        return reply(&mut s, 403, "text/plain", b"forbidden", None);
    }
    if path.is_dir() {
        let mut out = String::from("<pre>\n");
        let mut names: Vec<_> = std::fs::read_dir(&path)?
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        names.sort();
        for name in names {
            out.push_str(&format!("<a href=\"{rel}/{name}\">{name}</a>\n"));
        }
        out.push_str("</pre>\n");
        return reply(
            &mut s,
            200,
            "text/html; charset=utf-8",
            out.as_bytes(),
            None,
        );
    }

    let mut f = std::fs::File::open(&path)?;
    let len = f.metadata()?.len();
    let ct = mime(&path);
    if let Some(r) = range {
        // "bytes=START-[END]", which is all a tile reader ever sends.
        let mut it = r.trim_start_matches("bytes=").split('-');
        let start: u64 = it.next().unwrap_or("0").trim().parse().unwrap_or(0);
        let end: u64 = it
            .next()
            .and_then(|e| e.trim().parse().ok())
            .unwrap_or(len.saturating_sub(1))
            .min(len.saturating_sub(1));
        if start > end {
            return reply(&mut s, 416, "text/plain", b"bad range", None);
        }
        use std::io::Seek;
        f.seek(std::io::SeekFrom::Start(start))?;
        let mut body = vec![0u8; (end - start + 1) as usize];
        f.read_exact(&mut body)?;
        let cr = format!("bytes {start}-{end}/{len}");
        return reply(&mut s, 206, ct, &body, Some(&cr));
    }
    let mut body = Vec::with_capacity(len as usize);
    f.read_to_end(&mut body)?;
    reply(&mut s, 200, ct, &body, None)
}

fn reply(
    s: &mut std::net::TcpStream,
    code: u16,
    ct: &str,
    body: &[u8],
    content_range: Option<&str>,
) -> std::io::Result<()> {
    let reason = match code {
        200 => "OK",
        206 => "Partial Content",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        416 => "Range Not Satisfiable",
        _ => "OK",
    };
    let mut head = format!(
        "HTTP/1.1 {code} {reason}\r\nContent-Type: {ct}\r\nContent-Length: {}\r\n\
         Accept-Ranges: bytes\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n",
        body.len()
    );
    if let Some(cr) = content_range {
        head.push_str(&format!("Content-Range: {cr}\r\n"));
    }
    head.push_str("\r\n");
    s.write_all(head.as_bytes())?;
    s.write_all(body)?;
    s.flush()
}
