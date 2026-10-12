//! Every bundled asset, corrupted every way that is cheap to corrupt, through
//! its own parser.
//!
//! Six binary formats here are parsed by hand — Draco, USD crate files, glTF
//! binary, FBX, Alembic's Ogawa and KTX2 — and every one of them reads offsets
//! and lengths out of the file and then uses them. A length field is a number
//! an attacker controls, and the distance between "returns an error" and
//! "indexes out of bounds" is one missing bounds check in code that looks
//! identical either way.
//!
//! So this does not check that anything *parses*. It checks that nothing
//! panics: a malformed asset has to come back as `Err`, because a library that
//! aborts the process on a bad file cannot be put behind a file upload.
//!
//! It is deliberately cheap — a fraction of a second for the whole sweep —
//! because the value is in it running every time, not in it running deep. For
//! depth, point `cargo fuzz` at the same entry points.

use std::panic;

/// Every asset the crate carries, whatever format it is in.
fn bundled_assets() -> Vec<(String, Vec<u8>)> {
    let dirs = [
        "src/loaders/draco/testdata",
        "src/loaders/testdata",
        "src/loaders/testdata/ktx",
        "src/loaders/usd/testdata",
        "src/loaders/alembic/testdata",
        "src/loaders/fbx/testdata",
    ];
    let mut out = Vec::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_file() {
                if let Ok(b) = std::fs::read(&p) {
                    out.push((p.display().to_string(), b));
                }
            }
        }
    }
    out
}

/// Truncations, scattered corruption, and a length field blown up to 2 GB.
///
/// The last of those is the one worth having: a parser that trusts a count and
/// reserves on it turns a 3 KB file into an allocation the machine cannot
/// serve, which is a denial of service rather than a panic and would not show
/// up as one.
fn corruptions(bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    // Absolute lengths first, and they are the ones that matter: fractions of a
    // 120 KB asset are all still tens of kilobytes, so a sweep built only from
    // those never hands a header parser a file shorter than its own header.
    // Mutation-testing this caught exactly that — an unchecked `bytes[64]`
    // added to `decode_mesh` went undetected until these were added.
    for n in [0usize, 1, 2, 4, 8, 11, 12, 16, 24, 32, 48, 64, 96, 128, 256] {
        if n <= bytes.len() {
            out.push(bytes[..n].to_vec());
        }
    }
    for frac in [1usize, 2, 3, 7] {
        out.push(bytes[..bytes.len() * frac / 8].to_vec());
    }
    let mut seed = 0x1234_5678u32;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    for _ in 0..24 {
        let mut v = bytes.to_vec();
        let n = v.len();
        if n > 16 {
            for _ in 0..(n / 20).max(4) {
                let i = 16 + (rnd() as usize) % (n - 16);
                v[i] = (rnd() >> 7) as u8;
            }
        }
        out.push(v);
    }
    for off in [8usize, 12, 16, 20, 24, 28, 32] {
        if bytes.len() > off + 4 {
            let mut v = bytes.to_vec();
            v[off..off + 4].copy_from_slice(&0x7FFF_FFF0u32.to_le_bytes());
            out.push(v);
        }
    }
    out
}

/// Hand `bytes` to whichever parser owns that extension. Unknown extensions are
/// skipped rather than guessed at.
fn parse_by_extension(name: &str, bytes: &[u8]) -> bool {
    let lower = name.to_lowercase();
    if lower.ends_with(".drc") {
        let _ = threers::loaders::draco::decode_mesh(bytes);
    } else if lower.ends_with(".ktx2") {
        let _ = threers::loaders::ktx2::parse(bytes);
    } else if lower.ends_with(".fbx") {
        let _ = threers::loaders::FbxLoader::parse_scene(bytes);
    } else if lower.ends_with(".abc") {
        let _ = threers::loaders::alembic::Archive::open(bytes);
    } else if lower.ends_with(".glb") {
        let _ = threers::loaders::GltfLoader::parse_glb(bytes);
    } else if cfg!(feature = "usd")
        && (lower.ends_with(".usda") || lower.ends_with(".usdc") || lower.ends_with(".usdz"))
    {
        #[cfg(feature = "usd")]
        let _ = threers::loaders::UsdLoader::parse(bytes);
    } else {
        return false;
    }
    true
}

#[test]
fn a_malformed_asset_is_an_error_and_not_a_panic() {
    let assets = bundled_assets();
    assert!(
        assets.len() > 40,
        "only found {} bundled assets; the paths above have moved",
        assets.len()
    );

    // Quiet, or the sweep prints a backtrace per input it is *meant* to break.
    let previous = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));

    let mut checked = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for (name, bytes) in &assets {
        for (i, variant) in corruptions(bytes).into_iter().enumerate() {
            let tag = format!("{name} (corruption {i})");
            let ran = panic::catch_unwind(|| parse_by_extension(name, &variant));
            match ran {
                Ok(true) => checked += 1,
                Ok(false) => {}
                Err(_) => {
                    checked += 1;
                    failures.push(tag);
                }
            }
        }
    }
    panic::set_hook(previous);

    assert!(
        checked > 500,
        "only {checked} inputs reached a parser; the extension matching is wrong"
    );
    assert!(
        failures.is_empty(),
        "{} of {checked} malformed inputs panicked instead of returning an error:\n  {}",
        failures.len(),
        failures
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}
