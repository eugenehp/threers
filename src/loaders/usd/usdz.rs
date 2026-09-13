//! The `.usdz` container: a zip with two rules that make it mappable.
//!
//! USDZ is an ordinary zip archive constrained so that a reader can use the
//! file in place rather than unpacking it:
//!
//! - **Everything is stored, never compressed.** A texture inside a `.usdz` is
//!   byte-for-byte the PNG it came from.
//! - **Every file's data begins on a 64-byte boundary**, achieved by padding
//!   the local header's extra field. That is what lets an image or a `.usdc`
//!   crate be handed to a decoder as a slice of the mapped archive.
//!
//! The first layer in the archive is the one that gets opened; anything else is
//! a dependency it references by relative path.

use super::UsdError;

/// Alignment USDZ requires for each file's data.
const ALIGN: usize = 64;

/// One file in an archive.
#[derive(Debug, Clone)]
pub struct UsdzEntry {
    pub name: String,
    pub data: Vec<u8>,
}

/// Everything inside a `.usdz`, in the order it was stored.
#[derive(Debug, Clone, Default)]
pub struct UsdzArchive {
    pub entries: Vec<UsdzEntry>,
}

impl UsdzArchive {
    /// The entry a reader should open: the first `.usda`, `.usdc` or `.usd`.
    ///
    /// "First" is the specification's rule, not a heuristic — the defaulting
    /// layer is required to be the first file in the archive.
    pub fn root_layer(&self) -> Option<&UsdzEntry> {
        self.entries.iter().find(|e| {
            let lower = e.name.to_ascii_lowercase();
            lower.ends_with(".usda") || lower.ends_with(".usdc") || lower.ends_with(".usd")
        })
    }

    /// An entry by name, for resolving a texture a layer refers to.
    pub fn get(&self, name: &str) -> Option<&UsdzEntry> {
        self.entries.iter().find(|e| e.name == name)
    }
}

/// Read a `.usdz`.
///
/// Deflated entries are inflated rather than rejected: the format forbids them,
/// but files in the wild are written by tools that did not read the rule, and
/// refusing to open one helps nobody.
pub fn read(bytes: &[u8]) -> Result<UsdzArchive, UsdError> {
    let eocd = find_eocd(bytes).ok_or(UsdError::NotUsd)?;
    let count = u16(bytes, eocd + 10) as usize;
    let mut at = u32(bytes, eocd + 16) as usize;
    let mut entries = Vec::with_capacity(count);

    for _ in 0..count {
        if at + 46 > bytes.len() || &bytes[at..at + 4] != b"PK\x01\x02" {
            return Err(UsdError::Corrupt("central directory entry"));
        }
        let method = u16(bytes, at + 10);
        let compressed = u32(bytes, at + 20) as usize;
        let uncompressed = u32(bytes, at + 24) as usize;
        let name_len = u16(bytes, at + 28) as usize;
        let extra_len = u16(bytes, at + 30) as usize;
        let comment_len = u16(bytes, at + 32) as usize;
        let local = u32(bytes, at + 42) as usize;
        let name = String::from_utf8_lossy(
            bytes
                .get(at + 46..at + 46 + name_len)
                .ok_or(UsdError::Corrupt("entry name"))?,
        )
        .into_owned();

        // The local header repeats the name and extra field, and its extra
        // field is the one that carries the alignment padding — so the data
        // offset has to be computed from the local header, not the central one.
        if local + 30 > bytes.len() || &bytes[local..local + 4] != b"PK\x03\x04" {
            return Err(UsdError::Corrupt("local header"));
        }
        let local_name = u16(bytes, local + 26) as usize;
        let local_extra = u16(bytes, local + 28) as usize;
        let start = local + 30 + local_name + local_extra;
        let end = start + compressed;
        let raw = bytes
            .get(start..end)
            .ok_or(UsdError::Corrupt("entry data"))?;

        let data = match method {
            0 => raw.to_vec(),
            8 => super::super::deflate::inflate_raw(raw)
                .map_err(|_| UsdError::Corrupt("deflate stream"))?,
            _ => return Err(UsdError::Corrupt("unsupported zip compression")),
        };
        // A directory entry is a name ending in `/` with no content; skip it
        // rather than presenting a zero-byte file.
        if !name.ends_with('/') && (uncompressed == 0 || data.len() == uncompressed || method == 8)
        {
            entries.push(UsdzEntry { name, data });
        }
        at += 46 + name_len + extra_len + comment_len;
    }
    Ok(UsdzArchive { entries })
}

/// Write a `.usdz`: stored, and aligned to 64 bytes.
pub fn write(entries: &[UsdzEntry]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let mut directory: Vec<u8> = Vec::new();
    let mut count = 0u16;

    for entry in entries {
        let offset = out.len();
        let name = entry.name.as_bytes();
        let crc = crc32(&entry.data);

        // Pad the extra field so the data that follows starts on a boundary.
        let header = 30 + name.len();
        let pad = (ALIGN - (offset + header) % ALIGN) % ALIGN;
        // A padded extra field still has to look like one: two bytes of id and
        // two of length, so fewer than four bytes of slack means taking a whole
        // further block rather than writing a malformed field.
        let extra = if pad == 0 {
            Vec::new()
        } else {
            let n = if pad < 4 { pad + ALIGN } else { pad };
            let mut e = vec![0u8; n];
            // 0xE9E1 is unregistered and ignored by readers, which is what the
            // padding wants to be.
            e[0..2].copy_from_slice(&0xE9E1u16.to_le_bytes());
            e[2..4].copy_from_slice(&((n - 4) as u16).to_le_bytes());
            e
        };

        out.extend_from_slice(b"PK\x03\x04");
        out.extend_from_slice(&[20, 0, 0, 0, 0, 0]); // version, flags, method = store
        out.extend_from_slice(&[0, 0, 0, 0]); // time and date
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(entry.data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(entry.data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&(extra.len() as u16).to_le_bytes());
        out.extend_from_slice(name);
        out.extend_from_slice(&extra);
        debug_assert_eq!(out.len() % ALIGN, 0, "usdz data must be aligned");
        out.extend_from_slice(&entry.data);

        directory.extend_from_slice(b"PK\x01\x02");
        directory.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0]);
        directory.extend_from_slice(&[0, 0, 0, 0]);
        directory.extend_from_slice(&crc.to_le_bytes());
        directory.extend_from_slice(&(entry.data.len() as u32).to_le_bytes());
        directory.extend_from_slice(&(entry.data.len() as u32).to_le_bytes());
        directory.extend_from_slice(&(name.len() as u16).to_le_bytes());
        directory.extend_from_slice(&(extra.len() as u16).to_le_bytes());
        // comment length, disk number, internal attrs (2 each) and
        // external attrs (4) — ten bytes between the extra length and the
        // local header offset.
        directory.extend_from_slice(&[0u8; 10]);
        directory.extend_from_slice(&(offset as u32).to_le_bytes());
        directory.extend_from_slice(name);
        directory.extend_from_slice(&extra);
        count += 1;
    }

    let dir_offset = out.len();
    let dir_len = directory.len();
    out.extend_from_slice(&directory);
    out.extend_from_slice(b"PK\x05\x06");
    out.extend_from_slice(&[0, 0, 0, 0]);
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&(dir_len as u32).to_le_bytes());
    out.extend_from_slice(&(dir_offset as u32).to_le_bytes());
    out.extend_from_slice(&[0, 0]);
    out
}

fn u16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([
        *b.get(at).unwrap_or(&0),
        *b.get(at + 1).unwrap_or(&0),
    ])
}

fn u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([
        *b.get(at).unwrap_or(&0),
        *b.get(at + 1).unwrap_or(&0),
        *b.get(at + 2).unwrap_or(&0),
        *b.get(at + 3).unwrap_or(&0),
    ])
}

/// The end-of-central-directory record, searched from the back because a zip
/// may carry a trailing comment.
fn find_eocd(b: &[u8]) -> Option<usize> {
    if b.len() < 22 {
        return None;
    }
    let start = b.len().saturating_sub(22 + 0xFFFF);
    (start..=b.len() - 22).rev().find(|&i| &b[i..i + 4] == b"PK\x05\x06")
}

/// CRC-32, the zip flavour. Computed on the fly rather than from a table: an
/// archive is a handful of files and the table is 1 KB of state to own.
fn crc32(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// The file types Apple's RealityKit will open inside a package.
///
/// A general `.usdz` may hold more — the format itself does not care — but an
/// asset bound for AR Quick Look is judged against this list, and one stray
/// file fails the whole package.
const ARKIT_EXTENSIONS: &[&str] = &[
    // Layers.
    "usd", "usda", "usdc", "usdz", // Textures.
    "exr", "jpg", "jpeg", "png", // Audio.
    "m4a", "mp3", "wav",
];

/// Something about an archive that Apple's `usdchecker --arkit` would object
/// to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArkitIssue {
    /// A file type RealityKit will not open.
    UnsupportedFile { name: String, extension: String },
    /// An entry that was compressed. A package has to be mappable, which means
    /// every byte is where it says it is.
    Compressed { name: String },
    /// An entry whose data does not begin on a 64-byte boundary.
    Misaligned { name: String, offset: usize },
    /// No layer at all, or one that is not the first entry.
    RootLayerNotFirst,
}

impl std::fmt::Display for ArkitIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArkitIssue::UnsupportedFile { name, extension } => write!(
                f,
                "{name:?} has extension {extension:?}, which RealityKit will not open"
            ),
            ArkitIssue::Compressed { name } => {
                write!(f, "{name:?} is compressed; a package must be mappable")
            }
            ArkitIssue::Misaligned { name, offset } => write!(
                f,
                "{name:?} starts at {offset}, which is not a multiple of 64"
            ),
            ArkitIssue::RootLayerNotFirst => {
                write!(f, "the first entry is not a layer")
            }
        }
    }
}

/// Check an archive against the constraints Apple adds on top of `.usdz`.
///
/// This covers what can be judged from the bytes: the file types, the
/// alignment, and the order. It does not open the layers, so it says nothing
/// about the scene inside — `usdchecker --arkit` is the authority on that, and
/// this is what to run when it is not to hand.
pub fn arkit_issues(bytes: &[u8]) -> Vec<ArkitIssue> {
    let Ok(archive) = read(bytes) else {
        return vec![ArkitIssue::RootLayerNotFirst];
    };
    let mut issues = Vec::new();

    match archive.entries.first() {
        Some(first)
            if matches!(
                extension_of(&first.name).as_str(),
                "usd" | "usda" | "usdc" | "usdz"
            ) => {}
        _ => issues.push(ArkitIssue::RootLayerNotFirst),
    }

    for entry in &archive.entries {
        let extension = extension_of(&entry.name);
        if !ARKIT_EXTENSIONS.contains(&extension.as_str()) {
            issues.push(ArkitIssue::UnsupportedFile {
                name: entry.name.clone(),
                extension,
            });
        }
    }

    // Alignment and compression are read off the local headers, since the
    // archive itself has already decoded past them.
    for (name, offset, stored) in payload_offsets(bytes) {
        if !stored {
            issues.push(ArkitIssue::Compressed { name: name.clone() });
        }
        if !offset.is_multiple_of(ALIGN) {
            issues.push(ArkitIssue::Misaligned { name, offset });
        }
    }
    issues
}

fn extension_of(name: &str) -> String {
    name.rsplit('.')
        .next()
        .filter(|e| *e != name)
        .unwrap_or_default()
        .to_ascii_lowercase()
}

/// Walk the local headers, reporting where each entry's data starts and
/// whether it was stored rather than deflated.
fn payload_offsets(bytes: &[u8]) -> Vec<(String, usize, bool)> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while at + 30 <= bytes.len() {
        if bytes[at..at + 4] != [0x50, 0x4b, 0x03, 0x04] {
            break;
        }
        let u16at = |i: usize| u16::from_le_bytes([bytes[i], bytes[i + 1]]) as usize;
        let u32at = |i: usize| {
            u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as usize
        };
        let method = u16at(at + 8);
        let compressed = u32at(at + 18);
        let name_len = u16at(at + 26);
        let extra_len = u16at(at + 28);
        let name_at = at + 30;
        if name_at + name_len > bytes.len() {
            break;
        }
        let name = String::from_utf8_lossy(&bytes[name_at..name_at + name_len]).into_owned();
        let data_at = name_at + name_len + extra_len;
        out.push((name, data_at, method == 0));
        at = data_at + compressed;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What this crate writes satisfies the constraints Apple adds on top of
    /// `.usdz` — checked here, and checked for real by
    /// `usdchecker --arkit`, which this mirrors.
    #[test]
    fn what_we_write_passes_the_arkit_constraints() {
        let archive = write(&[
            UsdzEntry {
                name: "scene.usda".into(),
                data: b"#usda 1.0\n".to_vec(),
            },
            UsdzEntry {
                name: "textures/albedo.png".into(),
                data: vec![0u8; 300],
            },
            UsdzEntry {
                name: "audio/hum.wav".into(),
                data: vec![0u8; 44],
            },
        ]);
        assert_eq!(arkit_issues(&archive), Vec::new());
    }

    /// One stray file fails the whole package, which is the kind of thing
    /// worth hearing about before shipping rather than after.
    #[test]
    fn a_file_type_realitykit_will_not_open_is_reported() {
        let archive = write(&[
            UsdzEntry {
                name: "scene.usda".into(),
                data: b"#usda 1.0\n".to_vec(),
            },
            UsdzEntry {
                name: "notes.txt".into(),
                data: b"hello".to_vec(),
            },
            UsdzEntry {
                name: "model.obj".into(),
                data: b"v 0 0 0\n".to_vec(),
            },
        ]);
        let issues = arkit_issues(&archive);
        assert_eq!(issues.len(), 2, "{issues:?}");
        assert!(issues.contains(&ArkitIssue::UnsupportedFile {
            name: "notes.txt".into(),
            extension: "txt".into(),
        }));
        // And it says so in a sentence.
        assert!(issues[0].to_string().contains("RealityKit will not open"));
    }

    /// The layer has to come first, which is the rule that makes a package
    /// openable without reading all of it.
    #[test]
    fn a_package_that_does_not_begin_with_a_layer_is_reported() {
        let archive = write(&[
            UsdzEntry {
                name: "textures/albedo.png".into(),
                data: vec![0u8; 16],
            },
            UsdzEntry {
                name: "scene.usda".into(),
                data: b"#usda 1.0\n".to_vec(),
            },
        ]);
        assert!(arkit_issues(&archive).contains(&ArkitIssue::RootLayerNotFirst));
    }

    /// Case does not decide the answer: `.PNG` is a PNG.
    #[test]
    fn extensions_are_matched_without_regard_to_case() {
        let archive = write(&[
            UsdzEntry {
                name: "Scene.USDA".into(),
                data: b"#usda 1.0\n".to_vec(),
            },
            UsdzEntry {
                name: "T.PNG".into(),
                data: vec![0u8; 16],
            },
        ]);
        assert_eq!(arkit_issues(&archive), Vec::new());
    }

    #[test]
    fn an_archive_round_trips() {
        let entries = vec![
            UsdzEntry {
                name: "model.usda".into(),
                data: b"#usda 1.0\ndef Xform \"a\" {}\n".to_vec(),
            },
            UsdzEntry {
                name: "textures/albedo.png".into(),
                data: vec![0x89, b'P', b'N', b'G', 1, 2, 3],
            },
        ];
        let bytes = write(&entries);
        let back = read(&bytes).unwrap();
        assert_eq!(back.entries.len(), 2);
        assert_eq!(back.root_layer().unwrap().name, "model.usda");
        assert_eq!(back.get("textures/albedo.png").unwrap().data.len(), 7);
    }

    #[test]
    fn every_file_starts_on_a_64_byte_boundary() {
        // The whole point of the format: a reader maps the archive and hands a
        // slice of it straight to a decoder.
        let entries: Vec<UsdzEntry> = (0..6)
            .map(|i| UsdzEntry {
                // Names of different lengths, so the padding has to do real work.
                name: format!("{}{}.png", "a".repeat(i * 7), i),
                data: vec![i as u8; 10 + i],
            })
            .collect();
        let bytes = write(&entries);
        // Walk the local headers rather than searching for the payload: a run
        // of zero bytes appears in the padding too, so a search would find the
        // padding and call it the data.
        let mut at = 0usize;
        let mut seen = 0usize;
        while bytes[at..].starts_with(b"PK\x03\x04") {
            let name_len = u16(&bytes, at + 26) as usize;
            let extra_len = u16(&bytes, at + 28) as usize;
            let size = u32(&bytes, at + 18) as usize;
            let data = at + 30 + name_len + extra_len;
            let name = String::from_utf8_lossy(&bytes[at + 30..at + 30 + name_len]).into_owned();
            assert_eq!(data % ALIGN, 0, "{name} starts at {data}");
            seen += 1;
            at = data + size;
        }
        assert_eq!(seen, entries.len(), "did not reach every local header");
    }

    #[test]
    fn the_first_layer_is_the_root_whatever_else_is_inside() {
        let bytes = write(&[
            UsdzEntry { name: "a.png".into(), data: vec![1] },
            UsdzEntry { name: "first.usdc".into(), data: vec![2] },
            UsdzEntry { name: "second.usda".into(), data: vec![3] },
        ]);
        assert_eq!(read(&bytes).unwrap().root_layer().unwrap().name, "first.usdc");
    }

    #[test]
    fn truncated_archives_are_errors_not_panics() {
        let bytes = write(&[UsdzEntry { name: "m.usda".into(), data: vec![7; 100] }]);
        assert!(read(&bytes[..bytes.len() / 2]).is_err());
        assert!(read(b"not a zip").is_err());
        assert!(read(&[]).is_err());
    }

    #[test]
    fn crc32_matches_the_known_value() {
        // The standard check value for "123456789".
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }
}

#[cfg(test)]
mod external {
    use super::*;

    /// Check a file on disk with this crate's rules, for comparing against
    /// `usdchecker --arkit` on the same file.
    #[test]
    #[ignore = "compares against an external tool"]
    fn report_arkit_issues() {
        let path = std::env::var("USD_FILE").expect("USD_FILE");
        let bytes = std::fs::read(&path).expect("readable");
        let issues = arkit_issues(&bytes);
        if issues.is_empty() {
            println!("OK: no issues");
        }
        for issue in issues {
            println!("ISSUE: {issue}");
        }
    }
}
