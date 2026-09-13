//! The `.usdc` *crate* format — the binary layout large USD assets ship as.
//!
//! A crate file is a bootstrap header pointing at a table of contents, which
//! names half a dozen sections: a token table, a string table, the field and
//! fieldset tables that hold every authored value, the path table that gives
//! the prim tree its shape, and the spec table that ties them together. Most of
//! those sections are LZ4-compressed and several are additionally packed with
//! USD's own integer coder.
//!
//! This module handles the bootstrap and the table of contents — enough to
//! identify a crate file exactly, report its version, and say which sections it
//! holds. [`super::crate_read`] does the rest.
//!
//! Everything below the table of contents was worked out against files written
//! by OpenUSD's own `usdcat`, not from memory, and the fixtures in `testdata`
//! are those files. That distinction matters for a binary format: the type
//! enumeration in particular is not what you would guess — matrices are
//! numbered ahead of vectors — and a reader checked only against its own writer
//! would have agreed with itself all the way to the first real asset.

use super::UsdError;

/// The eight bytes every crate file starts with.
pub const MAGIC: &[u8; 8] = b"PXR-USDC";

/// What a crate file says about itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrateInfo {
    pub version: (u8, u8, u8),
    /// Section names in the order the table of contents lists them, with each
    /// one's offset and size.
    pub sections: Vec<(String, u64, u64)>,
}

impl CrateInfo {
    /// Whether a section is present.
    pub fn has(&self, name: &str) -> bool {
        self.sections.iter().any(|(n, _, _)| n == name)
    }
}

fn u64_at(b: &[u8], at: usize) -> Option<u64> {
    b.get(at..at + 8)
        .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
}

/// Read the bootstrap and table of contents.
///
/// The bootstrap is 88 bytes: the magic, three version bytes padded to eight,
/// the offset of the table of contents, and reserved space. The table itself is
/// a count followed by entries of a sixteen-byte name, an offset and a size.
pub fn info(bytes: &[u8]) -> Result<CrateInfo, UsdError> {
    if !bytes.starts_with(MAGIC) {
        return Err(UsdError::NotUsd);
    }
    if bytes.len() < 88 {
        return Err(UsdError::Corrupt("crate bootstrap is truncated"));
    }
    let version = (bytes[8], bytes[9], bytes[10]);
    let toc = u64_at(bytes, 16).ok_or(UsdError::Corrupt("missing toc offset"))? as usize;
    // Checked: an offset read out of the file can be anything, and adding to
    // it is how a corrupt one becomes a panic rather than an error.
    if toc == 0 || toc.checked_add(8).is_none_or(|end| end > bytes.len()) {
        return Err(UsdError::Corrupt("toc offset out of range"));
    }
    let count = u64_at(bytes, toc).ok_or(UsdError::Corrupt("missing toc count"))? as usize;
    // A crate has a handful of sections; a count in the millions means the
    // offset was wrong and the next read would be enormous.
    if count > 64 {
        return Err(UsdError::Corrupt("implausible section count"));
    }
    let mut sections = Vec::with_capacity(count);
    for i in 0..count {
        let at = toc + 8 + i * 32;
        let raw = bytes
            .get(at..at + 16)
            .ok_or(UsdError::Corrupt("truncated toc entry"))?;
        let name = String::from_utf8_lossy(raw)
            .trim_end_matches('\0')
            .to_string();
        let offset = u64_at(bytes, at + 16).ok_or(UsdError::Corrupt("truncated toc entry"))?;
        let size = u64_at(bytes, at + 24).ok_or(UsdError::Corrupt("truncated toc entry"))?;
        sections.push((name, offset, size));
    }
    Ok(CrateInfo { version, sections })
}

/// Read a crate file as a layer.
///
/// Not implemented — see the module docs for why. The error names the version
/// so a caller can say something useful rather than "unsupported".
pub fn read(bytes: &[u8]) -> Result<super::UsdLayer, UsdError> {
    let info = info(bytes)?;
    // The reader was written against 0.8.0 and the layout of the six sections
    // has been stable for far longer, so a newer minor version is worth
    // attempting rather than refusing. A newer *major* one is not.
    if info.version.0 != 0 {
        // Leaked deliberately: the message outlives the call, and one per file
        // that fails to open is not a leak worth a lifetime parameter.
        let message: &'static str = Box::leak(
            format!(
                "crate version {}.{}.{} is newer than this reader understands",
                info.version.0, info.version.1, info.version.2
            )
            .into_boxed_str(),
        );
        return Err(UsdError::UnsupportedCrate(message));
    }
    super::crate_read::read(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bootstrap and table of contents in the documented layout.
    fn crate_file(sections: &[(&str, u64, u64)]) -> Vec<u8> {
        let toc_at = 88u64;
        let mut out = vec![0u8; 88];
        out[..8].copy_from_slice(MAGIC);
        out[8] = 0;
        out[9] = 8;
        out[10] = 0;
        out[16..24].copy_from_slice(&toc_at.to_le_bytes());
        out.extend_from_slice(&(sections.len() as u64).to_le_bytes());
        for (name, offset, size) in sections {
            let mut padded = [0u8; 16];
            padded[..name.len()].copy_from_slice(name.as_bytes());
            out.extend_from_slice(&padded);
            out.extend_from_slice(&offset.to_le_bytes());
            out.extend_from_slice(&size.to_le_bytes());
        }
        out
    }

    #[test]
    fn the_bootstrap_and_toc_are_read() {
        let bytes = crate_file(&[("TOKENS", 200, 40), ("PATHS", 240, 80)]);
        let info = info(&bytes).unwrap();
        assert_eq!(info.version, (0, 8, 0));
        assert_eq!(info.sections.len(), 2);
        assert_eq!(info.sections[0], ("TOKENS".to_string(), 200, 40));
        assert!(info.has("PATHS"));
        assert!(!info.has("SPECS"));
    }

    #[test]
    fn a_version_this_reader_predates_is_declined_by_name() {
        let mut bytes = crate_file(&[]);
        bytes[8] = 9; // a major version that does not exist
        match read(&bytes) {
            Err(UsdError::UnsupportedCrate(why)) => {
                assert!(why.contains("9.8.0"), "the message should say what it found: {why}");
            }
            other => panic!("expected UnsupportedCrate, got {other:?}"),
        }
    }

    #[test]
    fn nonsense_is_rejected_rather_than_trusted() {
        assert_eq!(info(b"not usdc at all").unwrap_err(), UsdError::NotUsd);
        // Right magic, truncated body.
        assert!(info(&MAGIC[..]).is_err());
        // A toc offset past the end.
        let mut bytes = crate_file(&[("TOKENS", 0, 0)]);
        bytes[16..24].copy_from_slice(&9_999_999u64.to_le_bytes());
        assert!(info(&bytes).is_err());
    }

    #[test]
    fn an_implausible_section_count_is_not_allocated_for() {
        let mut bytes = crate_file(&[]);
        let toc = 88;
        bytes[toc..toc + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(info(&bytes).is_err());
    }
}
