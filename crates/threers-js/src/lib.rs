//! Packaging crate for the **npm / Deno / Node ESM** distribution of threers.
//!
//! The published artifact is JavaScript + wasm (`dist/`), not this rlib.
//! Build and publish:
//!
//! ```bash
//! ./crates/threers-js/build.sh
//! cd crates/threers-js && npm publish
//! ```
//!
//! See the crate [`README.md`](../README.md).

#![doc = include_str!("../README.md")]

/// npm package version — keep in sync with `package.json`.
pub const NPM_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Canonical npm package name.
pub const NPM_NAME: &str = "threers";
