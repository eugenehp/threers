//! Both *Drosophila* connectomes as one node per neuron, in the browser.
//!
//! The two trees under `connectome-fs` are directories of plain text — one
//! directory per neuron, one row per neuron in `index.tsv`. This crate reads
//! them, flattens the 350,825 neurons into little-endian arrays a browser can
//! fetch in one request each, and hosts the viewer that draws them with the
//! threers wasm build.
//!
//! Nothing here links `threers`: the renderer runs in the browser, and the Rust
//! side is a reader and a file server. It has no dependencies either — the trees
//! are text, the tables are typed arrays, and the two API routes are a file read
//! and a parse.
//!
//! ```no_run
//! use threers_connectome::{Release, Tree};
//!
//! let tree = Tree::read(std::path::Path::new("/Volumes/C4TB/connectome-fs"), &Release::ALL[0])?;
//! println!("{} placeable of {} neurons", tree.nodes.len(), tree.indexed);
//! # Ok::<(), std::io::Error>(())
//! ```

pub mod http;
pub mod tables;
pub mod tree;

pub use tables::{write_tables, Meta};
pub use tree::{Node, Release, Tree};

/// Where a neuron's node position came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placed {
    /// The column the release publishes — a soma for FlyWire, a centroid for
    /// Male CNS.
    Index,
    /// Computed here, from the neuron's own `skeleton.swc`.
    Skeleton,
}
