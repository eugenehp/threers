//! Universal Scene Description — `.usda`, `.usdc` and `.usdz`, read and written.
//!
//! Three file formats for one data model, and the difference between them is
//! only how the bytes are laid out:
//!
//! | | What it is |
//! |---|---|
//! | `.usda` | The text form. A layer as ASCII, readable and diffable. |
//! | `.usdc` | The *crate* form. The same layer as a binary table of tokens, fields and specs — what large assets actually ship as. |
//! | `.usdz` | A zip of the above plus its textures, stored uncompressed and aligned so a reader can map it without unpacking. |
//!
//! All three are read and written here in Rust with no external dependency,
//! which is what keeps them available on wasm where OpenUSD's C++ cannot go.
//!
//! Behind the `usd` feature. It is pure code, so enabling it costs compile time
//! and nothing else.
//!
//! ```toml
//! threers = { version = "0.0.5", features = ["usd"] }
//! ```
//!
//! ```no_run
//! use threers::loaders::UsdLoader;
//!
//! // Any of the three, told apart by content rather than by file name.
//! let scene = UsdLoader::parse(&std::fs::read("model.usdz").unwrap()).unwrap();
//! println!("{} root prims", scene.roots.len());
//! ```
//!
//! # What is carried across
//!
//! Meshes (`UsdGeomMesh`) with their points, face counts and indices, normals
//! and `primvars:st`; transforms (`UsdGeomXform`) as translate/rotate/scale ops
//! or a matrix; the prim hierarchy; and `UsdPreviewSurface` materials. Polygons
//! of any size are triangulated on the way in, since that is what a
//! [`BufferGeometry`](crate::core::BufferGeometry) holds.
//!
//! What a layer says about anything else — physics, skeletons, variants — is
//! parsed and kept rather than discarded, so [`UsdLayer`] round-trips through
//! the text writer even where this crate has no opinion about the schema.
//!
//! # Animation
//!
//! `timeSamples` are read from both the text and the binary form and become an
//! [`AnimationClip`](crate::animation::AnimationClip) on the returned
//! [`UsdScene`], with the scene itself posed at the first frame of the layer's
//! range. [`to_scene_at`] builds any other instant, and
//! [`UsdLayer::at_time`] resolves a whole layer to one moment — which is what
//! USD means by evaluating at a time code.
//!
//! Going the other way, [`animated_scene_to_usda`] and
//! [`animated_scene_to_usdz`] write clips back out as `timeSamples` on the
//! transform ops they drive.
//!
//! USD counts in time codes and this crate counts in seconds; the layer's
//! `timeCodesPerSecond` converts between them in both directions.

mod clips;
mod compose;
mod crate_read;
mod crate_write;
mod export;
mod expr;
mod ints;
mod lz4;
mod parse;
mod schemas;
mod skel;
mod subdiv;
mod scene;
mod shade;
mod usdc;
mod usdz;
mod value;
mod write;

pub use parse::{Specifier, UsdLayer, UsdPrim, UsdProperty, UsdVariantSet};
pub use value::{UsdReference, UsdValue};
pub use export::{
    animated_scene_to_layer, animated_scene_to_layer_with_textures, animated_scene_to_usda,
    geometry_to_layer, geometry_to_usda, material_prim, material_prim_with_textures, mesh_prim,
    scene_to_layer, scene_to_layer_with_textures, scene_to_usda, ExportedTexture,
    UsdExportOptions,
};
pub use crate::core::{MorphAttributes, MorphTarget};
pub use scene::attach_textures;
#[cfg(not(target_arch = "wasm32"))]
pub use scene::attach_textures_from_dir;
pub use scene::{
    mesh_geometry, mesh_geometry_at, to_scene, to_scene_at, to_scene_with, SceneOptions, UsdScene,
};
pub use clips::ClipSet;
pub use shade::{TextureRequest, TextureSlot, UsdMaterial};
pub use compose::{
    compose, mask, resolve_path, AssetResolver, ComposeOptions, MemoryResolver,
};
#[cfg(not(target_arch = "wasm32"))]
pub use compose::FileResolver;
pub use crate_write::write as layer_to_usdc;
pub use usdc::{info as usdc_info, CrateInfo};
pub use usdz::{arkit_issues, ArkitIssue, UsdzArchive, UsdzEntry};
pub use write::layer_to_usda;

/// Why a USD document could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UsdError {
    /// The text did not parse. `line` is 1-based where it is known.
    Syntax { line: usize, what: &'static str },
    /// The bytes are not USD in any of its three forms.
    NotUsd,
    /// A `.usdz` archive with no layer inside it.
    EmptyArchive,
    /// A `.usdc` crate this reader does not handle — the version is newer than
    /// the format it was written against.
    UnsupportedCrate(&'static str),
    /// The file is structurally USD but truncated or internally inconsistent.
    Corrupt(&'static str),
}

impl std::fmt::Display for UsdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UsdError::Syntax { line, what } => write!(f, "usda syntax error at line {line}: {what}"),
            UsdError::NotUsd => write!(f, "not a USD file"),
            UsdError::EmptyArchive => write!(f, "usdz archive contains no layer"),
            UsdError::UnsupportedCrate(why) => write!(f, "unsupported usdc crate: {why}"),
            UsdError::Corrupt(why) => write!(f, "corrupt USD file: {why}"),
        }
    }
}

impl std::error::Error for UsdError {}

/// Which of the three forms a byte string is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsdFormat {
    Usda,
    Usdc,
    Usdz,
}

/// Tell the three apart by what is in the bytes, not by the file name.
///
/// A `.usdz` is a zip and starts `PK`; a `.usdc` starts with the crate magic;
/// anything else that looks like text is treated as `.usda`, whose own `#usda`
/// header is a comment and therefore optional in practice.
pub fn sniff(bytes: &[u8]) -> Option<UsdFormat> {
    if bytes.starts_with(b"PK\x03\x04") || bytes.starts_with(b"PK\x05\x06") {
        return Some(UsdFormat::Usdz);
    }
    if bytes.starts_with(b"PXR-USDC") {
        return Some(UsdFormat::Usdc);
    }
    // Text: either the header is there, or the first non-blank line begins a
    // prim or a metadata block.
    let head = &bytes[..bytes.len().min(4096)];
    let text = std::str::from_utf8(head).ok()?;
    if text.starts_with("#usda") {
        return Some(UsdFormat::Usda);
    }
    let first = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))?;
    (first.starts_with("def ")
        || first.starts_with("over ")
        || first.starts_with("class ")
        || first.starts_with('('))
    .then_some(UsdFormat::Usda)
}

/// Reads any of the three USD forms.
pub struct UsdLoader;

impl UsdLoader {
    /// Parse USD bytes into a scene, whichever of the three forms they are.
    ///
    /// A `.usdz` arrives with its textures already loaded: the package carries
    /// them, so there is nothing for the caller to resolve. A `.usda` or
    /// `.usdc` names files beside itself that this cannot reach, and reports
    /// them in [`UsdScene::textures`] instead.
    pub fn parse(bytes: &[u8]) -> Result<UsdScene, UsdError> {
        let mut scene = scene::to_scene(&Self::parse_layer(bytes)?);
        if sniff(bytes) == Some(UsdFormat::Usdz) {
            if let Ok(archive) = usdz::read(bytes) {
                scene::attach_textures(&mut scene, &archive);
            }
        }
        Ok(scene)
    }

    /// Parse to a layer, without mapping it onto the scene graph — for callers
    /// that want the document rather than the geometry.
    pub fn parse_layer(bytes: &[u8]) -> Result<UsdLayer, UsdError> {
        match sniff(bytes).ok_or(UsdError::NotUsd)? {
            UsdFormat::Usda => {
                let text = std::str::from_utf8(bytes).map_err(|_| UsdError::NotUsd)?;
                parse::parse(text)
            }
            UsdFormat::Usdc => usdc::read(bytes),
            UsdFormat::Usdz => {
                let archive = usdz::read(bytes)?;
                let layer = archive.root_layer().ok_or(UsdError::EmptyArchive)?;
                // The layer inside may itself be either form, so this recurses
                // exactly once — a `.usdz` cannot contain a `.usdz`.
                Self::parse_layer(&layer.data)
            }
        }
    }

    /// The archive a `.usdz` is, for reaching the textures beside the layer.
    pub fn parse_archive(bytes: &[u8]) -> Result<UsdzArchive, UsdError> {
        usdz::read(bytes)
    }

    /// Open a *composed* stage rather than a single layer.
    ///
    /// [`parse`](Self::parse) reads one layer and stops. This follows what the
    /// layer points at — sublayers, references, payloads, variants, inherits —
    /// and hands back the scene those compose to, which for any asset built by
    /// a pipeline is the only reading of it that means anything.
    ///
    /// `name` is what the root layer is called, so that the relative paths
    /// inside it resolve; `resolver` says what a name means.
    ///
    /// ```no_run
    /// use threers::loaders::{ComposeOptions, FileResolver, UsdLoader};
    ///
    /// let bytes = std::fs::read("shots/010/shot.usda").unwrap();
    /// let scene = UsdLoader::open(
    ///     &bytes,
    ///     "shots/010/shot.usda",
    ///     &FileResolver,
    ///     &ComposeOptions::default(),
    /// )
    /// .unwrap();
    /// ```
    pub fn open(
        bytes: &[u8],
        name: &str,
        resolver: &dyn AssetResolver,
        options: &ComposeOptions,
    ) -> Result<UsdScene, UsdError> {
        Ok(scene::to_scene(&Self::open_layer(
            bytes, name, resolver, options,
        )?))
    }

    /// The composed layer, without mapping it onto the scene graph.
    pub fn open_layer(
        bytes: &[u8],
        name: &str,
        resolver: &dyn AssetResolver,
        options: &ComposeOptions,
    ) -> Result<UsdLayer, UsdError> {
        let root = Self::parse_layer(bytes)?;
        compose::compose(&root, name, resolver, options)
    }

    /// Open a `.usdz` as a composed stage, resolving arcs inside the archive.
    ///
    /// A `.usdz` whose root layer references others in the same archive — which
    /// is how a packaged asset with variants arrives — needs composing against
    /// the archive itself, not against the filesystem.
    pub fn open_archive(bytes: &[u8], options: &ComposeOptions) -> Result<UsdScene, UsdError> {
        let archive = usdz::read(bytes)?;
        let root = archive.root_layer().ok_or(UsdError::EmptyArchive)?;
        let name = root.name.clone();
        let data = root.data.clone();
        let layer = Self::parse_layer(&data)?;
        let mut scene = scene::to_scene(&compose::compose(&layer, &name, &archive, options)?);
        scene::attach_textures(&mut scene, &archive);
        Ok(scene)
    }
}

/// A document on its way out, in whichever of the three forms is wanted.
///
/// A USD document is not only a layer: a textured one is a layer *and* the
/// images it names. The three forms differ in what becomes of those. A
/// `.usdz` carries them, because a package is required to be self-contained.
/// A `.usda` or a `.usdc` refers to them by relative path, so they have to
/// land beside the file or every map dangles — which is what the plain
/// `scene_to_usdc` did, handing back bytes and quietly dropping the pictures
/// they referred to.
///
/// ```no_run
/// # use threers::loaders::usd::UsdExport;
/// # let (arena, roots) = (threers::core::ObjectArena::new(), vec![]);
/// let out = UsdExport::scene(&arena, &roots);
/// std::fs::write("model.usdz", out.usdz()).unwrap();   // carries its images
/// out.write_to("model.usda", std::path::Path::new("out")).unwrap();  // and beside it
/// ```
pub struct UsdExport {
    /// The composed document.
    pub layer: UsdLayer,
    /// The images the layer refers to, by the relative path written into it.
    pub textures: Vec<export::ExportedTexture>,
}

impl UsdExport {
    /// One geometry, under a name.
    pub fn geometry(g: &crate::core::BufferGeometry, name: &str) -> Self {
        Self::geometry_with(g, name, &UsdExportOptions::default())
    }

    /// The same, saying what the units and up-axis are.
    pub fn geometry_with(
        g: &crate::core::BufferGeometry,
        name: &str,
        options: &UsdExportOptions,
    ) -> Self {
        Self {
            layer: export::geometry_to_layer(g, name, options),
            textures: Vec::new(),
        }
    }

    /// A whole scene graph.
    pub fn scene(arena: &crate::core::ObjectArena, roots: &[crate::core::ObjectId]) -> Self {
        Self::scene_with(arena, roots, &UsdExportOptions::default())
    }

    /// The same, saying what the units and up-axis are.
    pub fn scene_with(
        arena: &crate::core::ObjectArena,
        roots: &[crate::core::ObjectId],
        options: &UsdExportOptions,
    ) -> Self {
        let (layer, textures) = export::scene_to_layer_with_textures(arena, roots, options);
        Self { layer, textures }
    }

    /// A scene graph with its animation.
    pub fn animated(
        arena: &crate::core::ObjectArena,
        roots: &[crate::core::ObjectId],
        clips: &[crate::animation::AnimationClip],
    ) -> Self {
        Self::animated_with(arena, roots, clips, &UsdExportOptions::default())
    }

    /// The same, saying what the units, up-axis and frame rate are.
    pub fn animated_with(
        arena: &crate::core::ObjectArena,
        roots: &[crate::core::ObjectId],
        clips: &[crate::animation::AnimationClip],
        options: &UsdExportOptions,
    ) -> Self {
        let (layer, textures) =
            export::animated_scene_to_layer_with_textures(arena, roots, clips, options);
        Self { layer, textures }
    }

    /// A layer that is already composed, with no images of its own.
    pub fn layer(layer: UsdLayer) -> Self {
        Self {
            layer,
            textures: Vec::new(),
        }
    }

    /// The document as `.usda` text.
    pub fn usda(&self) -> String {
        write::layer_to_usda(&self.layer)
    }

    /// The document as a `.usdc` crate.
    pub fn usdc(&self) -> Vec<u8> {
        crate_write::write(&self.layer)
    }

    /// The document as a `.usdz` package, images included.
    ///
    /// The layer inside is a crate, which is what a package from a phone or a
    /// DCC tool holds; [`usdz_as`](Self::usdz_as) writes text instead.
    pub fn usdz(&self) -> Vec<u8> {
        self.usdz_as(UsdzLayer::Crate, &[])
    }

    /// The same, choosing the form of the layer inside and adding extra files.
    pub fn usdz_as(&self, form: UsdzLayer, extras: &[UsdzEntry]) -> Vec<u8> {
        let (name, data) = match form {
            UsdzLayer::Text => ("scene.usda", self.usda().into_bytes()),
            UsdzLayer::Crate => ("scene.usdc", self.usdc()),
        };
        let mut entries = vec![UsdzEntry {
            name: name.into(),
            data,
        }];
        entries.extend(self.textures.iter().map(|t| UsdzEntry {
            name: t.path.clone(),
            data: t.data.clone(),
        }));
        entries.extend_from_slice(extras);
        usdz::write(&entries)
    }

    /// Write the document to `path`, with its images beside it.
    ///
    /// The form is taken from the extension — `.usda`, `.usdc` or `.usdz`.
    /// For the first two the images are written relative to `base`, which is
    /// the directory the layer's own relative paths resolve against; for a
    /// `.usdz` they go inside and `base` is unused.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn write_to(&self, path: impl AsRef<std::path::Path>, base: &std::path::Path) -> std::io::Result<()> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("usda")
            .to_ascii_lowercase();
        match extension.as_str() {
            "usdz" => return std::fs::write(path, self.usdz()),
            "usdc" => std::fs::write(path, self.usdc())?,
            _ => std::fs::write(path, self.usda())?,
        }
        for texture in &self.textures {
            let at = base.join(&texture.path);
            if let Some(parent) = at.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(at, &texture.data)?;
        }
        Ok(())
    }
}

/// One geometry as a `.usdc` crate file.
///
/// The binary form of [`geometry_to_usda`], byte for byte the same document.
pub fn geometry_to_usdc(g: &crate::core::BufferGeometry, name: &str) -> Vec<u8> {
    layer_to_usdc(&export::geometry_to_layer(
        g,
        name,
        &UsdExportOptions::default(),
    ))
}

/// A scene graph as a `.usdc` crate file.
/// Any images the materials refer to are *not* returned; a textured scene
/// wants [`UsdExport`], which keeps the two together.
pub fn scene_to_usdc(
    arena: &crate::core::ObjectArena,
    roots: &[crate::core::ObjectId],
) -> Vec<u8> {
    UsdExport::scene(arena, roots).usdc()
}

/// A scene graph and its animation as a `.usdc` crate file.
pub fn animated_scene_to_usdc(
    arena: &crate::core::ObjectArena,
    roots: &[crate::core::ObjectId],
    clips: &[crate::animation::AnimationClip],
) -> Vec<u8> {
    UsdExport::animated(arena, roots, clips).usdc()
}

/// Which form the layer inside a `.usdz` takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UsdzLayer {
    /// `.usda`. Readable with `unzip -p`, accepted everywhere, and the default.
    #[default]
    Text,
    /// `.usdc`. Smaller, and what large assets ship as.
    Crate,
}

/// A layer as a `.usdz`, in whichever form, with anything travelling beside it.
pub fn layer_to_usdz(layer: &UsdLayer, form: UsdzLayer, extras: &[UsdzEntry]) -> Vec<u8> {
    let (name, data) = match form {
        UsdzLayer::Text => ("scene.usda", layer_to_usda(layer).into_bytes()),
        UsdzLayer::Crate => ("scene.usdc", layer_to_usdc(layer)),
    };
    let mut entries = vec![UsdzEntry {
        name: name.into(),
        data,
    }];
    entries.extend_from_slice(extras);
    usdz::write(&entries)
}

/// One geometry as a `.usdz`.
///
/// The layer inside is `.usda`. That is a deliberate choice rather than a
/// shortcut: the archive format permits either, every reader accepts text, and
/// it keeps what this crate writes inspectable with `unzip -p`.
pub fn geometry_to_usdz(g: &crate::core::BufferGeometry, name: &str) -> Vec<u8> {
    usdz::write(&[UsdzEntry {
        // The root layer must be the first entry, and the convention is to name
        // it after the archive.
        name: format!("{}.usda", export::sanitize(name)),
        data: export::geometry_to_usda(g, name).into_bytes(),
    }])
}

/// A scene as a `.usdz`, with any extra files to travel beside the layer.
pub fn scene_to_usdz(
    arena: &crate::core::ObjectArena,
    roots: &[crate::core::ObjectId],
    extras: &[UsdzEntry],
) -> Vec<u8> {
    animated_scene_to_usdz(arena, roots, &[], extras)
}

/// A scene and its animation as a `.usdz`.
///
/// This is the format Quick Look opens, and an animated one plays on iOS
/// without anything else in the archive.
pub fn animated_scene_to_usdz(
    arena: &crate::core::ObjectArena,
    roots: &[crate::core::ObjectId],
    clips: &[crate::animation::AnimationClip],
    extras: &[UsdzEntry],
) -> Vec<u8> {
    let (layer, textures) = export::animated_scene_to_layer_with_textures(
        arena,
        roots,
        clips,
        &export::UsdExportOptions::default(),
    );
    let mut entries = vec![UsdzEntry {
        name: "scene.usda".into(),
        data: write::layer_to_usda(&layer).into_bytes(),
    }];
    // The layer refers to its textures by relative path, and a `.usdz` is
    // required to be self-contained, so they travel inside it. A package whose
    // maps resolve to nothing is a valid archive and a broken asset.
    entries.extend(textures.into_iter().map(|t| UsdzEntry {
        name: t.path,
        data: t.data,
    }));
    entries.extend_from_slice(extras);
    usdz::write(&entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape most real assets arrive in: a `.usdz` whose root layer is a
    /// binary crate, not text. This fixture was packaged by OpenUSD's own
    /// `usdzip`.
    #[test]
    fn a_usdz_holding_a_crate_layer_reads_through() {
        let bytes = include_bytes!("testdata/crate_inside.usdz");
        assert_eq!(sniff(bytes), Some(UsdFormat::Usdz));

        let archive = UsdLoader::parse_archive(bytes).unwrap();
        let root = archive.root_layer().unwrap();
        assert!(root.data.starts_with(b"PXR-USDC"), "the layer inside is binary");

        let scene = UsdLoader::parse(bytes).expect("archive parses");
        let names: Vec<&str> = scene
            .roots
            .iter()
            .filter_map(|r| scene.arena.get(*r))
            .map(|o| o.name.as_str())
            .collect();
        assert_eq!(names, vec!["World", "Mat"]);
        assert_eq!(scene.up_axis, 'Z');
    }

    /// Every writer has a binary counterpart, and both come back through the
    /// same reader as the same scene.
    #[test]
    fn the_text_and_binary_writers_agree() {
        use crate::core::{Mesh, Object3D, ObjectArena};
        use crate::geometries::BoxGeometry;
        use crate::materials::{Material, StandardMaterial};

        let geometry: crate::core::BufferGeometry = BoxGeometry::new(1.0, 2.0, 3.0);
        let mut arena = ObjectArena::new();
        let mut object = Object3D::mesh(Mesh::new(
            geometry.clone(),
            Material::Standard(StandardMaterial::default()),
        ));
        object.name = "Box".into();
        let id = arena.insert(object);

        for (text, binary) in [
            (
                geometry_to_usda(&geometry, "Widget").into_bytes(),
                geometry_to_usdc(&geometry, "Widget"),
            ),
            (
                scene_to_usda(&arena, &[id]).into_bytes(),
                scene_to_usdc(&arena, &[id]),
            ),
            (
                animated_scene_to_usda(&arena, &[id], &[]).into_bytes(),
                animated_scene_to_usdc(&arena, &[id], &[]),
            ),
        ] {
            assert_eq!(sniff(&text), Some(UsdFormat::Usda));
            assert_eq!(sniff(&binary), Some(UsdFormat::Usdc));
            let from_text = UsdLoader::parse(&text).expect("text parses");
            let from_binary = UsdLoader::parse(&binary).expect("binary parses");
            assert_eq!(from_text.roots.len(), from_binary.roots.len());
            assert_eq!(
                names_of(&from_text),
                names_of(&from_binary),
                "the two forms disagree about the scene"
            );
        }
    }

    fn names_of(scene: &UsdScene) -> Vec<String> {
        fn walk(scene: &UsdScene, id: crate::core::ObjectId, out: &mut Vec<String>) {
            let Some(node) = scene.arena.get(id) else { return };
            out.push(node.name.clone());
            for child in &node.children {
                walk(scene, *child, out);
            }
        }
        let mut out = Vec::new();
        for root in &scene.roots {
            walk(scene, *root, &mut out);
        }
        out
    }

    /// A `.usdz` can hold either form of layer, and both read back.
    #[test]
    fn an_archive_can_hold_either_form_of_layer() {
        let layer = parse::parse(include_str!("testdata/rich.usda")).unwrap();

        for (form, magic) in [
            (UsdzLayer::Text, &b"#usda"[..]),
            (UsdzLayer::Crate, &b"PXR-"[..]),
        ] {
            let archive = layer_to_usdz(&layer, form, &[]);
            assert_eq!(sniff(&archive), Some(UsdFormat::Usdz));

            let inner = UsdLoader::parse_archive(&archive).unwrap();
            let root = inner.root_layer().unwrap();
            assert!(root.data.starts_with(magic), "wrong layer form for {form:?}");

            let scene = UsdLoader::parse(&archive).expect("archive parses");
            assert_eq!(names_of(&scene), vec!["World", "Quad", "Sub", "Deep", "Mat", "Shader"]);
        }
    }

    /// The binary form is the smaller one, which is the reason it exists.
    #[test]
    fn the_crate_form_is_smaller_than_the_text() {
        let layer = parse::parse(include_str!("testdata/roundtrip.usda")).unwrap();
        let text = layer_to_usda(&layer).len();
        let binary = layer_to_usdc(&layer).len();
        assert!(binary < text, "{binary} bytes binary vs {text} text");
    }

    /// A `.usdz` whose layers reference each other composes against the
    /// archive, not against the filesystem — there is no filesystem inside a
    /// zip, and the paths are the names the entries carry.
    #[test]
    fn an_archive_composes_against_itself() {
        let archive = usdz::write(&[
            UsdzEntry {
                name: "shot.usda".into(),
                data: include_str!("testdata/comp/shot.usda").into(),
            },
            UsdzEntry {
                name: "base.usda".into(),
                data: include_str!("testdata/comp/base.usda").into(),
            },
            UsdzEntry {
                name: "asset.usda".into(),
                data: include_str!("testdata/comp/asset.usda").into(),
            },
        ]);

        // Read as a single layer, only the shot's own overrides are visible:
        // `Chair1` is an `over` whose definition lives in a layer the reader
        // never followed, so the geometry it references is simply not there.
        let flat = UsdLoader::parse(&archive).expect("the root layer parses");
        assert_eq!(names_of(&flat), vec!["Room", "Chair1"], "uncomposed");

        // Composed, the referenced asset arrives with it.
        let scene = UsdLoader::open_archive(&archive, &ComposeOptions::default())
            .expect("the archive composes");
        assert_eq!(names_of(&scene), vec!["Room", "Chair1", "Seat"], "composed");
    }

    /// The composed layer is the one with the answers in it.
    #[test]
    fn opening_composes_where_parsing_does_not() {
        let mut resolver = MemoryResolver::new();
        resolver
            .insert("asset.usda", include_str!("testdata/comp/asset.usda"))
            .insert("base.usda", include_str!("testdata/comp/base.usda"));
        let shot = include_str!("testdata/comp/shot.usda").as_bytes();

        let composed =
            UsdLoader::open_layer(shot, "shot.usda", &resolver, &ComposeOptions::default())
                .unwrap();
        let chair = composed.prim_at("/Room/Chair1").expect("composed");
        assert_eq!(chair.type_name, "Xform");
        assert_eq!(
            chair.value("xformOp:translate").unwrap().flat_f32(),
            vec![9.0, 1.0, 0.0]
        );

        // The same bytes read as one layer know none of that.
        let plain = UsdLoader::parse_layer(shot).unwrap();
        assert!(plain.prim_at("/Room/Chair1").unwrap().type_name.is_empty());
    }

    #[test]
    fn formats_are_told_apart_by_content() {
        assert_eq!(sniff(b"#usda 1.0\n"), Some(UsdFormat::Usda));
        assert_eq!(sniff(b"PXR-USDC\0\0\0\0"), Some(UsdFormat::Usdc));
        assert_eq!(sniff(b"PK\x03\x04rest"), Some(UsdFormat::Usdz));
        // No header, but it is plainly a layer.
        assert_eq!(sniff(b"def Xform \"a\" {}\n"), Some(UsdFormat::Usda));
        assert_eq!(sniff(b"\x89PNG\r\n"), None);
    }

    #[test]
    fn a_geometry_round_trips_through_a_usdz() {
        use crate::core::{BufferAttribute, BufferGeometry};
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 3),
        );
        g.set_index(vec![0, 1, 2]);

        let archive = geometry_to_usdz(&g, "Tri");
        assert_eq!(sniff(&archive), Some(UsdFormat::Usdz));

        // The loader unwraps the archive and reads the layer inside it.
        let scene = UsdLoader::parse(&archive).unwrap();
        let root = scene.arena.get(scene.roots[0]).unwrap();
        let mesh = scene.arena.get(root.children[0]).unwrap();
        assert_eq!(mesh.name, "Tri");

        // And the layer is plain text anyone can look at.
        let inner = UsdLoader::parse_archive(&archive).unwrap();
        let layer = inner.root_layer().unwrap();
        assert_eq!(layer.name, "Tri.usda");
        assert!(layer.data.starts_with(b"#usda 1.0"));
    }

    #[test]
    fn a_crate_file_loads_through_the_same_entry_point_as_the_text_form() {
        let bytes = include_bytes!("testdata/triangle.usdc");
        assert_eq!(sniff(bytes), Some(UsdFormat::Usdc));
        let layer = UsdLoader::parse_layer(bytes).expect("crate parses");
        assert_eq!(layer.prim_at("/Root/Tri").unwrap().type_name, "Mesh");
    }

    #[test]
    fn a_crate_from_a_future_major_version_is_declined_rather_than_guessed_at() {
        let mut bytes = vec![0u8; 88];
        bytes[..8].copy_from_slice(b"PXR-USDC");
        bytes[8] = 1; // major 1, which does not exist yet
        bytes[16..24].copy_from_slice(&88u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        match UsdLoader::parse_layer(&bytes) {
            Err(UsdError::UnsupportedCrate(why)) => assert!(why.contains("1.0.0"), "{why}"),
            other => panic!("expected UnsupportedCrate, got {other:?}"),
        }
    }

    #[test]
    fn errors_read_as_sentences() {
        assert_eq!(
            UsdError::Syntax { line: 4, what: "unterminated prim" }.to_string(),
            "usda syntax error at line 4: unterminated prim"
        );
    }
}
