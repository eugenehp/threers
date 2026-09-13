//! Composition — turning a stack of layers into one scene.
//!
//! A `.usda` file is not a scene; it is one *opinion* about a scene. What a
//! prim actually is comes from combining opinions across sublayers, references,
//! payloads, inherits, variants and specializes. That combining is what USD
//! calls composition, and it is the part that makes USD a scene description
//! rather than a mesh container.
//!
//! # Strength ordering
//!
//! Opinions are ranked, strongest first, by USD's **LIVRPS** order:
//!
//! | | | |
//! |---|---|---|
//! | **L** | local | what this layer stack says directly |
//! | **I** | inherits | a class this prim inherits from |
//! | **V** | variants | the selected variant's body |
//! | **R** | references | another prim, here or in another layer |
//! | **P** | payloads | a reference that may be left unloaded |
//! | **S** | specializes | a base whose opinions are always weakest |
//!
//! The first opinion found for a field wins; children are the union across all
//! of them. Specializes sits last for a reason worth knowing: a specialized
//! prim can be *refined* by anything that references it, which is the opposite
//! of inherits, where the class overrides its instances.
//!
//! # What this does not do
//!
//! There is no lazy stage, no prim index cached between edits and no
//! instancing — this composes eagerly into a flat layer. That is the same
//! thing `usdcat --flatten` produces, which is how the tests check it. A
//! population mask is applied afterwards rather than avoided up front, so it
//! narrows what you get without saving the work of composing it.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use super::clips;
use super::expr;
use super::parse::{Specifier, UsdLayer, UsdPrim, UsdProperty};
use super::value::{UsdReference, UsdValue};
use super::UsdError;

/// Where layers come from.
///
/// Composition reaches other files by name, and what a name means is the
/// caller's business: a directory on disk, the entries of a `.usdz`, a bundle
/// of bytes held in memory, or a server. Nothing here opens a file itself.
pub trait AssetResolver {
    /// Read the layer named by `path`, relative to `anchor` — the layer doing
    /// the asking, so a relative path resolves the way USD resolves it.
    fn resolve(&self, anchor: &str, path: &str) -> Option<Vec<u8>>;
}

/// A resolver over the local filesystem, anchored at each layer's own
/// directory.
#[cfg(not(target_arch = "wasm32"))]
pub struct FileResolver;

#[cfg(not(target_arch = "wasm32"))]
impl AssetResolver for FileResolver {
    fn resolve(&self, anchor: &str, path: &str) -> Option<Vec<u8>> {
        std::fs::read(resolve_path(anchor, path)).ok()
    }
}

/// A resolver over layers already in memory, keyed by name.
///
/// This is what a `.usdz` is: an archive whose entries reference each other by
/// the names they have inside it.
#[derive(Default)]
pub struct MemoryResolver {
    layers: HashMap<String, Vec<u8>>,
}

impl MemoryResolver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: impl Into<String>, data: impl Into<Vec<u8>>) -> &mut Self {
        self.layers.insert(name.into(), data.into());
        self
    }
}

impl AssetResolver for MemoryResolver {
    fn resolve(&self, anchor: &str, path: &str) -> Option<Vec<u8>> {
        // An exact name first, then the same name resolved against the layer
        // that asked — an archive holds both spellings in practice.
        self.layers
            .get(path)
            .or_else(|| self.layers.get(&resolve_path(anchor, path)))
            .cloned()
    }
}

/// An archive resolves the layers inside itself.
///
/// This is what a `.usdz` is for: not one file but a set of them that reference
/// each other by the names they carry inside the zip.
impl AssetResolver for super::UsdzArchive {
    fn resolve(&self, anchor: &str, path: &str) -> Option<Vec<u8>> {
        let resolved = resolve_path(anchor, path);
        self.entries
            .iter()
            .find(|e| e.name == path || e.name == resolved)
            .map(|e| e.data.clone())
    }
}

/// Resolve `path` against the directory holding `anchor`.
///
/// Absolute paths and paths with no anchor are returned as they are.
pub fn resolve_path(anchor: &str, path: &str) -> String {
    if path.starts_with('/') || anchor.is_empty() {
        return path.to_string();
    }
    let directory = match anchor.rfind('/') {
        Some(at) => &anchor[..at + 1],
        None => return path.to_string(),
    };
    // Fold away `./` and `../` so two spellings of one layer are one layer.
    let joined_raw = format!("{directory}{path}");
    let mut parts: Vec<&str> = Vec::new();
    for part in joined_raw.split('/') {
        match part {
            "." | "" => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    let joined = parts.join("/");
    if directory.starts_with('/') {
        format!("/{joined}")
    } else {
        joined
    }
}

/// What to compose, and how much of it.
pub struct ComposeOptions {
    /// Whether payloads are loaded. USD's point in distinguishing them from
    /// references is that a payload can be left out, so a scene too big to
    /// open can still be opened.
    pub load_payloads: bool,
    /// Variant choices to apply where the layers themselves select none, as
    /// `(set name, choice)`. An authored selection always wins.
    pub variant_fallbacks: Vec<(String, String)>,
    /// How deep composition may recurse before giving up. Cycles are detected
    /// exactly; this is the backstop for a chain that is merely absurd.
    pub max_depth: usize,
}

impl Default for ComposeOptions {
    fn default() -> Self {
        Self {
            load_payloads: true,
            variant_fallbacks: Vec::new(),
            max_depth: 64,
        }
    }
}

/// Compose a layer and everything it reaches into a single flat layer.
///
/// The result is what USD calls a flattened stage: every arc resolved, every
/// opinion merged, one prim per path.
pub fn compose(
    root: &UsdLayer,
    root_name: &str,
    resolver: &dyn AssetResolver,
    options: &ComposeOptions,
) -> Result<UsdLayer, UsdError> {
    let mut engine = Engine {
        resolver,
        options,
        cache: HashMap::new(),
        stacks: HashMap::new(),
        variables: expr::Variables::new(),
    };

    // The layer stack: the root plus its sublayers, strongest first.
    let stack = engine.stack_for(root, root_name);

    // Expression variables come from the whole stack before anything is
    // resolved, since an asset path in a weak layer may name a variable the
    // strongest one sets.
    for layer in stack.iter().rev() {
        engine
            .variables
            .extend(expr::variables_of(&layer.layer.metadata));
    }

    let mut out = UsdLayer::default();
    // Layer metadata takes the strongest opinion, as everything else does.
    for layer in stack.iter() {
        for (key, value) in &layer.layer.metadata {
            // The sublayers themselves are consumed here, and so is the
            // offset list that goes with them.
            if matches!(bare_key(key), "subLayers" | "subLayerOffsets") {
                continue;
            }
            if !out.metadata.iter().any(|(k, _)| k == key) {
                out.metadata.push((key.clone(), value.clone()));
            }
        }
    }

    // Relocations, which rename prims the arcs brought in. They are stated as
    // whole paths and applied to the composed result, so they are gathered
    // before anything is built.
    let relocations = relocations_of(&stack);

    // Every prim name across the stack, in the order the strongest layer that
    // mentions it puts them.
    let names = ordered_names(stack.iter().map(|l| &l.layer.prims));
    for name in names {
        let sources: Vec<Source> = stack
            .iter()
            .filter_map(|l| {
                l.layer.prims.iter().find(|p| p.name == name).map(|prim| {
                    let mut prim = prim.clone();
                    if !l.time.is_identity() {
                        shift_times(&mut prim, l.time);
                    }
                    Source {
                        prim,
                        anchor: l.name.clone(),
                        stack: stack.clone(),
                        time: l.time,
                    }
                })
            })
            .collect();
        if let Some(prim) = engine.compose_prim(&name, &sources, &mut HashSet::new(), 0) {
            out.prims.push(prim);
        }
    }

    for (from, to) in &relocations {
        relocate(&mut out, from, to);
    }
    Ok(out)
}

/// The relocations a layer stack authors, strongest layer first.
///
/// A relocation renames a prim that an arc brought in — the asset calls it
/// `OldName`, this stage calls it `NewName` — and everything that refers to it
/// by the new name then finds it. USD keeps the statement in the flattened
/// result as well as acting on it, since it still describes what happened.
fn relocations_of(stack: &[StackLayer]) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for layer in stack {
        let Some(UsdValue::Dict(entries)) = layer.layer.meta("relocates") else {
            continue;
        };
        for (from, to) in entries {
            let Some(to) = to.as_str() else { continue };
            // A stronger layer's relocation of the same path wins.
            if !out.iter().any(|(have, _)| have == from) {
                out.push((from.clone(), to.to_string()));
            }
        }
    }
    out
}

/// Move a prim from one path to another within a composed layer.
///
/// Only a rename among siblings is handled here — `</A/Old>` to `</A/New>` —
/// which is what a relocation is for. Moving a prim to a different parent is
/// syntactically expressible and means restructuring the tree; refusing it is
/// better than doing half of it.
fn relocate(layer: &mut UsdLayer, from: &str, to: &str) {
    let (from_parent, from_name) = match from.rsplit_once('/') {
        Some(split) => split,
        None => return,
    };
    let (to_parent, to_name) = match to.rsplit_once('/') {
        Some(split) => split,
        None => return,
    };
    if from_parent != to_parent {
        return;
    }

    let siblings = if from_parent.is_empty() {
        &mut layer.prims
    } else {
        match prim_at_mut(layer, from_parent) {
            Some(parent) => &mut parent.children,
            None => return,
        }
    };
    if let Some(prim) = siblings.iter_mut().find(|p| p.name == from_name) {
        prim.name = to_name.to_string();
    }
}

fn prim_at_mut<'a>(layer: &'a mut UsdLayer, path: &str) -> Option<&'a mut UsdPrim> {
    let mut parts = path.trim_start_matches('/').split('/').filter(|p| !p.is_empty());
    let first = parts.next()?;
    let mut current = layer.prims.iter_mut().find(|p| p.name == first)?;
    for part in parts {
        current = current.children.iter_mut().find(|p| p.name == part)?;
    }
    Some(current)
}

/// One layer of a layer stack, with the name it was reached by.
struct StackLayer {
    layer: UsdLayer,
    name: String,
    /// How this layer's time codes map onto the root's. A sublayer may shift
    /// and stretch everything beneath it — the same map an arc applies, one
    /// level up — and the shifts compose down a chain of sublayers.
    time: LayerOffset,
}

/// One opinion about a prim, and where it came from.
///
/// The origin travels with the opinion rather than being taken from whoever is
/// asking, because an opinion pulled in through a reference resolves its *own*
/// arcs against its *own* layer. An `inherits = </Class>` inside a referenced
/// asset means a class in that asset, not one in the shot that referenced it.
#[derive(Clone)]
struct Source {
    prim: UsdPrim,
    anchor: String,
    stack: Rc<Vec<StackLayer>>,
    /// How this opinion's time codes map onto the root's, as
    /// `(offset, scale)`. An arc may shift and stretch the animation it pulls
    /// in — that is what lets one walk cycle start at frame 100 here and frame
    /// 340 there — and the shifts compose through nested arcs.
    time: LayerOffset,
}

/// `t * scale + offset`, the map an arc applies to the time codes beneath it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct LayerOffset {
    offset: f64,
    scale: f64,
}

impl Default for LayerOffset {
    fn default() -> Self {
        Self {
            offset: 0.0,
            scale: 1.0,
        }
    }
}

impl LayerOffset {
    fn is_identity(&self) -> bool {
        self.offset == 0.0 && self.scale == 1.0
    }

    fn apply(&self, time: f64) -> f64 {
        time * self.scale + self.offset
    }

    /// This offset applied *after* `inner` — the map a nested arc ends up
    /// with, since the inner shift happens in the outer layer's frame.
    fn then(&self, inner: LayerOffset) -> LayerOffset {
        LayerOffset {
            offset: inner.offset * self.scale + self.offset,
            scale: inner.scale * self.scale,
        }
    }
}

/// Move every time sample in a subtree onto a new time line.
fn shift_times(prim: &mut UsdPrim, by: LayerOffset) {
    for property in &mut prim.properties {
        if let UsdValue::TimeSamples(samples) = &mut property.value {
            for (time, _) in samples.iter_mut() {
                *time = by.apply(*time);
            }
        }
    }
    for set in &mut prim.variant_sets {
        for (_, body) in &mut set.variants {
            shift_times(body, by);
        }
    }
    for child in &mut prim.children {
        shift_times(child, by);
    }
}

struct Engine<'a> {
    resolver: &'a dyn AssetResolver,
    options: &'a ComposeOptions,
    /// Layers already opened, so a file referenced fifty times is parsed once.
    cache: HashMap<String, UsdLayer>,
    /// Layer stacks already built, keyed by the layer that roots them.
    stacks: HashMap<String, Rc<Vec<StackLayer>>>,
    /// The variables an expression may refer to. A stronger layer's value for
    /// a name wins, which is how one shot layer drives what a whole stage
    /// pulls in.
    variables: expr::Variables,
}

impl Engine<'_> {
    /// Open a layer by name, through the resolver, once.
    fn layer(&mut self, anchor: &str, path: &str) -> Option<UsdLayer> {
        // An asset path may be an expression. One that does not evaluate is
        // not opened at all: naming the wrong file would be worse than naming
        // none.
        let path = &expr::evaluate(path, &self.variables)?;
        let key = resolve_path(anchor, path);
        if let Some(found) = self.cache.get(&key) {
            return Some(found.clone());
        }
        let bytes = self.resolver.resolve(anchor, path)?;
        let layer = super::UsdLoader::parse_layer(&bytes).ok()?;
        self.cache.insert(key, layer.clone());
        Some(layer)
    }

    /// A layer and its sublayers, strongest first.
    ///
    /// `subLayers` is authored strongest-first, which is the opposite of how
    /// most people read a list, and getting it backwards silently inverts
    /// every override in a shot.
    fn layer_stack(
        &mut self,
        layer: &UsdLayer,
        name: &str,
        seen: &mut HashSet<String>,
        depth: usize,
    ) -> Vec<StackLayer> {
        if depth > self.options.max_depth || !seen.insert(resolve_path("", name)) {
            return Vec::new();
        }
        let mut out = vec![StackLayer {
            layer: layer.clone(),
            name: name.to_string(),
            time: LayerOffset::default(),
        }];
        for (key, value) in &layer.metadata {
            if bare_key(key) != "subLayers" {
                continue;
            }
            for (sub, offset) in sublayers_of(value) {
                let Some(opened) = self.layer(name, &sub) else {
                    continue;
                };
                let resolved = resolve_path(name, &sub);
                // A sublayer of a shifted sublayer is shifted by both.
                for mut inner in self.layer_stack(&opened, &resolved, seen, depth + 1) {
                    inner.time = offset.then(inner.time);
                    out.push(inner);
                }
            }
        }
        out
    }

    /// The layer stack rooted at a named layer, built once.
    fn stack_for(&mut self, layer: &UsdLayer, name: &str) -> Rc<Vec<StackLayer>> {
        let key = resolve_path("", name);
        if let Some(found) = self.stacks.get(&key) {
            return found.clone();
        }
        let built = Rc::new(self.layer_stack(layer, name, &mut HashSet::new(), 0));
        self.stacks.insert(key, built.clone());
        built
    }

    /// Merge every opinion about one prim, in LIVRPS order.
    fn compose_prim(
        &mut self,
        name: &str,
        sources: &[Source],
        visiting: &mut HashSet<String>,
        depth: usize,
    ) -> Option<UsdPrim> {
        if sources.is_empty() || depth > self.options.max_depth {
            return None;
        }

        // Variant selections resolve across the prim's *local* opinions before
        // any arc is followed, because the layer that selects a variant is
        // very often not the layer that authored the variant set.
        let mut selections: Vec<(String, String)> = Vec::new();
        for source in sources {
                for (set, choice) in selections_of(&source.prim) {
                // A selection may be an expression too, which is how a shot
                // picks a look for everything at once.
                let Some(choice) = expr::evaluate(&choice, &self.variables) else {
                    continue;
                };
                if !selections.iter().any(|(s, _)| *s == set) {
                    selections.push((set, choice));
                }
            }
        }
        for (set, choice) in &self.options.variant_fallbacks {
            if !selections.iter().any(|(s, _)| s == set) {
                selections.push((set.clone(), choice.clone()));
            }
        }

        // The full opinion list: each source, then whatever its arcs bring in,
        // weaker than the source that named them.
        //
        // The arcs are combined across the whole stack before any is followed,
        // because a list operation composes across layers: the shot that
        // deletes a reference and the asset that prepended it are different
        // opinions, and resolving each on its own means the delete never meets
        // what it was deleting.
        let mut opinions: Vec<Source> = sources.to_vec();
        self.expand_arcs(sources, &selections, &mut opinions, visiting, depth);

        // A prim exists only if something defines it: a stack of `over`s with
        // no `def` anywhere is an override of nothing.
        if !opinions.iter().any(|o| o.prim.specifier == Specifier::Def) {
            return None;
        }
        // `active = false` takes the prim off the stage, and everything under
        // it with it. It is how a shot switches a prop off without editing the
        // asset, so the strongest opinion is the one that decides.
        if let Some(UsdValue::Bool(false)) = opinions.iter().find_map(|o| o.prim.meta("active")) {
            return None;
        }
        let specifier = opinions
            .iter()
            .map(|o| o.prim.specifier)
            .find(|s| *s != Specifier::Over)
            .unwrap_or(Specifier::Def);

        let mut prim = UsdPrim {
            specifier,
            type_name: opinions
                .iter()
                .map(|o| o.prim.type_name.as_str())
                .find(|t| !t.is_empty())
                .unwrap_or_default()
                .to_string(),
            name: name.to_string(),
            metadata: Vec::new(),
            properties: Vec::new(),
            children: Vec::new(),
            variant_sets: Vec::new(),
        };

        // Metadata and properties: strongest opinion wins, field by field.
        for opinion in &opinions {
            for (key, value) in &opinion.prim.metadata {
                // The arcs are what composition *is* and do not survive into
                // the result. A `reorder` does survive — USD keeps it in a
                // flattened layer as well as acting on it, since it still
                // describes the order.
                if is_arc_field(key) {
                    continue;
                }
                if !prim.metadata.iter().any(|(k, _)| k == key) {
                    prim.metadata.push((key.clone(), value.clone()));
                }
            }
            for property in &opinion.prim.properties {
                match prim.properties.iter_mut().find(|p| p.name == property.name) {
                    // A weaker opinion still fills in what the stronger one
                    // left unsaid — a type, an interpolation, a value.
                    Some(existing) => fill_in(existing, property),
                    None => prim.properties.push(property.clone()),
                }
            }
        }

        if let Some(order) = opinions
            .iter()
            .find_map(|o| o.prim.meta_exact("reorder properties"))
            .map(UsdValue::flat_tokens)
        {
            prim.properties.sort_by_key(|property| {
                order
                    .iter()
                    .position(|o| *o == property.name.as_str())
                    .unwrap_or(usize::MAX)
            });
        }

        // Value clips, which supply the animation for attributes the prim
        // already has. They are resolved after the opinions are merged,
        // because what they apply to is the merged result.
        self.apply_clips(&mut prim, sources);

        // Children: the union, each composed from every opinion that has one,
        // and each carrying the origin of the opinion it was found in.
        let mut names = ordered_names(opinions.iter().map(|o| &o.prim.children));
        // `reorder nameChildren` says what order they compose in — the
        // strongest opinion that states one wins, and anything it does not
        // name keeps its place behind those it does.
        if let Some(order) = opinions
            .iter()
            .find_map(|o| o.prim.meta_exact("reorder nameChildren"))
            .map(UsdValue::flat_tokens)
        {
            names.sort_by_key(|name| {
                order
                    .iter()
                    .position(|o| *o == name.as_str())
                    .unwrap_or(usize::MAX)
            });
        }
        for child_name in names {
            let child_sources: Vec<Source> = opinions
                .iter()
                .filter_map(|o| {
                    o.prim
                        .children
                        .iter()
                        .find(|c| c.name == child_name)
                        .map(|prim| Source {
                            prim: prim.clone(),
                            anchor: o.anchor.clone(),
                            stack: o.stack.clone(),
                            time: o.time,
                        })
                })
                .collect();
            if let Some(child) = self.compose_prim(&child_name, &child_sources, visiting, depth + 1)
            {
                prim.children.push(child);
            }
        }
        Some(prim)
    }

    /// Replace attribute values with what the clips hold, where a prim has
    /// them.
    ///
    /// Clips do not *create* attributes: a prim with nothing declared stays
    /// that way however many clips name it, which is why a topology layer
    /// usually sits underneath one.
    fn apply_clips(&mut self, prim: &mut UsdPrim, sources: &[Source]) {
        let sets = clips::sets_of(prim);
        if sets.is_empty() {
            return;
        }
        // Relative asset paths resolve against the layer that declared them.
        let anchor = sources
            .first()
            .map(|s| s.anchor.clone())
            .unwrap_or_default();

        for set in sets {
            if set.assets.is_empty() || set.prim_path.is_empty() {
                continue;
            }
            let manifest = (!set.manifest.is_empty())
                .then(|| self.layer(&anchor, &set.manifest))
                .flatten();
            let names = clips::animated_names(manifest.as_ref(), &set.prim_path, prim);

            // Opening a clip is not free, so each is opened once for the whole
            // set rather than once per attribute per frame.
            let layers: Vec<Option<UsdLayer>> = set
                .assets
                .iter()
                .map(|asset| self.layer(&anchor, asset))
                .collect();

            for name in names {
                // The times this attribute is sampled at, which without a
                // `times` table come from the clips themselves.
                let schedule = clips::sample_times(&set, |index| {
                    layers
                        .get(index)
                        .and_then(|l| l.as_ref())
                        .and_then(|l| l.prim_at(&set.prim_path))
                        .and_then(|p| p.value(&name))
                        .and_then(|v| v.samples().map(|s| s.iter().map(|(t, _)| *t).collect()))
                        .unwrap_or_default()
                });
                if schedule.is_empty() {
                    continue;
                }

                let mut samples = Vec::with_capacity(schedule.len());
                for (stage_time, clip_time) in schedule {
                    let Some(index) = set.clip_at(stage_time) else {
                        continue;
                    };
                    let Some(value) = layers
                        .get(index)
                        .and_then(|l| l.as_ref())
                        .and_then(|l| l.prim_at(&set.prim_path))
                        .and_then(|p| p.value(&name))
                    else {
                        continue;
                    };
                    samples.push((stage_time, value.at_time(clip_time)));
                }
                if samples.is_empty() {
                    continue;
                }
                // Only onto an attribute that exists.
                if let Some(property) = prim.properties.iter_mut().find(|p| p.name == name) {
                    property.value = UsdValue::TimeSamples(samples);
                }
            }
        }
    }

    /// Everything a stack of opinions' arcs contribute, appended weakest-last.
    fn expand_arcs(
        &mut self,
        sources: &[Source],
        selections: &[(String, String)],
        out: &mut Vec<Source>,
        visiting: &mut HashSet<String>,
        depth: usize,
    ) {
        if depth > self.options.max_depth {
            return;
        }

        // I — inherits.
        for (arc, origin) in combine_arcs(sources, "inherits") {
            self.follow(&sources[origin], &arc, selections, out, visiting, depth);
        }

        // V — variants, which are per-prim bodies rather than a shared list.
        for source in sources {
        for set in source.prim.variant_sets.clone() {
            let chosen = selections
                .iter()
                .find(|(s, _)| *s == set.name)
                .map(|(_, c)| c.clone())
                // With nothing selected anywhere, USD falls back to the first
                // variant authored.
                .or_else(|| set.variants.first().map(|(n, _)| n.clone()));
            let Some(body) = chosen.as_deref().and_then(|c| set.get(c)) else {
                continue;
            };
            let inner = Source {
                prim: body.clone(),
                anchor: source.anchor.clone(),
                stack: source.stack.clone(),
                time: source.time,
            };
            out.push(inner.clone());
            // A variant may itself carry arcs, and its own variant sets.
            self.expand_arcs(
                std::slice::from_ref(&inner),
                selections,
                out,
                visiting,
                depth + 1,
            );
        }
        }

        // R — references, then P — payloads, then S — specializes.
        for field in ["references", "payload", "specializes"] {
            if field == "payload" && !self.options.load_payloads {
                continue;
            }
            for (arc, origin) in combine_arcs(sources, field) {
                self.follow(&sources[origin], &arc, selections, out, visiting, depth);
            }
        }
    }

    /// Pull in what one arc points at.
    fn follow(
        &mut self,
        source: &Source,
        arc: &UsdReference,
        selections: &[(String, String)],
        out: &mut Vec<Source>,
        visiting: &mut HashSet<String>,
        depth: usize,
    ) {
        // A layer that references itself, directly or around a loop, would
        // otherwise recurse until the stack gives out.
        let key = format!(
            "{}|{}",
            resolve_path(&source.anchor, &arc.asset),
            arc.prim_path
        );
        if !visiting.insert(key.clone()) {
            return;
        }

        let (stack, anchor) = if arc.is_internal() {
            // An internal arc points inside the layer stack this opinion came
            // from — which is not necessarily the root's.
            (source.stack.clone(), source.anchor.clone())
        } else {
            let Some(opened) = self.layer(&source.anchor, &arc.asset) else {
                visiting.remove(&key);
                return;
            };
            let anchor = resolve_path(&source.anchor, &arc.asset);
            // The referenced layer brings its own sublayers with it.
            (self.stack_for(&opened, &anchor), anchor)
        };

        // Where to look inside it: the named prim, or the layer's default.
        let target_path = if !arc.prim_path.is_empty() {
            arc.prim_path.clone()
        } else {
            match stack.first().map(|l| &l.layer).and_then(default_prim_of) {
                Some(name) => format!("/{name}"),
                None => {
                    visiting.remove(&key);
                    return;
                }
            }
        };

        // The arc's own shift, on top of whatever is already in effect.
        let time = source.time.then(LayerOffset {
            offset: arc.offset,
            scale: arc.scale,
        });

        // Every layer of that stack that has an opinion about the target.
        let found: Vec<UsdPrim> = stack
            .iter()
            .filter_map(|l| l.layer.prim_at(&target_path).cloned())
            .collect();
        for mut target in found {
            if !time.is_identity() {
                shift_times(&mut target, time);
            }
            let inner = Source {
                prim: target,
                anchor: anchor.clone(),
                stack: stack.clone(),
                time,
            };
            out.push(inner.clone());
            self.expand_arcs(
                std::slice::from_ref(&inner),
                selections,
                out,
                visiting,
                depth + 1,
            );
        }
        visiting.remove(&key);
    }
}

/// The variant choices a prim's own metadata selects.
fn selections_of(prim: &UsdPrim) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (key, value) in &prim.metadata {
        if bare_key(key) != "variants" {
            continue;
        }
        if let UsdValue::Dict(entries) = value {
            for (set, choice) in entries {
                if let Some(choice) = choice.as_str() {
                    out.push((set.clone(), choice.to_string()));
                }
            }
        }
    }
    out
}

/// A metadata key without its list-operation qualifier.
fn bare_key(key: &str) -> &str {
    key.trim_start_matches("prepend ")
        .trim_start_matches("append ")
        .trim_start_matches("delete ")
        .trim_start_matches("add ")
}

fn default_prim_of(layer: &UsdLayer) -> Option<String> {
    layer
        .meta("defaultPrim")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .or_else(|| layer.prims.first().map(|p| p.name.clone()))
}

/// Whether a metadata key is a composition arc, which is consumed here rather
/// than carried through to the flattened result.
fn is_arc_field(key: &str) -> bool {
    let bare = key
        .trim_start_matches("prepend ")
        .trim_start_matches("append ")
        .trim_start_matches("delete ")
        .trim_start_matches("add ");
    matches!(
        bare,
        "references" | "payload" | "inherits" | "specializes" | "variants" | "variantSets"
    )
}

/// The arcs authored in one of a prim's composition fields, in the order they
/// take effect.
///
/// An arc has three spellings and all three mean something slightly different:
/// `@layer.usda@</Prim>` names a prim in another layer, `@layer.usda@` alone
/// means that layer's `defaultPrim`, and `</Prim>` alone is an internal arc to
/// this layer's own stack.
/// One field's arcs, composed across a stack of opinions.
///
/// A list operation is applied *over* the result of the weaker ones: an
/// explicit list replaces everything below it, `delete` removes, `prepend` goes
/// in front and `append` behind. Each arc is returned with the index of the
/// opinion that contributed it, because a relative asset path resolves against
/// the layer that named it and not against whichever layer happened to be
/// strongest.
fn combine_arcs(sources: &[Source], field: &str) -> Vec<(UsdReference, usize)> {
    let mut out: Vec<(UsdReference, usize)> = Vec::new();
    // Weakest first, so each stronger opinion is applied on top.
    for (index, source) in sources.iter().enumerate().rev() {
        let prim = &source.prim;
        let of = |key: &str| -> Vec<UsdReference> {
            let mut found = Vec::new();
            if let Some(value) = prim.meta_exact(key) {
                collect_arcs(value, &mut found);
            }
            found
        };

        // An explicit list says what the answer *is*, whatever was below.
        let explicit = of(field);
        let unqualified_is_explicit = prim.meta_exact(field).is_some();
        if unqualified_is_explicit {
            out = explicit.into_iter().map(|arc| (arc, index)).collect();
        }

        for arc in of(&format!("delete {field}")) {
            out.retain(|(have, _)| have.asset != arc.asset || have.prim_path != arc.prim_path);
        }
        // Prepended items go in front, keeping their own order.
        let prepended = of(&format!("prepend {field}"));
        for arc in prepended.into_iter().rev() {
            out.retain(|(have, _)| have.asset != arc.asset || have.prim_path != arc.prim_path);
            out.insert(0, (arc, index));
        }
        for arc in of(&format!("append {field}"))
            .into_iter()
            .chain(of(&format!("add {field}")))
        {
            out.retain(|(have, _)| have.asset != arc.asset || have.prim_path != arc.prim_path);
            out.push((arc, index));
        }
        // `reorder` moves what is there without adding or removing.
        let order = of(&format!("reorder {field}"));
        if !order.is_empty() {
            out.sort_by_key(|(arc, _)| {
                order
                    .iter()
                    .position(|o| o.asset == arc.asset && o.prim_path == arc.prim_path)
                    .unwrap_or(usize::MAX)
            });
        }
    }
    out
}

fn collect_arcs(value: &UsdValue, out: &mut Vec<UsdReference>) {
    match value {
        UsdValue::Reference(r) => out.push(r.clone()),
        // A layer with no prim named: its `defaultPrim`.
        UsdValue::Asset(a) => out.push(UsdReference::new(a.clone(), "")),
        // A path with no layer: inside this one.
        UsdValue::Path(p) => out.push(UsdReference::new("", p.clone())),
        UsdValue::Array(items) => {
            for item in items {
                collect_arcs(item, out);
            }
        }
        _ => {}
    }
}

/// The sublayers named by a value, each with the time offset it carries.
fn sublayers_of(value: &UsdValue) -> Vec<(String, LayerOffset)> {
    match value {
        UsdValue::Reference(r) => vec![(
            r.asset.clone(),
            LayerOffset {
                offset: r.offset,
                scale: r.scale,
            },
        )],
        UsdValue::Array(items) => items.iter().flat_map(sublayers_of).collect(),
        other => asset_paths(other)
            .into_iter()
            .map(|a| (a, LayerOffset::default()))
            .collect(),
    }
}

/// The asset paths in a `subLayers`-shaped value.
fn asset_paths(value: &UsdValue) -> Vec<String> {
    match value {
        UsdValue::Asset(a) => vec![a.clone()],
        UsdValue::String(s) => vec![s.clone()],
        UsdValue::Reference(r) => vec![r.asset.clone()],
        UsdValue::Array(items) => items.iter().flat_map(asset_paths).collect(),
        _ => Vec::new(),
    }
}

/// Names across several prim lists, strongest list first, each name once.
fn ordered_names<'a>(lists: impl Iterator<Item = &'a Vec<UsdPrim>>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for list in lists {
        for prim in list {
            if !out.contains(&prim.name) {
                out.push(prim.name.clone());
            }
        }
    }
    out
}

/// Let a weaker opinion supply what a stronger one left unsaid.
fn fill_in(strong: &mut UsdProperty, weak: &UsdProperty) {
    if strong.type_name.is_empty() {
        strong.type_name = weak.type_name.clone();
    }
    // A declaration with nothing said takes what a weaker layer says. A
    // *block* does not: `foo = None` exists to stop the weaker opinion, and
    // filling it in from below would do the opposite of what it asks.
    if matches!(strong.value, UsdValue::None) && !matches!(weak.value, UsdValue::None) {
        strong.value = weak.value.clone();
    }
    strong.uniform |= weak.uniform;
    strong.relationship |= weak.relationship;
    for (key, value) in &weak.metadata {
        if !strong.metadata.iter().any(|(k, _)| k == key) {
            strong.metadata.push((key.clone(), value.clone()));
        }
    }
}

/// Narrow a composed layer to a set of prims, as USD's population mask does.
///
/// Each path keeps the prim it names, everything beneath it, and its ancestors
/// — and an ancestor keeps its own properties while losing the children that
/// lead nowhere. Masking on `/World/Keep` leaves `World` with its transform
/// intact, because the kept prim is still positioned by it, and drops
/// `World`'s other children entirely.
///
/// A path naming nothing still keeps the ancestors it does name: masking on
/// `/World/Keep/Missing` leaves `World` and `Keep` standing and drops `Keep`'s
/// real children. Which prims are ancestors is decided by the path asked for,
/// not by what is found along it.
///
/// An empty mask means the whole stage, matching `UsdStagePopulationMask::All`.
///
/// This prunes after composing rather than before, which is the same answer
/// USD gives but not the same saving: `usdcat --mask` never builds the parts
/// it discards, and this builds them and then throws them away.
pub fn mask(layer: &UsdLayer, paths: &[&str]) -> UsdLayer {
    if paths.is_empty() {
        return layer.clone();
    }
    let wanted: Vec<String> = paths
        .iter()
        .map(|p| {
            let p = p.trim().trim_end_matches('/');
            if p.starts_with('/') {
                p.to_string()
            } else {
                format!("/{p}")
            }
        })
        .filter(|p| p.len() > 1)
        .collect();
    let mut out = layer.clone();
    out.prims = layer
        .prims
        .iter()
        .filter_map(|prim| keep(prim, "", &wanted))
        .collect();
    out
}

/// One prim under the mask, or `None` if nothing at or below it is wanted.
fn keep(prim: &UsdPrim, parent: &str, wanted: &[String]) -> Option<UsdPrim> {
    let path = format!("{parent}/{}", prim.name);
    // At or below something asked for: the whole subtree comes through.
    if wanted
        .iter()
        .any(|w| path == *w || path.starts_with(&format!("{w}/")))
    {
        return Some(prim.clone());
    }
    // On the way to something asked for: this prim stays, with its own
    // properties, and keeps only the children that lead onwards.
    //
    // This is decided by the path asked for, not by what is found along it. A
    // mask naming a prim that does not exist still keeps the part of the chain
    // that does — `--mask /World/Keep/Missing` leaves `World` and `Keep`
    // standing with their properties and drops `Keep`'s actual children.
    // Pruning on "did any child survive" instead would empty the stage.
    let prefix = format!("{path}/");
    if !wanted.iter().any(|w| w.starts_with(&prefix)) {
        return None;
    }
    let mut kept = prim.clone();
    kept.children = prim
        .children
        .iter()
        .filter_map(|child| keep(child, &path, wanted))
        .collect();
    Some(kept)
}

#[cfg(test)]
mod tests {
    /// A population mask keeps what it names, what is under it, and the
    /// ancestors that position it — and nothing else.
    ///
    /// The expectations here are `usdcat --mask /World/Keep --flatten`:
    /// `World` survives *with its transform*, `Keep` and `Inner` survive
    /// whole, and `Drop` and its child are gone. Getting the ancestor rule
    /// wrong in either direction is easy — dropping `World` moves the geometry,
    /// keeping all of `World` defeats the mask.
    #[test]
    fn a_population_mask_keeps_ancestors_and_descendants() {
        let layer = super::super::parse::parse(include_str!("testdata/mask/src.usda")).unwrap();
        let masked = mask(&layer, &["/World/Keep"]);
        let paths: Vec<String> = masked
            .prims_by_path()
            .into_iter()
            .map(|(p, _)| p)
            .collect();
        assert_eq!(paths, ["/World", "/World/Keep", "/World/Keep/Inner"]);
        assert!(
            masked.prim_at("/World").unwrap().value("xformOp:translate").is_some(),
            "the ancestor keeps the transform that places what was kept"
        );
    }

    /// Several paths union, and an empty mask is the whole stage.
    #[test]
    fn a_population_mask_unions_and_defaults_to_everything() {
        let layer = super::super::parse::parse(include_str!("testdata/mask/src.usda")).unwrap();
        let both: Vec<String> = mask(&layer, &["/World/Keep", "/Other"])
            .prims_by_path()
            .into_iter()
            .map(|(p, _)| p)
            .collect();
        assert_eq!(
            both,
            [
                "/World",
                "/World/Keep",
                "/World/Keep/Inner",
                "/Other",
                "/Other/Elsewhere"
            ]
        );
        assert_eq!(
            mask(&layer, &[]).prims_by_path().len(),
            layer.prims_by_path().len()
        );
    }

    /// A path naming nothing leaves an empty stage rather than the whole one.
    #[test]
    fn a_mask_that_matches_nothing_keeps_nothing() {
        let layer = super::super::parse::parse(include_str!("testdata/mask/src.usda")).unwrap();
        assert!(mask(&layer, &["/Nowhere"]).prims.is_empty());
    }

    /// A mask through a prim that does not exist keeps the chain that does.
    ///
    /// `--mask /World/Keep/Missing` leaves `World` and `Keep` standing, with
    /// their properties, and drops `Keep`'s real child — the ancestors are
    /// decided by the path asked for, not by what was found along it. Deciding
    /// instead on "did any child survive" empties the stage, which is what
    /// this crate did until `usdcat --mask` was asked.
    #[test]
    fn a_mask_through_a_missing_prim_keeps_the_chain_that_exists() {
        let layer = super::super::parse::parse(include_str!("testdata/mask/src.usda")).unwrap();
        let masked = mask(&layer, &["/World/Keep/Missing"]);
        let paths: Vec<String> = masked.prims_by_path().into_iter().map(|(p, _)| p).collect();
        assert_eq!(paths, ["/World", "/World/Keep"]);
        assert!(masked.prim_at("/World/Keep").unwrap().value("size").is_some());

        // And the same one level deeper, where the surviving chain is longer.
        let deep = mask(&layer, &["/World/Drop/Gone/Deeper"]);
        let paths: Vec<String> = deep.prims_by_path().into_iter().map(|(p, _)| p).collect();
        assert_eq!(paths, ["/World", "/World/Drop", "/World/Drop/Gone"]);
        assert!(deep.prim_at("/World/Drop/Gone").unwrap().value("radius").is_some());
    }

    use super::*;
    use super::super::parse::parse;
    use super::super::write::layer_to_usda;

    /// The three layers of `testdata/comp`, as an in-memory resolver.
    fn pipeline() -> MemoryResolver {
        let mut r = MemoryResolver::new();
        r.insert("asset.usda", include_str!("testdata/comp/asset.usda"))
            .insert("base.usda", include_str!("testdata/comp/base.usda"))
            .insert("shot.usda", include_str!("testdata/comp/shot.usda"));
        r
    }

    fn compose_shot(options: &ComposeOptions) -> UsdLayer {
        let root = parse(include_str!("testdata/comp/shot.usda")).expect("the shot parses");
        compose(&root, "shot.usda", &pipeline(), options).expect("composes")
    }

    /// The whole of it, against what `usdcat --flatten` produced from the same
    /// three files: sublayers, a reference, an inherit, a variant selected in
    /// one layer and overridden in another, and `over` opinions merging.
    #[test]
    fn a_composed_stage_matches_what_openusd_flattens() {
        let ours = compose_shot(&ComposeOptions::default());
        let theirs = parse(include_str!("testdata/comp/flat.usda")).expect("the truth parses");

        let chair = ours
            .prim_at("/Room/Chair1")
            .expect("the referenced chair is there");
        let expected = theirs.prim_at("/Room/Chair1").unwrap();

        // The type came through the reference, from the asset.
        assert_eq!(chair.type_name, expected.type_name, "type name");
        assert_eq!(chair.type_name, "Xform");

        for property in &expected.properties {
            let mine = chair
                .value(&property.name)
                .unwrap_or_else(|| panic!("{} is missing", property.name));
            if let Some(text) = property.value.as_str() {
                assert_eq!(mine.as_str(), Some(text), "{}", property.name);
            } else {
                assert_eq!(
                    mine.flat_f32(),
                    property.value.flat_f32(),
                    "{}",
                    property.name
                );
            }
        }

        // And the child that came in with the reference.
        let seat = ours.prim_at("/Room/Chair1/Seat").expect("the seat");
        assert_eq!(seat.type_name, "Mesh");
        assert_eq!(seat.value("points").unwrap().flat_f32().len(), 9);
    }

    /// Every prim and every value, against OpenUSD's own flattening — not a
    /// sample of them. A composition engine that gets the common case right
    /// and quietly drops a prim is worse than one that fails loudly.
    #[test]
    fn nothing_is_missing_and_nothing_is_invented() {
        let ours = compose_shot(&ComposeOptions::default());
        let theirs = parse(include_str!("testdata/comp/flat.usda")).unwrap();

        let ours_paths: Vec<String> = ours.prims_by_path().into_iter().map(|(p, _)| p).collect();
        let theirs_paths: Vec<String> =
            theirs.prims_by_path().into_iter().map(|(p, _)| p).collect();
        assert_eq!(ours_paths, theirs_paths, "the set of prims differs");

        for (path, expected) in theirs.prims_by_path() {
            let mine = ours.prim_at(&path).unwrap();
            assert_eq!(mine.type_name, expected.type_name, "type of {path}");

            let mut mine_names: Vec<&str> =
                mine.properties.iter().map(|p| p.name.as_str()).collect();
            let mut expected_names: Vec<&str> =
                expected.properties.iter().map(|p| p.name.as_str()).collect();
            mine_names.sort();
            expected_names.sort();
            assert_eq!(mine_names, expected_names, "properties of {path}");

            for property in &expected.properties {
                let mine = mine.value(&property.name).unwrap();
                match property.value.as_str() {
                    Some(text) => assert_eq!(mine.as_str(), Some(text), "{path}.{}", property.name),
                    None => assert_eq!(
                        mine.flat_f32(),
                        property.value.flat_f32(),
                        "{path}.{}",
                        property.name
                    ),
                }
            }
        }
    }

    /// The same three-layer pipeline, composed from `.usdc` instead of
    /// `.usda`, must give the same answer.
    ///
    /// Composition arcs are stored in a crate under names a document never
    /// uses — `inheritPaths`, `variantSelection`, `variantSetNames` — and
    /// variant bodies live at their own paths rather than inside the prim, so
    /// a reader that handles the text form tells you nothing about this one.
    #[test]
    fn a_binary_pipeline_composes_to_the_same_thing_as_a_text_one() {
        let mut binary = MemoryResolver::new();
        binary
            .insert("asset.usda", include_bytes!("testdata/comp/asset.usdc").as_slice())
            .insert("base.usda", include_bytes!("testdata/comp/base.usdc").as_slice());
        let root = super::super::UsdLoader::parse_layer(include_bytes!(
            "testdata/comp/shot.usdc"
        ))
        .expect("the binary shot parses");
        let from_binary = compose(&root, "shot.usda", &binary, &ComposeOptions::default())
            .expect("the binary pipeline composes");

        let from_text = compose_shot(&ComposeOptions::default());
        // By meaning rather than by spelling: a document remembers that `10`
        // was written without a decimal point and a crate does not.
        let paths = |l: &UsdLayer| -> Vec<String> {
            l.prims_by_path().into_iter().map(|(p, _)| p).collect()
        };
        assert_eq!(paths(&from_binary), paths(&from_text), "different prims");
        for (path, expected) in from_text.prims_by_path() {
            let mine = from_binary.prim_at(&path).unwrap();
            assert_eq!(mine.type_name, expected.type_name, "type of {path}");
            for property in &expected.properties {
                let mine = mine
                    .value(&property.name)
                    .unwrap_or_else(|| panic!("{path}.{} missing", property.name));
                match property.value.as_str() {
                    Some(text) => assert_eq!(mine.as_str(), Some(text), "{path}.{}", property.name),
                    None => assert_eq!(
                        mine.flat_f32(),
                        property.value.flat_f32(),
                        "{path}.{}",
                        property.name
                    ),
                }
            }
        }
    }

    /// Every arc spelling survives a trip through the binary form: an asset
    /// with a prim path, an asset without one, an internal path, a payload,
    /// a specializes, and a layer offset.
    #[test]
    fn every_arc_spelling_survives_the_binary_form() {
        let from_text = parse(include_str!("testdata/comp/arcs.usda")).unwrap();
        let from_binary =
            super::super::UsdLoader::parse_layer(include_bytes!("testdata/comp/arcs.usdc"))
                .expect("parses");

        let a = from_binary.prim_at("/A").expect("A");
        let refs = a.meta("references").expect("references survived");
        let refs = refs.references();
        assert_eq!(refs.len(), 3, "{refs:?}");
        assert_eq!(refs[0].asset, "asset.usda");
        assert_eq!(refs[0].prim_path, "/Chair");
        assert_eq!(refs[1].asset, "other.usda");
        assert_eq!(refs[1].prim_path, "", "no prim path means the default prim");
        assert_eq!(refs[2].asset, "", "an internal arc has no asset");
        assert_eq!(refs[2].prim_path, "/A/Local");

        let payload = a.meta("payload").expect("payload survived").references();
        assert_eq!(payload.len(), 1);
        assert_eq!(payload[0].asset, "heavy.usda");
        assert_eq!(payload[0].prim_path, "/Deep");

        assert_eq!(
            a.meta("specializes").unwrap().references()[0].prim_path,
            "/A/Base"
        );

        // The layer offset, which is what lets a clip start somewhere else.
        let b = from_binary.prim_at("/B").unwrap().meta("references").unwrap();
        let b = b.references();
        assert_eq!((b[0].offset, b[0].scale), (10.0, 2.0));

        // And the text form agrees about all of it.
        let text_refs = from_text.prim_at("/A").unwrap().meta("references").unwrap();
        assert_eq!(text_refs.references().len(), 3);
    }

    /// Variant sets and the selection of one, read out of a crate file.
    #[test]
    fn variants_survive_the_binary_form() {
        let layer = super::super::UsdLoader::parse_layer(include_bytes!(
            "testdata/comp/asset.usdc"
        ))
        .unwrap();
        let chair = layer.prim_at("/Chair").expect("the chair");

        let set = chair
            .variant_sets
            .iter()
            .find(|s| s.name == "look")
            .expect("the look variant set");
        let mut names: Vec<&str> = set.variants.iter().map(|(n, _)| n.as_str()).collect();
        names.sort();
        assert_eq!(names, vec!["oak", "steel"]);

        // Each variant carries its own body.
        let oak = set.get("oak").expect("oak");
        assert_eq!(
            oak.value("primvars:displayColor").unwrap().flat_f32(),
            vec![0.6, 0.4, 0.2]
        );

        // The asset's own selection, under the name a document uses for it.
        let UsdValue::Dict(entries) = chair.meta("variants").expect("variants") else {
            panic!("expected a dictionary");
        };
        assert_eq!(entries[0].0, "look");
        assert_eq!(entries[0].1.as_str(), Some("oak"));

        // And the inherit, which a crate calls `inheritPaths`.
        assert_eq!(
            chair.meta("inherits").unwrap().references()[0].prim_path,
            "/Furniture"
        );
    }

    /// An arc may shift and stretch the animation it pulls in, and this crate
    /// read and wrote those numbers faithfully while ignoring them — so a clip
    /// referenced with `(offset = 100)` composed at the wrong time.
    ///
    /// Checked against `usdcat --flatten`, which maps a sample at `t` to
    /// `t * scale + offset`.
    #[test]
    fn an_arc_shifts_the_animation_it_pulls_in() {
        let mut r = MemoryResolver::new();
        r.insert("clip.usda", include_str!("testdata/comp/offset_clip.usda"));
        let root = parse(include_str!("testdata/comp/offset_use.usda")).unwrap();
        let out = compose(&root, "use.usda", &r, &ComposeOptions::default()).unwrap();

        let samples = out
            .prim_at("/Shifted")
            .expect("the referenced prim")
            .value("xformOp:translate")
            .expect("still animated")
            .samples()
            .expect("time samples")
            .to_vec();

        let theirs = parse(include_str!("testdata/comp/offset_flat.usda")).unwrap();
        let expected = theirs
            .prim_at("/Shifted")
            .unwrap()
            .value("xformOp:translate")
            .unwrap()
            .samples()
            .unwrap();

        assert_eq!(samples.len(), expected.len());
        for (mine, theirs) in samples.iter().zip(expected) {
            assert_eq!(mine.0, theirs.0, "sample time");
            assert_eq!(mine.1.flat_f32(), theirs.1.flat_f32(), "sample value");
        }
        // Explicitly: 0 and 10 became 100 and 120, not 100 and 110.
        assert_eq!(samples[0].0, 100.0);
        assert_eq!(samples[1].0, 120.0);
    }

    /// Shifts compose: an arc inside a shifted arc is shifted by both.
    #[test]
    fn nested_arcs_compose_their_shifts() {
        let mut r = MemoryResolver::new();
        r.insert("clip.usda", include_str!("testdata/comp/offset_clip.usda"))
            // Shifts the clip by 10 at double speed.
            .insert(
                "mid.usda",
                "#usda 1.0\n(\n    defaultPrim = \"Mid\"\n)\ndef Xform \"Mid\" (\n                     references = @clip.usda@ (offset = 10; scale = 2)\n)\n{\n}\n",
            );
        // ...and then shifts that by 100 at triple speed.
        let root = parse(
            "#usda 1.0\ndef Xform \"Top\" (\n    references = @mid.usda@ (offset = 100; scale = 3)\n)\n{\n}\n",
        )
        .unwrap();
        let out = compose(&root, "root.usda", &r, &ComposeOptions::default()).unwrap();
        let samples = out
            .prim_at("/Top")
            .unwrap()
            .value("xformOp:translate")
            .unwrap()
            .samples()
            .unwrap()
            .to_vec();

        // A sample at t lands at (t * 2 + 10) * 3 + 100.
        assert_eq!(samples[0].0, 130.0, "0 -> (0*2+10)*3+100");
        assert_eq!(samples[1].0, 190.0, "10 -> (10*2+10)*3+100");
    }

    /// `delete` takes an arc away, and `reorder` says what order what is
    /// left composes in. Both were parsed and then ignored: a deleted
    /// reference still applied, which is how a shot that drops a prop gets the
    /// prop anyway.
    #[test]
    fn delete_and_reorder_compose_as_openusd_does() {
        let mut r = MemoryResolver::new();
        r.insert("a.usda", include_str!("testdata/listops/a.usda"))
            .insert("b.usda", include_str!("testdata/listops/b.usda"))
            .insert("base.usda", include_str!("testdata/listops/base.usda"));
        let root = parse(include_str!("testdata/listops/shot.usda")).unwrap();
        let ours = compose(&root, "shot.usda", &r, &ComposeOptions::default()).unwrap();
        let theirs = parse(include_str!("testdata/listops/flat.usda")).unwrap();

        // The deleted reference is gone and the other one is not.
        let holder = ours.prim_at("/Root/Holder").expect("Holder");
        assert!(
            holder.value("fromA").is_none(),
            "the deleted reference still applied: {:?}",
            holder.properties
        );
        assert_eq!(holder.value("fromB").unwrap().flat_f32(), vec![2.0]);

        // Children compose in the order the reorder names.
        let order: Vec<&str> = ours
            .prim_at("/Root/Ordered")
            .unwrap()
            .children
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        assert_eq!(order, vec!["Third", "First", "Second"]);

        // Which is what OpenUSD produced from the same files.
        let expected: Vec<&str> = theirs
            .prim_at("/Root/Ordered")
            .unwrap()
            .children
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        assert_eq!(order, expected);
    }

    /// A reorder survives into the flattened layer as well as acting on it —
    /// it still describes the order, so USD keeps it and so does this.
    #[test]
    fn a_reorder_is_kept_as_well_as_applied() {
        let mut r = MemoryResolver::new();
        r.insert("base.usda", include_str!("testdata/listops/base.usda"));
        let root = parse(include_str!("testdata/listops/shot.usda")).unwrap();
        let ours = compose(&root, "shot.usda", &r, &ComposeOptions::default()).unwrap();
        assert!(
            ours.prim_at("/Root/Ordered")
                .unwrap()
                .meta_exact("reorder nameChildren")
                .is_some(),
            "the reorder statement was consumed rather than kept"
        );
    }

    /// The shot's `over` beats the base layer's opinion, which beats the
    /// asset's. Getting this backwards is the classic composition bug.
    #[test]
    fn the_strongest_layer_wins() {
        let ours = compose_shot(&ComposeOptions::default());
        assert_eq!(
            ours.prim_at("/Room/Chair1")
                .unwrap()
                .value("xformOp:translate")
                .unwrap()
                .flat_f32(),
            vec![9.0, 1.0, 0.0],
            "the shot's override should win over the base layer and the asset"
        );
    }

    /// The shot selects `steel`; the asset selects `oak`. The shot is stronger.
    #[test]
    fn a_variant_selection_can_be_overridden_from_a_stronger_layer() {
        let ours = compose_shot(&ComposeOptions::default());
        let colour = ours
            .prim_at("/Room/Chair1")
            .unwrap()
            .value("primvars:displayColor")
            .unwrap()
            .flat_f32();
        assert_eq!(colour, vec![0.7, 0.7, 0.75], "steel, not oak");
    }

    /// A class contributes its opinions and is not itself part of the scene.
    #[test]
    fn an_inherited_class_contributes_but_does_not_appear() {
        let ours = compose_shot(&ComposeOptions::default());
        let chair = ours.prim_at("/Room/Chair1").unwrap();
        assert_eq!(chair.value("purpose").unwrap().as_str(), Some("render"));
        assert_eq!(chair.value("weight").unwrap().flat_f32(), vec![10.0]);
    }

    #[test]
    fn a_payload_can_be_left_unloaded() {
        let mut r = MemoryResolver::new();
        r.insert(
            "heavy.usda",
            "#usda 1.0\n(\n    defaultPrim = \"Heavy\"\n)\ndef Mesh \"Heavy\"\n{\n    int[] faceVertexCounts = [3]\n}\n",
        );
        let root = parse(
            "#usda 1.0\ndef Xform \"Holder\" (\n    payload = @heavy.usda@\n)\n{\n}\n",
        )
        .unwrap();

        let loaded = compose(&root, "root.usda", &r, &ComposeOptions::default()).unwrap();
        assert!(loaded.prim_at("/Holder").unwrap().value("faceVertexCounts").is_some());

        let unloaded = compose(
            &root,
            "root.usda",
            &r,
            &ComposeOptions {
                load_payloads: false,
                ..Default::default()
            },
        )
        .unwrap();
        let holder = unloaded.prim_at("/Holder").expect("the prim is still there");
        assert!(
            holder.value("faceVertexCounts").is_none(),
            "an unloaded payload should bring nothing in"
        );
    }

    /// A layer that references itself must stop rather than recurse forever.
    #[test]
    fn a_cycle_terminates() {
        let mut r = MemoryResolver::new();
        r.insert(
            "a.usda",
            "#usda 1.0\n(\n    defaultPrim = \"A\"\n)\ndef Xform \"A\" (\n    references = @b.usda@\n)\n{\n}\n",
        )
        .insert(
            "b.usda",
            "#usda 1.0\n(\n    defaultPrim = \"B\"\n)\ndef Xform \"B\" (\n    references = @a.usda@\n)\n{\n}\n",
        );
        let root = parse(include_str!("testdata/comp/asset.usda")).unwrap();
        let _ = compose(&root, "asset.usda", &r, &ComposeOptions::default());

        let a = parse("#usda 1.0\ndef Xform \"Top\" (\n    references = @a.usda@\n)\n{\n}\n").unwrap();
        let out = compose(&a, "root.usda", &r, &ComposeOptions::default()).unwrap();
        assert!(out.prim_at("/Top").is_some(), "it should still compose");
    }

    /// A sublayer that includes itself is the same hazard one level up.
    #[test]
    fn a_sublayer_cycle_terminates() {
        let mut r = MemoryResolver::new();
        r.insert(
            "loop.usda",
            "#usda 1.0\n(\n    subLayers = [\n        @loop.usda@\n    ]\n)\ndef Xform \"X\"\n{\n}\n",
        );
        let root = parse(
            "#usda 1.0\n(\n    subLayers = [\n        @loop.usda@\n    ]\n)\n",
        )
        .unwrap();
        let out = compose(&root, "root.usda", &r, &ComposeOptions::default()).unwrap();
        assert!(out.prim_at("/X").is_some());
    }

    #[test]
    fn paths_resolve_against_the_layer_that_named_them() {
        assert_eq!(resolve_path("shots/010/shot.usda", "asset.usda"), "shots/010/asset.usda");
        assert_eq!(resolve_path("shots/010/shot.usda", "../lib/chair.usda"), "shots/lib/chair.usda");
        assert_eq!(resolve_path("shots/010/shot.usda", "./here.usda"), "shots/010/here.usda");
        assert_eq!(resolve_path("shot.usda", "asset.usda"), "asset.usda");
        // Absolute stays absolute, and so does anything with no anchor.
        assert_eq!(resolve_path("shots/shot.usda", "/abs/a.usda"), "/abs/a.usda");
        assert_eq!(resolve_path("", "a.usda"), "a.usda");
    }

    /// Composition is idempotent: flattening a flat layer changes nothing.
    #[test]
    fn composing_a_flat_layer_is_a_no_op() {
        let layer = parse(include_str!("testdata/rich.usda")).unwrap();
        let composed = compose(
            &layer,
            "rich.usda",
            &MemoryResolver::new(),
            &ComposeOptions::default(),
        )
        .unwrap();
        assert_eq!(layer_to_usda(&composed), layer_to_usda(&layer));
    }
}

#[cfg(test)]
mod gaps {
    use super::*;
    use super::super::parse::parse;

    /// `foo = None` blocks a weaker layer's opinion. It is not the same as
    /// saying nothing: a declaration with no value takes what is below it, and
    /// treating a block that way does the opposite of what it asks.
    #[test]
    fn a_blocked_value_stays_blocked() {
        let mut r = MemoryResolver::new();
        r.insert("weak.usda", include_str!("testdata/block/weak.usda"));
        let root = parse(include_str!("testdata/block/strong.usda")).unwrap();
        let ours = compose(&root, "strong.usda", &r, &ComposeOptions::default()).unwrap();
        let theirs = parse(include_str!("testdata/block/flat.usda")).unwrap();

        let root_prim = ours.prim_at("/Root").expect("Root");
        assert_eq!(
            root_prim.value("blocked"),
            Some(&UsdValue::Block),
            "the block should have stopped the weaker `= 2`"
        );
        // Everything else still comes through from below.
        assert_eq!(root_prim.value("kept").unwrap().flat_f32(), vec![1.0]);
        assert_eq!(root_prim.value("stamp").unwrap().flat_f32(), vec![1.0, 2.0, 3.0]);
        // And a time code is a value like any other.
        assert_eq!(root_prim.value("when").unwrap().flat_f32(), vec![24.0]);

        // Which is the stage OpenUSD composes from the same files.
        assert_eq!(
            theirs.prim_at("/Root").unwrap().value("blocked"),
            Some(&UsdValue::Block)
        );
    }

    /// A variable in an asset path is how one shot layer drives which of
    /// several assets a whole stage pulls in.
    #[test]
    fn an_expression_in_an_asset_path_resolves() {
        let mut r = MemoryResolver::new();
        r.insert("red_asset.usda", include_str!("testdata/expr/red_asset.usda"))
            .insert("blue_asset.usda", include_str!("testdata/expr/blue_asset.usda"));
        let root = parse(include_str!("testdata/expr/expr.usda")).unwrap();
        let ours = compose(&root, "expr.usda", &r, &ComposeOptions::default()).unwrap();

        let thing = ours.prim_at("/Root/Thing").expect("the referenced prim");
        assert_eq!(
            thing.value("which").unwrap().flat_f32(),
            vec![1.0],
            "COLOR was red, so the red asset should have been pulled in"
        );

        // Which is what OpenUSD composes from the same files.
        let theirs = parse(include_str!("testdata/expr/flat.usda")).unwrap();
        assert_eq!(
            thing.value("which").unwrap().flat_f32(),
            theirs.prim_at("/Root/Thing").unwrap().value("which").unwrap().flat_f32()
        );
    }

    /// Change the variable and a different asset arrives, with nothing else
    /// edited — the whole point of the feature.
    #[test]
    fn changing_the_variable_changes_the_asset() {
        let mut r = MemoryResolver::new();
        r.insert("red_asset.usda", include_str!("testdata/expr/red_asset.usda"))
            .insert("blue_asset.usda", include_str!("testdata/expr/blue_asset.usda"));
        let root = parse(
            &include_str!("testdata/expr/expr.usda").replace("\"red\"", "\"blue\""),
        )
        .unwrap();
        let ours = compose(&root, "expr.usda", &r, &ComposeOptions::default()).unwrap();
        assert_eq!(
            ours.prim_at("/Root/Thing").unwrap().value("which").unwrap().flat_f32(),
            vec![2.0]
        );
    }

    /// An expression that cannot be evaluated opens nothing, rather than
    /// opening whatever the unevaluated text happens to name.
    #[test]
    fn an_unresolvable_expression_opens_nothing() {
        let mut r = MemoryResolver::new();
        r.insert("red_asset.usda", include_str!("testdata/expr/red_asset.usda"));
        let root = parse(
            r#"#usda 1.0
def Xform "Root"
{
    def Scope "Thing" (
        references = @`"${NOT_SET}_asset.usda"`@
    )
    {
    }
}
"#,
        )
        .unwrap();
        let ours = compose(&root, "expr.usda", &r, &ComposeOptions::default()).unwrap();
        assert!(
            ours.prim_at("/Root/Thing").unwrap().value("which").is_none(),
            "an undefined variable should not have resolved to anything"
        );
    }

    /// `active = false` takes a prim off the stage, and its children with it.
    /// It is how a shot switches a prop off without editing the asset.
    #[test]
    fn an_inactive_prim_is_not_on_the_stage() {
        let root = parse(include_str!("testdata/gaps/active.usda")).unwrap();
        let ours = compose(
            &root,
            "active.usda",
            &MemoryResolver::new(),
            &ComposeOptions::default(),
        )
        .unwrap();
        let theirs = parse(include_str!("testdata/gaps/flat_active.usda")).unwrap();

        assert!(ours.prim_at("/Root/Kept").is_some());
        assert!(
            ours.prim_at("/Root/Dropped").is_none(),
            "an inactive prim should be gone: {:?}",
            ours.prim_at("/Root").map(|p| p.children.iter().map(|c| &c.name).collect::<Vec<_>>())
        );
        assert!(ours.prim_at("/Root/Dropped/ChildOfDropped").is_none());

        // Which is the stage OpenUSD composes from the same file.
        let names = |l: &UsdLayer| -> Vec<String> {
            l.prims_by_path().into_iter().map(|(p, _)| p).collect()
        };
        assert_eq!(names(&ours), names(&theirs));
    }

    /// A sublayer may shift and stretch the time of everything in it — the
    /// same map an arc applies, one level up. It was read and written
    /// faithfully and then ignored, so a shot composed unshifted.
    #[test]
    fn a_sublayers_time_offset_applies() {
        let mut r = MemoryResolver::new();
        r.insert("anim_sub.usda", include_str!("testdata/gaps/anim_sub.usda"));
        let root = parse(include_str!("testdata/gaps/shifted.usda")).unwrap();
        let ours = compose(&root, "shifted.usda", &r, &ComposeOptions::default()).unwrap();

        let samples = ours
            .prim_at("/Moving")
            .expect("the prim came through the sublayer")
            .value("xformOp:translate")
            .expect("still animated")
            .samples()
            .expect("time samples")
            .to_vec();

        let theirs = parse(include_str!("testdata/gaps/flat_shifted.usda")).unwrap();
        let expected = theirs
            .prim_at("/Moving")
            .unwrap()
            .value("xformOp:translate")
            .unwrap()
            .samples()
            .unwrap();

        assert_eq!(samples.len(), expected.len());
        for (mine, theirs) in samples.iter().zip(expected) {
            assert_eq!(mine.0, theirs.0, "sample time");
        }
        // Spelled out: 0 and 10 became 100 and 120, not 100 and 110.
        assert_eq!(samples[0].0, 100.0);
        assert_eq!(samples[1].0, 120.0);
    }

    /// A relocation renames a prim an arc brought in: the asset calls it
    /// `OldName` and this stage calls it `NewName`.
    #[test]
    fn a_relocation_renames_what_an_arc_brought_in() {
        let mut r = MemoryResolver::new();
        r.insert("asset_r.usda", include_str!("testdata/gaps/asset_r.usda"));
        let root = parse(include_str!("testdata/gaps/reloc2.usda")).unwrap();
        let ours = compose(&root, "reloc2.usda", &r, &ComposeOptions::default()).unwrap();
        let theirs = parse(include_str!("testdata/gaps/flat_reloc2.usda")).unwrap();

        assert!(
            ours.prim_at("/Holder/OldName").is_none(),
            "the old name should be gone"
        );
        let renamed = ours
            .prim_at("/Holder/NewName")
            .expect("the prim should be under its new name");
        assert_eq!(renamed.value("marker").unwrap().flat_f32(), vec![7.0]);

        let names = |l: &UsdLayer| -> Vec<String> {
            l.prims_by_path().into_iter().map(|(p, _)| p).collect()
        };
        assert_eq!(names(&ours), names(&theirs));
    }

    /// A relocation that would move a prim to a different parent is not a
    /// rename, and doing half of it is worse than doing none.
    #[test]
    fn a_relocation_across_parents_is_refused() {
        let mut layer = parse(
            r#"#usda 1.0
def Scope "A"
{
    def Scope "Thing"
    {
    }
}

def Scope "B"
{
}
"#,
        )
        .unwrap();
        relocate(&mut layer, "/A/Thing", "/B/Thing");
        assert!(layer.prim_at("/A/Thing").is_some(), "left where it was");
        assert!(layer.prim_at("/B/Thing").is_none());
    }
}
