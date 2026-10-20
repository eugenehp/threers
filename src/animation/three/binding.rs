//! three.js r165 `PropertyBinding`: track-name parsing, node lookup and the
//! bound getter/setter pair, over a scene abstracted as an
//! [`AnimationHost`].
//!
//! A track name is `[dir/]nodeName[.objectName[objectIndex]].propertyName[propertyIndex]`
//! — `Box.position`, `.morphTargetInfluences[smile]`, `Arm.material.opacity`,
//! `.bones[hip].quaternion`, `mesh.materials[1].color`, or a node's uuid in
//! place of its name. [`parse_track_name`] is three.js's regular expression,
//! matched with its backtracking order, plus the allow-list that splits
//! `node.material` into node and object. [`find_node`] is
//! `PropertyBinding.findNode` over any [`NodeTree`].
//!
//! What a binding resolves to is the host's business: a native scene knows
//! transforms, visibility, morph influences, material and light parameters;
//! the wasm host forwards to the JS object three.js code would have written.

use super::track::ValueType;

/// `PropertyBinding.parseTrackName(trackName)`'s result.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ParsedPath {
    /// Parent directories (`a/b:`), unused as in three.js.
    pub directory_name: String,
    /// `undefined` when the name has no node part.
    pub node_name: Option<String>,
    pub object_name: Option<String>,
    pub object_index: Option<String>,
    pub property_name: String,
    pub property_index: Option<String>,
}

/// Characters three.js reserves for track syntax: `[ ] . : /`.
fn reserved(c: char) -> bool {
    matches!(c, '[' | ']' | '.' | ':' | '/')
}
/// `_wordChar`: anything not reserved.
fn word(c: char) -> bool {
    !reserved(c)
}
/// `_wordCharOrDot`: a word character or `.`.
fn word_or_dot(c: char) -> bool {
    !matches!(c, '[' | ']' | ':' | '/')
}

const SUPPORTED_OBJECT_NAMES: [&str; 4] = ["material", "materials", "bones", "map"];

/// Why a track name could not be parsed (three.js throws).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError(pub String);

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PropertyBinding: Cannot parse trackName: {}", self.0)
    }
}
impl std::error::Error for ParseError {}

/// `PropertyBinding.sanitizeNodeName(name)`: whitespace to `_`, reserved
/// characters removed.
pub fn sanitize_node_name(name: &str) -> String {
    name.chars()
        .filter_map(|c| {
            if c.is_whitespace() {
                Some('_')
            } else if reserved(c) {
                None
            } else {
                Some(c)
            }
        })
        .collect()
}

type Prop = (String, Option<String>);

/// `\.(WC+)(?:\[(.+)\])?$` against `r`.
fn match_property(r: &[char]) -> Option<Prop> {
    if r.first() != Some(&'.') {
        return None;
    }
    let wmax = r[1..].iter().take_while(|&&c| word(c)).count();
    for wl in (1..=wmax).rev() {
        let name: String = r[1..1 + wl].iter().collect();
        let after = &r[1 + wl..];
        // greedy optional bracket first
        if after.len() >= 3 && after[0] == '[' && after[after.len() - 1] == ']' {
            let inner: String = after[1..after.len() - 1].iter().collect();
            if !inner.contains(['\n', '\r', '\u{2028}', '\u{2029}']) {
                return Some((name, Some(inner)));
            }
        }
        if after.is_empty() {
            return Some((name, None));
        }
    }
    None
}

type Tail = (Option<String>, Option<String>, Prop);

/// `(?:\.(WC+)(?:\[(.+)\])?)?` then the property, against `r`.
fn match_tail(r: &[char]) -> Option<Tail> {
    if r.first() == Some(&'.') {
        let wmax = r[1..].iter().take_while(|&&c| word(c)).count();
        for wl in (1..=wmax).rev() {
            let obj: String = r[1..1 + wl].iter().collect();
            let after = &r[1 + wl..];
            if after.first() == Some(&'[') {
                // `.+` is greedy: the latest `]` first.
                let mut close = after.len();
                while close > 2 {
                    close -= 1;
                    if after[close] != ']' {
                        continue;
                    }
                    let inner = &after[1..close];
                    if inner.is_empty()
                        || inner
                            .iter()
                            .any(|c| matches!(c, '\n' | '\r' | '\u{2028}' | '\u{2029}'))
                    {
                        continue;
                    }
                    if let Some(p) = match_property(&after[close + 1..]) {
                        return Some((Some(obj), Some(inner.iter().collect()), p));
                    }
                }
            }
            if let Some(p) = match_property(after) {
                return Some((Some(obj), None, p));
            }
        }
    }
    match_property(r).map(|p| (None, None, p))
}

/// `PropertyBinding.parseTrackName(trackName)`.
pub fn parse_track_name(track_name: &str) -> Result<ParsedPath, ParseError> {
    let chars: Vec<char> = track_name.chars().collect();
    let err = || ParseError(track_name.to_string());
    // ((?:WC+[\/:])*): everything through the last '/' or ':' (nothing after
    // may contain one), in non-empty word segments.
    let dir_end = chars
        .iter()
        .rposition(|&c| c == '/' || c == ':')
        .map(|i| i + 1)
        .unwrap_or(0);
    let dir = &chars[..dir_end];
    if dir_end > 0 {
        let mut seg = 0usize;
        for &c in dir {
            if c == '/' || c == ':' {
                if seg == 0 {
                    return Err(err());
                }
                seg = 0;
            } else if word(c) {
                seg += 1;
            } else {
                return Err(err());
            }
        }
    }
    let rest = &chars[dir_end..];
    // (WCOD+)?: greedy, longest first, then absent.
    let node_max = rest.iter().take_while(|&&c| word_or_dot(c)).count();
    let mut found: Option<(Option<String>, Tail)> = None;
    for l in (1..=node_max).rev() {
        if let Some(t) = match_tail(&rest[l..]) {
            found = Some((Some(rest[..l].iter().collect()), t));
            break;
        }
    }
    if found.is_none() {
        if let Some(t) = match_tail(rest) {
            found = Some((None, t));
        }
    }
    let (node_name, (object_name, object_index, (property_name, property_index))) =
        found.ok_or_else(err)?;
    let mut results = ParsedPath {
        directory_name: dir.iter().collect(),
        node_name,
        object_name,
        object_index,
        property_name,
        property_index,
    };
    if let Some(node) = results.node_name.clone() {
        if let Some(last_dot) = node.rfind('.') {
            let object_name = &node[last_dot + 1..];
            // Object names must be checked against an allowlist.
            if SUPPORTED_OBJECT_NAMES.contains(&object_name) {
                results.object_name = Some(object_name.to_string());
                results.node_name = Some(node[..last_dot].to_string());
            }
        }
    }
    if results.property_name.is_empty() {
        return Err(ParseError(format!(
            "can not parse propertyName from trackName: {track_name}"
        )));
    }
    Ok(results)
}

/// The types a scene animates through: what a node is, and what a bound
/// property is. A mixer is generic over these, and takes the scene itself
/// (an [`AnimationHost`] of the same kind) as an argument to each call, so
/// it can outlive any borrow of the scene.
pub trait HostKind {
    /// A scene node (three.js `Object3D`, or any object used as a root).
    type Node: Clone + PartialEq + std::fmt::Debug;
    /// A resolved property (three.js's bound getter/setter pair).
    type Target: std::fmt::Debug;
}

/// The scene a mixer animates: how nodes are found, and how a parsed path
/// on a node becomes something values can be read from and written to.
pub trait AnimationHost<K: HostKind> {
    /// `node.uuid`.
    fn node_uuid(&self, node: &K::Node) -> String;

    /// `PropertyBinding.findNode(root, nodeName)`.
    fn find_node(&mut self, root: &K::Node, node_name: Option<&str>) -> Option<K::Node>;

    /// The rest of `PropertyBinding.bind()` once the node is known: the
    /// object (`material`, `materials[i]`, `bones[name]`, `map`, any
    /// `node[objectName]`), the property and its binding type. `None`: the
    /// binding is unavailable (three.js logs, then reads and writes nothing).
    /// `value_type` and `value_size` are the track's.
    fn bind(
        &mut self,
        node: &K::Node,
        parsed: &ParsedPath,
        path: &str,
        value_type: ValueType,
        value_size: usize,
    ) -> Option<K::Target>;

    /// `getValue(buffer, offset)`: `out` runs from the offset to the end of
    /// the mixer's buffer, as three.js's setters may read past one value.
    fn get_value(&mut self, target: &K::Target, out: &mut [f64]);

    /// `setValue(buffer, offset)`.
    fn set_value(&mut self, target: &K::Target, values: &[f64]);

    /// `getValue` for a string track.
    fn get_strings(&mut self, _target: &K::Target, _out: &mut [Option<String>]) {}

    /// `setValue` for a string track.
    fn set_strings(&mut self, _target: &K::Target, _values: &[Option<String>]) {}

    /// A target is no longer used (its binding was unbound or dropped).
    fn release(&mut self, _target: K::Target) {}
}

/// What [`find_node`] needs of a scene.
pub trait NodeTree {
    type Id: Clone + PartialEq;
    fn name_is(&self, node: &Self::Id, name: &str) -> bool;
    fn uuid_is(&self, node: &Self::Id, uuid: &str) -> bool;
    /// `node.skeleton.bones`, when the node has a skeleton.
    fn skeleton_bones(&self, node: &Self::Id) -> Option<Vec<Self::Id>>;
    fn children(&self, node: &Self::Id) -> Vec<Self::Id>;
}

/// `PropertyBinding.findNode(root, nodeName)`: the root itself for no name,
/// `.`, its name or its uuid; else a bone of its skeleton by name; else the
/// first node in a pre-order walk of its subtree with that name or uuid.
pub fn find_node<T: NodeTree>(tree: &T, root: &T::Id, node_name: Option<&str>) -> Option<T::Id> {
    let name = match node_name {
        None | Some("") | Some(".") => return Some(root.clone()),
        Some(n) => n,
    };
    if tree.name_is(root, name) || tree.uuid_is(root, name) {
        return Some(root.clone());
    }
    // search into skeleton bones.
    if let Some(bones) = tree.skeleton_bones(root) {
        if let Some(b) = bones.into_iter().find(|b| tree.name_is(b, name)) {
            return Some(b);
        }
    }
    // search into node subtree.
    fn search<T: NodeTree>(tree: &T, children: Vec<T::Id>, name: &str) -> Option<T::Id> {
        for child in children {
            if tree.name_is(&child, name) || tree.uuid_is(&child, name) {
                return Some(child);
            }
            if let Some(r) = search(tree, tree.children(&child), name) {
                return Some(r);
            }
        }
        None
    }
    search(tree, tree.children(root), name)
}

#[derive(Debug)]
enum BindState<T> {
    /// Not bound yet: the first get or set binds.
    Unbound,
    /// Bound to nothing (no node, no property): reads and writes do nothing.
    Unavailable,
    Bound(T),
}

/// three.js `PropertyBinding` (one root, one path).
#[derive(Debug)]
pub struct PropertyBinding<K: HostKind> {
    pub path: String,
    pub parsed_path: ParsedPath,
    pub root_node: K::Node,
    node: Option<K::Node>,
    value_type: ValueType,
    value_size: usize,
    state: BindState<K::Target>,
}

impl<K: HostKind> PropertyBinding<K> {
    /// `new PropertyBinding(rootNode, path, parsedPath)`: finds the node now.
    pub fn new<H: AnimationHost<K>>(
        host: &mut H,
        root_node: K::Node,
        path: &str,
        parsed_path: Option<ParsedPath>,
        value_type: ValueType,
        value_size: usize,
    ) -> Result<Self, ParseError> {
        let parsed_path = match parsed_path {
            Some(p) => p,
            None => parse_track_name(path)?,
        };
        let node = host.find_node(&root_node, parsed_path.node_name.as_deref());
        Ok(Self {
            path: path.to_string(),
            parsed_path,
            root_node,
            node,
            value_type,
            value_size,
            state: BindState::Unbound,
        })
    }

    /// The node found for the path, if any.
    pub fn node(&self) -> Option<&K::Node> {
        self.node.as_ref()
    }

    /// The resolved target, once bound.
    pub fn target(&self) -> Option<&K::Target> {
        match &self.state {
            BindState::Bound(t) => Some(t),
            _ => None,
        }
    }

    /// Whether `bind()` has run (successfully or not).
    pub fn is_bound(&self) -> bool {
        !matches!(self.state, BindState::Unbound)
    }

    /// `bind()`.
    pub fn bind<H: AnimationHost<K>>(&mut self, host: &mut H) {
        if self.node.is_none() {
            self.node = host.find_node(&self.root_node, self.parsed_path.node_name.as_deref());
        }
        if let BindState::Bound(t) = std::mem::replace(&mut self.state, BindState::Unavailable) {
            host.release(t);
        }
        let Some(node) = self.node.clone() else {
            log::warn!(
                "THREE.PropertyBinding: No target node found for track: {}.",
                self.path
            );
            return;
        };
        if let Some(t) = host.bind(
            &node,
            &self.parsed_path,
            &self.path,
            self.value_type,
            self.value_size,
        ) {
            self.state = BindState::Bound(t);
        }
    }

    /// `unbind()`.
    pub fn unbind<H: AnimationHost<K>>(&mut self, host: &mut H) {
        self.node = None;
        if let BindState::Bound(t) = std::mem::replace(&mut self.state, BindState::Unbound) {
            host.release(t);
        }
    }

    /// `getValue(buffer, offset)` (binding first if needed).
    pub fn get_value<H: AnimationHost<K>>(&mut self, host: &mut H, out: &mut [f64]) {
        if let BindState::Unbound = self.state {
            self.bind(host);
        }
        if let BindState::Bound(t) = &self.state {
            host.get_value(t, out);
        }
    }

    /// `setValue(buffer, offset)` (binding first if needed).
    pub fn set_value<H: AnimationHost<K>>(&mut self, host: &mut H, values: &[f64]) {
        if let BindState::Unbound = self.state {
            self.bind(host);
        }
        if let BindState::Bound(t) = &self.state {
            host.set_value(t, values);
        }
    }

    pub fn get_strings<H: AnimationHost<K>>(&mut self, host: &mut H, out: &mut [Option<String>]) {
        if let BindState::Unbound = self.state {
            self.bind(host);
        }
        if let BindState::Bound(t) = &self.state {
            host.get_strings(t, out);
        }
    }

    pub fn set_strings<H: AnimationHost<K>>(&mut self, host: &mut H, values: &[Option<String>]) {
        if let BindState::Unbound = self.state {
            self.bind(host);
        }
        if let BindState::Bound(t) = &self.state {
            host.set_strings(t, values);
        }
    }

    /// Give the target back to the host (the binding is being dropped).
    pub fn release<H: AnimationHost<K>>(&mut self, host: &mut H) {
        if let BindState::Bound(t) = std::mem::replace(&mut self.state, BindState::Unbound) {
            host.release(t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> ParsedPath {
        parse_track_name(s).unwrap()
    }

    #[test]
    fn parses_three_examples() {
        let a = p("Box.position");
        assert_eq!(a.node_name.as_deref(), Some("Box"));
        assert_eq!(a.property_name, "position");
        let a = p(".morphTargetInfluences[smile]");
        assert_eq!(a.node_name, None);
        assert_eq!(a.property_name, "morphTargetInfluences");
        assert_eq!(a.property_index.as_deref(), Some("smile"));
        let a = p("Box.material.color");
        assert_eq!(a.node_name.as_deref(), Some("Box"));
        assert_eq!(a.object_name.as_deref(), Some("material"));
        assert_eq!(a.property_name, "color");
        let a = p(".bones[hip].position");
        assert_eq!(a.object_name.as_deref(), Some("bones"));
        assert_eq!(a.object_index.as_deref(), Some("hip"));
        assert_eq!(a.property_name, "position");
        let a = p("mesh.materials[1].opacity");
        assert_eq!(a.node_name.as_deref(), Some("mesh"));
        assert_eq!(a.object_name.as_deref(), Some("materials"));
        assert_eq!(a.object_index.as_deref(), Some("1"));
        let a = p("dir/sub:Arm.L.quaternion");
        assert_eq!(a.directory_name, "dir/sub:");
        assert_eq!(a.node_name.as_deref(), Some("Arm.L"));
        assert_eq!(a.property_name, "quaternion");
        let a = p("uuid-1234.scale[0]");
        assert_eq!(a.node_name.as_deref(), Some("uuid-1234"));
        assert_eq!(a.property_index.as_deref(), Some("0"));
        assert!(parse_track_name("position").is_err());
        assert!(parse_track_name("").is_err());
    }

    #[test]
    fn sanitizes() {
        assert_eq!(sanitize_node_name("a b.c[d]:e/f"), "a_bcdef");
    }
}
