//! three.js r165 `AnimationObjectGroup`: objects that share one animation
//! state, used as a mixer root. Bindings are kept per path and per object,
//! with the same cache bookkeeping as three.js — objects removed with
//! [`remove`](AnimationObjectGroup::remove) stay cached (and keep their
//! bindings) until [`uncache`](AnimationObjectGroup::uncache).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::binding::{AnimationHost, HostKind, ParsedPath, PropertyBinding};
use super::track::ValueType;

/// A shared handle to a group (three.js passes the object around).
pub type GroupRef<K> = Arc<Mutex<AnimationObjectGroup<K>>>;

/// three.js `AnimationObjectGroup`.
pub struct AnimationObjectGroup<K: HostKind> {
    pub uuid: String,
    objects: Vec<K::Node>,
    /// Objects `[0, n_cached_objects)` are cached (removed); the rest active.
    n_cached_objects: usize,
    indices_by_uuid: HashMap<String, usize>,
    paths: Vec<String>,
    parsed_paths: Vec<ParsedPath>,
    value_types: Vec<(ValueType, usize)>,
    /// Per path, one binding per object (absent for objects cached when the
    /// path was subscribed).
    bindings: Vec<Vec<Option<PropertyBinding<K>>>>,
    bindings_indices_by_path: HashMap<String, usize>,
}

impl<K: HostKind> std::fmt::Debug for AnimationObjectGroup<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnimationObjectGroup")
            .field("uuid", &self.uuid)
            .field("objects", &self.objects)
            .field("n_cached_objects", &self.n_cached_objects)
            .field("paths", &self.paths)
            .finish()
    }
}

impl<K: HostKind> AnimationObjectGroup<K> {
    /// `new AnimationObjectGroup(...objects)`.
    pub fn new<H: AnimationHost<K>>(host: &H, objects: Vec<K::Node>) -> Self {
        let mut indices = HashMap::new();
        for (i, o) in objects.iter().enumerate() {
            indices.insert(host.node_uuid(o), i);
        }
        Self {
            uuid: super::clip::generate_uuid(),
            objects,
            n_cached_objects: 0,
            indices_by_uuid: indices,
            paths: Vec::new(),
            parsed_paths: Vec::new(),
            value_types: Vec::new(),
            bindings: Vec::new(),
            bindings_indices_by_path: HashMap::new(),
        }
    }

    /// Wrap in a shared handle.
    pub fn into_ref(self) -> GroupRef<K> {
        Arc::new(Mutex::new(self))
    }

    /// `stats.objects.total`.
    pub fn total_objects(&self) -> usize {
        self.objects.len()
    }
    /// `stats.objects.inUse`.
    pub fn objects_in_use(&self) -> usize {
        self.objects.len() - self.n_cached_objects
    }
    /// `stats.bindingsPerObject`.
    pub fn bindings_per_object(&self) -> usize {
        self.bindings.len()
    }
    /// `nCachedObjects_`.
    pub fn n_cached_objects(&self) -> usize {
        self.n_cached_objects
    }
    /// The objects, cached ones first.
    pub fn objects(&self) -> &[K::Node] {
        &self.objects
    }

    fn new_binding<H: AnimationHost<K>>(
        host: &mut H,
        object: &K::Node,
        path: &str,
        parsed: &ParsedPath,
        ty: (ValueType, usize),
    ) -> Option<PropertyBinding<K>> {
        PropertyBinding::new(host, object.clone(), path, Some(parsed.clone()), ty.0, ty.1).ok()
    }

    /// `add(...objects)`.
    pub fn add<H: AnimationHost<K>>(&mut self, host: &mut H, objects: &[K::Node]) {
        let n_bindings = self.bindings.len();
        let mut known_object: Option<K::Node> = None;
        let mut n_objects = self.objects.len();
        let mut n_cached = self.n_cached_objects;
        for object in objects {
            let uuid = host.node_uuid(object);
            match self.indices_by_uuid.get(&uuid).copied() {
                None => {
                    let index = n_objects;
                    n_objects += 1;
                    self.indices_by_uuid.insert(uuid, index);
                    self.objects.push(object.clone());
                    for j in 0..n_bindings {
                        let b = Self::new_binding(
                            host,
                            object,
                            &self.paths[j],
                            &self.parsed_paths[j],
                            self.value_types[j],
                        );
                        let list = &mut self.bindings[j];
                        if list.len() < index {
                            list.resize_with(index, || None);
                        }
                        list.push(b);
                    }
                }
                Some(index) if index < n_cached => {
                    known_object = Some(self.objects[index].clone());
                    // move existing object into the ACTIVE region
                    n_cached -= 1;
                    let first_active = n_cached;
                    let last_cached_object = self.objects[first_active].clone();
                    self.indices_by_uuid
                        .insert(host.node_uuid(&last_cached_object), index);
                    self.objects[index] = last_cached_object;
                    self.indices_by_uuid.insert(uuid, first_active);
                    self.objects[first_active] = object.clone();
                    for j in 0..n_bindings {
                        let list = &mut self.bindings[j];
                        let need = index.max(first_active) + 1;
                        if list.len() < need {
                            list.resize_with(need, || None);
                        }
                        let last_cached = list[first_active].take();
                        let mut binding = list[index].take();
                        list[index] = last_cached;
                        if binding.is_none() {
                            binding = Self::new_binding(
                                host,
                                object,
                                &self.paths[j],
                                &self.parsed_paths[j],
                                self.value_types[j],
                            );
                        }
                        list[first_active] = binding;
                    }
                }
                Some(index) => {
                    if Some(&self.objects[index]) != known_object.as_ref() {
                        log::error!(
                            "THREE.AnimationObjectGroup: Different objects with the same UUID detected. \
                             Clean the caches or recreate your infrastructure when reloading scenes."
                        );
                    }
                    // else the object is already where we want it to be
                }
            }
        }
        self.n_cached_objects = n_cached;
    }

    /// `remove(...objects)`: move active objects to the cached region.
    pub fn remove<H: AnimationHost<K>>(&mut self, host: &H, objects: &[K::Node]) {
        let n_bindings = self.bindings.len();
        let mut n_cached = self.n_cached_objects;
        for object in objects {
            let uuid = host.node_uuid(object);
            let Some(index) = self.indices_by_uuid.get(&uuid).copied() else {
                continue;
            };
            if index >= n_cached {
                // move existing object into the CACHED region
                let last_cached_index = n_cached;
                n_cached += 1;
                let first_active_object = self.objects[last_cached_index].clone();
                self.indices_by_uuid
                    .insert(host.node_uuid(&first_active_object), index);
                self.objects[index] = first_active_object;
                self.indices_by_uuid.insert(uuid, last_cached_index);
                self.objects[last_cached_index] = object.clone();
                for j in 0..n_bindings {
                    let list = &mut self.bindings[j];
                    let need = index.max(last_cached_index) + 1;
                    if list.len() < need {
                        list.resize_with(need, || None);
                    }
                    list.swap(index, last_cached_index);
                }
            }
        }
        self.n_cached_objects = n_cached;
    }

    /// `uncache(...objects)`: forget objects (and their bindings) entirely.
    pub fn uncache<H: AnimationHost<K>>(&mut self, host: &mut H, objects: &[K::Node]) {
        let n_bindings = self.bindings.len();
        let mut n_cached = self.n_cached_objects;
        let mut n_objects = self.objects.len();
        for object in objects {
            let uuid = host.node_uuid(object);
            let Some(index) = self.indices_by_uuid.remove(&uuid) else {
                continue;
            };
            if index < n_cached {
                // object is cached, shrink the CACHED region
                n_cached -= 1;
                let first_active_index = n_cached;
                let last_cached_object = self.objects[first_active_index].clone();
                n_objects -= 1;
                let last_index = n_objects;
                let last_object = self.objects[last_index].clone();
                // last cached object takes this object's place
                self.indices_by_uuid
                    .insert(host.node_uuid(&last_cached_object), index);
                self.objects[index] = last_cached_object;
                // last object goes to the activated slot and pop
                self.indices_by_uuid
                    .insert(host.node_uuid(&last_object), first_active_index);
                self.objects[first_active_index] = last_object;
                self.objects.pop();
                for j in 0..n_bindings {
                    let list = &mut self.bindings[j];
                    if list.len() < last_index + 1 {
                        list.resize_with(last_index + 1, || None);
                    }
                    let last_cached = list[first_active_index].take();
                    let last = list[last_index].take();
                    if let Some(mut gone) = std::mem::replace(&mut list[index], last_cached) {
                        gone.release(host);
                    }
                    list[first_active_index] = last;
                    list.pop();
                }
            } else {
                // object is active, just swap with the last and pop
                n_objects -= 1;
                let last_index = n_objects;
                if last_index > 0 {
                    let last_object = self.objects[last_index].clone();
                    self.indices_by_uuid
                        .insert(host.node_uuid(&last_object), index);
                }
                let last_object = self.objects[last_index].clone();
                self.objects[index] = last_object;
                self.objects.pop();
                for j in 0..n_bindings {
                    let list = &mut self.bindings[j];
                    if list.len() < last_index + 1 {
                        list.resize_with(last_index + 1, || None);
                    }
                    let last = list[last_index].take();
                    if let Some(mut gone) = std::mem::replace(&mut list[index], last) {
                        gone.release(host);
                    }
                    list.pop();
                }
            }
        }
        self.n_cached_objects = n_cached;
    }

    /// `subscribe_(path, parsedPath)`: the per-object bindings for a path,
    /// created for the active objects on first use. Returns the path index.
    pub fn subscribe<H: AnimationHost<K>>(
        &mut self,
        host: &mut H,
        path: &str,
        parsed: &ParsedPath,
        value_type: ValueType,
        value_size: usize,
    ) -> usize {
        if let Some(&i) = self.bindings_indices_by_path.get(path) {
            return i;
        }
        let index = self.bindings.len();
        self.bindings_indices_by_path
            .insert(path.to_string(), index);
        self.paths.push(path.to_string());
        self.parsed_paths.push(parsed.clone());
        self.value_types.push((value_type, value_size));
        let mut list: Vec<Option<PropertyBinding<K>>> =
            (0..self.objects.len()).map(|_| None).collect();
        for (slot, object) in list.iter_mut().zip(&self.objects).skip(self.n_cached_objects) {
            *slot = Self::new_binding(host, object, path, parsed, (value_type, value_size));
        }
        self.bindings.push(list);
        index
    }

    /// `unsubscribe_(path)`.
    pub fn unsubscribe<H: AnimationHost<K>>(&mut self, host: &mut H, path: &str) {
        let Some(index) = self.bindings_indices_by_path.remove(path) else {
            return;
        };
        let last = self.bindings.len() - 1;
        if index != last {
            let last_path = self.paths[last].clone();
            self.bindings_indices_by_path.insert(last_path, index);
        }
        let mut gone = self.bindings.swap_remove(index);
        for b in gone.iter_mut().flatten() {
            b.release(host);
        }
        self.parsed_paths.swap_remove(index);
        self.paths.swap_remove(index);
        self.value_types.swap_remove(index);
    }

    fn path_index(&self, path: &str) -> Option<usize> {
        self.bindings_indices_by_path.get(path).copied()
    }

    /// Composite `bind()`: every active object's binding.
    pub fn bind_path<H: AnimationHost<K>>(&mut self, host: &mut H, path: &str) {
        let Some(j) = self.path_index(path) else {
            return;
        };
        let start = self.n_cached_objects;
        for b in self.bindings[j].iter_mut().skip(start).flatten() {
            b.bind(host);
        }
    }

    /// Composite `unbind()`.
    pub fn unbind_path<H: AnimationHost<K>>(&mut self, host: &mut H, path: &str) {
        let Some(j) = self.path_index(path) else {
            return;
        };
        let start = self.n_cached_objects;
        for b in self.bindings[j].iter_mut().skip(start).flatten() {
            b.unbind(host);
        }
    }

    /// Composite `getValue`: binds all, then reads the first active object.
    pub fn get_value<H: AnimationHost<K>>(&mut self, host: &mut H, path: &str, out: &mut [f64]) {
        self.bind_path(host, path);
        let Some(j) = self.path_index(path) else {
            return;
        };
        let first = self.n_cached_objects;
        if let Some(Some(b)) = self.bindings[j].get_mut(first) {
            b.get_value(host, out);
        }
    }

    /// Composite `setValue`: every active object.
    pub fn set_value<H: AnimationHost<K>>(&mut self, host: &mut H, path: &str, values: &[f64]) {
        let Some(j) = self.path_index(path) else {
            return;
        };
        let start = self.n_cached_objects;
        for b in self.bindings[j].iter_mut().skip(start).flatten() {
            b.set_value(host, values);
        }
    }

    pub fn get_strings<H: AnimationHost<K>>(
        &mut self,
        host: &mut H,
        path: &str,
        out: &mut [Option<String>],
    ) {
        self.bind_path(host, path);
        let Some(j) = self.path_index(path) else {
            return;
        };
        let first = self.n_cached_objects;
        if let Some(Some(b)) = self.bindings[j].get_mut(first) {
            b.get_strings(host, out);
        }
    }

    pub fn set_strings<H: AnimationHost<K>>(
        &mut self,
        host: &mut H,
        path: &str,
        values: &[Option<String>],
    ) {
        let Some(j) = self.path_index(path) else {
            return;
        };
        let start = self.n_cached_objects;
        for b in self.bindings[j].iter_mut().skip(start).flatten() {
            b.set_strings(host, values);
        }
    }
}
