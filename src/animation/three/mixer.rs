//! three.js r165 `AnimationMixer`, with the parts of `AnimationAction` that
//! go through it.
//!
//! The memory manager is three.js's: actions and bindings live in arrays
//! whose first `n` entries are active, moved across the boundary by swaps,
//! so the order actions update and bindings apply in — and with it every
//! floating-point sum — is the order three.js would use for the same calls.
//!
//! Events are dispatched in the middle of an update in three.js, and
//! listeners may play, stop or fade actions before the rest of the frame
//! runs. [`AnimationMixer::begin_update`] / [`AnimationMixer::resume`] give
//! that control back at each event; [`AnimationMixer::update`] collects them
//! for callers that only want to know what happened, and
//! [`AnimationMixer::update_with`] runs a listener at the exact point.

use std::collections::HashMap;
use std::sync::Arc;

use slotmap::{new_key_type, SlotMap};

use super::action::{ActionEvent, AnimationAction};
use super::binding::{parse_track_name, AnimationHost, HostKind, ParseError, PropertyBinding};
use super::clip::AnimationClip;
use super::group::GroupRef;
use super::interpolant::{Interpolant, InterpolantKind, Seek};
use super::property_mixer::{Binding, PropertyMixer};
use super::track::{TrackModifier, TrackValues};
use super::{ADDITIVE_ANIMATION_BLEND_MODE, NORMAL_ANIMATION_BLEND_MODE};

new_key_type! {
    /// An action of one mixer.
    pub struct ActionId;
    /// A property mixer (binding) of one mixer.
    pub struct BindingId;
    /// A fade or warp curve (three.js's control interpolants).
    pub struct ControlId;
}

/// What a mixer (or one action) animates: an object, or an
/// [`AnimationObjectGroup`](super::AnimationObjectGroup).
pub enum MixerRoot<K: HostKind> {
    Node(K::Node),
    Group(GroupRef<K>),
}

impl<K: HostKind> Clone for MixerRoot<K> {
    fn clone(&self) -> Self {
        match self {
            MixerRoot::Node(n) => MixerRoot::Node(n.clone()),
            MixerRoot::Group(g) => MixerRoot::Group(g.clone()),
        }
    }
}

impl<K: HostKind> std::fmt::Debug for MixerRoot<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MixerRoot::Node(n) => f.debug_tuple("Node").field(n).finish(),
            MixerRoot::Group(g) => f
                .debug_tuple("Group")
                .field(&g.lock().map(|g| g.uuid.clone()).unwrap_or_default())
                .finish(),
        }
    }
}

impl<K: HostKind> MixerRoot<K> {
    /// `root.uuid`.
    pub fn uuid<H: AnimationHost<K>>(&self, host: &H) -> String {
        match self {
            MixerRoot::Node(n) => host.node_uuid(n),
            MixerRoot::Group(g) => g.lock().map(|g| g.uuid.clone()).unwrap_or_default(),
        }
    }
}

/// A mixer event: `'finished'` or `'loop'`, from an action.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MixerEvent {
    pub action: ActionId,
    pub kind: ActionEvent,
}

/// Which clip [`AnimationMixer::clip_action`] is for: a clip, or (three.js's
/// string form, once resolved by the caller) just a uuid already known to
/// the mixer.
#[derive(Debug, Clone)]
pub enum ClipRef {
    Clip(Arc<AnimationClip>),
    Uuid(String),
}

impl From<Arc<AnimationClip>> for ClipRef {
    fn from(c: Arc<AnimationClip>) -> Self {
        ClipRef::Clip(c)
    }
}
impl From<AnimationClip> for ClipRef {
    fn from(c: AnimationClip) -> Self {
        ClipRef::Clip(Arc::new(c))
    }
}

#[derive(Debug, Default)]
struct ActionsForClip {
    known_actions: Vec<ActionId>,
    action_by_root: HashMap<String, ActionId>,
}

#[derive(Debug)]
struct ControlInterpolant {
    times: [f32; 2],
    values: [f32; 2],
    interpolant: Interpolant,
    cache_index: usize,
}

impl ControlInterpolant {
    /// `evaluate(t)[0]`, into three.js's shared `Float32Array(1)`.
    fn evaluate(&mut self, t: f64) -> f64 {
        let mut result = [0.0f32];
        self.interpolant
            .evaluate(&self.times, &self.values, t, None, result.as_mut_slice());
        result[0] as f64
    }
}

#[derive(Debug, Clone, Copy)]
struct PendingUpdate {
    delta: f64,
    time: f64,
    direction: f64,
    accu_index: usize,
    n_actions: usize,
    next: usize,
    /// The action whose event was just handed out, with its clip time.
    mid: Option<(usize, f64, f64)>,
}

/// Counts as three.js's `mixer.stats`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MixerStats {
    pub actions_total: usize,
    pub actions_in_use: usize,
    pub bindings_total: usize,
    pub bindings_in_use: usize,
    pub control_interpolants_total: usize,
    pub control_interpolants_in_use: usize,
}

/// three.js `AnimationMixer`.
pub struct AnimationMixer<K: HostKind> {
    root: MixerRoot<K>,
    action_store: SlotMap<ActionId, AnimationAction<K>>,
    /// `_actions`: the first `n_active_actions` are active.
    actions: Vec<ActionId>,
    n_active_actions: usize,
    actions_by_clip: HashMap<String, ActionsForClip>,
    binding_store: SlotMap<BindingId, PropertyMixer<K>>,
    /// `_bindings`: the first `n_active_bindings` are active.
    bindings: Vec<BindingId>,
    n_active_bindings: usize,
    bindings_by_root_and_name: HashMap<String, HashMap<String, BindingId>>,
    control_store: SlotMap<ControlId, ControlInterpolant>,
    control_interpolants: Vec<ControlId>,
    n_active_control_interpolants: usize,
    accu_index: usize,
    /// Global mixer time, seconds.
    pub time: f64,
    /// Scales every update's delta.
    pub time_scale: f64,
    pending: Option<PendingUpdate>,
}

impl<K: HostKind> std::fmt::Debug for AnimationMixer<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnimationMixer")
            .field("root", &self.root)
            .field("time", &self.time)
            .field("time_scale", &self.time_scale)
            .field("stats", &self.stats())
            .finish()
    }
}

impl<K: HostKind> AnimationMixer<K> {
    /// `new AnimationMixer(root)`.
    pub fn new(root: MixerRoot<K>) -> Self {
        Self {
            root,
            action_store: SlotMap::with_key(),
            actions: Vec::new(),
            n_active_actions: 0,
            actions_by_clip: HashMap::new(),
            binding_store: SlotMap::with_key(),
            bindings: Vec::new(),
            n_active_bindings: 0,
            bindings_by_root_and_name: HashMap::new(),
            control_store: SlotMap::with_key(),
            control_interpolants: Vec::new(),
            n_active_control_interpolants: 0,
            accu_index: 0,
            time: 0.0,
            time_scale: 1.0,
            pending: None,
        }
    }

    /// `getRoot()`.
    pub fn root(&self) -> &MixerRoot<K> {
        &self.root
    }

    /// `stats`.
    pub fn stats(&self) -> MixerStats {
        MixerStats {
            actions_total: self.actions.len(),
            actions_in_use: self.n_active_actions,
            bindings_total: self.bindings.len(),
            bindings_in_use: self.n_active_bindings,
            control_interpolants_total: self.control_interpolants.len(),
            control_interpolants_in_use: self.n_active_control_interpolants,
        }
    }

    /// An action's state.
    pub fn action(&self, id: ActionId) -> Option<&AnimationAction<K>> {
        self.action_store.get(id)
    }
    /// An action's state, to change its public fields (`weight`, `loop_mode` …).
    pub fn action_mut(&mut self, id: ActionId) -> Option<&mut AnimationAction<K>> {
        self.action_store.get_mut(id)
    }
    /// The cached actions, active ones first (three.js `_actions`).
    pub fn cached_actions(&self) -> &[ActionId] {
        &self.actions
    }
    /// A binding's mixer.
    pub fn binding(&self, id: BindingId) -> Option<&PropertyMixer<K>> {
        self.binding_store.get(id)
    }
    /// The cached bindings, active ones first (three.js `_bindings`).
    pub fn cached_bindings(&self) -> &[BindingId] {
        &self.bindings
    }
    /// An action's bindings, one per track (`_propertyBindings`).
    pub fn action_bindings(&self, id: ActionId) -> &[Option<BindingId>] {
        self.action_store
            .get(id)
            .map(|a| a.property_bindings.as_slice())
            .unwrap_or(&[])
    }

    fn root_of(&self, a: &AnimationAction<K>) -> MixerRoot<K> {
        a.local_root.clone().unwrap_or_else(|| self.root.clone())
    }

    // ---- clipAction / existingAction ----

    /// `clipAction(clip, optionalRoot, blendMode)`: the cached action for
    /// this clip and root (with the same blend mode), or a new one. `None`
    /// for an unknown uuid; an error for a track name three.js cannot parse.
    pub fn clip_action<H: AnimationHost<K>>(
        &mut self,
        host: &mut H,
        clip: impl Into<ClipRef>,
        optional_root: Option<MixerRoot<K>>,
        blend_mode: Option<u32>,
    ) -> Result<Option<ActionId>, ParseError> {
        let clip = clip.into();
        let root = optional_root.clone().unwrap_or_else(|| self.root.clone());
        let root_uuid = root.uuid(host);
        let mut clip_object = match &clip {
            ClipRef::Clip(c) => Some(c.clone()),
            ClipRef::Uuid(_) => None,
        };
        let clip_uuid = match &clip {
            ClipRef::Clip(c) => c.uuid.clone(),
            ClipRef::Uuid(u) => u.clone(),
        };
        let blend_mode = blend_mode.unwrap_or_else(|| {
            clip_object
                .as_ref()
                .map(|c| c.blend_mode)
                .unwrap_or(NORMAL_ANIMATION_BLEND_MODE)
        });
        let mut prototype = None;
        if let Some(afc) = self.actions_by_clip.get(&clip_uuid) {
            if let Some(&existing) = afc.action_by_root.get(&root_uuid) {
                if self.action_store[existing].blend_mode == blend_mode {
                    return Ok(Some(existing));
                }
            }
            // we know the clip, so we don't have to parse all
            // the bindings again but can just copy
            prototype = afc.known_actions.first().copied();
            if clip_object.is_none() {
                clip_object = prototype.map(|p| self.action_store[p].clip.clone());
            }
        }
        // clip must be known when specified via string
        let Some(clip_object) = clip_object else {
            return Ok(None);
        };
        // three.js throws while binding; check every name first so a bad
        // one leaves no half-built state behind.
        for t in &clip_object.tracks {
            parse_track_name(&t.name)?;
        }
        let action = AnimationAction::new(clip_object, optional_root, blend_mode);
        let id = self.action_store.insert(action);
        self.bind_action(host, id, prototype)?;
        self.add_inactive_action(id, &clip_uuid, &root_uuid);
        Ok(Some(id))
    }

    /// `existingAction(clip, optionalRoot)`.
    pub fn existing_action<H: AnimationHost<K>>(
        &self,
        host: &H,
        clip_uuid: &str,
        optional_root: Option<&MixerRoot<K>>,
    ) -> Option<ActionId> {
        let root_uuid = optional_root.unwrap_or(&self.root).uuid(host);
        self.actions_by_clip
            .get(clip_uuid)
            .and_then(|afc| afc.action_by_root.get(&root_uuid).copied())
    }

    fn set_action_binding(&mut self, action: ActionId, i: usize, binding: BindingId) {
        let a = &mut self.action_store[action];
        let old = a.property_bindings[i].replace(binding);
        if old != Some(binding) {
            if let Some(b) = self.binding_store.get_mut(binding) {
                b.holders += 1;
            }
            if let Some(old) = old {
                self.release_holder::<NoHost>(old, None);
            }
        }
    }

    fn release_holder<H: AnimationHost<K>>(&mut self, id: BindingId, host: Option<&mut H>) {
        let free = match self.binding_store.get_mut(id) {
            Some(b) => {
                b.holders = b.holders.saturating_sub(1);
                b.holders == 0 && b.cache_index.is_none()
            }
            None => false,
        };
        if free {
            if let Some(mut b) = self.binding_store.remove(id) {
                if let (Some(host), Binding::Single(pb)) = (host, &mut b.binding) {
                    pb.release(host);
                }
            }
        }
    }

    /// `_bindAction(action, prototypeAction)`.
    fn bind_action<H: AnimationHost<K>>(
        &mut self,
        host: &mut H,
        action: ActionId,
        prototype: Option<ActionId>,
    ) -> Result<(), ParseError> {
        let a = &self.action_store[action];
        let root = self.root_of(a);
        let root_uuid = root.uuid(host);
        let clip = a.clip.clone();
        self.bindings_by_root_and_name
            .entry(root_uuid.clone())
            .or_default();
        for (i, track) in clip.tracks.iter().enumerate() {
            let track_name = &track.name;
            let known = self
                .bindings_by_root_and_name
                .get(&root_uuid)
                .and_then(|m| m.get(track_name))
                .copied();
            if let Some(binding) = known {
                self.binding_store[binding].reference_count += 1;
                self.set_action_binding(action, i, binding);
            } else {
                if let Some(binding) = self.action_store[action].property_bindings[i] {
                    // existing binding, make sure the cache knows
                    if self.binding_store[binding].cache_index.is_none() {
                        self.binding_store[binding].reference_count += 1;
                        self.add_inactive_binding(binding, &root_uuid, track_name);
                    }
                    continue;
                }
                let path = prototype.and_then(|p| {
                    let pb = (*self.action_store[p].property_bindings.get(i)?)?;
                    match &self.binding_store.get(pb)?.binding {
                        Binding::Single(b) => Some(b.parsed_path.clone()),
                        Binding::Composite { .. } => None,
                    }
                });
                let property_binding = match &root {
                    MixerRoot::Node(n) => Binding::Single(PropertyBinding::new(
                        host,
                        n.clone(),
                        track_name,
                        path,
                        track.value_type,
                        track.value_size(),
                    )?),
                    MixerRoot::Group(g) => {
                        let parsed = match path {
                            Some(p) => p,
                            None => parse_track_name(track_name)?,
                        };
                        if let Ok(mut gl) = g.lock() {
                            gl.subscribe(
                                host,
                                track_name,
                                &parsed,
                                track.value_type,
                                track.value_size(),
                            );
                        }
                        Binding::Composite {
                            group: g.clone(),
                            path: track_name.clone(),
                        }
                    }
                };
                let mut mixer =
                    PropertyMixer::new(property_binding, track.value_type, track.value_size());
                mixer.reference_count += 1;
                let binding = self.binding_store.insert(mixer);
                self.add_inactive_binding(binding, &root_uuid, track_name);
                self.set_action_binding(action, i, binding);
            }
        }
        Ok(())
    }

    // ---- activation ----

    fn is_active_action(&self, id: ActionId) -> bool {
        match self.action_store.get(id).and_then(|a| a.cache_index) {
            Some(i) => i < self.n_active_actions,
            None => false,
        }
    }

    /// `_activateAction(action)`.
    fn activate_action<H: AnimationHost<K>>(&mut self, host: &mut H, id: ActionId) {
        if self.is_active_action(id) {
            return;
        }
        if self.action_store[id].cache_index.is_none() {
            // this action has been forgotten by the cache, but the user
            // appears to be still using it -> rebind
            let a = &self.action_store[id];
            let root_uuid = self.root_of(a).uuid(host);
            let clip_uuid = a.clip.uuid.clone();
            let prototype = self
                .actions_by_clip
                .get(&clip_uuid)
                .and_then(|afc| afc.known_actions.first().copied());
            // Names were checked when the action was made.
            let _ = self.bind_action(host, id, prototype);
            self.add_inactive_action(id, &clip_uuid, &root_uuid);
        }
        let bindings: Vec<BindingId> = self.action_store[id]
            .property_bindings
            .iter()
            .flatten()
            .copied()
            .collect();
        for b in bindings {
            let first = {
                let m = &mut self.binding_store[b];
                let was = m.use_count;
                m.use_count += 1;
                was == 0
            };
            if first {
                self.lend_binding(b);
                self.binding_store[b].save_original_state(host);
            }
        }
        self.lend_action(id);
    }

    /// `_deactivateAction(action)`.
    fn deactivate_action<H: AnimationHost<K>>(&mut self, host: &mut H, id: ActionId) {
        if !self.is_active_action(id) {
            return;
        }
        let bindings: Vec<BindingId> = self.action_store[id]
            .property_bindings
            .iter()
            .flatten()
            .copied()
            .collect();
        for b in bindings {
            let last = {
                let m = &mut self.binding_store[b];
                m.use_count = m.use_count.saturating_sub(1);
                m.use_count == 0
            };
            if last {
                self.binding_store[b].restore_original_state(host);
                self.take_back_binding(b);
            }
        }
        self.take_back_action(id);
    }

    // ---- memory manager ----

    fn add_inactive_action(&mut self, id: ActionId, clip_uuid: &str, root_uuid: &str) {
        let afc = self
            .actions_by_clip
            .entry(clip_uuid.to_string())
            .or_default();
        self.action_store[id].by_clip_cache_index = Some(afc.known_actions.len());
        afc.known_actions.push(id);
        self.action_store[id].cache_index = Some(self.actions.len());
        self.actions.push(id);
        afc.action_by_root.insert(root_uuid.to_string(), id);
    }

    fn remove_inactive_action<H: AnimationHost<K>>(&mut self, host: &mut H, id: ActionId) {
        let Some(cache_index) = self.action_store[id].cache_index else {
            return;
        };
        let last = *self
            .actions
            .last()
            .expect("cached action list is not empty");
        self.action_store[last].cache_index = Some(cache_index);
        self.actions[cache_index] = last;
        self.actions.pop();
        self.action_store[id].cache_index = None;

        let a = &self.action_store[id];
        let clip_uuid = a.clip.uuid.clone();
        let root_uuid = self.root_of(a).uuid(host);
        if let Some(afc) = self.actions_by_clip.get_mut(&clip_uuid) {
            if let Some(by_clip) = self.action_store[id].by_clip_cache_index {
                if let Some(&last_known) = afc.known_actions.last() {
                    self.action_store[last_known].by_clip_cache_index = Some(by_clip);
                    if by_clip < afc.known_actions.len() {
                        afc.known_actions[by_clip] = last_known;
                    }
                    afc.known_actions.pop();
                }
            }
            afc.action_by_root.remove(&root_uuid);
            if afc.known_actions.is_empty() {
                self.actions_by_clip.remove(&clip_uuid);
            }
        }
        self.action_store[id].by_clip_cache_index = None;
        self.remove_inactive_bindings_for_action(id);
    }

    fn remove_inactive_bindings_for_action(&mut self, id: ActionId) {
        let bindings: Vec<BindingId> = self.action_store[id]
            .property_bindings
            .iter()
            .flatten()
            .copied()
            .collect();
        for b in bindings {
            let gone = {
                let m = &mut self.binding_store[b];
                m.reference_count = m.reference_count.saturating_sub(1);
                m.reference_count == 0
            };
            if gone {
                self.remove_inactive_binding(b);
            }
        }
    }

    fn lend_action(&mut self, id: ActionId) {
        let prev = self.action_store[id].cache_index.expect("cached action");
        let last_active = self.n_active_actions;
        self.n_active_actions += 1;
        let first_inactive = self.actions[last_active];
        self.action_store[id].cache_index = Some(last_active);
        self.actions[last_active] = id;
        self.action_store[first_inactive].cache_index = Some(prev);
        self.actions[prev] = first_inactive;
    }

    fn take_back_action(&mut self, id: ActionId) {
        let prev = self.action_store[id].cache_index.expect("cached action");
        self.n_active_actions -= 1;
        let first_inactive = self.n_active_actions;
        let last_active = self.actions[first_inactive];
        self.action_store[id].cache_index = Some(first_inactive);
        self.actions[first_inactive] = id;
        self.action_store[last_active].cache_index = Some(prev);
        self.actions[prev] = last_active;
    }

    fn add_inactive_binding(&mut self, id: BindingId, root_uuid: &str, track_name: &str) {
        self.bindings_by_root_and_name
            .entry(root_uuid.to_string())
            .or_default()
            .insert(track_name.to_string(), id);
        let m = &mut self.binding_store[id];
        m.cache_index = Some(self.bindings.len());
        m.root_uuid = root_uuid.to_string();
        m.track_name = track_name.to_string();
        self.bindings.push(id);
    }

    fn remove_inactive_binding(&mut self, id: BindingId) {
        let Some(cache_index) = self.binding_store[id].cache_index else {
            return;
        };
        let root_uuid = self.binding_store[id].root_uuid.clone();
        let track_name = self.binding_store[id].track_name.clone();
        let last = *self
            .bindings
            .last()
            .expect("cached binding list is not empty");
        self.binding_store[last].cache_index = Some(cache_index);
        self.bindings[cache_index] = last;
        self.bindings.pop();
        self.binding_store[id].cache_index = None;
        if let Some(by_name) = self.bindings_by_root_and_name.get_mut(&root_uuid) {
            by_name.remove(&track_name);
            if by_name.is_empty() {
                self.bindings_by_root_and_name.remove(&root_uuid);
            }
        }
        if self.binding_store[id].holders == 0 {
            self.binding_store.remove(id);
        }
    }

    fn lend_binding(&mut self, id: BindingId) {
        let prev = self.binding_store[id].cache_index.expect("cached binding");
        let last_active = self.n_active_bindings;
        self.n_active_bindings += 1;
        let first_inactive = self.bindings[last_active];
        self.binding_store[id].cache_index = Some(last_active);
        self.bindings[last_active] = id;
        self.binding_store[first_inactive].cache_index = Some(prev);
        self.bindings[prev] = first_inactive;
    }

    fn take_back_binding(&mut self, id: BindingId) {
        let prev = self.binding_store[id].cache_index.expect("cached binding");
        self.n_active_bindings -= 1;
        let first_inactive = self.n_active_bindings;
        let last_active = self.bindings[first_inactive];
        self.binding_store[id].cache_index = Some(first_inactive);
        self.bindings[first_inactive] = id;
        self.binding_store[last_active].cache_index = Some(prev);
        self.bindings[prev] = last_active;
    }

    fn lend_control_interpolant(&mut self) -> ControlId {
        let last_active = self.n_active_control_interpolants;
        self.n_active_control_interpolants += 1;
        if let Some(&id) = self.control_interpolants.get(last_active) {
            return id;
        }
        let id = self.control_store.insert(ControlInterpolant {
            times: [0.0; 2],
            values: [0.0; 2],
            interpolant: Interpolant::new(InterpolantKind::Linear, 1),
            cache_index: last_active,
        });
        self.control_interpolants.push(id);
        id
    }

    fn take_back_control_interpolant(&mut self, id: ControlId) {
        let prev = self.control_store[id].cache_index;
        self.n_active_control_interpolants -= 1;
        let first_inactive = self.n_active_control_interpolants;
        let last_active = self.control_interpolants[first_inactive];
        self.control_store[id].cache_index = first_inactive;
        self.control_interpolants[first_inactive] = id;
        self.control_store[last_active].cache_index = prev;
        self.control_interpolants[prev] = last_active;
    }

    // ---- AnimationAction API ----

    /// `action.play()`.
    pub fn play<H: AnimationHost<K>>(&mut self, host: &mut H, id: ActionId) {
        if self.action_store.contains_key(id) {
            self.activate_action(host, id);
        }
    }

    /// `action.stop()`.
    pub fn stop<H: AnimationHost<K>>(&mut self, host: &mut H, id: ActionId) {
        if self.action_store.contains_key(id) {
            self.deactivate_action(host, id);
            self.reset(id);
        }
    }

    /// `action.reset()`.
    pub fn reset(&mut self, id: ActionId) {
        let Some(a) = self.action_store.get_mut(id) else {
            return;
        };
        a.paused = false;
        a.enabled = true;
        a.time = 0.0; // restart clip
        a.loop_count = -1.0; // forget previous loops
        a.start_time = None; // forget scheduling
        self.stop_fading(id);
        self.stop_warping(id);
    }

    /// `action.isRunning()`.
    pub fn is_running(&self, id: ActionId) -> bool {
        let Some(a) = self.action_store.get(id) else {
            return false;
        };
        a.enabled
            && !a.paused
            && a.time_scale != 0.0
            && a.start_time.is_none()
            && self.is_active_action(id)
    }

    /// `action.isScheduled()`.
    pub fn is_scheduled(&self, id: ActionId) -> bool {
        self.is_active_action(id)
    }

    /// `action.setEffectiveWeight(weight)`.
    pub fn set_effective_weight(&mut self, id: ActionId, weight: f64) {
        let Some(a) = self.action_store.get_mut(id) else {
            return;
        };
        a.weight = weight;
        // note: same logic as when updated at runtime
        a.effective_weight = if a.enabled { weight } else { 0.0 };
        self.stop_fading(id);
    }

    /// `action.fadeIn(duration)`.
    pub fn fade_in(&mut self, id: ActionId, duration: f64) {
        self.schedule_fading(id, duration, 0.0, 1.0);
    }

    /// `action.fadeOut(duration)`.
    pub fn fade_out(&mut self, id: ActionId, duration: f64) {
        self.schedule_fading(id, duration, 1.0, 0.0);
    }

    /// `action.crossFadeFrom(fadeOutAction, duration, warp)`.
    pub fn cross_fade_from(
        &mut self,
        id: ActionId,
        fade_out_action: ActionId,
        duration: f64,
        warp: bool,
    ) {
        self.fade_out(fade_out_action, duration);
        self.fade_in(id, duration);
        if warp {
            let (Some(a), Some(o)) = (
                self.action_store.get(id),
                self.action_store.get(fade_out_action),
            ) else {
                return;
            };
            let fade_in_duration = a.clip.duration;
            let fade_out_duration = o.clip.duration;
            let start_end_ratio = fade_out_duration / fade_in_duration;
            let end_start_ratio = fade_in_duration / fade_out_duration;
            self.warp(fade_out_action, 1.0, start_end_ratio, duration);
            self.warp(id, end_start_ratio, 1.0, duration);
        }
    }

    /// `action.crossFadeTo(fadeInAction, duration, warp)`.
    pub fn cross_fade_to(
        &mut self,
        id: ActionId,
        fade_in_action: ActionId,
        duration: f64,
        warp: bool,
    ) {
        self.cross_fade_from(fade_in_action, id, duration, warp);
    }

    /// `action.stopFading()`.
    pub fn stop_fading(&mut self, id: ActionId) {
        let Some(a) = self.action_store.get_mut(id) else {
            return;
        };
        if let Some(c) = a.weight_interpolant.take() {
            self.take_back_control_interpolant(c);
        }
    }

    /// `action.setEffectiveTimeScale(timeScale)`.
    pub fn set_effective_time_scale(&mut self, id: ActionId, time_scale: f64) {
        let Some(a) = self.action_store.get_mut(id) else {
            return;
        };
        a.time_scale = time_scale;
        a.effective_time_scale = if a.paused { 0.0 } else { time_scale };
        self.stop_warping(id);
    }

    /// `action.setDuration(duration)`.
    pub fn set_duration(&mut self, id: ActionId, duration: f64) {
        let Some(a) = self.action_store.get_mut(id) else {
            return;
        };
        a.time_scale = a.clip.duration / duration;
        self.stop_warping(id);
    }

    /// `action.syncWith(other)`.
    pub fn sync_with(&mut self, id: ActionId, other: ActionId) {
        let Some(o) = self.action_store.get(other) else {
            return;
        };
        let (time, time_scale) = (o.time, o.time_scale);
        let Some(a) = self.action_store.get_mut(id) else {
            return;
        };
        a.time = time;
        a.time_scale = time_scale;
        self.stop_warping(id);
    }

    /// `action.halt(duration)`.
    pub fn halt(&mut self, id: ActionId, duration: f64) {
        let Some(a) = self.action_store.get(id) else {
            return;
        };
        let ets = a.effective_time_scale;
        self.warp(id, ets, 0.0, duration);
    }

    /// `action.warp(startTimeScale, endTimeScale, duration)`.
    pub fn warp(
        &mut self,
        id: ActionId,
        start_time_scale: f64,
        end_time_scale: f64,
        duration: f64,
    ) {
        if !self.action_store.contains_key(id) {
            return;
        }
        let now = self.time;
        let c = match self.action_store[id].time_scale_interpolant {
            Some(c) => c,
            None => {
                let c = self.lend_control_interpolant();
                self.action_store[id].time_scale_interpolant = Some(c);
                c
            }
        };
        let time_scale = self.action_store[id].time_scale;
        let ci = &mut self.control_store[c];
        ci.times[0] = now as f32;
        ci.times[1] = (now + duration) as f32;
        ci.values[0] = (start_time_scale / time_scale) as f32;
        ci.values[1] = (end_time_scale / time_scale) as f32;
    }

    /// `action.stopWarping()`.
    pub fn stop_warping(&mut self, id: ActionId) {
        let Some(a) = self.action_store.get_mut(id) else {
            return;
        };
        if let Some(c) = a.time_scale_interpolant.take() {
            self.take_back_control_interpolant(c);
        }
    }

    /// `action.getRoot()`.
    pub fn action_root(&self, id: ActionId) -> Option<MixerRoot<K>> {
        self.action_store.get(id).map(|a| self.root_of(a))
    }

    fn schedule_fading(&mut self, id: ActionId, duration: f64, weight_now: f64, weight_then: f64) {
        if !self.action_store.contains_key(id) {
            return;
        }
        let now = self.time;
        let c = match self.action_store[id].weight_interpolant {
            Some(c) => c,
            None => {
                let c = self.lend_control_interpolant();
                self.action_store[id].weight_interpolant = Some(c);
                c
            }
        };
        let ci = &mut self.control_store[c];
        ci.times[0] = now as f32;
        ci.values[0] = weight_now as f32;
        ci.times[1] = (now + duration) as f32;
        ci.values[1] = weight_then as f32;
    }

    /// `_updateWeight(time)`.
    fn update_weight(&mut self, id: ActionId, time: f64) -> f64 {
        let mut weight = 0.0;
        let a = &self.action_store[id];
        if a.enabled {
            weight = a.weight;
            if let Some(c) = a.weight_interpolant {
                let v = self.control_store[c].evaluate(time);
                weight *= v;
                if time > self.control_store[c].times[1] as f64 {
                    self.stop_fading(id);
                    if v == 0.0 {
                        // faded out, disable
                        self.action_store[id].enabled = false;
                    }
                }
            }
        }
        self.action_store[id].effective_weight = weight;
        weight
    }

    /// `_updateTimeScale(time)`.
    fn update_time_scale(&mut self, id: ActionId, time: f64) -> f64 {
        let mut time_scale = 0.0;
        let a = &self.action_store[id];
        if !a.paused {
            time_scale = a.time_scale;
            if let Some(c) = a.time_scale_interpolant {
                let v = self.control_store[c].evaluate(time);
                time_scale *= v;
                if time > self.control_store[c].times[1] as f64 {
                    self.stop_warping(id);
                    if time_scale == 0.0 {
                        // motion has halted, pause
                        self.action_store[id].paused = true;
                    } else {
                        // warp done - apply final time scale
                        self.action_store[id].time_scale = time_scale;
                    }
                }
            }
        }
        self.action_store[id].effective_time_scale = time_scale;
        time_scale
    }

    /// The first half of `action._update`: up to and including
    /// `_updateTime`. `None`: nothing more to do for this action.
    fn update_action_time(
        &mut self,
        id: ActionId,
        time: f64,
        delta: f64,
        direction: f64,
    ) -> Option<(f64, Option<ActionEvent>)> {
        if !self.action_store[id].enabled {
            // call ._updateWeight() to update ._effectiveWeight
            self.update_weight(id, time);
            return None;
        }
        let mut delta_time = delta;
        if let Some(start_time) = self.action_store[id].start_time {
            // check for scheduled start of action
            let time_running = (time - start_time) * direction;
            if time_running < 0.0 || direction == 0.0 {
                delta_time = 0.0;
            } else {
                self.action_store[id].start_time = None; // unschedule
                delta_time = direction * time_running;
            }
        }
        // apply time scale and advance time
        delta_time *= self.update_time_scale(id, time);
        Some(self.action_store[id].update_time(delta_time))
    }

    /// The second half of `action._update`: weight, sample, accumulate.
    fn update_action_weight(&mut self, id: ActionId, time: f64, clip_time: f64, accu_index: usize) {
        let weight = self.update_weight(id, time);
        if weight <= 0.0 {
            return;
        }
        let a = &mut self.action_store[id];
        let clip = a.clip.clone();
        let additive = a.blend_mode == ADDITIVE_ANIMATION_BLEND_MODE;
        let settings = a.interpolant_settings;
        for (j, track) in clip.tracks.iter().enumerate() {
            let Some(bid) = a.property_bindings.get(j).copied().flatten() else {
                continue;
            };
            let Some(m) = self.binding_store.get_mut(bid) else {
                continue;
            };
            let interp = &mut a.interpolants[j];
            let clip_time = if track.modifiers.is_empty() {
                clip_time
            } else {
                let duration = track.times.last().map(|&t| t as f64).unwrap_or(0.0);
                TrackModifier::remap_time(&track.modifiers, clip_time, duration)
            };
            match &track.values {
                TrackValues::Numeric(values) => {
                    interp.evaluate(
                        &track.times,
                        values,
                        clip_time,
                        Some(&settings),
                        m.buffer.as_mut_slice(),
                    );
                    if !track.modifiers.is_empty() {
                        let n = interp.value_size.min(m.buffer.len());
                        TrackModifier::apply_value(&track.modifiers, &mut m.buffer[..n], clip_time);
                    }
                }
                TrackValues::Bool(values) => {
                    let index = discrete_index(interp, &track.times, clip_time);
                    let size = interp.value_size;
                    for k in 0..size {
                        let v = values
                            .get(index * size + k)
                            .map(|&b| b as u8 as f64)
                            .unwrap_or(f64::NAN);
                        if let Some(s) = m.buffer.get_mut(k) {
                            *s = v;
                        }
                    }
                }
                TrackValues::String(values) => {
                    let index = discrete_index(interp, &track.times, clip_time);
                    let size = interp.value_size;
                    for k in 0..size {
                        if let Some(s) = m.string_buffer.get_mut(k) {
                            *s = values.get(index * size + k).cloned();
                        }
                    }
                }
            }
            if additive {
                m.accumulate_additive(weight);
            } else {
                m.accumulate(accu_index, weight);
            }
        }
    }

    // ---- update ----

    /// Start `update(deltaTime)`; drive it with [`resume`](Self::resume).
    pub fn begin_update(&mut self, delta_time: f64) {
        let delta = delta_time * self.time_scale;
        self.time += delta;
        let direction = js_sign(delta);
        self.accu_index ^= 1;
        self.pending = Some(PendingUpdate {
            delta,
            time: self.time,
            direction,
            accu_index: self.accu_index,
            n_actions: self.n_active_actions,
            next: 0,
            mid: None,
        });
    }

    /// Whether an update is under way (an event was handed out).
    pub fn is_updating(&self) -> bool {
        self.pending.is_some()
    }

    /// Run the update until the next event (returned, with the mixer state
    /// as three.js has it when it dispatches) or the end (`None`, after the
    /// bindings are applied).
    pub fn resume<H: AnimationHost<K>>(&mut self, host: &mut H) -> Option<MixerEvent> {
        let mut p = self.pending?;
        if let Some((i, time, clip_time)) = p.mid.take() {
            if let Some(&id) = self.actions.get(i) {
                self.update_action_weight(id, time, clip_time, p.accu_index);
            }
        }
        while p.next < p.n_actions {
            let i = p.next;
            p.next += 1;
            let Some(&id) = self.actions.get(i) else {
                continue;
            };
            let Some((clip_time, event)) =
                self.update_action_time(id, p.time, p.delta, p.direction)
            else {
                continue;
            };
            if let Some(kind) = event {
                p.mid = Some((i, p.time, clip_time));
                self.pending = Some(p);
                return Some(MixerEvent { action: id, kind });
            }
            self.update_action_weight(id, p.time, clip_time, p.accu_index);
        }
        // update scene graph
        let n_bindings = self.n_active_bindings;
        for i in 0..n_bindings {
            if let Some(&b) = self.bindings.get(i) {
                if let Some(m) = self.binding_store.get_mut(b) {
                    m.apply(host, p.accu_index);
                }
            }
        }
        self.pending = None;
        None
    }

    /// `update(deltaTime)`, returning the events it dispatched in order
    /// (listeners that change actions need [`update_with`](Self::update_with)).
    pub fn update<H: AnimationHost<K>>(
        &mut self,
        host: &mut H,
        delta_time: f64,
    ) -> Vec<MixerEvent> {
        self.begin_update(delta_time);
        let mut events = Vec::new();
        while let Some(e) = self.resume(host) {
            events.push(e);
        }
        events
    }

    /// `update(deltaTime)` with a listener run at each event, free to change
    /// the mixer as a three.js listener may.
    pub fn update_with<H: AnimationHost<K>>(
        &mut self,
        host: &mut H,
        delta_time: f64,
        mut on_event: impl FnMut(&mut Self, &mut H, MixerEvent),
    ) {
        self.begin_update(delta_time);
        while let Some(e) = self.resume(host) {
            on_event(self, host, e);
        }
    }

    /// Start `setTime(timeInSeconds)`: zero the mixer's and every cached
    /// action's time, then [`begin_update`](Self::begin_update).
    pub fn begin_set_time(&mut self, time_in_seconds: f64) {
        self.time = 0.0;
        for &id in &self.actions {
            self.action_store[id].time = 0.0;
        }
        self.begin_update(time_in_seconds);
    }

    /// `setTime(timeInSeconds)`.
    pub fn set_time<H: AnimationHost<K>>(
        &mut self,
        host: &mut H,
        time_in_seconds: f64,
    ) -> Vec<MixerEvent> {
        self.begin_set_time(time_in_seconds);
        let mut events = Vec::new();
        while let Some(e) = self.resume(host) {
            events.push(e);
        }
        events
    }

    /// `stopAllAction()`.
    pub fn stop_all_action<H: AnimationHost<K>>(&mut self, host: &mut H) {
        let n = self.n_active_actions;
        for i in (0..n).rev() {
            if let Some(&id) = self.actions.get(i) {
                self.stop(host, id);
            }
        }
    }

    /// `uncacheClip(clip)`: forget every action of the clip.
    pub fn uncache_clip<H: AnimationHost<K>>(&mut self, host: &mut H, clip_uuid: &str) {
        let Some(known) = self
            .actions_by_clip
            .get(clip_uuid)
            .map(|a| a.known_actions.clone())
        else {
            return;
        };
        for id in known {
            self.deactivate_action(host, id);
            if let Some(cache_index) = self.action_store[id].cache_index {
                let last = *self
                    .actions
                    .last()
                    .expect("cached action list is not empty");
                self.action_store[id].cache_index = None;
                self.action_store[id].by_clip_cache_index = None;
                if last != id {
                    self.action_store[last].cache_index = Some(cache_index);
                    self.actions[cache_index] = last;
                }
                self.actions.pop();
            }
            self.remove_inactive_bindings_for_action(id);
        }
        self.actions_by_clip.remove(clip_uuid);
    }

    /// `uncacheRoot(root)`: forget every action and binding of the root.
    pub fn uncache_root<H: AnimationHost<K>>(&mut self, host: &mut H, root_uuid: &str) {
        let clip_uuids: Vec<String> = self.actions_by_clip.keys().cloned().collect();
        for clip_uuid in clip_uuids {
            let action = self
                .actions_by_clip
                .get(&clip_uuid)
                .and_then(|afc| afc.action_by_root.get(root_uuid).copied());
            if let Some(id) = action {
                self.deactivate_action(host, id);
                self.remove_inactive_action(host, id);
            }
        }
        if let Some(by_name) = self.bindings_by_root_and_name.get(root_uuid) {
            let ids: Vec<BindingId> = by_name.values().copied().collect();
            for b in ids {
                if let Some(m) = self.binding_store.get_mut(b) {
                    m.restore_original_state(host);
                }
                self.remove_inactive_binding(b);
            }
        }
    }

    /// `uncacheAction(clip, optionalRoot)`.
    pub fn uncache_action<H: AnimationHost<K>>(
        &mut self,
        host: &mut H,
        clip_uuid: &str,
        optional_root: Option<&MixerRoot<K>>,
    ) {
        if let Some(id) = self.existing_action(host, clip_uuid, optional_root) {
            self.deactivate_action(host, id);
            self.remove_inactive_action(host, id);
        }
    }

    /// Drop an action the caller no longer holds. Cached actions stay (the
    /// mixer still uses them); returns whether it was dropped.
    pub fn forget_action<H: AnimationHost<K>>(&mut self, host: &mut H, id: ActionId) -> bool {
        match self.action_store.get(id) {
            Some(a) if a.cache_index.is_none() => {}
            _ => return false,
        }
        let Some(a) = self.action_store.remove(id) else {
            return false;
        };
        for c in [a.weight_interpolant, a.time_scale_interpolant]
            .into_iter()
            .flatten()
        {
            // an uncached action keeps its curves until reset; give them back
            if self.control_store.contains_key(c)
                && self.control_store[c].cache_index < self.n_active_control_interpolants
            {
                self.take_back_control_interpolant(c);
            }
        }
        for b in a.property_bindings.into_iter().flatten() {
            self.release_holder(b, Some(&mut *host));
        }
        true
    }
}

/// The sample index a discrete interpolant copies at `t`.
fn discrete_index(interp: &mut Interpolant, times: &[f32], t: f64) -> usize {
    match interp.seek(times, t) {
        Seek::Copy(i) => i,
        Seek::Interpolate { i1, .. } => i1.saturating_sub(1),
    }
}

/// JS `Math.sign`.
fn js_sign(v: f64) -> f64 {
    if v.is_nan() {
        f64::NAN
    } else if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        v
    }
}

/// A host type for calls that need none.
struct NoHost;
impl<K: HostKind> AnimationHost<K> for NoHost {
    fn node_uuid(&self, _: &K::Node) -> String {
        String::new()
    }
    fn find_node(&mut self, _: &K::Node, _: Option<&str>) -> Option<K::Node> {
        None
    }
    fn bind(
        &mut self,
        _: &K::Node,
        _: &super::ParsedPath,
        _: &str,
        _: super::ValueType,
        _: usize,
    ) -> Option<K::Target> {
        None
    }
    fn get_value(&mut self, _: &K::Target, _: &mut [f64]) {}
    fn set_value(&mut self, _: &K::Target, _: &[f64]) {}
}
