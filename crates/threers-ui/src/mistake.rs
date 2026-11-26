//! Mistakes in how the UI is used (CSS it doesn't support, a node id
//! that's gone): a panic naming it in debug builds, so it's found and
//! fixed; in a release build, reported and skipped (a property left as it
//! was, a gone node's change dropped), so an app on someone's phone keeps
//! running. Where reports go: [`set_hook`], else stderr; each once.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Mutex;

type Hook = Box<dyn Fn(&str) + Send + Sync>;

static HOOK: Mutex<Option<Hook>> = Mutex::new(None);
static SEEN: Mutex<Option<HashSet<String>>> = Mutex::new(None);
/// 0: as the build is (debug panics); 1: always panic; 2: never.
static MODE: AtomicU8 = AtomicU8::new(0);
/// How many distinct mistakes are reported (the rest only counted).
const MOST: usize = 256;

/// Where mistakes are reported (an app's log), in place of stderr.
pub fn set_hook(f: impl Fn(&str) + Send + Sync + 'static) {
    *HOOK.lock().unwrap_or_else(|e| e.into_inner()) = Some(Box::new(f));
}

/// Whether a mistake panics: `Some(true)` always, `Some(false)` never (as
/// a release build), `None` as the build is (debug builds panic).
pub fn set_panics(on: Option<bool>) {
    MODE.store(
        match on {
            None => 0,
            Some(true) => 1,
            Some(false) => 2,
        },
        Ordering::Relaxed,
    );
}

/// Whether a mistake panics now.
pub fn panics() -> bool {
    match MODE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => cfg!(debug_assertions),
    }
}

/// A mistake: a panic where they panic, else reported once (with where it
/// was made) and the caller carries on.
#[track_caller]
pub(crate) fn report(what: impl std::fmt::Display) {
    let at = std::panic::Location::caller();
    if panics() {
        panic!("{what}");
    }
    let line = format!("threers-ui: {what} (at {}:{}); skipped", at.file(), at.line());
    {
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let seen = seen.get_or_insert_with(HashSet::new);
        if seen.len() >= MOST || !seen.insert(line.clone()) {
            return;
        }
    }
    match &*HOOK.lock().unwrap_or_else(|e| e.into_inner()) {
        Some(f) => f(&line),
        None => eprintln!("{line}"),
    }
}
