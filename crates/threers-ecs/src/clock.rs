//! Wall-clock milliseconds for measuring, the same way on every target.

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn now_ms() -> f64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
}

#[cfg(all(target_arch = "wasm32", feature = "web"))]
pub(crate) fn now_ms() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map_or(0.0, |p| p.now())
}

/// No clock to read: everything measures as instant.
#[cfg(all(target_arch = "wasm32", not(feature = "web")))]
pub(crate) fn now_ms() -> f64 {
    0.0
}

/// Whether parallel systems really run at once here: the `parallel` feature,
/// and threads to run on.
pub(crate) fn threads_available() -> bool {
    cfg!(all(
        feature = "parallel",
        any(not(target_arch = "wasm32"), target_feature = "atomics")
    ))
}

/// Run each job, on the thread pool where there is one, and return what each
/// gave, in order.
#[cfg(all(
    feature = "parallel",
    any(not(target_arch = "wasm32"), target_feature = "atomics")
))]
pub(crate) fn run_together<T: Send>(jobs: Vec<impl FnOnce() -> T + Send>) -> Vec<T> {
    use rayon::prelude::*;
    jobs.into_par_iter().map(|job| job()).collect()
}

#[cfg(not(all(
    feature = "parallel",
    any(not(target_arch = "wasm32"), target_feature = "atomics")
)))]
pub(crate) fn run_together<T: Send>(jobs: Vec<impl FnOnce() -> T + Send>) -> Vec<T> {
    jobs.into_iter().map(|job| job()).collect()
}
