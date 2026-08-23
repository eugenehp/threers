//! Data-parallel helpers, backed by rayon when the `parallel` feature is on.
//!
//! Everything here has an identical sequential fallback, so calling code is
//! written once and does not branch on the feature. Results keep input order in
//! both modes — geometry pipelines must stay deterministic, or the same model
//! would produce different vertex ordering depending on a build flag.
//!
//! # When this actually runs in parallel
//!
//! Only on native builds with `--features parallel`. rayon needs OS threads;
//! `wasm32` has none unless the page is cross-origin isolated and the toolchain
//! is set up for `wasm-bindgen-rayon`, so the feature is deliberately a no-op
//! there rather than a build error. For parallel work in the browser, reach for
//! the GPU instead.

/// Whether [`par_map`] and friends will really use multiple threads.
///
/// ```
/// // False on wasm32, or without `--features parallel`.
/// let _ = threers::utils::parallel::is_parallel();
/// ```
pub const fn is_parallel() -> bool {
    cfg!(all(feature = "parallel", not(target_arch = "wasm32")))
}

/// Below this many items, run on one thread regardless.
///
/// Handing work to a thread pool costs a few microseconds, and for a short
/// slice of cheap items that is more than the work itself. Measured on the
/// physics narrow phase, a scene of two dozen shape pairs got *74% slower* with
/// the feature on before this threshold existed — the parallel build was worse
/// than the sequential one at the size most scenes actually are.
///
/// A count is a crude stand-in for the amount of work, since items are not all
/// equally expensive. It is the right kind of crude: it costs nothing to
/// evaluate and it only has to be roughly right, because near the crossover the
/// two paths cost about the same by definition.
pub const MIN_PARALLEL: usize = 128;

/// Map over a slice, in parallel where available. Output order matches input.
///
/// ```
/// use threers::utils::parallel::par_map;
///
/// let squares = par_map(&[1u32, 2, 3, 4], |&n| n * n);
/// assert_eq!(squares, vec![1, 4, 9, 16]);
/// ```
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
pub fn par_map<T, R, F>(items: &[T], f: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync + Send,
{
    use rayon::prelude::*;
    if items.len() < MIN_PARALLEL {
        return items.iter().map(f).collect();
    }
    items.par_iter().map(f).collect()
}

/// Sequential fallback. See the parallel version for the contract.
#[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
pub fn par_map<T, R, F>(items: &[T], f: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync + Send,
{
    items.iter().map(f).collect()
}

/// Map with the element index, in parallel where available.
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
pub fn par_map_indexed<T, R, F>(items: &[T], f: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(usize, &T) -> R + Sync + Send,
{
    use rayon::prelude::*;
    if items.len() < MIN_PARALLEL {
        return items.iter().enumerate().map(|(i, t)| f(i, t)).collect();
    }
    items.par_iter().enumerate().map(|(i, t)| f(i, t)).collect()
}

/// Sequential fallback. See the parallel version for the contract.
#[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
pub fn par_map_indexed<T, R, F>(items: &[T], f: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(usize, &T) -> R + Sync + Send,
{
    items.iter().enumerate().map(|(i, t)| f(i, t)).collect()
}

/// Map over an index range, in parallel where available.
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
pub fn par_map_range<R, F>(len: usize, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync + Send,
{
    use rayon::prelude::*;
    if len < MIN_PARALLEL {
        return (0..len).map(f).collect();
    }
    (0..len).into_par_iter().map(f).collect()
}

/// Sequential fallback. See the parallel version for the contract.
#[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
pub fn par_map_range<R, F>(len: usize, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync + Send,
{
    (0..len).map(f).collect()
}

/// Fill a slice a chunk at a time, in parallel where available. `f` is handed
/// the chunk's index and the chunk itself.
///
/// The difference from [`par_map`] is that nothing is allocated and nothing is
/// copied: the caller owns the buffer and each chunk is written where it will
/// live. For a big grid that matters more than the arithmetic — a
/// `Vec`-per-chunk version has to allocate once per chunk and then copy the
/// whole result into place, which for a few tens of megabytes costs more than
/// the parallelism saves.
///
/// ```
/// use threers::utils::parallel::par_fill_chunks;
///
/// let mut out = vec![0u32; 6];
/// par_fill_chunks(&mut out, 2, |i, chunk| chunk.fill(i as u32));
/// assert_eq!(out, vec![0, 0, 1, 1, 2, 2]);
/// ```
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
pub fn par_fill_chunks<T, F>(out: &mut [T], chunk: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    use rayon::prelude::*;
    if chunk == 0 {
        return;
    }
    // On the total, not the chunk count: a hundred chunks of ten thousand is
    // plenty of work to spread even though a hundred *items* would not be.
    if out.len() < MIN_PARALLEL {
        out.chunks_mut(chunk).enumerate().for_each(|(i, c)| f(i, c));
        return;
    }
    out.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(i, c)| f(i, c));
}

/// Sequential fallback. See the parallel version for the contract.
#[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
pub fn par_fill_chunks<T, F>(out: &mut [T], chunk: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    if chunk == 0 {
        return;
    }
    out.chunks_mut(chunk).enumerate().for_each(|(i, c)| f(i, c));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn par_map_preserves_order() {
        let input: Vec<u32> = (0..1000).collect();
        let out = par_map(&input, |&n| n * 2);
        assert_eq!(out.len(), 1000);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, i as u32 * 2, "order changed at {i}");
        }
    }

    #[test]
    fn par_map_indexed_sees_the_right_index() {
        let input = vec!['a', 'b', 'c'];
        let out = par_map_indexed(&input, |i, c| (i, *c));
        assert_eq!(out, vec![(0, 'a'), (1, 'b'), (2, 'c')]);
    }

    #[test]
    fn par_map_range_covers_the_whole_range_in_order() {
        let out = par_map_range(500, |i| i * i);
        assert_eq!(out.len(), 500);
        assert_eq!(out[0], 0);
        assert_eq!(out[499], 499 * 499);
    }

    #[test]
    fn empty_input_is_fine() {
        let empty: Vec<u32> = Vec::new();
        assert!(par_map(&empty, |&n| n).is_empty());
        assert!(par_map_range(0, |i| i).is_empty());
    }

    #[test]
    fn par_fill_chunks_writes_every_element_where_it_belongs() {
        // Big enough to cross MIN_PARALLEL, and with a ragged last chunk.
        let (len, chunk) = (1003usize, 10usize);
        let mut out = vec![u32::MAX; len];
        par_fill_chunks(&mut out, chunk, |i, c| {
            for (j, slot) in c.iter_mut().enumerate() {
                *slot = (i * chunk + j) as u32;
            }
        });
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, i as u32, "wrong value at {i}");
        }
    }

    #[test]
    fn par_fill_chunks_handles_the_degenerate_cases() {
        let mut empty: Vec<u32> = Vec::new();
        par_fill_chunks(&mut empty, 4, |_, _| unreachable!("nothing to fill"));

        // A zero chunk would be an infinite number of chunks; leave the buffer
        // alone rather than divide by it.
        let mut untouched = vec![7u32; 3];
        par_fill_chunks(&mut untouched, 0, |_, c| c.fill(0));
        assert_eq!(untouched, vec![7, 7, 7]);

        // A chunk larger than the slice is one short chunk.
        let mut short = vec![0u32; 3];
        par_fill_chunks(&mut short, 100, |i, c| {
            assert_eq!(i, 0);
            c.fill(5);
        });
        assert_eq!(short, vec![5, 5, 5]);
    }
}
