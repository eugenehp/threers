//! The radix sort against a plain stable comparison sort of the same keys,
//! and its edge cases. `timing` (ignored) measures it against the sorter it
//! replaced, in thread CPU time.

use threers_splat_sort::DepthSort;

fn rng(seed: u32) -> impl FnMut() -> f32 {
    let mut x = seed.wrapping_mul(2_654_435_761) | 1;
    move || {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        (x >> 8) as f32 / (1u32 << 24) as f32
    }
}

fn cloud(n: usize, seed: u32) -> Vec<[f32; 3]> {
    let mut r = rng(seed);
    (0..n)
        .map(|_| [r() * 40.0 - 20.0, r() * 8.0 - 4.0, r() * 100.0 - 90.0])
        .collect()
}

/// The order a stable sort by depth gives, with the keys quantised as the
/// radix sort quantises them (so ties fall the same way).
fn reference(centres: &[[f32; 3]], axis: [f32; 3]) -> Vec<u32> {
    let finite: Vec<_> = centres.iter().filter(|p| p.iter().all(|v| v.is_finite())).collect();
    let (mut lo, mut hi) = (0.0f32, 0.0f32);
    for k in 0..3 {
        let min = finite.iter().map(|p| p[k]).fold(f32::INFINITY, f32::min);
        let max = finite.iter().map(|p| p[k]).fold(f32::NEG_INFINITY, f32::max);
        let (a, b) = (axis[k] * min, axis[k] * max);
        lo += a.min(b);
        hi += a.max(b);
    }
    let max_key = ((1u32 << 24) - 1) as f32;
    let scale = if hi > lo { max_key / (hi - lo) } else { 0.0 };
    let key = |p: &[f32; 3]| {
        let d = ((axis[0] * p[0] + axis[1] * p[1] + axis[2] * p[2]) - lo) * scale;
        d.max(0.0).min(max_key) as u32
    };
    let mut order: Vec<u32> = (0..centres.len() as u32).collect();
    order.sort_by_key(|&i| key(&centres[i as usize]));
    order
}

#[test]
fn it_matches_a_stable_comparison_sort() {
    for (n, seed) in [(1, 1), (15, 2), (16, 3), (17, 4), (1000, 5), (100_003, 6)] {
        let centres = cloud(n, seed);
        let mut s = DepthSort::new(&centres);
        let mut r = rng(seed + 100);
        for _ in 0..4 {
            let a = [r() - 0.5, r() - 0.5, r() - 0.5];
            let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
            let axis = a.map(|v| v / l);
            s.sort(axis);
            assert_eq!(s.order(), reference(&centres, axis).as_slice(), "n {n}");
        }
    }
}

#[test]
fn far_first_along_the_view_axis() {
    let centres = [[0.0, 0.0, -1.0], [0.0, 0.0, -5.0], [0.0, 0.0, -3.0]];
    let mut s = DepthSort::new(&centres);
    s.sort([0.0, 0.0, 1.0]);
    assert_eq!(s.order(), &[1, 2, 0]);
    let mut f = Vec::new();
    s.order_f32(&mut f);
    assert_eq!(f, [1.0, 2.0, 0.0]);
}

#[test]
fn ties_keep_their_order_and_nothing_breaks_it() {
    // All at one depth: the order stays as it was given.
    let same = vec![[1.0, 2.0, 3.0]; 50];
    let mut s = DepthSort::new(&same);
    s.sort([0.0, 0.0, 1.0]);
    assert_eq!(s.order(), (0..50).collect::<Vec<u32>>().as_slice());
    // Empty.
    let mut s = DepthSort::new(&[]);
    s.sort([0.0, 0.0, 1.0]);
    assert!(s.order().is_empty());
    // Non-finite centres take key 0, with the farthest splat, so they are
    // among the first drawn; the box ignores them.
    let mut c = cloud(40, 9);
    c[7] = [f32::NAN, 0.0, 0.0];
    c[20] = [0.0, f32::INFINITY, 0.0];
    let mut s = DepthSort::new(&c);
    s.sort([0.0, 0.0, 1.0]);
    let first: Vec<u32> = s.order()[..3].to_vec();
    assert!(first.contains(&7) && first.contains(&20), "{first:?}");
    assert_eq!(s.order(), reference(&c, [0.0, 0.0, 1.0]).as_slice());
}

/// The sorter this replaced: centres as `[f32; 3]`, a sweep for the range,
/// histograms made with the keys, and a second pass gathering keys by index.
fn previous(centres: &[[f32; 3]], axis: [f32; 3], keys: &mut Vec<u32>, scratch: &mut Vec<u32>, order: &mut Vec<u32>) {
    let n = centres.len();
    let depth = |p: &[f32; 3]| axis[0] * p[0] + axis[1] * p[1] + axis[2] * p[2];
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for p in centres {
        let d = depth(p);
        lo = lo.min(d);
        hi = hi.max(d);
    }
    const BITS: u32 = 24;
    const MAX: f32 = ((1u32 << BITS) - 1) as f32;
    const RADIX: u32 = BITS / 2;
    const BUCKETS: usize = 1 << RADIX;
    const MASK: u32 = BUCKETS as u32 - 1;
    let scale = if hi > lo { MAX / (hi - lo) } else { 0.0 };
    let mut counts = vec![0usize; BUCKETS * 2];
    keys.clear();
    keys.extend(centres.iter().map(|p| {
        let k = ((depth(p) - lo) * scale).clamp(0.0, MAX) as u32;
        counts[(k & MASK) as usize] += 1;
        counts[BUCKETS + (k >> RADIX) as usize] += 1;
        k
    }));
    for half in counts.as_chunks_mut::<BUCKETS>().0 {
        let mut sum = 0;
        for c in half {
            let here = *c;
            *c = sum;
            sum += here;
        }
    }
    let (low, high) = counts.split_at_mut(BUCKETS);
    scratch.resize(n, 0);
    order.resize(n, 0);
    for (i, &k) in keys.iter().enumerate() {
        let slot = &mut low[(k & MASK) as usize];
        scratch[*slot] = i as u32;
        *slot += 1;
    }
    for &i in scratch.iter() {
        let slot = &mut high[(keys[i as usize] >> RADIX) as usize];
        order[*slot] = i;
        *slot += 1;
    }
}

fn thread_cpu_ms() -> f64 {
    let mut t = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut t) };
    t.tv_sec as f64 * 1e3 + t.tv_nsec as f64 * 1e-6
}

/// `cargo test --release -p threers-splat-sort -- --ignored --nocapture`
#[test]
#[ignore]
fn timing() {
    let n = 1_000_000;
    let centres = cloud(n, 42);
    let mut s = DepthSort::new(&centres);
    let (mut keys, mut scratch, mut order) = (Vec::new(), Vec::new(), Vec::new());
    let (mut new, mut old) = (Vec::new(), Vec::new());
    let mut floats = Vec::new();
    for f in 0..40 {
        let a = 0.3 + f as f32 * 0.01;
        let axis = [a.sin() * 0.94, 0.34, a.cos() * 0.94];
        let t = thread_cpu_ms();
        s.sort(axis);
        s.order_f32(&mut floats);
        new.push(thread_cpu_ms() - t);
        let t = thread_cpu_ms();
        previous(&centres, axis, &mut keys, &mut scratch, &mut order);
        let _: Vec<f32> = order.iter().map(|&i| i as f32).collect();
        old.push(thread_cpu_ms() - t);
    }
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        (v[v.len() / 2], v[0])
    };
    let (n_med, n_min) = median(new);
    let (o_med, o_min) = median(old);
    println!("1M splats, thread CPU ms, to floats: new {n_med:.2} (least {n_min:.2}), previous {o_med:.2} (least {o_min:.2})");
}
