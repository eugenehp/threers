//! C++'s `std::sort` as libc++ has it (LLVM 17 and later: `__introsort` in
//! `__algorithm/sort.h`, for a type that is not sorted branchlessly), for
//! results that must match a C++ program's to the order of equal items.
//!
//! `std::sort` is not stable: items that compare equal come out in the
//! order this algorithm leaves them in, which a stable sort does not give.
//!
//! The algorithm follows libc++ (the LLVM Project, Apache License 2.0 with
//! LLVM Exceptions).

/// Sorts `v` the way `std::sort( v.begin(), v.end(), less )` does.
pub fn sort<T: Clone>(v: &mut [T], mut less: impl FnMut(&T, &T) -> bool) {
    if v.is_empty() {
        return;
    }

    // __sort_dispatch: 2 * log2(len)
    let depth = 2 * (usize::BITS - 1 - v.len().leading_zeros()) as usize;
    introsort(v, 0, v.len(), &mut less, depth, true);
}

fn sort3<T>(
    v: &mut [T],
    x: usize,
    y: usize,
    z: usize,
    less: &mut impl FnMut(&T, &T) -> bool,
) -> bool {
    if !less(&v[y], &v[x]) {
        if !less(&v[z], &v[y]) {
            return false;
        }

        v.swap(y, z);

        if less(&v[y], &v[x]) {
            v.swap(x, y);
        }

        return true;
    }

    if less(&v[z], &v[y]) {
        v.swap(x, z);
        return true;
    }

    v.swap(x, y);

    if less(&v[z], &v[y]) {
        v.swap(y, z);
    }

    true
}

fn sort4<T>(
    v: &mut [T],
    x1: usize,
    x2: usize,
    x3: usize,
    x4: usize,
    less: &mut impl FnMut(&T, &T) -> bool,
) {
    sort3(v, x1, x2, x3, less);

    if less(&v[x4], &v[x3]) {
        v.swap(x3, x4);

        if less(&v[x3], &v[x2]) {
            v.swap(x2, x3);

            if less(&v[x2], &v[x1]) {
                v.swap(x1, x2);
            }
        }
    }
}

fn sort5<T>(v: &mut [T], x1: usize, less: &mut impl FnMut(&T, &T) -> bool) {
    let (x2, x3, x4, x5) = (x1 + 1, x1 + 2, x1 + 3, x1 + 4);
    sort4(v, x1, x2, x3, x4, less);

    if less(&v[x5], &v[x4]) {
        v.swap(x4, x5);

        if less(&v[x4], &v[x3]) {
            v.swap(x3, x4);

            if less(&v[x3], &v[x2]) {
                v.swap(x2, x3);

                if less(&v[x2], &v[x1]) {
                    v.swap(x1, x2);
                }
            }
        }
    }
}

/// The ranges of up to five items, which every path sorts the same way.
fn sort_small<T>(
    v: &mut [T],
    first: usize,
    last: usize,
    less: &mut impl FnMut(&T, &T) -> bool,
) -> bool {
    match last - first {
        0 | 1 => {}
        2 => {
            if less(&v[last - 1], &v[first]) {
                v.swap(first, last - 1);
            }
        }
        3 => {
            sort3(v, first, first + 1, last - 1, less);
        }
        4 => sort4(v, first, first + 1, first + 2, last - 1, less),
        5 => sort5(v, first, less),
        _ => return false,
    }

    true
}

/// `__insertion_sort`, and `__insertion_sort_unguarded` when the item ahead
/// of `first` is known not to be greater than any in the range.
fn insertion_sort<T: Clone>(
    v: &mut [T],
    first: usize,
    last: usize,
    less: &mut impl FnMut(&T, &T) -> bool,
    guarded: bool,
) {
    for i in first + 1..last {
        let mut j = i - 1;

        if less(&v[i], &v[j]) {
            let t = v[i].clone();
            let mut k = j;
            j = i;

            loop {
                v[j] = v[k].clone();
                j = k;

                if guarded && j == first {
                    break;
                }

                k -= 1;

                if !less(&t, &v[k]) {
                    break;
                }
            }

            v[j] = t;
        }
    }
}

fn insertion_sort_incomplete<T: Clone>(
    v: &mut [T],
    first: usize,
    last: usize,
    less: &mut impl FnMut(&T, &T) -> bool,
) -> bool {
    if sort_small(v, first, last, less) {
        return true;
    }

    let mut j = first + 2;
    sort3(v, first, first + 1, j, less);

    const LIMIT: u32 = 8;
    let mut count = 0;
    let mut i = j + 1;

    while i != last {
        if less(&v[i], &v[j]) {
            let t = v[i].clone();
            let mut k = j;
            j = i;

            loop {
                v[j] = v[k].clone();
                j = k;

                if j == first {
                    break;
                }

                k -= 1;

                if !less(&t, &v[k]) {
                    break;
                }
            }

            v[j] = t;
            count += 1;

            if count == LIMIT {
                return i + 1 == last;
            }
        }

        j = i;
        i += 1;
    }

    true
}

/// `__partition_with_equals_on_right`: where the pivot ends up, and whether
/// the range was partitioned already.
fn partition_with_equals_on_right<T: Clone>(
    v: &mut [T],
    begin: usize,
    end: usize,
    less: &mut impl FnMut(&T, &T) -> bool,
) -> (usize, bool) {
    let mut first = begin;
    let mut last = end;
    let pivot = v[first].clone();

    loop {
        first += 1;

        if !less(&v[first], &pivot) {
            break;
        }
    }

    if begin == first - 1 {
        while first < last {
            last -= 1;

            if less(&v[last], &pivot) {
                break;
            }
        }
    } else {
        loop {
            last -= 1;

            if less(&v[last], &pivot) {
                break;
            }
        }
    }

    let already_partitioned = first >= last;

    while first < last {
        v.swap(first, last);

        loop {
            first += 1;

            if !less(&v[first], &pivot) {
                break;
            }
        }

        loop {
            last -= 1;

            if less(&v[last], &pivot) {
                break;
            }
        }
    }

    let pivot_pos = first - 1;

    if begin != pivot_pos {
        v[begin] = v[pivot_pos].clone();
    }

    v[pivot_pos] = pivot;
    (pivot_pos, already_partitioned)
}

/// `__partition_with_equals_on_left`: the start of the items greater than
/// the pivot.
fn partition_with_equals_on_left<T: Clone>(
    v: &mut [T],
    begin: usize,
    end: usize,
    less: &mut impl FnMut(&T, &T) -> bool,
) -> usize {
    let mut first = begin;
    let mut last = end;
    let pivot = v[first].clone();

    if less(&pivot, &v[last - 1]) {
        loop {
            first += 1;

            if less(&pivot, &v[first]) {
                break;
            }
        }
    } else {
        loop {
            first += 1;

            if !(first < last && !less(&pivot, &v[first])) {
                break;
            }
        }
    }

    if first < last {
        loop {
            last -= 1;

            if !less(&pivot, &v[last]) {
                break;
            }
        }
    }

    while first < last {
        v.swap(first, last);

        loop {
            first += 1;

            if less(&pivot, &v[first]) {
                break;
            }
        }

        loop {
            last -= 1;

            if !less(&pivot, &v[last]) {
                break;
            }
        }
    }

    let pivot_pos = first - 1;

    if begin != pivot_pos {
        v[begin] = v[pivot_pos].clone();
    }

    v[pivot_pos] = pivot;
    first
}

/// `__sift_down`, on the heap of `len` items that starts at `first`.
fn sift_down<T: Clone>(
    v: &mut [T],
    first: usize,
    less: &mut impl FnMut(&T, &T) -> bool,
    len: usize,
    mut start: usize,
) {
    let mut child = start;

    if len < 2 || (len - 2) / 2 < child {
        return;
    }

    child = 2 * child + 1;

    if child + 1 < len && less(&v[first + child], &v[first + child + 1]) {
        child += 1;
    }

    if less(&v[first + child], &v[first + start]) {
        return;
    }

    let top = v[first + start].clone();

    loop {
        v[first + start] = v[first + child].clone();
        start = child;

        if (len - 2) / 2 < child {
            break;
        }

        child = 2 * child + 1;

        if child + 1 < len && less(&v[first + child], &v[first + child + 1]) {
            child += 1;
        }

        if less(&v[first + child], &top) {
            break;
        }
    }

    v[first + start] = top;
}

/// `__pop_heap`
fn pop_heap<T: Clone>(
    v: &mut [T],
    first: usize,
    last: usize,
    less: &mut impl FnMut(&T, &T) -> bool,
    len: usize,
) {
    if len <= 1 {
        return;
    }

    let top = v[first].clone();

    // __floyd_sift_down
    let mut hole = first;
    let mut child_i = first;
    let mut child = 0;

    loop {
        child_i += child + 1;
        child = 2 * child + 1;

        if child + 1 < len && less(&v[child_i], &v[child_i + 1]) {
            child_i += 1;
            child += 1;
        }

        v[hole] = v[child_i].clone();
        hole = child_i;

        if child > (len - 2) / 2 {
            break;
        }
    }

    let last = last - 1;

    if hole == last {
        v[hole] = top;
        return;
    }

    v[hole] = v[last].clone();
    hole += 1;
    v[last] = top;

    // __sift_up
    let mut len = hole - first;

    if len > 1 {
        len = (len - 2) / 2;
        let mut ptr = first + len;
        let mut last = hole - 1;

        if less(&v[ptr], &v[last]) {
            let t = v[last].clone();

            loop {
                v[last] = v[ptr].clone();
                last = ptr;

                if len == 0 {
                    break;
                }

                len = (len - 1) / 2;
                ptr = first + len;

                if !less(&v[ptr], &t) {
                    break;
                }
            }

            v[last] = t;
        }
    }
}

/// `__partial_sort` of a whole range: a heap sort.
fn heap_sort<T: Clone>(
    v: &mut [T],
    first: usize,
    last: usize,
    less: &mut impl FnMut(&T, &T) -> bool,
) {
    let n = last - first;

    if n > 1 {
        for start in (0..=(n - 2) / 2).rev() {
            sift_down(v, first, less, n, start);
        }
    }

    let mut last = last;
    let mut len = n;

    while len > 1 {
        pop_heap(v, first, last, less, len);
        last -= 1;
        len -= 1;
    }
}

fn introsort<T: Clone>(
    v: &mut [T],
    mut first: usize,
    mut last: usize,
    less: &mut impl FnMut(&T, &T) -> bool,
    mut depth: usize,
    mut leftmost: bool,
) {
    // upper bound for using insertion sort
    const LIMIT: usize = 24;
    // lower bound for using Tuckey's ninther technique
    const NINTHER_THRESHOLD: usize = 128;

    loop {
        let len = last - first;

        if sort_small(v, first, last, less) {
            return;
        }

        if len < LIMIT {
            insertion_sort(v, first, last, less, leftmost);
            return;
        }

        if depth == 0 {
            heap_sort(v, first, last, less);
            return;
        }

        depth -= 1;

        let half_len = len / 2;

        if len > NINTHER_THRESHOLD {
            sort3(v, first, first + half_len, last - 1, less);
            sort3(v, first + 1, first + (half_len - 1), last - 2, less);
            sort3(v, first + 2, first + (half_len + 1), last - 3, less);
            sort3(
                v,
                first + (half_len - 1),
                first + half_len,
                first + (half_len + 1),
                less,
            );
            v.swap(first, first + half_len);
        } else {
            sort3(v, first + half_len, first, last - 1, less);
        }

        // the items to the left are sorted already: if the pivot equals
        // the greatest of them, all of [first, pivot] equal it
        if !leftmost && !less(&v[first - 1], &v[first]) {
            first = partition_with_equals_on_left(v, first, last, less);
            continue;
        }

        let (i, already_partitioned) = partition_with_equals_on_right(v, first, last, less);

        if already_partitioned {
            let fs = insertion_sort_incomplete(v, first, i, less);

            if insertion_sort_incomplete(v, i + 1, last, less) {
                if fs {
                    return;
                }

                last = i;
                continue;
            } else if fs {
                first = i + 1;
                continue;
            }
        }

        introsort(v, first, i, less, depth, leftmost);
        leftmost = false;
        first = i + 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reproducible stream of small numbers.
    fn numbers(count: usize, modulus: u32, mut seed: u32) -> Vec<(u32, usize)> {
        (0..count)
            .map(|i| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 16) % modulus, i)
            })
            .collect()
    }

    #[test]
    fn sorts() {
        for (count, modulus) in [
            (0, 1),
            (1, 1),
            (2, 2),
            (5, 3),
            (23, 4),
            (24, 5),
            (100, 7),
            (129, 3),
            (5000, 50),
        ] {
            let mut v = numbers(count, modulus, count as u32);
            let mut expected = v.clone();
            sort(&mut v, |a, b| a.0 < b.0);
            assert!(
                v.windows(2).all(|w| w[0].0 <= w[1].0),
                "{count} items are sorted"
            );

            // the same items, whatever the order of the equal ones
            v.sort();
            expected.sort();
            assert_eq!(v, expected);
        }
    }

    /// What `std::sort` of Apple's libc++ makes of 30 pairs compared by
    /// their first number alone: (i * 7 % 5, i) for i in 0..30.
    #[test]
    fn equal_items_come_out_as_libcxx_leaves_them() {
        let mut v: Vec<(u32, u32)> = (0..30).map(|i| (i * 7 % 5, i)).collect();
        sort(&mut v, |a, b| a.0 < b.0);
        let order: Vec<u32> = v.iter().map(|p| p.1).collect();
        assert_eq!(order, LIBCXX_ORDER);
    }

    const LIBCXX_ORDER: [u32; 30] = [
        0, 5, 25, 15, 20, 10, 3, 8, 13, 18, 23, 28, 1, 6, 11, 16, 21, 26, 9, 24, 19, 14, 4, 29, 2,
        7, 12, 17, 22, 27,
    ];

    /// The same pairs, 200 and 1000 of them (the second takes the ninther
    /// path): an FNV-1a hash over the order libc++ leaves them in.
    #[test]
    fn longer_runs_of_equal_items() {
        for (count, expected) in [
            (200u32, 5914336536531584605u64),
            (1000, 16863546940096229641),
        ] {
            let mut v: Vec<(u32, u32)> = (0..count).map(|i| (i * 7 % 5, i)).collect();
            sort(&mut v, |a, b| a.0 < b.0);

            let hash = v.iter().fold(1469598103934665603u64, |h, p| {
                (h ^ u64::from(p.1)).wrapping_mul(1099511628211)
            });
            assert_eq!(hash, expected, "{count} items");
        }
    }
}
