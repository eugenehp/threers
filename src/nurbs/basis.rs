//! B-spline basis functions — the numerical core everything else stands on.
//!
//! All of it is f64. The exact-CSG kernel works in f64 (`V3 = [f64; 3]`,
//! `exact_csg::mod`), and feeding f32-derived points into `orient3d` would throw
//! away the exact predicates, so the NURBS layer never narrows until it hands a
//! `BufferGeometry` to the renderer.
//!
//! Algorithm numbers refer to Piegl & Tiller, *The NURBS Book* (2nd ed.).

/// Find the knot span index `i` with `knots[i] <= u < knots[i + 1]`, clamped to
/// the valid range `[degree, n_ctrl - 1]`. Binary search — A2.1.
///
/// The predecessor in `curves::nurbs` scanned linearly from `degree` and had a
/// `saturating_sub` fallback that silently produced the wrong span for `u`
/// below the domain; this clamps instead, which is what every caller wants.
pub fn find_span(degree: usize, n_ctrl: usize, knots: &[f64], u: f64) -> usize {
    debug_assert!(n_ctrl > degree, "need at least degree + 1 control points");
    let n = n_ctrl - 1;

    // Clamp to the domain [knots[degree], knots[n + 1]].
    if u >= knots[n + 1] {
        return n;
    }
    if u <= knots[degree] {
        return degree;
    }

    let mut low = degree;
    let mut high = n + 1;
    let mut mid = (low + high) / 2;
    while u < knots[mid] || u >= knots[mid + 1] {
        if u < knots[mid] {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
    }
    mid
}

/// Evaluate the `degree + 1` non-zero basis functions at `u` — A2.2.
///
/// Returns `N[0..=degree]`, where `N[j]` multiplies control point
/// `span - degree + j`. No division by zero is possible: the denominators are
/// `right[r + 1] + left[j - r]`, which the span search guarantees positive.
pub fn basis_funcs(span: usize, u: f64, degree: usize, knots: &[f64]) -> Vec<f64> {
    let mut n = vec![0.0; degree + 1];
    let mut left = vec![0.0; degree + 1];
    let mut right = vec![0.0; degree + 1];
    n[0] = 1.0;

    for j in 1..=degree {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        let mut saved = 0.0;
        for r in 0..j {
            let temp = n[r] / (right[r + 1] + left[j - r]);
            n[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        n[j] = saved;
    }
    n
}

/// Basis functions and their derivatives up to order `n_ders` — A2.3.
///
/// `ders[k][j]` is the `k`-th derivative of the basis function for control
/// point `span - degree + j`. Orders above `degree` are identically zero (a
/// degree-`p` spline is `C^∞` only piecewise; its `(p+1)`-th derivative
/// vanishes), and the loop bound reflects that rather than computing garbage.
#[allow(clippy::needless_range_loop)]
pub fn ders_basis_funcs(
    span: usize,
    u: f64,
    degree: usize,
    n_ders: usize,
    knots: &[f64],
) -> Vec<Vec<f64>> {
    let p = degree;
    let mut ders = vec![vec![0.0; p + 1]; n_ders + 1];

    // ndu holds the basis values (upper triangle) and knot differences (lower).
    let mut ndu = vec![vec![0.0; p + 1]; p + 1];
    let mut left = vec![0.0; p + 1];
    let mut right = vec![0.0; p + 1];
    ndu[0][0] = 1.0;

    for j in 1..=p {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        let mut saved = 0.0;
        for r in 0..j {
            ndu[j][r] = right[r + 1] + left[j - r];
            let temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }

    for (j, d) in ders[0].iter_mut().enumerate() {
        *d = ndu[j][p];
    }

    // A degree-`p` spline has identically zero derivatives past order `p`, and
    // A2.3's index arithmetic goes negative there. Compute up to `p` and leave
    // the higher rows as the zeros they mathematically are.
    let k_max = n_ders.min(p);
    if k_max == 0 {
        return ders;
    }

    // `a` alternates between two rows of coefficients; `s1`/`s2` index them.
    let mut a = vec![vec![0.0; p + 1]; 2];
    for r in 0..=p {
        let (mut s1, mut s2) = (0usize, 1usize);
        a[0][0] = 1.0;

        for k in 1..=k_max {
            let mut d = 0.0;
            let rk = r as isize - k as isize;
            let pk = p as isize - k as isize;

            if r as isize >= k as isize {
                a[s2][0] = a[s1][0] / ndu[(pk + 1) as usize][rk as usize];
                d = a[s2][0] * ndu[rk as usize][pk as usize];
            }

            let j1 = if rk >= -1 { 1isize } else { -rk };
            let j2 = if (r as isize - 1) <= pk {
                k as isize - 1
            } else {
                p as isize - r as isize
            };

            let mut j = j1;
            while j <= j2 {
                a[s2][j as usize] = (a[s1][j as usize] - a[s1][(j - 1) as usize])
                    / ndu[(pk + 1) as usize][(rk + j) as usize];
                d += a[s2][j as usize] * ndu[(rk + j) as usize][pk as usize];
                j += 1;
            }

            if (r as isize) <= pk {
                a[s2][k] = -a[s1][k - 1] / ndu[(pk + 1) as usize][r];
                d += a[s2][k] * ndu[r][pk as usize];
            }

            ders[k][r] = d;
            std::mem::swap(&mut s1, &mut s2);
        }
    }

    // Multiply through by the falling factorial p!/(p-k)!.
    let mut acc = p as f64;
    for k in 1..=k_max {
        for j in 0..=p {
            ders[k][j] *= acc;
        }
        acc *= (p as isize - k as isize) as f64;
    }
    ders
}

/// Binomial coefficient `n choose k`, exact for the small `n` derivative
/// formulas use (nothing here goes past a handful of orders).
pub fn binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut acc = 1.0;
    for i in 0..k {
        acc = acc * (n - i) as f64 / (i + 1) as f64;
    }
    acc
}

/// A clamped uniform knot vector for `n_ctrl` control points of degree `p`:
/// `p + 1` zeros, `n_ctrl - p - 1` uniformly spaced interior knots, `p + 1` ones.
pub fn uniform_clamped_knots(degree: usize, n_ctrl: usize) -> Vec<f64> {
    let p = degree;
    let m = n_ctrl + p + 1;
    let interior = n_ctrl - p - 1;
    let mut knots = Vec::with_capacity(m);
    knots.extend(std::iter::repeat_n(0.0, p + 1));
    for i in 1..=interior {
        knots.push(i as f64 / (interior + 1) as f64);
    }
    knots.extend(std::iter::repeat_n(1.0, p + 1));
    debug_assert_eq!(knots.len(), m);
    knots
}

/// Is `knots` a valid knot vector for `n_ctrl` control points of `degree`?
/// Checks length and monotonicity — the two things that turn into out-of-bounds
/// indexing or NaN downstream rather than a merely wrong answer.
pub fn knots_valid(degree: usize, n_ctrl: usize, knots: &[f64]) -> bool {
    if n_ctrl <= degree || knots.len() != n_ctrl + degree + 1 {
        return false;
    }
    if knots.windows(2).any(|w| w[1] < w[0]) {
        return false;
    }
    // The domain must be non-degenerate.
    knots[n_ctrl] > knots[degree]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open_knots() -> Vec<f64> {
        // degree 3, 7 control points → 11 knots.
        vec![0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]
    }

    #[test]
    fn span_search_matches_linear_scan() {
        let knots = open_knots();
        let (p, n_ctrl) = (3, 7);
        for i in 0..=1000 {
            let u = i as f64 / 1000.0;
            let span = find_span(p, n_ctrl, &knots, u);
            assert!(span >= p && span < n_ctrl);
            if u < 1.0 {
                assert!(knots[span] <= u && u < knots[span + 1], "u = {u}");
            }
        }
    }

    #[test]
    fn span_clamps_outside_the_domain() {
        let knots = open_knots();
        assert_eq!(find_span(3, 7, &knots, -5.0), 3);
        assert_eq!(find_span(3, 7, &knots, 5.0), 6);
    }

    #[test]
    fn basis_is_a_partition_of_unity() {
        let knots = open_knots();
        for i in 0..=500 {
            let u = i as f64 / 500.0;
            let span = find_span(3, 7, &knots, u);
            let n = basis_funcs(span, u, 3, &knots);
            let sum: f64 = n.iter().sum();
            assert!((sum - 1.0).abs() < 1e-14, "sum = {sum} at u = {u}");
            assert!(n.iter().all(|&x| x >= -1e-15), "negative basis at u = {u}");
        }
    }

    #[test]
    fn zeroth_derivative_row_equals_the_basis() {
        let knots = open_knots();
        for i in 0..=100 {
            let u = i as f64 / 100.0;
            let span = find_span(3, 7, &knots, u);
            let n = basis_funcs(span, u, 3, &knots);
            let ders = ders_basis_funcs(span, u, 3, 2, &knots);
            for j in 0..=3 {
                assert!((n[j] - ders[0][j]).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn derivative_sums_vanish() {
        // Σ_j N_j(u) ≡ 1 ⇒ every derivative order sums to zero.
        let knots = open_knots();
        for i in 1..100 {
            let u = i as f64 / 100.0;
            let span = find_span(3, 7, &knots, u);
            let ders = ders_basis_funcs(span, u, 3, 3, &knots);
            for (k, d) in ders.iter().enumerate().take(4).skip(1) {
                let sum: f64 = d.iter().sum();
                assert!(sum.abs() < 1e-9, "order {k} sum = {sum} at u = {u}");
            }
        }
    }

    #[test]
    fn binomials() {
        assert_eq!(binomial(0, 0), 1.0);
        assert_eq!(binomial(5, 0), 1.0);
        assert_eq!(binomial(5, 2), 10.0);
        assert_eq!(binomial(5, 5), 1.0);
        assert_eq!(binomial(5, 6), 0.0);
        assert_eq!(binomial(10, 5), 252.0);
    }

    #[test]
    fn knot_validation() {
        assert!(knots_valid(3, 7, &open_knots()));
        assert!(!knots_valid(3, 6, &open_knots()), "wrong length");
        let mut bad = open_knots();
        bad[5] = 0.1; // now decreasing
        assert!(!knots_valid(3, 7, &bad));
        assert!(knots_valid(2, 4, &uniform_clamped_knots(2, 4)));
    }
}
