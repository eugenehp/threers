//! Connectionist temporal classification: the loss that lets the model
//! learn letters from a swipe without anyone saying which point is which
//! letter.
//!
//! Class 0 is the blank ("no letter here"); a target is the classes of a
//! word's keys. The loss is −log of the probability, summed over every
//! way of spreading the target over the frames, computed forward and
//! backward over the trellis (log space, f64).

/// Each row of `z` (`frames × classes`) made log-probabilities.
pub fn log_softmax(z: &[f32], classes: usize) -> Vec<f32> {
    let mut out = vec![0.0; z.len()];
    for (row, o) in z.chunks(classes).zip(out.chunks_mut(classes)) {
        let m = row.iter().copied().fold(f32::MIN, f32::max);
        let s: f32 = row.iter().map(|v| (v - m).exp()).sum();
        let l = m + s.ln();
        for (o, v) in o.iter_mut().zip(row) {
            *o = v - l;
        }
    }
    out
}

fn lse(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// −log p(`target` | `logits`) and its gradient with respect to the
/// logits (`frames × classes`, blank 0, softmax over each row). `None`
/// when the target cannot fit the frames.
pub fn loss(logits: &[f32], classes: usize, target: &[usize]) -> Option<(f32, Vec<f32>)> {
    let frames = logits.len() / classes;
    let lp = log_softmax(logits, classes);
    // The target with blanks around and between: b l1 b l2 … b.
    let mut ext = Vec::with_capacity(2 * target.len() + 1);
    ext.push(0);
    for &l in target {
        debug_assert!(l > 0 && l < classes, "label {l} of {classes}");
        ext.push(l);
        ext.push(0);
    }
    let s = ext.len();
    let need = target.len() + target.windows(2).filter(|w| w[0] == w[1]).count();
    if frames == 0 || need > frames {
        return None;
    }
    let ninf = f64::NEG_INFINITY;
    let at = |t: usize, k: usize| lp[t * classes + k] as f64;
    let skip = |i: usize| i >= 2 && ext[i] != 0 && ext[i] != ext[i - 2];

    let mut alpha = vec![ninf; frames * s];
    alpha[0] = at(0, ext[0]);
    if s > 1 {
        alpha[1] = at(0, ext[1]);
    }
    for t in 1..frames {
        for i in 0..s {
            let mut a = alpha[(t - 1) * s + i];
            if i >= 1 {
                a = lse(a, alpha[(t - 1) * s + i - 1]);
            }
            if skip(i) {
                a = lse(a, alpha[(t - 1) * s + i - 2]);
            }
            alpha[t * s + i] = if a == ninf { ninf } else { a + at(t, ext[i]) };
        }
    }
    let last = (frames - 1) * s;
    let logp = if s > 1 {
        lse(alpha[last + s - 1], alpha[last + s - 2])
    } else {
        alpha[last]
    };
    if !logp.is_finite() {
        return None;
    }

    let mut beta = vec![ninf; frames * s];
    beta[last + s - 1] = at(frames - 1, ext[s - 1]);
    if s > 1 {
        beta[last + s - 2] = at(frames - 1, ext[s - 2]);
    }
    for t in (0..frames - 1).rev() {
        for i in 0..s {
            let mut b = beta[(t + 1) * s + i];
            if i + 1 < s {
                b = lse(b, beta[(t + 1) * s + i + 1]);
            }
            if i + 2 < s && skip(i + 2) {
                b = lse(b, beta[(t + 1) * s + i + 2]);
            }
            beta[t * s + i] = if b == ninf { ninf } else { b + at(t, ext[i]) };
        }
    }

    // d(−log p)/dz = softmax − occupancy, occupancy of class k at t being
    // Σ α·β over its positions / (p · y_t(k)).
    let mut grad = vec![0.0f32; logits.len()];
    let mut occ = vec![ninf; classes];
    for t in 0..frames {
        occ.fill(ninf);
        for i in 0..s {
            let v = alpha[t * s + i] + beta[t * s + i];
            occ[ext[i]] = lse(occ[ext[i]], v);
        }
        for k in 0..classes {
            let y = at(t, k);
            let g = y.exp()
                - if occ[k] == ninf {
                    0.0
                } else {
                    (occ[k] - y - logp).exp()
                };
            grad[t * classes + k] = g as f32;
        }
    }
    Some((-logp as f32, grad))
}

/// The best class per frame, repeats merged, blanks dropped.
pub fn greedy(logits: &[f32], classes: usize) -> Vec<usize> {
    let mut out: Vec<usize> = vec![];
    let mut prev = 0;
    for row in logits.chunks(classes) {
        let best = (0..classes)
            .max_by(|&a, &b| row[a].total_cmp(&row[b]))
            .unwrap_or(0);
        if best != 0 && best != prev {
            out.push(best);
        }
        prev = best;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every frame-level path, summed by brute force.
    fn brute(logits: &[f32], classes: usize, target: &[usize]) -> f64 {
        let frames = logits.len() / classes;
        let lp = log_softmax(logits, classes);
        let mut total = 0.0f64;
        let n = classes.pow(frames as u32);
        for code in 0..n {
            let (mut c, mut path, mut logp) = (code, vec![], 0.0f64);
            for t in 0..frames {
                let k = c % classes;
                c /= classes;
                path.push(k);
                logp += lp[t * classes + k] as f64;
            }
            let mut collapsed = vec![];
            let mut prev = usize::MAX;
            for &k in &path {
                if k != prev && k != 0 {
                    collapsed.push(k);
                }
                prev = k;
            }
            if collapsed == target {
                total += logp.exp();
            }
        }
        -total.ln()
    }

    fn logits(frames: usize, classes: usize, seed: u32) -> Vec<f32> {
        (0..frames * classes)
            .map(|i| (((i as u32).wrapping_mul(2654435761) ^ seed) % 1000) as f32 / 300.0 - 1.5)
            .collect()
    }

    #[test]
    fn matches_brute_force() {
        for (target, seed) in [
            (vec![1, 2], 1),
            (vec![1, 1], 2),
            (vec![2], 3),
            (vec![1, 2, 1], 4),
        ] {
            let z = logits(5, 3, seed);
            let (l, _) = loss(&z, 3, &target).unwrap();
            let b = brute(&z, 3, &target);
            assert!((l as f64 - b).abs() < 1e-4, "{target:?}: {l} vs {b}");
        }
        assert!(
            loss(&logits(2, 3, 0), 3, &[1, 1]).is_none(),
            "needs a blank between"
        );
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let (classes, target) = (4, vec![1, 3, 3, 2]);
        let mut z = logits(9, classes, 7);
        let (_, g) = loss(&z, classes, &target).unwrap();
        for i in (0..z.len()).step_by(3) {
            let h = 1e-2;
            z[i] += h;
            let up = loss(&z, classes, &target).unwrap().0;
            z[i] -= 2.0 * h;
            let down = loss(&z, classes, &target).unwrap().0;
            z[i] += h;
            let fd = (up - down) / (2.0 * h);
            assert!((fd - g[i]).abs() < 2e-3, "{i}: {fd} vs {}", g[i]);
        }
    }

    #[test]
    fn greedy_merges_and_drops_blanks() {
        let z = [0.0, 5.0, 0.0, 0.0, 5.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0];
        assert_eq!(greedy(&z, 3), vec![1, 1]);
    }
}
