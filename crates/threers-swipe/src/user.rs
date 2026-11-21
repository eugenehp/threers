//! What one person types, learnt as they type it: a small 3-gram of their
//! own words, mixed in with the language's next-word models so the bar and
//! the swipes come to know their names, phrases and habits, as a phone's
//! keyboard does.
//!
//! ```
//! use threers_swipe::next_word::WordModel;
//! use threers_swipe::user::UserModel;
//!
//! let me = UserModel::new();
//! me.learn_sentence(&["see", "you", "at", "the", "climbing", "gym"]);
//! assert_eq!(me.predict(&["the"], "", 1)[0].0, "climbing");
//! let saved = me.to_text();
//! assert_eq!(UserModel::from_text(&saved).unwrap().predict(&["the"], "", 1)[0].0, "climbing");
//! ```
//!
//! Witten–Bell smoothing (each context's own counts, then the shorter
//! context's in proportion to how many different words it has been seen
//! followed by), which needs no tuning and does well on few counts. It
//! knows only words it has seen: any other word is `None`, so in an
//! [`Interpolated`](crate::Interpolated) mix it takes nothing from the
//! language's models. Past [`MAX_ENTRIES`] different 3-grams every count
//! halves and what reaches zero goes: old habits fade.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::next_word::{normalize, WordModel, BOS};

/// Different 3-grams kept before every count halves.
pub const MAX_ENTRIES: usize = 50_000;

#[derive(Debug, Clone, Default)]
struct Counts {
    /// (word before last, last word) → next word → count.
    tri: HashMap<(String, String), HashMap<String, u32>>,
    /// last word → next word → count.
    bi: HashMap<String, HashMap<String, u32>>,
    uni: HashMap<String, u32>,
    total: u64,
    entries: usize,
}

impl Counts {
    fn add(&mut self, a: &str, b: &str, w: &str, n: u32) {
        let t = self.tri.entry((a.to_owned(), b.to_owned())).or_default();
        let c = t.entry(w.to_owned()).or_default();
        if *c == 0 {
            self.entries += 1;
        }
        *c += n;
        *self.bi.entry(b.to_owned()).or_default().entry(w.to_owned()).or_default() += n;
        *self.uni.entry(w.to_owned()).or_default() += n;
        self.total += n as u64;
    }

    /// Every 3-gram's count halved, the zeros gone, the rest recounted.
    fn halve(&mut self) {
        let old = std::mem::take(&mut self.tri);
        *self = Counts::default();
        for ((a, b), next) in old {
            for (w, c) in next {
                if c / 2 > 0 {
                    self.add(&a, &b, &w, c / 2);
                }
            }
        }
    }
}

/// Witten–Bell: `(c(h w) + T(h) · lower) / (c(h) + T(h))`.
fn wb(next: Option<&HashMap<String, u32>>, w: &str, lower: f64) -> f64 {
    let Some(next) = next else {
        return lower;
    };
    let total: u32 = next.values().sum();
    let kinds = next.len() as f64;
    if total == 0 {
        return lower;
    }
    let c = next.get(w).copied().unwrap_or(0) as f64;
    (c + kinds * lower) / (total as f64 + kinds)
}

/// One person's words, learnt as they type ([module docs](self)).
#[derive(Debug, Default)]
pub struct UserModel {
    counts: RefCell<Counts>,
}

impl UserModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// `word` typed after `context` (oldest first, a sentence's start as
    /// [`BOS`]).
    pub fn learn(&self, context: &[&str], word: &str) {
        let w = normalize(word);
        if w.is_empty() || !w.chars().any(char::is_alphanumeric) {
            return;
        }
        let at = |i: usize| -> String {
            context
                .len()
                .checked_sub(i)
                .and_then(|j| context.get(j))
                .map_or_else(|| BOS.to_owned(), |c| normalize(c))
        };
        // Before a sentence's start there is only its start.
        let b = at(1);
        let a = if b == BOS { BOS.to_owned() } else { at(2) };
        let mut c = self.counts.borrow_mut();
        c.add(&a, &b, &w, 1);
        if c.entries > MAX_ENTRIES {
            c.halve();
        }
    }

    /// Each word of a sentence after the ones before it.
    pub fn learn_sentence(&self, words: &[&str]) {
        let mut ctx: Vec<&str> = vec![BOS];
        for w in words {
            self.learn(&ctx, w);
            ctx.push(w);
        }
    }

    /// How many words it has learnt.
    pub fn len(&self) -> u64 {
        self.counts.borrow().total
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Forgets everything.
    pub fn clear(&self) {
        *self.counts.borrow_mut() = Counts::default();
    }

    /// Saved: `#threers-user`, then `a b w<TAB>count` a line.
    pub fn to_text(&self) -> String {
        let c = self.counts.borrow();
        let mut lines: Vec<String> = c
            .tri
            .iter()
            .flat_map(|((a, b), next)| next.iter().map(move |(w, n)| format!("{a} {b} {w}\t{n}")))
            .collect();
        lines.sort();
        let mut out = "#threers-user\n".to_owned();
        for l in lines {
            out.push_str(&l);
            out.push('\n');
        }
        out
    }

    pub fn from_text(text: &str) -> Result<Self, String> {
        let mut lines = text.lines();
        if lines.next() != Some("#threers-user") {
            return Err("not a threers user model: no `#threers-user` line".into());
        }
        let mut c = Counts::default();
        for (i, line) in lines.enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let bad = || format!("line {}: `{line}`", i + 2);
            let (g, n) = line.rsplit_once('\t').ok_or_else(bad)?;
            let n: u32 = n.trim().parse().map_err(|_| bad())?;
            let mut it = g.split(' ');
            let (Some(a), Some(b), Some(w), None) = (it.next(), it.next(), it.next(), it.next()) else {
                return Err(bad());
            };
            if n > 0 {
                c.add(a, b, w, n);
            }
        }
        while c.entries > MAX_ENTRIES {
            c.halve();
        }
        Ok(Self {
            counts: RefCell::new(c),
        })
    }

    fn prob(c: &Counts, a: &str, b: &str, w: &str) -> f64 {
        let p1 = c.uni.get(w).copied().unwrap_or(0) as f64 / c.total.max(1) as f64;
        let p2 = wb(c.bi.get(b), w, p1);
        wb(c.tri.get(&(a.to_owned(), b.to_owned())), w, p2)
    }

    fn last_two(context: &[&str]) -> (String, String) {
        let at = |i: usize| {
            context
                .len()
                .checked_sub(i)
                .and_then(|j| context.get(j))
                .map_or_else(|| BOS.to_owned(), |c| normalize(c))
        };
        let b = at(1);
        let a = if b == BOS { BOS.to_owned() } else { at(2) };
        (a, b)
    }
}

impl WordModel for UserModel {
    fn context_len(&self) -> usize {
        2
    }

    fn log_probs(&self, context: &[&str], words: &[&str]) -> Vec<Option<f32>> {
        let c = self.counts.borrow();
        let (a, b) = Self::last_two(context);
        words
            .iter()
            .map(|w| {
                c.uni
                    .contains_key(*w)
                    .then(|| Self::prob(&c, &a, &b, w).ln() as f32)
            })
            .collect()
    }

    fn predict(&self, context: &[&str], prefix: &str, n: usize) -> Vec<(String, f32)> {
        let c = self.counts.borrow();
        let (a, b) = Self::last_two(context);
        let prefix = normalize(prefix);
        let mut out: Vec<(String, f64)> = c
            .uni
            .keys()
            .filter(|w| w.starts_with(&prefix))
            .map(|w| (w.clone(), Self::prob(&c, &a, &b, w)))
            .collect();
        out.sort_by(|x, y| y.1.total_cmp(&x.1).then_with(|| x.0.cmp(&y.0)));
        out.truncate(n);
        out.into_iter().map(|(w, p)| (w, p.ln() as f32)).collect()
    }
}

/// A language's models with one person's mixed in, `weight` theirs.
pub struct WithUser<'a> {
    pub base: &'a dyn WordModel,
    pub user: &'a UserModel,
    pub weight: f32,
}

impl WordModel for WithUser<'_> {
    fn context_len(&self) -> usize {
        self.base.context_len().max(2)
    }

    fn log_probs(&self, context: &[&str], words: &[&str]) -> Vec<Option<f32>> {
        let w = self.weight.clamp(0.0, 1.0) as f64;
        self.base
            .log_probs(context, words)
            .into_iter()
            .zip(self.user.log_probs(context, words))
            .map(|(b, u)| {
                let p = (1.0 - w) * b.map_or(0.0, |l| (l as f64).exp())
                    + w * u.map_or(0.0, |l| (l as f64).exp());
                (p > 0.0).then(|| p.ln() as f32)
            })
            .collect()
    }

    fn predict(&self, context: &[&str], prefix: &str, n: usize) -> Vec<(String, f32)> {
        let mut words: Vec<String> = self
            .base
            .predict(context, prefix, 2 * n)
            .into_iter()
            .map(|e| e.0)
            .collect();
        for (w, _) in self.user.predict(context, prefix, 2 * n) {
            if !words.contains(&w) {
                words.push(w);
            }
        }
        let refs: Vec<&str> = words.iter().map(String::as_str).collect();
        let mut out: Vec<(String, f32)> = words
            .iter()
            .cloned()
            .zip(self.log_probs(context, &refs))
            .filter_map(|(w, lp)| Some((w, lp?)))
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out.truncate(n);
        out
    }

    fn unknown_log_prob(&self, context: &[&str]) -> Option<f32> {
        // The user's model has no unknown mass of its own: the base's share.
        self.base
            .unknown_log_prob(context)
            .map(|l| l + (1.0 - self.weight.clamp(0.0, 1.0)).ln())
    }
}

/// The words newly finished between the text before the caret as it was
/// (`was`) and as it is (`now`), each with the words before it: a space or
/// punctuation after a word, or a swipe's word after the last.
pub fn finished_words(was: &str, now: &str) -> Vec<(Vec<String>, String)> {
    let done = |t: &str| t.len() - crate::next_word::typing_len(t);
    let (a, b) = (&was[..done(was)], &now[..done(now)]);
    if b.len() <= a.len() || !b.starts_with(a) || b.len() - a.len() > 256 {
        return vec![];
    }
    let mut out = vec![];
    let mut start = None;
    for (i, ch) in b.char_indices().skip_while(|(i, _)| *i < a.len()) {
        let word = ch.is_alphanumeric() || ch == '\'' || ch == '’';
        match (word, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                let (ctx, _) = crate::next_word::split_context(&b[..s], 2);
                out.push((ctx, normalize(&b[s..i])));
                start = None;
            }
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_what_follows_and_forgets_slowly() {
        let me = UserModel::new();
        assert!(me.predict(&["the"], "", 3).is_empty());
        for _ in 0..3 {
            me.learn_sentence(&["meet", "me", "at", "the", "climbing", "gym"]);
        }
        me.learn_sentence(&["the", "cat", "sat"]);
        assert_eq!(me.predict(&["at", "the"], "", 1)[0].0, "climbing");
        assert_eq!(me.predict(&[BOS], "", 1)[0].0, "meet");
        assert_eq!(me.predict(&["the"], "c", 2).len(), 2);
        assert_eq!(me.log_probs(&["the"], &["zebra"]), [None]);
        // A distribution over the words it knows.
        let all = ["meet", "me", "at", "the", "climbing", "gym", "cat", "sat"];
        let sum: f64 = me
            .log_probs(&["at", "the"], &all)
            .iter()
            .map(|l| (l.unwrap() as f64).exp())
            .sum();
        assert!((sum - 1.0).abs() < 1e-6, "{sum}");
        let back = UserModel::from_text(&me.to_text()).unwrap();
        assert_eq!(back.to_text(), me.to_text());
        assert_eq!(back.len(), me.len());
        assert!(UserModel::from_text("nope").is_err());
        assert!(UserModel::from_text("#threers-user\na b\t1\n").is_err());
    }

    #[test]
    fn halving_keeps_it_bounded() {
        let me = UserModel::new();
        for i in 0..(MAX_ENTRIES + 10) {
            me.learn(&["x", &format!("w{i}")], "y");
        }
        assert!(me.counts.borrow().entries <= MAX_ENTRIES);
    }

    #[test]
    fn words_finish_on_a_space_or_the_next_swipe() {
        assert_eq!(finished_words("I am goi", "I am going "), vec![(vec!["i".to_owned(), "am".into()], "going".to_owned())]);
        assert_eq!(finished_words("Fine. ", "Fine. Thanks"), vec![]);
        // A swipe after a word: the space and the swiped word at once.
        assert_eq!(
            finished_words("Hi there", "Hi there you"),
            vec![(vec![BOS.to_owned(), "hi".into()], "there".to_owned())]
        );
        assert_eq!(finished_words("Hi there ", "Hi the"), vec![], "deleting teaches nothing");
        assert_eq!(finished_words("", "Ok. "), vec![(vec![BOS.to_owned()], "ok".to_owned())]);
    }

    #[test]
    fn mixed_in_it_lifts_the_users_words() {
        // A language's model: many words may follow, none very likely.
        let base = UserModel::new();
        for end in ["soon", "later", "then", "there", "tomorrow"] {
            base.learn_sentence(&["see", "you", end]);
        }
        let me = UserModel::new();
        for _ in 0..3 {
            me.learn_sentence(&["see", "you", "tmrw"]);
        }
        let mixed = WithUser { base: &base, user: &me, weight: 0.3 };
        assert_eq!(mixed.predict(&["see", "you"], "", 1)[0].0, "tmrw");
        let none = WithUser { base: &base, user: &UserModel::new(), weight: 0.3 };
        let p = none.predict(&["see", "you"], "", 2);
        assert_eq!(p.len(), 2, "an empty user model changes nothing: {p:?}");
    }
}
