//! An n-gram model with interpolated Kneser–Ney smoothing: counts of the
//! word sequences in a text, and how many different words each was seen
//! after for the shorter ones.
//!
//! ```
//! use threers_swipe::next_word::WordModel;
//! use threers_swipe::ngram::{NgramConfig, NgramModel};
//!
//! let lm = NgramModel::train("I am going to the shop. I am going to bed. Too late.", &NgramConfig::default());
//! let next = lm.predict(&["going"], "", 1);
//! assert_eq!(next[0].0, "to");
//! let p = lm.log_probs(&["am", "going"], &["to", "too"]);
//! assert!(p[0] > p[1]);
//! ```
//!
//! `p(w | h) = max(c(h w) − D, 0) / c(h ·) + D · N₁₊(h ·) / c(h ·) · p(w | h′)`,
//! `h′` being `h` less its oldest word, down to a uniform `1 / V`. The
//! highest order counts what was seen; the lower ones count distinct words
//! seen before (raw counts for those starting a sentence, which nothing
//! precedes).
//!
//! Held as sorted flat arrays per context length, found by bisection.
//! Saved two ways: binary ([`NgramModel::to_bytes`], what
//! [`save`](NgramModel::save) writes: the arrays varint-coded, read back in
//! one pass), or as text: a `#threers-ngram order=N discount=D` line, then
//! one line per highest-order n-gram, `w₁ … wₙ<TAB>count`, sentences padded
//! with `<s>`, everything else rebuilt from those on load, so counts from
//! elsewhere load the same way. [`load`](NgramModel::load) reads either.

use std::collections::HashMap;
use std::path::Path;

use crate::next_word::{normalize, sentences, WordModel, BOS, EOS, UNK};

/// How the model is made from a text.
#[derive(Debug, Clone)]
pub struct NgramConfig {
    /// Words per n-gram: 3 reads the two words before.
    pub order: usize,
    /// Most words kept, the commonest; the rest count as `<unk>`, so the
    /// model learns how often a word it does not know comes (a swipe's
    /// rare words get that share; with every word kept it is only the
    /// smoothing's). 0 keeps all.
    pub vocab: usize,
    /// Highest-order n-grams seen fewer times than this are dropped.
    pub min_count: u32,
    /// Kneser–Ney's absolute discount.
    pub discount: f32,
}

impl Default for NgramConfig {
    fn default() -> Self {
        Self {
            order: 3,
            vocab: 40_000,
            min_count: 1,
            discount: 0.75,
        }
    }
}

/// The words seen after one context.
/// The contexts of one length and the words seen after each: sorted flat
/// arrays, searched by bisection.
#[derive(Debug, Clone, Default)]
struct Level {
    /// `k` words a context, the contexts in order.
    contexts: Vec<u32>,
    /// Context `c`'s followers are `next[starts[c]..starts[c + 1]]`, by
    /// word id; one more entry than contexts.
    starts: Vec<u32>,
    /// Each context's followers' counts summed.
    totals: Vec<u32>,
}

impl Level {
    fn len(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }

    /// The index of context `ctx` (as long as this level's), if seen.
    fn find(&self, ctx: &[u32]) -> Option<usize> {
        let k = ctx.len();
        if k == 0 {
            return (self.len() > 0).then_some(0);
        }
        let (mut lo, mut hi) = (0, self.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            match self.contexts[mid * k..mid * k + k].cmp(ctx) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }
}

const UNK_ID: u32 = 0;
const BOS_ID: u32 = 1;
const EOS_ID: u32 = 2;
/// The binary form's first bytes.
const MAGIC: &[u8; 5] = b"TNGR\x01";

/// An interpolated Kneser–Ney n-gram model.
#[derive(Debug, Clone)]
pub struct NgramModel {
    order: usize,
    discount: f32,
    /// By id: `<unk>`, `<s>`, `</s>`, then the words, commonest first.
    words: Vec<String>,
    ids: HashMap<String, u32>,
    /// By context length, 0 to order − 1; the top one's counts are the
    /// raw counts, the others' Kneser–Ney's adjusted ones.
    levels: Vec<Level>,
    /// Every level's followers, word id and count.
    next: Vec<u32>,
    counts: Vec<u32>,
    /// Word ids by spelling, for completions.
    by_spelling: Vec<u32>,
    /// Word ids by how many words they follow, for a next word with no
    /// context to go on.
    by_count: Vec<u32>,
}

/// A context found in the model: its level (length) and index there.
type Found = (usize, usize);

impl NgramModel {
    /// Counts `text`'s sentences.
    pub fn train(text: &str, config: &NgramConfig) -> Self {
        Self::train_sentences(&sentences(text), config)
    }

    /// Counts sentences already split into words.
    pub fn train_sentences(sentences: &[Vec<String>], config: &NgramConfig) -> Self {
        let order = config.order.max(1);
        let mut freq: HashMap<&str, u64> = HashMap::new();
        for s in sentences {
            for w in s {
                *freq.entry(w.as_str()).or_default() += 1;
            }
        }
        let mut vocab: Vec<(&str, u64)> = freq.into_iter().collect();
        vocab.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        if config.vocab > 0 {
            vocab.truncate(config.vocab);
        }
        let mut words: Vec<String> = [UNK, BOS, EOS].map(String::from).to_vec();
        words.extend(vocab.iter().map(|(w, _)| w.to_string()));
        let ids: HashMap<&str, u32> = words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i as u32))
            .collect();
        let mut counts: HashMap<Box<[u32]>, u32> = HashMap::new();
        let mut padded = vec![];
        for s in sentences {
            padded.clear();
            padded.resize(order - 1, BOS_ID);
            padded.extend(s.iter().map(|w| ids.get(w.as_str()).copied().unwrap_or(UNK_ID)));
            padded.push(EOS_ID);
            for g in padded.windows(order) {
                *counts.entry(g.into()).or_default() += 1;
            }
        }
        let grams = counts
            .into_iter()
            .filter(|(_, c)| *c >= config.min_count)
            .collect();
        Self::build(order, config.discount, words, grams)
    }

    /// From the highest-order counts: the lower orders' adjusted counts,
    /// then every level laid out.
    fn build(order: usize, discount: f32, words: Vec<String>, grams: Vec<(Box<[u32]>, u32)>) -> Self {
        let mut by_length: Vec<Vec<(Box<[u32]>, u32)>> = vec![vec![]; order + 1];
        by_length[order] = grams;
        for n in (1..order).rev() {
            let mut lower: HashMap<Box<[u32]>, u32> = HashMap::new();
            for (g, c) in &by_length[n + 1] {
                // Nothing precedes a sentence's start: count it as seen.
                let add = if g[1] == BOS_ID { *c } else { 1 };
                *lower.entry(g[1..].into()).or_default() += add;
            }
            by_length[n] = lower.into_iter().collect();
        }
        let mut model = Self::empty(order, discount, words);
        for (n, mut grams) in by_length.into_iter().enumerate().skip(1) {
            grams.sort_unstable();
            let k = n - 1;
            let level = &mut model.levels[k];
            level.starts.push(model.next.len() as u32);
            let mut i = 0;
            while i < grams.len() {
                let ctx = &grams[i].0[..k];
                let mut total = 0u32;
                while i < grams.len() && &grams[i].0[..k] == ctx {
                    model.next.push(grams[i].0[k]);
                    model.counts.push(grams[i].1);
                    total += grams[i].1;
                    i += 1;
                }
                level.contexts.extend_from_slice(ctx);
                level.starts.push(model.next.len() as u32);
                level.totals.push(total);
            }
        }
        model.index();
        model
    }

    fn empty(order: usize, discount: f32, words: Vec<String>) -> Self {
        Self {
            order,
            discount,
            ids: HashMap::new(),
            words,
            levels: vec![Level::default(); order],
            next: vec![],
            counts: vec![],
            by_spelling: vec![],
            by_count: vec![],
        }
    }

    /// The lookups derived from the levels: ids, spellings, commonest.
    fn index(&mut self) {
        self.ids = self
            .words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i as u32))
            .collect();
        let words = &self.words;
        let mut by_spelling: Vec<u32> = (EOS_ID + 1..words.len() as u32).collect();
        by_spelling.sort_unstable_by(|&a, &b| words[a as usize].cmp(&words[b as usize]));
        self.by_spelling = by_spelling;
        let mut by_count: Vec<(u32, u32)> = self
            .followers(0, 0)
            .filter(|&(w, _)| w > EOS_ID)
            .collect();
        by_count.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        self.by_count = by_count.into_iter().map(|e| e.0).collect();
    }

    /// Context `c` of level `k`'s followers: (word, count) by word id.
    fn followers(&self, k: usize, c: usize) -> impl Iterator<Item = (u32, u32)> + '_ {
        let range = self.levels[k]
            .starts
            .get(c..c + 2)
            .map_or(0..0, |s| s[0] as usize..s[1] as usize);
        self.next[range.clone()]
            .iter()
            .copied()
            .zip(self.counts[range].iter().copied())
    }

    /// Highest-order n-grams and their counts, word by word, as the saved
    /// file has them; vocabulary and all else follow from them.
    pub fn from_counts<S: AsRef<str>>(
        order: usize,
        discount: f32,
        counts: impl IntoIterator<Item = (Vec<S>, u32)>,
    ) -> Result<Self, String> {
        let order = order.max(1);
        let mut words: Vec<String> = [UNK, BOS, EOS].map(String::from).to_vec();
        let mut ids: HashMap<String, u32> = words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i as u32))
            .collect();
        let mut freq: Vec<u64> = vec![0; 3];
        let mut grams: HashMap<Box<[u32]>, u32> = HashMap::new();
        for (g, c) in counts {
            if g.len() != order {
                return Err(format!("an n-gram of {} words in an order-{order} model", g.len()));
            }
            let g: Box<[u32]> = g
                .iter()
                .map(|w| {
                    let w = w.as_ref();
                    let w = if [UNK, BOS, EOS].contains(&w) { w.to_owned() } else { normalize(w) };
                    let id = *ids.entry(w.clone()).or_insert_with(|| {
                        words.push(w);
                        freq.push(0);
                        (words.len() - 1) as u32
                    });
                    freq[id as usize] += c as u64;
                    id
                })
                .collect();
            *grams.entry(g).or_default() += c;
        }
        // Commonest first, as from a text.
        let mut order_ids: Vec<u32> = (3..words.len() as u32).collect();
        order_ids.sort_by(|&a, &b| freq[b as usize].cmp(&freq[a as usize]).then(words[a as usize].cmp(&words[b as usize])));
        let mut remap: Vec<u32> = (0..words.len() as u32).collect();
        let mut sorted: Vec<String> = words[..3].to_vec();
        for (new, &old) in order_ids.iter().enumerate() {
            remap[old as usize] = new as u32 + 3;
            sorted.push(words[old as usize].clone());
        }
        let grams = grams
            .into_iter()
            .map(|(g, c)| (g.iter().map(|&w| remap[w as usize]).collect(), c))
            .collect();
        Ok(Self::build(order, discount, sorted, grams))
    }

    /// Reads the text form ([module docs](self)).
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut lines = text.lines();
        let header = lines.next().unwrap_or_default();
        let mut order = None;
        let mut discount = 0.75;
        let Some(fields) = header.strip_prefix("#threers-ngram") else {
            return Err("not a threers n-gram file: no `#threers-ngram` line".into());
        };
        for f in fields.split_whitespace() {
            match f.split_once('=') {
                Some(("order", v)) => order = v.parse::<usize>().ok(),
                Some(("discount", v)) => discount = v.parse().map_err(|_| format!("discount `{v}`"))?,
                _ => {}
            }
        }
        let order = order.ok_or("no order in the header")?;
        let mut counts = vec![];
        for (i, line) in lines.enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let (g, c) = line
                .rsplit_once('\t')
                .ok_or_else(|| format!("line {}: no tab before the count", i + 2))?;
            let c: u32 = c
                .trim()
                .parse()
                .map_err(|_| format!("line {}: count `{c}`", i + 2))?;
            counts.push((g.split(' ').collect::<Vec<_>>(), c));
        }
        Self::from_counts(order, discount, counts)
    }

    /// The text form ([module docs](self)): the highest order's counts.
    pub fn to_text(&self) -> String {
        let mut out = format!("#threers-ngram order={} discount={}\n", self.order, self.discount);
        let k = self.order - 1;
        let top = &self.levels[k];
        for c in 0..top.len() {
            let ctx: Vec<&str> = top.contexts[c * k..c * k + k]
                .iter()
                .map(|&w| self.words[w as usize].as_str())
                .collect();
            for (w, n) in self.followers(k, c) {
                for x in &ctx {
                    out.push_str(x);
                    out.push(' ');
                }
                out.push_str(&self.words[w as usize]);
                out.push('\t');
                out.push_str(&n.to_string());
                out.push('\n');
            }
        }
        out
    }

    /// The binary form: every level as it is held, varint-coded (each
    /// context's followers by the difference of their ids), so a load is
    /// one pass with nothing to sort or hash but the words.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        put(&mut out, self.order as u64);
        out.extend_from_slice(&self.discount.to_le_bytes());
        let words = self.words.join("\n");
        put(&mut out, self.words.len() as u64);
        put(&mut out, words.len() as u64);
        out.extend_from_slice(words.as_bytes());
        for (k, level) in self.levels.iter().enumerate() {
            put(&mut out, level.len() as u64);
            for c in 0..level.len() {
                for &w in &level.contexts[c * k..c * k + k] {
                    put(&mut out, w as u64);
                }
                let (lo, hi) = (level.starts[c] as usize, level.starts[c + 1] as usize);
                put(&mut out, (hi - lo) as u64);
                let mut last = 0u32;
                for i in lo..hi {
                    put(&mut out, (self.next[i] - last) as u64);
                    put(&mut out, self.counts[i] as u64);
                    last = self.next[i];
                }
            }
        }
        out
    }

    /// A saved model's bytes, either form.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        match bytes.strip_prefix(MAGIC) {
            Some(rest) => Self::read_binary(rest).ok_or_else(|| "a damaged binary n-gram file".to_owned()),
            None => Self::parse(std::str::from_utf8(bytes).map_err(|e| e.to_string())?),
        }
    }

    fn read_binary(b: &[u8]) -> Option<Self> {
        let mut at = 0usize;
        let order = get(b, &mut at)? as usize;
        if order == 0 || order > 16 {
            return None;
        }
        let discount = f32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?);
        at += 4;
        let n_words = get(b, &mut at)? as usize;
        let len = get(b, &mut at)? as usize;
        let text = std::str::from_utf8(b.get(at..at.checked_add(len)?)?).ok()?;
        at += len;
        let words: Vec<String> = text.split('\n').map(str::to_owned).collect();
        if words.len() != n_words || n_words < 3 {
            return None;
        }
        let mut m = Self::empty(order, discount, words);
        let v = n_words as u64;
        for k in 0..order {
            let n = get(b, &mut at)? as usize;
            // Each context takes a byte at least: a bound before reserving.
            if n > b.len() {
                return None;
            }
            let level = &mut m.levels[k];
            level.contexts.reserve(n * k);
            level.starts.reserve(n + 1);
            level.totals.reserve(n);
            level.starts.push(m.next.len() as u32);
            for _ in 0..n {
                for _ in 0..k {
                    let w = get(b, &mut at)?;
                    (w < v).then_some(())?;
                    level.contexts.push(w as u32);
                }
                let f = get(b, &mut at)? as usize;
                (f <= b.len()).then_some(())?;
                let (mut last, mut total) = (0u64, 0u64);
                for _ in 0..f {
                    last += get(b, &mut at)?;
                    let c = get(b, &mut at)?;
                    (last < v && c <= u32::MAX as u64).then_some(())?;
                    m.next.push(last as u32);
                    m.counts.push(c as u32);
                    total += c;
                }
                level.starts.push(m.next.len() as u32);
                level.totals.push(u32::try_from(total).ok()?);
            }
        }
        (at == b.len()).then_some(())?;
        m.index();
        Some(m)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let p = path.as_ref();
        std::fs::read(p)
            .map_err(|e| format!("{}: {e}", p.display()))
            .and_then(|b| Self::from_bytes(&b))
            .map_err(|e| format!("{}: {e}", p.display()))
    }

    /// Saves the binary form ([`to_bytes`](Self::to_bytes)).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        std::fs::write(path, self.to_bytes()).map_err(|e| e.to_string())
    }

    /// Saves the text form ([`to_text`](Self::to_text)).
    pub fn save_text(&self, path: impl AsRef<Path>) -> Result<(), String> {
        std::fs::write(path, self.to_text()).map_err(|e| e.to_string())
    }

    pub fn order(&self) -> usize {
        self.order
    }

    /// Words known, `<unk>`, `<s>` and `</s>` not counted.
    pub fn vocab_len(&self) -> usize {
        self.words.len() - 3
    }

    fn id(&self, word: &str) -> u32 {
        self.ids.get(word).copied().unwrap_or(UNK_ID)
    }

    /// The last `order − 1` words of `context` as ids, a sentence's start
    /// padded with `<s>` as in training.
    fn context_ids(&self, context: &[&str]) -> Vec<u32> {
        let keep = context.len().min(self.order - 1);
        let mut ids: Vec<u32> = context[context.len() - keep..]
            .iter()
            .map(|w| self.id(w))
            .collect();
        if ids.first() == Some(&BOS_ID) {
            let pad = self.order - 1 - ids.len();
            ids.splice(0..0, std::iter::repeat_n(BOS_ID, pad));
        }
        ids
    }

    /// The suffixes of `ctx` seen, shortest first, found once for every
    /// word scored after them.
    fn found(&self, ctx: &[u32]) -> Vec<Found> {
        let mut out = vec![];
        for k in 0..=ctx.len() {
            match self.levels[k].find(&ctx[ctx.len() - k..]) {
                Some(c) if self.levels[k].totals[c] > 0 => out.push((k, c)),
                _ => break,
            }
        }
        out
    }

    fn prob_found(&self, found: &[Found], w: u32) -> f64 {
        let d = self.discount as f64;
        // Uniform over every word that can follow: all but `<s>`.
        let mut p = 1.0 / (self.words.len() - 1) as f64;
        for &(k, c) in found {
            let level = &self.levels[k];
            let (lo, hi) = (level.starts[c] as usize, level.starts[c + 1] as usize);
            let count = self.next[lo..hi]
                .binary_search(&w)
                .map_or(0, |i| self.counts[lo + i]) as f64;
            p = ((count - d).max(0.0) + d * (hi - lo) as f64 * p) / level.totals[c] as f64;
        }
        p
    }

    #[cfg(test)]
    fn prob(&self, ctx: &[u32], w: u32) -> f64 {
        self.prob_found(&self.found(ctx), w)
    }

    /// Word ids spelled with `prefix`, from the spelling index.
    fn spelled(&self, prefix: &str) -> &[u32] {
        let lo = self
            .by_spelling
            .partition_point(|&w| self.words[w as usize].as_str() < prefix);
        let hi = lo
            + self.by_spelling[lo..].partition_point(|&w| self.words[w as usize].starts_with(prefix));
        &self.by_spelling[lo..hi]
    }
}

/// `v` as a LEB128 varint.
fn put(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push(v as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// The varint at `at`, `at` moved past it.
fn get(b: &[u8], at: &mut usize) -> Option<u64> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = *b.get(*at)?;
        *at += 1;
        v |= ((byte & 0x7f) as u64) << shift;
        if byte < 0x80 {
            return Some(v);
        }
    }
    None
}

/// Candidates from each seen context: its commonest followers, this many.
const PER_CONTEXT: usize = 32;

impl WordModel for NgramModel {
    fn context_len(&self) -> usize {
        self.order - 1
    }

    fn unknown_log_prob(&self, context: &[&str]) -> Option<f32> {
        let found = self.found(&self.context_ids(context));
        Some(self.prob_found(&found, UNK_ID).ln() as f32)
    }

    fn log_probs(&self, context: &[&str], words: &[&str]) -> Vec<Option<f32>> {
        let found = self.found(&self.context_ids(context));
        words
            .iter()
            .map(|w| {
                let id = *self.ids.get(*w)?;
                (id > EOS_ID).then(|| self.prob_found(&found, id).ln() as f32)
            })
            .collect()
    }

    fn predict(&self, context: &[&str], prefix: &str, n: usize) -> Vec<(String, f32)> {
        if n == 0 {
            return vec![];
        }
        let prefix = normalize(prefix);
        let found = self.found(&self.context_ids(context));
        let fits = |w: u32| w > EOS_ID && self.words[w as usize].starts_with(&prefix);
        // The words seen after each longer context (the commonest of them
        // with no letters to go on), then the commonest words.
        let mut cands: Vec<u32> = vec![];
        let keep = PER_CONTEXT.max(4 * n);
        for &(k, c) in found.iter().filter(|f| f.0 > 0) {
            let mut seen: Vec<(u32, u32)> = self.followers(k, c).filter(|&(w, _)| fits(w)).collect();
            if seen.len() > keep {
                seen.select_nth_unstable_by(keep, |a, b| b.1.cmp(&a.1));
                seen.truncate(keep);
            }
            cands.extend(seen.iter().map(|e| e.0));
        }
        if prefix.is_empty() {
            cands.extend(self.by_count.iter().take(keep));
        } else {
            let spelled = self.spelled(&prefix);
            if spelled.len() <= 4096 {
                cands.extend(spelled);
            } else {
                cands.extend(self.by_count.iter().filter(|&&w| fits(w)).take(keep));
            }
        }
        cands.sort_unstable();
        cands.dedup();
        let mut scored: Vec<(u32, f64)> = cands
            .into_iter()
            .map(|w| (w, self.prob_found(&found, w)))
            .collect();
        let best = |a: &(u32, f64), b: &(u32, f64)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
        if scored.len() > n {
            scored.select_nth_unstable_by(n, best);
            scored.truncate(n);
        }
        scored.sort_by(best);
        scored
            .into_iter()
            .map(|(w, p)| (self.words[w as usize].clone(), p.ln() as f32))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = "I am going to the shop. I am going to bed. \
        It is too late. Me too. I want to go to the park. \
        Going to the beach is fun. I am going to swim.";

    fn lm() -> NgramModel {
        NgramModel::train(TEXT, &NgramConfig::default())
    }

    #[test]
    fn sums_to_one() {
        let lm = lm();
        let all: Vec<&str> = lm.words[3..].iter().map(String::as_str).collect();
        for ctx in [&[][..], &["going"], &["am", "going"], &["<s>"], &["never", "seen"]] {
            let p: f64 = lm
                .log_probs(ctx, &all)
                .iter()
                .map(|l| (l.unwrap() as f64).exp())
                .sum::<f64>()
                + lm.prob(&lm.context_ids(ctx), EOS_ID)
                + lm.prob(&lm.context_ids(ctx), UNK_ID);
            assert!((p - 1.0).abs() < 1e-4, "{ctx:?}: {p}");
        }
    }

    #[test]
    fn context_decides() {
        let lm = lm();
        let p = lm.log_probs(&["am", "going"], &["to", "too"]);
        assert!(p[0].unwrap() > p[1].unwrap() + 1.0);
        let p = lm.log_probs(&["me"], &["to", "too"]);
        assert!(p[1].unwrap() > p[0].unwrap());
        assert_eq!(lm.log_probs(&[], &["zebra", "<s>"]), [None, None]);
        let next: Vec<String> = lm.predict(&["going", "to"], "", 2).into_iter().map(|e| e.0).collect();
        assert_eq!(next[0], "the");
        let next = lm.predict(&["<s>"], "", 1);
        assert_eq!(next[0].0, "i", "sentences start with I here");
        let done = lm.predict(&["to", "the"], "pa", 3);
        assert_eq!(done[0].0, "park");
        assert!(done.iter().all(|e| e.0.starts_with("pa")));
    }

    #[test]
    fn saves_and_loads() {
        let lm = lm();
        let back = NgramModel::parse(&lm.to_text()).unwrap();
        assert_eq!(back.vocab_len(), lm.vocab_len());
        for ctx in [&["am", "going"][..], &["<s>"], &[]] {
            assert_eq!(lm.log_probs(ctx, &["to", "the", "late"]), back.log_probs(ctx, &["to", "the", "late"]));
        }
        assert!(NgramModel::parse("the 3\n").is_err());
        assert!(NgramModel::parse("#threers-ngram order=2\na b c\t1\n").is_err());
    }

    #[test]
    fn the_binary_form_is_the_same_model() {
        for order in [1, 2, 3, 4] {
            let lm = NgramModel::train(TEXT, &NgramConfig { order, ..Default::default() });
            let bytes = lm.to_bytes();
            let back = NgramModel::from_bytes(&bytes).unwrap();
            assert_eq!(back.to_text(), lm.to_text(), "order {order}");
            for ctx in [&["am", "going"][..], &["<s>"], &[], &["zzz", "to"]] {
                assert_eq!(back.log_probs(ctx, &["to", "the", "late"]), lm.log_probs(ctx, &["to", "the", "late"]));
                assert_eq!(back.predict(ctx, "", 3), lm.predict(ctx, "", 3));
            }
            assert!(bytes.len() < lm.to_text().len(), "{} vs {}", bytes.len(), lm.to_text().len());
        }
    }

    #[test]
    fn damaged_files_are_refused_not_panicked_on() {
        let bytes = lm().to_bytes();
        for cut in 0..bytes.len() {
            assert!(NgramModel::from_bytes(&bytes[..cut]).is_err(), "cut at {cut}");
        }
        let mut rng = crate::synth::Rng::new(3);
        for _ in 0..2000 {
            let mut b = bytes.clone();
            for _ in 0..1 + rng.next_u64() % 4 {
                let i = 5 + (rng.next_u64() as usize) % (b.len() - 5);
                b[i] = rng.next_u64() as u8;
            }
            if let Ok(m) = NgramModel::from_bytes(&b) {
                // Whatever it read, it answers without panicking.
                let _ = m.predict(&["going", "to"], "", 3);
                let _ = m.predict(&["<s>"], "t", 3);
                let _ = m.to_text();
            }
        }
    }

    #[test]
    fn unigram_and_vocab_limit() {
        let lm = NgramModel::train(TEXT, &NgramConfig { order: 1, vocab: 3, ..Default::default() });
        assert_eq!(lm.vocab_len(), 3);
        assert_eq!(lm.predict(&["anything"], "", 1)[0].0, "to");
    }
}
