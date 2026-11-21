//! Words from the words before them: what the bar offers next, and what
//! settles a swipe the keys cannot ("I'm going **to**", not "too").
//!
//! A [`WordModel`] gives `log p(word | the words before it)`. Two come
//! with the crate, each behind its own feature: an n-gram model
//! (`ngram`, feature `ngram`) and a small neural one run
//! by RLX (`neural`, feature `rlx`). Either works alone;
//! [`Interpolated`] mixes them. Neither is the swipe model: that one reads
//! the finger, these read the sentence.
//!
//! Words are lower case; a sentence starts with [`BOS`], so the context of
//! a sentence's first word is `["<s>"]`.

/// Before a sentence's first word.
pub const BOS: &str = "<s>";
/// After its last.
pub const EOS: &str = "</s>";
/// Any word out of the vocabulary.
pub const UNK: &str = "<unk>";

/// `log p(word | context)` for the words of a language.
pub trait WordModel {
    /// How many words before the next one it reads (n − 1 for an n-gram).
    fn context_len(&self) -> usize;

    /// `log p(w | context)` for each of `words`, `None` for a word it does
    /// not know. `context` is oldest first, its last word the one just
    /// before; only its last [`context_len`](Self::context_len) words
    /// count.
    fn log_probs(&self, context: &[&str], words: &[&str]) -> Vec<Option<f32>>;

    /// The likeliest `n` words to follow `context` that start with
    /// `prefix` (any, if empty), best first, with their log-probability.
    fn predict(&self, context: &[&str], prefix: &str, n: usize) -> Vec<(String, f32)>;

    /// `log p(some word it does not know | context)`, all of them
    /// together; `None` if it cannot say. A decoder shares it among the
    /// unknown words by their frequency.
    fn unknown_log_prob(&self, context: &[&str]) -> Option<f32> {
        let _ = context;
        None
    }
}

/// A model behind a pointer is a model: one [`Rc`](std::rc::Rc) shared
/// alone, in an [`Interpolated`] and in a keyboard; a [`Box`] of any of
/// them chosen at run time.
macro_rules! word_model_through {
    ($($ptr:ident)::+) => {
        impl<T: WordModel + ?Sized> WordModel for $($ptr)::+<T> {
            fn context_len(&self) -> usize {
                (**self).context_len()
            }

            fn log_probs(&self, context: &[&str], words: &[&str]) -> Vec<Option<f32>> {
                (**self).log_probs(context, words)
            }

            fn predict(&self, context: &[&str], prefix: &str, n: usize) -> Vec<(String, f32)> {
                (**self).predict(context, prefix, n)
            }

            fn unknown_log_prob(&self, context: &[&str]) -> Option<f32> {
                (**self).unknown_log_prob(context)
            }
        }
    };
}

word_model_through!(std::rc::Rc);
word_model_through!(Box);

/// A word's letters as the models store it: lower case, `’` as `'`.
pub fn normalize(word: &str) -> String {
    word.to_lowercase().replace('’', "'")
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '\'' || c == '’'
}

fn ends_sentence(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '\n' | '…')
}

/// `text`'s sentences as words, lower case; a sentence ends at `.`, `!`,
/// `?`, `…` or a line break.
pub fn sentences(text: &str) -> Vec<Vec<String>> {
    let mut out = vec![];
    let mut sentence = vec![];
    let mut word = String::new();
    for c in text.chars() {
        if is_word_char(c) {
            word.push(c);
            continue;
        }
        if !word.is_empty() {
            sentence.push(normalize(word.trim_matches(['\'', '’'])));
            if sentence.last().is_some_and(String::is_empty) {
                sentence.pop();
            }
            word.clear();
        }
        if ends_sentence(c) && !sentence.is_empty() {
            out.push(std::mem::take(&mut sentence));
        }
    }
    if !word.is_empty() {
        let w = normalize(word.trim_matches(['\'', '’']));
        if !w.is_empty() {
            sentence.push(w);
        }
    }
    if !sentence.is_empty() {
        out.push(sentence);
    }
    out
}

/// Bytes of the word the caret ends (the one being typed) in `before`.
pub(crate) fn typing_len(before: &str) -> usize {
    before.len() - word_start(before)
}

/// Where the word the caret ends (the one being typed) starts in `before`.
fn word_start(before: &str) -> usize {
    before
        .char_indices()
        .rev()
        .find(|(_, c)| !is_word_char(*c))
        .map_or(0, |(i, c)| i + c.len_utf8())
}

/// The text before the caret split for a model: the sentence's words
/// before the one being typed, after [`BOS`] if the sentence starts in
/// `before`, at most `max` of them; and the word being typed (empty after
/// a space).
pub fn split_context(before: &str, max: usize) -> (Vec<String>, String) {
    let start = word_start(before);
    let typing = normalize(&before[start..]);
    let head = &before[..start];
    // Back to where the sentence began, or as far as `max` words need.
    let mut words: Vec<String> = vec![];
    let mut word = String::new();
    let mut began = true;
    for c in head.chars().rev() {
        if is_word_char(c) {
            word.insert(0, c);
            continue;
        }
        if !word.is_empty() {
            let w = normalize(word.trim_matches(['\'', '’']));
            if !w.is_empty() {
                words.push(w);
            }
            word.clear();
            if words.len() > max {
                began = false;
                break;
            }
        }
        if ends_sentence(c) {
            break;
        }
    }
    if !word.is_empty() {
        let w = normalize(word.trim_matches(['\'', '’']));
        if !w.is_empty() {
            words.push(w);
        }
    }
    if began {
        words.push(BOS.to_owned());
    }
    words.reverse();
    let skip = words.len().saturating_sub(max);
    (words.split_off(skip), typing)
}

/// `log p(w | context)` for each of `words`, a word `model` does not know
/// given its share of the unknown words' probability by `frequency`
/// (`log p(w)` among a word list's words), or that frequency alone when
/// the model cannot say how likely an unknown word is.
pub fn log_probs_or(
    model: &dyn WordModel,
    context: &[&str],
    words: &[&str],
    frequency: impl Fn(&str) -> f32,
) -> Vec<f32> {
    let lps = model.log_probs(context, words);
    let unknown = lps
        .iter()
        .any(Option::is_none)
        .then(|| model.unknown_log_prob(context))
        .flatten();
    lps.into_iter()
        .zip(words)
        .map(|(lp, w)| lp.unwrap_or_else(|| unknown.unwrap_or(0.0) + frequency(w)))
        .collect()
}

/// Words for the bar from the text before the caret: completions of the
/// word being typed, or the next word after a space.
pub fn suggest(model: &dyn WordModel, before: &str, n: usize) -> Vec<String> {
    // Right after punctuation (no space yet) a word would run into it.
    if before
        .chars()
        .last()
        .is_some_and(|c| !c.is_whitespace() && !is_word_char(c))
    {
        return vec![];
    }
    let (context, typing) = split_context(before, model.context_len());
    let first = context == [BOS];
    let typed = &before[word_start(before)..];
    let context: Vec<&str> = context.iter().map(String::as_str).collect();
    let mut out: Vec<String> = vec![];
    for (w, _) in model.predict(&context, &typing, n) {
        let w = cased(&w, typed, first);
        if !out.contains(&w) {
            out.push(w);
        }
    }
    out
}

/// `word` (as the models keep it, lower case) as it should be typed: "I"
/// and its contractions capital, the first word of a sentence capitalised,
/// and a completion in the case of the letters `typed` so far ("Th" →
/// "The", "NA" → "NASA").
pub fn cased(word: &str, typed: &str, sentence_start: bool) -> String {
    let capital = |w: &str| {
        let mut c = w.chars();
        c.next()
            .map_or_else(String::new, |f| f.to_uppercase().collect::<String>() + c.as_str())
    };
    let mut letters = typed.chars().filter(|c| c.is_alphabetic());
    let shouting = typed.chars().filter(|c| c.is_alphabetic()).count() > 1
        && typed.chars().filter(|c| c.is_alphabetic()).all(char::is_uppercase);
    if shouting {
        return word.to_uppercase();
    }
    let is_i = word == "i" || word.strip_prefix("i'").is_some_and(|r| ["m", "ll", "d", "ve"].contains(&r));
    if is_i || sentence_start || letters.next().is_some_and(char::is_uppercase) {
        capital(word)
    } else {
        word.to_owned()
    }
}

/// Several models as one: `p = Σ wᵢ pᵢ / Σ wᵢ` (linear interpolation), a
/// model that does not know the word giving it nothing, so a word only
/// one knows keeps only that one's share.
pub struct Interpolated {
    models: Vec<(Box<dyn WordModel>, f32)>,
}

impl Interpolated {
    pub fn new() -> Self {
        Self { models: vec![] }
    }

    /// `model`, weighted `weight` against the others.
    pub fn with(mut self, model: impl WordModel + 'static, weight: f32) -> Self {
        self.models.push((Box::new(model), weight.max(0.0)));
        self
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

impl Default for Interpolated {
    fn default() -> Self {
        Self::new()
    }
}

impl WordModel for Interpolated {
    fn context_len(&self) -> usize {
        self.models
            .iter()
            .map(|(m, _)| m.context_len())
            .max()
            .unwrap_or(0)
    }

    fn log_probs(&self, context: &[&str], words: &[&str]) -> Vec<Option<f32>> {
        let total: f64 = self.models.iter().map(|(_, w)| *w as f64).sum();
        let mut sum = vec![0.0f64; words.len()];
        for (m, w) in &self.models {
            for (s, lp) in sum.iter_mut().zip(m.log_probs(context, words)) {
                if let Some(lp) = lp {
                    *s += *w as f64 * (lp as f64).exp();
                }
            }
        }
        sum.into_iter()
            .map(|p| (total > 0.0 && p > 0.0).then(|| (p / total).ln() as f32))
            .collect()
    }

    fn unknown_log_prob(&self, context: &[&str]) -> Option<f32> {
        let (mut p, mut w) = (0.0f64, 0.0f64);
        for (m, weight) in &self.models {
            if let Some(lp) = m.unknown_log_prob(context) {
                p += *weight as f64 * (lp as f64).exp();
                w += *weight as f64;
            }
        }
        (w > 0.0 && p > 0.0).then(|| (p / w).ln() as f32)
    }

    fn predict(&self, context: &[&str], prefix: &str, n: usize) -> Vec<(String, f32)> {
        let mut words: Vec<String> = vec![];
        for (m, _) in &self.models {
            for (w, _) in m.predict(context, prefix, 2 * n) {
                if !words.contains(&w) {
                    words.push(w);
                }
            }
        }
        let refs: Vec<&str> = words.iter().map(String::as_str).collect();
        let lps = self.log_probs(context, &refs);
        let mut out: Vec<(String, f32)> = words
            .into_iter()
            .zip(lps)
            .filter_map(|(w, lp)| Some((w, lp?)))
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        out.truncate(n);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentences_split_and_normalise() {
        let s = sentences("Hello there. I'm here!\nIt’s 'quoted' ok");
        assert_eq!(
            s,
            vec![
                vec!["hello", "there"],
                vec!["i'm", "here"],
                vec!["it's", "quoted", "ok"],
            ]
        );
    }

    #[test]
    fn context_before_the_caret() {
        let (c, t) = split_context("Fine. See you to", 3);
        assert_eq!((c, t.as_str()), (vec![BOS.to_owned(), "see".into(), "you".into()], "to"));
        let (c, t) = split_context("one two three four ", 2);
        assert_eq!((c, t.as_str()), (vec!["three".to_owned(), "four".into()], ""));
        let (c, _) = split_context("", 2);
        assert_eq!(c, vec![BOS]);
        let (c, _) = split_context("a b ", 3);
        assert_eq!(c, vec![BOS, "a", "b"]);
    }

    struct Fixed(Vec<(&'static str, f32)>);

    impl WordModel for Fixed {
        fn context_len(&self) -> usize {
            1
        }
        fn log_probs(&self, _: &[&str], words: &[&str]) -> Vec<Option<f32>> {
            words
                .iter()
                .map(|w| self.0.iter().find(|(x, _)| x == w).map(|(_, p)| p.ln()))
                .collect()
        }
        fn predict(&self, _: &[&str], prefix: &str, n: usize) -> Vec<(String, f32)> {
            let mut v: Vec<_> = self
                .0
                .iter()
                .filter(|(w, _)| w.starts_with(prefix))
                .map(|(w, p)| (w.to_string(), p.ln()))
                .collect();
            v.sort_by(|a, b| b.1.total_cmp(&a.1));
            v.truncate(n);
            v
        }
    }

    #[test]
    fn unknown_words_share_the_unknown_mass() {
        struct Unk(Fixed);
        impl WordModel for Unk {
            fn context_len(&self) -> usize {
                1
            }
            fn log_probs(&self, c: &[&str], w: &[&str]) -> Vec<Option<f32>> {
                self.0.log_probs(c, w)
            }
            fn predict(&self, c: &[&str], p: &str, n: usize) -> Vec<(String, f32)> {
                self.0.predict(c, p, n)
            }
            fn unknown_log_prob(&self, _: &[&str]) -> Option<f32> {
                Some(0.1f32.ln())
            }
        }
        let freq = |w: &str| if w == "zed" { 0.5f32.ln() } else { 0.25f32.ln() };
        let plain = log_probs_or(&Fixed(vec![("a", 0.9)]), &[], &["a", "zed"], freq);
        assert_eq!(plain, [0.9f32.ln(), 0.5f32.ln()], "the frequency alone");
        let m = Unk(Fixed(vec![("a", 0.9)]));
        let lp = log_probs_or(&m, &[], &["a", "zed", "qua"], freq);
        assert!((lp[1].exp() - 0.05).abs() < 1e-6 && (lp[2].exp() - 0.025).abs() < 1e-6);
        let mixed = Interpolated::new().with(m, 1.0).with(Fixed(vec![]), 1.0);
        assert!((mixed.unknown_log_prob(&[]).unwrap().exp() - 0.1).abs() < 1e-6, "only the model that can say");
    }

    #[test]
    fn suggestions_are_cased_as_typed() {
        assert_eq!(cased("i'm", "", false), "I'm");
        assert_eq!(cased("is", "", false), "is", "only I itself");
        assert_eq!(cased("the", "", true), "The");
        assert_eq!(cased("the", "Th", false), "The");
        assert_eq!(cased("nasa", "NA", false), "NASA");
        assert_eq!(cased("the", "th", false), "the");
        let m = Fixed(vec![("the", 0.5), ("i", 0.3), ("then", 0.2)]);
        assert_eq!(suggest(&m, "", 3), ["The", "I", "Then"], "a sentence's start");
        assert_eq!(suggest(&m, "Fine. ", 3), ["The", "I", "Then"]);
        assert_eq!(suggest(&m, "so ", 3), ["the", "I", "then"]);
        assert_eq!(suggest(&m, "so Th", 3), ["The", "Then"]);
    }

    #[test]
    fn interpolation_mixes_probabilities() {
        let m = Interpolated::new()
            .with(Fixed(vec![("a", 0.8), ("b", 0.2)]), 1.0)
            .with(Fixed(vec![("a", 0.2), ("b", 0.4), ("c", 0.4)]), 1.0);
        let lp = m.log_probs(&[], &["a", "b", "c", "d"]);
        assert!((lp[0].unwrap().exp() - 0.5).abs() < 1e-6);
        assert!((lp[1].unwrap().exp() - 0.3).abs() < 1e-6);
        assert!((lp[2].unwrap().exp() - 0.2).abs() < 1e-6, "only the share of the model knowing it");
        assert_eq!(lp[3], None);
        let p: Vec<String> = m.predict(&[], "", 2).into_iter().map(|x| x.0).collect();
        assert_eq!(p, ["a", "b"], "c's 0.4 is half of it: 0.2 < b's 0.3");
        assert_eq!(suggest(&m, "x.", 3), Vec::<String>::new(), "no word glued to a period");
        assert_eq!(suggest(&m, "x b", 3), ["b"]);
    }
}
