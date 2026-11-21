//! What the text before the caret tells the keyboard: to start a sentence
//! with a capital, that two spaces after a word end the sentence, which
//! word is being typed (for suggestions). The keyboard learns the text from
//! [`VirtualKeyboard::set_context`](super::VirtualKeyboard::set_context);
//! without it, none of this happens.
//!
//! ```
//! use threers_ui::virtual_keyboard::{AutoCapitalize, current_word, wants_capital};
//!
//! assert!(wants_capital(AutoCapitalize::Sentences, ""));
//! assert!(wants_capital(AutoCapitalize::Sentences, "Done. "));
//! assert!(!wants_capital(AutoCapitalize::Sentences, "Done, "));
//! assert!(wants_capital(AutoCapitalize::Words, "New "));
//! assert_eq!(current_word("I said hel"), "hel");
//! ```

/// When Shift comes on by itself, as `UITextAutocapitalizationType` has it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AutoCapitalize {
    /// Never.
    None,
    /// At the start, and after `.`, `!` or `?` and a space, or a new line.
    #[default]
    Sentences,
    /// At the start of every word.
    Words,
    /// Always.
    All,
}

/// Whether the next letter after `before` (the text before the caret) is
/// a capital.
pub fn wants_capital(mode: AutoCapitalize, before: &str) -> bool {
    match mode {
        AutoCapitalize::None => false,
        AutoCapitalize::All => true,
        AutoCapitalize::Words => before.is_empty() || before.ends_with(char::is_whitespace),
        AutoCapitalize::Sentences => {
            if before.is_empty() || before.ends_with('\n') {
                return true;
            }
            if !before.ends_with(char::is_whitespace) {
                return false;
            }
            let text = before.trim_end();
            text.is_empty() || text.ends_with(['.', '!', '?', '…', '¡', '¿'])
        }
    }
}

/// The word the caret is at the end of: the letters, digits and
/// apostrophes after the last space or punctuation.
pub fn current_word(before: &str) -> &str {
    let start = before
        .char_indices()
        .rev()
        .find(|(_, c)| !(c.is_alphanumeric() || *c == '\'' || *c == '’'))
        .map_or(0, |(i, c)| i + c.len_utf8());
    &before[start..]
}

/// Whether a space typed after `before` should end the sentence instead:
/// `before` ends with a word and one space, as Apple's “.” shortcut has it.
pub fn period_shortcut(before: &str) -> bool {
    let Some(rest) = before.strip_suffix(' ') else {
        return false;
    };
    rest.chars().last().is_some_and(char::is_alphanumeric)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentences_start_with_capitals() {
        let s = AutoCapitalize::Sentences;
        for yes in ["", "Hi. ", "Hi!  ", "Why? ", "one\n", "   "] {
            assert!(wants_capital(s, yes), "{yes:?}");
        }
        for no in ["H", "Hi ", "Hi.", "e.g, "] {
            assert!(!wants_capital(s, no), "{no:?}");
        }
        assert!(!wants_capital(AutoCapitalize::None, ""));
        assert!(wants_capital(AutoCapitalize::All, "abc"));
    }

    #[test]
    fn the_word_typed_and_the_period_shortcut() {
        assert_eq!(current_word(""), "");
        assert_eq!(current_word("an apple"), "apple");
        assert_eq!(current_word("don't"), "don't");
        assert_eq!(current_word("end. Wö"), "Wö");
        assert_eq!(current_word("end "), "");
        assert!(period_shortcut("word "));
        assert!(!period_shortcut("word.  "));
        assert!(!period_shortcut("word"));
        assert!(!period_shortcut(" "));
    }
}
