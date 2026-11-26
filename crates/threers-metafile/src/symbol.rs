//! The Symbol font's encoding, and Unicode.
//!
//! Text set in Symbol (an Equation Editor formula's Greek letters,
//! operators and the pieces its tall brackets are built of) is stored in
//! the font's own encoding: `p` is π, `æ` the top of a tall `(`, `@` is ≅.
//! A font other than Symbol draws those bytes as what they say in Latin-1,
//! so they are made the characters they stand for (the table is Unicode's
//! `SYMBOL.TXT`; the private-use extenders as their nearest characters).
//! Characters in the private-use block a symbol font's Unicode text uses
//! (`U+F020`–`U+F0FF`) are taken as the byte below them.

/// Whether `family` is the Symbol font, whose text is in its own encoding.
pub fn is_symbol(family: &str) -> bool {
    family.trim().eq_ignore_ascii_case("symbol")
}

/// Bytes `0x20`–`0xFF` of the Symbol encoding (`'\0'` where it has none).
const TABLE: [char; 224] = [
    // 0x20
    ' ', '!', '∀', '#', '∃', '%', '&', '∋', '(', ')', '∗', '+', ',', '−', '.', '/',
    // 0x30
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?',
    // 0x40
    '≅', 'Α', 'Β', 'Χ', 'Δ', 'Ε', 'Φ', 'Γ', 'Η', 'Ι', 'ϑ', 'Κ', 'Λ', 'Μ', 'Ν', 'Ο',
    // 0x50
    'Π', 'Θ', 'Ρ', 'Σ', 'Τ', 'Υ', 'ς', 'Ω', 'Ξ', 'Ψ', 'Ζ', '[', '∴', ']', '⊥', '_',
    // 0x60
    '‾', 'α', 'β', 'χ', 'δ', 'ε', 'φ', 'γ', 'η', 'ι', 'ϕ', 'κ', 'λ', 'μ', 'ν', 'ο',
    // 0x70
    'π', 'θ', 'ρ', 'σ', 'τ', 'υ', 'ϖ', 'ω', 'ξ', 'ψ', 'ζ', '{', '|', '}', '∼', '\0',
    // 0x80
    '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
    // 0x90
    '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
    // 0xA0
    '€', 'ϒ', '′', '≤', '⁄', '∞', 'ƒ', '♣', '♦', '♥', '♠', '↔', '←', '↑', '→', '↓',
    // 0xB0
    '°', '±', '″', '≥', '×', '∝', '∂', '•', '÷', '≠', '≡', '≈', '…', '⏐', '⎯', '↵',
    // 0xC0
    'ℵ', 'ℑ', 'ℜ', '℘', '⊗', '⊕', '∅', '∩', '∪', '⊃', '⊇', '⊄', '⊂', '⊆', '∈', '∉',
    // 0xD0
    '∠', '∇', '®', '©', '™', '∏', '√', '⋅', '¬', '∧', '∨', '⇔', '⇐', '⇑', '⇒', '⇓',
    // 0xE0
    '◊', '〈', '®', '©', '™', '∑', '⎛', '⎜', '⎝', '⎡', '⎢', '⎣', '⎧', '⎨', '⎩', '⎪',
    // 0xF0
    '\0', '〉', '∫', '⌠', '⎮', '⌡', '⎞', '⎟', '⎠', '⎤', '⎥', '⎦', '⎫', '⎬', '⎭', '\0',
];

/// One character of Symbol-encoded text as what it stands for.
pub fn char_to_unicode(c: char) -> char {
    let code = c as u32;
    // a symbol font's private-use Unicode: the byte below
    let code = if (0xF020..=0xF0FF).contains(&code) { code - 0xF000 } else { code };
    match code {
        0x20..=0xFF => match TABLE[(code - 0x20) as usize] {
            '\0' => c,
            u => u,
        },
        _ => c,
    }
}

/// Symbol-encoded text as the characters it stands for.
pub fn to_unicode(text: &str) -> String {
    text.chars().map(char_to_unicode).collect()
}

/// Text back in the Symbol encoding (for a PDF's Symbol font): each
/// character's byte, `?` for one the font lacks.
pub fn from_unicode(text: &str) -> Vec<u8> {
    text.chars()
        .map(|c| match TABLE.iter().position(|&t| t == c && t != '\0') {
            Some(i) => (i + 0x20) as u8,
            None if (c as u32) < 0x80 => c as u8,
            None => b'?',
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_equations_symbols_are_the_characters_they_stand_for() {
        assert_eq!(to_unicode("p"), "π");
        assert_eq!(to_unicode("2pl"), "2πλ");
        assert_eq!(to_unicode("@"), "≅");
        // a tall bracket's pieces: top, extender, bottom
        assert_eq!(to_unicode("\u{E6}\u{E7}\u{E8}"), "⎛⎜⎝");
        assert_eq!(to_unicode("\u{F6}\u{F7}\u{F8}"), "⎞⎟⎠");
        assert_eq!(to_unicode("\u{E5}\u{F2}\u{B4}\u{B6}"), "∑∫×∂");
        // a symbol font's private-use Unicode too
        assert_eq!(to_unicode("\u{F070}"), "π");
        assert!(is_symbol(" SYMBOL"));
        assert!(!is_symbol("Times New Roman"));
    }

    #[test]
    fn back_to_the_fonts_bytes() {
        assert_eq!(from_unicode("2πλ≅⎛"), b"2pl@\xE6");
        assert_eq!(from_unicode(&to_unicode("\u{E5}x\u{F2}")), b"\xE5x\xF2");
    }
}
