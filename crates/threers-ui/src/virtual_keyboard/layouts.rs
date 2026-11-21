//! Languages' letters, and what holding a key offers.

/// The letters of a keyboard, three rows of them, and its name.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyboardLayout {
    /// Shown on the space bar and said to assistive technology.
    pub name: String,
    /// A language tag it is for (`en`, `de`, …).
    pub language: String,
    /// Its letters, row by row, lower case.
    pub rows: [String; 3],
    /// Letters with accents the three rows leave out, as a row over them
    /// (French's ê, ô, ï…); empty for none.
    pub accents: String,
}

impl KeyboardLayout {
    pub fn new(name: &str, language: &str, rows: [&str; 3]) -> Self {
        Self {
            name: name.into(),
            language: language.into(),
            rows: rows.map(String::from),
            accents: String::new(),
        }
    }

    /// With a row of accented letters over the others.
    pub fn with_accents(mut self, accents: &str) -> Self {
        self.accents = accents.into();
        self
    }

    pub fn english() -> Self {
        Self::new("English", "en", ["qwertyuiop", "asdfghjkl", "zxcvbnm"])
    }
    pub fn spanish() -> Self {
        Self::new("Español", "es", ["qwertyuiop", "asdfghjklñ", "zxcvbnm"])
    }
    pub fn german() -> Self {
        Self::new("Deutsch", "de", ["qwertzuiopü", "asdfghjklöä", "yxcvbnmß"])
    }
    pub fn french() -> Self {
        Self::new("Français", "fr", ["azertyuiop", "qsdfghjklm", "wxcvbnéèàç"])
            .with_accents("êëâùûüôîïœ«»")
    }
    pub fn russian() -> Self {
        Self::new("Русский", "ru", ["йцукенгшщзх", "фывапролджэ", "ячсмитьбю"])
    }
    pub fn ukrainian() -> Self {
        Self::new(
            "Українська",
            "uk",
            ["йцукенгшщзхї", "фівапролджє", "ячсмитьбюґ"],
        )
    }
    pub fn greek() -> Self {
        Self::new("Ελληνικά", "el", ["ςερτυθιοπ", "ασδφγηξκλ", "ζχψωβνμ"])
    }

    /// Those built in.
    pub fn all() -> Vec<Self> {
        vec![
            Self::english(),
            Self::spanish(),
            Self::german(),
            Self::french(),
            Self::russian(),
            Self::ukrainian(),
            Self::greek(),
        ]
    }

    /// The one for `locale`'s language (English for one not built in).
    pub fn for_locale(locale: &str) -> Self {
        let lang = locale.split(['-', '_']).next().unwrap_or("").to_lowercase();
        Self::all()
            .into_iter()
            .find(|l| l.language == lang)
            .unwrap_or_else(Self::english)
    }
}

/// What holding a key offers on Apple's keyboards: accented letters, and
/// dashes, quotes and currencies by their plain forms. Upper case for an
/// upper-case letter. Empty for a key with none.
pub fn alternates_for(key: &str) -> Vec<String> {
    let lower = key.to_lowercase();
    let found = match lower.as_str() {
        "a" => "à á â ä æ ã å ā",
        "c" => "ç ć č",
        "e" => "è é ê ë ē ė ę",
        "i" => "î ï í ī į ì",
        "l" => "ł",
        "n" => "ñ ń",
        "o" => "ô ö ò ó œ ø ō õ",
        "s" => "ß ś š",
        "u" => "û ü ù ú ū",
        "y" => "ÿ",
        "z" => "ž ź ż",
        "е" => "ё",
        "ь" => "ъ",
        "-" => "– — •",
        "/" => "\\",
        "$" => "₽ ¥ € ¢ £ ₩",
        "&" => "§",
        "." => "…",
        "?" => "¿",
        "!" => "¡",
        "'" => "’ ‘ `",
        "\"" => "” “ „ » «",
        "0" => "°",
        "%" => "‰",
        _ => "",
    };
    let upper = key != lower;
    found
        .split_whitespace()
        .map(|a| {
            // ß has no capital on the keyboard.
            if upper && a != "ß" {
                a.to_uppercase()
            } else {
                a.to_owned()
            }
        })
        .collect()
}
