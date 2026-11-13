//! Words that change with a number, and which way a language reads.
//! [`plural`] picks the form of a phrase for a count by the language's
//! plural rules (CLDR's, for the languages here): English has one and
//! other, Russian one, few, many, Arabic six. [`Locale`] is an
//! [environment](crate::environment) value; [`direction_of`] says whether
//! its language reads right to left.
//!
//! ```
//! use threers_ui::i18n::{plural, Forms};
//!
//! let en = Forms::new("{} task left").other("{} tasks left");
//! assert_eq!(plural("en", 1, &en), "1 task left");
//! assert_eq!(plural("en", 3, &en), "3 tasks left");
//! let ru = Forms::new("{} задача").few("{} задачи").many("{} задач");
//! assert_eq!(plural("ru", 2, &ru), "2 задачи");
//! assert_eq!(plural("ru", 5, &ru), "5 задач");
//! assert_eq!(plural("ru", 21, &ru), "21 задача");
//! ```

use crate::environment::LayoutDirection;

/// The language (a BCP 47 tag: `en`, `ru-RU`, `ar`), set on a UI or a
/// part of it as an environment value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Locale(pub String);

/// A plural category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    Zero,
    One,
    Two,
    Few,
    Many,
    Other,
}

/// A phrase in each plural form a language has; `{}` is the count. A
/// missing form falls back to `other`, then to `one`.
#[derive(Debug, Clone, Default)]
pub struct Forms {
    pub zero: Option<String>,
    pub one: String,
    pub two: Option<String>,
    pub few: Option<String>,
    pub many: Option<String>,
    pub other: Option<String>,
}

impl Forms {
    /// The `one` form.
    pub fn new(one: impl Into<String>) -> Self {
        Self { one: one.into(), ..Self::default() }
    }
    pub fn zero(mut self, s: impl Into<String>) -> Self {
        self.zero = Some(s.into());
        self
    }
    pub fn two(mut self, s: impl Into<String>) -> Self {
        self.two = Some(s.into());
        self
    }
    pub fn few(mut self, s: impl Into<String>) -> Self {
        self.few = Some(s.into());
        self
    }
    pub fn many(mut self, s: impl Into<String>) -> Self {
        self.many = Some(s.into());
        self
    }
    pub fn other(mut self, s: impl Into<String>) -> Self {
        self.other = Some(s.into());
        self
    }

    fn get(&self, c: Category) -> &str {
        let other = self.other.as_deref().unwrap_or(&self.one);
        match c {
            Category::Zero => self.zero.as_deref().unwrap_or(other),
            Category::One => &self.one,
            Category::Two => self.two.as_deref().unwrap_or(other),
            Category::Few => self.few.as_deref().unwrap_or(other),
            Category::Many => self.many.as_deref().unwrap_or(other),
            Category::Other => other,
        }
    }
}

fn language(locale: &str) -> String {
    locale.split(['-', '_']).next().unwrap_or("").to_lowercase()
}

/// The plural category of a whole number `n` in `locale`.
pub fn category(locale: &str, n: u64) -> Category {
    let (m10, m100) = (n % 10, n % 100);
    match language(locale).as_str() {
        // No plurals.
        "ja" | "zh" | "ko" | "vi" | "th" | "id" | "ms" => Category::Other,
        "fr" | "pt" => {
            if n <= 1 {
                Category::One
            } else {
                Category::Other
            }
        }
        "ru" | "uk" | "be" => {
            if m10 == 1 && m100 != 11 {
                Category::One
            } else if (2..=4).contains(&m10) && !(12..=14).contains(&m100) {
                Category::Few
            } else {
                Category::Many
            }
        }
        "pl" => {
            if n == 1 {
                Category::One
            } else if (2..=4).contains(&m10) && !(12..=14).contains(&m100) {
                Category::Few
            } else {
                Category::Many
            }
        }
        "cs" | "sk" => match n {
            1 => Category::One,
            2..=4 => Category::Few,
            _ => Category::Other,
        },
        "ar" => match (n, m100) {
            (0, _) => Category::Zero,
            (1, _) => Category::One,
            (2, _) => Category::Two,
            (_, 3..=10) => Category::Few,
            (_, 11..=99) => Category::Many,
            _ => Category::Other,
        },
        "he" => match n {
            1 => Category::One,
            2 => Category::Two,
            _ => Category::Other,
        },
        // English, German, Spanish, Italian, Dutch, Swedish and the like.
        _ => {
            if n == 1 {
                Category::One
            } else {
                Category::Other
            }
        }
    }
}

/// The form of `forms` for `n` in `locale`, with `{}` replaced by `n`.
pub fn plural(locale: &str, n: u64, forms: &Forms) -> String {
    forms.get(category(locale, n)).replace("{}", &n.to_string())
}

/// Which way `locale`'s language reads.
pub fn direction_of(locale: &str) -> LayoutDirection {
    match language(locale).as_str() {
        "ar" | "he" | "fa" | "ur" | "ps" | "yi" | "dv" | "ku" | "sd" | "ug" => LayoutDirection::RightToLeft,
        _ => LayoutDirection::LeftToRight,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categories_follow_each_languages_rules() {
        assert_eq!([0, 1, 2, 5].map(|n| category("en-US", n)), [Category::Other, Category::One, Category::Other, Category::Other]);
        assert_eq!([0, 1, 2].map(|n| category("fr", n)), [Category::One, Category::One, Category::Other]);
        assert_eq!([1, 11, 22, 25, 111].map(|n| category("uk", n)), [Category::One, Category::Many, Category::Few, Category::Many, Category::Many]);
        assert_eq!([1, 22, 12, 5].map(|n| category("pl", n)), [Category::One, Category::Few, Category::Many, Category::Many]);
        assert_eq!([0, 1, 2, 3, 11, 100].map(|n| category("ar", n)), [Category::Zero, Category::One, Category::Two, Category::Few, Category::Many, Category::Other]);
        assert_eq!(category("ja", 1), Category::Other);
        assert_eq!(direction_of("he-IL"), LayoutDirection::RightToLeft);
        assert_eq!(direction_of("en"), LayoutDirection::LeftToRight);
    }
}
