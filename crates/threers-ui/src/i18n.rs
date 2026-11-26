//! An interface in the reader's language: words that change with a
//! number, which way a language reads, how it writes numbers and dates,
//! which languages the user reads, and threers' own words in them.
//!
//! [`plural`] picks the form of a phrase for a count by the language's
//! plural rules (CLDR's, for the languages here): English has one and
//! other, Russian one, few, many, Arabic six. [`Locale`] is an
//! [environment](crate::environment) value; [`direction_of`] says whether
//! its language reads right to left. [`format_decimal`] and
//! [`format_date_time`] write numbers and dates as the language does.
//! [`system_languages`] are the user's, and [`best_match`] picks a
//! translation from them. [`set_language`] puts threers' own [`word`]s —
//! the menu over selected text, Back, the ✕ buttons' names — in it.
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

impl Category {
    /// CLDR's name for it: `zero`, `one`, `two`, `few`, `many`, `other` —
    /// the suffix a translation table keeps each form under.
    pub fn as_str(self) -> &'static str {
        match self {
            Category::Zero => "zero",
            Category::One => "one",
            Category::Two => "two",
            Category::Few => "few",
            Category::Many => "many",
            Category::Other => "other",
        }
    }
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
        Self {
            one: one.into(),
            ..Self::default()
        }
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

/// `ja` of `ja-JP`, lower case.
pub(crate) fn base_of(locale: &str) -> String {
    locale.split(['-', '_']).next().unwrap_or("").to_lowercase()
}

/// The plural category of a whole number `n` in `locale`.
pub fn category(locale: &str, n: u64) -> Category {
    let (m10, m100) = (n % 10, n % 100);
    match base_of(locale).as_str() {
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
    match base_of(locale).as_str() {
        "ar" | "he" | "fa" | "ur" | "ps" | "yi" | "dv" | "ku" | "sd" | "ug" => {
            LayoutDirection::RightToLeft
        }
        _ => LayoutDirection::LeftToRight,
    }
}

// ── The language threers' own words are in ──────────────────────────────

static LANGUAGE: std::sync::RwLock<String> = std::sync::RwLock::new(String::new());

type Translator = fn(Word, &str) -> Option<String>;
static TRANSLATOR: std::sync::RwLock<Option<Translator>> = std::sync::RwLock::new(None);

/// Sets the language (a BCP 47 tag) threers' own words are shown in: the
/// menu over selected text, a navigation stack's Back, the names a screen
/// reader gives its ✕ buttons (see [`word`]). English until set; an app
/// shown in another language sets its own here, as
/// [`best_match`]`(…, &`[`system_languages`]`())` or the user's choice.
pub fn set_language(tag: &str) {
    if let Ok(mut l) = LANGUAGE.write() {
        *l = tag.to_string();
    }
}

/// The language [`set_language`] set; `en` until then.
pub fn language() -> String {
    LANGUAGE
        .read()
        .ok()
        .filter(|l| !l.is_empty())
        .map_or_else(|| "en".into(), |l| l.clone())
}

/// Words threers' widgets show or say themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Word {
    Cut,
    Copy,
    Paste,
    Select,
    SelectAll,
    LookUp,
    /// A search field's ✕, emptying it.
    Clear,
    /// A notice's ✕.
    Close,
    /// A navigation stack's way back.
    Back,
    /// A chart's range, for a screen reader: `{max}` and `{min}` filled in.
    ChartRange,
    /// A chart with nothing to show.
    NoValues,
    /// A search field's results, named by its placeholder (`{query}`).
    ResultsFor,
    /// A search that found nothing.
    NoResults,
    /// How to close a sheet, for a screen reader.
    SheetHint,
}

/// Lets the app say threers' words its own way, or in a language threers has
/// none for: `f(word, language)` is asked first, and `None` leaves the word
/// to threers.
pub fn set_translator(f: Translator) {
    if let Ok(mut t) = TRANSLATOR.write() {
        *t = Some(f);
    }
}

/// `w` in the [`language`]: the app's ([`set_translator`]), else threers'
/// own (English, German, Spanish, French, Italian, Portuguese, Dutch,
/// Russian, Ukrainian, Polish, Hebrew, Arabic, Japanese, Korean, Chinese
/// Simplified and Traditional), else English.
pub fn word(w: Word) -> String {
    word_in(w, &language())
}

/// [`word`] in `lang` rather than the [`language`] set.
pub fn word_in(w: Word, lang: &str) -> String {
    if let Some(f) = TRANSLATOR.read().ok().and_then(|t| *t)
        && let Some(s) = f(w, lang)
    {
        return s;
    }
    builtin_words(lang)
        .or_else(|| builtin_words("en"))
        .map_or_else(String::new, |row| row[w as usize].to_string())
}

/// The [`Word`]s, in their order, in `tag`'s language.
fn builtin_words(tag: &str) -> Option<[&'static str; 14]> {
    Some(match base_of(tag).as_str() {
        "en" => [
            "Cut",
            "Copy",
            "Paste",
            "Select",
            "Select All",
            "Look Up",
            "Clear",
            "Close",
            "Back",
            "Highest {max}, lowest {min}",
            "no values",
            "Results for {query}",
            "No results",
            "Drag down or press Escape to close",
        ],
        "de" => [
            "Ausschneiden",
            "Kopieren",
            "Einsetzen",
            "Auswählen",
            "Alles auswählen",
            "Nachschlagen",
            "Löschen",
            "Schließen",
            "Zurück",
            "Höchstwert {max}, Tiefstwert {min}",
            "keine Werte",
            "Ergebnisse für {query}",
            "Keine Ergebnisse",
            "Zum Schließen nach unten ziehen oder Esc drücken",
        ],
        "es" => [
            "Cortar",
            "Copiar",
            "Pegar",
            "Seleccionar",
            "Seleccionar todo",
            "Consultar",
            "Borrar",
            "Cerrar",
            "Atrás",
            "Máximo {max}, mínimo {min}",
            "sin valores",
            "Resultados de {query}",
            "Sin resultados",
            "Arrastra hacia abajo o pulsa Esc para cerrar",
        ],
        "fr" => [
            "Couper",
            "Copier",
            "Coller",
            "Sélectionner",
            "Tout sélectionner",
            "Rechercher",
            "Effacer",
            "Fermer",
            "Retour",
            "Maximum {max}, minimum {min}",
            "aucune valeur",
            "Résultats pour {query}",
            "Aucun résultat",
            "Faites glisser vers le bas ou appuyez sur Échap pour fermer",
        ],
        "it" => [
            "Taglia",
            "Copia",
            "Incolla",
            "Seleziona",
            "Seleziona tutto",
            "Cerca",
            "Cancella",
            "Chiudi",
            "Indietro",
            "Massimo {max}, minimo {min}",
            "nessun valore",
            "Risultati per {query}",
            "Nessun risultato",
            "Trascina verso il basso o premi Esc per chiudere",
        ],
        "pt" => [
            "Recortar",
            "Copiar",
            "Colar",
            "Selecionar",
            "Selecionar tudo",
            "Pesquisar",
            "Limpar",
            "Fechar",
            "Voltar",
            "Máximo {max}, mínimo {min}",
            "sem valores",
            "Resultados para {query}",
            "Nenhum resultado",
            "Arraste para baixo ou pressione Esc para fechar",
        ],
        "nl" => [
            "Knip",
            "Kopieer",
            "Plak",
            "Selecteer",
            "Selecteer alles",
            "Zoek op",
            "Wis",
            "Sluit",
            "Terug",
            "Hoogste {max}, laagste {min}",
            "geen waarden",
            "Resultaten voor {query}",
            "Geen resultaten",
            "Sleep omlaag of druk op Esc om te sluiten",
        ],
        "ru" => [
            "Вырезать",
            "Копировать",
            "Вставить",
            "Выбрать",
            "Выбрать все",
            "Найти",
            "Очистить",
            "Закрыть",
            "Назад",
            "Максимум {max}, минимум {min}",
            "нет значений",
            "Результаты для «{query}»",
            "Нет результатов",
            "Потяните вниз или нажмите Esc, чтобы закрыть",
        ],
        "uk" => [
            "Вирізати",
            "Копіювати",
            "Вставити",
            "Вибрати",
            "Вибрати все",
            "Шукати",
            "Очистити",
            "Закрити",
            "Назад",
            "Максимум {max}, мінімум {min}",
            "немає значень",
            "Результати для «{query}»",
            "Немає результатів",
            "Потягніть униз або натисніть Esc, щоб закрити",
        ],
        "pl" => [
            "Wytnij",
            "Kopiuj",
            "Wklej",
            "Zaznacz",
            "Zaznacz wszystko",
            "Sprawdź",
            "Wyczyść",
            "Zamknij",
            "Wstecz",
            "Najwyższa {max}, najniższa {min}",
            "brak wartości",
            "Wyniki dla „{query}”",
            "Brak wyników",
            "Przeciągnij w dół lub naciśnij Esc, aby zamknąć",
        ],
        "he" => [
            "גזור",
            "העתק",
            "הדבק",
            "בחר",
            "בחר הכל",
            "חפש",
            "נקה",
            "סגור",
            "חזרה",
            "הגבוה ביותר {max}, הנמוך ביותר {min}",
            "אין ערכים",
            "תוצאות עבור {query}",
            "אין תוצאות",
            "גרור למטה או הקש Esc כדי לסגור",
        ],
        "ar" => [
            "قص",
            "نسخ",
            "لصق",
            "تحديد",
            "تحديد الكل",
            "بحث",
            "مسح",
            "إغلاق",
            "رجوع",
            "الأعلى {max}، الأدنى {min}",
            "لا توجد قيم",
            "نتائج {query}",
            "لا توجد نتائج",
            "اسحب لأسفل أو اضغط Esc للإغلاق",
        ],
        "ja" => [
            "カット",
            "コピー",
            "ペースト",
            "選択",
            "すべてを選択",
            "調べる",
            "消去",
            "閉じる",
            "戻る",
            "最高 {max}、最低 {min}",
            "値なし",
            "「{query}」の結果",
            "結果なし",
            "下にドラッグするか Esc キーで閉じます",
        ],
        "ko" => [
            "오려두기",
            "복사하기",
            "붙여넣기",
            "선택",
            "전체 선택",
            "찾아보기",
            "지우기",
            "닫기",
            "뒤로",
            "최고 {max}, 최저 {min}",
            "값 없음",
            "'{query}' 결과",
            "결과 없음",
            "아래로 끌거나 Esc 키를 눌러 닫습니다",
        ],
        "zh" if traditional_chinese(tag) => [
            "剪下",
            "拷貝",
            "貼上",
            "選取",
            "全選",
            "查詢",
            "清除",
            "關閉",
            "返回",
            "最高 {max}，最低 {min}",
            "沒有數值",
            "「{query}」的結果",
            "沒有結果",
            "向下拖曳或按 Esc 關閉",
        ],
        "zh" => [
            "剪切",
            "拷贝",
            "粘贴",
            "选择",
            "全选",
            "查询",
            "清除",
            "关闭",
            "返回",
            "最高 {max}，最低 {min}",
            "无数值",
            "“{query}”的结果",
            "无结果",
            "向下拖动或按 Esc 关闭",
        ],
        _ => return None,
    })
}

/// `zh-TW`, `zh-HK`, `zh-MO` and `zh-Hant…`: Chinese in Traditional characters.
pub(crate) fn traditional_chinese(tag: &str) -> bool {
    tag.split(['-', '_'])
        .skip(1)
        .any(|p| ["hant", "tw", "hk", "mo"].contains(&p.to_ascii_lowercase().as_str()))
}

/// A navigation stack's back button: the arrow points the way a row reads
/// back, left in English, right in Hebrew.
pub fn back_label() -> String {
    back_label_in(&language())
}

fn back_label_in(lang: &str) -> String {
    let arrow = match direction_of(lang) {
        LayoutDirection::RightToLeft => '›',
        _ => '‹',
    };
    format!("{arrow} {}", word_in(Word::Back, lang))
}

// ── Numbers and dates ────────────────────────────────────────────────────

/// What separates whole and fraction in `locale`: a comma in most of
/// Europe (German, French, Spanish, Russian, Ukrainian…), a point in
/// English, Hebrew, Japanese, Korean and Chinese.
pub fn decimal_separator(locale: &str) -> char {
    match base_of(locale).as_str() {
        "de" | "fr" | "es" | "it" | "pt" | "nl" | "ru" | "uk" | "be" | "pl" | "cs" | "sk"
        | "sv" | "da" | "nb" | "nn" | "no" | "fi" | "tr" | "el" | "ro" | "hu" | "bg" | "hr"
        | "sr" | "sl" | "lt" | "lv" | "et" | "id" | "vi" => ',',
        _ => '.',
    }
}

/// `x` with `places` decimals, written as `locale` writes them: `4.4` in
/// English, `4,4` in German.
pub fn format_decimal(locale: &str, x: f64, places: usize) -> String {
    let s = format!("{x:.places$}");
    match decimal_separator(locale) {
        '.' => s,
        c => s.replace('.', &c.to_string()),
    }
}

/// A date and time (month 1–12) as `locale` writes it, short: `Nov 20, 2026,
/// 21:34` in English, `20.11.2026, 21:34` in German, `20/11/2026 21:34` in
/// French, `2026/11/20 21:34` in Japanese and Chinese, `2026. 11. 20. 21:34`
/// in Korean. Day before month where the language puts it so, the year last
/// or first as it does; elsewhere the ISO order, `2026-11-20 21:34`.
pub fn format_date_time(
    locale: &str,
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
) -> String {
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let time = format!("{hour:02}:{minute:02}");
    match base_of(locale).as_str() {
        "en" => {
            let m = MONTHS[(month.clamp(1, 12) - 1) as usize];
            format!("{m} {day}, {year}, {time}")
        }
        "de" | "ru" | "uk" | "be" | "pl" | "cs" | "sk" | "fi" | "nb" | "nn" | "no" | "da"
        | "tr" | "ro" | "he" => format!("{day:02}.{month:02}.{year}, {time}"),
        "fr" | "pt" => format!("{day:02}/{month:02}/{year} {time}"),
        "es" | "it" | "el" | "ar" | "id" | "vi" => format!("{day:02}/{month:02}/{year}, {time}"),
        "nl" => format!("{day:02}-{month:02}-{year} {time}"),
        "ja" | "zh" => format!("{year}/{month:02}/{day:02} {time}"),
        "ko" => format!("{year}. {month}. {day}. {time}"),
        _ => format!("{year}-{month:02}-{day:02} {time}"),
    }
}

/// A day as `locale` writes it short, with its weekday (0 Monday … 6
/// Sunday) and no year: `Fri, Nov 5` in English, `Fr., 05.11.` in German,
/// `11/5（金）` in Japanese. Languages without a table here get the ISO
/// order, `11-05`.
pub fn format_weekday_date(locale: &str, weekday: u32, month: u32, day: u32) -> String {
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let w = (weekday % 7) as usize;
    let pick = |names: [&'static str; 7]| names[w];
    match base_of(locale).as_str() {
        "en" => format!(
            "{}, {} {day}",
            pick(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
            MONTHS[(month.clamp(1, 12) - 1) as usize]
        ),
        "de" => format!(
            "{}, {day:02}.{month:02}.",
            pick(["Mo.", "Di.", "Mi.", "Do.", "Fr.", "Sa.", "So."])
        ),
        "es" => format!(
            "{}, {day:02}/{month:02}",
            pick(["lun", "mar", "mié", "jue", "vie", "sáb", "dom"])
        ),
        "fr" => format!(
            "{} {day:02}/{month:02}",
            pick(["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."])
        ),
        "it" => format!(
            "{} {day:02}/{month:02}",
            pick(["lun", "mar", "mer", "gio", "ven", "sab", "dom"])
        ),
        "pt" => format!(
            "{}, {day:02}/{month:02}",
            pick(["seg.", "ter.", "qua.", "qui.", "sex.", "sáb.", "dom."])
        ),
        "nl" => format!(
            "{} {day:02}-{month:02}",
            pick(["ma", "di", "wo", "do", "vr", "za", "zo"])
        ),
        "ru" => format!(
            "{}, {day:02}.{month:02}",
            pick(["пн", "вт", "ср", "чт", "пт", "сб", "вс"])
        ),
        "uk" => format!(
            "{}, {day:02}.{month:02}",
            pick(["пн", "вт", "ср", "чт", "пт", "сб", "нд"])
        ),
        "pl" => format!(
            "{}, {day:02}.{month:02}",
            pick(["pon.", "wt.", "śr.", "czw.", "pt.", "sob.", "niedz."])
        ),
        "he" => format!(
            "{}, {day:02}.{month:02}",
            pick([
                "יום ב׳",
                "יום ג׳",
                "יום ד׳",
                "יום ה׳",
                "יום ו׳",
                "שבת",
                "יום א׳"
            ])
        ),
        "ja" => format!(
            "{month}/{day}（{}）",
            pick(["月", "火", "水", "木", "金", "土", "日"])
        ),
        "ko" => format!(
            "{month}. {day}. ({})",
            pick(["월", "화", "수", "목", "금", "토", "일"])
        ),
        "zh" => format!(
            "{month}/{day} {}",
            pick(["周一", "周二", "周三", "周四", "周五", "周六", "周日"])
        ),
        _ => format!("{month:02}-{day:02}"),
    }
}

// ── The user's languages ─────────────────────────────────────────────────

/// The languages the user reads, best first, as BCP 47 tags (`en-US`,
/// `zh-Hans-CN`): the system's preferred languages on macOS and iOS
/// (`AppleLanguages`), Windows (the user's UI languages) and the web
/// (`navigator.languages`); elsewhere, and where those give none, from the
/// environment: `LANGUAGE` (a `:`-separated list), then the first of
/// `LC_ALL`, `LC_MESSAGES` and `LANG` (`C` and `POSIX` are none).
pub fn system_languages() -> Vec<String> {
    let mut out = platform_languages();
    if out.is_empty() {
        out = env_languages(|v| std::env::var(v).ok());
    }
    let mut seen = Vec::new();
    out.retain(|t| {
        let fresh = !seen.contains(t);
        seen.push(t.clone());
        fresh
    });
    out
}

/// What of `available` (the languages an app has translations in) suits a
/// reader of `wanted` (best first, as [`system_languages`]) best: the
/// first wanted tag that is available as it is, else in its language with
/// another region or script (`pt-BR` takes `pt`, `en-GB` takes `en-US`,
/// `zh-Hant-TW` takes `zh-TW` before `zh`). `None` when none of them is.
pub fn best_match<'a>(available: &[&'a str], wanted: &[impl AsRef<str>]) -> Option<&'a str> {
    let norm = |t: &str| t.replace('_', "-").to_ascii_lowercase();
    for w in wanted {
        let w = norm(w.as_ref());
        if let Some(a) = available.iter().find(|a| norm(a) == w) {
            return Some(a);
        }
        let lang = base_of(&w);
        let same: Vec<&&str> = available.iter().filter(|a| base_of(a) == lang).collect();
        if lang == "zh" {
            // Traditional and Simplified are different writing, not regions.
            let trad = traditional_chinese(&w);
            if let Some(a) = same.iter().find(|a| traditional_chinese(a) == trad) {
                return Some(a);
            }
        }
        if let Some(a) = same.first() {
            return Some(a);
        }
    }
    None
}

/// `en_US.UTF-8@euro` → `en-US`; `C`, `POSIX` and empty → none.
fn posix_tag(v: &str) -> Option<String> {
    let v = v.split(['.', '@']).next().unwrap_or("").trim();
    (!v.is_empty() && v != "C" && v != "POSIX").then(|| v.replace('_', "-"))
}

fn env_languages(var: impl Fn(&str) -> Option<String>) -> Vec<String> {
    let mut out: Vec<String> = var("LANGUAGE")
        .map(|l| l.split(':').filter_map(posix_tag).collect())
        .unwrap_or_default();
    if let Some(t) = ["LC_ALL", "LC_MESSAGES", "LANG"]
        .iter()
        .find_map(|v| var(v).as_deref().and_then(posix_tag))
    {
        out.push(t);
    }
    out
}

#[cfg(target_vendor = "apple")]
fn platform_languages() -> Vec<String> {
    use std::ffi::{c_char, c_void, CStr};
    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFLocaleCopyPreferredLanguages() -> *const c_void;
        fn CFArrayGetCount(array: *const c_void) -> isize;
        fn CFArrayGetValueAtIndex(array: *const c_void, i: isize) -> *const c_void;
        fn CFStringGetCString(s: *const c_void, buf: *mut c_char, size: isize, encoding: u32)
            -> u8;
        fn CFRelease(cf: *const c_void);
    }
    const UTF8: u32 = 0x0800_0100;
    let mut out = Vec::new();
    // SAFETY: CFLocaleCopyPreferredLanguages returns a CFArray of CFStrings
    // we own (released below) or null; each index is within its count, and
    // CFStringGetCString writes at most `size` bytes, NUL-terminated, into
    // the buffer when it returns true.
    unsafe {
        let array = CFLocaleCopyPreferredLanguages();
        if array.is_null() {
            return out;
        }
        for i in 0..CFArrayGetCount(array) {
            let s = CFArrayGetValueAtIndex(array, i);
            let mut buf = [0 as c_char; 64];
            if !s.is_null()
                && CFStringGetCString(s, buf.as_mut_ptr(), buf.len() as isize, UTF8) != 0
            {
                out.push(CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned());
            }
        }
        CFRelease(array);
    }
    out
}

#[cfg(windows)]
fn platform_languages() -> Vec<String> {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetUserPreferredUILanguages(
            flags: u32,
            count: *mut u32,
            buf: *mut u16,
            len: *mut u32,
        ) -> i32;
    }
    const MUI_LANGUAGE_NAME: u32 = 0x8;
    let (mut count, mut len) = (0u32, 0u32);
    // SAFETY: with a null buffer the call only writes the needed length
    // (UTF-16 units, the final NULs included) into `len`.
    if unsafe {
        GetUserPreferredUILanguages(
            MUI_LANGUAGE_NAME,
            &mut count,
            std::ptr::null_mut(),
            &mut len,
        )
    } == 0
    {
        return Vec::new();
    }
    let mut buf = vec![0u16; len as usize];
    // SAFETY: `buf` holds the `len` units the first call asked for.
    if unsafe {
        GetUserPreferredUILanguages(MUI_LANGUAGE_NAME, &mut count, buf.as_mut_ptr(), &mut len)
    } == 0
    {
        return Vec::new();
    }
    buf.split(|&c| c == 0)
        .filter(|s| !s.is_empty())
        .map(String::from_utf16_lossy)
        .collect()
}

#[cfg(target_arch = "wasm32")]
fn platform_languages() -> Vec<String> {
    web_sys::window().map_or_else(Vec::new, |w| {
        w.navigator()
            .languages()
            .iter()
            .filter_map(|v| v.as_string())
            .collect()
    })
}

#[cfg(not(any(target_vendor = "apple", windows, target_arch = "wasm32")))]
fn platform_languages() -> Vec<String> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threers_words_in_each_language() {
        // Not `set_language`: the language is the process's, and other tests
        // read the English words while this one runs.
        assert_eq!(word(Word::SelectAll), "Select All", "English until set");
        assert_eq!(word_in(Word::Paste, "de-AT"), "Einsetzen");
        assert_eq!(word_in(Word::Copy, "zh-Hant-TW"), "拷貝");
        assert_eq!(word_in(Word::Copy, "zh-CN"), "拷贝");
        assert_eq!(back_label_in("he"), "› חזרה");
        assert_eq!(back_label_in("en-GB"), "‹ Back");
        assert_eq!(
            word_in(Word::Close, "xx"),
            "Close",
            "English where there is none"
        );
        set_translator(|w, lang| (w == Word::Close && lang == "xx").then(|| "Zu".into()));
        assert_eq!(word_in(Word::Close, "xx"), "Zu");
        assert_eq!(word_in(Word::Back, "xx"), "Back");
        // Every language has every word.
        for lang in [
            "en", "de", "es", "fr", "it", "pt", "nl", "ru", "uk", "pl", "he", "ar", "ja", "ko",
            "zh", "zh-TW",
        ] {
            let row = builtin_words(lang).unwrap();
            assert!(row.iter().all(|w| !w.is_empty()), "{lang}");
        }
    }

    #[test]
    fn numbers_and_dates_as_each_language_writes_them() {
        assert_eq!(format_decimal("en-US", 4.44, 1), "4.4");
        assert_eq!(format_decimal("de", 4.44, 1), "4,4");
        assert_eq!(format_decimal("ja", 1536.0, 0), "1536");
        let at = |l| format_date_time(l, 2026, 11, 5, 9, 4);
        assert_eq!(at("en"), "Nov 5, 2026, 09:04");
        assert_eq!(at("de-DE"), "05.11.2026, 09:04");
        assert_eq!(at("fr"), "05/11/2026 09:04");
        assert_eq!(at("es"), "05/11/2026, 09:04");
        assert_eq!(at("ja"), "2026/11/05 09:04");
        assert_eq!(at("ko"), "2026. 11. 5. 09:04");
        assert_eq!(at("xx"), "2026-11-05 09:04");
        // 2026-11-06 is a Friday (weekday 4).
        let day = |l| format_weekday_date(l, 4, 11, 6);
        assert_eq!(day("en-US"), "Fri, Nov 6");
        assert_eq!(day("de"), "Fr., 06.11.");
        assert_eq!(day("ja"), "11/6（金）");
        assert_eq!(day("he"), "יום ו׳, 06.11");
        assert_eq!(day("xx"), "11-06");
    }

    #[test]
    fn the_best_translation_for_a_reader() {
        let have = [
            "de", "en", "es", "fr", "he", "ja", "ko", "uk", "zh", "zh-TW",
        ];
        assert_eq!(best_match(&have, &["fr-CA", "en"]), Some("fr"));
        assert_eq!(
            best_match(&have, &["nl", "de-CH"]),
            Some("de"),
            "the first one there is"
        );
        assert_eq!(best_match(&have, &["zh-Hant-HK"]), Some("zh-TW"));
        assert_eq!(best_match(&have, &["zh-Hans-CN"]), Some("zh"));
        assert_eq!(best_match(&have, &["ja_JP"]), Some("ja"));
        assert_eq!(best_match(&["en-US", "pt"], &["en-GB"]), Some("en-US"));
        assert_eq!(best_match(&have, &["nl", "pl"]), None);
        assert_eq!(best_match(&have, &[] as &[&str]), None);
    }

    #[test]
    fn the_environment_names_languages_as_posix_does() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |v: &str| {
                pairs
                    .iter()
                    .find(|(k, _)| *k == v)
                    .map(|(_, x)| x.to_string())
            }
        };
        assert_eq!(env_languages(env(&[("LANG", "uk_UA.UTF-8")])), ["uk-UA"]);
        assert_eq!(
            env_languages(env(&[
                ("LANGUAGE", "fr_CA:fr"),
                ("LC_ALL", "C"),
                ("LANG", "de_DE@euro")
            ])),
            ["fr-CA", "fr", "de-DE"]
        );
        assert!(env_languages(env(&[("LANG", "POSIX")])).is_empty());
        assert!(!system_languages().iter().any(|t| t.is_empty()));
    }

    #[test]
    fn categories_follow_each_languages_rules() {
        assert_eq!(
            [0, 1, 2, 5].map(|n| category("en-US", n)),
            [
                Category::Other,
                Category::One,
                Category::Other,
                Category::Other
            ]
        );
        assert_eq!(
            [0, 1, 2].map(|n| category("fr", n)),
            [Category::One, Category::One, Category::Other]
        );
        assert_eq!(
            [1, 11, 22, 25, 111].map(|n| category("uk", n)),
            [
                Category::One,
                Category::Many,
                Category::Few,
                Category::Many,
                Category::Many
            ]
        );
        assert_eq!(
            [1, 22, 12, 5].map(|n| category("pl", n)),
            [Category::One, Category::Few, Category::Many, Category::Many]
        );
        assert_eq!(
            [0, 1, 2, 3, 11, 100].map(|n| category("ar", n)),
            [
                Category::Zero,
                Category::One,
                Category::Two,
                Category::Few,
                Category::Many,
                Category::Other
            ]
        );
        assert_eq!(category("ja", 1), Category::Other);
        assert_eq!(direction_of("he-IL"), LayoutDirection::RightToLeft);
        assert_eq!(direction_of("en"), LayoutDirection::LeftToRight);
    }
}
