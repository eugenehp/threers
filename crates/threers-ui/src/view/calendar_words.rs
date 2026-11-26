//! A calendar's words in the languages threers speaks: months, weekdays,
//! how a month and a day are written, and the day a week starts on. From
//! Unicode CLDR 48 (as ICU 78 gives it), but for the words that turn the
//! months.

use super::kit_more::Day;
use crate::i18n::{base_of, language, traditional_chinese};

/// A [`calendar`](super::calendar)'s words.
#[derive(Debug, Clone, PartialEq)]
pub struct CalendarWords {
    /// As they are on their own ("май").
    pub months: [String; 12],
    /// As they are in a date ("мая").
    pub months_in_date: [String; 12],
    /// From Monday, short.
    pub weekdays: [String; 7],
    /// A month of a year, `{m}` and `{y}` filled in.
    pub title: String,
    /// A day, `{d}`, `{m}` and `{y}` filled in.
    pub date: String,
    pub previous: String,
    pub next: String,
    /// The first day of a week: 0 Monday to 6 Sunday.
    pub week_starts: u32,
}

impl Default for CalendarWords {
    /// In the [`language`](crate::i18n::language) set.
    fn default() -> Self {
        Self::for_language(&language())
    }
}

impl CalendarWords {
    /// In `tag`'s language (those of [`word`](crate::i18n::word); English
    /// for others), weeks starting as they do in its region (`en-GB`), or
    /// where it is spoken most.
    pub fn for_language(tag: &str) -> Self {
        let base = base_of(tag);
        let c = cal(if base == "zh" && traditional_chinese(tag) {
            "zh-Hant"
        } else {
            &base
        })
        .unwrap_or_else(english);
        Self {
            months: c.months.map(String::from),
            months_in_date: c.in_date.map(String::from),
            weekdays: c.weekdays.map(String::from),
            title: c.title.into(),
            date: c.date.into(),
            previous: c.turn[0].into(),
            next: c.turn[1].into(),
            week_starts: region(tag).map_or(c.week_starts, week_starts),
        }
    }

    /// `day`'s month and year, as a heading.
    pub fn month_title(&self, day: Day) -> String {
        let m = &self.months[(day.month.clamp(1, 12) - 1) as usize];
        self.title
            .replace("{m}", m)
            .replace("{y}", &day.year.to_string())
    }

    /// `day` written out.
    pub fn day(&self, day: Day) -> String {
        let m = &self.months_in_date[(day.month.clamp(1, 12) - 1) as usize];
        self.date
            .replace("{m}", m)
            .replace("{d}", &day.day.to_string())
            .replace("{y}", &day.year.to_string())
    }
}

/// `tag`'s region, upper case (`en-GB`'s GB; `zh-Hant-TW`'s TW).
fn region(tag: &str) -> Option<String> {
    tag.split(['-', '_'])
        .skip(1)
        .take_while(|s| s.len() > 1)
        .find(|s| s.len() == 2 && s.chars().all(|c| c.is_ascii_alphabetic()))
        .map(str::to_ascii_uppercase)
}

/// The day weeks start on in `region`: 0 Monday to 6 Sunday.
fn week_starts(region: String) -> u32 {
    match region.as_str() {
        "AG" | "AS" | "BD" | "BR" | "BS" | "BT" | "BU" | "BW" | "BZ" | "CA" | "CO" | "DM" | "DO" | "ET" | "GT" | "GU" | "HK" | "HN" | "ID" | "IL" | "IN" | "IS" | "JM" | "JP" | "KE" | "KH" | "KR" | "LA" | "MH" | "MM" | "MO" | "MT" | "MX" | "MZ" | "NI" | "NP" | "PA" | "PE" | "PH" | "PK" | "PR" | "PT" | "PY" | "RH" | "SA" | "SG" | "SV" | "TH" | "TT" | "TW" | "UM" | "US" | "VE" | "VI" | "WS" | "YD" | "YE" | "ZA" | "ZW" => 6,
        "AF" | "BH" | "DJ" | "DZ" | "EG" | "IQ" | "IR" | "JO" | "KW" | "LY" | "OM" | "QA" | "SD" | "SY" => 5,
        "MV" => 4,
        _ => 0,
    }
}

struct Cal {
    title: &'static str,
    date: &'static str,
    turn: [&'static str; 2],
    months: [&'static str; 12],
    in_date: [&'static str; 12],
    weekdays: [&'static str; 7],
    week_starts: u32,
}

fn english() -> Cal {
    cal("en").unwrap_or(Cal {
        title: "",
        date: "",
        turn: [""; 2],
        months: [""; 12],
        in_date: [""; 12],
        weekdays: [""; 7],
        week_starts: 0,
    })
}

fn cal(lang: &str) -> Option<Cal> {
    Some(match lang {
        "en" => Cal {
            turn: ["Previous month", "Next month"],
            title: "{m} {y}",
            date: "{m} {d}, {y}",
            months: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
            in_date: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
            weekdays: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            week_starts: 6,
        },
        "de" => Cal {
            turn: ["Vorheriger Monat", "Nächster Monat"],
            title: "{m} {y}",
            date: "{d}. {m} {y}",
            months: ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"],
            in_date: ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"],
            weekdays: ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"],
            week_starts: 0,
        },
        "es" => Cal {
            turn: ["Mes anterior", "Mes siguiente"],
            title: "{m} de {y}",
            date: "{d} de {m} de {y}",
            months: ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"],
            in_date: ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"],
            weekdays: ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"],
            week_starts: 0,
        },
        "fr" => Cal {
            turn: ["Mois précédent", "Mois suivant"],
            title: "{m} {y}",
            date: "{d} {m} {y}",
            months: ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"],
            in_date: ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"],
            weekdays: ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."],
            week_starts: 0,
        },
        "it" => Cal {
            turn: ["Mese precedente", "Mese successivo"],
            title: "{m} {y}",
            date: "{d} {m} {y}",
            months: ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"],
            in_date: ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"],
            weekdays: ["lun", "mar", "mer", "gio", "ven", "sab", "dom"],
            week_starts: 0,
        },
        "pt" => Cal {
            turn: ["Mês anterior", "Próximo mês"],
            title: "{m} de {y}",
            date: "{d} de {m} de {y}",
            months: ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"],
            in_date: ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"],
            weekdays: ["seg.", "ter.", "qua.", "qui.", "sex.", "sáb.", "dom."],
            week_starts: 6,
        },
        "nl" => Cal {
            turn: ["Vorige maand", "Volgende maand"],
            title: "{m} {y}",
            date: "{d} {m} {y}",
            months: ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober", "november", "december"],
            in_date: ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober", "november", "december"],
            weekdays: ["ma", "di", "wo", "do", "vr", "za", "zo"],
            week_starts: 0,
        },
        "ru" => Cal {
            turn: ["Предыдущий месяц", "Следующий месяц"],
            title: "{m} {y} г.",
            date: "{d} {m} {y} г.",
            months: ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"],
            in_date: ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"],
            weekdays: ["пн", "вт", "ср", "чт", "пт", "сб", "вс"],
            week_starts: 0,
        },
        "uk" => Cal {
            turn: ["Попередній місяць", "Наступний місяць"],
            title: "{m} {y} р.",
            date: "{d} {m} {y} р.",
            months: ["січень", "лютий", "березень", "квітень", "травень", "червень", "липень", "серпень", "вересень", "жовтень", "листопад", "грудень"],
            in_date: ["січня", "лютого", "березня", "квітня", "травня", "червня", "липня", "серпня", "вересня", "жовтня", "листопада", "грудня"],
            weekdays: ["пн", "вт", "ср", "чт", "пт", "сб", "нд"],
            week_starts: 0,
        },
        "pl" => Cal {
            turn: ["Poprzedni miesiąc", "Następny miesiąc"],
            title: "{m} {y}",
            date: "{d} {m} {y}",
            months: ["styczeń", "luty", "marzec", "kwiecień", "maj", "czerwiec", "lipiec", "sierpień", "wrzesień", "październik", "listopad", "grudzień"],
            in_date: ["stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca", "lipca", "sierpnia", "września", "października", "listopada", "grudnia"],
            weekdays: ["pon.", "wt.", "śr.", "czw.", "pt.", "sob.", "niedz."],
            week_starts: 0,
        },
        "he" => Cal {
            turn: ["החודש הקודם", "החודש הבא"],
            title: "{m} {y}",
            date: "{d} ב{m} {y}",
            months: ["ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני", "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר"],
            in_date: ["ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני", "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר"],
            weekdays: ["ב׳", "ג׳", "ד׳", "ה׳", "ו׳", "ש׳", "א׳"],
            week_starts: 6,
        },
        "ar" => Cal {
            turn: ["الشهر السابق", "الشهر التالي"],
            title: "{m} {y}",
            date: "{d} {m} {y}",
            months: ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"],
            in_date: ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"],
            weekdays: ["ن", "ث", "ر", "خ", "ج", "س", "ح"],
            week_starts: 5,
        },
        "ja" => Cal {
            turn: ["前の月", "次の月"],
            title: "{y}年{m}月",
            date: "{y}年{m}月{d}日",
            months: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
            in_date: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
            weekdays: ["月", "火", "水", "木", "金", "土", "日"],
            week_starts: 6,
        },
        "ko" => Cal {
            turn: ["이전 달", "다음 달"],
            title: "{y}년 {m}",
            date: "{y}년 {m} {d}일",
            months: ["1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월"],
            in_date: ["1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월"],
            weekdays: ["월", "화", "수", "목", "금", "토", "일"],
            week_starts: 6,
        },
        "zh" => Cal {
            turn: ["上个月", "下个月"],
            title: "{y}年{m}月",
            date: "{y}年{m}月{d}日",
            months: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
            in_date: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
            weekdays: ["周一", "周二", "周三", "周四", "周五", "周六", "周日"],
            week_starts: 0,
        },
        "zh-Hant" => Cal {
            turn: ["上個月", "下個月"],
            title: "{y}年{m}月",
            date: "{y}年{m}月{d}日",
            months: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
            in_date: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
            weekdays: ["週一", "週二", "週三", "週四", "週五", "週六", "週日"],
            week_starts: 6,
        },
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// As ICU 78 writes them.
    const ICU: [(&str, &str, &str); 16] = [
        ("en", "May 2026", "May 9, 2026"),
        ("de", "Mai 2026", "9. Mai 2026"),
        ("es", "mayo de 2026", "9 de mayo de 2026"),
        ("fr", "mai 2026", "9 mai 2026"),
        ("it", "maggio 2026", "9 maggio 2026"),
        ("pt", "maio de 2026", "9 de maio de 2026"),
        ("nl", "mei 2026", "9 mei 2026"),
        ("ru", "май 2026 г.", "9 мая 2026 г."),
        ("uk", "травень 2026 р.", "9 травня 2026 р."),
        ("pl", "maj 2026", "9 maja 2026"),
        ("he", "מאי 2026", "9 במאי 2026"),
        ("ar", "مايو 2026", "9 مايو 2026"),
        ("ja", "2026年5月", "2026年5月9日"),
        ("ko", "2026년 5월", "2026년 5월 9일"),
        ("zh", "2026年5月", "2026年5月9日"),
        ("zh-Hant", "2026年5月", "2026年5月9日"),
    ];

    #[test]
    fn months_and_days_are_written_as_icu_writes_them() {
        for (tag, title, date) in ICU {
            let w = CalendarWords::for_language(tag);
            let d = Day::new(2026, 5, 9).unwrap();
            assert_eq!((w.month_title(d).as_str(), w.day(d).as_str()), (title, date), "{tag}");
            let all = w.months.iter().chain(&w.months_in_date).chain(&w.weekdays);
            assert!(all.chain([&w.previous, &w.next]).all(|s| !s.is_empty()), "{tag}");
        }
    }

    #[test]
    fn a_region_says_when_weeks_start_and_other_languages_are_english() {
        let starts = |t: &str| CalendarWords::for_language(t).week_starts;
        assert_eq!((starts("en"), starts("en-US"), starts("en-GB")), (6, 6, 0));
        assert_eq!((starts("de"), starts("pt-PT"), starts("ar-EG"), starts("ar-MA")), (0, 6, 5, 0));
        assert_eq!(starts("zh-Hant-TW"), 6);
        assert_eq!(starts("en-u-ca-gregory"), 6, "an extension is no region");
        assert_eq!(CalendarWords::for_language("xx").months[0], "January");
        assert_eq!(CalendarWords::for_language("zh-TW").previous, "上個月");
    }
}
