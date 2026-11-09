//! Faces the operating system ships, for the scripts the built-in faces lack.
//!
//! The bitmap face and SF Compact cover Latin, Cyrillic and Greek. Chinese,
//! Japanese, Korean, Devanagari and Thai each need a face of their own, and
//! every platform keeps one in a different place. [`CaptionFont::system_for_language`]
//! looks in the usual places for a language and returns the first face that has
//! **every** character of the text it is to draw, so a window that shows its
//! words in another language gets letters, not boxes.
//!
//! A face is taken only if it covers the text. A collection (`.ttc`, as macOS
//! ships Hiragino and Apple SD Gothic Neo) is searched face by face.
//! OpenType/CFF faces load as well as TrueType ones. Nothing here runs on the
//! web: there is no file system to look in.

use super::font::CaptionFont;
use crate::loaders::TtfFont;

impl CaptionFont {
    /// Whether the face draws its own glyph for every character of `text`,
    /// not a stand-in box. White space and control characters are skipped:
    /// a face need not have a glyph for them to lay them out.
    pub fn covers(&self, text: &str) -> bool {
        text.chars()
            .filter(|c| !c.is_whitespace() && !c.is_control())
            .all(|c| self.has_glyph(c))
    }

    /// The first face of the font file or collection in `bytes` that covers
    /// `covering`, or `None` if the file is not a font this reads or no face
    /// in it has every character.
    ///
    /// A plain `.ttf`/`.otf` has one face. A `.ttc` has several, tried in
    /// order.
    pub fn from_collection(bytes: &[u8], covering: &str) -> Option<Self> {
        for index in 0..64u32 {
            // Past the last face the parse fails, which ends the search.
            let Ok(face) = TtfFont::parse_face(bytes, index, &[]) else {
                break;
            };
            let font = Self::from_ttf(face);
            if font.covers(covering) {
                return Some(font);
            }
        }
        None
    }

    /// A face installed on this machine for `language` that covers
    /// `covering`.
    ///
    /// `language` is a language tag: `"ja"`, `"zh"` (Simplified), `"zh-TW"`
    /// (Traditional; `"zh-Hant"` and `"zh-HK"` too), `"ko"`, `"hi"`, `"th"`.
    /// Other languages have no candidates and give `None`: the built-in face
    /// covers them. Pass the text the language will show (see
    /// [`CaptionFont::covers`]); a face that lacks some of it is skipped, and
    /// `None` means the machine has none that covers it all.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn system_for_language(language: &str, covering: &str) -> Option<Self> {
        system_font_paths(language).into_iter().find_map(|path| {
            shared_covering(&path.to_string_lossy(), covering).map(Self::from_shared_face)
        })
    }
}

impl CaptionFont {
    /// The system's ordinary sans-serif face, for a user interface: SF Compact where the system has it
    /// (macOS), otherwise the first of Noto Sans, Ubuntu, Liberation Sans, DejaVu Sans, Droid Sans,
    /// Segoe UI, Arial or Helvetica that is installed and covers Latin and Cyrillic. `None` where there is
    /// none; [`CaptionFont::ui`] then gives the bitmap face, which is monospaced and wide.
    ///
    /// The family's own bold, italic and bold-italic files are used where it has them, so `styled(700, false)`
    /// is the real bold and not the regular grown. [`CaptionFont::ui`] stays as it was, so a program that wants the
    /// better face asks for this one.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn system_sans() -> Option<Self> {
        if let Some(font) = Self::try_sf_compact() {
            return Some(font);
        }
        family_from(&generic_paths(Generic::Sans), INTERFACE_SAMPLE)
    }

    /// The system's monospaced face (Menlo, DejaVu Sans Mono, Liberation Mono, Noto Sans Mono, Ubuntu Mono, Consolas
    /// or Courier), with its bold and italic where it has them; `None` if there is none.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn system_mono() -> Option<Self> {
        family_from(&generic_paths(Generic::Mono), INTERFACE_SAMPLE)
    }

    /// The system's serif face (New York, Times New Roman, Noto Serif, Liberation Serif, DejaVu Serif, Georgia), with
    /// its bold and italic where it has them; `None` if there is none.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn system_serif() -> Option<Self> {
        family_from(&generic_paths(Generic::Serif), INTERFACE_SAMPLE)
    }

    /// This face with an installed face behind it for each of Japanese, Chinese,
    /// Korean, Hindi, Thai, Hebrew and Arabic that the machine has, so text in any of them mixed
    /// with this face's own draws letters wherever the first face has none (see
    /// [`with_fallback`](CaptionFont::with_fallback)).
    ///
    /// `preferred` names the languages (tags as in [`system_for_language`](Self::system_for_language))
    /// whose faces go first. The order matters where scripts overlap: Han characters are drawn
    /// by the first face that has them, so `&["zh-TW"]` draws Chinese text in a Traditional
    /// face and `&["ja"]` Japanese in a Japanese one. The rest follow in a fixed order.
    ///
    /// Each face is read from disk and held in memory, a few megabytes each: do it once,
    /// not per frame.
    #[cfg(not(target_arch = "wasm32"))]
    #[must_use]
    pub fn with_system_fallbacks(mut self, preferred: &[&str]) -> Self {
        // A few letters every face for the script has, to take the first face that is one.
        const SAMPLES: [(&str, &str); 7] = [
            ("ja", "日本語かな"),
            ("zh", "中文"),
            ("ko", "한국어"),
            ("hi", "हिन्दी"),
            ("th", "ไทย"),
            ("he", "שלום"),
            ("ar", "مرحبا"),
        ];
        let base = |tag: &str| {
            tag.split(['-', '_'])
                .next()
                .unwrap_or("")
                .to_ascii_lowercase()
        };
        let mut order: Vec<(String, &str)> = Vec::new();
        for tag in preferred {
            if let Some(&(b, sample)) = SAMPLES.iter().find(|(b, _)| *b == base(tag))
                && !order.iter().any(|(t, _)| base(t) == b)
            {
                order.push(((*tag).to_string(), sample));
            }
        }
        for &(b, sample) in &SAMPLES {
            if !order.iter().any(|(t, _)| base(t) == b) {
                order.push((b.to_string(), sample));
            }
        }
        for (tag, sample) in order {
            if let Some(face) = Self::system_for_language(&tag, sample) {
                self = self.with_fallback(face);
            }
        }
        self
    }
}

/// Letters an interface face must have: Latin with accents, digits, punctuation, Cyrillic.
#[cfg(not(target_arch = "wasm32"))]
const INTERFACE_SAMPLE: &str = "Hamburgefonstiv 0123456789 .,:;!?%()+- éüñçÅøß Привет";

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
enum Generic {
    Sans,
    Mono,
    Serif,
}

/// Where the regular files of a generic family usually are, best first, on every platform.
#[cfg(not(target_arch = "wasm32"))]
fn generic_paths(kind: Generic) -> Vec<String> {
    let roots = [
        "/usr/share/fonts/truetype",
        "/usr/share/fonts/opentype",
        "/usr/share/fonts/TTF",
        "/usr/share/fonts/liberation",
        "/usr/share/fonts/dejavu",
        "/usr/share/fonts",
    ];
    let in_roots = |sub: &str, file: &str| -> Vec<String> {
        roots.iter().map(|r| format!("{r}/{sub}/{file}")).collect()
    };
    let mut paths: Vec<String> = Vec::new();
    match kind {
        Generic::Sans => {
            for (sub, file) in [
                ("noto", "NotoSans-Regular.ttf"),
                ("ubuntu", "Ubuntu-R.ttf"),
                ("ubuntu", "UbuntuSans[wdth,wght].ttf"),
                ("liberation", "LiberationSans-Regular.ttf"),
                ("liberation2", "LiberationSans-Regular.ttf"),
                ("dejavu", "DejaVuSans.ttf"),
                ("droid", "DroidSans.ttf"),
            ] {
                paths.extend(in_roots(sub, file));
            }
            paths.extend(
                [
                    "/usr/share/fonts/TTF/DejaVuSans.ttf",
                    "/usr/share/fonts/NotoSans-Regular.ttf",
                    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
                ]
                .map(String::from),
            );
            paths.extend(["segoeui.ttf", "arial.ttf"].map(|f| format!("C:/Windows/Fonts/{f}")));
            paths.extend(
                ["Helvetica.ttc", "HelveticaNeue.ttc"]
                    .map(|f| format!("/System/Library/Fonts/{f}")),
            );
        }
        Generic::Mono => {
            paths.extend(
                ["Menlo.ttc", "Monaco.ttf", "Courier.ttc"]
                    .map(|f| format!("/System/Library/Fonts/{f}")),
            );
            for (sub, file) in [
                ("dejavu", "DejaVuSansMono.ttf"),
                ("liberation", "LiberationMono-Regular.ttf"),
                ("noto", "NotoSansMono-Regular.ttf"),
                ("ubuntu", "UbuntuMono-R.ttf"),
            ] {
                paths.extend(in_roots(sub, file));
            }
            paths.extend(["consola.ttf", "cour.ttf"].map(|f| format!("C:/Windows/Fonts/{f}")));
        }
        Generic::Serif => {
            paths.extend(
                [
                    "/System/Library/Fonts/NewYork.ttf",
                    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
                    "/System/Library/Fonts/Times.ttc",
                ]
                .map(String::from),
            );
            for (sub, file) in [
                ("noto", "NotoSerif-Regular.ttf"),
                ("liberation", "LiberationSerif-Regular.ttf"),
                ("dejavu", "DejaVuSerif.ttf"),
            ] {
                paths.extend(in_roots(sub, file));
            }
            paths.extend(["times.ttf", "georgia.ttf"].map(|f| format!("C:/Windows/Fonts/{f}")));
        }
    }
    paths
}

/// The bold, italic and bold-italic files beside a regular one, by the ways fonts are named: `-Regular` becomes `-Bold`, `-Italic`
/// and `-BoldItalic`; Ubuntu's `-R` becomes `-B`, `-RI` and `-BI`; DejaVu's plain name gains `-Bold`, `-Oblique` and `-BoldOblique`;
/// Windows' gains `b`/`bd`, `i` and `z`/`bi`. Only files that exist are returned.
#[cfg(not(target_arch = "wasm32"))]
fn siblings(regular: &str) -> [Option<std::path::PathBuf>; 3] {
    let path = std::path::Path::new(regular);
    let (Some(stem), Some(ext)) = (
        path.file_stem().and_then(|s| s.to_str()),
        path.extension().and_then(|e| e.to_str()),
    ) else {
        return [None, None, None];
    };
    let dir = path.parent().unwrap_or(std::path::Path::new(""));
    let find = |names: &[String]| {
        names
            .iter()
            .map(|n| dir.join(format!("{n}.{ext}")))
            .find(|p| p.exists())
    };
    let (base, bold, italic, bold_italic): (&str, Vec<&str>, Vec<&str>, Vec<&str>) =
        if let Some(b) = stem.strip_suffix("-Regular") {
            (
                b,
                vec!["-Bold"],
                vec!["-Italic", "-Oblique"],
                vec!["-BoldItalic", "-BoldOblique"],
            )
        } else if let Some(b) = stem.strip_suffix("-R") {
            (b, vec!["-B"], vec!["-RI"], vec!["-BI"])
        } else {
            (
                stem,
                vec!["-Bold", "b", "bd"],
                vec!["-Italic", "-Oblique", "i"],
                vec!["-BoldItalic", "-BoldOblique", "z", "bi"],
            )
        };
    let names = |suffixes: Vec<&str>| {
        suffixes
            .into_iter()
            .map(|s| format!("{base}{s}"))
            .collect::<Vec<_>>()
    };
    [
        find(&names(bold)),
        find(&names(italic)),
        find(&names(bold_italic)),
    ]
}

/// The first candidate that is installed and covers `sample`, as a family with the bold and italic files it has.
#[cfg(not(target_arch = "wasm32"))]
fn family_from(candidates: &[String], sample: &str) -> Option<CaptionFont> {
    candidates.iter().find_map(|path| {
        let bytes = std::fs::read(path).ok()?;
        // A collection's first face covering the sample is the regular; a family's own files are single faces.
        let regular = CaptionFont::from_collection(&bytes, sample)?;
        let [bold, italic, bold_italic] =
            siblings(path).map(|p| p.and_then(|p| std::fs::read(p).ok()));
        // Only files, not collections, make a family; a `.ttc` regular stays as it is.
        if path.ends_with(".ttc") {
            return Some(regular);
        }
        CaptionFont::from_family_files(
            &bytes,
            bold.as_deref(),
            italic.as_deref(),
            bold_italic.as_deref(),
        )
        .ok()
        .filter(|f| f.covers(sample))
        .or(Some(regular))
    })
}

/// Faces parsed from system files, kept for the life of the process: a window that changes language, or a second
/// window, takes the same face again for the price of a reference instead of reading a twenty-megabyte collection.
#[cfg(not(target_arch = "wasm32"))]
fn face_cache() -> &'static std::sync::Mutex<std::collections::HashMap<String, CachedFile>> {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, CachedFile>>,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(Default::default)
}

/// The faces of one file parsed so far, and whether that is all it has.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct CachedFile {
    faces: Vec<std::sync::Arc<TtfFont>>,
    complete: bool,
    /// The file could not be read at all.
    missing: bool,
}

/// The first face of `path` that covers `covering`, from the cache where it has been parsed already.
#[cfg(not(target_arch = "wasm32"))]
fn shared_covering(path: &str, covering: &str) -> Option<std::sync::Arc<TtfFont>> {
    let covers = |f: &TtfFont| {
        covering
            .chars()
            .filter(|c| !c.is_whitespace() && !c.is_control())
            .all(|c| f.cmap.contains_key(&(c as u32)))
    };
    let mut cache = face_cache().lock().unwrap_or_else(|e| e.into_inner());
    let entry = cache.entry(path.to_string()).or_default();
    if entry.missing {
        return None;
    }
    if let Some(found) = entry.faces.iter().find(|f| covers(f)) {
        return Some(found.clone());
    }
    if entry.complete {
        return None;
    }
    // Not found among what is parsed: read the file and go on from the next face.
    let Ok(bytes) = std::fs::read(path) else {
        entry.missing = true;
        return None;
    };
    while entry.faces.len() < 64 {
        let Ok(face) = TtfFont::parse_face(&bytes, entry.faces.len() as u32, &[]) else {
            entry.complete = true;
            return None;
        };
        let face = std::sync::Arc::new(face);
        entry.faces.push(face.clone());
        if covers(&face) {
            return Some(face);
        }
    }
    entry.complete = true;
    None
}

/// How many system files have faces held, for tests and for a program that wants to know what it is keeping.
#[cfg(not(target_arch = "wasm32"))]
pub fn system_face_cache_len() -> usize {
    face_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .values()
        .filter(|f| !f.faces.is_empty())
        .count()
}

/// Lets go of every cached system face. Faces already handed out keep their own reference.
#[cfg(not(target_arch = "wasm32"))]
pub fn clear_system_face_cache() {
    face_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clear();
}

/// Where faces for `language` are usually kept, best first, on every platform
/// (the ones that do not exist are skipped when they are read). The last
/// entries are the broad faces that cover several scripts at once.
pub fn system_font_paths(language: &str) -> Vec<std::path::PathBuf> {
    let tag = language.replace('_', "-").to_ascii_lowercase();
    let script = match tag.as_str() {
        "ja" | "ja-jp" => "ja",
        "zh" | "zh-cn" | "zh-hans" | "zh-sg" => "zh",
        "zh-tw" | "zh-hant" | "zh-hk" | "zh-mo" => "zh-hant",
        "ko" | "ko-kr" => "ko",
        "hi" | "hi-in" | "mr" | "ne" | "sa" => "hi",
        "th" | "th-th" => "th",
        "he" | "iw" | "he-il" => "he",
        "ar" | "ar-sa" | "fa" | "ur" => "ar",
        _ => return Vec::new(),
    };
    let mac = "/System/Library/Fonts";
    let mac_extra = "/System/Library/Fonts/Supplemental";
    let win = "C:/Windows/Fonts";
    let linux = [
        "/usr/share/fonts/truetype/noto",
        "/usr/share/fonts/opentype/noto",
        "/usr/share/fonts/noto-cjk",
        "/usr/share/fonts/truetype/noto",
        "/usr/share/fonts/noto",
        "/usr/share/fonts/google-noto-cjk",
        "/usr/share/fonts/google-noto",
    ];
    let mut paths: Vec<String> = Vec::new();
    let mut add = |dir: &str, name: &str| paths.push(format!("{dir}/{name}"));
    match script {
        "ja" => {
            add(mac, "ヒラギノ角ゴシック W3.ttc");
            add(mac, "ヒラギノ丸ゴ ProN W4.ttc");
            for f in ["YuGothR.ttc", "YuGothM.ttc", "meiryo.ttc", "msgothic.ttc"] {
                add(win, f);
            }
            for d in linux {
                add(d, "NotoSansCJK-Regular.ttc");
                add(d, "NotoSansCJKjp-Regular.otf");
            }
        }
        "zh" => {
            add(mac, "Hiragino Sans GB.ttc");
            add(mac, "STHeiti Light.ttc");
            add(mac_extra, "Songti.ttc");
            for f in ["msyh.ttc", "msyh.ttf", "simsun.ttc", "simhei.ttf"] {
                add(win, f);
            }
            for d in linux {
                add(d, "NotoSansCJK-Regular.ttc");
                add(d, "NotoSansCJKsc-Regular.otf");
            }
        }
        "zh-hant" => {
            add(mac, "STHeiti Light.ttc");
            add(mac_extra, "Songti.ttc");
            add(mac, "Hiragino Sans GB.ttc");
            for f in ["msjh.ttc", "msjh.ttf", "mingliu.ttc"] {
                add(win, f);
            }
            for d in linux {
                add(d, "NotoSansCJK-Regular.ttc");
                add(d, "NotoSansCJKtc-Regular.otf");
            }
        }
        "ko" => {
            add(mac, "AppleSDGothicNeo.ttc");
            for f in ["malgun.ttf", "gulim.ttc", "batang.ttc"] {
                add(win, f);
            }
            for d in linux {
                add(d, "NotoSansCJK-Regular.ttc");
                add(d, "NotoSansCJKkr-Regular.otf");
            }
        }
        "hi" => {
            // The system's own: one face per Indic script, Devanagari the second of them.
            add(mac, "SFIndia.ttc");
            add(mac, "Kohinoor.ttc");
            add(mac_extra, "DevanagariMT.ttc");
            add(mac_extra, "ITFDevanagari.ttc");
            for f in ["Nirmala.ttf", "nirmala.ttf", "mangal.ttf"] {
                add(win, f);
            }
            for d in linux {
                add(d, "NotoSansDevanagari-Regular.ttf");
            }
        }
        "he" => {
            add(mac, "SFHebrew.ttf");
            add(mac_extra, "Arial Hebrew.ttc");
            for f in ["arial.ttf", "david.ttf"] {
                add(win, f);
            }
            for d in linux {
                add(d, "NotoSansHebrew-Regular.ttf");
            }
        }
        "ar" => {
            add(mac, "SFArabic.ttf");
            add(mac, "GeezaPro.ttc");
            for f in ["arial.ttf", "segoeui.ttf", "tahoma.ttf"] {
                add(win, f);
            }
            for d in linux {
                add(d, "NotoSansArabic-Regular.ttf");
            }
        }
        "th" => {
            // Tahoma draws Thai larger and cleaner than the system UI face.
            add(mac_extra, "Tahoma.ttf");
            add(mac, "ThonburiUI.ttc");
            add(mac_extra, "Sathu.ttf");
            for f in ["LeelawUI.ttf", "leelawui.ttf", "tahoma.ttf"] {
                add(win, f);
            }
            for d in linux {
                add(d, "NotoSansThai-Regular.ttf");
            }
        }
        _ => {}
    }
    // Where the packages name their files differently (Noto has Thai as `NotoSansThai`, `NotoLoopedThai` or
    // `NotoSansThaiUI`, by distribution), look in the folders for the script's name.
    let keywords: &[&str] = match script {
        "hi" => &["devanagari"],
        "th" => &["thai"],
        "he" => &["hebrew"],
        "ar" => &["arabic"],
        _ => &[],
    };
    if !keywords.is_empty() {
        paths.extend(scan_script_fonts(&linux, keywords));
    }
    // The broad faces, last: they have all of the script, rarely well.
    paths.push(format!("{mac_extra}/Arial Unicode.ttf"));
    paths.push("/Library/Fonts/Arial Unicode.ttf".to_string());
    paths.into_iter().map(Into::into).collect()
}

/// The font files in `dirs` whose names mention one of `keywords`, the plain-weight sans ones first: `NotoSansThai-Regular.ttf`
/// before `NotoSerifThai-Bold.ttf`. Folders that do not exist are skipped.
fn scan_script_fonts(dirs: &[&str], keywords: &[&str]) -> Vec<String> {
    let mut found: Vec<(u8, String)> = Vec::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let lower = name.to_ascii_lowercase();
            if !(lower.ends_with(".ttf") || lower.ends_with(".otf") || lower.ends_with(".ttc"))
                || !keywords.iter().any(|k| lower.contains(k))
            {
                continue;
            }
            // Lower ranks first: regular beats bold and italic, sans beats serif, and the plain design beats a stylised one (Noto's
            // looped Thai, which comes first alphabetically).
            let rank = u8::from(!lower.contains("regular")) * 4
                + u8::from(lower.contains("serif")) * 2
                + u8::from(
                    lower.contains("bold")
                        || lower.contains("italic")
                        || lower.contains("mono")
                        || lower.contains("looped"),
                );
            found.push((rank, format!("{dir}/{name}")));
        }
    }
    found.sort();
    found.into_iter().map(|(_, p)| p).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dejavu_bytes() -> &'static [u8] {
        include_bytes!("../openscad/DejaVuSans.ttf")
    }

    #[test]
    fn a_face_covers_the_text_it_has_letters_for() {
        let font = CaptionFont::from_ttf_bytes(dejavu_bytes()).unwrap();
        assert!(font.covers("Hamburgefonstiv 0123, Ünïcödé"));
        assert!(font.covers("  \t\n"), "white space needs no glyph");
        assert!(font.covers(""));
        assert!(!font.covers("日本語"), "DejaVu has no Han");
        assert!(!font.covers("abc한"), "one missing letter is enough");
        // The bitmap face has no Han either.
        assert!(!CaptionFont::builtin().covers("日"));
    }

    #[test]
    fn a_collection_gives_the_first_face_that_covers_the_text() {
        let face =
            CaptionFont::from_collection(dejavu_bytes(), "abc").expect("a plain font is one face");
        assert!(face.covers("abc"));
        assert!(
            CaptionFont::from_collection(dejavu_bytes(), "日本語").is_none(),
            "no face has it"
        );
        assert!(CaptionFont::from_collection(b"not a font at all", "a").is_none());
        assert!(CaptionFont::from_collection(&[], "a").is_none());
        // A collection header with no faces behind it ends the search at once.
        let mut ttc = b"ttcf\x00\x01\x00\x00\x00\x00\x00\x00".to_vec();
        ttc.extend([0u8; 16]);
        assert!(CaptionFont::from_collection(&ttc, "a").is_none());
    }

    #[test]
    fn each_language_that_needs_a_face_has_places_to_look() {
        for tag in [
            "ja", "zh", "zh-CN", "zh-Hans", "zh-TW", "zh_HK", "zh-Hant", "ko", "hi", "th", "he",
            "ar",
        ] {
            let paths = system_font_paths(tag);
            assert!(paths.len() > 3, "{tag}: {paths:?}");
            assert!(
                paths.iter().any(|p| p.ends_with("Arial Unicode.ttf")),
                "{tag} ends with the broad face"
            );
        }
        // Simplified and Traditional start in different places.
        assert_ne!(system_font_paths("zh")[0], system_font_paths("zh-TW")[0]);
        // The scripts the built-in faces cover need nothing.
        for tag in ["en", "fr", "ru", "vi", "id", "", "xx"] {
            assert!(system_font_paths(tag).is_empty(), "{tag}");
        }
    }

    /// With the system's faces behind a Latin one, a line of all five scripts is drawn, each by its own face; and
    /// the language asked for first decides who draws the Han characters the languages share.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_system_faces_can_stand_behind_a_face_and_the_preferred_language_goes_first() {
        let text = "Hello 日本語 中文 한국어 हिन्दी ไทย";
        let chain = CaptionFont::builtin().with_system_fallbacks(&["ko"]);
        // Hebrew and Arabic are there too, where the machine has them.
        let semitic = CaptionFont::builtin().with_system_fallbacks(&[]);
        if semitic.fallback_count() >= 7 {
            assert!(semitic.can_draw_all("שלום مرحبا"));
        }
        if chain.fallback_count() < 5 {
            return; // a machine without the faces is not a failure of the code
        }
        assert!(chain.can_draw_all(text));
        assert!(!CaptionFont::builtin().can_draw_all(text));
        // Korean first: its face is the first fallback, and has the Hangul.
        assert_eq!(chain.gid('한') >> 24, 1);
        // A Han character both the Japanese and the Chinese face have goes to whichever came first.
        let zh_first = CaptionFont::builtin().with_system_fallbacks(&["zh"]);
        let ja_first = CaptionFont::builtin().with_system_fallbacks(&["ja"]);
        assert_eq!(zh_first.gid('中') >> 24, 1);
        assert_eq!(ja_first.gid('中') >> 24, 1);
        // The same languages again add nothing.
        assert_eq!(
            CaptionFont::builtin()
                .with_system_fallbacks(&["zh", "zh-TW", "zh"])
                .fallback_count(),
            chain.fallback_count()
        );
    }

    #[test]
    fn bold_and_italic_files_are_found_beside_a_regular_one_by_the_ways_fonts_are_named() {
        let dir = std::env::temp_dir().join(format!("threers-family-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let touch = |names: &[&str]| {
            names
                .iter()
                .for_each(|n| std::fs::write(dir.join(n), b"x").unwrap())
        };
        touch(&[
            "Alpha-Regular.ttf",
            "Alpha-Bold.ttf",
            "Alpha-Italic.ttf",
            "Alpha-BoldItalic.ttf",
            "Beta-R.ttf",
            "Beta-B.ttf",
            "Beta-RI.ttf",
            "Beta-BI.ttf",
        ]);
        touch(&[
            "Gamma.ttf",
            "Gamma-Bold.ttf",
            "Gamma-Oblique.ttf",
            "Gamma-BoldOblique.ttf",
            "delta.ttf",
            "deltab.ttf",
            "deltai.ttf",
            "deltaz.ttf",
            "eps.ttf",
            "epsbd.ttf",
            "Zeta.ttf",
        ]);
        let found = |regular: &str| {
            siblings(dir.join(regular).to_str().unwrap())
                .map(|p| p.map(|p| p.file_name().unwrap().to_str().unwrap().to_string()))
        };
        let s = |n: &str| Some(n.to_string());
        assert_eq!(
            found("Alpha-Regular.ttf"),
            [
                s("Alpha-Bold.ttf"),
                s("Alpha-Italic.ttf"),
                s("Alpha-BoldItalic.ttf")
            ]
        );
        assert_eq!(
            found("Beta-R.ttf"),
            [s("Beta-B.ttf"), s("Beta-RI.ttf"), s("Beta-BI.ttf")]
        );
        assert_eq!(
            found("Gamma.ttf"),
            [
                s("Gamma-Bold.ttf"),
                s("Gamma-Oblique.ttf"),
                s("Gamma-BoldOblique.ttf")
            ]
        );
        assert_eq!(
            found("delta.ttf"),
            [s("deltab.ttf"), s("deltai.ttf"), s("deltaz.ttf")]
        );
        assert_eq!(
            found("eps.ttf"),
            [s("epsbd.ttf"), None, None],
            "only what exists"
        );
        assert_eq!(found("Zeta.ttf"), [None, None, None]);
        assert_eq!(siblings("no extension"), [None, None, None]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn the_monospaced_and_serif_faces_are_what_they_say() {
        if let Some(mut mono) = CaptionFont::system_mono() {
            assert!(mono.covers("Hamburgefonstiv 0123"));
            assert!(
                (mono.advance('i', 16.0) - mono.advance('m', 16.0)).abs() < 0.5,
                "a monospaced face: {} vs {}",
                mono.advance('i', 16.0),
                mono.advance('m', 16.0)
            );
        }
        if let (Some(mut serif), Some(mut sans)) =
            (CaptionFont::system_serif(), CaptionFont::system_sans())
        {
            assert!(serif.covers("Hamburgefonstiv 0123"));
            // Not the same face as the sans: some letter is a different width.
            assert!("Hamburgefonstiv"
                .chars()
                .any(|c| (serif.advance(c, 20.0) - sans.advance(c, 20.0)).abs() > 0.3));
        }
    }

    /// A system file is parsed once: the second language for a script takes what the first parsed.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn a_system_face_is_parsed_once_and_shared() {
        // Other tests load system faces into the same process-wide cache while
        // this runs, so it checks the sharing itself, not how full the cache is.
        let text = "한국어";
        let Some(path) = system_font_paths("ko")
            .into_iter()
            .filter_map(|p| p.to_str().map(str::to_owned))
            .find(|p| shared_covering(p, text).is_some())
        else {
            return;
        };
        let first = shared_covering(&path, text).expect("the face found above");
        let again = shared_covering(&path, text).expect("the same again");
        assert!(
            std::sync::Arc::ptr_eq(&first, &again),
            "the second asking parsed nothing new"
        );
        let mut font = CaptionFont::system_for_language("ko", text).expect("a Korean face");
        assert!(font.covers(text));
        assert!(font.advance('한', 16.0) > 8.0);
        clear_system_face_cache();
        // What was handed out still draws, and the next asking parses afresh.
        assert!(font.glyph('한', 16.0).width > 0);
        let fresh = shared_covering(&path, text).expect("parsed again");
        assert!(!std::sync::Arc::ptr_eq(&first, &fresh));
    }

    /// The interface face is a real sans where the machine has one, not the monospaced bitmap.
    #[test]
    fn the_system_has_a_sans_face_for_an_interface() {
        let Some(font) = CaptionFont::system_sans() else {
            // A machine with no fonts at all is not a failure of the code.
            assert!(
                [
                    "/usr/share/fonts",
                    "/System/Library/Fonts",
                    "C:/Windows/Fonts"
                ]
                .iter()
                .all(|d| !std::path::Path::new(d).exists())
                    || cfg!(not(any(
                        target_os = "macos",
                        target_os = "linux",
                        target_os = "windows"
                    )))
            );
            return;
        };
        assert!(font.covers("Hamburgefonstiv 0123 Привет éüñ"));
        let mut font = font;
        // It is proportional: an i is narrower than an m, which the bitmap face never is.
        assert!(font.advance('i', 16.0) < font.advance('m', 16.0) - 2.0);
        assert!(!CaptionFont::builtin().covers("Привет"));
    }

    #[test]
    fn script_fonts_are_found_by_name_and_the_plain_ones_come_first() {
        let dir = std::env::temp_dir().join(format!("threers-fonts-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for name in [
            "NotoSerifThai-Bold.ttf",
            "NotoSansThai-Regular.ttf",
            "NotoSansThai-Bold.ttf",
            "NotoLoopedThai-Regular.ttf",
            "NotoSansHebrew-Regular.ttf",
            "readme.txt",
            "NotoSansThai.json",
        ] {
            std::fs::write(dir.join(name), b"x").unwrap();
        }
        let found: Vec<String> =
            scan_script_fonts(&[dir.to_str().unwrap(), "/no/such/folder"], &["thai"])
                .into_iter()
                .map(|p| p.rsplit('/').next().unwrap().to_string())
                .collect();
        assert_eq!(found.len(), 4, "{found:?}");
        // The plain regular sans first, then the stylised and the bold, then the serif.
        assert_eq!(found[0], "NotoSansThai-Regular.ttf", "{found:?}");
        assert_eq!(found[3], "NotoSerifThai-Bold.ttf");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// On a Mac the system's own faces are there: each language gets one that covers its text.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_mac_has_a_face_for_each_asian_language() {
        for (tag, text) in [
            ("ja", "設定 保存 ミックス表示"),
            ("zh", "设置 保存 显示混音"),
            ("zh-TW", "設定 儲存 顯示混音"),
            ("ko", "설정 저장 믹스 보기"),
            ("hi", "सहेजें मिक्स दिखाएँ"),
            ("th", "บันทึก แสดงมิกซ์"),
        ] {
            let Some(font) = CaptionFont::system_for_language(tag, text) else {
                // A machine with the fonts removed is not a failure of the code.
                assert!(
                    system_font_paths(tag)
                        .iter()
                        .all(|p| !p.exists() || std::fs::read(p).is_err()),
                    "{tag}: a face exists but none covered {text}"
                );
                continue;
            };
            assert!(font.covers(text), "{tag}");
        }
        // A language that needs none gets none.
        assert!(CaptionFont::system_for_language("en", "abc").is_none());
    }
}
