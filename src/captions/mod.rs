//! Subtitles and captions — a timed-text track you can burn into exported
//! video, ship as a sidecar `.srt`/`.vtt`, mux as a soft subtitle track, or
//! draw on screen over a live render.
//!
//! Pure Rust with no dependencies and no OS/process/filesystem use, so it
//! builds and runs on `wasm32` as well as native (mirroring
//! [`crate::codec`]).
//!
//! ```
//! use threers::captions::{CaptionTrack, CaptionPainter};
//!
//! let track = CaptionTrack::parse_srt(
//!     "1\n00:00:00,000 --> 00:00:02,500\nHello, world!\n",
//! ).unwrap();
//! assert_eq!(track.active_at(1.0).len(), 1);
//!
//! // Burn the active cue into an RGBA frame.
//! let (w, h) = (320u32, 180u32);
//! let mut frame = vec![0u8; (w * h * 4) as usize];
//! let mut painter = CaptionPainter::new();
//! painter.burn_in(&mut frame, w, h, &track, 1.0);
//! ```
//!
//! | Piece | Role |
//! |-------|------|
//! | [`Cue`] / [`CaptionTrack`] | timed-text model, `active_at` lookup |
//! | [`srt`] / [`vtt`] | SubRip and WebVTT parse + serialize |
//! | [`CaptionStyle`] | type size, colors, outline, box, placement |
//! | [`CaptionFont`] | built-in 8×8 bitmap face, or a TrueType face |
//! | [`CaptionPainter`] | wraps text and composites it into RGBA pixels |
//! | [`CaptionOverlay`] | caches a frame-sized overlay for on-screen drawing |
//!
//! Video export wires these in through
//! [`VideoOptions::captions`](crate::video::VideoOptions::captions); the wgpu
//! renderer draws them through
//! [`Renderer::draw_caption_overlay`](crate::renderer::Renderer::draw_caption_overlay).

pub mod font;
pub mod layout;
pub mod overlay;
pub mod raster;
pub mod srt;
pub mod vtt;

pub use font::{CaptionFont, GlyphBitmap};
pub use layout::{LaidOutText, LineBox};
pub use overlay::CaptionOverlay;
pub use raster::CaptionPainter;

/// Horizontal alignment of caption lines within the caption box.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CaptionAlign {
    Left,
    #[default]
    Center,
    Right,
}

/// Which edge of the frame the caption block is anchored to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CaptionAnchor {
    Top,
    Middle,
    #[default]
    Bottom,
}

/// Timed-text file format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptionFormat {
    /// SubRip (`.srt`) — the most widely accepted sidecar format.
    Srt,
    /// WebVTT (`.vtt`) — the format `<track>` takes in browsers.
    Vtt,
}

impl CaptionFormat {
    /// File extension without the dot (`"srt"`, `"vtt"`).
    pub fn extension(self) -> &'static str {
        match self {
            CaptionFormat::Srt => "srt",
            CaptionFormat::Vtt => "vtt",
        }
    }

    /// MIME type, for browser downloads / `Blob` construction.
    pub fn mime_type(self) -> &'static str {
        match self {
            CaptionFormat::Srt => "application/x-subrip",
            CaptionFormat::Vtt => "text/vtt",
        }
    }

    /// Infer the format from a file name / path extension.
    pub fn from_path(path: &str) -> Option<Self> {
        let ext = path.rsplit('.').next()?.to_ascii_lowercase();
        match ext.as_str() {
            "srt" => Some(CaptionFormat::Srt),
            "vtt" | "webvtt" => Some(CaptionFormat::Vtt),
            _ => None,
        }
    }
}

/// Why parsing a subtitle file failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CaptionError {
    /// A timestamp line was not `HH:MM:SS,mmm --> HH:MM:SS,mmm`.
    BadTimestamp { line: usize, text: String },
    /// A cue block had no `-->` timing line.
    MissingTiming { line: usize },
    /// The input had no cues at all.
    Empty,
    /// Text did not look like the requested format (e.g. no `WEBVTT` header).
    NotRecognized,
}

impl std::fmt::Display for CaptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CaptionError::BadTimestamp { line, text } => {
                write!(f, "line {line}: malformed cue timing {text:?}")
            }
            CaptionError::MissingTiming { line } => {
                write!(f, "line {line}: cue block has no \"-->\" timing line")
            }
            CaptionError::Empty => write!(f, "no cues found"),
            CaptionError::NotRecognized => write!(f, "input is not SubRip or WebVTT"),
        }
    }
}
impl std::error::Error for CaptionError {}

/// One timed caption: a time range plus the text shown during it.
///
/// Times are seconds from the start of the media. `text` keeps its original
/// line breaks (`\n`) and any inline markup; use [`Cue::plain_text`] for a
/// markup-free version suitable for rasterizing.
#[derive(Clone, Debug, PartialEq)]
pub struct Cue {
    /// Optional cue identifier (the numeric counter in SubRip, the free-form
    /// name in WebVTT). Regenerated on serialize when `None`.
    pub id: Option<String>,
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
    /// Cue text; `\n` separates lines.
    pub text: String,
    /// Per-cue horizontal alignment override (WebVTT `align:`).
    pub align: Option<CaptionAlign>,
    /// Per-cue vertical placement as a fraction of frame height from the top
    /// (WebVTT `line:NN%`). `None` uses the style's anchor + margin.
    pub line: Option<f32>,
    /// Per-cue horizontal placement as a fraction of frame width from the left
    /// (WebVTT `position:NN%`). `None` centers within the style's max width.
    pub position: Option<f32>,
}

impl Cue {
    /// A cue spanning `start..end` seconds showing `text`.
    pub fn new(start: f64, end: f64, text: impl Into<String>) -> Self {
        Self {
            id: None,
            start,
            end,
            text: text.into(),
            align: None,
            line: None,
            position: None,
        }
    }

    /// Set the cue identifier.
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Override horizontal alignment for this cue.
    pub fn align(mut self, align: CaptionAlign) -> Self {
        self.align = Some(align);
        self
    }

    /// Place the cue vertically at `line` (0 = top, 1 = bottom of the frame).
    pub fn at_line(mut self, line: f32) -> Self {
        self.line = Some(line);
        self
    }

    /// Place the cue horizontally at `position` (0 = left, 1 = right).
    pub fn at_position(mut self, position: f32) -> Self {
        self.position = Some(position);
        self
    }

    /// Duration in seconds (never negative).
    pub fn duration(&self) -> f64 {
        (self.end - self.start).max(0.0)
    }

    /// Whether the cue is showing at `time` seconds (start inclusive, end
    /// exclusive — so back-to-back cues never both match).
    pub fn is_active_at(&self, time: f64) -> bool {
        time >= self.start && time < self.end
    }

    /// Cue text with inline markup removed and HTML entities decoded — what
    /// the rasterizer draws.
    ///
    /// Strips WebVTT/SubRip tags (`<i>`, `<b>`, `<u>`, `<c.yellow>`, `<v Bob>`,
    /// `<00:00:01.000>`), ASS-style overrides (`{\an8}`), and decodes `&amp;`,
    /// `&lt;`, `&gt;`, `&nbsp;`, `&lrm;`, `&rlm;`.
    pub fn plain_text(&self) -> String {
        strip_markup(&self.text)
    }
}

/// Remove inline caption markup and decode the entity set WebVTT defines.
pub fn strip_markup(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            // <i> </i> <c.classname> <v Speaker> <00:00:01.000> …
            '<' => {
                for t in chars.by_ref() {
                    if t == '>' {
                        break;
                    }
                }
            }
            // ASS/SSA override blocks such as {\an8}.
            '{' if matches!(chars.peek(), Some('\\')) => {
                for t in chars.by_ref() {
                    if t == '}' {
                        break;
                    }
                }
            }
            '&' => {
                let mut name = String::new();
                let mut closed = false;
                // Entities are short; anything longer is literal text.
                while let Some(&t) = chars.peek() {
                    if t == ';' {
                        chars.next();
                        closed = true;
                        break;
                    }
                    if !t.is_ascii_alphanumeric() && t != '#' || name.len() > 8 {
                        break;
                    }
                    name.push(t);
                    chars.next();
                }
                match (closed, name.as_str()) {
                    (true, "amp") => out.push('&'),
                    (true, "lt") => out.push('<'),
                    (true, "gt") => out.push('>'),
                    (true, "quot") => out.push('"'),
                    (true, "apos") => out.push('\''),
                    (true, "nbsp") => out.push(' '),
                    // Bidi marks are invisible controls — drop them.
                    (true, "lrm") | (true, "rlm") => {}
                    _ => {
                        out.push('&');
                        out.push_str(&name);
                        if closed {
                            out.push(';');
                        }
                    }
                }
            }
            _ => out.push(c),
        }
    }
    out
}

/// An ordered list of [`Cue`]s plus the metadata a subtitle track carries.
///
/// ```
/// use threers::captions::CaptionTrack;
/// let track = CaptionTrack::new()
///     .language("en")
///     .label("English")
///     .cue(0.0, 2.0, "First line")
///     .cue(2.0, 4.0, "Second line");
/// assert_eq!(track.duration(), 4.0);
/// assert!(track.to_vtt().starts_with("WEBVTT"));
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CaptionTrack {
    /// Cues, kept sorted by start time by [`push`](Self::push) / [`cue`](Self::cue).
    pub cues: Vec<Cue>,
    /// BCP-47 language tag (`"en"`, `"pt-BR"`, …). Empty means undetermined.
    pub language: String,
    /// Human-readable track name shown in player menus.
    pub label: String,
}

impl CaptionTrack {
    /// An empty track.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the BCP-47 language tag.
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    /// Set the human-readable track label.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Append a cue (builder form). Keeps `cues` sorted by start time.
    pub fn cue(mut self, start: f64, end: f64, text: impl Into<String>) -> Self {
        self.push(Cue::new(start, end, text));
        self
    }

    /// Append a cue, keeping `cues` sorted by start time.
    pub fn push(&mut self, cue: Cue) {
        let at = self
            .cues
            .iter()
            .rposition(|c| c.start <= cue.start)
            .map(|i| i + 1)
            .unwrap_or(0);
        self.cues.insert(at, cue);
    }

    /// Re-sort cues by start time (then end time). Only needed if `cues` was
    /// mutated directly.
    pub fn sort(&mut self) {
        self.cues.sort_by(|a, b| {
            a.start
                .partial_cmp(&b.start)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    a.end
                        .partial_cmp(&b.end)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });
    }

    /// Number of cues.
    pub fn len(&self) -> usize {
        self.cues.len()
    }

    /// Whether the track has no cues.
    pub fn is_empty(&self) -> bool {
        self.cues.is_empty()
    }

    /// End time of the last cue, in seconds (`0.0` when empty).
    pub fn duration(&self) -> f64 {
        self.cues.iter().fold(0.0f64, |acc, c| acc.max(c.end))
    }

    /// Every cue showing at `time` seconds, in track order.
    pub fn active_at(&self, time: f64) -> Vec<&Cue> {
        self.cues.iter().filter(|c| c.is_active_at(time)).collect()
    }

    /// The first cue showing at `time` seconds — the common case, since most
    /// tracks show one cue at a time.
    pub fn cue_at(&self, time: f64) -> Option<&Cue> {
        self.cues.iter().find(|c| c.is_active_at(time))
    }

    /// Concatenated [`plain_text`](Cue::plain_text) of every cue active at
    /// `time`, joined with newlines. Empty when nothing is showing.
    pub fn text_at(&self, time: f64) -> String {
        let mut out = String::new();
        for c in self.cues.iter().filter(|c| c.is_active_at(time)) {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(c.plain_text().trim_end_matches('\n'));
        }
        out
    }

    /// Shift every cue by `seconds` (negative moves earlier). Cues never move
    /// before zero.
    pub fn shift(&mut self, seconds: f64) {
        for c in &mut self.cues {
            c.start = (c.start + seconds).max(0.0);
            c.end = (c.end + seconds).max(0.0);
        }
    }

    /// Scale every cue time by `factor` — e.g. `24.0 / 25.0` to retime a 25 fps
    /// track for 24 fps playback.
    pub fn scale(&mut self, factor: f64) {
        for c in &mut self.cues {
            c.start *= factor;
            c.end *= factor;
        }
    }

    /// Parse SubRip (`.srt`).
    pub fn parse_srt(text: &str) -> Result<Self, CaptionError> {
        srt::parse(text)
    }

    /// Parse WebVTT (`.vtt`).
    pub fn parse_vtt(text: &str) -> Result<Self, CaptionError> {
        vtt::parse(text)
    }

    /// Parse either format, sniffing the `WEBVTT` magic.
    pub fn parse(text: &str) -> Result<Self, CaptionError> {
        if text
            .trim_start_matches('\u{feff}')
            .trim_start()
            .starts_with("WEBVTT")
        {
            vtt::parse(text)
        } else {
            srt::parse(text)
        }
    }

    /// Serialize to SubRip.
    pub fn to_srt(&self) -> String {
        srt::write(self)
    }

    /// Serialize to WebVTT.
    pub fn to_vtt(&self) -> String {
        vtt::write(self)
    }

    /// Serialize to `format`.
    pub fn to_format(&self, format: CaptionFormat) -> String {
        match format {
            CaptionFormat::Srt => self.to_srt(),
            CaptionFormat::Vtt => self.to_vtt(),
        }
    }
}

/// Look, placement, and wrapping of burned-in / on-screen captions.
///
/// Sizes are in pixels at the frame resolution the caption is drawn into. Use
/// [`scaled`](Self::scaled) to keep the look consistent across resolutions.
#[derive(Clone, Debug, PartialEq)]
pub struct CaptionStyle {
    /// Cap-to-descender type size in pixels.
    pub font_size: f32,
    /// Fill color, RGBA.
    pub color: [u8; 4],
    /// Outline (stroke) color, RGBA. Alpha `0` disables the outline.
    pub outline_color: [u8; 4],
    /// Outline half-width in pixels. `0` disables the outline.
    pub outline_width: f32,
    /// Drop-shadow color, RGBA. Alpha `0` disables the shadow.
    pub shadow_color: [u8; 4],
    /// Drop-shadow offset in pixels, `[dx, dy]`.
    pub shadow_offset: [f32; 2],
    /// Background box color, RGBA. Alpha `0` disables the box.
    pub background: [u8; 4],
    /// Padding between the text and the background box edge, in pixels.
    pub padding: f32,
    /// Line advance as a multiple of `font_size`.
    pub line_height: f32,
    /// Extra tracking between glyphs, in pixels.
    pub letter_spacing: f32,
    /// Horizontal alignment of lines within the caption block.
    pub align: CaptionAlign,
    /// Frame edge the block is anchored to.
    pub anchor: CaptionAnchor,
    /// Distance from the anchored edge, in pixels. Ignored for
    /// [`CaptionAnchor::Middle`].
    pub margin: f32,
    /// Widest a line may get before wrapping, as a fraction of frame width.
    pub max_width: f32,
}

impl Default for CaptionStyle {
    /// Broadcast-style bottom-centered white text with a black outline and a
    /// translucent box, sized for 1080p.
    fn default() -> Self {
        Self {
            font_size: 34.0,
            color: [255, 255, 255, 255],
            outline_color: [0, 0, 0, 255],
            outline_width: 2.0,
            shadow_color: [0, 0, 0, 0],
            shadow_offset: [2.0, 2.0],
            background: [0, 0, 0, 140],
            padding: 10.0,
            line_height: 1.25,
            letter_spacing: 0.0,
            align: CaptionAlign::Center,
            anchor: CaptionAnchor::Bottom,
            margin: 48.0,
            max_width: 0.8,
        }
    }
}

impl CaptionStyle {
    /// Type size in pixels.
    pub fn font_size(mut self, px: f32) -> Self {
        self.font_size = px.max(1.0);
        self
    }

    /// Text fill color.
    pub fn color(mut self, rgba: [u8; 4]) -> Self {
        self.color = rgba;
        self
    }

    /// Outline color and half-width in pixels. Width `0` removes the outline.
    pub fn outline(mut self, rgba: [u8; 4], width: f32) -> Self {
        self.outline_color = rgba;
        self.outline_width = width.max(0.0);
        self
    }

    /// Drop shadow color and offset. Alpha `0` removes the shadow.
    pub fn shadow(mut self, rgba: [u8; 4], dx: f32, dy: f32) -> Self {
        self.shadow_color = rgba;
        self.shadow_offset = [dx, dy];
        self
    }

    /// Background box color. Alpha `0` removes the box.
    pub fn background(mut self, rgba: [u8; 4]) -> Self {
        self.background = rgba;
        self
    }

    /// Padding inside the background box, in pixels.
    pub fn padding(mut self, px: f32) -> Self {
        self.padding = px.max(0.0);
        self
    }

    /// Line advance as a multiple of the type size.
    pub fn line_height(mut self, factor: f32) -> Self {
        self.line_height = factor.max(0.1);
        self
    }

    /// Extra tracking between glyphs, in pixels.
    pub fn letter_spacing(mut self, px: f32) -> Self {
        self.letter_spacing = px;
        self
    }

    /// Horizontal alignment of lines within the block.
    pub fn align(mut self, align: CaptionAlign) -> Self {
        self.align = align;
        self
    }

    /// Frame edge to anchor the block to.
    pub fn anchor(mut self, anchor: CaptionAnchor) -> Self {
        self.anchor = anchor;
        self
    }

    /// Distance from the anchored edge, in pixels.
    pub fn margin(mut self, px: f32) -> Self {
        self.margin = px.max(0.0);
        self
    }

    /// Wrap width as a fraction (`0..=1`) of the frame width.
    pub fn max_width(mut self, fraction: f32) -> Self {
        self.max_width = fraction.clamp(0.05, 1.0);
        self
    }

    /// A copy with every pixel measurement multiplied by `factor` — the way to
    /// keep one style looking right at several output sizes.
    ///
    /// ```
    /// use threers::captions::CaptionStyle;
    /// // 1080p defaults, retargeted to 720p.
    /// let style = CaptionStyle::default().scaled(720.0 / 1080.0);
    /// assert!((style.font_size - 34.0 * 2.0 / 3.0).abs() < 1e-4);
    /// ```
    pub fn scaled(&self, factor: f32) -> Self {
        let f = factor.max(0.01);
        Self {
            font_size: self.font_size * f,
            outline_width: self.outline_width * f,
            shadow_offset: [self.shadow_offset[0] * f, self.shadow_offset[1] * f],
            padding: self.padding * f,
            letter_spacing: self.letter_spacing * f,
            margin: self.margin * f,
            ..self.clone()
        }
    }

    /// A copy sized for a frame `height` pixels tall, treating the current
    /// values as authored for 1080p.
    ///
    /// ```
    /// use threers::captions::CaptionStyle;
    /// let s = CaptionStyle::default().for_height(2160);
    /// assert!((s.font_size - 68.0).abs() < 1e-4);
    /// ```
    pub fn for_height(&self, height: u32) -> Self {
        self.scaled(height.max(1) as f32 / 1080.0)
    }
}

/// Best-effort ISO-639-2/T three-letter language code for a BCP-47 tag, as MP4
/// (`mdhd.language`) and Matroska (`Language`) both want.
///
/// Three-letter tags pass through, common two-letter tags are mapped, and
/// anything unrecognized becomes `und` (undetermined).
///
/// ```
/// use threers::captions::iso639_2;
/// assert_eq!(&iso639_2("en-US"), b"eng");
/// assert_eq!(&iso639_2("por"), b"por");
/// assert_eq!(&iso639_2(""), b"und");
/// ```
pub fn iso639_2(language: &str) -> [u8; 3] {
    let primary = language
        .split(['-', '_'])
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    let bytes = primary.as_bytes();
    if bytes.len() == 3 && bytes.iter().all(|b| b.is_ascii_lowercase()) {
        return [bytes[0], bytes[1], bytes[2]];
    }
    let mapped = match primary.as_str() {
        "aa" => "aar",
        "ab" => "abk",
        "af" => "afr",
        "am" => "amh",
        "ar" => "ara",
        "az" => "aze",
        "be" => "bel",
        "bg" => "bul",
        "bn" => "ben",
        "bs" => "bos",
        "ca" => "cat",
        "cs" => "ces",
        "cy" => "cym",
        "da" => "dan",
        "de" => "deu",
        "el" => "ell",
        "en" => "eng",
        "eo" => "epo",
        "es" => "spa",
        "et" => "est",
        "eu" => "eus",
        "fa" => "fas",
        "fi" => "fin",
        "fr" => "fra",
        "ga" => "gle",
        "gl" => "glg",
        "he" => "heb",
        "hi" => "hin",
        "hr" => "hrv",
        "hu" => "hun",
        "hy" => "hye",
        "id" => "ind",
        "is" => "isl",
        "it" => "ita",
        "ja" => "jpn",
        "ka" => "kat",
        "kk" => "kaz",
        "km" => "khm",
        "kn" => "kan",
        "ko" => "kor",
        "lt" => "lit",
        "lv" => "lav",
        "mk" => "mkd",
        "ml" => "mal",
        "mn" => "mon",
        "mr" => "mar",
        "ms" => "msa",
        "mt" => "mlt",
        "my" => "mya",
        "nb" => "nob",
        "ne" => "nep",
        "nl" => "nld",
        "nn" => "nno",
        "no" => "nor",
        "pa" => "pan",
        "pl" => "pol",
        "ps" => "pus",
        "pt" => "por",
        "ro" => "ron",
        "ru" => "rus",
        "si" => "sin",
        "sk" => "slk",
        "sl" => "slv",
        "sq" => "sqi",
        "sr" => "srp",
        "sv" => "swe",
        "sw" => "swa",
        "ta" => "tam",
        "te" => "tel",
        "th" => "tha",
        "tl" => "tgl",
        "tr" => "tur",
        "uk" => "ukr",
        "ur" => "urd",
        "uz" => "uzb",
        "vi" => "vie",
        "zh" => "zho",
        "zu" => "zul",
        _ => "und",
    };
    let b = mapped.as_bytes();
    [b[0], b[1], b[2]]
}

/// Format `seconds` as `HH:MM:SS,mmm` (SubRip) or `HH:MM:SS.mmm` (WebVTT).
pub(crate) fn format_timestamp(seconds: f64, millis_sep: char) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let m = (total_s / 60) % 60;
    let h = total_s / 3600;
    format!("{h:02}:{m:02}:{s:02}{millis_sep}{ms:03}")
}

/// Parse `HH:MM:SS,mmm`, `HH:MM:SS.mmm`, or `MM:SS.mmm` into seconds.
pub(crate) fn parse_timestamp(text: &str) -> Option<f64> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    // Split the fractional part off first — it may use ',' or '.'.
    let (clock, frac) = match text.rfind([',', '.']) {
        Some(i) => (&text[..i], &text[i + 1..]),
        None => (text, ""),
    };
    if !frac.is_empty() && !frac.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let mut seconds = 0f64;
    let mut parts = 0;
    for part in clock.split(':') {
        if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        seconds = seconds * 60.0 + part.parse::<f64>().ok()?;
        parts += 1;
        if parts > 3 {
            return None;
        }
    }
    if parts == 0 {
        return None;
    }
    if !frac.is_empty() {
        // Right-pad/truncate to milliseconds so "5", "50" and "500" all mean 0.5 s.
        let mut digits = frac.to_string();
        digits.truncate(3);
        while digits.len() < 3 {
            digits.push('0');
        }
        seconds += digits.parse::<f64>().ok()? / 1000.0;
    }
    Some(seconds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps_round_trip() {
        assert_eq!(parse_timestamp("00:00:01,500"), Some(1.5));
        assert_eq!(parse_timestamp("00:00:01.500"), Some(1.5));
        assert_eq!(parse_timestamp("01:02:03,004"), Some(3723.004));
        assert_eq!(parse_timestamp("02:03.250"), Some(123.25));
        assert_eq!(parse_timestamp("00:00:01,5"), Some(1.5));
        assert_eq!(parse_timestamp("bogus"), None);
        assert_eq!(parse_timestamp("00:xx:01,500"), None);
        assert_eq!(format_timestamp(3723.004, ','), "01:02:03,004");
        assert_eq!(format_timestamp(1.5, '.'), "00:00:01.500");
        assert_eq!(format_timestamp(-4.0, ','), "00:00:00,000");
    }

    #[test]
    fn cues_stay_sorted_and_lookup_by_time() {
        let track = CaptionTrack::new()
            .cue(4.0, 6.0, "third")
            .cue(0.0, 2.0, "first")
            .cue(2.0, 4.0, "second");
        let order: Vec<&str> = track.cues.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(order, ["first", "second", "third"]);
        assert_eq!(track.cue_at(0.0).unwrap().text, "first");
        // End is exclusive, so the boundary belongs to exactly one cue.
        assert_eq!(track.cue_at(2.0).unwrap().text, "second");
        assert!(track.cue_at(6.0).is_none());
        assert_eq!(track.duration(), 6.0);
    }

    #[test]
    fn overlapping_cues_all_report_active() {
        let track = CaptionTrack::new()
            .cue(0.0, 4.0, "speaker one")
            .cue(1.0, 3.0, "speaker two");
        assert_eq!(track.active_at(2.0).len(), 2);
        assert_eq!(track.text_at(2.0), "speaker one\nspeaker two");
        assert_eq!(track.text_at(3.5), "speaker one");
        assert_eq!(track.text_at(9.0), "");
    }

    #[test]
    fn markup_and_entities_are_stripped() {
        let cue = Cue::new(
            0.0,
            1.0,
            "<i>Ich</i> &amp; <c.yellow>du</c> {\\an8}<v Bob>hi",
        );
        assert_eq!(cue.plain_text(), "Ich & du hi");
        // A bare ampersand is not an entity and must survive verbatim.
        assert_eq!(strip_markup("Tom & Jerry"), "Tom & Jerry");
        assert_eq!(strip_markup("a &lt;b&gt; c"), "a <b> c");
        assert_eq!(strip_markup("x&nbsp;y"), "x y");
    }

    #[test]
    fn shift_and_scale_retime_the_track() {
        let mut track = CaptionTrack::new().cue(1.0, 2.0, "a").cue(10.0, 11.0, "b");
        track.shift(-2.0);
        assert_eq!(track.cues[0].start, 0.0); // clamped, never negative
        assert_eq!(track.cues[1].start, 8.0);
        track.scale(0.5);
        assert_eq!(track.cues[1].start, 4.0);
    }

    #[test]
    fn style_scaling_is_proportional() {
        let base = CaptionStyle::default();
        let half = base.scaled(0.5);
        assert_eq!(half.font_size, base.font_size * 0.5);
        assert_eq!(half.margin, base.margin * 0.5);
        assert_eq!(half.padding, base.padding * 0.5);
        // Non-pixel properties are carried through untouched.
        assert_eq!(half.color, base.color);
        assert_eq!(half.max_width, base.max_width);
    }

    #[test]
    fn format_detection_from_path() {
        assert_eq!(
            CaptionFormat::from_path("a/b.SRT"),
            Some(CaptionFormat::Srt)
        );
        assert_eq!(CaptionFormat::from_path("x.vtt"), Some(CaptionFormat::Vtt));
        assert_eq!(CaptionFormat::from_path("x.mp4"), None);
    }
}
