//! WebVTT (`.vtt`) parse and serialize — the format browsers accept in
//! `<track kind="subtitles" src="…">`.
//!
//! A WebVTT file is the magic word `WEBVTT`, optional header metadata, then
//! cue blocks. Relative to SubRip a cue may carry a free-form identifier and a
//! settings list after the timing (`align:`, `line:`, `position:`), and
//! `NOTE` / `STYLE` / `REGION` blocks may appear between cues.
//!
//! The parser understands the cue settings this crate can render — `align`,
//! `line`, and `position` — and skips the rest rather than failing.

use super::{format_timestamp, parse_timestamp, CaptionAlign, CaptionError, CaptionTrack, Cue};

/// Parse WebVTT text into a [`CaptionTrack`].
///
/// ```
/// use threers::captions::vtt;
/// let track = vtt::parse(
///     "WEBVTT\n\nintro\n00:00.000 --> 00:02.000 align:left line:10%\nHi there\n",
/// ).unwrap();
/// assert_eq!(track.cues[0].id.as_deref(), Some("intro"));
/// assert_eq!(track.cues[0].line, Some(0.1));
/// ```
pub fn parse(text: &str) -> Result<CaptionTrack, CaptionError> {
    let text = text.trim_start_matches('\u{feff}');
    let lines: Vec<&str> = text.lines().map(|l| l.trim_end_matches('\r')).collect();
    let mut track = CaptionTrack::new();

    let mut i = 0;
    // Header: "WEBVTT" plus anything up to the first blank line.
    if lines.first().map(|l| l.trim_start().starts_with("WEBVTT")) != Some(true) {
        return Err(CaptionError::NotRecognized);
    }
    while i < lines.len() && !lines[i].trim().is_empty() {
        i += 1;
    }

    while i < lines.len() {
        if lines[i].trim().is_empty() {
            i += 1;
            continue;
        }
        let block_start = i;
        let mut end = i;
        while end < lines.len() && !lines[end].trim().is_empty() {
            end += 1;
        }
        let block = &lines[i..end];
        i = end;

        // NOTE / STYLE / REGION blocks carry no cues.
        let head = block[0].trim();
        if head == "NOTE"
            || head.starts_with("NOTE ")
            || head.starts_with("NOTE\t")
            || head == "STYLE"
            || head == "REGION"
        {
            continue;
        }

        let timing_idx = match block.iter().position(|l| l.contains("-->")) {
            Some(idx) => idx,
            None => {
                return Err(CaptionError::MissingTiming {
                    line: block_start + 1,
                })
            }
        };
        let (start, stop, settings) =
            parse_timing(block[timing_idx], block_start + timing_idx + 1)?;
        let id = block[..timing_idx]
            .iter()
            .map(|l| l.trim())
            .find(|l| !l.is_empty())
            .map(|l| l.to_string());
        let body = block[timing_idx + 1..].join("\n");

        track.cues.push(Cue {
            id,
            start,
            end: stop,
            text: body.trim_end().to_string(),
            align: settings.align,
            line: settings.line,
            position: settings.position,
        });
    }

    if track.is_empty() {
        return Err(CaptionError::Empty);
    }
    track.sort();
    Ok(track)
}

/// The subset of WebVTT cue settings this crate can honor.
#[derive(Default)]
struct CueSettings {
    align: Option<CaptionAlign>,
    line: Option<f32>,
    position: Option<f32>,
}

fn parse_timing(line: &str, line_no: usize) -> Result<(f64, f64, CueSettings), CaptionError> {
    let bad = || CaptionError::BadTimestamp {
        line: line_no,
        text: line.to_string(),
    };
    let (left, right) = line.split_once("-->").ok_or_else(bad)?;
    let start = parse_timestamp(left).ok_or_else(bad)?;
    let mut rest = right.split_whitespace();
    let end = parse_timestamp(rest.next().ok_or_else(bad)?).ok_or_else(bad)?;

    let mut settings = CueSettings::default();
    for token in rest {
        let Some((key, value)) = token.split_once(':') else {
            continue;
        };
        match key {
            "align" => {
                settings.align = match value {
                    "start" | "left" => Some(CaptionAlign::Left),
                    "end" | "right" => Some(CaptionAlign::Right),
                    "center" | "middle" => Some(CaptionAlign::Center),
                    _ => None,
                }
            }
            // "line" is a percentage from the top, or a line count we can't map.
            "line" => settings.line = parse_percent(value),
            "position" => settings.position = parse_percent(value),
            _ => {}
        }
    }
    Ok((start, end.max(start), settings))
}

/// `"90%"` → `Some(0.9)`. Alignment suffixes (`"90%,start"`) are dropped.
fn parse_percent(value: &str) -> Option<f32> {
    let value = value.split(',').next()?;
    let number = value.strip_suffix('%')?;
    number
        .parse::<f32>()
        .ok()
        .map(|v| (v / 100.0).clamp(0.0, 1.0))
}

/// Serialize a [`CaptionTrack`] to WebVTT text.
///
/// Emits the track language/label as a header comment, then each cue with its
/// identifier (when set) and any placement settings it carries.
pub fn write(track: &CaptionTrack) -> String {
    let mut out = String::from("WEBVTT");
    if !track.label.is_empty() || !track.language.is_empty() {
        out.push_str(" - ");
        if !track.label.is_empty() {
            out.push_str(&track.label);
        }
        if !track.language.is_empty() {
            if !track.label.is_empty() {
                out.push(' ');
            }
            out.push('[');
            out.push_str(&track.language);
            out.push(']');
        }
    }
    out.push_str("\n\n");

    for cue in &track.cues {
        if let Some(id) = &cue.id {
            // A bare number would be read as a cue id anyway; keep it verbatim.
            out.push_str(id.trim());
            out.push('\n');
        }
        out.push_str(&format_timestamp(cue.start, '.'));
        out.push_str(" --> ");
        out.push_str(&format_timestamp(cue.end, '.'));
        if let Some(align) = cue.align {
            out.push_str(match align {
                CaptionAlign::Left => " align:left",
                CaptionAlign::Center => " align:center",
                CaptionAlign::Right => " align:right",
            });
        }
        if let Some(line) = cue.line {
            out.push_str(&format!(" line:{:.0}%", line * 100.0));
        }
        if let Some(position) = cue.position {
            out.push_str(&format!(" position:{:.0}%", position * 100.0));
        }
        out.push('\n');
        out.push_str(cue.text.trim_end());
        out.push_str("\n\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "WEBVTT - Example [en]\n\
                          \n\
                          NOTE this comment is skipped\n\
                          and so is its continuation\n\
                          \n\
                          intro\n\
                          00:00.000 --> 00:02.000 align:left line:10% position:25%\n\
                          Hi there\n\
                          \n\
                          00:00:02.000 --> 00:00:04.000\n\
                          <i>Second</i> cue\n";

    #[test]
    fn parses_header_notes_settings_and_ids() {
        let track = parse(SAMPLE).unwrap();
        assert_eq!(track.len(), 2);
        let first = &track.cues[0];
        assert_eq!(first.id.as_deref(), Some("intro"));
        assert_eq!(first.start, 0.0);
        assert_eq!(first.end, 2.0);
        assert_eq!(first.align, Some(CaptionAlign::Left));
        assert_eq!(first.line, Some(0.1));
        assert_eq!(first.position, Some(0.25));
        assert_eq!(track.cues[1].text, "<i>Second</i> cue");
        assert_eq!(track.cues[1].plain_text(), "Second cue");
    }

    #[test]
    fn round_trips_settings_through_write() {
        let track = parse(SAMPLE).unwrap();
        let text = write(&track);
        assert!(text.starts_with("WEBVTT"), "{text}");
        let again = parse(&text).unwrap();
        assert_eq!(again.cues[0].align, Some(CaptionAlign::Left));
        assert_eq!(again.cues[0].line, Some(0.1));
        assert_eq!(again.cues[0].position, Some(0.25));
        assert_eq!(again.cues[1].text, track.cues[1].text);
    }

    #[test]
    fn missing_magic_is_rejected() {
        assert_eq!(
            parse("1\n00:00:00,000 --> 00:00:01,000\nx\n"),
            Err(CaptionError::NotRecognized)
        );
    }

    #[test]
    fn style_and_region_blocks_are_skipped() {
        let text = "WEBVTT\n\nSTYLE\n::cue { color: red }\n\nREGION\nid:r1\n\n\
                    00:00.000 --> 00:01.000\nonly cue\n";
        let track = parse(text).unwrap();
        assert_eq!(track.len(), 1);
        assert_eq!(track.cues[0].text, "only cue");
    }

    #[test]
    fn sniffing_picks_the_right_parser() {
        let vtt = CaptionTrack::parse(SAMPLE).unwrap();
        assert_eq!(vtt.len(), 2);
        let srt = CaptionTrack::parse("1\n00:00:00,000 --> 00:00:01,000\nx\n").unwrap();
        assert_eq!(srt.len(), 1);
    }
}
