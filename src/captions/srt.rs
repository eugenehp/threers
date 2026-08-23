//! SubRip (`.srt`) parse and serialize.
//!
//! SubRip is the lowest-common-denominator sidecar format: a numeric counter,
//! an `HH:MM:SS,mmm --> HH:MM:SS,mmm` timing line, then one or more text lines,
//! with a blank line between cues.
//!
//! The parser is deliberately forgiving — it accepts `.` as the decimal
//! separator, tolerates a missing counter line, ignores the optional
//! `X1:.. Y1:..` coordinate suffix, and accepts both LF and CRLF.

use super::{format_timestamp, parse_timestamp, CaptionError, CaptionTrack, Cue};

/// Parse SubRip text into a [`CaptionTrack`].
///
/// ```
/// use threers::captions::srt;
/// let track = srt::parse("1\n00:00:01,000 --> 00:00:03,000\nLine one\nLine two\n").unwrap();
/// assert_eq!(track.len(), 1);
/// assert_eq!(track.cues[0].text, "Line one\nLine two");
/// ```
pub fn parse(text: &str) -> Result<CaptionTrack, CaptionError> {
    let text = text.trim_start_matches('\u{feff}');
    let mut track = CaptionTrack::new();
    let lines: Vec<&str> = text.lines().map(|l| l.trim_end_matches('\r')).collect();
    let mut i = 0;

    while i < lines.len() {
        // Skip blank separators.
        if lines[i].trim().is_empty() {
            i += 1;
            continue;
        }
        let block_start = i;
        // Gather the block up to the next blank line.
        let mut end = i;
        while end < lines.len() && !lines[end].trim().is_empty() {
            end += 1;
        }
        let block = &lines[i..end];
        i = end;

        // The timing line is the first line containing "-->".
        let timing_idx = match block.iter().position(|l| l.contains("-->")) {
            Some(idx) => idx,
            None => {
                return Err(CaptionError::MissingTiming {
                    line: block_start + 1,
                })
            }
        };
        let (start, stop) = parse_timing(block[timing_idx], block_start + timing_idx + 1)?;

        // Anything before the timing line is the cue identifier.
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
            align: None,
            line: None,
            position: None,
        });
    }

    if track.is_empty() {
        return Err(CaptionError::Empty);
    }
    track.sort();
    Ok(track)
}

/// Split `"00:00:01,000 --> 00:00:03,000 X1:0 X2:100"` into start/end seconds.
fn parse_timing(line: &str, line_no: usize) -> Result<(f64, f64), CaptionError> {
    let bad = || CaptionError::BadTimestamp {
        line: line_no,
        text: line.to_string(),
    };
    let (left, right) = line.split_once("-->").ok_or_else(bad)?;
    let start = parse_timestamp(left).ok_or_else(bad)?;
    // The end timestamp may be followed by SubRip's optional coordinates.
    let end_text = right.split_whitespace().next().ok_or_else(bad)?;
    let end = parse_timestamp(end_text).ok_or_else(bad)?;
    Ok((start, end.max(start)))
}

/// Serialize a [`CaptionTrack`] to SubRip text.
///
/// Cues are numbered from 1 regardless of [`Cue::id`], because SubRip requires
/// the counter to be an integer.
pub fn write(track: &CaptionTrack) -> String {
    let mut out = String::new();
    for (n, cue) in track.cues.iter().enumerate() {
        out.push_str(&(n + 1).to_string());
        out.push('\n');
        out.push_str(&format_timestamp(cue.start, ','));
        out.push_str(" --> ");
        out.push_str(&format_timestamp(cue.end, ','));
        out.push('\n');
        out.push_str(cue.text.trim_end());
        out.push_str("\n\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\u{feff}1\r\n00:00:00,000 --> 00:00:02,500\r\nHello, world!\r\n\r\n\
                          2\r\n00:00:02,500 --> 00:00:05,000\r\nSecond cue\r\nwith two lines\r\n";

    #[test]
    fn parses_bom_crlf_and_multiline_text() {
        let track = parse(SAMPLE).unwrap();
        assert_eq!(track.len(), 2);
        assert_eq!(track.cues[0].start, 0.0);
        assert_eq!(track.cues[0].end, 2.5);
        assert_eq!(track.cues[0].text, "Hello, world!");
        assert_eq!(track.cues[1].text, "Second cue\nwith two lines");
        assert_eq!(track.cues[1].id.as_deref(), Some("2"));
    }

    #[test]
    fn round_trips_through_write() {
        let track = parse(SAMPLE).unwrap();
        let again = parse(&write(&track)).unwrap();
        assert_eq!(again.cues.len(), track.cues.len());
        for (a, b) in again.cues.iter().zip(track.cues.iter()) {
            assert_eq!(a.text, b.text);
            assert!((a.start - b.start).abs() < 1e-6);
            assert!((a.end - b.end).abs() < 1e-6);
        }
    }

    #[test]
    fn tolerates_missing_counter_and_coordinates() {
        let track = parse("00:00:01.000 --> 00:00:02.000 X1:10 X2:20\nno counter\n").unwrap();
        assert_eq!(track.len(), 1);
        assert_eq!(track.cues[0].start, 1.0);
        assert_eq!(track.cues[0].end, 2.0);
        assert!(track.cues[0].id.is_none());
    }

    #[test]
    fn reversed_times_clamp_instead_of_going_negative() {
        let track = parse("1\n00:00:05,000 --> 00:00:02,000\noops\n").unwrap();
        assert_eq!(track.cues[0].end, 5.0);
        assert_eq!(track.cues[0].duration(), 0.0);
    }

    #[test]
    fn errors_are_reported_with_line_numbers() {
        assert_eq!(parse(""), Err(CaptionError::Empty));
        assert_eq!(
            parse("1\njust text\nno timing\n"),
            Err(CaptionError::MissingTiming { line: 1 })
        );
        match parse("1\n00:00:0X,000 --> 00:00:02,000\nx\n") {
            Err(CaptionError::BadTimestamp { line, .. }) => assert_eq!(line, 2),
            other => panic!("expected BadTimestamp, got {other:?}"),
        }
    }

    #[test]
    fn writer_renumbers_cues_from_one() {
        let mut track = CaptionTrack::new().cue(0.0, 1.0, "a").cue(1.0, 2.0, "b");
        track.cues[0].id = Some("99".into());
        let text = write(&track);
        assert!(
            text.starts_with("1\n00:00:00,000 --> 00:00:01,000\na\n\n"),
            "{text}"
        );
        assert!(text.contains("\n2\n"), "{text}");
    }
}
