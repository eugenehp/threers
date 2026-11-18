//! One game on two devices: messages from one to the other, and keeping
//! only the newest of a stream of snapshots.
//!
//! A [`Link`] carries text messages (JSON, say) between two ends. What
//! carries them is the host's: a socket, a WebRTC channel, or Apple's
//! MultipeerConnectivity, which [`nearby`](crate::nearby) has (the `nearby`
//! feature). [`Loopback::pair`] gives two ends joined in memory, for
//! tests and for trying both sides on one device. [`Newest`] keeps a
//! snapshot stream in order: each message carries a sequence number, and one
//! older than what was shown is dropped.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// A way for messages to travel to the other device.
pub trait Link {
    /// Send `message`. One that cannot go now is dropped: snapshots go
    /// again soon.
    fn send(&mut self, message: &str);
    /// Send `message`, which may be lost or overtaken: for a stream where
    /// the newest is what counts. A link that cannot tell sends it as
    /// [`send`](Link::send) does.
    fn send_lossy(&mut self, message: &str) {
        self.send(message);
    }
    /// Every message that has come since last asked, oldest first.
    fn receive(&mut self) -> Vec<String>;
    /// The names of the devices at the other end now; none, if it cannot tell.
    fn peers(&self) -> Vec<String> {
        Vec::new()
    }
}

type Queue = Arc<Mutex<VecDeque<String>>>;

/// One end of two joined in memory.
#[derive(Debug, Clone, Default)]
pub struct Loopback {
    inbox: Queue,
    outbox: Queue,
}

impl Loopback {
    /// Two ends: what one sends, the other receives.
    pub fn pair() -> (Loopback, Loopback) {
        let (a, b) = (Queue::default(), Queue::default());
        (
            Loopback {
                inbox: a.clone(),
                outbox: b.clone(),
            },
            Loopback {
                inbox: b,
                outbox: a,
            },
        )
    }
}

impl Link for Loopback {
    fn send(&mut self, message: &str) {
        if let Ok(mut queue) = self.outbox.lock() {
            queue.push_back(message.to_owned());
        }
    }

    fn receive(&mut self) -> Vec<String> {
        self.inbox
            .lock()
            .map(|mut queue| queue.drain(..).collect())
            .unwrap_or_default()
    }

    fn peers(&self) -> Vec<String> {
        vec!["loopback".to_owned()]
    }
}

const BASE64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Bytes as text, to go over a link: base64, padded.
pub fn to_text(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk.iter().enumerate().fold(0u32, |n, (i, b)| n | (*b as u32) << (16 - 8 * i));
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(BASE64[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// The bytes [`to_text`] made; `None` if `text` is not base64.
pub fn from_text(text: &str) -> Option<Vec<u8>> {
    let text = text.trim_end_matches('=');
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let (mut n, mut bits) = (0u32, 0);
    for c in text.bytes() {
        let v = BASE64.iter().position(|b| *b == c)? as u32;
        n = n << 6 | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((n >> bits) as u8);
            n &= (1 << bits) - 1;
        }
    }
    Some(out)
}

/// The newest of a stream of numbered snapshots.
#[derive(Debug, Clone, Default)]
pub struct Newest {
    last: Option<u64>,
}

impl Newest {
    /// Of `messages`, each with its sequence number, the newest that is newer
    /// than any before; `None` if none is.
    pub fn pick<T>(&mut self, messages: impl IntoIterator<Item = (u64, T)>) -> Option<T> {
        let mut best: Option<(u64, T)> = None;
        for (seq, message) in messages {
            let newer = self.last.is_none_or(|last| seq > last) && best.as_ref().is_none_or(|(b, _)| seq > *b);
            if newer {
                best = Some((seq, message));
            }
        }
        let (seq, message) = best?;
        self.last = Some(seq);
        Some(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_one_end_sends_the_other_receives() {
        let (mut a, mut b) = Loopback::pair();
        a.send("hello");
        a.send("again");
        b.send("back");
        assert_eq!(b.receive(), ["hello", "again"]);
        assert!(b.receive().is_empty());
        assert_eq!(a.receive(), ["back"]);
    }

    #[test]
    fn bytes_go_as_text_and_back() {
        for bytes in [&b""[..], b"f", b"fo", b"foo", b"foob", &[0, 255, 128, 7, 9]] {
            assert_eq!(from_text(&to_text(bytes)).unwrap(), bytes);
        }
        assert_eq!(to_text(b"foobar"), "Zm9vYmFy");
        assert_eq!(to_text(b"fo"), "Zm8=");
        assert!(from_text("not*base64").is_none());
    }

    #[test]
    fn only_newer_snapshots_are_shown() {
        let mut newest = Newest::default();
        assert_eq!(newest.pick([(3, "c"), (5, "e"), (4, "d")]), Some("e"));
        // Late, or the same again: dropped.
        assert_eq!(newest.pick([(2, "b"), (5, "e")]), None);
        assert_eq!(newest.pick([(6, "f")]), Some("f"));
    }
}
