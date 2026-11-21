//! Keyboards by what a field holds and what it runs on, as a platform
//! picks them: say a field takes an email address and the keyboard has
//! `@` and `.`; run on a tablet and it is a tablet's.
//!
//! ```
//! use threers_ui::virtual_keyboard::{Device, KeyboardType};
//!
//! assert_eq!(Device::for_size(390.0, 844.0), Device::Phone);
//! assert_eq!(Device::for_size(820.0, 1180.0), Device::Tablet);
//! assert_eq!(Device::for_size(1366.0, 1024.0), Device::LargeTablet);
//! let pin = KeyboardType::NumberPad.config(Device::Tablet);
//! assert_eq!(pin.letters.len(), 4, "a number pad wherever it runs");
//! ```

use super::config::{Key, KeyboardConfig, Page, ReturnKey};
use super::context::AutoCapitalize;

/// What the keyboard is shaped for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// An iPhone's ([`KeyboardConfig::iphone`]).
    Phone,
    /// An iPad's ([`KeyboardConfig::ipad`]).
    Tablet,
    /// A large iPad Pro's, with its number row ([`KeyboardConfig::ipad_pro`]).
    LargeTablet,
    /// Whichever suits the window, again when it changes size
    /// ([`Device::for_size`]).
    #[default]
    Auto,
}

impl Device {
    /// The device a window this size is: a phone under 600 pixels across
    /// its shorter side, a large tablet from 1000, a tablet between.
    pub fn for_size(width: f32, height: f32) -> Device {
        let short = width.min(height);
        if short < 600.0 {
            Device::Phone
        } else if short >= 1000.0 {
            Device::LargeTablet
        } else {
            Device::Tablet
        }
    }

    /// `Auto` made a device for a window this size; the others as they are.
    pub fn resolve(self, width: f32, height: f32) -> Device {
        match self {
            Device::Auto => Device::for_size(width, height),
            d => d,
        }
    }
}

/// What a field takes, as a platform's text fields say
/// (`UIKeyboardType`, HTML's `inputmode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KeyboardType {
    /// Text: letters, sentences capitalised.
    #[default]
    Default,
    /// Starts on the digits and punctuation page.
    NumbersAndPunctuation,
    /// An email address: `@` and `.` by the space bar, no capitals.
    Email,
    /// A web address: `.`, `/` and `.com`, Go, no capitals.
    Url,
    /// A search box: Search on Return.
    WebSearch,
    /// Digits only (a PIN, a quantity).
    NumberPad,
    /// Digits and a decimal point (a price).
    DecimalPad,
    /// A phone number: digits, `+`, `*`, `#`.
    PhonePad,
}

impl KeyboardType {
    /// The keyboard for this type on `device` (`Auto` taken as a phone).
    pub fn config(self, device: Device) -> KeyboardConfig {
        let base = KeyboardConfig::for_device(device);
        let tablet = !matches!(device, Device::Phone | Device::Auto);
        let mut c = match self {
            KeyboardType::Default => base,
            KeyboardType::NumbersAndPunctuation => KeyboardConfig {
                start_page: Page::Symbols,
                ..base
            },
            KeyboardType::WebSearch => base.return_key(ReturnKey::Search),
            KeyboardType::Email if !tablet => KeyboardConfig::email(),
            KeyboardType::Url if !tablet => KeyboardConfig::url(),
            // A tablet's keeps its keys and adds the address's beside
            // the space bar.
            KeyboardType::Email => with_keys_by_space(base, &["@", "."]),
            KeyboardType::Url => with_keys_by_space(base, &["/", ".com"]).return_key(ReturnKey::Go),
            KeyboardType::NumberPad => KeyboardConfig::number_pad(),
            KeyboardType::DecimalPad => KeyboardConfig::decimal_pad(),
            KeyboardType::PhonePad => KeyboardConfig::phone_pad(),
        };
        if matches!(self, KeyboardType::Email | KeyboardType::Url) {
            c.auto_capitalize = AutoCapitalize::None;
            c.period_shortcut = false;
        }
        c
    }
}

/// `keys` put after the space bar on every page.
fn with_keys_by_space(mut c: KeyboardConfig, keys: &[&str]) -> KeyboardConfig {
    for page in [&mut c.letters, &mut c.symbols, &mut c.more] {
        for row in page.iter_mut() {
            if let Some(i) = row.iter().position(|k| k.name.as_deref() == Some("Space")) {
                for (j, t) in keys.iter().enumerate() {
                    row.insert(i + 1 + j, Key::text(t));
                }
            }
        }
    }
    c
}
