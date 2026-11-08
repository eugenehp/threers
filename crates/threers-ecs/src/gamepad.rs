//! Real gamepads, through gilrs.
//!
//! gilrs reports events per device; [`PadTracker`] does the combining. A window
//! from the `window` module polls this for you. Any other host keeps a
//! [`Gamepads`] and polls it before each [`App::frame`](crate::App::frame).

use crate::input::{Input, PadAxis, PadButton, PadEvent, PadTracker, Rumble};
use gilrs::ff::{BaseEffect, BaseEffectType, Effect, EffectBuilder, Repeat, Replay, Ticks};
use gilrs::{Axis, Button, Event, EventType, Gilrs};

/// Every connected gamepad, feeding one [`Input`].
pub struct Gamepads {
    gilrs: Gilrs,
    tracker: PadTracker,
    effect: Option<Effect>,
}

impl Gamepads {
    /// # Errors
    ///
    /// If the platform's gamepad service cannot be reached. Games usually carry
    /// on without pads.
    pub fn new() -> Result<Self, Box<gilrs::Error>> {
        Ok(Self {
            gilrs: Gilrs::new().map_err(Box::new)?,
            tracker: PadTracker::default(),
            effect: None,
        })
    }

    /// Bring `input` up to date. While `focused` is false events are still read,
    /// so nothing is stale on return, but the game does not see them: a pad in
    /// use in another app should not be driving this one.
    pub fn poll(&mut self, input: &mut Input, focused: bool) {
        while let Some(Event { id, event, .. }) = self.gilrs.next_event() {
            if let Some(event) = translate(event) {
                self.tracker.update(id.into(), event);
                // Per event, not per poll, so a tap between two polls still lands.
                if focused {
                    self.tracker.sync(input);
                }
            }
        }
        if focused {
            self.tracker.sync(input);
        }
        if let Some(rumble) = input.take_rumble() {
            self.play_rumble(rumble);
        }
        self.gilrs.inc();
    }

    /// Pads that cannot shake are skipped, and a platform with no force feedback
    /// plays nothing. The effect is kept, since dropping it would stop it.
    fn play_rumble(&mut self, rumble: Rumble) {
        let pads: Vec<_> = self
            .gilrs
            .gamepads()
            .filter(|(_, pad)| pad.is_ff_supported())
            .map(|(id, _)| id)
            .collect();
        if pads.is_empty() {
            return;
        }
        let effect = EffectBuilder::new()
            .add_effect(BaseEffect {
                kind: BaseEffectType::Strong {
                    magnitude: (rumble.strength * u16::MAX as f32) as u16,
                },
                scheduling: Replay {
                    play_for: Ticks::from_ms((rumble.seconds * 1000.0) as u32),
                    ..Replay::default()
                },
                ..BaseEffect::default()
            })
            .repeat(Repeat::For(Ticks::from_ms(
                (rumble.seconds * 1000.0) as u32,
            )))
            .gamepads(&pads)
            .finish(&mut self.gilrs);
        match effect.and_then(|effect| effect.play().map(|()| effect)) {
            Ok(effect) => self.effect = Some(effect),
            Err(error) => eprintln!("rumble: {error}"),
        }
    }

    /// How many pads are connected now.
    pub fn connected(&self) -> usize {
        self.gilrs.gamepads().count()
    }
}

fn translate(event: EventType) -> Option<PadEvent> {
    match event {
        EventType::ButtonPressed(b, _) => pad_button(b).map(PadEvent::Pressed),
        EventType::ButtonReleased(b, _) => pad_button(b).map(PadEvent::Released),
        EventType::ButtonChanged(b, value, _) => trigger(b).map(|a| PadEvent::Axis(a, value)),
        EventType::AxisChanged(a, value, _) => pad_axis(a).map(|a| PadEvent::Axis(a, value)),
        EventType::Disconnected => Some(PadEvent::Disconnected),
        _ => None,
    }
}

/// gilrs says `LeftTrigger` for the bumper and `LeftTrigger2` for the trigger, so
/// the names here cross over on purpose.
pub fn pad_button(button: Button) -> Option<PadButton> {
    Some(match button {
        Button::South => PadButton::South,
        Button::East => PadButton::East,
        Button::West => PadButton::West,
        Button::North => PadButton::North,
        Button::LeftTrigger => PadButton::LeftShoulder,
        Button::RightTrigger => PadButton::RightShoulder,
        Button::LeftTrigger2 => PadButton::LeftTrigger,
        Button::RightTrigger2 => PadButton::RightTrigger,
        Button::LeftThumb => PadButton::LeftThumb,
        Button::RightThumb => PadButton::RightThumb,
        Button::Start => PadButton::Start,
        Button::Select => PadButton::Select,
        Button::DPadUp => PadButton::DPadUp,
        Button::DPadDown => PadButton::DPadDown,
        Button::DPadLeft => PadButton::DPadLeft,
        Button::DPadRight => PadButton::DPadRight,
        _ => return None,
    })
}

/// A trigger's analog travel, which gilrs reports as a button's value.
pub fn trigger(button: Button) -> Option<PadAxis> {
    match button {
        Button::LeftTrigger2 => Some(PadAxis::LeftTrigger),
        Button::RightTrigger2 => Some(PadAxis::RightTrigger),
        _ => None,
    }
}

/// The [`PadAxis`] for one of gilrs's axes, if it has one.
pub fn pad_axis(axis: Axis) -> Option<PadAxis> {
    match axis {
        Axis::LeftStickX => Some(PadAxis::LeftX),
        Axis::LeftStickY => Some(PadAxis::LeftY),
        Axis::RightStickX => Some(PadAxis::RightX),
        Axis::RightStickY => Some(PadAxis::RightY),
        _ => None,
    }
}
