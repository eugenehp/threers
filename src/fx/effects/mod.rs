//! The effects that ship with [`fx`](crate::fx). Each implements
//! [`Effect`](crate::fx::Effect); add your own the same way.

pub mod gloss;
pub mod venom;

pub use gloss::{Gloss, GlossStyle};
pub use venom::{Venom, VenomState, VenomStyle};
