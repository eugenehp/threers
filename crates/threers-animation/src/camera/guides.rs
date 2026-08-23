//! Composition guides, safe areas, and passepartout.

/// Overlay guide drawn in the viewport / HUD.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositionGuide {
    Center,
    Thirds,
    Golden,
    HarmonyTriangleA,
    HarmonyTriangleB,
}

/// SMPTE / broadcast safe areas as fractions of the frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SafeAreas {
    /// Title-safe inset from each edge (`0.1` = 10%).
    pub title: f32,
    /// Action-safe inset.
    pub action: f32,
}

impl Default for SafeAreas {
    fn default() -> Self {
        Self {
            title: 0.1,
            action: 0.05,
        }
    }
}

impl SafeAreas {
    pub const BROADCAST: Self = Self {
        title: 0.1,
        action: 0.05,
    };
}

/// Passepartout — dim outside the render gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Passepartout {
    /// Opacity of the outside dim `0..1`.
    pub alpha: f32,
    /// Gate aspect; `None` = use camera aspect.
    pub aspect: Option<f32>,
}

impl Default for Passepartout {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            aspect: None,
        }
    }
}

/// Bundle of viewport camera chrome.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraGuides {
    pub guides: Vec<CompositionGuide>,
    pub safe: SafeAreas,
    pub passepartout: Passepartout,
    pub show_safe: bool,
    pub show_passepartout: bool,
}

impl Default for CameraGuides {
    fn default() -> Self {
        Self {
            guides: vec![CompositionGuide::Thirds],
            safe: SafeAreas::default(),
            passepartout: Passepartout::default(),
            show_safe: true,
            show_passepartout: true,
        }
    }
}

/// Normalised line segment in frame space (`0..1`, origin top-left or centre — caller picks).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuideLine {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl CompositionGuide {
    /// Lines in normalised frame coords with origin at top-left, y down.
    pub fn lines(self) -> Vec<GuideLine> {
        match self {
            Self::Center => vec![
                GuideLine {
                    x0: 0.5,
                    y0: 0.0,
                    x1: 0.5,
                    y1: 1.0,
                },
                GuideLine {
                    x0: 0.0,
                    y0: 0.5,
                    x1: 1.0,
                    y1: 0.5,
                },
            ],
            Self::Thirds => vec![
                GuideLine {
                    x0: 1.0 / 3.0,
                    y0: 0.0,
                    x1: 1.0 / 3.0,
                    y1: 1.0,
                },
                GuideLine {
                    x0: 2.0 / 3.0,
                    y0: 0.0,
                    x1: 2.0 / 3.0,
                    y1: 1.0,
                },
                GuideLine {
                    x0: 0.0,
                    y0: 1.0 / 3.0,
                    x1: 1.0,
                    y1: 1.0 / 3.0,
                },
                GuideLine {
                    x0: 0.0,
                    y0: 2.0 / 3.0,
                    x1: 1.0,
                    y1: 2.0 / 3.0,
                },
            ],
            Self::Golden => {
                let g = 1.0 / 1.618_034;
                let og = 1.0 - g;
                vec![
                    GuideLine {
                        x0: g,
                        y0: 0.0,
                        x1: g,
                        y1: 1.0,
                    },
                    GuideLine {
                        x0: og,
                        y0: 0.0,
                        x1: og,
                        y1: 1.0,
                    },
                    GuideLine {
                        x0: 0.0,
                        y0: g,
                        x1: 1.0,
                        y1: g,
                    },
                    GuideLine {
                        x0: 0.0,
                        y0: og,
                        x1: 1.0,
                        y1: og,
                    },
                ]
            }
            Self::HarmonyTriangleA => vec![
                GuideLine {
                    x0: 0.0,
                    y0: 0.0,
                    x1: 1.0,
                    y1: 1.0,
                },
                GuideLine {
                    x0: 0.0,
                    y0: 1.0,
                    x1: 0.5,
                    y1: 0.5,
                },
            ],
            Self::HarmonyTriangleB => vec![
                GuideLine {
                    x0: 1.0,
                    y0: 0.0,
                    x1: 0.0,
                    y1: 1.0,
                },
                GuideLine {
                    x0: 1.0,
                    y0: 1.0,
                    x1: 0.5,
                    y1: 0.5,
                },
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thirds_has_four_lines() {
        assert_eq!(CompositionGuide::Thirds.lines().len(), 4);
    }
}
