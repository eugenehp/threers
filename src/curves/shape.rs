use super::Path;

/// A closed 2D outline plus optional holes. Mirrors three.js's `Shape`.
pub struct Shape {
    pub outline: Path,
    pub holes: Vec<Path>,
}

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shape")
            .field("outline", &self.outline)
            .field("holes", &self.holes.len())
            .finish()
    }
}

impl Default for Shape {
    fn default() -> Self {
        Self::new()
    }
}

impl Shape {
    pub fn new() -> Self {
        Self {
            outline: Path::new(),
            holes: Vec::new(),
        }
    }

    pub fn from_path(outline: Path) -> Self {
        Self {
            outline,
            holes: Vec::new(),
        }
    }

    pub fn add_hole(&mut self, hole: Path) -> &mut Self {
        self.holes.push(hole);
        self
    }
}
