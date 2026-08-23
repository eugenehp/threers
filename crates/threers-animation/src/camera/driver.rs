//! Simple drivers — link camera channels (Blender driver lite).

use super::pose::CameraPose;

/// Animatable camera channel a driver can read/write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriverChannel {
    Fov,
    Roll,
    Radius,
    Azimuth,
    Elevation,
    FocusDistance,
    Aperture,
    FStop,
    ShiftX,
    ShiftY,
    TargetX,
    TargetY,
    TargetZ,
}

impl DriverChannel {
    pub fn get(self, pose: &CameraPose) -> f32 {
        match self {
            Self::Fov => pose.fov,
            Self::Roll => pose.roll,
            Self::Radius => pose.radius,
            Self::Azimuth => pose.azimuth,
            Self::Elevation => pose.elevation,
            Self::FocusDistance => pose.focus_distance,
            Self::Aperture => pose.aperture,
            Self::FStop => pose.f_stop,
            Self::ShiftX => pose.shift_x,
            Self::ShiftY => pose.shift_y,
            Self::TargetX => pose.target.x,
            Self::TargetY => pose.target.y,
            Self::TargetZ => pose.target.z,
        }
    }

    pub fn set(self, pose: &mut CameraPose, value: f32) {
        match self {
            Self::Fov => pose.fov = value.max(1e-3),
            Self::Roll => pose.roll = value,
            Self::Radius => pose.radius = value.max(1e-4),
            Self::Azimuth => pose.azimuth = value,
            Self::Elevation => pose.elevation = value,
            Self::FocusDistance => pose.focus_distance = value.max(0.0),
            Self::Aperture => pose.aperture = value.max(0.0),
            Self::FStop => pose.f_stop = value.max(0.0),
            Self::ShiftX => pose.shift_x = value,
            Self::ShiftY => pose.shift_y = value,
            Self::TargetX => pose.target.x = value,
            Self::TargetY => pose.target.y = value,
            Self::TargetZ => pose.target.z = value,
        }
    }
}

/// How the source value maps to the destination.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DriverOp {
    Copy,
    Scale(f32),
    Offset(f32),
    ScaleOffset { scale: f32, offset: f32 },
    /// dest = clamp(scale * src + offset, min, max)
    Expression {
        scale: f32,
        offset: f32,
        min: f32,
        max: f32,
    },
}

impl DriverOp {
    pub fn apply(self, src: f32) -> f32 {
        match self {
            Self::Copy => src,
            Self::Scale(s) => src * s,
            Self::Offset(o) => src + o,
            Self::ScaleOffset { scale, offset } => src * scale + offset,
            Self::Expression {
                scale,
                offset,
                min,
                max,
            } => (src * scale + offset).clamp(min, max),
        }
    }
}

/// One driver link.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Driver {
    pub source: DriverChannel,
    pub dest: DriverChannel,
    pub op: DriverOp,
    pub enabled: bool,
}

impl Driver {
    pub fn copy(source: DriverChannel, dest: DriverChannel) -> Self {
        Self {
            source,
            dest,
            op: DriverOp::Copy,
            enabled: true,
        }
    }

    pub fn apply(&self, pose: &mut CameraPose) {
        if !self.enabled {
            return;
        }
        let v = self.op.apply(self.source.get(pose));
        self.dest.set(pose, v);
    }
}

/// Ordered driver list.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DriverStack {
    pub drivers: Vec<Driver>,
}

impl DriverStack {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, d: Driver) {
        self.drivers.push(d);
    }

    pub fn apply(&self, pose: &mut CameraPose) {
        for d in &self.drivers {
            d.apply(pose);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_driver_fov_from_radius() {
        let mut pose = CameraPose::new(threers::math::Vector3::ZERO, 0.0, 1.2, 4.0);
        pose.fov = 1.0;
        let mut stack = DriverStack::new();
        stack.push(Driver {
            source: DriverChannel::Radius,
            dest: DriverChannel::Fov,
            op: DriverOp::Scale(0.2),
            enabled: true,
        });
        stack.apply(&mut pose);
        assert!((pose.fov - 0.8).abs() < 1e-4);
    }
}
