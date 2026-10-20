/// A CSG boolean operation (three-bvh-csg's constants).
pub type Operation = u8;

pub const ADDITION: Operation = 0;
pub const SUBTRACTION: Operation = 1;
pub const REVERSE_SUBTRACTION: Operation = 2;
pub const INTERSECTION: Operation = 3;
pub const DIFFERENCE: Operation = 4;
pub const HOLLOW_SUBTRACTION: Operation = 5;
pub const HOLLOW_INTERSECTION: Operation = 6;

pub const FRONT_SIDE: i8 = 1;
pub const BACK_SIDE: i8 = -1;
pub const COPLANAR_ALIGNED: i8 = 2;
pub const COPLANAR_OPPOSITE: i8 = -2;

pub const ADD_TRI: u8 = 1;
pub const INVERT_TRI: u8 = 0;
pub const SKIP_TRI: u8 = 2;

pub const DOUBLE_SIDE: u32 = 2;
