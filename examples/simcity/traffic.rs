//! Part of the `simcity` example; see `mod.rs`.
#![allow(dead_code)]

use super::*;

// ---------------------------------------------------------------------------
// Traffic.
// ---------------------------------------------------------------------------

/// Something travelling along a road: a car in a lane, or a pedestrian on the
/// pavement beside it. The two differ only in how they are placed and drawn.
#[derive(Clone, Copy)]
pub(crate) struct Mover {
    /// True when it runs along the X axis (`fixed` is then its z).
    pub(crate) along_x: bool,
    pub(crate) fixed: f32,
    /// Distance along the road, wrapped into `s_lo..s_hi`.
    pub(crate) s: f32,
    /// The run this mover is allowed on. Streets that stop at the river get
    /// one of the two banks rather than the whole width.
    pub(crate) s_lo: f32,
    pub(crate) s_hi: f32,
    /// Current speed. Traffic drives it down behind a queue or a red light
    /// and back up to `cruise` when the road clears.
    pub(crate) speed: f32,
    /// The speed this driver would hold on an empty road.
    pub(crate) cruise: f32,
    /// Bumper to bumper, for the following distance.
    pub(crate) length: f32,
    /// Which stream this belongs to: same road, same direction. Vehicles only
    /// queue behind others in their own lane, and lanes span fleets — a car
    /// has to see the bus in front of it.
    pub(crate) lane: u16,
    /// True on an avenue, which is where the signals are.
    pub(crate) signalled: bool,
    /// +1 or -1.
    pub(crate) dir: f32,
    /// Per-mover PRNG state, advanced at each junction. Turning has to be
    /// decided without a shared `Rng` — the traffic pass runs from `&mut self`
    /// and must stay deterministic frame to frame.
    pub(crate) seed: u32,
    /// Position along the NEW lane at which the current corner began, or NaN
    /// when driving straight. The simulation switches lane instantly — that is
    /// what keeps the queueing and give-way logic one-dimensional — but a
    /// vehicle that pivots ninety degrees in one frame looks like a glitch, so
    /// this drives an arc for the length of the turn.
    pub(crate) turn_s0: f32,
    /// `[entry x, entry z, corner x, corner z]`: the first two control points
    /// of the corner. The third is wherever the new lane is when the arc ends,
    /// so the vehicle rejoins it exactly.
    pub(crate) turn: [f32; 4],
    /// Height the model sits at before any gait animation.
    pub(crate) base_y: f32,
    /// Per-instance size. One pedestrian model at one size reads as a rank of
    /// clones; a spread of 0.9–1.1 does not.
    pub(crate) scale: f32,
    /// Which skin-tone mesh this one is drawn by. Instancing cannot vary a
    /// material per instance, so the population is split across a few meshes
    /// instead — a crowd of identical faces is worse than three extra draws.
    pub(crate) tone: u8,
}

/// Radius of a corner, in metres. Also how far before a junction a vehicle
/// commits to the turn, so there is road left to sweep the arc through.
pub(crate) const TURN_R: f32 = 5.0;

/// Which side of the centreline a vehicle travelling in `dir` belongs on.
///
/// The two axes disagreed, and the offset was the same expression on both:
/// `center - dir * half * 0.44`. Work it out from the engine's own rotation
/// rather than from a cross product — `Quaternion::from_axis_angle(UP, yaw)`
/// maps a vehicle's local `+X` to `(cos yaw, 0, -sin yaw)`, and the drawn yaw
/// is `atan2(fx, fz)`. Travelling `+X` that puts the vehicle's right at `-Z`;
/// travelling `+Z` it puts it at `+X`. So the sign has to flip between the
/// axes, and it was the roads running along Z that were driving on the wrong
/// side. Nothing crashed, because each road is internally consistent — it
/// shows the moment a vehicle turns from one onto the other and swaps sides
/// doing it.
pub(crate) fn lane_side(along_x: bool, dir: f32) -> f32 {
    if along_x {
        -dir
    } else {
        dir
    }
}

impl Mover {
    /// How far through its corner this mover is, `0..1`, or `None` when it is
    /// driving straight.
    /// Distance from the arc's entry point to the corner.
    ///
    /// Derived rather than stored, so the exit leg can be laid out to match it
    /// exactly. Equal legs are what make the quadratic Bezier leave along the
    /// old lane and arrive along the new one instead of rotating the heading
    /// discontinuously at either end.
    pub(crate) fn turn_leg(&self) -> f32 {
        let dx = self.turn[2] - self.turn[0];
        let dz = self.turn[3] - self.turn[1];
        (dx * dx + dz * dz).sqrt()
    }

    pub(crate) fn cornering(&self) -> Option<f32> {
        if self.turn_s0.is_nan() {
            return None;
        }
        let leg = self.turn_leg();
        if leg < 0.05 {
            return None;
        }
        let t = (self.s - self.turn_s0) * self.dir / leg;
        if !(0.0..1.0).contains(&t) {
            return None;
        }
        Some(t)
    }

    pub(crate) fn pose(&self) -> (f32, f32, f32) {
        let lane = |s: f32| -> (f32, f32) {
            if self.along_x {
                (s, self.fixed)
            } else {
                (self.fixed, s)
            }
        };
        let Some(t) = self.cornering() else {
            let (x, z) = lane(self.s);
            let (fx, fz) = if self.along_x {
                (self.dir, 0.0)
            } else {
                (0.0, self.dir)
            };
            return (x, z, fx.atan2(fz));
        };
        let (p0x, p0z) = (self.turn[0], self.turn[1]);
        let (p1x, p1z) = (self.turn[2], self.turn[3]);
        let (p2x, p2z) = lane(self.turn_s0 + self.dir * self.turn_leg());
        let u = 1.0 - t;
        let x = u * u * p0x + 2.0 * u * t * p1x + t * t * p2x;
        let z = u * u * p0z + 2.0 * u * t * p1z + t * t * p2z;
        // Bezier derivative, which is the heading.
        let tx = 2.0 * (u * (p1x - p0x) + t * (p2x - p1x));
        let tz = 2.0 * (u * (p1z - p0z) + t * (p2z - p1z));
        (x, z, tx.atan2(tz))
    }
}

/// How a mover is placed and animated.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Gait {
    /// On the carriageway, flat.
    Drive,
    /// On the pavement, with a walking bob.
    Walk,
    /// On the water, with a long slow roll.
    Float,
}

pub(crate) const BOAT_COLORS: [u32; 3] = [0x7d3a2c, 0x2c4a6b, 0x54585c];

pub(crate) const WALKER_COLORS: [u32; 5] =
    [0x4a5a72, 0x936a48, 0xa8524f, 0x5c7f57, 0x8b8d93];

// Proportions of a 1.75 m adult, in metres from the ground. Everything below
// is measured off these rather than invented per limb, which is what stops the
// figure drifting into caricature.
pub(crate) const P_ANKLE: f32 = 0.10;
pub(crate) const P_KNEE: f32 = 0.48;
pub(crate) const P_HIP: f32 = 0.88;
pub(crate) const P_WAIST: f32 = 1.06;
pub(crate) const P_SHOULDER: f32 = 1.42;
pub(crate) const P_CHIN: f32 = 1.50;
/// Half the distance between the shoulder joints.
pub(crate) const P_SHOULDER_HALF: f32 = 0.155;
/// Half the distance between the hip joints.
pub(crate) const P_HIP_HALF: f32 = 0.085;
/// A torso is about twice as wide as it is deep.
pub(crate) const P_TORSO_FLAT: f32 = 0.70;

/// The clothed part of a person, facing +Z, feet on `y = 0`, 1.73 m tall.
///
/// `swing` in `-1..1` throws one leg and the opposite arm forward and the
/// other pair back. Two of these at `+1` and `-1`, alternated by distance
/// walked, is a walk cycle — which instancing cannot otherwise have, since
/// every instance of a mesh is the same mesh.
///
/// Skin, hair and shoes are NOT here — see `walker_skin_geometry`. The
/// instanced mesh's material colour multiplies every vertex, so a navy coat on
/// one mesh would give its wearer navy hands and a navy face.
pub(crate) fn walker_geometry(swing: f32) -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let coat = Color::WHITE;
    let trousers = Color::new(0.60, 0.60, 0.68);
    let throw = swing * 0.17;

    // Legs: thigh and calf, with a knee between them. The bend is what
    // separates a stride from a pair of scissors.
    for (side, sign) in [(-1.0f32, 1.0f32), (1.0, -1.0)] {
        let x = P_HIP_HALF * side;
        let fwd = throw * sign;
        m.add_limb(
            Vector3::new(x, P_HIP, 0.0),
            Vector3::new(x, P_KNEE, fwd * 0.75),
            0.098,
            0.064,
            8,
            trousers,
            false,
        );
        m.add_limb(
            Vector3::new(x, P_KNEE, fwd * 0.75),
            Vector3::new(x, P_ANKLE, fwd * 1.15),
            0.064,
            0.050,
            8,
            trousers,
            true,
        );
    }

    // Pelvis, then chest: both flattened, and both wider at the ends people
    // are wide at.
    m.add_limb_flat(
        Vector3::new(0.0, P_HIP - 0.03, 0.0),
        Vector3::new(0.0, P_WAIST, 0.0),
        0.150,
        0.132,
        P_TORSO_FLAT,
        14,
        coat,
        true,
    );
    m.add_limb_flat(
        Vector3::new(0.0, P_WAIST - 0.02, 0.0),
        Vector3::new(0.0, P_SHOULDER, 0.0),
        0.132,
        0.158,
        P_TORSO_FLAT,
        14,
        coat,
        // No cap: a flat elliptical plate across the top of the chest reads as
        // a shelf, and the shoulders cover it anyway.
        false,
    );
    // Shoulder line, uncapped for the same reason, with a deltoid rounding
    // each end. Capped, the ends showed as flat wings past the arms.
    m.add_limb_flat(
        Vector3::new(-P_SHOULDER_HALF, P_SHOULDER - 0.03, 0.0),
        Vector3::new(P_SHOULDER_HALF, P_SHOULDER - 0.03, 0.0),
        0.072,
        0.072,
        0.86,
        10,
        coat,
        false,
    );
    for side in [-1.0f32, 1.0] {
        m.add_ellipsoid(
            Vector3::new(P_SHOULDER_HALF * side, P_SHOULDER - 0.035, 0.0),
            0.076,
            0.078,
            0.066,
            4,
            10,
            coat,
        );
    }

    // Arms: upper arm and forearm, swinging opposite the leg on their side,
    // and ending at the wrist — the hand belongs to the skin mesh.
    for (side, sign) in [(-1.0f32, -1.0f32), (1.0, 1.0)] {
        let x = P_SHOULDER_HALF * side;
        let fwd = throw * sign * 0.85;
        let elbow = Vector3::new(x + 0.014 * side, P_SHOULDER - 0.30, fwd * 0.8);
        m.add_limb(
            Vector3::new(x, P_SHOULDER - 0.03, 0.0),
            elbow,
            0.060,
            0.048,
            6,
            coat,
            false,
        );
        m.add_limb(
            elbow,
            Vector3::new(x + 0.026 * side, P_SHOULDER - 0.56, fwd * 1.2),
            0.048,
            0.041,
            6,
            coat,
            true,
        );
    }
    m.build()
}

/// Head, hair, hands and shoes: the parts of a person that are not the coat.
///
/// Drawn over the clothed mesh with a skin-toned material, one mesh per tone
/// per pose, shared by everyone using that tone. An instanced mesh's material
/// colour multiplies every vertex, so folding these into the clothes would
/// give every wearer a face the colour of their jacket — and a single skin
/// mesh for the whole city would give everyone the same face.
///
/// Hair is in here too, as a vertex-colour multiple of the skin tone, so it
/// darkens with it instead of being one colour on every head.
pub(crate) fn walker_skin_geometry(swing: f32) -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let skin = Color::WHITE;
    let hair = Color::new(0.24, 0.20, 0.18);
    let shoe = Color::new(0.26, 0.24, 0.24);
    let throw = swing * 0.17;

    for (side, sign) in [(-1.0f32, 1.0f32), (1.0, -1.0)] {
        let x = P_HIP_HALF * side;
        let fwd = throw * sign * 1.15;
        // A shoe, not a brick: a low sole, an upper that tapers forward, and
        // a rounded toe. At street level the feet are the closest part of a
        // person to the camera and a pair of boxes is the first thing that
        // gives the model away.
        m.add_box(
            Vector3::new(x - 0.052, 0.0, fwd - 0.062),
            Vector3::new(x + 0.052, 0.028, fwd + 0.142),
            scale_color(shoe, 0.8),
            Uv::Unit,
        );
        m.add_limb_flat(
            Vector3::new(x, 0.026, fwd - 0.055),
            Vector3::new(x, 0.026, fwd + 0.095),
            0.052,
            0.044,
            1.15,
            7,
            shoe,
            true,
        );
        m.add_ellipsoid(
            Vector3::new(x, 0.040, fwd + 0.095),
            0.048,
            0.040,
            0.052,
            3,
            8,
            shoe,
        );
    }
    for (side, sign) in [(-1.0f32, -1.0f32), (1.0, 1.0)] {
        let x = P_SHOULDER_HALF * side + 0.026 * side;
        m.add_ellipsoid(
            Vector3::new(x, P_SHOULDER - 0.60, throw * sign * 1.2),
            0.046,
            0.060,
            0.042,
            4,
            8,
            skin,
        );
    }
    m.add_limb(
        Vector3::new(0.0, P_SHOULDER - 0.04, 0.0),
        Vector3::new(0.0, P_CHIN, 0.0),
        0.054,
        0.049,
        9,
        skin,
        false,
    );
    m.add_ellipsoid(
        Vector3::new(0.0, P_CHIN + 0.115, 0.008),
        0.088,
        0.112,
        0.098,
        6,
        12,
        skin,
    );
    // Hair: a cap set back and up, intersecting the skull rather than
    // wrapping it, so there is a face left at the front.
    m.add_ellipsoid(
        Vector3::new(0.0, P_CHIN + 0.150, -0.012),
        0.092,
        0.084,
        0.100,
        5,
        10,
        hair,
    );
    m.build()
}

/// The far level of detail for a person: two boxes. At the distance this
/// takes over it is a couple of pixels tall, and the whole population shares
/// one draw call.
pub(crate) fn walker_impostor() -> BufferGeometry {
    let mut m = MeshBuilder::default();
    m.add_box(
        Vector3::new(-0.20, 0.0, -0.12),
        Vector3::new(0.20, 1.42, 0.12),
        Color::new(0.42, 0.42, 0.46),
        Uv::Unit,
    );
    m.add_box(
        Vector3::new(-0.10, 1.42, -0.09),
        Vector3::new(0.10, 1.70, 0.09),
        Color::new(0.60, 0.46, 0.38),
        Uv::Unit,
    );
    m.build()
}

/// How many skin-tone meshes the pedestrians are split across. Three is
/// enough that a crowd stops reading as clones and costs four extra draws.
pub(crate) const SKIN_TONES: usize = 3;

/// Skin, from the palest to the darkest. The hair in the same mesh is a
/// vertex-colour multiple of these, so it tracks.
pub(crate) const SKIN_PALETTE: [u32; SKIN_TONES] = [0xe0b18e, 0xb9835e, 0x6f4a33];

/// The kinds of thing that use a lane. Larger vehicles matter more than they
/// look: from the air a car is two pixels, and a bus is what makes a street
/// read as a street.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Vehicle {
    Car,
    /// Short, tall, two-box: the other half of what is on a street.
    Hatch,
    /// Low and wide, and the only thing here that overtakes.
    Sports,
    Taxi,
    Pickup,
    Van,
    Bus,
    Lorry,
    /// Long roofline, load space behind the rear seats.
    Estate,
    /// Tall and square. The thing that blocks your view of the junction.
    Suv,
    /// Boxy, high-roofed, with a beacon bar and a stripe.
    Ambulance,
    /// Red, tall, with a ladder on the roof.
    FireEngine,
    /// Box-bodied delivery truck: roller shutter at the back, livery on the
    /// sides, and a tail lift.
    Delivery,
    /// Marked car with a light bar. Its beacons are the one thing in the city
    /// that alternates two colours rather than blinking one.
    Police,
    /// A bicycle and the person on it. Runs in the cycle lane rather than the
    /// running lane, which is the whole reason the cycle lane is there — a
    /// painted lane with nothing in it is paint.
    Bicycle,
}

// Real traffic is mostly white, silver, grey and black with a few strong
// colours in it. Weighting the list that way rather than spreading evenly is
// what stops a street looking like a bag of sweets.
/// Cycling kit, which is nothing like car paint: high-visibility yellows and
/// oranges, and the strong primaries sportswear comes in.
pub(crate) const BIKE_PAINT: [u32; 8] = [
    0xd8e04a, 0xe8853a, 0x2f6ab5, 0xd04a3a, 0x2a2d31, 0x3f8f5c, 0xe0e4e8, 0x8f4a8a,
];

pub(crate) const CAR_PAINT: [u32; 10] = [
    0xd8d9dd, 0xeceef0, 0x2b2d31, 0x8d9199, 0x5a5f66, 0xa3241f, 0x1d4e8a, 0x2f6b44,
    0xd8a13c, 0x6b4a7a,
];
pub(crate) const HATCH_PAINT: [u32; 6] =
    [0xe6e8ea, 0x3a3d42, 0x9aa0a6, 0xc0392b, 0x2e7d64, 0x2f5f8a];
pub(crate) const SPORTS_PAINT: [u32; 4] = [0xc0161a, 0xf2c200, 0x101215, 0x1b6fb5];
pub(crate) const TAXI_PAINT: [u32; 1] = [0xf5c518];
pub(crate) const PICKUP_PAINT: [u32; 4] = [0x2f4f7a, 0x8a3b2c, 0xd7d9db, 0x4a5a45];
pub(crate) const VAN_PAINT: [u32; 4] = [0xe4e6e8, 0x2f5f8a, 0xbdb6a6, 0x6d7a52];
pub(crate) const BUS_PAINT: [u32; 3] = [0xc4342a, 0x2e6f4f, 0x2d4f86];
pub(crate) const LORRY_PAINT: [u32; 4] = [0x39567d, 0x8d8f93, 0x7a3f33, 0xb0932f];
pub(crate) const POLICE_PAINT: [u32; 1] = [0xf0f2f4];
pub(crate) const ESTATE_PAINT: [u32; 5] =
    [0xdcdee0, 0x35383d, 0x7d8288, 0x2f5f4a, 0x8a4a3a];
pub(crate) const SUV_PAINT: [u32; 5] = [0x2b2d31, 0xd8d9dd, 0x4a5560, 0x6b4a30, 0x2f4f7a];
pub(crate) const AMBULANCE_PAINT: [u32; 1] = [0xf4f6f8];
pub(crate) const FIRE_PAINT: [u32; 1] = [0xc0231c];
pub(crate) const DELIVERY_PAINT: [u32; 5] =
    [0xd8a02a, 0x2f6bb5, 0x8f3f2c, 0xf0f2f4, 0x2f7d52];

impl Vehicle {
    pub(crate) fn colors(self) -> &'static [u32] {
        match self {
            Vehicle::Car => &CAR_PAINT,
            Vehicle::Hatch => &HATCH_PAINT,
            Vehicle::Sports => &SPORTS_PAINT,
            Vehicle::Taxi => &TAXI_PAINT,
            Vehicle::Pickup => &PICKUP_PAINT,
            Vehicle::Van => &VAN_PAINT,
            Vehicle::Bus => &BUS_PAINT,
            Vehicle::Lorry => &LORRY_PAINT,
            Vehicle::Estate => &ESTATE_PAINT,
            Vehicle::Suv => &SUV_PAINT,
            Vehicle::Ambulance => &AMBULANCE_PAINT,
            Vehicle::FireEngine => &FIRE_PAINT,
            Vehicle::Delivery => &DELIVERY_PAINT,
            Vehicle::Police => &POLICE_PAINT,
            Vehicle::Bicycle => &BIKE_PAINT,
        }
    }

    /// Metres per second, low and high.
    pub(crate) fn speed(self) -> (f32, f32) {
        match self {
            Vehicle::Car => (7.5, 15.0),
            Vehicle::Hatch => (7.0, 14.0),
            Vehicle::Sports => (9.5, 19.0),
            Vehicle::Taxi => (7.0, 14.5),
            Vehicle::Pickup => (7.0, 13.5),
            Vehicle::Van => (6.5, 12.0),
            Vehicle::Bus => (5.0, 9.5),
            Vehicle::Lorry => (5.5, 10.5),
            Vehicle::Estate => (7.2, 14.2),
            Vehicle::Suv => (7.0, 13.8),
            Vehicle::Ambulance => (8.0, 16.0),
            Vehicle::FireEngine => (6.0, 12.0),
            Vehicle::Delivery => (6.2, 12.5),
            Vehicle::Police => (8.5, 17.0),
            // Slower than everything, and the spread is wide: the difference
            // between somebody commuting and somebody in a hurry is bigger on
            // a bicycle than in a car.
            Vehicle::Bicycle => (3.4, 7.2),
        }
    }

    /// Bumper to bumper, in metres, as drawn — which is `VEHICLE_SCALE`
    /// larger than life. The simulation reads this for following distance and
    /// for fitting a vehicle on a run, so the two cannot drift apart.
    pub(crate) fn length(self) -> f32 {
        VEHICLE_SCALE * self.raw_length()
    }

    fn raw_length(self) -> f32 {
        match self {
            Vehicle::Car => 4.4,
            Vehicle::Hatch => 3.8,
            Vehicle::Sports => 4.2,
            Vehicle::Taxi => 4.6,
            Vehicle::Pickup => 5.2,
            Vehicle::Van => 5.6,
            Vehicle::Bus => 11.0,
            Vehicle::Lorry => 9.5,
            Vehicle::Estate => 4.9,
            Vehicle::Suv => 4.8,
            Vehicle::Ambulance => 6.0,
            Vehicle::FireEngine => 8.2,
            Vehicle::Delivery => 7.0,
            Vehicle::Police => 4.6,
            Vehicle::Bicycle => 1.8,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Vehicle::Car => "cars",
            Vehicle::Hatch => "hatchbacks",
            Vehicle::Sports => "sports cars",
            Vehicle::Taxi => "taxis",
            Vehicle::Pickup => "pickups",
            Vehicle::Van => "vans",
            Vehicle::Bus => "buses",
            Vehicle::Lorry => "lorries",
            Vehicle::Estate => "estates",
            Vehicle::Suv => "4x4s",
            Vehicle::Ambulance => "ambulances",
            Vehicle::FireEngine => "fire engines",
            Vehicle::Delivery => "delivery trucks",
            Vehicle::Police => "police cars",
            Vehicle::Bicycle => "bicycles",
        }
    }

    /// Body pointing +Z, wheels on `y = 0`. Paint is left white so the
    /// material colour tints it; glass and rubber are baked per vertex.
    pub(crate) fn geometry(self) -> BufferGeometry {
        self.body().build()
    }

    /// The body as a builder, so static copies of it can be merged into a
    /// batch rather than instanced. Parked cars are the reason.
    /// The full near mesh: what a vehicle in traffic is drawn with.
    pub(crate) fn body(self) -> MeshBuilder {
        self.body_detail(true)
    }

    /// The mesh a *parked* copy is stamped from.
    ///
    /// Two things are wrong with using the driving mesh for these. It has
    /// somebody sitting in it, which a parked car does not; and there are
    /// about five hundred parked cars in a city, each merged into a static
    /// batch as its own copy with no level of detail at any range — nearly
    /// three hundred thousand triangles, a sixth of the entire scene, and the
    /// only batch of that size with no LOD behind it at all.
    pub(crate) fn body_parked(self) -> MeshBuilder {
        self.body_detail(false)
    }

    fn body_detail(self, driver: bool) -> MeshBuilder {
        let mut m = MeshBuilder::default();
        let paint = Color::WHITE;
        let glass = Color::new(0.15, 0.16, 0.19);
        let rubber = Color::new(0.06, 0.06, 0.07);
        let hub = Color::new(0.62, 0.63, 0.65);
        // A wheel is a cylinder lying on its side, not a box. `add_box` is
        // axis-aligned and was doing the job because at a hundred metres
        // nobody looks — but a car is the one thing in this city the eye
        // already knows the shape of, and a square wheel is the first thing it
        // finds. `add_limb` builds a cylinder between two points, so the axis
        // runs across the car for free.
        // Wheels are sixty per cent of a car's triangles: four tyres at twelve
        // capped segments and four hubs at ten is three hundred and fifty, of
        // a body that comes to under six hundred. That is the right budget for
        // something driving past the camera and quite the wrong one for the
        // five hundred cars standing at kerbs, each merged into a static batch
        // as its own copy with no level of detail at any distance.
        let wheels = |m: &mut MeshBuilder, half_w: f32, r: f32, zs: &[f32]| {
            for &z in zs {
                for sx in [-1.0f32, 1.0] {
                    let x = half_w * sx;
                    m.add_limb(
                        Vector3::new(x - 0.17, r, z),
                        Vector3::new(x + 0.17, r, z),
                        r,
                        r,
                        if driver { 12 } else { 7 },
                        rubber,
                        true,
                    );
                    if !driver {
                        // The hub is what stops a wheel reading as a black
                        // disc — worth forty triangles on a moving car and not
                        // on one parked against a kerb with the hub half in
                        // shadow.
                        continue;
                    }
                    // A hub, set slightly proud of the tyre's outer face. Two
                    // tones is what stops a wheel reading as a black disc.
                    m.add_limb(
                        Vector3::new(x + sx * 0.09, r, z),
                        Vector3::new(x + sx * 0.185, r, z),
                        r * 0.60,
                        r * 0.54,
                        10,
                        hub,
                        true,
                    );
                }
            }
        };
        // The fittings every road vehicle carries. Added to the *near* mesh
        // only — every one of these types has an impostor for distance, which
        // is what makes it safe to spend triangles here: at the camera this
        // example now ships with, a car is the most-looked-at moving thing in
        // the frame, and none of this exists on the box it becomes at range.
        let fittings = |m: &mut MeshBuilder, hw: f32, front: f32, back: f32, sill: f32| {
            let glass = Color::new(0.15, 0.16, 0.19);
            let chrome = Color::new(0.62, 0.63, 0.66);
            let plate = Color::new(0.88, 0.88, 0.84);
            let dark = Color::new(0.10, 0.10, 0.11);
            for sx in [-1.0f32, 1.0] {
                // Wing mirror: a stalk and a housing, which is the one piece
                // of a car that breaks its silhouette.
                m.add_limb(
                    Vector3::new(sx * (hw - 0.04), sill + 0.30, front - 1.30),
                    Vector3::new(sx * (hw + 0.16), sill + 0.34, front - 1.42),
                    0.035,
                    0.030,
                    4,
                    dark,
                    false,
                );
                m.add_box(
                    Vector3::new(sx * (hw + 0.10) - 0.09, sill + 0.28, front - 1.50),
                    Vector3::new(sx * (hw + 0.10) + 0.09, sill + 0.42, front - 1.36),
                    dark,
                    Uv::Unit,
                );
                // Door handle.
                m.add_box(
                    Vector3::new(sx * (hw + 0.01) - 0.02, sill + 0.16, -0.30),
                    Vector3::new(sx * (hw + 0.01) + 0.02, sill + 0.24, 0.10),
                    chrome,
                    Uv::Unit,
                );
            }
            // Number plates, front and rear.
            for (z, d) in [(front + 0.01, 1.0f32), (back - 0.01, -1.0)] {
                m.add_box(
                    Vector3::new(-0.30, sill - 0.06, z - d * 0.02),
                    Vector3::new(0.30, sill + 0.10, z + d * 0.02),
                    plate,
                    Uv::Unit,
                );
            }
            // Exhaust, offset to one side as they are.
            m.add_limb(
                Vector3::new(0.42, sill - 0.30, back + 0.10),
                Vector3::new(0.42, sill - 0.30, back - 0.06),
                0.055,
                0.06,
                6,
                chrome,
                true,
            );
            // Somebody driving it. A head and shoulders behind the screen is
            // the difference between a car and a parked prop — which is
            // exactly why the parked copies leave it out.
            if !driver {
                return;
            }
            m.add_ellipsoid(
                Vector3::new(-0.34, sill + 0.86, 0.30),
                0.11,
                0.13,
                0.11,
                3,
                6,
                Color::new(0.55, 0.42, 0.34),
            );
            m.add_limb_flat(
                Vector3::new(-0.34, sill + 0.34, 0.24),
                Vector3::new(-0.34, sill + 0.76, 0.26),
                0.17,
                0.15,
                0.7,
                6,
                glass,
                false,
            );
        };
        // A saloon shell the marked variants reuse: same body, different kit.
        let saloon = |m: &mut MeshBuilder, roof: f32| {
            m.add_box(
                Vector3::new(-0.92, 0.30, -2.20),
                Vector3::new(0.92, 0.88, 2.20),
                paint,
                Uv::Unit,
            );
            m.add_box(
                Vector3::new(-0.86, 0.88, -2.05),
                Vector3::new(0.86, 1.04, 2.05),
                paint,
                Uv::Unit,
            );
            m.add_box(
                Vector3::new(-0.82, 1.04, -1.30),
                Vector3::new(0.82, roof, 0.95),
                glass,
                Uv::Unit,
            );
            m.add_box(
                Vector3::new(-0.78, 1.04, -1.24),
                Vector3::new(0.78, roof + 0.04, 0.88),
                paint,
                Uv::Unit,
            );
        };
        match self {
            Vehicle::Bicycle => {
                // Two wheels, a frame, and somebody on it. The rider is most
                // of the silhouette — a bicycle on its own is a few thin lines
                // and reads as nothing at all, which is why every game that
                // draws cyclists draws the person first.
                let frame = paint;
                let tyre = Color::from_hex(0x22262b);
                for z in [-0.58f32, 0.58] {
                    // Wheel: a thin cylinder on its side.
                    m.add_limb(
                        Vector3::new(-0.03, 0.34, z),
                        Vector3::new(0.03, 0.34, z),
                        0.34,
                        0.34,
                        9,
                        tyre,
                        true,
                    );
                }
                // Frame: down tube and seat tube.
                m.add_limb(
                    Vector3::new(0.0, 0.34, 0.58),
                    Vector3::new(0.0, 0.78, -0.16),
                    0.035,
                    0.035,
                    4,
                    frame,
                    false,
                );
                m.add_limb(
                    Vector3::new(0.0, 0.34, -0.58),
                    Vector3::new(0.0, 0.80, -0.10),
                    0.035,
                    0.035,
                    4,
                    frame,
                    false,
                );
                m.add_limb(
                    Vector3::new(0.0, 0.34, -0.58),
                    Vector3::new(0.0, 0.36, 0.58),
                    0.03,
                    0.03,
                    4,
                    frame,
                    false,
                );
                // Handlebars.
                m.add_limb(
                    Vector3::new(-0.22, 0.98, 0.50),
                    Vector3::new(0.22, 0.98, 0.50),
                    0.028,
                    0.028,
                    4,
                    Color::from_hex(0x3a3f45),
                    false,
                );
                // Rider: legs, torso, head, leaning forward over the bars.
                let kit = paint;
                let skin = Color::from_hex(0x9a6f52);
                for sx in [-0.11f32, 0.11] {
                    m.add_limb(
                        Vector3::new(sx, 0.34, 0.02),
                        Vector3::new(sx, 0.86, -0.10),
                        0.075,
                        0.065,
                        5,
                        Color::from_hex(0x2b3038),
                        false,
                    );
                }
                m.add_limb_flat(
                    Vector3::new(0.0, 0.84, -0.10),
                    Vector3::new(0.0, 1.34, 0.20),
                    0.20,
                    0.17,
                    0.62,
                    6,
                    kit,
                    false,
                );
                for sx in [-0.19f32, 0.19] {
                    m.add_limb(
                        Vector3::new(sx, 1.28, 0.16),
                        Vector3::new(sx * 0.9, 1.02, 0.46),
                        0.05,
                        0.045,
                        4,
                        kit,
                        false,
                    );
                }
                m.add_ellipsoid(
                    Vector3::new(0.0, 1.44, 0.26),
                    0.11,
                    0.13,
                    0.12,
                    3,
                    6,
                    skin,
                );
                // Helmet.
                m.add_ellipsoid(
                    Vector3::new(0.0, 1.49, 0.25),
                    0.13,
                    0.10,
                    0.14,
                    2,
                    6,
                    scale_color(kit, 1.15),
                );
            }
            Vehicle::Delivery => {
                // A box on a cab. The box is deliberately taller and squarer
                // than the van's: at a distance the silhouette is the only
                // thing telling the two apart.
                m.add_box(
                    Vector3::new(-1.08, 0.62, -3.45),
                    Vector3::new(1.08, 3.05, 1.35),
                    paint,
                    Uv::Unit,
                );
                m.add_box(
                    Vector3::new(-1.02, 0.55, 1.35),
                    Vector3::new(1.02, 2.35, 3.45),
                    scale_color(paint, 0.92),
                    Uv::Unit,
                );
                m.add_box(
                    Vector3::new(-0.96, 1.35, 1.40),
                    Vector3::new(0.96, 2.30, 3.38),
                    glass,
                    Uv::Unit,
                );
                // Roller shutter at the tail, and the lift folded up under it.
                m.add_box(
                    Vector3::new(-1.02, 0.72, -3.52),
                    Vector3::new(1.02, 2.75, -3.44),
                    Color::from_hex(0xb9b3a6),
                    Uv::Unit,
                );
                m.add_box(
                    Vector3::new(-0.98, 0.40, -3.72),
                    Vector3::new(0.98, 0.60, -3.44),
                    Color::from_hex(0x5a5f66),
                    Uv::Unit,
                );
                // A band of livery down each flank — a delivery truck is a
                // moving billboard and that is most of how it reads.
                for sx in [-1.0f32, 1.0] {
                    m.add_box(
                        Vector3::new(sx * 1.09 - 0.02, 1.55, -3.10),
                        Vector3::new(sx * 1.09 + 0.02, 2.55, 1.05),
                        scale_color(paint, 1.25),
                        Uv::Unit,
                    );
                }
                wheels(&mut m, 0.94, 0.42, &[-2.35, 2.30]);
            }
            Vehicle::Estate => {
                // A saloon with the roof carried straight back to the tail.
                saloon(&mut m, 1.44);
                fittings(&mut m, 0.92, 2.20, -2.45, 0.30);
                m.add_box(
                    Vector3::new(-0.92, 0.30, -2.45),
                    Vector3::new(0.92, 1.04, -2.05),
                    paint,
                    Uv::Unit,
                );
                m.add_box(
                    Vector3::new(-0.82, 1.04, -2.35),
                    Vector3::new(0.82, 1.42, -1.25),
                    glass,
                    Uv::Unit,
                );
                m.add_box(
                    Vector3::new(-0.80, 1.40, -2.30),
                    Vector3::new(0.80, 1.50, 0.90),
                    paint,
                    Uv::Unit,
                );
                // Roof bars, which is most of what says estate at a distance.
                for sx in [-1.0f32, 1.0] {
                    m.add_box(
                        Vector3::new(sx * 0.58 - 0.05, 1.50, -2.10),
                        Vector3::new(sx * 0.58 + 0.05, 1.57, 0.70),
                        Color::from_hex(0x2b2d31),
                        Uv::Unit,
                    );
                }
                wheels(&mut m, 0.80, 0.34, &[-1.55, 1.50]);
            }
            Vehicle::Suv => {
                // Tall, upright and square: everything sits higher.
                m.add_box(
                    Vector3::new(-0.96, 0.52, -2.35),
                    Vector3::new(0.96, 1.30, 2.35),
                    paint,
                    Uv::Unit,
                );
                m.add_box(
                    Vector3::new(-0.90, 1.30, -2.20),
                    Vector3::new(0.90, 2.02, 1.55),
                    glass,
                    Uv::Unit,
                );
                m.add_box(
                    Vector3::new(-0.86, 1.94, -2.15),
                    Vector3::new(0.86, 2.10, 1.45),
                    paint,
                    Uv::Unit,
                );
                // Wheel arches and a bull bar, both squared off.
                m.add_box(
                    Vector3::new(-0.99, 0.34, -2.42),
                    Vector3::new(0.99, 0.58, 2.42),
                    Color::from_hex(0x2b2d31),
                    Uv::Unit,
                );
                fittings(&mut m, 0.96, 2.35, -2.35, 0.52);
                wheels(&mut m, 0.86, 0.44, &[-1.48, 1.52]);
            }
            Vehicle::Ambulance => {
                // A box body on a van cab, with a beacon bar and a stripe.
                m.add_box(
                    Vector3::new(-1.00, 0.42, -2.95),
                    Vector3::new(1.00, 2.60, 1.15),
                    paint,
                    Uv::Unit,
                );
                m.add_box(
                    Vector3::new(-0.96, 0.42, 1.15),
                    Vector3::new(0.96, 1.95, 2.95),
                    paint,
                    Uv::Unit,
                );
                m.add_box(
                    Vector3::new(-0.90, 1.30, 1.20),
                    Vector3::new(0.90, 1.92, 2.88),
                    glass,
                    Uv::Unit,
                );
                for sx in [-1.0f32, 1.0] {
                    // Battenburg, in the green and yellow of an ambulance
                    // rather than the blue of a police car.
                    for i in 0..8 {
                        let z = -2.8 + i as f32 * 0.48;
                        m.add_box(
                            Vector3::new(sx * 1.01 - 0.015, 1.15, z),
                            Vector3::new(sx * 1.01 + 0.015, 1.62, z + 0.46),
                            if i % 2 == 0 {
                                Color::from_hex(0x2f8f4a)
                            } else {
                                Color::from_hex(0xe8e02c)
                            },
                            Uv::Unit,
                        );
                    }
                }
                m.add_box(
                    Vector3::new(-0.70, 2.60, 0.30),
                    Vector3::new(0.70, 2.74, 0.62),
                    Color::from_hex(0x26282c),
                    Uv::Unit,
                );
                wheels(&mut m, 0.88, 0.40, &[-1.90, 1.95]);
            }
            Vehicle::FireEngine => {
                m.add_box(
                    Vector3::new(-1.14, 0.55, -4.05),
                    Vector3::new(1.14, 2.75, 1.30),
                    paint,
                    Uv::Unit,
                );
                m.add_box(
                    Vector3::new(-1.10, 0.55, 1.30),
                    Vector3::new(1.10, 2.35, 4.05),
                    paint,
                    Uv::Unit,
                );
                m.add_box(
                    Vector3::new(-1.02, 1.55, 1.35),
                    Vector3::new(1.02, 2.30, 3.98),
                    glass,
                    Uv::Unit,
                );
                // Lockers down the flanks, and the ladder on the roof.
                for sx in [-1.0f32, 1.0] {
                    for i in 0..4 {
                        let z = -3.8 + i as f32 * 1.25;
                        m.add_box(
                            Vector3::new(sx * 1.15 - 0.03, 0.95, z),
                            Vector3::new(sx * 1.15 + 0.03, 1.95, z + 1.05),
                            Color::from_hex(0x8f8a80),
                            Uv::Unit,
                        );
                    }
                    m.add_box(
                        Vector3::new(sx * 0.55 - 0.06, 2.78, -3.90),
                        Vector3::new(sx * 0.55 + 0.06, 2.90, 3.20),
                        Color::from_hex(0xb9b3a6),
                        Uv::Unit,
                    );
                }
                for i in 0..9 {
                    let z = -3.7 + i as f32 * 0.8;
                    m.add_box(
                        Vector3::new(-0.55, 2.80, z),
                        Vector3::new(0.55, 2.87, z + 0.10),
                        Color::from_hex(0xb9b3a6),
                        Uv::Unit,
                    );
                }
                m.add_box(
                    Vector3::new(-0.80, 2.75, 0.20),
                    Vector3::new(0.80, 2.90, 0.55),
                    Color::from_hex(0x26282c),
                    Uv::Unit,
                );
                wheels(&mut m, 1.02, 0.48, &[-2.70, 2.60]);
            }
            Vehicle::Taxi => {
                saloon(&mut m, 1.44);
                fittings(&mut m, 0.92, 2.20, -2.20, 0.30);
                wheels(&mut m, 0.80, 0.34, &[-1.42, 1.46]);
                // Roof sign and a chequer band along the flank, which is what
                // makes a yellow car a taxi.
                m.add_box(
                    Vector3::new(-0.30, 1.48, -0.24),
                    Vector3::new(0.30, 1.70, 0.24),
                    Color::from_hex(0x2b2d31),
                    Uv::Unit,
                );
                for i in 0..8 {
                    let z = -1.6 + i as f32 * 0.42;
                    let c = if i % 2 == 0 {
                        Color::from_hex(0x1c1d20)
                    } else {
                        Color::from_hex(0xf0f0f0)
                    };
                    for sx in [-1.0f32, 1.0] {
                        m.add_box(
                            Vector3::new(sx * 0.93 - 0.015, 0.62, z),
                            Vector3::new(sx * 0.93 + 0.015, 0.80, z + 0.40),
                            c,
                            Uv::Unit,
                        );
                    }
                }
            }
            Vehicle::Police => {
                saloon(&mut m, 1.44);
                fittings(&mut m, 0.92, 2.20, -2.20, 0.30);
                wheels(&mut m, 0.80, 0.34, &[-1.42, 1.46]);
                // Battenburg down the side: high-contrast blocks, which is
                // what a marked car reads as at any distance.
                for i in 0..7 {
                    let z = -1.55 + i as f32 * 0.46;
                    let c = if i % 2 == 0 {
                        Color::from_hex(0x1d4e8a)
                    } else {
                        Color::from_hex(0xe8c81c)
                    };
                    for sx in [-1.0f32, 1.0] {
                        m.add_box(
                            Vector3::new(sx * 0.93 - 0.015, 0.46, z),
                            Vector3::new(sx * 0.93 + 0.015, 0.84, z + 0.44),
                            c,
                            Uv::Unit,
                        );
                    }
                }
                // The bar itself is unlit body; the lenses are lamps.
                m.add_box(
                    Vector3::new(-0.66, 1.46, -0.16),
                    Vector3::new(0.66, 1.60, 0.16),
                    Color::from_hex(0x26282c),
                    Uv::Unit,
                );
            }
            Vehicle::Hatch => {
                m.add_box(
                    Vector3::new(-0.88, 0.32, -1.86),
                    Vector3::new(0.88, 0.92, 1.86),
                    paint,
                    Uv::Unit,
                );
                // Two-box: the cabin runs right back to the tail, and the
                // rear screen is nearly upright.
                m.add_box(
                    Vector3::new(-0.82, 0.92, -1.80),
                    Vector3::new(0.82, 1.52, 0.72),
                    glass,
                    Uv::Unit,
                );
                m.add_box(
                    Vector3::new(-0.78, 1.30, -1.74),
                    Vector3::new(0.78, 1.58, 0.60),
                    paint,
                    Uv::Unit,
                );
                m.add_box(
                    Vector3::new(-0.84, 0.92, 0.72),
                    Vector3::new(0.84, 1.08, 1.80),
                    paint,
                    Uv::Unit,
                );
                fittings(&mut m, 0.88, 1.86, -1.86, 0.32);
                wheels(&mut m, 0.78, 0.32, &[-1.20, 1.24]);
            }
            Vehicle::Sports => {
                // Low, wide, and long-nosed. Nothing else on the road is under
                // a metre and a half tall, so the silhouette does the work.
                m.add_box(
                    Vector3::new(-0.96, 0.24, -2.10),
                    Vector3::new(0.96, 0.66, 2.10),
                    paint,
                    Uv::Unit,
                );
                m.add_box(
                    Vector3::new(-0.90, 0.66, -1.90),
                    Vector3::new(0.90, 0.80, 1.95),
                    paint,
                    Uv::Unit,
                );
                m.add_box(
                    Vector3::new(-0.74, 0.80, -1.05),
                    Vector3::new(0.74, 1.14, 0.30),
                    glass,
                    Uv::Unit,
                );
                m.add_box(
                    Vector3::new(-0.70, 0.80, -1.00),
                    Vector3::new(0.70, 1.18, 0.22),
                    paint,
                    Uv::Unit,
                );
                // Sill set low to suit the cabin: this is the one car whose
                // roof a driver at saloon height would come through.
                fittings(&mut m, 0.96, 2.10, -2.10, 0.14);
                // Rear wing.
                m.add_box(
                    Vector3::new(-0.80, 0.86, -2.16),
                    Vector3::new(0.80, 0.94, -1.96),
                    Color::from_hex(0x1a1c1f),
                    Uv::Unit,
                );
                wheels(&mut m, 0.86, 0.33, &[-1.36, 1.44]);
            }
            Vehicle::Pickup => {
                m.add_box(
                    Vector3::new(-0.94, 0.44, -2.55),
                    Vector3::new(0.94, 1.02, 2.55),
                    paint,
                    Uv::Unit,
                );
                // Cab forward, open bed behind, with sides standing proud.
                m.add_box(
                    Vector3::new(-0.88, 1.02, 0.10),
                    Vector3::new(0.88, 1.78, 1.95),
                    glass,
                    Uv::Unit,
                );
                m.add_box(
                    Vector3::new(-0.84, 1.02, 0.16),
                    Vector3::new(0.84, 1.84, 1.88),
                    paint,
                    Uv::Unit,
                );
                for (bx0, bz0, bx1, bz1) in [
                    (-0.94, -2.52, -0.80, 0.05),
                    (0.80, -2.52, 0.94, 0.05),
                    (-0.94, -2.52, 0.94, -2.40),
                ] {
                    m.add_box(
                        Vector3::new(bx0, 1.02, bz0),
                        Vector3::new(bx1, 1.42, bz1),
                        paint,
                        Uv::Unit,
                    );
                }
                wheels(&mut m, 0.86, 0.38, &[-1.62, 1.62]);
            }
            Vehicle::Car => {
                fittings(&mut m, 0.92, 2.20, -2.20, 0.30);
                m.add_box(
                    Vector3::new(-0.92, 0.30, -2.20),
                    Vector3::new(0.92, 0.88, 2.20),
                    paint,
                    Uv::Unit,
                );
                // Bonnet and boot: a narrower deck that stops short of the
                // cabin, so the silhouette is not one brick.
                m.add_box(
                    Vector3::new(-0.86, 0.88, -2.05),
                    Vector3::new(0.86, 1.04, 2.05),
                    paint,
                    Uv::Unit,
                );
                m.add_box(
                    Vector3::new(-0.80, 1.04, -1.05),
                    Vector3::new(0.80, 1.44, 1.00),
                    glass,
                    Uv::Unit,
                );
                m.add_box(
                    Vector3::new(-0.74, 1.44, -0.92),
                    Vector3::new(0.74, 1.52, 0.80),
                    paint,
                    Uv::Unit,
                );
                // Wing mirrors. Two boxes each 12 cm across, and the thing
                // that stops a car reading as a loaf at street level.
                for sx in [-1.0f32, 1.0] {
                    let x = 0.94 * sx;
                    m.add_box(
                        Vector3::new(x - 0.11, 1.10, 0.86),
                        Vector3::new(x + 0.11, 1.24, 1.04),
                        paint,
                        Uv::Unit,
                    );
                }
                wheels(&mut m, 0.86, 0.36, &[-1.40, 1.40]);
            }
            Vehicle::Van => {
                m.add_box(
                    Vector3::new(-1.00, 0.42, -2.80),
                    Vector3::new(1.00, 2.45, 1.55),
                    paint,
                    Uv::Unit,
                );
                m.add_box(
                    Vector3::new(-0.96, 0.42, 1.55),
                    Vector3::new(0.96, 1.30, 2.80),
                    paint,
                    Uv::Unit,
                );
                m.add_box(
                    Vector3::new(-0.90, 1.30, 1.50),
                    Vector3::new(0.90, 2.05, 1.92),
                    glass,
                    Uv::Unit,
                );
                wheels(&mut m, 0.94, 0.40, &[-1.85, 1.95]);
            }
            Vehicle::Bus => {
                m.add_box(
                    Vector3::new(-1.27, 0.52, -5.50),
                    Vector3::new(1.27, 3.10, 5.50),
                    paint,
                    Uv::Unit,
                );
                // Window band, proud of the body so it reads as glazing.
                m.add_box(
                    Vector3::new(-1.30, 1.62, -5.30),
                    Vector3::new(1.30, 2.52, 5.35),
                    glass,
                    Uv::Unit,
                );
                m.add_box(
                    Vector3::new(-1.18, 3.10, -5.10),
                    Vector3::new(1.18, 3.28, 4.60),
                    Color::new(0.80, 0.80, 0.78),
                    Uv::Unit,
                );
                // Destination blind over the windscreen.
                m.add_box(
                    Vector3::new(-0.80, 2.60, 5.48),
                    Vector3::new(0.80, 2.94, 5.56),
                    Color::new(0.90, 0.86, 0.55),
                    Uv::Unit,
                );
                wheels(&mut m, 1.22, 0.48, &[-3.70, 3.60]);
            }
            Vehicle::Lorry => {
                m.add_box(
                    Vector3::new(-1.24, 0.58, 1.95),
                    Vector3::new(1.24, 2.95, 4.70),
                    paint,
                    Uv::Unit,
                );
                m.add_box(
                    Vector3::new(-1.20, 1.75, 4.55),
                    Vector3::new(1.20, 2.72, 4.86),
                    glass,
                    Uv::Unit,
                );
                // Trailer, in a plain body colour rather than the cab paint.
                m.add_box(
                    Vector3::new(-1.25, 1.05, -4.75),
                    Vector3::new(1.25, 3.40, 1.70),
                    Color::new(0.82, 0.82, 0.80),
                    Uv::Unit,
                );
                wheels(&mut m, 1.19, 0.46, &[-3.90, -3.00, 3.70]);
            }
        }
        m
    }
}

/// The far level of detail for a vehicle: its silhouette as one box, in a
/// neutral body colour. Everything past the fleet's LOD distance shares this
/// single instanced mesh whatever its paint.
pub(crate) fn vehicle_impostor(kind: Vehicle) -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let (hw, h, hl) = match kind {
        Vehicle::Car | Vehicle::Taxi | Vehicle::Police => (0.90, 1.45, 2.20),
        Vehicle::Estate => (0.92, 1.55, 2.45),
        Vehicle::Suv => (0.98, 1.90, 2.40),
        Vehicle::Ambulance => (1.02, 2.55, 3.00),
        Vehicle::FireEngine => (1.20, 3.05, 4.10),
        Vehicle::Delivery => (1.10, 2.90, 3.50),
        Vehicle::Hatch => (0.88, 1.50, 1.90),
        Vehicle::Sports => (0.94, 1.18, 2.10),
        Vehicle::Bicycle => (0.30, 1.60, 0.90),
        Vehicle::Pickup => (0.95, 1.80, 2.60),
        Vehicle::Van => (0.99, 2.45, 2.80),
        Vehicle::Bus => (1.27, 3.10, 5.50),
        Vehicle::Lorry => (1.25, 3.30, 4.75),
    };
    m.add_box(
        Vector3::new(-hw, 0.10, -hl),
        Vector3::new(hw, h, hl),
        Color::new(0.55, 0.56, 0.58),
        Uv::Unit,
    );
    m.build()
}

/// The lamps at one end of a car, as their own mesh so every car in the city
/// can share two instanced draws for its lights instead of one per paint
/// colour. Modelled proud of the body so they read from the front.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lamp {
    Head,
    Tail,
    /// The pool the headlamps throw on the road ahead. Unlit and faded out
    /// per vertex, which an emissive material cannot do — emissive is one
    /// value for the whole mesh.
    Beam,
    /// Brake lamps, drawn only while the vehicle is stopping. The simulation
    /// already knows: a vehicle queueing at a red or behind a bus has its
    /// speed pulled down, and that is exactly when its brake lights are on.
    Brake,
    /// Indicators, drawn only while a corner is in progress and only on half
    /// the blink cycle. Also free — `Mover::cornering` is the same state the
    /// drawn pose is following round the turn.
    Indicate,
    /// The near-side lens of a light bar, and the off-side one. Two meshes
    /// shown on opposite halves of the blink so the bar alternates two colours
    /// rather than flashing one — which is the whole visual signature of an
    /// emergency vehicle, and nothing else in the city does it.
    BeaconA,
    BeaconB,
}

pub(crate) fn car_lamp_geometry(part: Lamp) -> BufferGeometry {
    let mut m = MeshBuilder::default();
    match part {
        Lamp::Head | Lamp::Tail | Lamp::Brake => {
            let (z0, z1) = if part == Lamp::Head {
                (2.06, 2.22)
            } else {
                (-2.24, -2.06)
            };
            // Brake lamps sit inboard of the tail lights and a little higher,
            // so both can be lit at once without becoming one red bar.
            let (half, y0, y1) = match part {
                Lamp::Brake => (0.20, 0.60, 0.86),
                _ => (0.26, 0.52, 0.84),
            };
            for sx in [-1.0f32, 1.0] {
                let x = 0.58 * sx;
                m.add_box(
                    Vector3::new(x - half, y0, z0),
                    Vector3::new(x + half, y1, z1),
                    Color::WHITE,
                    Uv::Unit,
                );
            }
            // High-level brake light in the rear screen, which is the part you
            // actually see from behind in traffic.
            if part == Lamp::Brake {
                m.add_box(
                    Vector3::new(-0.34, 1.36, -1.62),
                    Vector3::new(0.34, 1.46, -1.54),
                    Color::WHITE,
                    Uv::Unit,
                );
            }
        }
        Lamp::BeaconA | Lamp::BeaconB => {
            // One lens each side of the bar, swapped between the two meshes.
            let near = part == Lamp::BeaconA;
            for (sx, lit) in [(-1.0f32, near), (1.0, !near)] {
                if !lit {
                    continue;
                }
                // Two heights: a car's roof and a box body's. Drawing both
                // costs a handful of triangles and means one mesh serves the
                // police car, the ambulance and the fire engine.
                for y in [1.47f32, 2.62] {
                    m.add_box(
                        Vector3::new(sx * 0.36 - 0.28, y, -0.14),
                        Vector3::new(sx * 0.36 + 0.28, y + 0.14, 0.14),
                        Color::WHITE,
                        Uv::Unit,
                    );
                }
            }
        }
        Lamp::Indicate => {
            // Front and rear corners, outboard of everything else.
            for sx in [-1.0f32, 1.0] {
                let x = 0.86 * sx;
                for (z0, z1) in [(2.02, 2.20), (-2.22, -2.04)] {
                    m.add_box(
                        Vector3::new(x - 0.14, 0.56, z0),
                        Vector3::new(x + 0.14, 0.82, z1),
                        Color::WHITE,
                        Uv::Unit,
                    );
                }
            }
        }
        Lamp::Beam => {
            // A splayed quad on the tarmac, bright at the bumper and gone by
            // the far end. Wound counter-clockwise seen from above.
            let warm = Color::from_hex(0xffe9b8);
            let near = 2.3;
            let far = 11.0;
            let (wn, wf) = (0.95, 2.6);
            m.quad(
                [
                    [-wn, 0.02, near],
                    [wn, 0.02, near],
                    [wf, 0.02, far],
                    [-wf, 0.02, far],
                ],
                [0.0, 1.0, 0.0],
                Uv::Unit,
                warm,
            );
            // Overwrite the far pair's colours so the pool fades out.
            let n = m.col.len();
            for i in (n - 6)..n {
                m.col[i] = 0.0;
            }
        }
    }
    m.build()
}

/// A river barge, bow pointing +Z: hull, deck, wheelhouse aft, and a stack.
/// The kinds of vessel on the water.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Boat {
    /// Flat-decked freight barge. The one that was already here.
    Barge,
    /// Short, high-bowed, with a wheelhouse most of its length.
    Tug,
    /// A box boat: stacks of containers and an island right aft.
    Container,
    /// Sail up, hull low.
    Yacht,
}

pub(crate) const BOAT_KINDS: [Boat; 4] = [Boat::Barge, Boat::Tug, Boat::Container, Boat::Yacht];

impl Boat {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Boat::Barge => "barges",
            Boat::Tug => "tugs",
            Boat::Container => "container ships",
            Boat::Yacht => "yachts",
        }
    }

    pub(crate) fn length(self) -> f32 {
        match self {
            Boat::Barge => 22.0,
            Boat::Tug => 14.0,
            Boat::Container => 46.0,
            Boat::Yacht => 11.0,
        }
    }

    /// Metres per second, low and high.
    pub(crate) fn speed(self) -> (f32, f32) {
        match self {
            Boat::Barge => (2.4, 4.2),
            Boat::Tug => (3.0, 5.5),
            Boat::Container => (2.0, 3.4),
            Boat::Yacht => (2.6, 5.0),
        }
    }

    pub(crate) fn geometry(self) -> BufferGeometry {
        match self {
            Boat::Barge => boat_geometry(),
            Boat::Tug => tug_geometry(),
            Boat::Container => container_geometry(),
            Boat::Yacht => yacht_geometry(),
        }
    }
}

/// A tug: short, deep and mostly wheelhouse, with a heavy fendered bow.
pub(crate) fn tug_geometry() -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let hull = Color::WHITE;
    let deck = Color::new(0.30, 0.29, 0.26);
    let house = Color::new(0.88, 0.87, 0.84);
    let dark = Color::new(0.12, 0.13, 0.15);
    m.add_limb_flat(
        Vector3::new(0.0, -0.5, -6.6),
        Vector3::new(0.0, -0.4, 5.6),
        2.05,
        1.5,
        0.80,
        8,
        hull,
        true,
    );
    m.add_box(
        Vector3::new(-1.95, 0.28, -6.4),
        Vector3::new(1.95, 0.46, 5.4),
        deck,
        Uv::Unit,
    );
    // Wheelhouse in two tiers, well forward.
    m.add_box(
        Vector3::new(-1.5, 0.46, -2.6),
        Vector3::new(1.5, 2.5, 2.4),
        house,
        Uv::Unit,
    );
    m.add_box(
        Vector3::new(-1.15, 2.5, -1.9),
        Vector3::new(1.15, 4.1, 1.7),
        house,
        Uv::Unit,
    );
    m.add_box(
        Vector3::new(-1.18, 3.1, -1.95),
        Vector3::new(1.18, 3.9, 1.75),
        dark,
        Uv::Unit,
    );
    // Funnel and mast.
    m.add_cylinder(
        Vector3::new(0.0, 4.1, -1.0),
        0.42,
        0.40,
        1.5,
        8,
        Color::new(0.55, 0.14, 0.12),
        true,
        Uv::Unit,
    );
    m.add_cylinder(
        Vector3::new(0.0, 4.1, 1.0),
        0.08,
        0.06,
        3.4,
        5,
        house,
        false,
        Uv::Unit,
    );
    // Fenders: old tyres down the flanks, which is the whole look of a tug.
    for sx in [-1.0f32, 1.0] {
        for k in 0..7 {
            let z = -5.6 + k as f32 * 1.7;
            m.add_limb(
                Vector3::new(sx * 1.96, 0.1, z),
                Vector3::new(sx * 2.12, 0.1, z),
                0.34,
                0.34,
                7,
                dark,
                true,
            );
        }
    }
    m.build()
}

/// A container ship: a long low hull, stacks of boxes, and the island aft.
pub(crate) fn container_geometry() -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let hull = Color::WHITE;
    let deck = Color::new(0.34, 0.33, 0.30);
    let house = Color::new(0.86, 0.86, 0.83);
    let dark = Color::new(0.12, 0.13, 0.15);
    m.add_limb_flat(
        Vector3::new(0.0, -0.9, -22.0),
        Vector3::new(0.0, -0.8, 19.0),
        3.3,
        2.6,
        0.86,
        9,
        hull,
        true,
    );
    m.add_limb_flat(
        Vector3::new(0.0, -0.8, 19.0),
        Vector3::new(0.0, -0.2, 23.5),
        2.6,
        0.5,
        0.86,
        8,
        hull,
        true,
    );
    m.add_box(
        Vector3::new(-3.2, 0.5, -21.5),
        Vector3::new(3.2, 0.7, 22.0),
        deck,
        Uv::Unit,
    );
    // The boxes: stacked in bays, in a scatter of liveries, and not all the
    // same height — an even deck of them reads as one long block.
    const BOX: [u32; 6] = [0xb0442c, 0x2f6bb5, 0x3f8f57, 0xd8a02a, 0x8a8f95, 0x7a4a8a];
    let mut h = 0u32;
    let mut rnd = || {
        h = h.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (h >> 16) as usize
    };
    for bay in 0..11 {
        let z = -19.5 + bay as f32 * 3.7;
        let tiers = 1 + rnd() % 3;
        for t in 0..tiers {
            for lane in 0..3 {
                if rnd() % 10 == 0 {
                    continue;
                }
                let x = -2.2 + lane as f32 * 2.2;
                m.add_box(
                    Vector3::new(x - 1.0, 0.7 + t as f32 * 1.3, z - 1.6),
                    Vector3::new(x + 1.0, 1.95 + t as f32 * 1.3, z + 1.6),
                    Color::from_hex(BOX[rnd() % BOX.len()]),
                    Uv::Unit,
                );
            }
        }
    }
    // Island and funnel, right aft.
    m.add_box(
        Vector3::new(-2.6, 0.7, -21.0),
        Vector3::new(2.6, 6.4, -16.5),
        house,
        Uv::Unit,
    );
    m.add_box(
        Vector3::new(-2.65, 3.4, -21.05),
        Vector3::new(2.65, 4.4, -16.45),
        dark,
        Uv::Unit,
    );
    m.add_box(
        Vector3::new(-1.3, 6.4, -20.2),
        Vector3::new(1.3, 9.6, -17.6),
        Color::new(0.55, 0.14, 0.12),
        Uv::Unit,
    );
    m.build()
}

/// A yacht: low hull, cabin, and a sail that is most of what you see.
pub(crate) fn yacht_geometry() -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let hull = Color::WHITE;
    let deck = Color::new(0.72, 0.62, 0.44);
    let sail = Color::new(0.94, 0.94, 0.92);
    m.add_limb_flat(
        Vector3::new(0.0, -0.35, -5.0),
        Vector3::new(0.0, -0.25, 4.2),
        1.35,
        0.9,
        0.72,
        8,
        hull,
        true,
    );
    m.add_limb_flat(
        Vector3::new(0.0, -0.25, 4.2),
        Vector3::new(0.0, 0.1, 5.8),
        0.9,
        0.15,
        0.72,
        7,
        hull,
        true,
    );
    m.add_box(
        Vector3::new(-1.2, 0.18, -4.8),
        Vector3::new(1.2, 0.3, 5.2),
        deck,
        Uv::Unit,
    );
    m.add_box(
        Vector3::new(-0.85, 0.3, -1.6),
        Vector3::new(0.85, 1.15, 1.8),
        hull,
        Uv::Unit,
    );
    // Mast, boom, and a mainsail as two triangles so it is not invisible edge
    // on — the sail is the entire silhouette of a yacht.
    m.add_cylinder(
        Vector3::new(0.0, 0.3, 0.4),
        0.09,
        0.05,
        8.4,
        6,
        sail,
        false,
        Uv::Unit,
    );
    for s in [-1.0f32, 1.0] {
        m.tri(
            [[0.0, 8.4, 0.4], [s * 0.05, 1.0, -3.8], [s * 0.05, 1.0, 0.6]],
            [s, 0.0, 0.0],
            [[0.5, 0.5]; 3],
            [sail; 3],
        );
        m.tri(
            [[0.0, 8.4, 0.4], [s * 0.05, 2.2, 1.0], [s * 0.05, 1.0, 3.6]],
            [s, 0.0, 0.0],
            [[0.5, 0.5]; 3],
            [sail; 3],
        );
    }
    m.build()
}

pub(crate) fn boat_geometry() -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let hull = Color::WHITE;
    let deck = Color::new(0.42, 0.40, 0.36);
    let house = Color::new(0.86, 0.86, 0.84);
    m.add_box(
        Vector3::new(-2.3, -0.9, -11.0),
        Vector3::new(2.3, 0.5, 9.5),
        hull,
        Uv::Unit,
    );
    // Raked bow: a shorter, narrower block ahead of the hull.
    m.add_box(
        Vector3::new(-1.6, -0.5, 9.5),
        Vector3::new(1.6, 0.5, 11.4),
        hull,
        Uv::Unit,
    );
    m.add_box(
        Vector3::new(-2.1, 0.5, -10.6),
        Vector3::new(2.1, 0.66, 9.2),
        deck,
        Uv::Unit,
    );
    m.add_box(
        Vector3::new(-1.5, 0.66, -9.6),
        Vector3::new(1.5, 2.9, -5.6),
        house,
        Uv::Unit,
    );
    m.add_cylinder(
        Vector3::new(0.0, 2.9, -7.6),
        0.34,
        0.30,
        1.9,
        8,
        Color::new(0.20, 0.20, 0.22),
        true,
        Uv::Unit,
    );
    m.build()
}

/// Hand each contiguous colour run its own instanced mesh, once per walk
/// pose, and give the whole population one shared impostor for the far level
/// of detail.
///
/// Cost in draw calls is `colours x poses + 1`, and the impostor is the `+1`
/// — which is the point: the alternative, a simplified mesh per colour per
/// level, would cost more than it saves in a renderer bound by draw calls.
#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_fleet(
    scene: &mut Scene,
    name: &str,
    poses: &[Arc<BufferGeometry>],
    far: Option<Arc<BufferGeometry>>,
    colors: &[u32],
    roughness: f32,
    metalness: f32,
    clearcoat: f32,
    lod: f32,
    gait: Gait,
    sorted: Vec<(usize, Mover)>,
    triangles: &mut usize,
    draws: &mut usize,
) -> Fleet {
    let near_tris = poses
        .first()
        .and_then(|g| g.index.as_ref())
        .map(|i| i.len() / 3)
        .unwrap_or(0);
    let count = sorted.len();
    // One mesh per pose, not per pose and colour.
    //
    // This used to be a group per (colour, pose), which is a draw call per
    // colour: nine vehicle types with their palettes cost thirty-seven draws
    // of bodywork on their own, a quarter of the whole city. The paint is a
    // per-instance tint now, so a fleet costs what its animation costs and
    // nothing for its variety.
    let mut groups = Vec::new();
    for (pose, geometry) in poses.iter().enumerate() {
        let mut im = InstancedMesh::new(
            BufferGeometry::new(),
            // White base: the colour arrives per instance. Car paint is still
            // a coloured base under a clear lacquer, which is what the
            // clearcoat lobe is for; pedestrians pass 0 and get none.
            Material::Physical(
                PhysicalMaterial::new(Color::WHITE)
                    .with_roughness(roughness)
                    .with_metalness(metalness)
                    .with_clearcoat(clearcoat, 0.06),
            ),
            count,
        );
        im.geometry = geometry.clone();
        let mut obj = Object3D::instanced_mesh(im);
        obj.name = format!("{name}-{pose}");
        groups.push(Group {
            mesh: scene.add(obj),
            start: 0,
            count,
            pose,
        });
        *draws += 1;
    }
    // Counted by the caller as a ceiling, not added to the static total: with
    // instance packing what a fleet actually draws depends on where the camera
    // is standing.
    *triangles += near_tris * count * poses.len();

    let paints: Vec<Color> = sorted
        .iter()
        .map(|(c, _)| Color::from_hex(colors[*c % colors.len()]))
        .collect();
    let movers: Vec<Mover> = sorted.into_iter().map(|(_, m)| m).collect();
    let far = far.map(|geometry| {
        let tris = geometry.index.as_ref().map(|i| i.len() / 3).unwrap_or(0);
        let mut im = InstancedMesh::new(
            BufferGeometry::new(),
            Material::Standard(
                StandardMaterial::new(Color::WHITE)
                    .with_roughness(roughness)
                    .with_metalness(metalness * 0.5),
            ),
            movers.len(),
        );
        im.geometry = geometry;
        let mut obj = Object3D::instanced_mesh(im);
        obj.name = format!("{name}-far");
        *triangles += tris * movers.len();
        *draws += 1;
        scene.add(obj)
    });

    Fleet {
        movers,
        paints,
        groups,
        far,
        poses: poses.len(),
        variants: 1,
        lod,
        gait,
    }
}

/// Inset a run of road by half a vehicle's length.
///
/// `s_lo`/`s_hi` bound the mover's CENTRE, so a vehicle stopped at the end of
/// a run hangs half its body past it. On a street that dead-ends at the river
/// that is an eleven-metre bus with five and a half metres over the water —
/// invisible for a car, obvious for a bus, which is how it was found.
pub(crate) fn fit_run(lo: f32, hi: f32, length: f32) -> (f32, f32) {
    let half = length * 0.5;
    if hi - lo <= length * 1.2 {
        // Too short to inset without inverting; centre it instead.
        let mid = (lo + hi) * 0.5;
        return (mid, mid);
    }
    (lo + half, hi - half)
}

/// Pick which way to go on the road being turned onto, given how much of it
/// lies each way from the entry point.
///
/// Without this a vehicle can turn onto a lane it is already at the end of,
/// drive off it within a frame or two, and wrap — which kills the corner arc
/// half way through and snaps the heading by a full ninety degrees. The corner
/// needs room to be swept through before it is worth committing to.
fn pick_turn_dir(preferred: f32, start: f32, lo: f32, hi: f32, leg: f32) -> Option<f32> {
    let room = |d: f32| if d > 0.0 { hi - start } else { start - lo };
    [preferred, -preferred].into_iter().find(|&d| room(d) >= leg * 1.4)
}

/// Turn a vehicle onto the crossing road, if it has just passed the middle of
/// a junction and the dice say so.
///
/// Traffic that only ever runs straight and wraps at the end of the road is
/// what made the collision work easy — nothing ever changed lane, so "which
/// junction am I approaching" was a scalar. This is the thing that makes it a
/// network. The choice comes from the mover's own PRNG state so the whole
/// simulation stays deterministic without a shared `Rng`.
pub(crate) fn maybe_turn(m: &mut Mover, before: f32, junctions: &[Junction]) {
    for j in junctions {
        let (center, cross_fixed, cross_half) = if m.along_x {
            (j.cx, j.cz, j.hz)
        } else {
            (j.cz, j.cx, j.hx)
        };
        // Is this junction on the road we are actually driving down?
        if (m.fixed - cross_fixed).abs() > cross_half {
            continue;
        }
        // Half-width of the crossing road measured along the axis we are
        // driving down, less the padding the occupancy box carries.
        let half = if m.along_x {
            (j.hx - 1.0).max(1.0)
        } else {
            (j.hz - 1.0).max(1.0)
        };
        // Commit a corner radius early rather than at the middle, so there is
        // road left to sweep the arc through instead of pivoting on the spot.
        // The lead includes the lane offset, so the entry leg still measures at
        // least `TURN_R` whichever way the dice send us.
        let commit = center - m.dir * (TURN_R + half * 0.44);
        if (before - commit) * (m.s - commit) > 0.0 {
            continue;
        }

        m.seed = m.seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let roll = (m.seed >> 13) & 0xff;
        // About a fifth of vehicles turn at any given junction. A third was
        // measured: it holds a stable steady state but a congested one, ~100
        // of 234 stopped against ~60 here.
        if roll >= 52 {
            return;
        }
        let preferred = if roll & 1 == 0 { 1.0 } else { -1.0 };

        let (lo, hi) = if m.along_x {
            fit_run(j.z_lo, j.z_hi, m.length)
        } else {
            fit_run(j.x_lo, j.x_hi, m.length)
        };
        // The corner lands where the two centrelines cross, which is at our own
        // `fixed` along the new road. If that is outside the new run there is no
        // corner to drive -- clamping it instead would stretch the exit leg away
        // from the entry leg and put the heading jump straight back.
        if m.fixed < lo || m.fixed > hi {
            return;
        }
        let Some(new_dir) = pick_turn_dir(preferred, m.fixed, lo, hi, TURN_R + 2.0 * half * 0.44)
        else {
            return;
        };

        // Where we are now, and where the two lane centrelines cross.
        let (p0x, p0z) = if m.along_x {
            (m.s, m.fixed)
        } else {
            (m.fixed, m.s)
        };
        // The new road runs along the other axis, so the side is computed for
        // that axis, not the one being left.
        let new_fixed = (if m.along_x { j.cx } else { j.cz })
            + lane_side(!m.along_x, new_dir) * half * 0.44;
        if m.along_x {
            m.turn = [p0x, p0z, new_fixed, p0z];
            m.signalled = j.avenue_z;
            m.lane = ((j.road_z as u16) << 2) | u16::from(new_dir > 0.0);
        } else {
            m.turn = [p0x, p0z, p0x, new_fixed];
            m.signalled = j.avenue_x;
            m.lane = ((j.road_x as u16) << 2) | (1 << 1) | u16::from(new_dir > 0.0);
        }
        m.turn_s0 = m.fixed;
        m.s = m.fixed;
        m.fixed = new_fixed;
        m.s_lo = lo;
        m.s_hi = hi;
        m.along_x = !m.along_x;
        m.dir = new_dir;
        // Slow for the corner rather than pivoting at speed.
        m.speed = m.speed.min(m.cruise * 0.62);
        return;
    }
}
