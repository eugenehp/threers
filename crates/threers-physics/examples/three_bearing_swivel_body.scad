// A Three-Bearing Swivel Module, in both the arrangements its drive can take.
//
// Everything is here; nothing runs until `swivel_module(internal)` is called.
// `three_bearing_swivel.scad` calls it with the pinion outside its ring gear and
// `three_bearing_swivel_internal.scad` with the pinion inside it, and those two
// files are three lines each.
//
// ---------------------------------------------------------------------------
// The five modules below are shims. threers replaces them with the real thing
// when it reads this file as a mechanism; OpenSCAD, which has never heard of
// them, uses these and renders an ordinary static assembly. Without them
// OpenSCAD drops every part() *along with its children* and shows nothing.
module part(name, fixed = false, density = undef, collider = undef) { children(); }
module hinge(name, parts, at, axis, range, bearing, friction) { }
module weld(name, parts, at, axis) { }
module gear(name, parts, at, axis, ratio, carrier) { }
module drive(name, to, over, torque, speed) { }

// ---- the duct, in two builds ------------------------------------------------
//
// The external arrangement is a single-wall duct with the drive bolted to the
// outside of it. The internal one is double-walled — a liner carrying the gas
// and a casing carrying the loads, with a cooling annulus between them — and the
// whole drive lives in that annulus. Same gas path either way, so the two are
// comparable on the thing an exhaust duct is for.
R_GAS  = 0.484;            // gas-path bore, the same in both
WALL   = 0.016;            // structural wall; stands in for wall and stiffening
LINER  = 0.012;            // and the liner's own, thinner because it carries no load
BGAP   = 0.001;            // running clearance at a bearing face, so the two
                           // halves of a joint do not meet at exactly zero
LGAP   = 0.004;            // axial gap left at each end of a liner, so adjacent
                           // liners do not rub as the bearing between them turns.
                           // A real one bridges that gap with a sliding seal fed
                           // by the same cooling air the annulus carries -- which
                           // is what "the liner that directs bypass cooling air
                           // through the swivel joints" is describing. Without it
                           // the two faces meet at zero clearance and the
                           // clearance check says so.

// External build: one wall, and a race on the outside of it.
R_OUT  = R_GAS + WALL;     // 0.500
R_RACE = 0.548;            // the bearing race, a boss on the end of the duct

// Internal build: liner, annulus, casing, and a race on the casing.
R_LIN  = R_GAS + LINER;    // 0.496, outside of the liner
R_BORE = 0.740;            // casing bore — the far wall of the annulus
R_CASE = R_BORE + WALL;    // 0.756
R_SEAT = 0.750;            // race bore, where the ring gear seats
R_RIM  = 0.780;            // and the outside of the race
// Those sit 125 mm further out than the annulus strictly needs, and the reason
// is that a 23 mm design margin does not survive a mechanism whose parts swing.
// Sized to just clear, the liner's plain middle came within 0.3 mm of the pinion
// at 45 degrees of deflection — not at the joint, where everything is coaxial
// with the bearing and invariant under its rotation, but a third of a metre
// downstream where it is not. The clearance check found that; the arithmetic
// that sized the annulus never would have.

BOSS   = 0.090;            // how far a race stands along its bearing axis
LBOSS  = 0.200;            // and how far a *liner* stays coaxial with its bearing.
                           // Longer than the race on purpose: only the coaxial
                           // part of a liner is invariant under its own bearing's
                           // rotation, and the pinion for that bearing sits in the
                           // annulus right there.
                           // Longer than the race on purpose: the pinion for that
                           // bearing sits inside the annulus at this end, and only
                           // the coaxial part of the liner is invariant under the
                           // bearing's own rotation. Where the liner starts to
                           // blend back to the duct axis it sweeps, and at 0.09 it
                           // swept to within 0.3 mm of the pinion.
R_FLNG = 0.580;            // the flange this bolts to the engine on
FLNG_W = 0.050;
R_EXIT = 0.410;            // outer radius at the throat

// ---- stations, measured along the engine centreline -------------------------
X0 = -0.600;               // the engine flange this bolts to
X1 =  0.000;               // bearing 1 -- the roll bearing, square to the duct
X2 =  0.560;               // bearing 2 -- canted
X3 =  1.280;               // bearing 3 -- canted the other way
X4 =  1.680;               // exit plane
// Those spacings are not free either. A drive unit reaches 0.42 m back along its
// own bearing axis, and the segment in front of it has to be long enough that it
// does not arrive on top of the previous joint's pinion -- at X2 = 0.42 the two
// came within 1.2 mm of each other. And `swivel_b` has a canted joint at both
// ends, so it has to be longer than the two dips put together: 2 x 0.278 m of
// casing, before any plain duct at all.

// ---- cant angles ------------------------------------------------------------
// The angle each bearing's axis makes with the duct it is cut through, and the
// one thing about this mechanism that is *not* a design choice here.
//
// Lockheed describe three duct segments "cut on an angle and joined by two
// airtight circular bearings", with "the forward and aft segments maintain[ing]
// alignment with each other" while "the center segment rotates 180 degrees
// relative to them" -- and a third bearing "aft of the turbine stage" through
// which the nozzle "provides yaw control".
//
// So the first bearing is square to the duct. It is a roll bearing: it does
// nothing to the jet on its own, and turns the whole folded assembly about the
// engine axis, which is one for one with the jet's azimuth. That is the yaw
// control, and it is also what makes the deployment schedule closed form.
//
// The other two are a pair of mitre joints. Read as mitres, the exhaust leaves
// at 2 * (C1 - C2 + C3), so turning the centre segment half a turn against both
// of its neighbours folds the duct by 2 * (23.75 + 23.75) -- and four times
// 23.75 is 95 exactly, which is the published figure.
//
// An earlier draft of this file canted all three at 11.875 / -23.75 / 11.875.
// That also reaches 95 degrees, and by a completely different mechanism: no
// bearing sets azimuth, so there is no closed-form schedule, the jet leaves the
// vertical plane by 68 degrees on any fixed ratio, and the stowed pose has no
// first-order pitch authority at all. All three of those are properties of that
// arrangement and not of a 3BSM.
C1 =   0.000;
C2 = -23.750;
C3 =  23.750;

// ---- the drive train, in two arrangements -----------------------------------
//
// "External motors drive geared teeth in these segments to rotate them" is how
// Lockheed describe it, and that is what is below: each joint's ring gear sits
// on the race turned into the end of the segment it drives, and the pinion that
// meshes with it rides on the segment *upstream* of the joint — the only place
// it can ride, because that is the part the ring gear turns against.
//
// Which side of the ring gear the pinion runs on is a real choice, and it decides
// where the whole drive has to live.
//
//   external   Teeth on the outside of the ring, pinion beside it, centre
//              distance R_ring + R_pinion, and the pair counter-rotates. Nothing
//              needs a second wall — but the pinion and its motor are outside the
//              duct, and they are what sets the installed envelope.
//
//   internal   Teeth on the inside of a ring seated in the race bore, pinion
//              *inside* the annulus between the liner and the casing along with
//              its motor, centre distance R_ring - R_pinion, and the pair turns
//              the same way. Nothing projects at all; the envelope is the race.
//
// The internal build cannot use the same 8:1. Its pinion has to clear the liner
// from a centre distance of R_ring - R_pinion, which needs R_pinion small, which
// needs the tooth count high: 16:1 rather than 8:1, and half the gearbox behind
// it. The mesh ratio is not a free choice there — the annulus picks it.
Z_PIN  = 12;                     // teeth on a pinion, both builds
FACE   = 0.060;                  // gear face width
GEAR_S = 0.015;                  // how far along its race a ring gear sits

function teeth_ring(internal) = internal ? 192 : 96;
function gmod(internal)       = internal ? 0.0075 : 0.0125;
function rgear(internal)      = gmod(internal) * teeth_ring(internal) / 2;   // 0.600 both
function rpin(internal)       = gmod(internal) * Z_PIN / 2;
function addm(internal)       = gmod(internal);
function dedm(internal)       = 1.25 * gmod(internal);
function centres(internal)    = internal ? rgear(internal) - rpin(internal)
                                         : rgear(internal) + rpin(internal);
// Turns of the pinion per turn of the bearing. Two external gears counter-rotate
// and an internal pair does not, which is the sign.
function mesh(internal)       = internal ? teeth_ring(internal) / Z_PIN
                                         : -teeth_ring(internal) / Z_PIN;
// The plain side of a ring gear: its bore on the race for an external one, its
// seat in the race bore for an internal one.
function ring_back(internal)  = internal ? R_SEAT : R_RACE;
// Motor-to-pinion reduction inside each drive unit: whatever brings the whole
// train to 200:1, which is the ratio that puts a fueldraulic motor's useful
// speed against the bearing's. The internal build's mesh already carries twice
// as much, so its gearbox carries half — 12.5:1 against 25:1, and the motor
// turns at the same rpm in both.
function boxratio(internal)   = 200 / abs(mesh(internal));
// The drive unit's mounting flange, sized to land on the part it bolts to rather
// than near it: the casing bore inside the annulus, the duct outside it.
function flange(internal)     = internal ? R_BORE - centres(internal)
                                         : centres(internal) - R_OUT;

RHO       = 4430;          // Ti-6Al-4V for the cold structure
RHO_STEEL = 7850;          // gears are steel
SEGS      = 72;            // facets round a plain duct

// ---- how the parts are built ------------------------------------------------
//
// One polyhedron each, lofted along a list of rings, and no boolean anywhere.
//
// A ring is [radius, cx, cy, cz, cant, mode]: a circle of that radius about the
// point [cx, cy, cz], lying in the plane whose normal is canted by `cant` from
// +X toward -Z. Walk the list along the outside of the part and back along the
// bore, and the two radial steps at the ends become flat annular end faces.
//
// Which is the whole trick. A bearing plane through this duct is an oblique cut,
// and cutting it with a half-space costs an exact boolean that fragments a
// 900-triangle tube into twenty thousand. Ending the loft *on* that plane costs
// nothing and is exact: the end face is a circle about the bearing axis, which
// is what a machined bearing race is. The race either side of a joint is then
// the same circle, and the ring gear that bolts to one of them has the same bore
// -- so nothing here floats beside anything.
//
// `mode` is 0 for a plain ring, +1 for teeth pointing out and -1 for teeth
// pointing in. A toothed ring ignores its own radius and takes the pitch radius
// passed to the loft, modulated four samples to the tooth. Square teeth rather
// than involute ones: the mesh here is a *constraint*, not a contact, so a
// tooth's job is to carry the right mass and to look like a tooth. Cutting them
// this way costs exactly what a plain ring costs.
function ring(r, cx, cy, cz, cant, mode) = [r, cx, cy, cz, cant, mode];

// A point on bearing `x`'s axis, `d` along it, canted by `c`.
function on_axis(x, d, c) = [x + d * cos(c), 0, -d * sin(c)];

module loft(rings, segs, pitch = 0, add = 0, ded = 0) {
    n = len(rings);
    polyhedron(
        points = [ for (i = [0 : segs - 1], j = [0 : n - 1])
                     let (a   = 360 * i / segs,
                          tip = (i % 4 == 1 || i % 4 == 2),
                          m   = rings[j][5],
                          r   = m == 0 ? rings[j][0]
                              : m > 0  ? (tip ? pitch + add : pitch - ded)
                                       : (tip ? pitch - add : pitch + ded),
                          t   = rings[j][4])
                     [ rings[j][1] + r * cos(a) * sin(t),
                       rings[j][2] + r * sin(a),
                       rings[j][3] + r * cos(a) * cos(t) ] ],
        faces  = [ for (i = [0 : segs - 1], j = [0 : n - 1])
                     [ i * n + j,
                       ((i + 1) % segs) * n + j,
                       ((i + 1) % segs) * n + (j + 1) % n,
                       i * n + (j + 1) % n ] ]);
}

// ---- the shapes ------------------------------------------------------------
//
// One module per kind of piece, taking the radii that tell the two builds apart.
// `rin`/`rout` are the wall through the middle of a segment and `rseat`/`rrim`
// the race at each end — for the external build those are the bore and the race
// boss, and for the internal one the casing bore and the race the ring gear
// seats in.

// A length of duct between two bearing planes. The ends *are* the planes: a ring
// perpendicular to the bearing axis and centred on it, which is what a machined
// race is, so the two halves of a joint meet face to face and nothing has to be
// cut afterwards.
// `da` and `db` inset the two ends along their own bearing axes, which is how a
// liner is kept off the one in front of it and how a bearing face is given its
// running clearance.
//
// `ba`/`bb` are where the middle of the segment becomes a plain tube about the
// *duct* axis, and they are not free. A bearing plane is oblique, so a ring of
// radius R about the duct axis at station s reaches `(s - x)cos(c) + R|sin(c)|`
// past it — and a segment whose middle crosses its own joint plane will foul the
// segment on the other side of that joint however the bearing turns. At the
// middle bearing's 23.75 degrees a 0.631 m casing dips 278 mm, so the plain part
// of `swivel_a` has to live inside 0.133..0.282 and not the 0.20..0.37 that
// looks reasonable. Ending the loft *on* the plane is what makes this a
// constraint: the old half-space cut enforced it for free and this does not.
module segment(xa, ca, xb, cb, ba, bb, rin, rout, rseat, rrim, da = 0, db = 0, boss = BOSS) {
    loft([ring(rrim,  on_axis(xa, da, ca)[0], 0, on_axis(xa, da, ca)[2], ca, 0),
          ring(rrim,  on_axis(xa, da + boss, ca)[0], 0, on_axis(xa, da + boss, ca)[2], ca, 0),
          ring(rout,  ba, 0, 0, 0, 0),
          ring(rout,  bb, 0, 0, 0, 0),
          ring(rrim,  on_axis(xb, -db - boss, cb)[0], 0, on_axis(xb, -db - boss, cb)[2], cb, 0),
          ring(rrim,  on_axis(xb, -db, cb)[0], 0, on_axis(xb, -db, cb)[2], cb, 0),
          ring(rseat, on_axis(xb, -db, cb)[0], 0, on_axis(xb, -db, cb)[2], cb, 0),
          ring(rseat, on_axis(xb, -db - boss, cb)[0], 0, on_axis(xb, -db - boss, cb)[2], cb, 0),
          ring(rin,   bb, 0, 0, 0, 0),
          ring(rin,   ba, 0, 0, 0, 0),
          ring(rseat, on_axis(xa, da + boss, ca)[0], 0, on_axis(xa, da + boss, ca)[2], ca, 0),
          ring(rseat, on_axis(xa, da, ca)[0], 0, on_axis(xa, da, ca)[2], ca, 0)], SEGS);
}

// The fixed duct: an engine flange at one end and a bearing race at the other.
module inlet(rin, rout, rseat, rrim, flanged, db = 0, boss = BOSS) {
    loft(concat(flanged ? [ring(R_FLNG, X0, 0, 0, 0, 0),
                           ring(R_FLNG, X0 + FLNG_W, 0, 0, 0, 0),
                           ring(rout,   X0 + FLNG_W, 0, 0, 0, 0)]
                        : [ring(rout,   X0, 0, 0, 0, 0)],
                [ring(rout,  -0.22, 0, 0, 0, 0),
                 ring(rrim,  on_axis(X1, -db - boss, C1)[0], 0, on_axis(X1, -db - boss, C1)[2], C1, 0),
                 ring(rrim,  on_axis(X1, -db, C1)[0], 0, on_axis(X1, -db, C1)[2], C1, 0),
                 ring(rseat, on_axis(X1, -db, C1)[0], 0, on_axis(X1, -db, C1)[2], C1, 0),
                 ring(rseat, on_axis(X1, -db - boss, C1)[0], 0, on_axis(X1, -db - boss, C1)[2], C1, 0),
                 ring(rin,   -0.22, 0, 0, 0, 0),
                 ring(rin,   X0, 0, 0, 0, 0)]), SEGS);
}

// And the last one, which converges to the throat instead of ending on a plane.
// It is lighter than the other two and its centre of mass sits closer to its
// bearing, which is the moment the third drive has to hold.
module tail(rin, rseat, rrim, exit_out, exit_in, da = 0, boss = BOSS) {
    loft([ring(rrim,     on_axis(X3, da, C3)[0], 0, on_axis(X3, da, C3)[2], C3, 0),
          ring(rrim,     on_axis(X3, da + boss, C3)[0], 0, on_axis(X3, da + boss, C3)[2], C3, 0),
          ring(exit_out, X4, 0, 0, 0, 0),
          ring(exit_in,  X4, 0, 0, 0, 0),
          ring(rin,      on_axis(X3, da + boss, C3)[0], 0, on_axis(X3, da + boss, C3)[2], C3, 0),
          ring(rseat,    on_axis(X3, da, C3)[0], 0, on_axis(X3, da, C3)[2], C3, 0)], SEGS);
}

// ---- the drive-train pieces -------------------------------------------------

// A ring gear on the race at bearing `x`. Bored to the race when its teeth face
// out, seated in the race bore when they face in — an internal ring hangs inside
// the casing with the pinion under it.
module ring_gear(x, cant, internal) {
    a = addm(internal);
    d = dedm(internal);
    p = rgear(internal);
    b = ring_back(internal);
    // Inner radius first either way, so the loop keeps its handedness: for an
    // external gear that is the bore and for an internal one it is the teeth.
    loft(internal
         ? [ring(0, on_axis(x, GEAR_S, cant)[0], 0, on_axis(x, GEAR_S, cant)[2], cant, -1),
            ring(b, on_axis(x, GEAR_S, cant)[0], 0, on_axis(x, GEAR_S, cant)[2], cant, 0),
            ring(b, on_axis(x, GEAR_S + FACE, cant)[0], 0,
                    on_axis(x, GEAR_S + FACE, cant)[2], cant, 0),
            ring(0, on_axis(x, GEAR_S + FACE, cant)[0], 0,
                    on_axis(x, GEAR_S + FACE, cant)[2], cant, -1)]
         : [ring(b, on_axis(x, GEAR_S, cant)[0], 0, on_axis(x, GEAR_S, cant)[2], cant, 0),
            ring(0, on_axis(x, GEAR_S, cant)[0], 0, on_axis(x, GEAR_S, cant)[2], cant, 1),
            ring(0, on_axis(x, GEAR_S + FACE, cant)[0], 0,
                    on_axis(x, GEAR_S + FACE, cant)[2], cant, 1),
            ring(b, on_axis(x, GEAR_S + FACE, cant)[0], 0,
                    on_axis(x, GEAR_S + FACE, cant)[2], cant, 0)],
         teeth_ring(internal) * 4, p, a, d);
}

// Its pinion, on an axis parallel to the bearing and offset by the centre
// distance. Teeth face out whichever side of the ring it runs on — that is what
// makes one pair counter-rotate and the other not.
module pinion(x, cant, internal) {
    c = centres(internal);
    loft([ring(0.024, on_axis(x, GEAR_S, cant)[0], c, on_axis(x, GEAR_S, cant)[2], cant, 0),
          ring(0, on_axis(x, GEAR_S, cant)[0], c, on_axis(x, GEAR_S, cant)[2], cant, 1),
          ring(0, on_axis(x, GEAR_S + FACE, cant)[0], c,
                  on_axis(x, GEAR_S + FACE, cant)[2], cant, 1),
          ring(0.024, on_axis(x, GEAR_S + FACE, cant)[0], c,
                  on_axis(x, GEAR_S + FACE, cant)[2], cant, 0)],
         Z_PIN * 4, rpin(internal), addm(internal), dedm(internal));
}

// Motor and gearbox, coaxial with the pinion and reaching back over the joint to
// the part that carries them — the outside of the duct in one build, and the
// cooling annulus between liner and casing in the other. The mounting flange is
// sized to land on that part rather than near it. What is inside the case is a
// ratio, not a shape.
module drive_unit(x, cant, internal) {
    c  = centres(internal);
    f  = flange(internal);
    r1 = internal ? 0.050 : 0.100;
    r2 = internal ? 0.040 : 0.075;
    loft([ring(0.024, on_axis(x, -0.02, cant)[0], c, on_axis(x, -0.02, cant)[2], cant, 0),
          ring(f,     on_axis(x, -0.02, cant)[0], c, on_axis(x, -0.02, cant)[2], cant, 0),
          ring(f,     on_axis(x, -0.09, cant)[0], c, on_axis(x, -0.09, cant)[2], cant, 0),
          ring(r1,    on_axis(x, -0.12, cant)[0], c, on_axis(x, -0.12, cant)[2], cant, 0),
          ring(r1,    on_axis(x, -0.24, cant)[0], c, on_axis(x, -0.24, cant)[2], cant, 0),
          ring(r2,    on_axis(x, -0.24, cant)[0], c, on_axis(x, -0.24, cant)[2], cant, 0),
          ring(r2,    on_axis(x, -0.42, cant)[0], c, on_axis(x, -0.42, cant)[2], cant, 0),
          ring(0.024, on_axis(x, -0.42, cant)[0], c, on_axis(x, -0.42, cant)[2], cant, 0)],
         32);
}

// ---- the whole thing --------------------------------------------------------
//
// Every part is drawn where it sits, in one coordinate system. That is what lets
// the mates be written once and mean the same thing to the geometry, the
// colliders and the kinematics.
//
// `collider = "mesh"` throughout, and it is not decoration. A duct is a hole
// with metal round it, and the fits that approximate it fill the hole in: the
// default convex hull makes a segment weigh 8 tonnes and a convex decomposition
// 738 kg, where the metal in it weighs 211. Every bearing torque downstream of
// that is wrong by the same factor.
//
// A triangle mesh is normally the wrong collider for something that moves,
// because it is a hollow surface. It is the right one here: nothing in this
// mechanism touches anything. The parts are held by bearings, and each bearing's
// two halves lie on opposite sides of a plane that the bearing's own rotation
// maps to itself, so they cannot reach each other at any angle. The sweep in the
// example is what checks that rather than assuming it.
module swivel_module(internal) {

// Which wall this build has. `RI`/`RO` is the wall through the middle of a
// segment and `RS`/`RR` the race at each end — bore and boss for the external
// build, casing bore and ring-gear seat for the internal one.
RI = internal ? R_BORE : R_GAS;
RO = internal ? R_CASE : R_OUT;
RS = internal ? R_SEAT : R_GAS;
RR = internal ? R_RIM  : R_RACE;

part("engine_duct", fixed = true, density = RHO, collider = "mesh")
    color("dimgray") inlet(RI, RO, RS, RR, true, BGAP);

part("swivel_a", density = RHO, collider = "mesh")
    color("steelblue") segment(X1, C1, X2, C2, 0.03, 0.20, RI, RO, RS, RR, BGAP, BGAP);

part("swivel_b", density = RHO, collider = "mesh")
    color("lightsteelblue") segment(X2, C2, X3, C3, 0.86, 0.98, RI, RO, RS, RR, BGAP, BGAP);

part("nozzle", density = RHO, collider = "mesh")
    color("indianred") tail(RI, RS, RR, R_EXIT, R_EXIT - WALL, BGAP);

// The liner, in the build that has one: the gas path, hung inside the casing on
// the same bearings, with the cooling annulus between the two — and the whole
// drive train living in that annulus. Nothing projects from this machine at all.
if (internal) {
    part("liner_0", fixed = true, density = RHO, collider = "mesh")
        color("lightgray") inlet(R_GAS, R_LIN, R_GAS, R_LIN, false, LGAP, LBOSS);
    part("liner_a", density = RHO, collider = "mesh")
        color("lightgray") segment(X1, C1, X2, C2, 0.03, 0.20, R_GAS, R_LIN, R_GAS, R_LIN, LGAP, LGAP, LBOSS);
    part("liner_b", density = RHO, collider = "mesh")
        color("lightgray") segment(X2, C2, X3, C3, 0.86, 0.98, R_GAS, R_LIN, R_GAS, R_LIN, LGAP, LGAP, LBOSS);
    part("liner_n", density = RHO, collider = "mesh")
        color("lightgray") tail(R_GAS, R_GAS, R_LIN, R_EXIT - 0.030, R_EXIT - 0.030 - LINER, LGAP, LBOSS);
}

part("ring1", density = RHO_STEEL, collider = "mesh")
    color("darkgoldenrod") ring_gear(X1, C1, internal);
part("ring2", density = RHO_STEEL, collider = "mesh")
    color("darkgoldenrod") ring_gear(X2, C2, internal);
part("ring3", density = RHO_STEEL, collider = "mesh")
    color("darkgoldenrod") ring_gear(X3, C3, internal);

part("pinion1", density = RHO_STEEL, collider = "mesh")
    color("goldenrod") pinion(X1, C1, internal);
part("pinion2", density = RHO_STEEL, collider = "mesh")
    color("goldenrod") pinion(X2, C2, internal);
part("pinion3", density = RHO_STEEL, collider = "mesh")
    color("goldenrod") pinion(X3, C3, internal);

part("drive1", density = 4000, collider = "mesh")
    color("olivedrab") drive_unit(X1, C1, internal);
part("drive2", density = 4000, collider = "mesh")
    color("olivedrab") drive_unit(X2, C2, internal);
part("drive3", density = 4000, collider = "mesh")
    color("olivedrab") drive_unit(X3, C3, internal);

// ---- the mechanism ----------------------------------------------------------
//
// One hinge per bearing, written in the coordinates the parts are drawn in. The
// axis is the normal of the cut plane, which is what a bearing turns about.
//
// `bearing = [mu, radius]` is friction that scales with what the joint carries,
// which is the difference between a bearing and an ideal hinge: 0.004 is a large
// crossed-roller slew ring, and the radius is the race the load runs on.

hinge("bearing1", parts = ["swivel_a", "engine_duct"], at = [X1, 0, 0],
      axis = [cos(C1), 0, -sin(C1)], range = [-180, 180], bearing = [0.004, RR]);
hinge("bearing2", parts = ["swivel_b", "swivel_a"], at = [X2, 0, 0],
      axis = [cos(C2), 0, -sin(C2)], range = [-180, 180], bearing = [0.004, RR]);
hinge("bearing3", parts = ["nozzle", "swivel_b"], at = [X3, 0, 0],
      axis = [cos(C3), 0, -sin(C3)], range = [-180, 180], bearing = [0.004, RR]);

// Each ring gear is bolted to its bearing race, so it is part of the segment it
// drives and moves as one with it. Each drive unit is bolted to the segment
// upstream of its joint, which is what it has to push against.
weld("ring1_bolts",  parts = ["ring1",  "swivel_a"],    at = [X1, 0, 0], axis = [1, 0, 0]);
weld("ring2_bolts",  parts = ["ring2",  "swivel_b"],    at = [X2, 0, 0], axis = [1, 0, 0]);
weld("ring3_bolts",  parts = ["ring3",  "nozzle"],      at = [X3, 0, 0], axis = [1, 0, 0]);
weld("drive1_bolts", parts = ["drive1", "engine_duct"], at = [X1, 0, 0], axis = [1, 0, 0]);
weld("drive2_bolts", parts = ["drive2", "swivel_a"],    at = [X2, 0, 0], axis = [1, 0, 0]);
weld("drive3_bolts", parts = ["drive3", "swivel_b"],    at = [X3, 0, 0], axis = [1, 0, 0]);
if (internal) {
    weld("liner_a_bolts", parts = ["liner_a", "swivel_a"], at = [X1, 0, 0], axis = [1, 0, 0]);
    weld("liner_b_bolts", parts = ["liner_b", "swivel_b"], at = [X2, 0, 0], axis = [1, 0, 0]);
    weld("liner_n_bolts", parts = ["liner_n", "nozzle"],   at = [X3, 0, 0], axis = [1, 0, 0]);
}

// The pinion shafts. Each turns in its own drive unit's bearings, about an axis
// parallel to the bearing it drives and offset by the centre distance. No range:
// a pinion makes eight turns for every one of the bearing's, so a limit written
// in its own coordinate would wrap long before the nozzle reached anything.
hinge("shaft1", parts = ["pinion1", "engine_duct"],
      at = [on_axis(X1, GEAR_S, C1)[0], centres(internal),
            on_axis(X1, GEAR_S, C1)[2]],
      axis = [cos(C1), 0, -sin(C1)], bearing = [0.002, rpin(internal)]);
hinge("shaft2", parts = ["pinion2", "swivel_a"],
      at = [on_axis(X2, GEAR_S, C2)[0], centres(internal),
            on_axis(X2, GEAR_S, C2)[2]],
      axis = [cos(C2), 0, -sin(C2)], bearing = [0.002, rpin(internal)]);
hinge("shaft3", parts = ["pinion3", "swivel_b"],
      at = [on_axis(X3, GEAR_S, C3)[0], centres(internal),
            on_axis(X3, GEAR_S, C3)[2]],
      axis = [cos(C3), 0, -sin(C3)], bearing = [0.002, rpin(internal)]);

// And the meshes. `ratio` is turns of the first part per turn of the second, so
// -8 is the pinion turning eight times for each turn of the bearing, the other
// way round because two external gears counter-rotate. An internal pair does not
// counter-rotate, so the same 8:1 comes out +8 there, and the controller simply
// reads the sign off the model.
//
// `carrier` is the part the mesh is mounted on. For the first bearing that is
// the engine duct, which does not move, and naming it changes nothing. For the
// second and third it is a segment that is itself being swung by the bearing in
// front -- and there the carrier is the difference between this mechanism and a
// different one. Measured against the world instead, turning bearing 1 alone
// would drag bearings 2 and 3 round with it through their own meshes.
gear("mesh1", parts = ["pinion1", "swivel_a"], carrier = "engine_duct",
     at = [X1, 0, 0], axis = [cos(C1), 0, -sin(C1)], ratio = mesh(internal));
gear("mesh2", parts = ["pinion2", "swivel_b"], carrier = "swivel_a",
     at = [X2, 0, 0], axis = [cos(C2), 0, -sin(C2)], ratio = mesh(internal));
gear("mesh3", parts = ["pinion3", "nozzle"], carrier = "swivel_b",
     at = [X3, 0, 0], axis = [cos(C3), 0, -sin(C3)], ratio = mesh(internal));

// Torque ceilings, in N.m, at the *pinion shaft* -- which is where the drive
// acts, and eight times less than what the bearing sees. Divide again by the
// gearbox and these are 48, 24 and 12 N.m at the motor, which is a small
// fueldraulic unit rather than the enormous direct-drive actuator the bearing
// figures would otherwise imply. That is the whole point of the train.
//
// These are sized by the *loop*, not by the load, and the difference is a factor
// of five. Holding the worst pose takes 211 N.m at pinion 1 and following the
// schedule takes 231 -- but a drive given 400 lets the jet wander 4.3 degrees
// out of the vertical plane through the low-authority part of the deployment,
// where the schedule turns 28 degrees of bearing for one degree of jet. At 1200
// it wanders 1.0, and above 1200 nothing improves. Size a servo on its static
// load and it will not look undersized; it will look slow, and here that reads
// as yaw.
//
// A ceiling is what makes that visible at all: ask for more than the drive can
// produce and it stalls, exactly as the real one would, rather than turning the
// nozzle through whatever is in the way.
//
// Nothing drives a bearing directly. The bearings are free joints and the only
// things holding this nozzle up are three pinions.
// Scaled by the mesh ratio, so both builds get the same ceiling referred to the
// bearing and the comparison is of the mechanism rather than of the actuators.
SCALE = 8 / abs(mesh(internal));
drive("shaft1", to = 0, over = 0.5, torque = 1200 * SCALE);
drive("shaft2", to = 0, over = 0.5, torque =  600 * SCALE);
drive("shaft3", to = 0, over = 0.5, torque =  300 * SCALE);

}
