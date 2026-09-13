//! An iridescent logo mark: black metal under a drifting neon field, lit so
//! that almost all of it reads as silhouette and only the rim carries colour.
//!
//!     cargo run --release --example neon_logo
//!
//! Two layers make the picture, and neither is a conventional BRDF. A simplex
//! noise field, sampled in object space and scaled so the whole mark sits inside
//! a fraction of one noise cell, supplies a slow four-colour wash. Over it sits
//! a ten-stop ramp indexed by the half-Lambert term against the single spot
//! light. Most of that ramp is opaque black, so the mark reads as a hole; four
//! thin stops near the lit end — white, violet, pink — carve the bright edge.
//! Move the light and the edge sweeps around the shape.
//!
//! Writes `out/neon_logo.png`. Knobs:
//!
//! * `NEON_CAMERA=hero|tilt|close` — one of the three framings below.
//! * `NEON_RES=<n>` — marching-squares grid for the swept solids.
//! * `NEON_MESH=<dir>` — load `body.obj` / `leaf.obj` from `<dir>` instead of
//!   sweeping the outlines.
//! * `NEON_ANIM_FPS=<n>` (or `NEON_ANIM=1` for 30) — render the 15-second
//!   timeline to `out/neon_logo_anim/`, plus an `.mp4` with
//!   `--features native-codec`.
//! * `NEON_NO_POST=1` — skip bloom and the saturation lift.

// The figures here are carried at f64 precision and narrowed to f32. Trimming
// them would move the thin ramp stops, which are only ~0.012 apart, so the lint
// that wants them shortened is off for this file.
#![allow(clippy::excessive_precision)]

use threers::prelude::*;

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------
//
// Three orthographic framings that share one frustum. The frustum is expressed
// in *pixels* — half-extents of 624 x 464.5, i.e. a 1248x929 window — and each
// camera divides it by its own zoom. Two things follow:
//
//   * the scene has no intrinsic aspect ratio. Changing the viewport shows
//     *more or less of the scene* rather than rescaling the same framing, so
//     the output size has to keep the viewport's aspect or the composition
//     shears. The assert in `main` enforces that.
//   * rotations are given in degrees and composed as an XYZ Euler, and the
//     look-at is derived from them: the camera looks along its own -Z, so the
//     target is `position + forward * target_offset`.
//
// Axes are three.js's throughout: Y up, -Z forward, `up` of (0, 1, 0).
const VIEW_W: f32 = 1248.0;
const VIEW_H: f32 = 929.0;
const W: u32 = 1600;
const H: u32 = 1191;

/// `NEON_VIEWPORT=WxH` renders as though the window were a different size.
/// Because the frustum is in pixels this changes how much of the scene is on
/// screen rather than rescaling it, so the output is rendered at that size too.
fn viewport() -> (f32, f32) {
    std::env::var("NEON_VIEWPORT")
        .ok()
        .and_then(|v| {
            let (a, b) = v.split_once('x')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .unwrap_or((VIEW_W, VIEW_H))
}

/// A framing. `zoom` divides the shared frustum:
/// half-extent = (right - left) / (2 * zoom).
struct CameraSpec {
    name: &'static str,
    position: [f32; 3],
    /// Euler XYZ, in **degrees**.
    rotation_deg: [f32; 3],
    zoom: f32,
    target_offset: f32,
}

/// The default framing: square on, rotation exactly zero, so it looks straight
/// down -Z and the mark sits centred.
const CAMERA_HERO: CameraSpec = CameraSpec {
    name: "hero",
    position: [-642.9856, 846.9931, 1603.9453],
    rotation_deg: [0.0, 0.0, 0.0],
    zoom: 0.32353354497370923,
    target_offset: 1511.35,
};

/// Closer and tipped down a few degrees, which drops the bright edge toward the
/// bottom of the mark.
const CAMERA_TILT: CameraSpec = CameraSpec {
    name: "tilt",
    position: [-432.2856, 363.4933, 1794.2207],
    rotation_deg: [7.460768, 0.910433, -0.119175],
    zoom: 0.3584859224085421,
    target_offset: 1511.35,
};

/// Tighter still, yawed 13 degrees so the mark turns slightly into the light.
const CAMERA_CLOSE: CameraSpec = CameraSpec {
    name: "close",
    position: [-200.13276808978668, 803.5491492800749, 1715.379767204384],
    rotation_deg: [1.1426908883474647, 12.927035489481012, -0.2556644796343047],
    zoom: 0.5403600876626369,
    target_offset: 1511.35,
};

/// `near` is genuinely negative, which an orthographic projection is happy
/// with: it keeps geometry behind the camera plane in view.
const NEAR: f32 = -100_000.0;
const FAR: f32 = 100_000.0;

// ---------------------------------------------------------------------------
// The mark
// ---------------------------------------------------------------------------
//
// Two solids under one group: a body and a leaf. Both are swept from a 2D
// outline rather than loaded — see `inflated_geometry` — so the whole mark is
// generated from the coordinate tables further down.
const LOGO_POSITION: [f32; 3] = [-664.7068, 886.5597, 101.1524];
/// 8.653 degrees about Y, in radians — just enough yaw to keep the bright edge
/// off the silhouette.
const LOGO_ROTATION_Y: f32 = 0.15102;
const LOGO_SCALE: f32 = 11.7026;

/// The body sits at the group origin, squashed in Z.
///
/// Deliberately an object scale rather than baked into the sweep: the solid is
/// built at its full half-thickness and flattened by the node, which leaves the
/// renderer's normal matrix to tilt the normals accordingly. Baking it would
/// need the same inverse-transpose applied by hand.
const BODY_SCALE_Z: f32 = 0.5755;
/// The leaf's offset within the group.
const LEAF_POSITION: [f32; 3] = [0.3067, 0.0, 0.0];

// ---------------------------------------------------------------------------
// Lights
// ---------------------------------------------------------------------------

/// The only real light. Note `distance` (2521) is far shorter than the ~5487
/// units between it and the mark, which matters — see `LOGO_FRAGMENT`.
const SPOT_POSITION: [f32; 3] = [1715.9844, -1470.7504, 4447.7187];
const SPOT_TARGET: [f32; 3] = [1699.3385, -225.8405, 1780.0101];
const SPOT_COLOR: u32 = 0xb2bde6;
const SPOT_INTENSITY: f32 = 21.409953934214442;
const SPOT_DISTANCE: f32 = 2521.0;
const SPOT_ANGLE: f32 = 0.4886921905584123;
const SPOT_PENUMBRA: f32 = 0.6;
const SPOT_DECAY: f32 = 1.0;

/// A neutral hemisphere fill at 0.75 * PI. It reaches the picture only through
/// the material's indirect term, which is zero here — the base colour is black
/// at full metalness, so there is no Lambert lobe for it to feed. It is kept
/// because anything added to the scene that is *not* the mark will want it.
const HEMI_SKY: u32 = 0xd3d3d3;
const HEMI_GROUND: u32 = 0x828282;
const HEMI_INTENSITY: f32 = 2.356194490192345;

// ---------------------------------------------------------------------------
// Post-processing
// ---------------------------------------------------------------------------
//
// Two effects: a bright-pass bloom screened back over the frame, then a
// saturation lift. Both run in linear light — see `post`.
const BLOOM_INTENSITY: f32 = 0.834;
const BLOOM_THRESHOLD: f32 = 0.635;
const BLOOM_SMOOTHING: f32 = 0.719;
/// Blur support as a power of two: `2^3` texels of a half-resolution buffer,
/// so 16 pixels of the full frame.
const BLOOM_KERNEL: u32 = 3;
/// ...scaled by this. At 0.01 the support is 0.16 px, so **the bloom does not
/// blur**: it is a bright-pass lift screened straight back over the frame.
///
/// That is deliberate, and it is the setting most worth resisting the urge to
/// "improve". A wide Gaussian here wraps the mark in a soft halo that reads as
/// glow at a glance but flattens the thin ramp stops the whole design rests on,
/// and it inflates the lit-pixel count by a third. Raise it only if you want
/// that.
const BLOOM_BLUR_SCALE: f32 = 0.01;
const SATURATION: f32 = 0.576;

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// The body's silhouette, at **uniform arc length** and counter-clockwise.
///
/// Even spacing matters more than point count here. Fitting an outline to a
/// tolerance — Douglas-Peucker and friends — packs knots into the curved
/// stretches and strands them across the flat ones, and Catmull-Rom through
/// unevenly spaced knots wobbles in curvature. The ramp below turns a curvature
/// wobble into a visible ripple along the bright edge, so the knots are spread
/// evenly and the count raised until the ripple goes. Points that merely trace
/// the silhouette accurately are not enough.
const BODY_OUTLINE: [[f32; 2]; 192] = [
    [19.4758, 25.7923],
    [18.0310, 25.8014],
    [16.5908, 25.7351],
    [15.1615, 25.5468],
    [13.7411, 25.2866],
    [12.3398, 24.9363],
    [10.9549, 24.5260],
    [9.5899, 24.0526],
    [8.2370, 23.5470],
    [6.8893, 23.0274],
    [5.5476, 22.4914],
    [4.1910, 21.9953],
    [2.8166, 21.5538],
    [1.4055, 21.2483],
    [-0.0307, 21.2400],
    [-1.4510, 21.4882],
    [-2.8326, 21.9100],
    [-4.1871, 22.4113],
    [-5.5289, 22.9469],
    [-6.8708, 23.4824],
    [-8.2197, 23.9986],
    [-9.5846, 24.4722],
    [-10.9697, 24.8822],
    [-12.3779, 25.2017],
    [-13.8035, 25.4222],
    [-15.2401, 25.5267],
    [-16.6853, 25.5266],
    [-18.1214, 25.4079],
    [-19.5486, 25.2016],
    [-20.9628, 24.9091],
    [-22.3547, 24.5230],
    [-23.7245, 24.0659],
    [-25.0587, 23.5132],
    [-26.3588, 22.8838],
    [-27.6227, 22.1860],
    [-28.8332, 21.3975],
    [-29.9994, 20.5471],
    [-31.1151, 19.6333],
    [-32.1626, 18.6408],
    [-33.1787, 17.6135],
    [-34.1125, 16.5150],
    [-34.9890, 15.3685],
    [-35.8028, 14.1749],
    [-36.5454, 12.9365],
    [-37.2341, 11.6676],
    [-37.8579, 10.3649],
    [-38.4105, 9.0304],
    [-38.9027, 7.6725],
    [-39.3302, 6.2926],
    [-39.7004, 4.8965],
    [-40.0050, 3.4845],
    [-40.2569, 2.0628],
    [-40.4548, 0.6345],
    [-40.5833, -0.8009],
    [-40.6901, -2.2374],
    [-40.7450, -3.6782],
    [-40.7428, -5.1234],
    [-40.6873, -6.5642],
    [-40.5802, -8.0006],
    [-40.4614, -9.4367],
    [-40.2962, -10.8687],
    [-40.0838, -12.2957],
    [-39.8480, -13.7201],
    [-39.5636, -15.1359],
    [-39.2596, -16.5480],
    [-38.9185, -17.9516],
    [-38.5446, -19.3470],
    [-38.1386, -20.7331],
    [-37.7139, -22.1138],
    [-37.2416, -23.4792],
    [-36.7508, -24.8377],
    [-36.2279, -26.1842],
    [-35.6730, -27.5176],
    [-35.0789, -28.8337],
    [-34.4764, -30.1459],
    [-33.8300, -31.4385],
    [-33.1408, -32.7072],
    [-32.4307, -33.9641],
    [-31.6892, -35.2036],
    [-30.9008, -36.4141],
    [-30.0762, -37.5999],
    [-29.2363, -38.7746],
    [-28.3881, -39.9432],
    [-27.5216, -41.0987],
    [-26.6367, -42.2388],
    [-25.7307, -43.3613],
    [-24.7785, -44.4443],
    [-23.8064, -45.5093],
    [-22.7589, -46.5018],
    [-21.6519, -47.4261],
    [-20.4674, -48.2525],
    [-19.2070, -48.9562],
    [-17.8746, -49.5106],
    [-16.4753, -49.8655],
    [-15.0439, -50.0288],
    [-13.6031, -49.9739],
    [-12.1745, -49.7823],
    [-10.7712, -49.4423],
    [-9.4012, -48.9860],
    [-8.0600, -48.4488],
    [-6.7301, -47.8861],
    [-5.3891, -47.3500],
    [-4.0282, -46.8657],
    [-2.6371, -46.4772],
    [-1.2226, -46.1870],
    [0.2088, -46.0175],
    [1.6496, -45.9625],
    [3.0946, -45.9687],
    [4.5271, -46.1197],
    [5.9504, -46.3597],
    [7.3477, -46.7269],
    [8.7127, -47.2000],
    [10.0523, -47.7399],
    [11.3770, -48.3147],
    [12.7127, -48.8632],
    [14.0828, -49.3195],
    [15.4883, -49.6520],
    [16.9161, -49.8518],
    [18.3562, -49.9142],
    [19.7908, -49.7857],
    [21.2019, -49.4880],
    [22.5636, -49.0089],
    [23.8594, -48.3728],
    [25.0764, -47.5977],
    [26.2218, -46.7206],
    [27.2950, -45.7576],
    [28.3169, -44.7357],
    [29.2792, -43.6618],
    [30.1887, -42.5425],
    [31.0736, -41.4022],
    [31.9367, -40.2444],
    [32.7756, -39.0689],
    [33.6022, -37.8845],
    [34.4161, -36.6910],
    [35.1899, -35.4714],
    [35.9247, -34.2282],
    [36.6353, -32.9715],
    [37.3031, -31.6909],
    [37.9494, -30.3982],
    [38.5508, -29.0855],
    [39.1256, -27.7608],
    [39.6618, -26.4193],
    [40.1562, -25.0621],
    [40.6113, -23.6905],
    [39.6034, -22.9169],
    [38.3415, -22.2156],
    [37.1151, -21.4530],
    [35.9397, -20.6140],
    [34.8284, -19.6944],
    [33.7589, -18.7276],
    [32.7568, -17.6886],
    [31.8230, -16.5901],
    [30.9603, -15.4327],
    [30.1847, -14.2143],
    [29.4867, -12.9505],
    [28.8917, -11.6349],
    [28.3846, -10.2825],
    [27.9760, -8.8972],
    [27.6608, -7.4876],
    [27.4442, -6.0613],
    [27.3069, -4.6274],
    [27.2824, -3.1840],
    [27.3192, -1.7420],
    [27.4470, -0.3068],
    [27.6721, 1.1184],
    [27.9953, 2.5259],
    [28.4190, 3.9069],
    [28.9244, 5.2599],
    [29.5350, 6.5685],
    [30.2180, 7.8405],
    [30.9820, 9.0661],
    [31.8408, 10.2261],
    [32.7547, 11.3415],
    [33.7404, 12.3952],
    [34.7879, 13.3877],
    [35.8898, 14.3179],
    [37.0539, 15.1726],
    [37.9513, 16.0624],
    [37.0764, 17.2100],
    [36.1426, 18.3086],
    [35.1389, 19.3462],
    [34.0661, 20.3096],
    [32.9419, 21.2133],
    [31.7527, 22.0332],
    [30.5176, 22.7802],
    [29.2329, 23.4387],
    [27.9126, 24.0233],
    [26.5543, 24.5147],
    [25.1702, 24.9279],
    [23.7636, 25.2568],
    [22.3425, 25.5102],
    [20.9118, 25.6886],
];

/// The leaf's silhouette, on the same terms as `BODY_OUTLINE`.
const LEAF_OUTLINE: [[f32; 2]; 128] = [
    [19.1869, 49.9918],
    [18.6652, 49.9586],
    [18.1450, 49.9035],
    [17.6264, 49.8315],
    [17.1102, 49.7412],
    [16.5973, 49.6327],
    [16.0877, 49.5092],
    [15.5818, 49.3709],
    [15.0807, 49.2160],
    [14.5833, 49.0502],
    [14.0922, 48.8665],
    [13.6041, 48.6751],
    [13.1218, 48.4695],
    [12.6453, 48.2509],
    [12.1733, 48.0224],
    [11.7042, 47.7874],
    [11.2449, 47.5349],
    [10.7892, 47.2759],
    [10.3403, 47.0052],
    [9.8957, 46.7273],
    [9.4585, 46.4372],
    [9.0289, 46.1367],
    [8.6034, 45.8304],
    [8.1872, 45.5119],
    [7.7790, 45.1837],
    [7.3777, 44.8476],
    [6.9818, 44.5053],
    [6.5951, 44.1523],
    [6.2235, 43.7819],
    [5.8525, 43.4109],
    [5.4952, 43.0281],
    [5.1467, 42.6374],
    [4.8081, 42.2382],
    [4.4773, 41.8323],
    [4.1573, 41.4172],
    [3.8482, 40.9938],
    [3.5469, 40.5647],
    [3.2559, 40.1285],
    [2.9723, 39.6875],
    [2.6999, 39.2394],
    [2.4400, 38.7842],
    [2.1871, 38.3252],
    [1.9464, 37.8593],
    [1.7117, 37.3900],
    [1.4898, 36.9150],
    [1.2779, 36.4356],
    [1.0777, 35.9509],
    [0.8881, 35.4620],
    [0.7110, 34.9684],
    [0.5452, 34.4709],
    [0.3888, 33.9703],
    [0.2495, 33.4647],
    [0.1185, 32.9568],
    [0.0035, 32.4451],
    [-0.0957, 31.9304],
    [-0.1844, 31.4139],
    [-0.2565, 30.8953],
    [-0.3112, 30.3749],
    [-0.3559, 29.8539],
    [-0.3716, 29.3305],
    [-0.3872, 28.8070],
    [-0.3668, 28.2839],
    [-0.3151, 27.7635],
    [-0.2398, 27.2452],
    [0.1318, 26.9939],
    [0.6517, 26.9367],
    [1.1748, 26.9159],
    [1.6982, 26.9333],
    [2.2192, 26.9785],
    [2.7383, 27.0471],
    [3.2548, 27.1354],
    [3.7675, 27.2458],
    [4.2764, 27.3732],
    [4.7800, 27.5197],
    [5.2778, 27.6855],
    [5.7715, 27.8621],
    [6.2595, 28.0544],
    [6.7408, 28.2621],
    [7.2163, 28.4828],
    [7.6856, 28.7174],
    [8.1491, 28.9624],
    [8.6070, 29.2174],
    [9.0562, 29.4880],
    [9.4989, 29.7690],
    [9.9352, 30.0598],
    [10.3642, 30.3612],
    [10.7853, 30.6732],
    [11.1985, 30.9954],
    [11.6023, 31.3287],
    [11.9969, 31.6727],
    [12.3866, 32.0220],
    [12.7588, 32.3917],
    [13.1298, 32.7627],
    [13.4841, 33.1482],
    [13.8301, 33.5410],
    [14.1603, 33.9474],
    [14.4887, 34.3555],
    [14.8090, 34.7702],
    [15.1156, 35.1955],
    [15.4169, 35.6246],
    [15.7045, 36.0632],
    [15.9865, 36.5053],
    [16.2567, 36.9547],
    [16.5166, 37.4099],
    [16.7636, 37.8723],
    [17.0043, 38.3381],
    [17.2326, 38.8103],
    [17.4533, 39.2858],
    [17.6611, 39.7671],
    [17.8588, 40.2528],
    [18.0438, 40.7435],
    [18.2189, 41.2378],
    [18.3847, 41.7353],
    [18.5379, 42.2368],
    [18.6796, 42.7418],
    [18.8105, 43.2497],
    [18.9247, 43.7615],
    [19.0308, 44.2751],
    [19.1218, 44.7913],
    [19.2004, 45.3092],
    [19.2665, 45.8286],
    [19.3143, 46.3495],
    [19.3588, 46.8705],
    [19.3760, 47.3939],
    [19.3915, 47.9173],
    [19.3760, 48.4407],
    [19.3474, 48.9630],
    [19.2959, 49.4837],
];

/// Half-thickness in Z, and the inset at which the solid closes.
///
/// The imported solid is not an extrusion with a bevel — it is the outline
/// *inflated*, and the inflation is an ellipse. Sampling the real mesh's
/// vertices against a distance field built from its own silhouette, every
/// vertex satisfies
///
///     (1 + sd/D)^2 + (z/Z)^2 = 1
///
/// to within ~2% of the body's width, where `sd` is the signed distance to the
/// outline (negative inside). A cosine or quadratic profile is 4-5x worse on the
/// same fit, so the ellipse is the shape, not a guess. `D` comes out at very
/// nearly the shape's maximum inradius, which is what that equation implies:
/// the cross-section has shrunk to the medial axis by the time z reaches Z.
const BODY_HALF_Z: f32 = 37.29941;
const BODY_INSET: f32 = 32.976;
const LEAF_HALF_Z: f32 = 5.4970155;
const LEAF_INSET: f32 = 5.705;

/// Catmull-Rom subdivisions between stored outline points. Enough that the
/// remaining polyline facets in the distance field are far below one output
/// pixel; the knots themselves are already evenly spaced, which is what keeps
/// the interpolant's curvature well behaved.
const OUTLINE_SUBDIV: usize = 6;

/// A closed polygon, with the signed distance and gradient the sweep needs.
struct Outline {
    pts: Vec<[f32; 2]>,
}

impl Outline {
    /// Resample a stored outline onto a closed Catmull-Rom curve through its
    /// points.
    ///
    /// Not cosmetic. The distance field of a polygon is only C1 *inside* each
    /// segment's Voronoi cell; across those boundaries the gradient kinks, and
    /// since the gradient *is* the normal here, every kink is a visible facet
    /// on the rim — which is exactly where this material does all its shading.
    /// At 90 stored points the facets were plainly countable. Subdividing the
    /// curve shrinks each one instead of trying to hide it, and Catmull-Rom
    /// interpolates rather than approximates, so the silhouette does not move.
    fn from_knots(pts: &[[f32; 2]], subdiv: usize) -> Self {
        let n = pts.len();
        let mut out = Vec::with_capacity(n * subdiv);
        for i in 0..n {
            let p0 = pts[(i + n - 1) % n];
            let p1 = pts[i];
            let p2 = pts[(i + 1) % n];
            let p3 = pts[(i + 2) % n];
            for s in 0..subdiv {
                let t = s as f32 / subdiv as f32;
                let (t2, t3) = (t * t, t * t * t);
                let mut p = [0.0f32; 2];
                for c in 0..2 {
                    p[c] = 0.5
                        * ((2.0 * p1[c])
                            + (-p0[c] + p2[c]) * t
                            + (2.0 * p0[c] - 5.0 * p1[c] + 4.0 * p2[c] - p3[c]) * t2
                            + (-p0[c] + 3.0 * p1[c] - 3.0 * p2[c] + p3[c]) * t3);
                }
                out.push(p);
            }
        }
        Self { pts: out }
    }

    /// Exact signed distance to the polygon (negative inside) and the unit
    /// gradient of that distance. Both fall out of the closest point: the
    /// gradient of a distance field is the unit vector away from it.
    fn sample(&self, px: f32, py: f32) -> (f32, f32, f32) {
        let n = self.pts.len();
        let mut best = f32::INFINITY;
        let (mut cx, mut cy) = (0.0f32, 0.0f32);
        let mut inside = false;
        for i in 0..n {
            let a = self.pts[i];
            let b = self.pts[(i + 1) % n];
            let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
            let (wx, wy) = (px - a[0], py - a[1]);
            let len2 = ex * ex + ey * ey;
            let t = if len2 > 0.0 {
                ((wx * ex + wy * ey) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let (qx, qy) = (a[0] + ex * t, a[1] + ey * t);
            let d2 = (px - qx) * (px - qx) + (py - qy) * (py - qy);
            if d2 < best {
                best = d2;
                cx = qx;
                cy = qy;
            }
            // Crossing-number test, folded into the same loop.
            if (a[1] > py) != (b[1] > py) {
                let x_int = a[0] + (py - a[1]) / (b[1] - a[1]) * ex;
                if px < x_int {
                    inside = !inside;
                }
            }
        }
        let d = best.sqrt();
        let s = if inside { -1.0 } else { 1.0 };
        let inv = if d > 1e-6 { s / d } else { 0.0 };
        (s * d, (px - cx) * inv, (py - cy) * inv)
    }
}

/// Sweep an outline into the inflated solid described above.
///
/// The surface is a height field over the outline's interior — `z = +-Z *
/// sqrt(1 - u^2)` with `u = clamp(1 + sd/D, 0, 1)` — so it is built as two
/// sheets that meet at the silhouette, where `u = 1` drives the height to zero
/// and both sheets close. Cells are clipped against `sd <= 0` by
/// Sutherland-Hodgman, which handles the concave bite and the leaf's points
/// without special cases.
///
/// Normals are analytic rather than averaged from the triangles: for
/// `F = u^2 + (z/Z)^2 - 1` the gradient is `(2u*sdx/D, 2u*sdy/D, 2z/Z^2)`.
/// Taking them from the field is what keeps the rim smooth — and the rim is the
/// whole picture here, because the material shades almost entirely off the
/// normal (see `LOGO_FRAGMENT`).
fn inflated_geometry(outline: &Outline, inset: f32, half_z: f32, res: usize) -> BufferGeometry {
    let (mut lo_x, mut lo_y) = (f32::INFINITY, f32::INFINITY);
    let (mut hi_x, mut hi_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for p in &outline.pts {
        lo_x = lo_x.min(p[0]);
        lo_y = lo_y.min(p[1]);
        hi_x = hi_x.max(p[0]);
        hi_y = hi_y.max(p[1]);
    }
    // One cell of margin so the outline never touches the grid edge.
    let step = ((hi_x - lo_x).max(hi_y - lo_y)) / res as f32;
    lo_x -= step;
    lo_y -= step;
    let nx = ((hi_x + step - lo_x) / step).ceil() as usize + 1;
    let ny = ((hi_y + step - lo_y) / step).ceil() as usize + 1;

    // Sample the field once per grid node; cells share corners.
    let mut sd = vec![0.0f32; nx * ny];
    let mut grad = vec![[0.0f32; 2]; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            let x = lo_x + i as f32 * step;
            let y = lo_y + j as f32 * step;
            let (d, gx, gy) = outline.sample(x, y);
            sd[j * nx + i] = d;
            grad[j * nx + i] = [gx, gy];
        }
    }

    let mut positions: Vec<f32> = Vec::new();
    let mut normals: Vec<f32> = Vec::new();
    let mut uvs: Vec<f32> = Vec::new();

    // A clipped corner: position, signed distance and gradient interpolated
    // together so the normal stays consistent with the height.
    #[derive(Clone, Copy)]
    struct P {
        x: f32,
        y: f32,
        d: f32,
        gx: f32,
        gy: f32,
    }
    let lerp = |a: P, b: P| -> P {
        let t = a.d / (a.d - b.d);
        P {
            x: a.x + (b.x - a.x) * t,
            y: a.y + (b.y - a.y) * t,
            d: 0.0,
            gx: a.gx + (b.gx - a.gx) * t,
            gy: a.gy + (b.gy - a.gy) * t,
        }
    };

    let span_x = hi_x - lo_x;
    let span_y = hi_y - lo_y;
    let mut poly: Vec<P> = Vec::with_capacity(8);
    for j in 0..ny - 1 {
        for i in 0..nx - 1 {
            let corner = |ii: usize, jj: usize| -> P {
                let k = jj * nx + ii;
                P {
                    x: lo_x + ii as f32 * step,
                    y: lo_y + jj as f32 * step,
                    d: sd[k],
                    gx: grad[k][0],
                    gy: grad[k][1],
                }
            };
            let sq = [
                corner(i, j),
                corner(i + 1, j),
                corner(i + 1, j + 1),
                corner(i, j + 1),
            ];
            if sq.iter().all(|p| p.d > 0.0) {
                continue;
            }
            // Sutherland-Hodgman against the half-space `d <= 0`.
            poly.clear();
            for k in 0..4 {
                let a = sq[k];
                let b = sq[(k + 1) % 4];
                let a_in = a.d <= 0.0;
                let b_in = b.d <= 0.0;
                if a_in {
                    poly.push(a);
                }
                if a_in != b_in {
                    poly.push(lerp(a, b));
                }
            }
            if poly.len() < 3 {
                continue;
            }

            // Fan-triangulate, emitting the top sheet and the mirrored bottom.
            for side in [1.0f32, -1.0] {
                for t in 1..poly.len() - 1 {
                    let tri = [poly[0], poly[t], poly[t + 1]];
                    // Bottom sheet reverses winding so both face outward.
                    let order: [usize; 3] = if side > 0.0 { [0, 1, 2] } else { [0, 2, 1] };
                    for &o in &order {
                        let p = tri[o];
                        let u = (1.0 + p.d / inset).clamp(0.0, 1.0);
                        let h = side * half_z * (1.0 - u * u).max(0.0).sqrt();
                        positions.extend_from_slice(&[p.x, p.y, h]);
                        // grad F, normalised.
                        let (mut nx_, mut ny_, mut nz_) = (
                            2.0 * u * p.gx / inset,
                            2.0 * u * p.gy / inset,
                            2.0 * h / (half_z * half_z),
                        );
                        let l = (nx_ * nx_ + ny_ * ny_ + nz_ * nz_).sqrt();
                        if l > 1e-12 {
                            nx_ /= l;
                            ny_ /= l;
                            nz_ /= l;
                        } else {
                            nz_ = side;
                        }
                        normals.extend_from_slice(&[nx_, ny_, nz_]);
                        uvs.extend_from_slice(&[(p.x - lo_x) / span_x, (p.y - lo_y) / span_y]);
                    }
                }
            }
        }
    }

    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(positions, 3));
    g.set_attribute("normal", BufferAttribute::new(normals, 3));
    g.set_attribute("uv", BufferAttribute::new(uvs, 2));
    g
}

// ---------------------------------------------------------------------------
// The material
// ---------------------------------------------------------------------------
//
// Both meshes run the same shader, differing only in their parameters. It is
// three layers composited in order:
//
//     base = mix(black, physical_brdf, 0.6)   // black metal, roughness 0.59
//     col  = screen(base, noise)              // four-colour simplex field
//     col  = mix(col, ramp.rgb, ramp.a)       // ten-stop light-angle ramp
//
// The base colour is black at full metalness, so the Lambert lobe is zero and,
// with no environment map, so is the indirect specular. The only term that can
// survive is the spot's direct specular — and it does not, because the light is
// 5487 units from the mark against a `distance` cutoff of 2521, which sends
// three.js's `pow2(saturate(1 - pow4(d / cutoff)))` attenuation to zero.
//
// So in this configuration the picture is *entirely* the noise and ramp layers,
// and the base evaluates to black. That looks like a bug and is not: it is what
// makes the interior read as a clean hole rather than a dim grey shape. The
// physical term is still computed rather than dropped, so moving the light
// inside its own range, or widening that range, lights the mark properly.
const LOGO_FRAGMENT: &str = r#"
const NL_F3: f32 = 0.3333333;
const NL_G3: f32 = 0.1666667;
const NL_PI: f32 = 3.14159265359;

// A cheap gradient-hash simplex: the gradients come out of a sin-based hash
// rather than a permutation table, so the exact arithmetic is part of the look
// and a "better" hash gives a different picture.
//
// It is also the one thing here that is not reproducible across GPUs, which is
// worth knowing before chasing a colour difference. `nl_hash3` computes
// `fract(512.0 * (4096.0 * sin x))` — around 2e6 before the `fract`, where an
// f32 ulp is 0.125. The fractional part is decided by the last couple of bits
// of that multiply, so two shader compilers that disagree by one ulp on `sin`
// return different gradients. That is unusually visible at this scale: `st`
// spans a few hundredths of a cell across the whole mark, so only about eight
// lattice corners are ever hashed and each one paints a large region. The
// blue-to-violet wash on the outer rim is where it shows.
//
// The cell *structure* is safe — `floor` lands unambiguously — only the
// gradient values inside it move.
fn nl_hash3(c: vec3<f32>) -> vec3<f32> {
    var j = 4096.0 * sin(dot(c, vec3<f32>(17.0, 59.4, 15.0)));
    var r = vec3<f32>(0.0, 0.0, 0.0);
    r.z = fract(512.0 * j);
    j = j * 0.125;
    r.x = fract(512.0 * j);
    j = j * 0.125;
    r.y = fract(512.0 * j);
    return r - vec3<f32>(0.5, 0.5, 0.5);
}

fn nl_simplex(p: vec3<f32>) -> f32 {
    let s = floor(p + vec3<f32>(dot(p, vec3<f32>(NL_F3, NL_F3, NL_F3))));
    let x = p - s + vec3<f32>(dot(s, vec3<f32>(NL_G3, NL_G3, NL_G3)));
    let e = step(vec3<f32>(0.0), x - x.yzx);
    let i1 = e * (1.0 - e.zxy);
    let i2 = 1.0 - e.zxy * (1.0 - e);
    let x1 = x - i1 + vec3<f32>(NL_G3);
    let x2 = x - i2 + vec3<f32>(2.0 * NL_G3);
    let x3 = x - vec3<f32>(1.0) + vec3<f32>(3.0 * NL_G3);
    var w = vec4<f32>(dot(x, x), dot(x1, x1), dot(x2, x2), dot(x3, x3));
    w = max(vec4<f32>(0.6) - w, vec4<f32>(0.0));
    var d = vec4<f32>(
        dot(nl_hash3(s), x),
        dot(nl_hash3(s + i1), x1),
        dot(nl_hash3(s + i2), x2),
        dot(nl_hash3(s + vec3<f32>(1.0)), x3),
    );
    w = w * w;
    w = w * w;
    d = d * w;
    return dot(d, vec4<f32>(52.0));
}

/// Screen blend. `a` is what is already there, `b` the layer above it.
fn nl_screen(a: vec3<f32>, b: vec3<f32>, alpha: f32) -> vec3<f32> {
    let t = vec3<f32>(1.0) - (vec3<f32>(1.0) - a) * (vec3<f32>(1.0) - b);
    return mix(a, t, alpha);
}

/// The colour field: three layers of warped simplex noise selecting between
/// four colours.
///
/// It is evaluated in **object space** (`in.local_pos`), and `size` divides
/// rather than multiplies — at ~(-364, 294, -80) against a body only 81 units
/// across, the sample point moves a few hundredths of a cell from one side of
/// the mark to the other. That is why this reads as a smooth iridescent sweep
/// and not as noise. Shrink `size` and it becomes recognisably noisy.
fn nl_noise(local: vec3<f32>) -> vec3<f32> {
    let scale = max(abs(u_data.data[1].x), 0.001);
    let size = u_data.data[1].yzw;
    let mv = u_data.data[2].x;
    let fa = u_data.data[2].yz;
    let fb = vec2<f32>(u_data.data[2].w, u_data.data[3].x);
    let dist = u_data.data[3].yz;

    var st = local / size;
    st = st / scale;

    let q = vec3<f32>(
        nl_simplex(st),
        nl_simplex(st + vec3<f32>(1.0)),
        nl_simplex(st + vec3<f32>(1.0)),
    );
    let r = vec3<f32>(
        nl_simplex(st + vec3<f32>(dist, 1.0) * q + vec3<f32>(fa, 1.0) + vec3<f32>(mv)),
        nl_simplex(st + vec3<f32>(dist, 1.0) * q + vec3<f32>(fb, 1.0) + vec3<f32>(mv)),
        nl_simplex(st * q),
    );
    let f = nl_simplex(st + r);

    var color = mix(u_data.data[4], u_data.data[5], clamp(f * f * 4.0, 0.0, 1.0));
    color = mix(color, u_data.data[6], clamp(length(q), 0.0, 1.0));
    // `length()` of a scalar in GLSL is `abs()`.
    color = mix(color, u_data.data[7], clamp(abs(r.x), 0.0, 1.0));
    return clamp(color, vec4<f32>(0.0), vec4<f32>(1.0)).rgb;
}

/// The ramp layer: a ten-stop gradient indexed by the half-Lambert term against
/// the spot light. Hemisphere lights do not enter — only punctual ones aim this.
///
/// Applied as a running `mix` over all ten stops rather than a lookup. With
/// ascending stops the two agree, but the alpha channel rides along, and that
/// alpha is what carves the mark: stops 3 and 4 are fully transparent, so the
/// mid-range of `t` shows the noise layer beneath, while everything past 0.799
/// is opaque black and gives the dark face.
fn nl_ramp(world_pos: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
    var spot = u_data.data[8].xyz;
    if (frame.light_counts.z > 0u) {
        spot = frame.spot_lights[0].position.xyz;
    }
    let l = normalize(spot - world_pos);
    let t = clamp(dot(l, n) * 0.5 + 0.5, 0.0, 1.0);

    var color = vec4<f32>(u_s0[0], u_s0[1], u_s0[2], u_s0[3]);
    for (var i: u32 = 1u; i < 10u; i = i + 1u) {
        let s0 = u_s0[40u + i - 1u];
        let s1 = u_s0[40u + i];
        let p = clamp((t - s0) / max(s1 - s0, 1e-6), 0.0, 1.0);
        let ci = vec4<f32>(u_s0[i * 4u], u_s0[i * 4u + 1u], u_s0[i * 4u + 2u], u_s0[i * 4u + 3u]);
        color = mix(color, ci, smoothstep(0.0, 1.0, p));
    }
    return color;
}

fn nl_d_ggx(noh: f32, a: f32) -> f32 {
    let a2 = a * a;
    let d = noh * noh * (a2 - 1.0) + 1.0;
    return a2 / (NL_PI * d * d);
}

fn nl_v_smith(nov: f32, nol: f32, a: f32) -> f32 {
    let a2 = a * a;
    let gv = nol * sqrt(nov * nov * (1.0 - a2) + a2);
    let gl = nov * sqrt(nol * nol * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-5);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let v = normalize(frame.camera_position.xyz - in.world_pos);
    let nov = max(dot(n, v), 1e-4);

    let roughness = u_data.data[0].x;
    let f0 = vec3<f32>(u_data.data[0].z);
    let a = max(roughness * roughness, 1e-3);

    // Black base colour at metalness 1: the Lambert term is
    // `diffuse * (1 - metalness)` = 0, and with no environment map the indirect
    // specular is 0 too. Only the spot's direct specular can survive.
    var lit = vec3<f32>(0.0);
    if (frame.light_counts.z > 0u) {
        let sl = frame.spot_lights[0];
        let to_l = sl.position.xyz - in.world_pos;
        let d = length(to_l);
        let l = to_l / max(d, 1e-4);
        let cos_angle = dot(-l, normalize(sl.direction.xyz));
        var cone = 0.0;
        if (cos_angle > sl.params.z) {
            cone = smoothstep(sl.params.z, sl.params.w, cos_angle);
        }
        // three.js `getDistanceAttenuation`. `params.x` is the cutoff distance,
        // `params.y` the decay exponent.
        var falloff = 1.0 / max(pow(d, sl.params.y), 0.01);
        if (sl.params.x > 0.0) {
            let k = clamp(1.0 - pow(d / sl.params.x, 4.0), 0.0, 1.0);
            falloff = falloff * k * k;
        }
        let nol = max(dot(n, l), 0.0);
        if (nol > 0.0 && cone > 0.0) {
            let h = normalize(l + v);
            let noh = max(dot(n, h), 0.0);
            let voh = max(dot(v, h), 0.0);
            let fr = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - voh, 5.0);
            let spec = fr * nl_d_ggx(noh, a) * nl_v_smith(nov, nol, a);
            lit = lit + sl.color.rgb * falloff * cone * nol * spec;
        }
    }

    // `spe_blend(diffuseColor /* black */, outgoingLight, nodeU5, NORMAL)`.
    let base = mix(vec3<f32>(0.0), lit, u_data.data[0].w);

    // Noise layer: Screen, alpha exactly 1 (see the note above).
    var col = nl_screen(base, nl_noise(in.local_pos), 1.0);

    // Ramp layer: Normal, alpha = the ramp's own.
    let ramp = nl_ramp(in.world_pos, n);
    col = mix(col, ramp.rgb, ramp.a);

    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
"#;

/// The ten-stop ramp shared by both meshes: `nodeUA0` (colours) then `nodeUA1`
/// (positions), laid out as 40 floats followed by 10.
///
/// Reading it top to bottom explains the whole image. `t` is the half-Lambert
/// against the spot, so the logo's face — pointed at the light — sits at
/// t ~ 0.90 and lands on the last stop: opaque black. Turning away through the
/// rim sweeps t down through white (0.760), cyan (0.656) and violet/pink
/// (0.775-0.787) — the bright band — and then into stops 3 and 4, which are
/// fully transparent and let the noise layer's blues through on the far side.
const RAMP: [[f32; 4]; 10] = [
    [0.82074863470873793, 0.84512682038834963, 1.0, 1.0],
    [
        0.72139692050970861,
        0.53720418689320382,
        1.0,
        0.55400000000000005,
    ],
    [
        0.066775006586013042,
        0.042176125737047411,
        0.089786862864077666,
        0.115,
    ],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.36862745098039218, 0.52941176470588236, 0.0],
    [0.0, 0.66696958434466014, 0.96105127427184467, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [
        0.56067839805825193,
        0.18037014563106779,
        1.0,
        0.40100000000000002,
    ],
    [
        0.98672633495145645,
        0.32584646395592426,
        0.67479103584156497,
        0.71599999999999997,
    ],
    [0.0, 0.0, 0.0, 1.0],
];
const RAMP_STOPS: [f32; 10] = [
    0.0,
    0.01458688699315892,
    0.037397308906647844,
    0.48698600874926812,
    0.55911664427566887,
    0.6561921417300205,
    0.75963228228034629,
    0.77473705131752224,
    0.78690278148096471,
    0.79915239779569935,
];

/// Per-mesh material parameters. Everything not listed is shared.
struct MaterialParams {
    /// Alpha of the black colour layer underneath the physical term.
    color_alpha: f32,
    /// Divides the noise sample point, after `noise_size`.
    noise_scale: f32,
    /// Divides object-space position. Large values keep the whole mark inside a
    /// fraction of one noise cell, which is what makes the field read as a wash.
    noise_size: [f32; 3],
    /// A constant offset into the field — slides the pattern without changing it.
    noise_move: f32,
    /// The two warp offsets and the distortion applied to the second octave.
    f_a: [f32; 2],
    f_b: [f32; 2],
    distortion: [f32; 2],
    /// The four colours the field selects between. Their alphas are 1, which is
    /// why the noise layer always fully screens.
    colors: [[f32; 3]; 4],
}

/// The body: blues and a violet.
// `distortion.x` really is 3.14; clippy reads it as a fumbled `PI`, which it is
// not.
#[allow(clippy::approx_constant)]
const BODY_PARAMS: MaterialParams = MaterialParams {
    color_alpha: 1.0,
    noise_scale: 3.76,
    noise_size: [-364.0, 294.0, -80.0],
    noise_move: 4.54,
    f_a: [3.7, 30.28],
    f_b: [11.12, 4.4],
    distortion: [3.14, 5.94],
    colors: [
        [0.4058138652912622, 0.7029069326456311, 1.0], // #67b3ff
        [0.0, 0.0, 0.0],                               // #000000
        [0.0, 0.6399999999999997, 1.0],                // #00a3ff
        [0.5455539542880261, 0.2209496359223302, 1.0], // #8b38ff
    ],
};

/// The leaf: a hotter palette — cyan, purple, magenta.
const LEAF_PARAMS: MaterialParams = MaterialParams {
    color_alpha: 0.0,
    noise_scale: 6.26,
    noise_size: [-364.0, 294.0, -118.0],
    noise_move: 4.54,
    f_a: [1.7, 13.06],
    f_b: [11.3, 4.4],
    distortion: [6.48, -14.92],
    colors: [
        [0.217308859223301, 0.9671269720873787, 1.0], // #37f7ff
        [0.0, 0.0, 0.0],                              // #000000
        [0.5, 0.0, 1.0],                              // #8000ff
        [1.0, 0.0, 0.75],                             // #ff00bf
    ],
};

/// Shared across both meshes. The shader squares `SPECULAR_INTENSITY` and
/// scales it by 0.16 to get F0; `LIGHT_ALPHA` is how far the physical term is
/// mixed up from black.
const ROUGHNESS: f32 = 0.59;
const METALNESS: f32 = 1.0;
const SPECULAR_INTENSITY: f32 = 1.17;
const LIGHT_ALPHA: f32 = 0.6;

// ---------------------------------------------------------------------------
// The timeline
// ---------------------------------------------------------------------------
//
// A 15-second clip driven by three chained tweens. "Chained" is the important
// word: the legs run one after another, so a leg's start time is the sum of
// everything before it, not zero.
//
//   body        1 s hold, then 8 s linear into its second parameter set
//   leaf        1 s hold, then 15 s ease-in-out into its second set
//   spot light  1 s hold, then 2 s, 3 s, 3 s, 3 s, 3 s through five positions
//
// Each opens with the same one-second hold so the clip starts on a settled
// frame. The spot's chain is what sets the length, and moving it is what
// animates the picture: the ramp indexes off the angle to the light, so
// carrying the light around the mark sweeps the bright edge around with it.
// The two meshes morph their noise fields underneath, which recolours the edge
// as it travels.
//
// Nothing loops — each chain plays once and holds its final value.
const ANIM_SECONDS: f32 = 15.0;
const ANIM_FPS: u32 = 30;

/// Easing by index: 4 is ease-in-out, anything else is linear. Only those two
/// are used here.
fn ease(kind: u8, x: f32) -> f32 {
    match kind {
        4 => cubic_bezier(0.42, 0.0, 0.58, 1.0, x),
        _ => x.clamp(0.0, 1.0),
    }
}

/// CSS-style `cubic-bezier(x1, y1, x2, y2)`: solve x(t) = `x` for t by
/// bisection, then return y(t). Bisection rather than Newton because the curve
/// is evaluated a handful of times per frame and monotonicity makes it
/// unconditionally safe.
fn cubic_bezier(x1: f32, y1: f32, x2: f32, y2: f32, x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    let bez = |a: f32, b: f32, t: f32| {
        let u = 1.0 - t;
        3.0 * u * u * t * a + 3.0 * u * t * t * b + t * t * t
    };
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    for _ in 0..32 {
        let mid = 0.5 * (lo + hi);
        if bez(x1, x2, mid) < x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    bez(y1, y2, 0.5 * (lo + hi))
}

/// One leg of a chained `Transition`: how long it lasts and how it eases.
struct Leg {
    seconds: f32,
    easing: u8,
}

/// Walk a chain of legs at time `t` and return `(index, progress)` for the leg
/// in flight, eased. Past the end, the last leg is held complete.
fn chain_at(legs: &[Leg], t: f32) -> (usize, f32) {
    let mut start = 0.0;
    for (i, leg) in legs.iter().enumerate() {
        if t < start + leg.seconds || i == legs.len() - 1 {
            let raw = ((t - start) / leg.seconds).clamp(0.0, 1.0);
            return (i, ease(leg.easing, raw));
        }
        start += leg.seconds;
    }
    (0, 0.0)
}

fn lerp(a: f32, b: f32, k: f32) -> f32 {
    a + (b - a) * k
}
fn lerp2(a: [f32; 2], b: [f32; 2], k: f32) -> [f32; 2] {
    [lerp(a[0], b[0], k), lerp(a[1], b[1], k)]
}
fn lerp3(a: [f32; 3], b: [f32; 3], k: f32) -> [f32; 3] {
    [
        lerp(a[0], b[0], k),
        lerp(a[1], b[1], k),
        lerp(a[2], b[2], k),
    ]
}

/// The body's second parameter set, reached over 8 s of **linear** tween.
///
/// Only six properties move; everything the `..BODY_PARAMS` tail picks up —
/// `noise_size`, `f_a`, colours B and D — holds its opening value for the whole
/// clip. Colours are interpolated componentwise, which is the space the shader
/// consumes them in.
fn body_params_at(t: f32) -> MaterialParams {
    let legs = [
        Leg {
            seconds: 1.0,
            easing: 4,
        },
        Leg {
            seconds: 8.0,
            easing: 0,
        },
    ];
    let (leg, k) = chain_at(&legs, t);
    let k = if leg == 0 { 0.0 } else { k };
    MaterialParams {
        noise_scale: lerp(BODY_PARAMS.noise_scale, 8.85, k),
        noise_move: lerp(BODY_PARAMS.noise_move, 5.64, k),
        f_b: lerp2(BODY_PARAMS.f_b, [11.12, 5.86], k),
        distortion: lerp2(BODY_PARAMS.distortion, [6.16, 5.94], k),
        colors: [
            lerp3(
                BODY_PARAMS.colors[0],
                [0.40392156862745093, 0.9463529411764705, 0.9999999999999999],
                k,
            ),
            BODY_PARAMS.colors[1],
            lerp3(BODY_PARAMS.colors[2], [0.5179999999999998, 0.0, 1.0], k),
            BODY_PARAMS.colors[3],
        ],
        ..BODY_PARAMS
    }
}

/// The leaf's second parameter set, over 15 s of ease-in-out. Only `distortion`
/// and `f_a` move.
///
/// This leg starts at t = 1 s and runs 15 s, so it lands at 16 — a second past
/// the end of the clip. The leaf is still travelling when the clip stops, which
/// is intentional: it keeps the leaf from settling in step with the body.
fn leaf_params_at(t: f32) -> MaterialParams {
    let legs = [
        Leg {
            seconds: 1.0,
            easing: 4,
        },
        Leg {
            seconds: 15.0,
            easing: 4,
        },
    ];
    let (leg, k) = chain_at(&legs, t);
    let k = if leg == 0 { 0.0 } else { k };
    MaterialParams {
        distortion: lerp2(LEAF_PARAMS.distortion, [10.42, -23.1], k),
        f_a: lerp2(LEAF_PARAMS.f_a, [1.7, 17.68], k),
        ..LEAF_PARAMS
    }
}

/// Where the spot light goes: four positions and then home, so the clip closes
/// on its opening frame and loops cleanly.
///
/// Only position is keyed. The cone direction is left pointing at its original
/// target because it changes nothing visible — the cone feeds the physical term
/// alone, and that term is zero at every one of these positions (the nearest,
/// the third, is 3421 units from the mark against a `distance` cutoff of 2521).
/// What the light does in this scene is aim the ramp, and the ramp reads
/// position only.
const SPOT_KEYS: [[f32; 3]; 5] = [
    [-1317.2792351911544, -5476.234252158654, 2832.0019076994704],
    [-4042.8866578062266, -1270.2884042234177, -798.8651297606409],
    [-2079.216456755957, 2368.1696703753196, 2843.1388646693026],
    [2163.777075056271, 3654.6562396309573, 1871.1778707384328],
    SPOT_POSITION,
];

fn spot_position_at(t: f32) -> [f32; 3] {
    let legs = [
        Leg {
            seconds: 1.0,
            easing: 4,
        },
        Leg {
            seconds: 2.0,
            easing: 0,
        },
        Leg {
            seconds: 3.0,
            easing: 0,
        },
        Leg {
            seconds: 3.0,
            easing: 0,
        },
        Leg {
            seconds: 3.0,
            easing: 0,
        },
        Leg {
            seconds: 3.0,
            easing: 4,
        },
    ];
    let (leg, k) = chain_at(&legs, t);
    if leg == 0 {
        return SPOT_POSITION;
    }
    let from = if leg == 1 {
        SPOT_POSITION
    } else {
        SPOT_KEYS[leg - 2]
    };
    lerp3(from, SPOT_KEYS[leg - 1], k)
}

fn neon_material(u: &MaterialParams) -> Material {
    let mut ramp = Vec::with_capacity(50);
    for c in RAMP {
        ramp.extend_from_slice(&c);
    }
    ramp.extend_from_slice(&RAMP_STOPS);
    Material::Shader(ShaderMaterial {
        storage0: ramp,
        transparent: false,
        ..ShaderMaterial::new(LOGO_FRAGMENT).with_data(vec![
            [
                ROUGHNESS,
                METALNESS,
                0.16 * SPECULAR_INTENSITY * SPECULAR_INTENSITY,
                LIGHT_ALPHA,
            ],
            [
                u.noise_scale,
                u.noise_size[0],
                u.noise_size[1],
                u.noise_size[2],
            ],
            // `nodeU9` move, then fA and the first half of fB.
            [u.noise_move, u.f_a[0], u.f_a[1], u.f_b[0]],
            [u.f_b[1], u.distortion[0], u.distortion[1], u.color_alpha],
            [u.colors[0][0], u.colors[0][1], u.colors[0][2], 1.0],
            [u.colors[1][0], u.colors[1][1], u.colors[1][2], 1.0],
            [u.colors[2][0], u.colors[2][1], u.colors[2][2], 1.0],
            [u.colors[3][0], u.colors[3][1], u.colors[3][2], 1.0],
            // Fallback spot position, used only if the scene has no spot light.
            [SPOT_POSITION[0], SPOT_POSITION[1], SPOT_POSITION[2], 0.0],
        ])
    })
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

/// Load `$NEON_MESH/<name>.obj` in place of sweeping the outline, if that
/// variable points at a directory holding `body.obj` and `leaf.obj`.
///
/// Here because the sweep has a ceiling worth being able to see past. It is a
/// model of the solid, not the solid, and its normals sit a few degrees off
/// whatever an authored mesh would carry — while the ramp's top four stops are
/// only 0.012-0.015 apart in `t`, which is about *two degrees* of normal
/// rotation each. Those thin white/violet/pink bands therefore land where the
/// swept normal puts them. Everything coarser than them — silhouette, framing,
/// the broad cyan of the edge, the dark face — is insensitive to it.
fn imported_mesh(name: &str) -> Option<BufferGeometry> {
    let dir = std::env::var("NEON_MESH").ok()?;
    let path = std::path::Path::new(&dir).join(format!("{name}.obj"));
    let src = std::fs::read_to_string(&path).ok()?;
    println!("  {name}: loaded {} ({} bytes)", path.display(), src.len());
    // OBJ rather than PLY on purpose: `ObjLoader` keeps the file's own `vn`
    // normals, and this whole scene is shaded off the normal. (`PlyLoader`'s
    // binary branch is also unimplemented as of this writing.)
    Some(threers::prelude::loaders::ObjLoader::parse(&src))
}

/// World direction for a three.js `Euler(x, y, z, "XYZ")`, in degrees.
///
/// three.js composes that order into a matrix whose third column is
/// `(sin y, -sin x cos y, cos x cos y)`; the camera looks down **-Z**, so the
/// forward vector is that column negated.
fn forward_from_euler_deg(r: [f32; 3]) -> Vector3 {
    let (x, y) = (r[0].to_radians(), r[1].to_radians());
    Vector3::new(-y.sin(), x.sin() * y.cos(), -(x.cos() * y.cos()))
}

fn build_camera(spec: &CameraSpec, view_w: f32, view_h: f32) -> OrthographicCamera {
    let half_w = (view_w / 2.0) / spec.zoom;
    let half_h = (view_h / 2.0) / spec.zoom;
    let mut camera = OrthographicCamera::new(-half_w, half_w, half_h, -half_h, NEAR, FAR);
    let eye = Vector3::new(spec.position[0], spec.position[1], spec.position[2]);
    let fwd = forward_from_euler_deg(spec.rotation_deg);
    camera.position = eye;
    camera.target = Vector3::new(
        eye.x + fwd.x * spec.target_offset,
        eye.y + fwd.y * spec.target_offset,
        eye.z + fwd.z * spec.target_offset,
    );
    println!(
        "  camera: {} — ortho frustum +-{half_w:.1} x +-{half_h:.1} world units \
         (viewport {view_w}x{view_h} px / zoom {:.4})",
        spec.name, spec.zoom
    );
    camera
}

/// Handles for the three things the timeline touches, so frames after the first
/// can move them instead of rebuilding the scene. The geometry is static — only
/// the two materials and the light's position change — and `Mesh::geometry` is an
/// `Arc` the renderer caches GPU buffers against, so leaving it alone keeps those
/// buffers alive across the whole clip.
struct Animated {
    body: ObjectId,
    leaf: ObjectId,
    spot: ObjectId,
}

fn build_scene(res: usize) -> (Scene, Animated) {
    let mut scene = Scene::new();
    // `Scene::new()` starts at 0x111111. Pure black matters here: the mark's
    // interior is opaque black, so anything lighter turns the silhouette into a
    // visible shape sitting on a lighter field.
    scene.background = Color::BLACK;

    // ---- lights -------------------------------------------------------
    // The spot contributes nothing at this range (see LOGO_FRAGMENT), but it is
    // the light the ramp layer measures its angle against, so it has to be here
    // and it has to be in the right place.
    let spot_pos = Vector3::new(SPOT_POSITION[0], SPOT_POSITION[1], SPOT_POSITION[2]);
    let spot_target = Vector3::new(SPOT_TARGET[0], SPOT_TARGET[1], SPOT_TARGET[2]);
    let mut spot = Object3D::light(SpotLight {
        color: Color::from_hex(SPOT_COLOR),
        intensity: SPOT_INTENSITY,
        direction: Vector3::new(
            spot_target.x - spot_pos.x,
            spot_target.y - spot_pos.y,
            spot_target.z - spot_pos.z,
        )
        .normalize(),
        distance: SPOT_DISTANCE,
        decay: SPOT_DECAY,
        angle: SPOT_ANGLE,
        penumbra: SPOT_PENUMBRA,
        cast_shadow: false,
        ..Default::default()
    });
    spot.position = spot_pos;
    spot.name = "Spot Light".into();
    let spot_id = scene.add(spot);

    scene.add_light(HemisphereLight::new(
        Color::from_hex(HEMI_SKY),
        Color::from_hex(HEMI_GROUND),
        HEMI_INTENSITY,
    ));

    // ---- the "Asset" group --------------------------------------------
    let mut group = Object3D::group();
    group.name = "Asset".into();
    group.position = Vector3::new(LOGO_POSITION[0], LOGO_POSITION[1], LOGO_POSITION[2]);
    group.quaternion = Quaternion::from_euler_xyz(0.0, LOGO_ROTATION_Y, 0.0);
    group.scale = Vector3::new(LOGO_SCALE, LOGO_SCALE, LOGO_SCALE);
    let group_id = scene.add(group);

    let body_outline = Outline::from_knots(&BODY_OUTLINE, OUTLINE_SUBDIV);
    let leaf_outline = Outline::from_knots(&LEAF_OUTLINE, OUTLINE_SUBDIV);

    // Counts triangles for indexed and non-indexed geometry alike: the rebuild
    // emits a triangle soup, an imported OBJ arrives indexed.
    let tris = |g: &BufferGeometry| match &g.index {
        Some(ix) => ix.len() / 3,
        None => g
            .get_attribute("position")
            .map(|a| a.count() / 3)
            .unwrap_or(0),
    };
    let body_geo = imported_mesh("body")
        .unwrap_or_else(|| inflated_geometry(&body_outline, BODY_INSET, BODY_HALF_Z, res));
    let leaf_geo = imported_mesh("leaf").unwrap_or_else(|| {
        inflated_geometry(&leaf_outline, LEAF_INSET, LEAF_HALF_Z, (res * 3) / 4)
    });
    println!("  body: {} triangles", tris(&body_geo));
    println!("  leaf: {} triangles", tris(&leaf_geo));

    let mut body = Object3D::mesh(Mesh::new(body_geo, neon_material(&BODY_PARAMS)));
    body.name = "body".into();
    body.scale = Vector3::new(1.0, 1.0, BODY_SCALE_Z);
    let body_id = scene.add_to(group_id, body);

    let mut leaf = Object3D::mesh(Mesh::new(leaf_geo, neon_material(&LEAF_PARAMS)));
    leaf.name = "leaf".into();
    leaf.position = Vector3::new(LEAF_POSITION[0], LEAF_POSITION[1], LEAF_POSITION[2]);
    let leaf_id = scene.add_to(group_id, leaf);

    (
        scene,
        Animated {
            body: body_id,
            leaf: leaf_id,
            spot: spot_id,
        },
    )
}

/// Move the scene to `t` seconds: two new materials and a light position.
fn seek(scene: &mut Scene, ids: &Animated, t: f32) {
    let set_material = |scene: &mut Scene, id: ObjectId, u: &MaterialParams| {
        if let Some(ObjectKind::Mesh(mesh)) = scene.get_mut(id).map(|o| &mut o.kind) {
            mesh.material = std::sync::Arc::new(neon_material(u));
        }
    };
    set_material(scene, ids.body, &body_params_at(t));
    set_material(scene, ids.leaf, &leaf_params_at(t));
    let p = spot_position_at(t);
    if let Some(obj) = scene.get_mut(ids.spot) {
        obj.position = Vector3::new(p[0], p[1], p[2]);
    }
}

// ---------------------------------------------------------------------------
// Post
// ---------------------------------------------------------------------------

/// The `postprocessing` library's luminance weights.
fn luminance(c: [f32; 3]) -> f32 {
    0.2125 * c[0] + 0.7154 * c[1] + 0.0721 * c[2]
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// The whole post chain: bloom, then saturation, then encode.
///
/// **All of it runs in linear light** — decode, grade, encode once at the end.
/// That is not a detail, and it is the easiest thing here to get quietly wrong,
/// because grading the encoded bytes looks fine on muted colours and falls apart
/// on saturated ones:
///
///     ramp stop 5, the cyan (0, 0.667, 0.961)
///       saturated in linear, then encoded -> (0, 236, 255)
///       encoded, then saturated in sRGB   -> (0, 255, 255)
///
/// The second pins green to full and turns the broad cyan sweep white, taking
/// the thin stops with it.
fn post(rgba: &mut [u8], w: u32, h: u32) {
    let (wi, hi) = (w as usize, h as usize);
    let n = wi * hi;

    let mut lin = vec![0.0f32; n * 3];
    for i in 0..n {
        for ch in 0..3 {
            lin[i * 3 + ch] = srgb_to_linear(rgba[i * 4 + ch] as f32 / 255.0);
        }
    }

    bloom(&mut lin, wi, hi);
    saturate(&mut lin, n);

    for i in 0..n {
        for ch in 0..3 {
            rgba[i * 4 + ch] =
                (linear_to_srgb(lin[i * 3 + ch].clamp(0.0, 1.0)) * 255.0).round() as u8;
        }
    }
}

/// `BloomEffect`: threshold the frame by luminance, blur, scale by `intensity`,
/// Screen the result back over the original.
///
/// The library blurs with a Kawase ladder; this is a separable Gaussian of the
/// same support instead, which is the one deliberate approximation in the post
/// chain — and at this file's `blurScale` the support is sub-pixel, so the blur
/// is skipped outright and the approximation costs nothing. What actually
/// shapes the result is the threshold, which decides *which* of the rim
/// survives into the lift.
fn bloom(lin: &mut [f32], wi: usize, hi: usize) {
    let n = wi * hi;

    // Bright pass. `luminanceSmoothing` widens the threshold into a smoothstep,
    // exactly as `LuminanceMaterial` does when `smoothing > 0`.
    let mut bright = vec![0.0f32; n * 3];
    for i in 0..n {
        let c = [lin[i * 3], lin[i * 3 + 1], lin[i * 3 + 2]];
        let l = luminance(c);
        let t = ((l - BLOOM_THRESHOLD) / BLOOM_SMOOTHING).clamp(0.0, 1.0);
        let k = t * t * (3.0 - 2.0 * t);
        for ch in 0..3 {
            bright[i * 3 + ch] = c[ch] * k;
        }
    }

    // Separable Gaussian standing in for the Kawase ladder: `2^kernelSize`
    // texels of a half-resolution buffer, scaled by `blurScale`. Here that is
    // 2^3 * 2 * 0.01 = 0.16 px, so the branch below skips the blur and the
    // bright pass is screened back unblurred.
    let sigma = 2.0f32.powi(BLOOM_KERNEL as i32) * 2.0 * BLOOM_BLUR_SCALE;
    if sigma >= 0.3 {
        let radius = (sigma * 3.0).ceil() as i32;
        let kernel: Vec<f32> = (-radius..=radius)
            .map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp())
            .collect();
        let ksum: f32 = kernel.iter().sum();
        let mut tmp = vec![0.0f32; n * 3];
        for y in 0..hi {
            for x in 0..wi {
                let mut acc = [0.0f32; 3];
                for (ki, kv) in kernel.iter().enumerate() {
                    let sx = (x as i32 + ki as i32 - radius).clamp(0, wi as i32 - 1) as usize;
                    for ch in 0..3 {
                        acc[ch] += bright[(y * wi + sx) * 3 + ch] * kv;
                    }
                }
                for ch in 0..3 {
                    tmp[(y * wi + x) * 3 + ch] = acc[ch] / ksum;
                }
            }
        }
        for x in 0..wi {
            for y in 0..hi {
                let mut acc = [0.0f32; 3];
                for (ki, kv) in kernel.iter().enumerate() {
                    let sy = (y as i32 + ki as i32 - radius).clamp(0, hi as i32 - 1) as usize;
                    for ch in 0..3 {
                        acc[ch] += tmp[(sy * wi + x) * 3 + ch] * kv;
                    }
                }
                for ch in 0..3 {
                    bright[(y * wi + x) * 3 + ch] = acc[ch] / ksum;
                }
            }
        }
    }

    // Screen the bright pass back over the frame — blend function 16 at
    // `opacity: 1`, scaled by `intensity`.
    for i in 0..n {
        for ch in 0..3 {
            let base = lin[i * 3 + ch];
            let add = (bright[i * 3 + ch] * BLOOM_INTENSITY).clamp(0.0, 1.0);
            lin[i * 3 + ch] = 1.0 - (1.0 - base) * (1.0 - add);
        }
    }
}

/// `HueSaturationEffect` with `hue: 0`, so only the saturation half runs. The
/// library's own formula, which is not a plain lerp toward grey:
///
///     average = (r + g + b) / 3
///     diff    = average - colour
///     colour += diff * (1 - 1 / (1.001 - saturation))      // saturation > 0
fn saturate(lin: &mut [f32], n: usize) {
    let k = 1.0 - 1.0 / (1.001 - SATURATION);
    for i in 0..n {
        let c = [lin[i * 3], lin[i * 3 + 1], lin[i * 3 + 2]];
        let average = (c[0] + c[1] + c[2]) / 3.0;
        for ch in 0..3 {
            lin[i * 3 + ch] = (c[ch] + (average - c[ch]) * k).clamp(0.0, 1.0);
        }
    }
}

fn main() {
    let _ = std::fs::create_dir_all("out");

    let spec = match std::env::var("NEON_CAMERA").as_deref() {
        Ok("tilt") => &CAMERA_TILT,
        Ok("close") => &CAMERA_CLOSE,
        _ => &CAMERA_HERO,
    };

    // Stretching the frustum against a different output aspect would shear the
    // composition, so say so rather than quietly producing a wrong picture.
    let (view_w, view_h) = viewport();
    let (w, h) = if (view_w, view_h) == (VIEW_W, VIEW_H) {
        (W, H)
    } else {
        (view_w as u32, view_h as u32)
    };
    let view_aspect = view_w / view_h;
    let out_aspect = w as f32 / h as f32;
    assert!(
        (view_aspect - out_aspect).abs() / view_aspect < 0.002,
        "output {w}x{h} (aspect {out_aspect:.5}) does not match the viewport \
         {view_w}x{view_h} (aspect {view_aspect:.5}); the frustum is in pixels, \
         so these have to agree"
    );

    let res: usize = std::env::var("NEON_RES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);

    let camera = build_camera(spec, view_w, view_h);
    let (mut scene, ids) = build_scene(res);

    let mut renderer = HeadlessRenderer::builder()
        .size(w, h)
        .build()
        .expect("headless renderer (is a GPU adapter available?)");

    let do_post = std::env::var("NEON_NO_POST").is_err();

    // `NEON_ANIM_FPS=n` renders the Start-event timeline instead of the still:
    // 15 seconds, which is one full pass of the spot light's chain and the
    // duration the scene's own export settings name.
    if let Some(fps) = std::env::var("NEON_ANIM_FPS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .or_else(|| std::env::var("NEON_ANIM").ok().map(|_| ANIM_FPS))
    {
        let fps = fps.max(1);
        let total = (ANIM_SECONDS * fps as f32).round() as u32;
        let dir = "out/neon_logo_anim";
        let _ = std::fs::create_dir_all(dir);
        println!("  timeline: {ANIM_SECONDS}s at {fps} fps = {total} frames");
        for i in 0..total {
            let t = i as f32 / fps as f32;
            seek(&mut scene, &ids, t);
            let mut rgba = renderer.render_to_rgba(&mut scene, &camera);
            if do_post {
                post(&mut rgba, w, h);
            }
            std::fs::write(format!("{dir}/frame_{i:04}.png"), encode_png(w, h, &rgba))
                .expect("write png");
            if i % (fps.max(1)) == 0 {
                let p = spot_position_at(t);
                println!(
                    "    t={t:5.2}s  spot=({:8.1},{:8.1},{:8.1})  body.scale={:5.2}",
                    p[0],
                    p[1],
                    p[2],
                    body_params_at(t).noise_scale
                );
            }
        }
        println!("wrote {total} frames to {dir}/");
        encode_clip(dir, total, w, h, fps);
        return;
    }

    seek(&mut scene, &ids, 0.0);
    let mut rgba = renderer.render_to_rgba(&mut scene, &camera);

    let lit = rgba
        .chunks_exact(4)
        .filter(|p| p[0] as u32 + p[1] as u32 + p[2] as u32 > 24)
        .count();
    println!(
        "  render: {w}x{h}, {:.2}% of pixels above black",
        100.0 * lit as f32 / (w * h) as f32
    );

    if do_post {
        post(&mut rgba, w, h);
    }

    let path = "out/neon_logo.png";
    std::fs::write(path, encode_png(w, h, &rgba)).expect("write png");
    println!("wrote {path}");
}

/// Mux the frames into the container the scene asks for — `publish.settings.video`
/// says `mp4` — using this crate's own H.264 encoder. Needs the `native-codec`
/// feature; without it the PNG sequence is the output and ffmpeg can take it
/// from there.
///
/// Frames are read back off disk one at a time rather than kept from the render
/// loop: 450 of them at this size is 3.4 GB resident, and the encoder only ever
/// looks at one.
#[cfg(feature = "native-codec")]
fn encode_clip(dir: &str, total: u32, w: u32, h: u32, fps: u32) {
    use threers::{encode_animation_rgba, AnimationEncodeOptions, BrowserCodec};
    // H.264 chroma is subsampled 2x2, so both dimensions have to be even. This
    // scene's frustum is 1248x929 — odd — so the odd edge is dropped rather than
    // rescaled, which would resample every pixel to fix one row.
    let (cw, ch) = (w & !1, h & !1);
    if (cw, ch) != (w, h) {
        println!("  mp4: cropping {w}x{h} to {cw}x{ch} (H.264 needs even dimensions)");
    }
    let frames = (0..total).map(|i| {
        let bytes = std::fs::read(format!("{dir}/frame_{i:04}.png")).expect("read frame");
        let img = decode_png(&bytes).expect("decode frame");
        if (cw, ch) == (w, h) {
            return img.rgba;
        }
        let mut out = Vec::with_capacity((cw * ch * 4) as usize);
        for y in 0..ch {
            let row = (y * w * 4) as usize;
            out.extend_from_slice(&img.rgba[row..row + (cw * 4) as usize]);
        }
        out
    });
    let opts = AnimationEncodeOptions {
        width: cw,
        height: ch,
        fps,
        codec: BrowserCodec::Mp4,
        ..Default::default()
    };
    match encode_animation_rgba(&opts, frames) {
        Ok(bytes) => {
            let path = "out/neon_logo.mp4";
            std::fs::write(path, &bytes).expect("write mp4");
            println!("wrote {path} ({:.1} MB)", bytes.len() as f32 / 1e6);
        }
        Err(e) => println!("  mp4 encode failed: {e}"),
    }
}

#[cfg(not(feature = "native-codec"))]
fn encode_clip(_dir: &str, _total: u32, _w: u32, _h: u32, _fps: u32) {
    println!("  (build with --features native-codec to also write out/neon_logo.mp4)");
}
