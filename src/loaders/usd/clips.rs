//! Value clips — animation streamed from a sequence of files.
//!
//! A film shot does not hold its animation in the layer that describes it. The
//! layer says *where the animation lives*: a list of files, which of them is
//! active over which frames, and how stage time maps onto the time inside
//! them. That is a value clip, and it is how a crowd or an effects cache gets
//! onto a stage without any one file having to hold all of it.
//!
//! ```text
//! clips = {
//!     dictionary default = {
//!         asset[] assetPaths = [@cache.0.usd@, @cache.1.usd@]
//!         string primPath = "/Cache"
//!         double2[] active = [(0, 0), (2, 1)]      # frame 0 → clip 0, 2 → clip 1
//!         double2[] times  = [(0, 0), (1, 1), ...] # stage frame → clip frame
//!     }
//! }
//! ```
//!
//! # Two things that are easy to get wrong
//!
//! Clips supply values for attributes that **already exist** on the composed
//! prim; they do not bring attributes into being. A prim with no
//! `xformOp:translate` declared stays without one however many clips name it,
//! which is why the manifest exists and why a topology layer usually sits
//! underneath.
//!
//! And the two tables do different jobs: `active` picks *which file*, `times`
//! picks *when inside it*. Reading either for the other's purpose gives an
//! animation that plays, and plays wrongly.

use super::parse::{UsdLayer, UsdPrim};
use super::value::UsdValue;

/// One clip set off a prim's `clips` metadata.
#[derive(Debug, Default, Clone)]
pub struct ClipSet {
    pub name: String,
    /// The files, in the order `active` indexes them by.
    pub assets: Vec<String>,
    /// Where inside each file to look.
    pub prim_path: String,
    /// `(stage time, clip index)`, in ascending stage time.
    pub active: Vec<(f64, usize)>,
    /// `(stage time, clip time)`, the map from one time line to the other.
    pub times: Vec<(f64, f64)>,
    /// The layer naming which attributes the clips animate.
    pub manifest: String,
}

impl ClipSet {
    /// Which file is active at a stage time.
    ///
    /// The table is a step function: an entry takes effect at its time and
    /// holds until the next one, so the answer is the last entry at or before
    /// the time asked about.
    pub fn clip_at(&self, time: f64) -> Option<usize> {
        if self.active.is_empty() {
            // With no table at all there is one clip and it is always on.
            return (!self.assets.is_empty()).then_some(0);
        }
        let mut chosen = self.active.first()?.1;
        for (at, index) in &self.active {
            if *at <= time {
                chosen = *index;
            } else {
                break;
            }
        }
        Some(chosen)
    }
}

/// Read the clip sets a prim declares.
///
/// USD has two spellings: a `clips` dictionary keyed by set name, and the older
/// flat `clipAssetPaths` / `clipPrimPath` / `clipActive` / `clipTimes`. Both
/// appear in the wild, and a reader that knows only the modern one silently
/// ignores half the caches ever written.
pub fn sets_of(prim: &UsdPrim) -> Vec<ClipSet> {
    let mut out = Vec::new();

    if let Some(UsdValue::Dict(entries)) = prim.meta("clips") {
        for (name, value) in entries {
            if let UsdValue::Dict(fields) = value {
                out.push(read_set(name, fields));
            }
        }
    }

    // The flat form, which is one unnamed set.
    if out.is_empty() {
        let flat: Vec<(String, UsdValue)> = [
            ("assetPaths", "clipAssetPaths"),
            ("primPath", "clipPrimPath"),
            ("active", "clipActive"),
            ("times", "clipTimes"),
            ("manifestAssetPath", "clipManifestAssetPath"),
        ]
        .iter()
        .filter_map(|(to, from)| prim.meta(from).map(|v| (to.to_string(), v.clone())))
        .collect();
        if !flat.is_empty() {
            out.push(read_set("default", &flat));
        }
    }
    out
}

fn read_set(name: &str, fields: &[(String, UsdValue)]) -> ClipSet {
    let field = |key: &str| fields.iter().find(|(k, _)| k == key).map(|(_, v)| v);
    let pairs = |key: &str| -> Vec<(f64, f64)> {
        let Some(value) = field(key) else {
            return Vec::new();
        };
        let flat = value.flat_f32();
        flat.chunks_exact(2)
            .map(|p| (p[0] as f64, p[1] as f64))
            .collect()
    };

    ClipSet {
        name: name.to_string(),
        assets: field("assetPaths")
            .map(|v| match v {
                UsdValue::Array(items) => items
                    .iter()
                    .filter_map(|i| i.as_str())
                    .map(str::to_string)
                    .collect(),
                other => other.as_str().map(str::to_string).into_iter().collect(),
            })
            .unwrap_or_default(),
        prim_path: field("primPath")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        active: pairs("active")
            .into_iter()
            .map(|(at, index)| (at, index.max(0.0) as usize))
            .collect(),
        times: pairs("times"),
        manifest: field("manifestAssetPath")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
    }
}

/// Which attributes the clips are allowed to animate.
///
/// The manifest is a layer whose prims declare the attributes, with no values.
/// Without one every attribute on the prim is a candidate, which is what USD
/// falls back to and is only slower rather than wrong.
pub fn animated_names(manifest: Option<&UsdLayer>, prim_path: &str, prim: &UsdPrim) -> Vec<String> {
    if let Some(manifest) = manifest {
        if let Some(found) = manifest.prim_at(prim_path) {
            let names: Vec<String> = found.properties.iter().map(|p| p.name.clone()).collect();
            if !names.is_empty() {
                return names;
            }
        }
    }
    prim.properties.iter().map(|p| p.name.clone()).collect()
}

/// The stage times a clip set produces samples at, and the clip time each maps
/// to.
///
/// With a `times` table those pairs *are* the answer. Without one, stage time
/// and clip time are the same thing, and the samples are whatever the clips
/// themselves hold — so the times have to be gathered from them.
pub fn sample_times(set: &ClipSet, from_clips: impl Fn(usize) -> Vec<f64>) -> Vec<(f64, f64)> {
    if !set.times.is_empty() {
        let mut out = set.times.clone();
        out.sort_by(|a, b| a.0.total_cmp(&b.0));
        return out;
    }
    let mut out: Vec<(f64, f64)> = Vec::new();
    for index in 0..set.assets.len() {
        for time in from_clips(index) {
            // Only where this clip is the active one, or two clips covering
            // the same frame would each contribute a sample to it.
            if set.clip_at(time) == Some(index) && !out.iter().any(|(t, _)| *t == time) {
                out.push((time, time));
            }
        }
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::compose::{compose, ComposeOptions, MemoryResolver};
    use super::super::parse::parse;

    fn pipeline() -> MemoryResolver {
        let mut r = MemoryResolver::new();
        r.insert("clip_0.usda", include_str!("testdata/clips/clip_0.usda"))
            .insert("clip_1.usda", include_str!("testdata/clips/clip_1.usda"))
            .insert("manifest.usda", include_str!("testdata/clips/manifest.usda"));
        r
    }

    /// The whole feature, against `usdcat --flatten` on the same files.
    ///
    /// Two tables do different jobs: `active` picks which *file*, `times`
    /// picks *when inside it*. Reading either for the other's purpose gives an
    /// animation that plays, and plays wrongly — which is why the check is
    /// against OpenUSD's own answer rather than against a reading of the spec.
    #[test]
    fn clips_resolve_to_the_same_samples_openusd_produces() {
        let stage = parse(include_str!("testdata/clips/stage.usda")).expect("parses");
        let out = compose(&stage, "stage.usda", &pipeline(), &ComposeOptions::default())
            .expect("composes");

        let mine = out
            .prim_at("/Shot")
            .expect("the shot")
            .value("xformOp:translate")
            .expect("the clipped attribute")
            .samples()
            .expect("clips should have made it animated")
            .to_vec();

        let theirs = parse(include_str!("testdata/clips/flat.usda")).unwrap();
        let expected = theirs
            .prim_at("/Shot")
            .unwrap()
            .value("xformOp:translate")
            .unwrap()
            .samples()
            .unwrap();

        assert_eq!(mine.len(), expected.len(), "sample count");
        for (mine, theirs) in mine.iter().zip(expected) {
            assert_eq!(mine.0, theirs.0, "sample time");
            assert_eq!(mine.1.flat_f32(), theirs.1.flat_f32(), "at {}", mine.0);
        }

        // Spelled out: the first two frames come from clip 0 and the next two
        // from clip 1, each at its own clip-local time.
        assert_eq!(mine[0].1.flat_f32(), vec![0.0, 0.0, 0.0]);
        assert_eq!(mine[1].1.flat_f32(), vec![0.0, 1.0, 0.0]);
        assert_eq!(mine[2].1.flat_f32(), vec![1.0, 0.0, 0.0]);
        assert_eq!(mine[3].1.flat_f32(), vec![1.0, 1.0, 0.0]);
    }

    /// `active` is a step function: an entry holds until the next one.
    #[test]
    fn the_active_table_holds_between_entries() {
        let set = ClipSet {
            assets: vec!["a".into(), "b".into(), "c".into()],
            active: vec![(0.0, 0), (10.0, 2), (20.0, 1)],
            ..Default::default()
        };
        assert_eq!(set.clip_at(-5.0), Some(0), "before the first, the first");
        assert_eq!(set.clip_at(0.0), Some(0));
        assert_eq!(set.clip_at(9.9), Some(0), "holds until the next entry");
        assert_eq!(set.clip_at(10.0), Some(2));
        assert_eq!(set.clip_at(19.0), Some(2));
        assert_eq!(set.clip_at(1000.0), Some(1), "the last one holds forever");
    }

    /// With one clip and no table, it is always the active one.
    #[test]
    fn a_single_clip_needs_no_table() {
        let set = ClipSet {
            assets: vec!["only".into()],
            ..Default::default()
        };
        assert_eq!(set.clip_at(42.0), Some(0));
        assert_eq!(ClipSet::default().clip_at(0.0), None, "no clips, no answer");
    }

    /// Without a `times` table, stage time and clip time are the same and the
    /// samples come from the clips themselves.
    #[test]
    fn missing_times_fall_back_to_the_clips_own_samples() {
        let set = ClipSet {
            assets: vec!["a".into(), "b".into()],
            active: vec![(0.0, 0), (2.0, 1)],
            ..Default::default()
        };
        let schedule = sample_times(&set, |index| match index {
            0 => vec![0.0, 1.0, 2.0],
            _ => vec![2.0, 3.0],
            });
        // Frame 2 belongs to clip 1, so clip 0's sample there is not used.
        assert_eq!(schedule, vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]);
    }

    /// The older flat spelling is the same set under different names, and
    /// plenty of caches were written with it.
    #[test]
    fn the_flat_spelling_is_read_too() {
        let layer = parse(
            r#"#usda 1.0
def Xform "Old" (
    clipAssetPaths = [@a.usda@, @b.usda@]
    clipPrimPath = "/Cache"
    clipActive = [(0, 0), (5, 1)]
    clipTimes = [(0, 0), (5, 0)]
)
{
    double3 xformOp:translate
}
"#,
        )
        .unwrap();
        let sets = sets_of(layer.prim_at("/Old").unwrap());
        assert_eq!(sets.len(), 1);
        assert_eq!(sets[0].assets, vec!["a.usda", "b.usda"]);
        assert_eq!(sets[0].prim_path, "/Cache");
        assert_eq!(sets[0].active, vec![(0.0, 0), (5.0, 1)]);
    }

    /// Clips supply values for attributes that already exist; they do not
    /// bring attributes into being. This is the behaviour OpenUSD showed when
    /// the same stage, minus the declaration, flattened to nothing at all.
    #[test]
    fn clips_do_not_invent_attributes() {
        let stage = parse(
            &include_str!("testdata/clips/stage.usda")
                .replace("    double3 xformOp:translate\n", ""),
        )
        .unwrap();
        let out = compose(&stage, "stage.usda", &pipeline(), &ComposeOptions::default()).unwrap();
        assert!(
            out.prim_at("/Shot").unwrap().value("xformOp:translate").is_none(),
            "an undeclared attribute stays undeclared"
        );
    }
}
