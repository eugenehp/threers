//! Data derived from a [`BufferGeometry`] on first use and kept until the
//! geometry changes: the bounding volumes a raycast needs when the geometry
//! carries none, and the polygon prefix sums behind
//! [`BufferGeometry::polygon_of_triangle`].
//!
//! A geometry sits behind an `Arc` and is read, not written, while it is
//! raycast, so the cache fills through `&self`. Each entry is stamped with the
//! geometry's [`geometry_version`](BufferGeometry::geometry_version) and the
//! lengths of the position, index and polygon arrays, and an entry whose
//! stamp no longer matches is rebuilt.
//!
//! The contract it keeps is on [`DerivedCache`].

use super::BufferGeometry;
use crate::math::{Box3, Sphere};
use std::sync::{Arc, RwLock};

/// The lazily filled cache on `BufferGeometry::derived`. Opaque: it only
/// exists so that the geometry can stay `Clone + Default + Debug`. A clone
/// starts empty.
///
/// # The contract
///
/// This is the renderer's contract, not a weaker one: the GPU upload is also
/// keyed on `geometry_version`, so a change the version does not record is a
/// change the screen does not show either.
///
/// - Every mutator ([`set_attribute`](BufferGeometry::set_attribute),
///   [`set_index`](BufferGeometry::set_index),
///   [`set_polygons`](BufferGeometry::set_polygons)) bumps the version, so the
///   cache follows them.
/// - A direct write to the public `attributes` / `index` / `polygons` fields
///   is seen **only** if it is followed by
///   [`bump_version`](BufferGeometry::bump_version) — three.js's
///   `needsUpdate = true`. A write that changes an array's length is caught
///   anyway, as a convenience; one that keeps the length is not, whether it
///   edits values in place or swaps in a new array (the allocator readily
///   hands a freed address back, so an address cannot tell two arrays apart).
/// - Bounds follow three.js: a stored
///   [`bounding_sphere`](BufferGeometry::bounding_sphere) /
///   [`bounding_box`](BufferGeometry::bounding_box) is used as it is, stale or
///   not, exactly as three.js uses `geometry.boundingSphere` once set; only
///   while the field is `None` is the cached one used.
#[derive(Default)]
pub struct DerivedCache {
    slot: RwLock<Option<Arc<Derived>>>,
}

impl Clone for DerivedCache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl std::fmt::Debug for DerivedCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let filled = self.slot.read().map(|s| s.is_some()).unwrap_or(false);
        f.debug_struct("DerivedCache")
            .field("filled", &filled)
            .finish()
    }
}

impl DerivedCache {
    /// Drop whatever is cached. Not needed after a version bump — a stale
    /// entry is rebuilt on its next read — but frees the memory at once.
    pub fn clear(&mut self) {
        *self.slot.get_mut().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

/// What a cache entry was computed from: the version, and — as a cheap extra —
/// the array lengths. No addresses: a replaced array often lands at the
/// address of the one it replaced, so an address match proves nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Stamp {
    version: u32,
    position: usize,
    index: usize,
    polygons: usize,
}

impl Stamp {
    fn of(g: &BufferGeometry) -> Self {
        let len = |s: Option<&[u32]>| s.map_or(usize::MAX, <[u32]>::len);
        Self {
            version: g.geometry_version,
            position: g
                .attributes
                .get("position")
                .map_or(usize::MAX, |a| a.array.len()),
            index: len(g.index.as_deref()),
            polygons: len(g.polygons.as_deref()),
        }
    }
}

/// Bounding volumes and polygon lookup, computed together on first use.
#[derive(Debug)]
pub(crate) struct Derived {
    stamp: Stamp,
    /// three.js `computeBoundingBox` over the positions; empty without any.
    pub bounding_box: Box3,
    /// three.js `computeBoundingSphere`; empty (negative radius) without
    /// positions.
    pub bounding_sphere: Sphere,
    /// `ends[p]` is one past polygon `p`'s last triangle. `None` without a
    /// grouping, or with one that does not cover exactly this geometry's
    /// triangles (the [`BufferGeometry::polygon_counts`] rule).
    polygon_ends: Option<Vec<u32>>,
}

impl Derived {
    fn compute(g: &BufferGeometry) -> Self {
        let mut bounding_box = Box3::empty();
        if let Some(points) = g.positions() {
            for p in points {
                bounding_box.expand_by_point(p);
            }
        }
        let bounding_sphere = if bounding_box.is_empty() {
            Sphere::empty()
        } else {
            let center = bounding_box.center();
            let mut max_r2 = 0.0f32;
            if let Some(points) = g.positions() {
                for p in points {
                    max_r2 = max_r2.max((p - center).length_sq());
                }
            }
            Sphere::new(center, max_r2.sqrt())
        };
        let polygon_ends = g.polygons.as_deref().and_then(|counts| {
            let mut ends = Vec::with_capacity(counts.len());
            let mut at = 0u64;
            for &n in counts {
                at += n as u64;
                ends.push(u32::try_from(at).ok()?);
            }
            (at == g.triangle_count() as u64).then_some(ends)
        });
        Self {
            stamp: Stamp::of(g),
            bounding_box,
            bounding_sphere,
            polygon_ends,
        }
    }

    /// Which polygon triangle `t` belongs to: a binary search of the prefix
    /// sums.
    pub fn polygon_of(&self, t: usize) -> Option<usize> {
        let ends = self.polygon_ends.as_deref()?;
        let p = ends.partition_point(|&end| end as usize <= t);
        (p < ends.len()).then_some(p)
    }
}

impl BufferGeometry {
    /// The derived-data entry for the geometry as it is now, computing it if
    /// the cache is empty or stale.
    pub(crate) fn derived(&self) -> Arc<Derived> {
        let stamp = Stamp::of(self);
        let lock = &self.derived.slot;
        if let Some(d) = lock.read().unwrap_or_else(|e| e.into_inner()).as_ref() {
            if d.stamp == stamp {
                return d.clone();
            }
        }
        let fresh = Arc::new(Derived::compute(self));
        *lock.write().unwrap_or_else(|e| e.into_inner()) = Some(fresh.clone());
        fresh
    }

    /// Which polygon of the grouping ([`polygons`](Self::polygons)) triangle
    /// `t` belongs to, or `None` without a (valid) grouping or past the end.
    ///
    /// The first call builds a prefix sum of the grouping and caches it on the
    /// geometry; every later call is a binary search, `O(log polygons)`. The
    /// cache follows the geometry through every mutator, and through a direct
    /// write to the public fields once it is followed by
    /// [`bump_version`](Self::bump_version) — see [`DerivedCache`].
    pub fn polygon_of_triangle(&self, t: usize) -> Option<usize> {
        self.polygons.as_ref()?;
        self.derived().polygon_of(t)
    }

    /// The geometry's bounding sphere: the stored
    /// [`bounding_sphere`](Self::bounding_sphere) when there is one — used as
    /// is, as three.js uses `geometry.boundingSphere` — else one computed the
    /// same way as [`compute_bounding_sphere`](Self::compute_bounding_sphere)
    /// and cached. Empty (negative radius) without positions.
    pub fn bounding_sphere_cached(&self) -> Sphere {
        self.bounding_sphere
            .unwrap_or_else(|| self.derived().bounding_sphere)
    }

    /// The geometry's bounding box, stored or cached like
    /// [`bounding_sphere_cached`](Self::bounding_sphere_cached).
    pub fn bounding_box_cached(&self) -> Box3 {
        self.bounding_box
            .unwrap_or_else(|| self.derived().bounding_box)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BufferAttribute;

    /// Four quads in a row, then the polygon table checked triangle by
    /// triangle against the linear walk it replaces.
    fn strip() -> BufferGeometry {
        crate::geometries::PlaneGeometry::with_segments(4.0, 1.0, 4, 1)
    }

    fn linear(g: &BufferGeometry, t: usize) -> Option<usize> {
        g.polygons_of_triangles(&[t])[0]
    }

    #[test]
    fn lookup_matches_the_linear_walk() {
        let mut g = strip();
        g.set_polygons(vec![1, 3, 2, 1, 1]);
        for t in 0..g.triangle_count() + 2 {
            assert_eq!(g.polygon_of_triangle(t), linear(&g, t), "triangle {t}");
        }
        assert_eq!(g.polygon_of_triangle(0), Some(0));
        assert_eq!(g.polygon_of_triangle(3), Some(1));
        assert_eq!(g.polygon_of_triangle(4), Some(2));
        assert_eq!(g.polygon_of_triangle(7), Some(4));
        assert_eq!(g.polygon_of_triangle(8), None);
    }

    #[test]
    fn set_polygons_refreshes_the_cache() {
        let mut g = strip();
        assert_eq!(g.polygon_of_triangle(2), Some(1));
        g.set_polygons(vec![4, 4]);
        assert_eq!(g.polygon_of_triangle(2), Some(0));
        assert_eq!(g.polygon_of_triangle(5), Some(1));
    }

    #[test]
    fn set_index_drops_the_grouping() {
        let mut g = strip();
        assert_eq!(g.polygon_of_triangle(2), Some(1));
        let index = g.index.clone().unwrap();
        g.set_index(index);
        assert_eq!(g.polygon_of_triangle(2), None);
    }

    /// The contract: a direct write to a public field is seen once the
    /// version is bumped. A length change is caught even without the bump.
    #[test]
    fn direct_writes_are_seen_after_a_version_bump() {
        let mut g = strip();
        assert_eq!(g.polygon_of_triangle(6), Some(3));
        // Same length: invisible until the bump, whatever the address.
        g.polygons = Some(vec![8, 0, 0, 0]);
        g.bump_version();
        assert_eq!(g.polygon_of_triangle(6), Some(0));
        // A table that no longer covers the triangles is refused.
        g.polygons = Some(vec![2, 2]);
        assert_eq!(g.polygon_of_triangle(1), None, "length change");
        g.polygons = Some(vec![2, 2, 2, 2]);
        assert_eq!(g.polygon_of_triangle(5), Some(2), "length change");
        // Dropping triangles from the index invalidates the table.
        let mut index = g.index.clone().unwrap();
        index.truncate(12);
        g.index = Some(index);
        assert_eq!(g.polygon_of_triangle(1), None, "length change");
    }

    /// An in-place edit of the values keeps the address and the length, and
    /// the bump is what makes it visible — to this cache and to the renderer.
    #[test]
    fn an_in_place_edit_needs_the_bump() {
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 3),
        );
        assert_eq!(g.bounding_box_cached().max.x, 1.0);
        g.attributes.get_mut("position").unwrap().array[3] = 5.0;
        let v = g.geometry_version;
        g.bump_version();
        assert_ne!(g.geometry_version, v);
        assert_eq!(g.bounding_box_cached().max.x, 5.0);
    }

    /// Replacing an array with a new one of the same length, then bumping:
    /// refreshed every time, whatever address the allocator hands back.
    #[test]
    fn a_same_length_replacement_with_a_bump_always_refreshes() {
        let mut g = strip();
        for trial in 0..200u32 {
            let first = 1 + trial % 7;
            let table = vec![first, 8 - first];
            g.polygons = Some(table);
            g.bump_version();
            assert_eq!(g.polygon_of_triangle(0), Some(0));
            assert_eq!(
                g.polygon_of_triangle(first as usize),
                Some(1),
                "trial {trial}"
            );
        }
    }

    /// A stored bounding volume is three.js's `geometry.boundingSphere`: used
    /// as is, even when the positions have moved on, until it is cleared or
    /// recomputed.
    #[test]
    fn stored_bounds_are_used_as_is() {
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 3),
        );
        g.bounding_sphere = Some(Sphere::new(crate::math::Vector3::ZERO, 42.0));
        assert_eq!(g.bounding_sphere_cached().radius, 42.0);
        g.bounding_sphere = None;
        assert!((g.bounding_sphere_cached().radius - 0.5f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn bounds_follow_set_attribute() {
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0], 3),
        );
        let s = g.bounding_sphere_cached();
        assert!((s.radius - 2f32.sqrt()).abs() < 1e-6);
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0, 0.0], 3),
        );
        let s = g.bounding_sphere_cached();
        assert!((s.radius - 8f32.sqrt()).abs() < 1e-6, "{s:?}");
        assert_eq!(g.bounding_box_cached().max.x, 4.0);
        // Positions replaced by a direct write of a new array, then the bump.
        g.attributes.insert(
            "position".into(),
            BufferAttribute::new(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 3),
        );
        g.bump_version();
        assert_eq!(g.bounding_box_cached().max.x, 1.0);
    }
}
