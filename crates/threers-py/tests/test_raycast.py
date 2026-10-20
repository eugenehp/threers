"""Scene.raycast / raycast_from_camera: three.js's intersection fields (in
snake_case), material sides, line thresholds, handle lists, recursion,
near/far and first_hit_only — run after `maturin develop`."""

import pytest

import threers
from threers import Geometry, Material

DOWN = (0, 0, -1)


def close(a, b, eps=1e-4):
    return abs(a - b) < eps


def test_a_mesh_hit_carries_three_js_fields():
    s = threers.Scene()
    h = s.add_mesh(Geometry.plane(2, 2), Material.basic())
    (hit,) = s.raycast((0.5, -0.25, 5), DOWN)
    assert hit["handle"] == h
    assert close(hit["distance"], 5)
    assert hit["face_index"] in (0, 1)
    face = hit["face"]
    assert all(0 <= face[k] < 4 for k in "abc")
    assert face["material_index"] == 0
    assert close(face["normal"][2], 1), "geometric normal"
    assert close(hit["normal"][2], 1), "normal attribute, facing the ray"
    assert close(hit["uv"][0], 0.75) and close(hit["uv"][1], 0.375), hit["uv"]
    assert close(sum(hit["barycoord"]), 1)
    assert hit["polygon_index"] == 0
    for k in ("uv1", "instance_id", "index", "distance_to_ray"):
        assert hit[k] is None, k


def test_material_sides_cull_as_three_js_does():
    s = threers.Scene()
    front = s.add_mesh(Geometry.plane(1, 1), Material.basic())
    back = s.add_mesh(Geometry.plane(1, 1), Material.basic(side="back"))
    both = s.add_mesh(Geometry.plane(1, 1), Material.basic(side="double"))
    # Off the plane's diagonal: a ray along a shared edge hits both triangles.
    from_front = sorted(h["handle"] for h in s.raycast((0.1, 0.3, 5), DOWN))
    from_back = sorted(h["handle"] for h in s.raycast((0.1, 0.3, -5), (0, 0, 1)))
    assert from_front == sorted([front, both])
    assert from_back == sorted([back, both])
    (behind,) = s.raycast((0.1, 0.3, -5), (0, 0, 1), objects=[both])
    assert close(behind["normal"][2], -1), "the normal faces the ray"


def test_line_segments_hit_within_the_threshold():
    s = threers.Scene()
    # The outline of a 2 x 0.6 rectangle: edges at y = +-0.3, x = +-1.
    line = s.add_line_segments(Geometry.edges(Geometry.plane(2, 0.6)), Material.line())
    at = (0.2, 0.35, 5)
    hits = s.raycast(at, DOWN, line_threshold=0.1)
    assert len(hits) == 1, "only the top edge is that close"
    hit = hits[0]
    assert hit["handle"] == line
    assert isinstance(hit["index"], int)
    assert close(hit["point"][1], 0.3), "the point is on the segment"
    for k in ("face", "face_index", "uv", "normal", "barycoord"):
        assert hit[k] is None, k
    assert s.raycast(at, DOWN, line_threshold=0.01) == []
    assert len(s.raycast((0.2, 1.2, 5), DOWN)) == 1, "three.js's default threshold is 1"
    with pytest.raises(ValueError, match="line_threshold"):
        s.raycast(at, DOWN, line_threshold=-1)


def test_objects_recursive_near_far_and_first_hit_only():
    s = threers.Scene()
    group = s.add_group()
    planes = []
    for z in (0, -1, -2):
        h = s.add_mesh(Geometry.plane(1, 1), Material.basic(side="double"), 0, 0, z)
        s.set_parent(h, group)
        planes.append(h)
    other = s.add_mesh(Geometry.plane(1, 1), Material.basic(), 0, 0, -3)
    o = (0.1, 0.3, 5)
    handles = lambda hits: [h["handle"] for h in hits]  # noqa: E731
    assert handles(s.raycast(o, DOWN)) == [*planes, other], "nearest first"
    assert handles(s.raycast(o, DOWN, objects=[group])) == planes
    assert s.raycast(o, DOWN, objects=[group], recursive=False) == []
    assert handles(s.raycast(o, DOWN, objects=[other, planes[2]])) == [planes[2], other]
    assert handles(s.raycast(o, DOWN, 5.5, 6.5)) == [planes[1]]
    first = s.raycast(o, DOWN, first_hit_only=True)
    assert handles(first) == [planes[0]]
    with pytest.raises(ValueError, match="handle"):
        s.raycast(o, DOWN, objects=[999])


def test_raycast_from_camera_takes_either_camera_and_the_options():
    s = threers.Scene()
    near = s.add_mesh(Geometry.plane(2, 2), Material.basic(), 0, 0, 0)
    far = s.add_mesh(Geometry.plane(2, 2), Material.basic(), 0, 0, -2)
    persp = threers.PerspectiveCamera(50, 1, 0.1, 100)
    persp.set_position(0, 0, 5)
    persp.look_at(0, 0, 0)
    hits = s.raycast_from_camera(persp, 0.1, 0.2)
    assert [h["handle"] for h in hits] == [near, far]
    assert hits[0]["uv"] is not None and hits[0]["face"] is not None
    assert [h["handle"] for h in s.raycast_from_camera(persp, 0.1, 0.2, first_hit_only=True)] == [near]
    assert [h["handle"] for h in s.raycast_from_camera(persp, 0.1, 0.2, objects=[far])] == [far]
    ortho = threers.OrthographicCamera(-2, 2, 2, -2, 0.1, 100)
    ortho.set_position(0, 0, 10)
    ortho.look_at(0, 0, 0)
    both = s.raycast_from_camera(ortho, 0.25, -0.5)
    assert both[0]["handle"] == near
    assert close(both[0]["point"][0], 0.5, 1e-3) and close(both[0]["point"][1], -1, 1e-3)
    # Where the ortho ray starts is the camera plane in three.js; only the
    # gap between the two planes is fixed.
    assert close(both[1]["distance"] - both[0]["distance"], 2, 1e-3)
    with pytest.raises(ValueError, match="far must be > near"):
        s.raycast_from_camera(persp, 0, 0, far=-1)
