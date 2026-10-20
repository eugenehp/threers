"""Type stubs for the native extension `threers._native` (PyO3)."""

from __future__ import annotations

from os import PathLike
from typing import Literal, Mapping, Optional, Sequence, Tuple, TypedDict, Union

Point3 = Union["Vector3", Tuple[float, float, float], Sequence[float]]
Point2 = Union[Tuple[float, float], Sequence[float]]
PaneName = Literal["top", "front", "side", "perspective"]
Camera = Union["PerspectiveCamera", "OrthographicCamera"]
ModelFormat = Literal["glb", "gltf", "obj", "stl", "ply", "fbx", "usd", "usda", "usdc", "usdz"]
ToneMappingMode = Literal["none", "linear", "aces", "neutral", "agx"]
Side = Literal["front", "back", "double"]
Wrap = Literal["repeat", "mirror", "clamp"]
Filter = Literal["linear", "nearest"]
ColorSpace = Literal["srgb", "linear"]
MapSlot = Literal[
    "map", "alphaMap", "normalMap", "aoMap", "emissiveMap", "roughnessMap",
    "metalnessMap", "lightMap", "bumpMap", "specularMap", "displacementMap",
    "matcap", "cloudShadowMap", "gradientMap",
    "clearcoatMap", "clearcoatRoughnessMap", "clearcoatNormalMap",
    "iridescenceMap", "iridescenceThicknessMap", "sheenColorMap",
    "sheenRoughnessMap", "specularIntensityMap", "specularColorMap",
    "transmissionMap", "thicknessMap", "anisotropyMap",
]

def version() -> str: ...

class Vector3:
    x: float
    y: float
    z: float
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None: ...
    def length(self) -> float: ...
    def normalized(self) -> Vector3: ...

class Color:
    @property
    def r(self) -> float: ...
    @property
    def g(self) -> float: ...
    @property
    def b(self) -> float: ...
    def __init__(self, r: float = 1.0, g: float = 1.0, b: float = 1.0) -> None: ...
    @staticmethod
    def from_hex(hex: int) -> Color: ...

class PerspectiveCamera:
    def __init__(
        self, fov: float = 50.0, aspect: float = 1.0, near: float = 0.1, far: float = 2000.0
    ) -> None: ...
    def set_position(self, x: float, y: float, z: float) -> None: ...
    def look_at(self, x: float, y: float, z: float) -> None: ...
    def set_aspect(self, aspect: float) -> None: ...

class OrthographicCamera:
    def __init__(
        self,
        left: float = -1.0,
        right: float = 1.0,
        top: float = 1.0,
        bottom: float = -1.0,
        near: float = 0.1,
        far: float = 2000.0,
    ) -> None: ...
    def set_frustum(
        self,
        left: float,
        right: float,
        top: float,
        bottom: float,
        near: Optional[float] = None,
        far: Optional[float] = None,
    ) -> None: ...
    def set_position(self, x: float, y: float, z: float) -> None: ...
    def look_at(self, x: float, y: float, z: float) -> None: ...

class Geometry:
    """Triangle mesh data. Build with the static factories (three.js
    ``*Geometry`` parameters and defaults)."""

    @staticmethod
    def box(width: float = 1.0, height: float = 1.0, depth: float = 1.0) -> Geometry: ...
    @staticmethod
    def plane(
        width: float = 1.0, height: float = 1.0, width_segments: int = 1, height_segments: int = 1
    ) -> Geometry: ...
    @staticmethod
    def sphere(radius: float = 1.0, width_segments: int = 32, height_segments: int = 16) -> Geometry: ...
    @staticmethod
    def cylinder(
        radius_top: float = 1.0,
        radius_bottom: float = 1.0,
        height: float = 1.0,
        radial_segments: int = 32,
        height_segments: int = 1,
        open_ended: bool = False,
    ) -> Geometry: ...
    @staticmethod
    def cone(
        radius: float = 1.0,
        height: float = 1.0,
        radial_segments: int = 32,
        height_segments: int = 1,
        open_ended: bool = False,
    ) -> Geometry: ...
    @staticmethod
    def torus(
        radius: float = 1.0,
        tube: float = 0.4,
        radial_segments: int = 12,
        tubular_segments: int = 48,
        arc: float = 6.283185307179586,
    ) -> Geometry: ...
    @staticmethod
    def torus_knot(
        radius: float = 1.0,
        tube: float = 0.4,
        tubular_segments: int = 64,
        radial_segments: int = 8,
        p: int = 2,
        q: int = 3,
    ) -> Geometry: ...
    @staticmethod
    def ring(
        inner_radius: float = 0.5,
        outer_radius: float = 1.0,
        theta_segments: int = 32,
        phi_segments: int = 1,
    ) -> Geometry: ...
    @staticmethod
    def circle(radius: float = 1.0, segments: int = 32) -> Geometry: ...
    @staticmethod
    def capsule(
        radius: float = 1.0, length: float = 1.0, cap_segments: int = 4, radial_segments: int = 8
    ) -> Geometry: ...
    @staticmethod
    def lathe(points: Union[Sequence[float], Sequence[Point2]], segments: int = 12) -> Geometry: ...
    @staticmethod
    def tube(
        points: Union[Sequence[float], Sequence[Point3]],
        tubular_segments: int = 64,
        radius: float = 1.0,
        radial_segments: int = 8,
        closed: bool = False,
    ) -> Geometry: ...
    @staticmethod
    def icosahedron(radius: float = 1.0, detail: int = 0) -> Geometry: ...
    @staticmethod
    def octahedron(radius: float = 1.0, detail: int = 0) -> Geometry: ...
    @staticmethod
    def tetrahedron(radius: float = 1.0, detail: int = 0) -> Geometry: ...
    @staticmethod
    def dodecahedron(radius: float = 1.0, detail: int = 0) -> Geometry: ...
    @staticmethod
    def quad(
        corners: Union[Sequence[float], Sequence[Point3]],
        width_segments: int = 1,
        height_segments: int = 1,
    ) -> Geometry: ...
    @staticmethod
    def from_arrays(
        positions: Sequence[float],
        indices: Optional[Sequence[int]] = None,
        normals: Optional[Sequence[float]] = None,
    ) -> Geometry: ...
    @staticmethod
    def edges(source: Geometry, threshold_angle: float = 1.0) -> Geometry:
        """Sharp + border edges as a line geometry (degrees); quad diagonals never appear."""
    @staticmethod
    def wireframe(source: Geometry) -> Geometry:
        """Every edge once as a line geometry; polygons draw their outlines."""
    def positions(self) -> list[float]: ...
    def index(self) -> Optional[list[int]]: ...
    def polygon_counts(self) -> Optional[list[int]]:
        """Triangles per polygon (2 = quad), or None without a grouping."""
    def set_polygons(self, counts: Sequence[int]) -> None: ...
    def polygon_count(self) -> int: ...
    def quad_count(self) -> int: ...
    def polygon_loops(self) -> list[Optional[list[int]]]: ...
    def polygon_edges(self) -> list[Tuple[int, int]]: ...
    def subdivide(self, levels: int = 1, sharp_boundary: bool = True) -> Geometry:
        """Catmull-Clark, a new geometry of grouped quads (levels 0..=6)."""
    def infer_quads(self, max_face_angle: float = 40.0, max_shape_angle: float = 40.0) -> int:
        """Pair triangles back into quads (degrees); returns how many were formed."""
    def vertex_count(self) -> int: ...
    def triangle_count(self) -> int: ...

class Material:
    """Surface material. Keyword arguments a factory does not use raise TypeError."""

    @property
    def kind(self) -> Literal["basic", "standard", "physical", "lambert", "phong", "toon", "normal", "line"]: ...
    @staticmethod
    def basic(
        *,
        color: int = ...,
        opacity: float = ...,
        transparent: bool = ...,
        wireframe: bool = ...,
        side: Side = ...,
    ) -> Material: ...
    @staticmethod
    def standard(
        *,
        color: int = ...,
        emissive: int = ...,
        emissive_intensity: float = ...,
        roughness: float = ...,
        metalness: float = ...,
        opacity: float = ...,
        transparent: bool = ...,
        wireframe: bool = ...,
        side: Side = ...,
    ) -> Material: ...
    @staticmethod
    def physical(
        *,
        color: int = ...,
        emissive: int = ...,
        emissive_intensity: float = ...,
        roughness: float = ...,
        metalness: float = ...,
        opacity: float = ...,
        transparent: bool = ...,
        wireframe: bool = ...,
        side: Side = ...,
        clearcoat: float = ...,
        clearcoat_roughness: float = ...,
        transmission: float = ...,
        ior: float = ...,
        thickness: float = ...,
        specular_intensity: float = ...,
        specular_color: int = ...,
        sheen: float = ...,
        sheen_color: int = ...,
        sheen_roughness: float = ...,
        iridescence: float = ...,
        iridescence_ior: float = ...,
        iridescence_thickness_range: Tuple[float, float] = ...,
        anisotropy: float = ...,
        anisotropy_rotation: float = ...,
        attenuation_color: int = ...,
        attenuation_distance: float = ...,
        dispersion: float = ...,
    ) -> Material: ...
    @staticmethod
    def lambert(
        *,
        color: int = ...,
        emissive: int = ...,
        opacity: float = ...,
        transparent: bool = ...,
        wireframe: bool = ...,
        side: Side = ...,
    ) -> Material: ...
    @staticmethod
    def phong(
        *,
        color: int = ...,
        emissive: int = ...,
        specular: int = ...,
        shininess: float = ...,
        opacity: float = ...,
        transparent: bool = ...,
        wireframe: bool = ...,
        side: Side = ...,
    ) -> Material: ...
    @staticmethod
    def toon(
        *,
        color: int = ...,
        emissive: int = ...,
        opacity: float = ...,
        transparent: bool = ...,
        wireframe: bool = ...,
        side: Side = ...,
    ) -> Material:
        """MeshToonMaterial; its ramp is ``set_map("gradientMap", texture)``."""
    @staticmethod
    def normal(*, opacity: float = ..., transparent: bool = ..., wireframe: bool = ...) -> Material: ...
    @staticmethod
    def line(*, color: int = ..., opacity: float = ..., depth_test: bool = ...) -> Material: ...
    def set_map(self, slot: MapSlot, texture: Optional[Texture]) -> None:
        """three.js ``material[slot] = texture``; ValueError for a slot this kind lacks."""
    def get_map(self, slot: MapSlot) -> Optional[Texture]: ...

class Texture:
    """three.js Texture: RGBA8 pixels (rows top first) plus sampling state."""

    def __init__(self, width: int, height: int, data: bytes, color_space: ColorSpace = "srgb") -> None: ...
    @staticmethod
    def from_file(path: Union[str, PathLike[str]], color_space: ColorSpace = "srgb") -> Texture:
        """A PNG file, as three.js's TextureLoader loads it."""
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    color_space: ColorSpace
    wrap_s: Wrap
    wrap_t: Wrap
    mag_filter: Filter
    min_filter: Filter
    mipmap_filter: Optional[Filter]
    offset: Tuple[float, float]
    repeat: Tuple[float, float]
    rotation: float
    center: Tuple[float, float]
    channel: int
    """The uv set read: 0 ``uv``, 1-3 ``uv1``-``uv3``."""
    flip_y: bool

class RaycastFace(TypedDict):
    """three.js ``intersection.face``: the triangle a mesh hit landed on."""

    a: int
    b: int
    c: int
    normal: Tuple[float, float, float]
    """Geometric normal, in the object's local space."""
    material_index: int

class RaycastHit(TypedDict):
    """three.js's intersection fields in snake_case; those that do not apply
    to the kind of object hit are ``None``."""

    handle: int
    distance: float
    point: Tuple[float, float, float]
    face_index: Optional[int]
    face: Optional[RaycastFace]
    normal: Optional[Tuple[float, float, float]]
    uv: Optional[Tuple[float, float]]
    uv1: Optional[Tuple[float, float]]
    barycoord: Optional[Tuple[float, float, float]]
    instance_id: Optional[int]
    index: Optional[int]
    distance_to_ray: Optional[float]
    polygon_index: Optional[int]

class Scene:
    def __init__(self) -> None: ...
    def add_box(
        self,
        width: float = 1.0,
        height: float = 1.0,
        depth: float = 1.0,
        color: int = 0xFF6633,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ) -> int: ...
    def add_sphere(
        self, radius: float = 0.5, color: int = 0x6EA8FE, x: float = 0.0, y: float = 0.0, z: float = 0.0
    ) -> int: ...
    def add_quad(
        self,
        corners: Union[Sequence[float], Sequence[Point3]],
        color: int = 0xFF6633,
        width_segments: int = 1,
        height_segments: int = 1,
    ) -> int: ...
    def add_mesh(
        self, geometry: Geometry, material: Material, x: float = 0.0, y: float = 0.0, z: float = 0.0
    ) -> int: ...
    def add_line_segments(
        self, geometry: Geometry, material: Material, x: float = 0.0, y: float = 0.0, z: float = 0.0
    ) -> int: ...
    def add_group(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> int: ...
    def set_parent(self, child: int, parent: Optional[int] = None) -> None: ...
    def add_ambient_light(self, color: int = 0xFFFFFF, intensity: float = 1.0) -> int: ...
    def add_directional_light(
        self,
        color: int = 0xFFFFFF,
        intensity: float = 1.0,
        position: Optional[Point3] = None,
        cast_shadow: bool = False,
    ) -> int: ...
    def add_point_light(
        self,
        color: int = 0xFFFFFF,
        intensity: float = 1.0,
        position: Optional[Point3] = None,
        distance: float = 0.0,
        decay: float = 2.0,
        cast_shadow: bool = False,
    ) -> int: ...
    def add_spot_light(
        self,
        color: int = 0xFFFFFF,
        intensity: float = 1.0,
        position: Optional[Point3] = None,
        target: Optional[Point3] = None,
        angle: float = 1.0471975511965976,
        penumbra: float = 0.0,
        distance: float = 0.0,
        decay: float = 2.0,
        cast_shadow: bool = False,
    ) -> int: ...
    def add_hemisphere_light(
        self, sky_color: int = 0xFFFFFF, ground_color: int = 0xFFFFFF, intensity: float = 1.0
    ) -> int: ...
    def light_count(self) -> int: ...
    def object_count(self) -> int: ...
    def set_background(self, hex: int) -> None: ...
    def set_position(self, handle: int, x: float, y: float, z: float) -> None: ...
    def get_position(self, handle: int) -> Vector3: ...
    def set_rotation(self, handle: int, x: float, y: float, z: float) -> None: ...
    def set_scale(self, handle: int, x: float, y: float, z: float) -> None: ...
    def set_visible(self, handle: int, visible: bool) -> None: ...
    def set_cast_shadow(self, handle: int, cast: bool) -> None: ...
    def set_receive_shadow(self, handle: int, receive: bool) -> None: ...
    def set_material(self, handle: int, material: Material) -> None: ...
    def get_geometry(self, handle: int) -> Geometry: ...
    def meshes(self, handle: int) -> list[int]: ...
    def remove(self, handle: int) -> None: ...
    def raycast(
        self,
        origin: Point3,
        direction: Point3,
        near: float = 0.0,
        far: float = ...,
        *,
        first_hit_only: bool = False,
        line_threshold: float = 1.0,
        points_threshold: float = 1.0,
        recursive: bool = True,
        objects: Optional[Sequence[int]] = None,
    ) -> list[RaycastHit]: ...
    def raycast_from_camera(
        self,
        camera: Camera,
        x: float,
        y: float,
        *,
        near: float = 0.0,
        far: float = ...,
        first_hit_only: bool = False,
        line_threshold: float = 1.0,
        points_threshold: float = 1.0,
        recursive: bool = True,
        objects: Optional[Sequence[int]] = None,
    ) -> list[RaycastHit]: ...
    def load(self, path: Union[str, PathLike[str]], material: Optional[Material] = None) -> int: ...
    def load_bytes(
        self, data: bytes, format: ModelFormat, material: Optional[Material] = None
    ) -> int: ...

class HeadlessRenderer:
    def __init__(self, width: int = 640, height: int = 480, supersample: int = 1) -> None: ...
    def render_rgba(self, scene: Scene, camera: Camera) -> bytes: ...
    def render_png(self, scene: Scene, camera: Camera) -> bytes: ...
    def render_quad_view(
        self,
        scene: Scene,
        camera: PerspectiveCamera,
        maximized: Optional[PaneName] = None,
        format: Literal["png", "rgba"] = "png",
        gap: Optional[float] = None,
        gap_color: Optional[int] = None,
        border: Optional[float] = None,
        layout: Optional[Sequence[PaneName]] = None,
        backgrounds: Optional[Mapping[PaneName, int]] = None,
        margin: Optional[float] = None,
    ) -> bytes: ...
    def size(self) -> Tuple[int, int]: ...
    def set_size(self, width: int, height: int) -> None: ...
    def set_viewport(self, x: float, y: float, width: float, height: float) -> None: ...
    def reset_viewport(self) -> None: ...
    def set_scissor(self, x: float, y: float, width: float, height: float) -> None: ...
    def set_scissor_test(self, enabled: bool) -> None: ...
    def set_auto_clear(self, enabled: bool) -> None: ...
    def auto_clear(self) -> bool: ...
    def clear(self, color: int = 0x000000, alpha: float = 1.0) -> None: ...
    def set_tone_mapping(self, mode: ToneMappingMode, exposure: float = 1.0) -> None: ...
    def tone_mapping(self) -> Tuple[ToneMappingMode, float]: ...
    def set_shadows_enabled(self, enabled: bool) -> None: ...
    def shadows_enabled(self) -> bool: ...

class Tween:
    def __init__(
        self, start: float = 0.0, end: float = 1.0, duration: float = 1.0, easing: str = "cubicOut"
    ) -> None: ...
    def update(self, dt: float) -> float: ...
    def value(self) -> float: ...
    def is_finished(self) -> bool: ...

class PhysicsWorld:
    def __init__(self) -> None: ...
    def add_ball(
        self,
        radius: float = 0.5,
        x: float = 0.0,
        y: float = 5.0,
        z: float = 0.0,
        restitution: float = 0.4,
    ) -> int: ...
    def step(self, dt: float) -> None: ...
    def body_translation(self, handle: int) -> Vector3: ...
    def body_count(self) -> int: ...
