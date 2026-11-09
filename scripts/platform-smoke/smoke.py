"""Import an installed threers wheel and draw one red sphere.

Exits non-zero unless the pixels come back right. Used by test-linux-builds.sh.
"""

import platform
import sys

import threers
from threers import Geometry, Material

print(f"python wheel loaded on {platform.system()}-{platform.machine()}, "
      f"python {platform.python_version()}")
r = threers.HeadlessRenderer(64, 64)
s = threers.Scene()
s.set_background(0x204060)
s.add_mesh(Geometry.sphere(1), Material.basic(color=0xFF0000))
c = threers.PerspectiveCamera(50.0, 1.0, 0.1, 100.0)
c.set_position(0.0, 0.0, 4.0)
c.look_at(0.0, 0.0, 0.0)
px = r.render_rgba(s, c)
i = (32 * 64 + 32) * 4
centre, corner = tuple(px[i:i + 3]), tuple(px[0:3])
ok = centre[0] > 240 and centre[1] < 10 and centre[2] < 10 and corner[2] > 80
print(f"python render: centre {centre}, corner {corner}: {'ok' if ok else 'WRONG'}")
sys.exit(0 if ok else 1)
