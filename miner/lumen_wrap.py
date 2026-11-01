"""lumen-kit's post-processing for versions C and D (owner 30/10: "add lumen-kit's post-processing to ver C and D"),
copied from the round-0 leader research-somthing-t/lumen-kit @3594fd1a sn17gen/jsutil.py (no licence file; public
contest submission, read in competition/cache). WRAP, FLAT, wrap and final are verbatim (typed here).

    wrap   the module's generate becomes __inner; a new generate puts its object in a pivot group (rotation rx, ry), then
           re-centers it and scales its largest side to 0.99. Every render of a module in C/D renders wrap(module), as
           lumen-kit's checks, refine sheets and panel do; the model only ever sees and edits the unwrapped module.
    final  wrap + a 15 degree tilt toward the camera (rotation x 0.261799 rad) of a flat object: one whose height is
           under half of its larger horizontal side. lumen-kit returns final(best) without rendering it.
"""

from __future__ import annotations

import re

_EXPORT = re.compile(r"export\s+default\s+function\s*([A-Za-z_$][\w$]*)?\s*\(\s*THREE\s*\)")

WRAP = """
export default function generate(THREE) {
  const inner = __inner(THREE);
  const pivot = new THREE.Group();
  pivot.add(inner);
  pivot.rotation.set(%(rx)s, %(ry)s, 0);
  const root = new THREE.Group();
  root.add(pivot);
  root.updateMatrixWorld(true);
  const parts = [];
  root.traverse((o) => {
    if ((o.isMesh || o.isLine || o.isPoints) && !o.isInstancedMesh && o.geometry && o.matrixWorld.determinant() > 0) parts.push(o);
  });
  const bake = new Set(parts);
  for (const o of parts) {
    if (o.children.some((c) => !bake.has(c))) { bake.delete(o); continue; }
    o.geometry = o.geometry.clone();
    o.geometry.applyMatrix4(o.matrixWorld);
    o.removeFromParent();
    o.position.set(0, 0, 0);
    o.quaternion.set(0, 0, 0, 1);
    o.scale.set(1, 1, 1);
    root.add(o);
  }
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const s = 0.99 / maxDim;
  root.scale.setScalar(s);
  root.position.set(-center.x * s, -center.y * s, -center.z * s);
  return root;
}
"""

FLAT = (
    "  pivot.updateMatrixWorld(true);\n"
    "  const box0 = new THREE.Box3().setFromObject(pivot);\n"
    "  const size0 = new THREE.Vector3();\n"
    "  box0.getSize(size0);\n"
    "  if (size0.y < 0.5 * Math.max(size0.x, size0.z)) pivot.rotation.set(0.261799, 0, 0);\n"
)

NO_EXPORT = "the module must contain `export default function generate(THREE)`"  # lumen-kit's error for wrap -> None


def wrap(js: str, ry: str = "0", rx: str = "0") -> str | None:
    m = _EXPORT.search(js)
    if not m or "__inner" in js:
        return None
    inner = js[: m.start()] + "function __inner(THREE)" + js[m.end() :]
    return inner.rstrip() + "\n" + WRAP % {"rx": rx, "ry": ry}


def final(js: str) -> str | None:
    out = wrap(js)
    return None if out is None else out.replace("  pivot.rotation.set(0, 0, 0);\n", FLAT, 1)
