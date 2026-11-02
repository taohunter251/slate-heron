"""The measured colour line of the refine after Sol: general, numbers only, no object names. Empty when the photo has no usable object mask."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from miner import measure_own as M


def line(photo_png: bytes, grid: bytes) -> str:
    q = M.photo_mask(photo_png)
    if q is None:
        return ""
    pc = M.color(q[1], q[0])
    a = np.asarray(Image.open(io.BytesIO(grid)).convert("RGB"))
    h, w = a.shape[:2]
    views = []
    for t in (a[: h // 2 - 2, : w // 2 - 2], a[: h // 2 - 2, w // 2 + 2 :], a[h // 2 + 2 :, : w // 2 - 2], a[h // 2 + 2 :, w // 2 + 2 :]):
        rgb = np.asarray(Image.fromarray(t).resize((M.S, M.S), Image.BILINEAR)).astype(np.float32)
        m = M.render_mask(rgb)
        if m.sum() >= 30:
            views.append(M.color(rgb, m))
    if not views:
        return ""
    c, l = np.median([v["C90"] for v in views]), np.median([v["L"] for v in views])
    return (
        "Measured on the object's pixels, median of the four views (lightness 0-100; colourfulness of the strongest colours, higher is more vivid). Estimates: the photo's light differs from the judge's.\n"
        f"- colourfulness: photo {pc['C90']:.0f}, render {c:.0f}\n- lightness: photo {pc['L']:.0f}, render {l:.0f}\n"
        "Move material colours, textures, roughness and metalness toward the photo where the photo itself confirms these numbers.\n"
    )
