"""CPU-only image analysis: reference object mask, candidate view sheets, similarity scores.

Everything here must be reproducible during replay: integer images, rounded scores and
index tie-breaks; decisions never depend on time or thread order.
"""
from __future__ import annotations

import io
from collections import deque
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from PIL import Image

N = 256
RING = 8
CANON = 64
SHEET_BG = (128, 128, 128)
SHEET_SIZE = 384
GAP = 5
# The judge's four front views, rendered once per candidate on a uniform grey background:
# used for CPU scoring and (with a white background) for the selection calls.
FRONT_VIEWS = [(30, 0), (330, 0), (0, 15), (0, -30)]
SHEET_VIEWS = FRONT_VIEWS
PROBE_FRONT = [0, 1, 2, 3]
PROBE_SIDE: list[int] = []


def _ring(shape: tuple[int, int], k: int) -> np.ndarray:
    r = np.zeros(shape, bool)
    r[:k], r[-k:], r[:, :k], r[:, -k:] = True, True, True, True
    return r


def _mad_cut(v: np.ndarray, k: float) -> float:
    med = float(np.median(v))
    return med + k * 1.4826 * float(np.median(np.abs(v - med)))


def _shift_or(m: np.ndarray) -> np.ndarray:
    o = m.copy()
    o[1:] |= m[:-1]
    o[:-1] |= m[1:]
    o[:, 1:] |= m[:, :-1]
    o[:, :-1] |= m[:, 1:]
    return o


def _dilate(m: np.ndarray, k: int) -> np.ndarray:
    for _ in range(k):
        m = _shift_or(m)
    return m


def _erode(m: np.ndarray, k: int) -> np.ndarray:
    for _ in range(k):
        m = ~_shift_or(~m)
    return m


def _components(m: np.ndarray) -> list[np.ndarray]:
    h, w = m.shape
    seen = np.zeros_like(m)
    out = []
    for sy, sx in zip(*np.nonzero(m)):
        if seen[sy, sx]:
            continue
        q = deque([(int(sy), int(sx))])
        seen[sy, sx] = True
        ys, xs = [], []
        while q:
            y, x = q.popleft()
            ys.append(y)
            xs.append(x)
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        c = np.zeros_like(m)
        c[ys, xs] = True
        out.append(c)
    return out


def _fill_holes(m: np.ndarray) -> np.ndarray:
    pad = np.pad(~m, 1, constant_values=True)
    for c in _components(pad):
        if c[0, 0]:
            return ~c[1:-1, 1:-1]
    return m


def _keep_large(m: np.ndarray, frac: float) -> np.ndarray:
    comps = _components(m)
    if not comps:
        return m
    big = max(int(c.sum()) for c in comps)
    out = np.zeros_like(m)
    for c in comps:
        if int(c.sum()) >= frac * big:
            out |= c
    return out


def _solve_exact(a: list[list[int]], b: list[int]) -> list[Fraction]:
    """Gaussian elimination in exact rationals: identical on every machine."""
    n = len(b)
    m = [[Fraction(v) for v in row] + [Fraction(b[i])] for i, row in enumerate(a)]
    for col in range(n):
        piv = next(r for r in range(col, n) if m[r][col] != 0)
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r != col and m[r][col] != 0:
                f = m[r][col] / m[col][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    return [m[i][n] / m[i][i] for i in range(n)]


def _background(img: np.ndarray) -> np.ndarray:
    """Quadratic background surface fitted to the border ring, rounded to integers.

    Integer pixel coordinates and int64 sums keep the least-squares system exact; the 6x6 solve uses
    rationals, so the result does not depend on BLAS/LAPACK or SIMD summation order.
    """
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x, y = (xx - w // 2).astype(np.int64), (yy - h // 2).astype(np.int64)
    basis = [np.ones_like(x), x, y, x * x, y * y, x * y]
    ring = _ring((h, w), RING)
    keep = ring.copy()
    bg = np.zeros(img.shape, np.int32)
    for _ in range(3):
        cols = [v[keep] for v in basis]
        ata = [[int((p * q).sum()) for q in cols] for p in cols]
        for c in range(3):
            target = img[..., c][keep].astype(np.int64)
            coef = _solve_exact(ata, [int((p * target).sum()) for p in cols])
            # scaled integer evaluation: value * 2^20, rounded once
            scaled = [round(k * (1 << 20)) for k in coef]
            acc = np.zeros((h, w), np.int64)
            for k, v in zip(scaled, basis):
                acc += k * v
            bg[..., c] = ((acc + (1 << 19)) >> 20).astype(np.int32)
        res = np.abs(img - bg).sum(-1)
        keep = ring & (res < _mad_cut(res[ring], 4.0) + 3)
    return bg


def reference_mask(png: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Object mask of the prompt photo at N px and the resized RGB image (uint8)."""
    rgb = np.asarray(Image.open(io.BytesIO(png)).convert("RGB").resize((N, N), Image.LANCZOS)).astype(np.int32)
    bg = _background(rgb)
    d2 = ((rgb - bg) ** 2).sum(-1)
    lum, blum = rgb.sum(-1), bg.sum(-1) + 1
    chroma = np.abs((3 * rgb - lum[..., None]) - (3 * bg - blum[..., None])).max(-1)
    shadow = (100 * lum > 55 * blum) & (100 * lum < 99 * blum) & (chroma < 30)
    ring = _ring((N, N), RING)
    t = max(6.0, 3.0 * _mad_cut(np.sqrt(d2[ring]), 1.0))
    gy, gx = np.gradient(rgb.sum(-1) - bg.sum(-1))
    g = np.abs(gx) + np.abs(gy)
    edge = g > max(12.0, _mad_cut(g[ring], 6.0))
    raw = ((d2 > t * t) & ~(shadow & (d2 < 16 * t * t))) | edge
    m = _erode(_fill_holes(_erode(_dilate(raw, 2), 1)), 1)
    return _keep_large(m, 0.02), rgb.astype(np.uint8)


def _canon(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(ys) < 8:
        return None
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = CANON / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    small = np.asarray(Image.fromarray(crop.astype(np.uint8) * 255).resize((nw, nh), Image.BILINEAR)) >= 128
    out = np.zeros((CANON, CANON), bool)
    oy, ox = (CANON - nh) // 2, (CANON - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = small
    return out


def _aspect(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    return round((int(xs.max()) - int(xs.min()) + 1) / (int(ys.max()) - int(ys.min()) + 1), 4)


def _mean_color(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    px = rgb[mask].astype(np.int64)
    return px.sum(0) // max(1, len(px))


@dataclass
class Reference:
    mask: np.ndarray
    rgb: np.ndarray
    canon: np.ndarray | None
    color: np.ndarray
    aspect: float


def analyze_reference(png: bytes) -> Reference:
    m, rgb = reference_mask(png)
    c = _canon(m)
    return Reference(m, rgb, c, _mean_color(rgb, m), _aspect(m) if c is not None else 1.0)


def sheet_options() -> dict:
    return dict(thetas=tuple(float(t) for t, _ in SHEET_VIEWS), phis=tuple(float(p) for _, p in SHEET_VIEWS),
                img_size=SHEET_SIZE, bg_color="808080", lighting="neutral")


def split_sheet(png: bytes) -> list[np.ndarray]:
    img = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    cols = int(np.ceil(np.sqrt(len(SHEET_VIEWS))))  # renderer layout: ceil(sqrt(n)) columns, row-major
    tiles = []
    for i in range(len(SHEET_VIEWS)):
        r, c = divmod(i, cols)
        y0, x0 = r * (SHEET_SIZE + GAP), c * (SHEET_SIZE + GAP)
        tiles.append(img[y0:y0 + SHEET_SIZE, x0:x0 + SHEET_SIZE])
    return tiles


def tile_mask(tile: np.ndarray) -> np.ndarray:
    return (tile != np.array(SHEET_BG, np.uint8)).any(-1)


def compose(tiles: list[np.ndarray], bg: tuple[int, int, int] | None = None, size: int | None = None) -> bytes:
    """2x2 PNG of four tiles; optionally replace the exact sheet background colour."""
    ts = []
    for t in tiles:
        if bg is not None:
            t = t.copy()
            t[~tile_mask(t)] = bg
        im = Image.fromarray(t)
        if size is not None and size != SHEET_SIZE:
            im = im.resize((size, size), Image.LANCZOS)
        ts.append(im)
    s = ts[0].width
    fill = bg if bg is not None else SHEET_BG
    canvas = Image.new("RGB", (2 * s + GAP, 2 * s + GAP), fill)
    for i, im in enumerate(ts):
        r, c = divmod(i, 2)
        canvas.paste(im, (c * (s + GAP), r * (s + GAP)))
    buf = io.BytesIO()
    canvas.save(buf, "PNG")
    return buf.getvalue()


@dataclass
class Score:
    total: float
    iou: float
    color: float
    view: int
    side_iou: float
    aspect: float


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = int((a | b).sum())
    return round(int((a & b).sum()) / u, 4) if u else 0.0


def score(ref: Reference, tiles: list[np.ndarray]) -> Score:
    best = Score(0.0, 0.0, 0.0, PROBE_FRONT[0], 0.0, 1.0)
    if ref.canon is None:
        return best
    for i in PROBE_FRONT:
        m = tile_mask(tiles[i])
        c = _canon(m)
        if c is None:
            continue
        iou = _iou(c, ref.canon)
        d = int(np.abs(_mean_color(tiles[i], m) - ref.color).sum())  # 0..765
        col = round(max(0.0, 1.0 - d / 300), 4)
        total = round(0.7 * iou + 0.3 * col, 4)
        if total > best.total:
            best = Score(total, iou, col, i, 0.0, _aspect(m))
    side = 0.0
    for i in PROBE_SIDE:
        c = _canon(tile_mask(tiles[i]))
        if c is not None:
            side = max(side, _iou(c, ref.canon))
    best.side_iou = side
    return best


def _hex(c: np.ndarray) -> str:
    return "#" + "".join(f"{int(v):02x}" for v in c)


def diagnostics(ref: Reference, tiles: list[np.ndarray], s: Score) -> tuple[str, bytes]:
    """Text hints and an overlay image comparing the target silhouette with the closest model view."""
    tile = tiles[s.view]
    mask = tile_mask(tile)
    mc = _canon(mask)
    lines = []
    if ref.canon is not None and mc is not None:
        ra, ma = ref.aspect, s.aspect
        if abs(ma / ra - 1) > 0.08:
            lines.append(f"In the closest matching view the model's width/height ratio is {ma:.2f}, the target's is "
                         f"{ra:.2f}: the model is too {'wide or short' if ma > ra else 'narrow or tall'}.")
        mcol = _mean_color(tile, mask)
        lines.append(f"Average object colour: target {_hex(ref.color)} (photo lighting), model {_hex(mcol)} "
                     f"(render lighting). Silhouette overlap {s.iou:.0%}.")
    tw = 192
    canvas = Image.new("RGB", (3 * tw + 20, tw), (255, 255, 255))
    for k, (rgb, m) in enumerate(((ref.rgb, ref.mask), (tile, mask))):
        ys, xs = np.nonzero(m)
        if len(ys):
            crop = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()
            if k == 1:
                crop[~m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]] = (255, 255, 255)
            im = Image.fromarray(crop)
            im.thumbnail((tw, tw))
            canvas.paste(im, (k * (tw + 10) + (tw - im.width) // 2, (tw - im.height) // 2))
    if ref.canon is not None and mc is not None:
        ov = np.full((CANON, CANON, 3), 255, np.uint8)
        ov[ref.canon & mc] = (150, 150, 150)
        ov[ref.canon & ~mc] = (220, 40, 40)
        ov[mc & ~ref.canon] = (40, 80, 220)
        canvas.paste(Image.fromarray(ov).resize((tw, tw), Image.NEAREST), (2 * tw + 20, 0))
    buf = io.BytesIO()
    canvas.save(buf, "PNG")
    lines.append("Attached comparison: left = target object cut out of the photo, middle = your model from the closest "
                  "matching angle, right = silhouette overlap after scaling both to the same size (grey = both, "
                  "red = only target, blue = only model). The photo's camera angle is only approximated.")
    return "\n".join(lines), buf.getvalue()


# Pose correction for objects photographed from well above (flat things lying down): one small probe
# render of the final module at theta 0 and several elevations; the judge's front views (phi 0, 0, +15,
# -30) see the object at phi - tilt after tilting it toward the camera by `tilt` degrees.
TILT_ELEVS = [15, 0, -15, -30, -45, -60]
TILT_JUDGE_PHIS = [0, 0, 15, -30]
TILT_SIZE = 192


def tilt_options() -> dict:
    return dict(thetas=tuple(0.0 for _ in TILT_ELEVS), phis=tuple(float(p) for p in TILT_ELEVS),
                img_size=TILT_SIZE, bg_color="808080", lighting="neutral")


def choose_tilt(ref: Reference, png: bytes, max_tilt: int, min_gain: float = 0.1) -> tuple[int, dict]:
    if ref.canon is None:
        return 0, {}
    img = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    cols = int(np.ceil(np.sqrt(len(TILT_ELEVS))))
    iou_at = {}
    for i, p in enumerate(TILT_ELEVS):
        r, c = divmod(i, cols)
        tile = img[r * (TILT_SIZE + GAP):r * (TILT_SIZE + GAP) + TILT_SIZE,
                   c * (TILT_SIZE + GAP):c * (TILT_SIZE + GAP) + TILT_SIZE]
        m = _canon(tile_mask(tile))
        iou_at[p] = _iou(m, ref.canon) if m is not None else 0.0
    tilts = [t for t in (0, 15, 30) if t <= max_tilt]
    scores = {t: round(sum(iou_at.get(p - t, 0.0) for p in TILT_JUDGE_PHIS) / len(TILT_JUDGE_PHIS), 4) for t in tilts}
    best = max(tilts, key=lambda t: (scores[t], -t))
    return (best if scores[best] - scores[0] >= min_gain else 0), scores


TILT_WRAP = """
export default function generate(THREE) {
  const root = new THREE.Group();
  root.add(__posed_inner(THREE));
  root.rotation.set(%d * Math.PI / 180, 0, 0);
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3());
  const extent = Math.max(size.x, size.y, size.z);
  if (extent > 0 && Number.isFinite(extent)) {
    const center = box.getCenter(new THREE.Vector3());
    const scale = 0.95 / extent;
    root.scale.setScalar(scale);
    root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
  }
  return root;
}
"""


def apply_tilt(src: str, deg: int) -> str:
    import re
    inner = re.sub(r"export\s+default\s+function\s+generate\s*\(", "function __posed_inner(", src, count=1)
    return inner + TILT_WRAP % deg
