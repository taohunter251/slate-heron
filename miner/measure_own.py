#!/usr/bin/env python3
"""Our own photo-vs-render measures (written from the idea only, no rival code). numpy + PIL only.
photo_mask: bilinear background plane fitted to the image border, soft shadow removal, closing, hole fill, largest pieces;
refuses (None) when the mask changes much with the threshold. render_mask: non-white pixels of a white-background render."""
import io
import numpy as np
from PIL import Image, ImageFilter

S = 256; N = 128; BANDS = 10

def rgb256(png):
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB").resize((S, S), Image.BILINEAR)).astype(np.float32)

def _plane(rgb):
    h, w = rgb.shape[:2]; yy, xx = np.mgrid[0:h, 0:w]; r = 8
    b = np.zeros((h, w), bool); b[:r] = b[-r:] = True; b[:, :r] = b[:, -r:] = True
    A = np.c_[xx[b] / w, yy[b] / h, np.ones(b.sum())]; Y = rgb[b]; keep = np.ones(len(Y), bool)
    for _ in range(3):
        c, *_ = np.linalg.lstsq(A[keep], Y[keep], rcond=None)
        e = np.abs(A @ c - Y).max(1); keep = e <= max(6.0, 2.5 * float(np.median(e)))
    bg = (np.c_[xx.ravel() / w, yy.ravel() / h, np.ones(h * w)] @ c).reshape(h, w, 3)
    return bg, float(np.median(e[keep])) if keep.any() else 0.0, b

def _morph(m, grow, shrink):
    im = Image.fromarray(m.astype(np.uint8) * 255)
    if grow: im = im.filter(ImageFilter.MaxFilter(2 * grow + 1))
    if shrink: im = im.filter(ImageFilter.MinFilter(2 * shrink + 1))
    return np.asarray(im) > 127

def _fill(m):
    im = np.pad(~m, 1, constant_values=True); seen = np.zeros_like(im); seen[0, 0] = True
    while True:
        g = seen.copy(); g[1:] |= seen[:-1]; g[:-1] |= seen[1:]; g[:, 1:] |= seen[:, :-1]; g[:, :-1] |= seen[:, 1:]; g &= im
        if (g == seen).all(): break
        seen = g
    return ~seen[1:-1, 1:-1]

def _pieces(m):
    """keep connected pieces of at least 4% of the largest"""
    try:
        from scipy import ndimage as ni
        lab, n = ni.label(m)
        if n <= 1: return m
        a = ni.sum(m, lab, range(1, n + 1)); ok = np.nonzero(a >= .04 * a.max())[0] + 1
        return np.isin(lab, ok)
    except ImportError:
        return m

def _seg(rgb, bg, noise, b, k):
    d = np.abs(rgb - bg).max(-1); cut = k * max(18.0, 3 * noise + 8)
    l, lb = rgb.mean(-1), bg.mean(-1)
    tint = np.abs((rgb - l[..., None]) - (bg - lb[..., None])).max(-1)
    shadow = (l < .98 * lb) & (l > .45 * lb) & (tint < 12) & (d < 4 * cut)
    m = (d > cut) & ~shadow
    m = _fill(_morph(_morph(m, 0, 1), 4, 3)); m = _pieces(m)
    if m.sum() < 80 or m[b].mean() > .5: return None
    return m

def norm(m):
    ys, xs = np.nonzero(m); c = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = c.shape; s = N / max(h, w); h2, w2 = max(1, round(h * s)), max(1, round(w * s))
    sm = np.asarray(Image.fromarray(c.astype(np.uint8) * 255).resize((w2, h2), Image.BILINEAR)) > 127
    o = np.zeros((N, N), bool); t, l = (N - h2) // 2, (N - w2) // 2; o[t:t + h2, l:l + w2] = sm
    return o

def iou(a, b):
    a, b = norm(a), norm(b); return float((a & b).sum() / max(1, (a | b).sum()))

def aspect(m):
    ys, xs = np.nonzero(m); return float((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1))

def photo_mask(png):
    rgb = rgb256(png); bg, noise, b = _plane(rgb)
    m = _seg(rgb, bg, noise, b, 1.0)
    if m is None: return None
    for k in (.75, 1.25):
        o = _seg(rgb, bg, noise, b, k)
        if o is None or iou(m, o) < .8: return None
    return m, rgb

def render_mask(rgb):
    return (255 - rgb).max(-1) > 8

def profile(m):
    """width of the object per horizontal band and height per vertical band, fractions of its own box."""
    ys, xs = np.nonzero(m); c = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]; h, w = c.shape
    wid = [float(c[int(i * h / BANDS):max(int(i * h / BANDS) + 1, int((i + 1) * h / BANDS))].any(0).sum() / w) for i in range(BANDS)]
    hei = [float(c[:, int(i * w / BANDS):max(int(i * w / BANDS) + 1, int((i + 1) * w / BANDS))].any(1).sum() / h) for i in range(BANDS)]
    return wid, hei

def lab(rgb):
    x = rgb / 255.0; x = np.where(x > .04045, ((x + .055) / 1.055) ** 2.4, x / 12.92)
    M = np.array([[.4124, .3576, .1805], [.2126, .7152, .0722], [.0193, .1192, .9505]]); xyz = x @ M.T / np.array([.9505, 1.0, 1.089])
    f = np.where(xyz > .008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)

def color(rgb, m):
    inner = _morph(m, 0, 2); inner = inner if inner.sum() >= 30 else m
    L = lab(rgb)[inner]; C = np.hypot(L[:, 1], L[:, 2])
    g = np.abs(np.diff(lab(rgb)[..., 0], axis=0))[:, :-1] + np.abs(np.diff(lab(rgb)[..., 0], axis=1))[:-1]
    return {"L": float(L[:, 0].mean()), "a": float(L[:, 1].mean()), "b": float(L[:, 2].mean()), "C": float(C.mean()), "C90": float(np.percentile(C, 90)),
            "Lsd": float(L[:, 0].std()), "L5": float(np.percentile(L[:, 0], 5)), "L95": float(np.percentile(L[:, 0], 95)),
            "shine": float(((L[:, 0] > 88) & (C < 8)).mean()), "dark": float((L[:, 0] < 20).mean()),
            "edge": float(g[inner[:-1, :-1]].mean())}
