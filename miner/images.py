"""Images sent to the model. The prompt photo is sent exactly as the base miner sends it (its original bytes as a PNG
data URL, detail auto; owner 29/10: image configurations stay as in the base miner). The review image is built as in the
SFT data-gen pipeline (sft-data-gen code/pipeline/visual.py and generate.py `_image_part`): compare.png = one row
photo | front-left | front-right | front-above | front-below, each render cropped to the object, 313 px tiles, sent
as a PNG data URL (the SFT run sent JPEG quality 90). Largest request on our 128 photos: ~4.3 MB of the 8 MiB limit.

The SFT pipeline rendered the 4 review views as separate images; the witness renders them as one 2x2 grid
(render-service-js renderGrid: cols = ceil(sqrt(n)), row-major, `gap` px apart), cut back into the 4 views here."""

from __future__ import annotations

import base64
import io
import math

from PIL import Image, ImageChops

COMPARE_MAX_SIDE = 1568
VIEW_SIZE = 1024
GRID_GAP = 5  # the service treats gap=0 as "default" (5), so the default is used explicitly


def _png_data_url(im: Image.Image, max_side: int) -> str:
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, "white")
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.standard_b64encode(buf.getvalue()).decode()


def photo_data_url(image: bytes) -> str:
    """The base miner's png_data_url: the prompt bytes as they are."""
    return "data:image/png;base64," + base64.b64encode(image).decode("ascii")


def _on_white(im: Image.Image) -> Image.Image:
    im = im.convert("RGBA")
    return Image.alpha_composite(Image.new("RGBA", im.size, "white"), im).convert("RGB")


def _crop_obj(im: Image.Image, pad: int = 24) -> Image.Image:
    diff = ImageChops.difference(im, Image.new("RGB", im.size, "white")).convert("L")
    bb = diff.point(lambda v: 255 if v > 12 else 0).getbbox()
    if not bb:
        return im
    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    h = max(bb[2] - bb[0], bb[3] - bb[1]) / 2 + pad
    return im.crop((int(cx - h), int(cy - h), int(cx + h), int(cy + h)))


def grid_views(grid_png: bytes, n: int = 4, size: int = VIEW_SIZE, gap: int = GRID_GAP) -> list[Image.Image]:
    grid = Image.open(io.BytesIO(grid_png))
    cols = math.ceil(math.sqrt(n))
    out = []
    for i in range(n):
        x, y = (i % cols) * (size + gap), (i // cols) * (size + gap)
        out.append(grid.crop((x, y, x + size, y + size)))
    return out


def view_data_urls(grid_png: bytes, size: int) -> list[str]:
    """The 4 views of a render grid, each its own PNG data URL (version C's rewrite turn, as golden217 sends them)."""
    return [_png_data_url(_on_white(v), size) for v in grid_views(grid_png, size=size)]


def sheet_data_url(grid_png: bytes, size: int) -> str:
    """The 4 views of a render grid as one gapless 2x2 sheet (version D, as lumen-kit's views._sheet builds it)."""
    sheet = Image.new("RGB", (2 * size, 2 * size), "white")
    for i, v in enumerate(grid_views(grid_png, size=size)):
        sheet.paste(_on_white(v), ((i % 2) * size, (i // 2) * size))
    return _png_data_url(sheet, 2 * size)


def compare_data_url(image: bytes, grid_png: bytes) -> str:
    views = grid_views(grid_png)
    tiles = [_on_white(Image.open(io.BytesIO(image)))] + [_crop_obj(_on_white(v)) for v in views]
    t = COMPARE_MAX_SIDE // len(tiles)
    sheet = Image.new("RGB", (t * len(tiles), t), "white")
    for k, im in enumerate(tiles):
        sheet.paste(im.resize((t, t)), (t * k, 0))
    return _png_data_url(sheet, COMPARE_MAX_SIDE)
