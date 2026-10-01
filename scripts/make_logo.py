#!/usr/bin/env python3
"""
Build the README logo: assets/logo-light.svg and assets/logo-dark.svg.

The mark is three wind strokes; the text ("VAYU" in Manrope ExtraBold and
"وایو" in Vazirmatn SemiBold) is shaped with HarfBuzz and written as outlines,
so the SVGs render the same everywhere without web fonts.

Usage:
    pip install fonttools uharfbuzz
    python scripts/make_logo.py

Fonts are downloaded from Google Fonts (both are under the SIL Open Font
License) into ~/.cache/vayu-logo-fonts, one file per font version, so a new
upstream version is fetched rather than mixed with an old cached copy.
"""

import math
import os
import pathlib
import re
import urllib.request

import uharfbuzz as hb
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.svgLib.path import parse_path
from fontTools.ttLib import TTFont

ASSETS = pathlib.Path(__file__).resolve().parent.parent / "assets"
CACHE = (
    pathlib.Path(os.environ.get("XDG_CACHE_HOME") or pathlib.Path.home() / ".cache")
    / "vayu-logo-fonts"
)
TIMEOUT = 30  # seconds per request

PALETTES = {
    # for light backgrounds
    "light": dict(ink="#0F172A", a1="#0F766E", a2="#EA580C", muted="#57606A"),
    # for dark backgrounds
    "dark": dict(ink="#F1F5F9", a1="#2DD4BF", a2="#FB923C", muted="#8B949E"),
}

# the mark: three wind strokes on an 80-unit grid, as (palette colour, path)
MARK_STROKES = [
    ("a1", "M8 30 H52 A10 10 0 1 0 42 20"),
    ("ink", "M8 48 H58 A8 8 0 1 1 50 56"),
    ("a2", "M18 66 H36"),
]
MARK_SCALE, MARK_STROKE = 1.1, 7  # grid units -> px, stroke width in grid units
GAP = 20  # px between the mark's ink and the text's ink
MARGIN = 2  # px of space around all the ink
WORD_SIZE, WORD_TRACKING = 54, 0.02  # px, em
FA_SIZE, FA_GAP = 22, 10


def num(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


def union(*boxes):
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def mark_ink():
    """Ink box of the mark in px: the strokes' centre lines plus half a stroke."""
    pen = BoundsPen(None)
    for _, d in MARK_STROKES:
        parse_path(d, pen)
    half = MARK_STROKE / 2
    x0, y0, x1, y1 = pen.bounds
    return tuple(v * MARK_SCALE for v in (x0 - half, y0 - half, x1 + half, y1 + half))


def google_font(family: str, weight: int) -> pathlib.Path:
    """Download a static TTF of `family` at `weight`, cached per font version."""
    css_url = (
        "https://fonts.googleapis.com/css2?family="
        f"{family.replace(' ', '+')}:wght@{weight}"
    )
    css = urllib.request.urlopen(css_url, timeout=TIMEOUT).read().decode()
    ttf_urls = re.findall(r"url\((https://fonts\.gstatic\.com/[^)]+\.ttf)\)", css)
    if len(ttf_urls) != 1:
        raise RuntimeError(
            f"Expected one TTF for {family} {weight} from Google Fonts, got "
            f"{len(ttf_urls)}; the CSS API may have changed:\n{css}"
        )
    # the URL path carries the font version (e.g. /s/manrope/v20/...)
    path = CACHE / re.sub(r"[^A-Za-z0-9.]+", "-", ttf_urls[0].split("/s/", 1)[1])
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        data = urllib.request.urlopen(ttf_urls[0], timeout=TIMEOUT).read()
        partial = path.with_suffix(".part")
        partial.write_bytes(data)
        os.replace(partial, path)  # never leave a truncated font in the cache
    return path


class Text:
    """A shaped line of text, positioned in SVG pixels."""

    def __init__(self, path, text, size, tracking=0.0, lang=None):
        self.font = TTFont(path)
        missing = [c for c in text if ord(c) not in self.font.getBestCmap()]
        if missing:
            raise RuntimeError(f"{path.name} has no glyphs for {missing!r}")
        hbfont = hb.Font(hb.Face(hb.Blob.from_file_path(str(path))))
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()  # direction and script, e.g. RTL Arabic
        if lang:
            buf.language = lang
        hb.shape(hbfont, buf)

        self.scale = size / self.font["head"].unitsPerEm
        order = self.font.getGlyphOrder()
        self.glyphs = []
        x = 0.0
        # glyphs come back in visual (left-to-right) order, RTL included
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            self.glyphs.append(
                (
                    order[info.codepoint],
                    x + pos.x_offset * self.scale,
                    pos.y_offset * self.scale,
                )
            )
            x += pos.x_advance * self.scale + tracking * size
        self.width = x - tracking * size  # no tracking after the last glyph
        hhea = self.font["hhea"]
        self.ascent = hhea.ascent * self.scale
        self.descent = -hhea.descent * self.scale

    def _draw(self, pen, left, baseline):
        glyphset = self.font.getGlyphSet()
        s = self.scale
        for name, gx, gy in self.glyphs:
            # font units (y up) -> SVG pixels (y down)
            glyphset[name].draw(
                TransformPen(pen, (s, 0, 0, -s, left + gx, baseline - gy))
            )

    def path(self, left, baseline):
        pen = SVGPathPen(self.font.getGlyphSet(), ntos=num)
        self._draw(pen, left, baseline)
        return pen.getCommands()

    def bounds(self, left, baseline):
        pen = BoundsPen(self.font.getGlyphSet())
        self._draw(pen, left, baseline)
        return pen.bounds

    def baseline_in_box(self, top, line_height):
        """Baseline of this text centred in a line box, as CSS lays it out."""
        half_leading = (line_height - (self.ascent + self.descent)) / 2
        return top + half_leading + self.ascent


def main():
    word = Text(google_font("Manrope", 800), "VAYU", WORD_SIZE, WORD_TRACKING)
    persian = Text(google_font("Vazirmatn", 600), "وایو", FA_SIZE, lang="fa")

    # set the two lines at the origin, as CSS line boxes would stack them
    word_baseline = word.baseline_in_box(0, WORD_SIZE)
    fa_baseline = persian.baseline_in_box(WORD_SIZE + FA_GAP, FA_SIZE)
    word_ink = word.bounds(0, word_baseline)
    # right-align the visible letters (not the advance boxes) under VAYU
    fa_left = word_ink[2] - persian.bounds(0, fa_baseline)[2]
    text_ink = union(word_ink, persian.bounds(fa_left, fa_baseline))

    # put the text GAP px right of the mark's ink, centred on it vertically
    mark = mark_ink()
    dx = mark[2] + GAP - text_ink[0]
    dy = (mark[1] + mark[3] - text_ink[1] - text_ink[3]) / 2
    ink = union(
        mark, (text_ink[0] + dx, text_ink[1] + dy, text_ink[2] + dx, text_ink[3] + dy)
    )

    # then shift everything so the frame holds all the ink with MARGIN to spare
    ox, oy = MARGIN - ink[0], MARGIN - ink[1]
    width = math.ceil(ink[2] - ink[0] + 2 * MARGIN)
    height = math.ceil(ink[3] - ink[1] + 2 * MARGIN)
    word_d = word.path(dx + ox, word_baseline + dy + oy)
    fa_d = persian.path(fa_left + dx + ox, fa_baseline + dy + oy)

    for mode, c in PALETTES.items():
        strokes = "\n".join(
            f'<path d="{d}" stroke="{c[colour]}"/>' for colour, d in MARK_STROKES
        )
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-labelledby="title">
<title id="title">Vayu (وایو)</title>
<g fill="none" stroke-linecap="round" stroke-linejoin="round" stroke-width="{MARK_STROKE}" transform="translate({num(ox)} {num(oy)}) scale({MARK_SCALE:g})">
{strokes}
</g>
<path d="{word_d}" fill="{c["ink"]}"/>
<path d="{fa_d}" fill="{c["muted"]}"/>
</svg>
"""
        out = ASSETS / f"logo-{mode}.svg"
        out.write_text(svg, encoding="utf-8")
        print(f"wrote {out} ({width}x{height})")


if __name__ == "__main__":
    main()
