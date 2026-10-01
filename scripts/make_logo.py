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
License) into a cache directory on first run.
"""

import pathlib
import re
import tempfile
import urllib.request

import uharfbuzz as hb
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

ASSETS = pathlib.Path(__file__).resolve().parent.parent / "assets"
CACHE = pathlib.Path(tempfile.gettempdir()) / "vayu-logo-fonts"

PALETTES = {
    # for light backgrounds
    "light": dict(ink="#0F172A", a1="#0F766E", a2="#EA580C", muted="#57606A"),
    # for dark backgrounds
    "dark": dict(ink="#F1F5F9", a1="#2DD4BF", a2="#FB923C", muted="#8B949E"),
}

MARK = 88  # px; the mark is drawn on an 80-unit grid
GAP = 20
WORD_SIZE, WORD_TRACKING = 54, 0.02  # px, em
FA_SIZE, FA_GAP = 22, 10


def google_font(family: str, weight: int) -> pathlib.Path:
    """Download a static TTF of `family` at `weight` (cached)."""
    path = CACHE / f"{family.replace(' ', '')}-{weight}.ttf"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        css_url = (
            "https://fonts.googleapis.com/css2?family="
            f"{family.replace(' ', '+')}:wght@{weight}"
        )
        css = urllib.request.urlopen(css_url).read().decode()
        ttf_url = re.search(
            r"url\((https://fonts\.gstatic\.com/[^)]+\.ttf)\)", css
        ).group(1)
        path.write_bytes(urllib.request.urlopen(ttf_url).read())
    return path


class Text:
    """A shaped line of text, positioned in SVG pixels."""

    def __init__(self, path, text, size, tracking=0.0, rtl=False, lang=None):
        self.font = TTFont(path)
        hbfont = hb.Font(hb.Face(hb.Blob.from_file_path(str(path))))
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        if rtl:
            buf.direction, buf.script = "rtl", "Arab"
        if lang:
            buf.language = lang
        hb.shape(hbfont, buf, {"kern": True, "liga": True})

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
        pen = SVGPathPen(
            self.font.getGlyphSet(), ntos=lambda v: f"{v:.2f}".rstrip("0").rstrip(".")
        )
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
    persian = Text(google_font("Vazirmatn", 600), "وایو", FA_SIZE, rtl=True, lang="fa")

    column_top = (MARK - (WORD_SIZE + FA_GAP + FA_SIZE)) / 2
    text_left = MARK + GAP
    word_baseline = word.baseline_in_box(column_top, WORD_SIZE)
    fa_baseline = persian.baseline_in_box(column_top + WORD_SIZE + FA_GAP, FA_SIZE)
    fa_left = text_left + word.width - persian.width  # right-aligned under VAYU

    width = round(text_left + word.width + 2)
    # the frame must hold the mark and every glyph, descenders included
    text_bottom = max(
        word.bounds(text_left, word_baseline)[3],
        persian.bounds(fa_left, fa_baseline)[3],
    )
    height = round(max(MARK, text_bottom + 2))
    word_d = word.path(text_left, word_baseline)
    fa_d = persian.path(fa_left, fa_baseline)

    for mode, c in PALETTES.items():
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-labelledby="title">
<title id="title">Vayu (وایو)</title>
<g fill="none" stroke-linecap="round" stroke-linejoin="round" stroke-width="7" transform="scale({MARK / 80:g})">
<path d="M8 30 H52 A10 10 0 1 0 42 20" stroke="{c["a1"]}"/>
<path d="M8 48 H58 A8 8 0 1 1 50 56" stroke="{c["ink"]}"/>
<path d="M18 66 H36" stroke="{c["a2"]}"/>
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
