"""render_ocr.py: 'retype' an AI-generated string by rendering it to a bitmap and
reading it back via OCR, producing a string with zero lineage to the original
text object (no clipboard flavors, no invisible Unicode, no HTML/RTF wrapper,
no hidden characters of any kind: OCR physically cannot see what isn't ink on
the page).

This is the API-delivery equivalent of a human retyping text on a keyboard.
It does NOT defeat a statistical (word-choice) watermark - the words that come
back out are the same words that went in. Pair with a paraphraser for that.

Public entrypoint: retype(text) -> (clean_text, diff_report)
"""

from __future__ import annotations

import difflib
import io
import os
import platform
import re
import unicodedata
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
import pytesseract

# Font path resolution: a monospace font with a known, fixed glyph width is
# required (CHAR_WIDTH_PX below depends on it), and it must be present on
# the host without requiring the end user to hunt for one. Checked in order;
# first match wins. HERMES_STRIP_METADATA_FONT overrides everything (set it
# if none of these paths exist on your system, e.g. a minimal Docker image
# or a Windows install without the bundled fallback below).
_FONT_CANDIDATES = [
    os.environ.get("HERMES_STRIP_METADATA_FONT"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  # Debian/Ubuntu
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",            # Fedora/RHEL
    "/usr/local/share/fonts/DejaVuSansMono.ttf",             # manual install, any Linux
    "/opt/homebrew/share/fonts/DejaVuSansMono.ttf",          # macOS (Apple Silicon Homebrew)
    "/usr/local/share/fonts/DejaVuSansMono.ttf",             # macOS (Intel Homebrew)
    "C:\\Windows\\Fonts\\consola.ttf",                        # Windows: Consolas (near-universal since Vista)
    "C:\\Windows\\Fonts\\cour.ttf",                           # Windows: Courier New (always present, fallback)
]


def _resolve_font_path() -> str:
    for candidate in _FONT_CANDIDATES:
        if candidate and os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "hermes-strip-metadata: no monospace font found at any known path "
        "for this OS (checked: " + ", ".join(c for c in _FONT_CANDIDATES if c) + "). "
        "Set HERMES_STRIP_METADATA_FONT to a .ttf path on this machine. "
        "Stage 2 (render-OCR) cannot run without a font and will be skipped "
        "(falls back to stage-1 output only) until this is set."
    )


try:
    FONT_PATH = _resolve_font_path()
except FileNotFoundError as _font_err:
    # Fail loud, not silent: a missing font used to mean every call quietly
    # degraded to stage-1-only with no visible signal anyone would notice.
    # Warn once at import time, then let callers keep working (retype()
    # below re-raises into the same fallback path, intentionally).
    warnings.warn(str(_font_err), RuntimeWarning, stacklevel=2)
    FONT_PATH = None

FONT_SIZE = 28
LINE_SPACING = 8
MARGIN = 20
MAX_CHARS_PER_LINE = 100

# Large pastes are paginated rather than rendered as one tall image: OCR
# accuracy degrades on very tall images, and a single huge render is slow
# and hard to debug. Each page is rendered/OCR'd/diffed independently, then
# stitched back together in original order.
MAX_LINES_PER_PAGE = 60


def _measure_char_width_px(font_path: str, font_size: int) -> float:
    """Measure the real advance width of this font's monospace glyphs rather
    than assuming a fixed constant. A single hardcoded CHAR_WIDTH_PX tuned
    for DejaVu Sans Mono is wrong for Consolas or Courier New (the Windows
    fallbacks in _FONT_CANDIDATES above), since different monospace fonts
    do not share the same glyph advance width at the same point size. This
    is measured once, from whichever font actually resolved on this host,
    not guessed per font family.

    Measuring a run of characters and dividing (rather than a single glyph)
    avoids integer-rounding noise in the per-glyph case.
    """
    font = ImageFont.truetype(font_path, font_size)
    sample = "M" * 50
    return font.getlength(sample) / len(sample)


# Default matches the old hardcoded guess, used only if font metrics can't
# be measured for some reason (FONT_PATH is None, or the measurement itself
# fails) - in both cases Stage 2 is already being skipped via FONT_PATH, so
# this value is never actually used to render anything in that case.
CHAR_WIDTH_PX: float = 17.0
if FONT_PATH:
    try:
        CHAR_WIDTH_PX = _measure_char_width_px(FONT_PATH, FONT_SIZE)
    except Exception as _measure_err:  # pragma: no cover - defensive, see warning
        warnings.warn(
            f"hermes-strip-metadata: failed to measure font metrics for "
            f"{FONT_PATH}, falling back to a default CHAR_WIDTH_PX="
            f"{CHAR_WIDTH_PX} (may misalign indentation on this font): "
            f"{_measure_err}",
            RuntimeWarning,
            stacklevel=2,
        )

# Cache the parsed font + cmap at module load instead of re-parsing the
# font file from disk on every retype() call. TTFont() is not free (it
# parses the whole font file), and find_unsupported_chars() previously ran
# it on every single call regardless of input size.
_CMAP_CACHE: Optional[dict] = None


def _get_cmap() -> dict:
    global _CMAP_CACHE
    if _CMAP_CACHE is None:
        if not FONT_PATH:
            raise FileNotFoundError("no font available; see warning at import time")
        tt = TTFont(FONT_PATH)
        _CMAP_CACHE = tt.getBestCmap()
    return _CMAP_CACHE


def find_unsupported_chars(text: str, font_path: Optional[str] = None) -> List[str]:
    """Return the sorted set of characters in `text` that the render font has
    no glyph for (emoji, CJK, most non-Latin scripts in a Latin monospace
    font, etc), checked against the font's actual cmap table.

    A bounding-box check on the rendered glyph is NOT sufficient here: most
    fonts draw a visible "tofu" fallback box for codepoints they don't
    support, which has a non-empty bbox and would be indistinguishable from
    a real glyph by that test alone. The cmap is the authoritative source.

    Any unsupported character means retyping would corrupt the text (either
    render as a tofu box that OCR misreads, or get silently dropped), the
    caller should skip the round trip for this input rather than risk it.

    The cmap is parsed once at module import and cached (see `_get_cmap`)
    rather than re-parsed from disk on every call - font files are not
    huge, but there is no reason to re-read and re-decode one on every
    single response that passes through this hook.
    """
    cmap = _get_cmap() if font_path is None else TTFont(font_path).getBestCmap()
    bad = set()
    for ch in set(text):
        if ch.isspace():
            continue
        if ord(ch) not in cmap:
            bad.add(ch)
    return sorted(bad)


@dataclass
class RetypeResult:
    clean_text: str
    original_text: str
    diff_ratio: float
    diff_lines: List[str] = field(default_factory=list)
    suspect: bool = False
    skipped: bool = False
    skip_reason: str = ""
    char_damage: List[str] = field(default_factory=list)


def _grapheme_clusters(word: str) -> List[str]:
    """Split `word` into user-perceived characters (base character plus any
    trailing Unicode combining marks), not raw code units.

    A plain `word[i:i+n]` slice can land exactly between a base character
    and a combining mark that is supposed to attach to it (e.g. "e" +
    U+0301 COMBINING ACUTE ACCENT), severing them onto two different
    render lines. Confirmed empirically (round 5 review): a 201-character
    unbroken word with a combining acute at index 100, hard-split at
    max_chars=100, put the bare base "e" at the end of line 0 and the
    orphaned combining mark alone at the start of line 1; line 1 rendered
    the mark floating with no base to attach to, and OCR read the whole
    thing back wrong, with the accent landing on an unrelated character
    two positions away. The split stayed byte-lossless and the suspect
    gate still caught the corruption, so nothing shipped silently wrong,
    but a correctly clustered split avoids manufacturing that corruption
    in the first place.

    `unicodedata.combining()` returns nonzero for any combining mark
    regardless of script, so this works for accented Latin, Hebrew
    niqqud, Arabic diacritics, Devanagari matras, etc, not just the Latin
    case that triggered this fix.
    """
    clusters: List[str] = []
    for ch in word:
        category = unicodedata.category(ch)
        # unicodedata.combining() only returns nonzero for characters with
        # an assigned canonical combining class, which covers Latin/Hebrew/
        # Arabic/Devanagari diacritics but NOT every script's visually
        # attaching mark. Confirmed empirically (round 6 review): Lao
        # vowel sign U+0EB4 is Unicode category Mn (a real combining mark
        # by general category) and IS present in DejaVu Sans Mono's cmap
        # (so find_unsupported_chars never intercepts it), but
        # unicodedata.combining("\u0EB4") == 0, so the old combining()-only
        # check treated it as its own independent cluster and manufactured
        # the exact severed-mark corruption this function exists to
        # prevent, just for a script family combining() doesn't cover.
        # Checking general category (Mn=nonspacing mark, Mc=spacing
        # combining mark, Me=enclosing mark) instead of the narrower
        # canonical-combining-class field covers every script whose marks
        # visually attach to a base character, not just the ones Unicode
        # happens to assign a nonzero combining class to.
        if clusters and (unicodedata.combining(ch) or category in ("Mn", "Mc", "Me")):
            clusters[-1] += ch
        else:
            clusters.append(ch)
    return clusters


def _wrap_for_render(text: str, max_chars: int = MAX_CHARS_PER_LINE) -> List[str]:
    """Wrap text into fixed-width lines without breaking words where avoidable,
    preserving existing newlines as hard breaks (so paragraph structure survives
    the round trip) AND preserving leading whitespace/indentation per source
    line (critical for code blocks: OCR round-trips must not silently change
    what the text means).

    Two defensive behaviors, both load-bearing even though stage 1
    (sanitize_text) already handles them before render_ocr normally sees
    text: this module's own public entrypoint (`retype`) takes a raw
    string, not "a string stage 1 already cleaned", so render_ocr must be
    safe to call directly without relying on an implicit caller contract.

    1. ALL tabs (leading or interior) are expanded to spaces up front, not
       just leading indentation. PIL's ImageDraw has no reliable tab-stop
       behavior, and OCR-side indent reconstruction always rebuilds
       indentation in space units regardless of the source; an interior
       tab left unexpanded renders as a stray glyph that OCR corrupts
       (confirmed empirically: "name\\tage\\tcity" round-tripped through
       real OCR as "name[jage[[city").

    2. A single "word" (no internal space) longer than fits on one line is
       hard-split at max_chars rather than left to overflow the fixed
       canvas width. Without this, a long unbroken token (a URL, API key,
       or hash with no spaces) runs off the right edge of the rendered
       image and OCR reads back garbage for the overflow portion
       (confirmed empirically: a 250-character unbroken run rendered at
       the current canvas width came back from OCR as unrelated garbage
       character sequences, not the original text). The existing suspect
       gate catches this and falls back to stage-1 output, so it is not a
       silent corruption risk, but it does mean stage 2 never actually
       retypes such tokens; splitting them keeps them inside the canvas so
       the round trip can succeed instead of always being rejected.
    """
    out: List[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            out.append("")
            continue
        # Expand tabs anywhere in the line, not just the leading run - see
        # docstring point 1 above.
        paragraph = paragraph.expandtabs(4)
        leading_ws_len = len(paragraph) - len(paragraph.lstrip(" "))
        indent = paragraph[:leading_ws_len]
        words = paragraph[leading_ws_len:].split(" ")
        line = indent
        first_word_on_line = True
        max_word_len = max(max_chars - len(indent), 1)
        for w in words:
            # Hard-split any word too long to ever fit on an indented line
            # by itself, regardless of what else is already on the current
            # line - see docstring point 2 above. Split on grapheme
            # clusters (_grapheme_clusters), not raw code units, so a
            # combining mark is never separated from its base character -
            # see that function's docstring for why.
            clusters = _grapheme_clusters(w)
            chunks = [
                "".join(clusters[i:i + max_word_len])
                for i in range(0, len(clusters), max_word_len)
            ] or [w]
            for chunk in chunks:
                candidate = f"{line}{'' if first_word_on_line else ' '}{chunk}"
                if len(candidate) > max_chars and not first_word_on_line:
                    out.append(line)
                    line = f"{indent}{chunk}"
                    first_word_on_line = False
                else:
                    line = candidate
                    first_word_on_line = False
        out.append(line)
    return out


def _render_lines(lines: List[str]) -> Image.Image:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    line_height = FONT_SIZE + LINE_SPACING
    width = round(MARGIN * 2 + MAX_CHARS_PER_LINE * CHAR_WIDTH_PX)
    height = MARGIN * 2 + line_height * max(len(lines), 1)

    img = Image.new("L", (width, height), color=255)  # white background, grayscale
    draw = ImageDraw.Draw(img)
    y = MARGIN
    for line in lines:
        draw.text((MARGIN, y), line, fill=0, font=font)
        y += line_height
    return img


def _ocr(img: Image.Image) -> str:
    """OCR via word-level bounding boxes rather than plain image_to_string.

    Plain string OCR discards leading whitespace per line (not visible
    "content" to a text recognizer), which would silently destroy code
    indentation on the round trip. Reconstructing indentation from each
    line's first word's pixel x-offset (which we control, since we rendered
    the image) recovers it exactly.
    """
    config = "--psm 6"
    data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)

    # tesseract's line_num resets per block/paragraph, so the sort key must be
    # the full (block_num, par_num, line_num) tuple plus top-pixel position,
    # not line_num alone - otherwise lines from different blocks collide and
    # get merged out of order.
    lines: dict[tuple[int, int, int], list[tuple[int, str]]] = {}
    tops: dict[tuple[int, int, int], int] = {}
    for i in range(len(data["text"])):
        word = data["text"][i]
        if not word.strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        left = data["left"][i]
        lines.setdefault(key, []).append((left, word))
        tops[key] = data["top"][i]

    out_lines = []
    sorted_keys = sorted(lines, key=lambda k: tops[k])
    line_height = FONT_SIZE + LINE_SPACING
    prev_top: Optional[int] = None
    for idx, key in enumerate(sorted_keys):
        if prev_top is not None:
            gap = tops[key] - prev_top
            blank_count = round(gap / line_height) - 1
            out_lines.extend([""] * max(0, blank_count))
        elif idx == 0:
            # Leading blank render lines (before the first detected word on
            # this page) produce no bounding box at all, so there is no
            # "prev_top" to diff against the way the inter-line gap above
            # does. Without this, a page that starts with one or more blank
            # lines (e.g. the second half of a paragraph break that landed
            # exactly on a pagination boundary) silently loses them: OCR
            # only ever reconstructs blank lines as GAPS between two known
            # text lines, never as a gap between the top margin and the
            # first text line. Confirmed empirically: a page whose first
            # render line is blank came back from _ocr() missing that line
            # entirely, round 5 review, verified via a direct retype() call
            # on text engineered to land a blank line at a page boundary.
            leading_gap = tops[key] - MARGIN
            leading_blank_count = round(leading_gap / line_height)
            out_lines.extend([""] * max(0, leading_blank_count))
        words = sorted(lines[key], key=lambda t: t[0])
        first_left = words[0][0]
        indent = " " * max(0, round((first_left - MARGIN) / CHAR_WIDTH_PX))
        out_lines.append(indent + " ".join(w for _, w in words))
        prev_top = tops[key]

    return "\n".join(out_lines)


def _postprocess_ocr(raw: str) -> str:
    """Undo wrapping artifacts: collapse the hard line-wraps we introduced for
    rendering back into the natural paragraph flow, while preserving blank
    lines (paragraph breaks) and intentional single newlines are NOT
    reconstructed perfectly here - by design, since we can't know which
    newlines were "real" vs render-wrap without a marker. See note in README.

    Trailing blank lines are stripped here unconditionally (OCR sometimes
    appends a stray blank at the very end that was never in the source);
    the caller (`_retype_page`) is responsible for restoring however many
    trailing blank lines the real original actually had, since this
    function has no way to tell "OCR noise" from "a real trailing blank
    the original text intentionally ended with" on its own.
    """
    # Tesseract tends to leave trailing whitespace; normalize line endings.
    lines = [l.rstrip() for l in raw.split("\n")]
    # Drop a possible trailing blank line OCR appends.
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _char_level_damage(original: str, ocr_text: str, max_report: int = 10) -> List[str]:
    """Report individual character substitutions the OCR round trip made,
    independent of the aggregate similarity ratio. A high overall ratio can
    hide a small number of semantically important substitutions (e.g.
    accented letters silently flattened to their ASCII equivalent by the OCR
    engine, even though the font rendered the correct glyph) - this walks
    the actual diff opcodes and surfaces every 'replace' span so the caller
    can see what specifically changed, not just how much. Capped at
    `max_report` entries for readability; see `_has_structural_damage` for
    the uncapped safety check used as the actual suspect gate.
    """
    sm = difflib.SequenceMatcher(None, original, ocr_text)
    reports = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace" and len(reports) < max_report:
            reports.append(f"{original[i1:i2]!r} -> {ocr_text[j1:j2]!r}")
    return reports


def _has_structural_damage(original: str, ocr_text: str) -> bool:
    """True if the OCR round trip changed, deleted, or inserted any
    non-whitespace content, OR changed the leading indentation of any line
    - not capped by `max_report`, not limited to non-ASCII characters.

    Two separate checks, deliberately:
    1. Non-whitespace changes (as before): a damaging edit can be pure
       ASCII (a fence marker like three backticks mangled into something
       else, or dropped entirely) and still score an aggregate diff_ratio
       above 0.9, so ratio alone is not a safe gate for structured/code
       content.
    2. Leading-indentation changes, even when every changed character is
       whitespace: confirmed empirically (independent review) that OCR can
       silently turn an 8-space indent into 4-space, which still scores
       ratio 0.86-0.97 and is pure whitespace, yet changes what the code
       means (Python nesting, YAML structure, etc). A generic
       whitespace-only diff elsewhere (e.g. tesseract inserting a stray
       space between two words, which is genuinely cosmetic) is NOT
       flagged by this second check - only a change to a line's OWN
       leading-indent run counts, since that is the only whitespace
       category that is reliably structural rather than cosmetic.
    """
    sm = difflib.SequenceMatcher(None, original, ocr_text)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        changed = original[i1:i2] + ocr_text[j1:j2]
        if changed.strip():
            return True

    def _leading_whitespace(line: str) -> str:
        match = re.match(r"[ \t]*", line)
        return match.group(0) if match else ""

    orig_lines = original.split("\n")
    ocr_lines = ocr_text.split("\n")
    if len(orig_lines) == len(ocr_lines):
        for o_line, c_line in zip(orig_lines, ocr_lines):
            if _leading_whitespace(o_line) != _leading_whitespace(c_line):
                return True

    return False


def _retype_page(render_lines: List[str], diff_threshold: float) -> RetypeResult:
    """Render/OCR a single page of already-wrapped lines (wrapping and
    pagination both happen once, up front, in retype())."""
    img = _render_lines(render_lines)
    raw_ocr = _ocr(img)
    clean = _postprocess_ocr(raw_ocr)

    # Compare against the *wrapped* original (same line breaks) for a fair diff,
    # not the pre-wrap original which has different newline positions.
    wrapped_original = "\n".join(render_lines)

    # _postprocess_ocr unconditionally strips trailing blank lines (it has
    # no way to tell OCR noise from a real trailing blank). Restore however
    # many trailing blank lines the original actually had, so an
    # intentional trailing blank (e.g. a markdown paragraph separator at
    # the very end of the response) survives the round trip instead of
    # silently vanishing. Confirmed empirically: "...nominal.\n\n" came
    # back as "...nominal." with no indication anything was dropped, and
    # the suspect gate did not catch it since it only checks non-whitespace
    # content and leading indentation, never trailing blank lines.
    orig_lines = wrapped_original.split("\n")
    trailing_blanks = 0
    for line in reversed(orig_lines):
        if line.strip() == "":
            trailing_blanks += 1
        else:
            break
    if trailing_blanks:
        clean = clean + ("\n" * trailing_blanks)

    ratio = difflib.SequenceMatcher(None, wrapped_original, clean).ratio()

    diff_lines = list(
        difflib.unified_diff(
            wrapped_original.splitlines(),
            clean.splitlines(),
            lineterm="",
            n=1,
        )
    )

    char_damage = _char_level_damage(wrapped_original, clean)
    # Reject ANY structural damage (not just non-ASCII replacements): a
    # damaging edit can be pure ASCII - e.g. a code fence marker mangled or
    # dropped - and still score an aggregate ratio above threshold. Ratio
    # alone is not a safe gate for structured/code content; see
    # _has_structural_damage for why.
    structural_damage = _has_structural_damage(wrapped_original, clean)

    return RetypeResult(
        clean_text=clean,
        original_text=wrapped_original,
        diff_ratio=ratio,
        diff_lines=diff_lines,
        suspect=(ratio < diff_threshold) or structural_damage,
        char_damage=char_damage,
    )


def _split_into_pages(render_lines: List[str], max_lines: int = MAX_LINES_PER_PAGE) -> List[List[str]]:
    """Split already-wrapped render lines into page-sized chunks on
    blank-line (paragraph) boundaries where possible, falling back to a
    hard line-count cut if a single paragraph alone exceeds max_lines.

    Must operate on WRAPPED lines, not raw source lines: a single long
    paragraph is one source line but many render lines once word-wrapped to
    MAX_CHARS_PER_LINE, and it's the render-line count that drives OCR
    image height (confirmed empirically: a 6900-character single paragraph
    passed the old raw-line check trivially since it was one source line,
    then rendered as a single ~70-line image, well past the 60-line
    reliability ceiling this function exists to enforce).

    Returns (pages, hard_cut_after) where hard_cut_after[i] is True if page
    i was cut mid-paragraph (no blank line at the boundary) rather than on
    a natural paragraph break. The caller needs this to know which page
    joins in retype() are two independently-OCR'd halves of what was
    originally one continuous run of text, with no blank line between them
    to lean on when reassembling.
    """
    if len(render_lines) <= max_lines:
        return [render_lines], [False]

    pages: List[List[str]] = []
    hard_cut_after: List[bool] = []
    current: List[str] = []
    for line in render_lines:
        current.append(line)
        at_blank_boundary = line.strip() == ""
        if len(current) >= max_lines and at_blank_boundary:
            pages.append(current)
            hard_cut_after.append(False)
            current = []
        elif len(current) >= max_lines:
            # No blank line at exactly max_lines: cut here rather than
            # waiting for one, so a page is never more than max_lines long
            # regardless of paragraph structure. This is the seam that
            # needs a marker on rejoin: page N's last rendered line and
            # page N+1's first rendered line were split out of what was,
            # in the original text, one continuous run with no blank line
            # between them.
            pages.append(current)
            hard_cut_after.append(True)
            current = []
    if current:
        pages.append(current)
        hard_cut_after.append(False)
    return pages, hard_cut_after


def retype(text: str, diff_threshold: float = 0.90) -> RetypeResult:
    """Render `text` to a bitmap, OCR it back, and return the OCR'd string plus
    a diff report against the original. If similarity falls below
    `diff_threshold`, `suspect=True` is set so the caller can decide to fall
    back to the original text rather than risk a garbled OCR read silently
    going out.

    If `text` contains characters the render font can't draw (emoji, CJK,
    most non-Latin scripts), the round trip is skipped entirely and the
    original text is returned unchanged with `skipped=True`: silently
    rendering those as tofu boxes and OCR-guessing them back would corrupt
    meaning-bearing content, and an aggregate similarity ratio over a long
    string is not sensitive enough to catch that kind of concentrated,
    small-character-count damage (confirmed empirically: an emoji-corrupting
    round trip can still score ratio > 0.9).

    Large input is paginated internally (see MAX_LINES_PER_PAGE): each page
    is rendered/OCR'd/diffed independently and the results are merged, so a
    long paste behaves the same as many small calls rather than one huge,
    OCR-unreliable image.
    """
    if not text or not text.strip():
        return RetypeResult(clean_text=text, original_text=text, diff_ratio=1.0)

    if not FONT_PATH:
        # No usable font on this host (see the warning emitted at import
        # time) - skip the round trip entirely rather than crashing. The
        # caller falls back to stage-1-only output, same as the
        # unsupported-character path below.
        return RetypeResult(
            clean_text=text,
            original_text=text,
            diff_ratio=1.0,
            skipped=True,
            skip_reason="no monospace font found on this host; set HERMES_STRIP_METADATA_FONT",
        )

    unsupported = find_unsupported_chars(text)
    if unsupported:
        return RetypeResult(
            clean_text=text,
            original_text=text,
            diff_ratio=1.0,
            skipped=True,
            skip_reason=f"font cannot render: {' '.join(unsupported)}",
        )

    pages, hard_cut_after = _split_into_pages(_wrap_for_render(text))
    if len(pages) == 1:
        return _retype_page(pages[0], diff_threshold)

    results = [_retype_page(p, diff_threshold) for p in pages]
    # Rejoin pages. A normal page boundary (hard_cut_after[i] is False) was
    # cut at a blank line, so a plain "\n" join reproduces the original
    # blank-line paragraph break correctly. A hard mid-paragraph cut
    # (hard_cut_after[i] is True) is different: page i's last rendered line
    # and page i+1's first rendered line are two halves of what was one
    # continuous wrapped line in the original text (the word-wrap in
    # _wrap_for_render breaks on a space, not a newline), so joining them
    # with "\n" would silently insert a line break that was never there,
    # and _postprocess_ocr has already stripped each page's own trailing/
    # leading whitespace independently, so naively concatenating also risks
    # losing the single space that belonged between the last word of page i
    # and the first word of page i+1. Join hard-cut seams with a single
    # space instead, so two independently-OCR'd halves of one continuous
    # run reassemble into text, not two arbitrarily-spliced fragments.
    merged_parts: List[str] = []
    for i, r in enumerate(results):
        merged_parts.append(r.clean_text)
        if i < len(results) - 1:
            merged_parts.append(" " if hard_cut_after[i] else "\n")
    merged_clean = "".join(merged_parts)
    merged_diff_lines: List[str] = []
    merged_char_damage: List[str] = []
    for i, r in enumerate(results):
        merged_diff_lines.extend(r.diff_lines)
        merged_char_damage.extend(r.char_damage)
    avg_ratio = sum(r.diff_ratio for r in results) / len(results)
    any_suspect = any(r.suspect for r in results)

    return RetypeResult(
        clean_text=merged_clean,
        original_text=text,
        diff_ratio=avg_ratio,
        diff_lines=merged_diff_lines,
        suspect=any_suspect,
        char_damage=merged_char_damage,
    )
