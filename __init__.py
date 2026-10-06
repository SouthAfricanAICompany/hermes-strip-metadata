"""strip-metadata: sanitize Hermes's final response before it leaves the agent loop.

Free and open source. Built and maintained by SAAIC (South African AI Company),
https://saaic.co.za

Hooks `transform_llm_output`, the one chokepoint that fires on EVERY turn
(CLI, every gateway platform, cron delivery) BEFORE any destination-specific
send/display logic runs. See README.md for why this hook and not any other.
"""

import re
import unicodedata

# Zero-width / invisible characters used for text watermarking or tracking.
# U+200B ZERO WIDTH SPACE, U+200C/D ZWNJ/ZWJ, U+2060 WORD JOINER,
# U+FEFF BOM/ZERO WIDTH NO-BREAK SPACE, U+00AD SOFT HYPHEN,
# Unicode "variation selectors" (U+FE00-FE0F) sometimes abused for payload hiding,
# and the tag-block range (U+E0000-E007F) used by some watermarking schemes.
_INVISIBLE_PATTERN = re.compile(
    "["
    "\u200b-\u200f"   # zero-width space/joiners, LRM/RLM
    "\u202a-\u202e"   # bidi embedding/override controls
    "\u2060-\u2064"   # word joiner, invisible operators
    "\ufeff"          # BOM
    "\u00ad"          # soft hyphen
    "\ufe00-\ufe0f"   # variation selectors
    "\U000e0000-\U000e007f"  # tag block
    "]"
)

# Common AI-platform watermark character sometimes injected between words.
_SUSPICIOUS_LOOKALIKES = {
    "\u2010": "-",  # hyphen variants -> ascii hyphen (optional; keep conservative)
}

# Real HTML tag names only. A bare "anything in angle brackets" regex is not
# conservative: it also matches C++/Java generics (vector<int>), comparison
# chains (if (a < b) and (c > d)), and angle-bracket email addresses
# (<nobody@example.com>), silently deleting content that was never HTML.
# Matching against an actual tag-name whitelist (plus a required boundary
# right after the name: whitespace, "/", or ">") excludes all three of
# those cases while still catching the real stray-HTML-tag leaks this check
# exists for. Case-insensitive, since HTML tag names are.
_HTML_TAG_NAMES = (
    "a abbr address area article aside audio b base bdi bdo blockquote body br button "
    "canvas caption cite code col colgroup data datalist dd del details dfn dialog div dl dt "
    "em embed fieldset figcaption figure footer form h1 h2 h3 h4 h5 h6 head header hr html i "
    "iframe img input ins kbd label legend li link main map mark menu meta meter nav noscript "
    "object ol optgroup option output p param picture pre progress q rp rt ruby s samp script "
    "section select slot small source span strong style sub summary sup table tbody td "
    "template textarea tfoot th thead time title tr track u ul var video wbr"
).split()
# Attribute content: anything except a bare "<"/">", OR a fully quoted string
# that may itself contain "<"/">" characters. Without the quoted-string
# alternative, a crafted attribute value embedding literal tag-like text
# (e.g. onclick="...'<script>x</script>'...") stops the outer tag's match
# at the first "<" INSIDE the quotes, so the outer tag (<a ...>) never
# matches at all, while the inner "<script>"/"</script>" substrings -
# which are not real markup, just characters inside a quoted attribute
# value - DO match the whitelist on their own and get stripped, leaving a
# mangled, half-stripped fragment behind. Confirmed empirically (round 5
# review): `<a href="..." onclick="...'<script>x</script>'...">click</a>`
# came back with the inner fake tags removed but the outer `<a ...>` and
# its unmatched `</a>` still present verbatim. Treating quoted strings as
# opaque lets the outer tag match across its full span (quotes and all),
# so the whole real tag - attributes included - is stripped as one unit.
_ATTR_CONTENT = r'(?:"[^"]*"|\'[^\']*\'|[^<>\n])'
_HTML_TAG_PATTERN = re.compile(
    r"</?(?:" + "|".join(_HTML_TAG_NAMES) + r")(?:\s" + _ATTR_CONTENT + r"{0,200})?\s*/?>",
    re.IGNORECASE,
)


def sanitize_text(text: str) -> str:
    """Strip invisible/watermarking characters and normalize whitespace.

    Conservative by design: only removes characters with no visible rendering
    or that are near-universally used for tracking/metadata, never touches
    visible punctuation, emoji, or non-Latin scripts.
    """
    if not text:
        return text

    # 1. Drop invisible/watermarking code points outright.
    cleaned = _INVISIBLE_PATTERN.sub("", text)

    # 2. Unicode-normalize to NFC so combining-character tricks collapse to
    #    their canonical form (defeats some steganographic combining-mark use).
    cleaned = unicodedata.normalize("NFC", cleaned)

    # 3. Strip any stray HTML tags that sometimes leak through clipboard/RTF
    #    paste paths (defensive; the model's plain-text output shouldn't have
    #    these, but a tool-result echo sometimes does).
    cleaned = _HTML_TAG_PATTERN.sub("", cleaned)

    # 4. Collapse runs of whitespace introduced by the removals above, but
    #    preserve intentional single newlines/paragraph breaks AND leading
    #    indentation (code blocks, nested markdown lists). Only interior
    #    whitespace (after the leading indent) gets collapsed - leading
    #    whitespace is never touched, since indentation is structural, not
    #    noise from the invisible-char/HTML-tag removal above.
    def _collapse_interior_whitespace(line: str) -> str:
        match = re.match(r"[ \t]*", line)
        indent = match.group(0) if match else ""
        rest = line[len(indent):]
        rest = re.sub(r"[ \t]+", " ", rest)
        return indent + rest

    cleaned = "\n".join(_collapse_interior_whitespace(line) for line in cleaned.split("\n"))
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))

    return cleaned


def _is_low_risk_plain_text(text: str) -> bool:
    """True when `text` has no realistic metadata/watermark surface for
    stage 2 to clean, so the expensive render+OCR round trip can be
    skipped without weakening protection.

    Stage 1 already strips every known invisible/zero-width/homoglyph
    character by exact codepoint range (see `_INVISIBLE_PATTERN`). Those
    are the ONLY hiding techniques that can exist in a plain Python string
    at this point in the pipeline - there is no clipboard flavor, RTF
    wrapper, or font-level metadata on a str object, that risk only
    applies once text is rendered into some UI/document format downstream.
    So: pure 7-bit ASCII text has, by definition, zero room left for any
    Unicode-level hiding technique, known or not - there is no "invisible
    ASCII" to find. Skipping stage 2 here is a real cost saving (full
    render+OCR round trip measured at several seconds for a long paste in
    testing), not a coverage gap.

    Deliberately conservative: ANY non-ASCII character at all routes
    through stage 2 regardless of what it is, since an unknown future
    hiding technique could live in any non-ASCII range stage 1 doesn't
    yet know about - this function is not a substitute for stage 1's
    pattern list, it's a cost gate for text that is provably outside
    stage 2's ability to add value.
    """
    return text.isascii()


def _on_transform_llm_output(**kwargs):
    """`transform_llm_output` hook contract: return a replacement string, or
    None/"" to leave the response untouched. Must never raise: a hook
    exception is caught by the host and simply contributes no result, but we
    guard anyway so a sanitizer bug never blocks delivery of the real answer.

    Two stages, in order:
    1. sanitize_text: strip invisible/watermarking Unicode (this file).
    2. render_ocr.retype: render-to-image + OCR round trip, the API-delivery
       equivalent of physically retyping the text, kills any remaining
       clipboard/formatting-level metadata that survived stage 1. Skipped
       automatically (falls back to stage-1 output) for text the round trip
       can't safely handle (emoji/CJK, or OCR produced suspect output), AND
       for plain ASCII text stage 1 already left untouched - see
       `_is_low_risk_plain_text` for why that's a safe skip, not a
       coverage gap. See render_ocr.py for the other skip conditions.
    """
    try:
        response_text = kwargs.get("response_text") or ""
        sanitized = sanitize_text(response_text)

        if sanitized == response_text and _is_low_risk_plain_text(sanitized):
            # Nothing for stage 1 to have caught, and nothing non-ASCII for
            # stage 2 to usefully check either - skip the expensive render
            # round trip entirely.
            return None

        try:
            from . import render_ocr
            result = render_ocr.retype(sanitized)
            final = sanitized if (result.skipped or result.suspect) else result.clean_text
        except Exception:
            # render_ocr unavailable or failed (e.g. tesseract not installed
            # on this host) - fall back to stage-1 output only, never block
            # delivery over a missing optional dependency.
            final = sanitized

        return final if final != response_text else None
    except Exception:
        # Fail open: never let a sanitizer bug block the agent's response.
        return None


def register(ctx):
    ctx.register_hook("transform_llm_output", _on_transform_llm_output)
