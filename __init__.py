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
# U+200B ZERO WIDTH SPACE, U+200C ZWNJ, U+2060 WORD JOINER,
# U+FEFF BOM/ZERO WIDTH NO-BREAK SPACE, U+00AD SOFT HYPHEN,
# Unicode "variation selectors" (U+FE00-FE0F) sometimes abused for payload hiding,
# and the tag-block range (U+E0000-E007F) used by some watermarking schemes.
#
# U+200D ZWJ (ZERO WIDTH JOINER) is handled separately by
# `_strip_watermark_zwj` below, NOT blanket-stripped here. Round 7 review:
# ZWJ is the real, required glue character in standard compound emoji
# (family emoji, "woman health worker", etc. - any ZWJ emoji sequence).
# Confirmed empirically: sanitize_text() on the 4-person family emoji
# (U+1F468 ZWJ U+1F469 ZWJ U+1F467 ZWJ U+1F466) used to strip all 3 ZWJ
# characters, turning one compound glyph into 4 unrelated emoji - visible
# content corruption of completely ordinary text, not a watermark hit.
# ZWJ is still a real steganographic vector when it sits between ordinary
# (non-emoji) characters, e.g. inserted mid-word to hide a payload - that
# case is still stripped, just contextually, by `_strip_watermark_zwj`.
_INVISIBLE_PATTERN = re.compile(
    "["
    "\u200b"          # zero-width space
    "\u200c"          # ZWNJ (word-ligature control, not emoji-joining)
    "\u200e\u200f"    # LRM/RLM
    "\u202a-\u202e"   # bidi embedding/override controls
    "\u2060-\u2064"   # word joiner, invisible operators
    "\ufeff"          # BOM
    "\u00ad"          # soft hyphen
    "\ufe00-\ufe0f"   # variation selectors
    "\U000e0000-\U000e007f"  # tag block
    "]"
)

# Characters that commonly serve as the BASE of a real Unicode ZWJ emoji
# sequence (per Unicode's own emoji-zwj-sequences.txt): the "people" block
# (man/woman/boy/girl and their many profession/hair/feature variants),
# plus the small set of symbols known to appear on the far side of a ZWJ in
# documented sequences (medical symbol, heart, airplane, gender symbols,
# rainbow, flag base, kiss mark, etc). Deliberately curated and narrow,
# NOT "any codepoint in the whole emoji plane" - see round 8 review below
# for why the broad version was unsafe.
#
# Round 8 review found the original _EMOJI_ISH check (any codepoint in
# U+2600-27BF or U+1F000-1FFFF) was far too permissive: it only required
# that SOME symbol sit on each side of the ZWJ, not that the two symbols
# actually form a real, documented compound-emoji relationship. Confirmed
# empirically: sanitize_text("\u2714\u200d\u2605") (heavy check mark + ZWJ
# + black star - not a defined emoji sequence) used to keep the ZWJ
# untouched, because both neighbors happen to fall in the dingbat range.
# That is the exact steganographic payload this function exists to strip,
# surviving because an attacker can pick any two symbols from a huge
# range rather than needing a real emoji pair. Narrowing the check to
# only the codepoints that actually appear as ZWJ-sequence components
# closes that gap while still preserving every real compound emoji this
# plugin has been tested against (family emoji, woman health worker).
_ZWJ_JOINABLE = re.compile(
    "["
    "\U0001f466-\U0001f487"  # boy..person-with-blond-hair (core "people" base range)
    "\U0001f9b0-\U0001f9ff"  # supplemental people/body features (red hair, beard, etc.)
    "\U0001f3f3"             # white flag (rainbow-flag ZWJ sequences)
    "\U0001f308"             # rainbow
    "\U0001f48b"             # kiss mark (kiss ZWJ sequences)
    "\u2695"                 # medical symbol (health worker sequences)
    "\u2696"                 # scales (judge sequences)
    "\u2708"                 # airplane (pilot sequences)
    "\u2764"                 # heavy black heart (couple/kiss sequences)
    "\u2640\u2642"           # female/male signs (gendered-profession sequences)
    "\u2620"                 # skull and crossbones (pirate sequences)
    "]"
)


def _strip_watermark_zwj(text: str) -> str:
    """Remove ZWJ (U+200D) everywhere EXCEPT where it joins two characters
    that actually appear as components of a real, documented Unicode ZWJ
    emoji sequence (see `_ZWJ_JOINABLE`'s docstring for why this is
    narrower than "any emoji-ish symbol")."""
    if "\u200d" not in text:
        return text

    def _keep_or_strip(m: "re.Match") -> str:
        idx = m.start()
        before = text[idx - 1] if idx > 0 else ""
        after = text[idx + 1] if idx + 1 < len(text) else ""
        if _ZWJ_JOINABLE.match(before) and _ZWJ_JOINABLE.match(after):
            return m.group(0)
        return ""

    return re.sub("\u200d", _keep_or_strip, text)

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
    r"</?(?:" + "|".join(_HTML_TAG_NAMES) + r")\b(?P<attrs>(?:\s" + _ATTR_CONTENT + r"{0,200})?)\s*/?>",
    re.IGNORECASE,
)


def _strip_html_tags(text: str) -> str:
    """Strip real HTML tags, but reject a candidate match whose "attributes"
    are non-empty plain words containing no "=" anywhere.

    Round 7 review found that a bare tag-name whitelist plus a lenient
    "anything that isn't < or >" attribute blob matches ordinary prose and
    code as if it were markup, whenever a short tag name (a, b, i, s, q, u,
    p, li, dd, ...) happens to collide with a variable name or word used
    right after a "<". Confirmed empirically:
      "results show x<a and y>b, so x<a<b is false when b<a"
      -> used to come back as "results show xb, so x<a<b is false when b<a"
         (the whole "<a and y>" comparison clause silently deleted)
      "if x<li and y>li: print('ok')"
      -> used to come back as "if xli: print('ok')" (working code destroyed)

    Real HTML attributes are almost always name=value pairs (href=, src=,
    class=, id=...). A run of bare words with no "=" anywhere in the
    attribute span is far more consistent with a comparison/code false
    positive than genuine markup, so such a match is left untouched rather
    than stripped. Known, accepted trade-off: a stray leaked HTML tag using
    only bare boolean attributes (e.g. "<input disabled>") will no longer
    be stripped either - rare in practice, and this is a defensive layer
    for accidental tag leakage, not a guarantee against deliberately
    crafted markup.
    """

    def _repl(m: "re.Match") -> str:
        attrs = m.group("attrs")
        if attrs and attrs.strip() and "=" not in attrs:
            return m.group(0)
        return ""

    return _HTML_TAG_PATTERN.sub(_repl, text)


# Cleanup pass for a tag the pattern above cannot fully match: an attribute
# value with an unterminated quote (no matching closing quote anywhere in
# the text) has no legal way to reach the required trailing ">" within the
# quote-balanced grammar above, so the whole tag fails to match and its
# literal opening fragment ("<a href="unterminated...) survives verbatim,
# even though real, well-formed tags around it DID get stripped by the
# pattern above. Confirmed empirically (round 6 review):
# '<a href="unterminated clicked</a> more text <b>bold</b>' sanitized to
# '<a href="unterminated clicked more text bold' - the well-formed </a>,
# <b>, </b> were correctly stripped, but the broken <a href="... opener
# survived as a literal, un-neutralized "<" in text that is supposed to
# guarantee none survive from real markup.
#
# Deliberately conservative about what it removes: only the bare opening
# "<tagname" / "</tagname" token itself, nothing after it. A broader
# match that consumed "everything up to the next < or end of line" would
# delete real surrounding prose that has nothing to do with the broken
# tag (e.g. "more text bold" in the repro above is legitimate content,
# not markup, and must survive). Stripping just the opening token removes
# the only "<" character the broken tag introduced - which is the actual
# contract (no literal angle brackets from real markup survive) - while
# leaving whatever attribute-like text followed it intact as harmless
# plain text, same as any other malformed snippet sanitize_text doesn't
# try to fully parse.
#
# Round 7 review found this pass is itself too eager when the "<tagname"
# token is the deliberate subject of discussion rather than a failed
# markup attempt, e.g. documentation or a chat message literally
# explaining what "<script" means, especially inside a markdown code
# span (backticks). Confirmed empirically:
#   'explain what `<script` means ... like this unterminated example:
#    `<script src="x` in a tutorial.'
#   -> both literal "<script" occurrences, each inside its own pair of
#      backticks, were deleted even though neither one is a broken
#      markup attempt - they're text about the token itself.
# Fixed by skipping any orphan-opener candidate that sits inside a
# backtick-delimited code span: if an odd number of backticks appear
# before the match on the same line, the match is inside an open code
# span and is left untouched (the whole point of a code span is "render
# this literally, don't interpret it as markup").
# Round 7 also found this pass doesn't share _strip_html_tags's "bare-word
# attributes with no '=' are probably code/comparison text, not markup"
# rejection - so a short tag name rejected by the main pattern for that
# reason (e.g. the "<a" in "x<a and y>b") still got eaten here as an
# "orphan opener", even though it was never a broken tag to begin with.
# Confirmed empirically: "results show x<a and y>b, so x<a<b is false
# when b<a" - after _strip_html_tags correctly left it alone, this pass
# still deleted the lone "<a" token. Round 7's fix required an "=" to
# appear ANYWHERE between the match and the next "<"/end-of-line - but
# round 8 review found that's too loose: a real "=" belonging to
# unrelated prose much later on the same line can make a genuinely
# harmless "<tagname" look like it has attributes. Confirmed empirically:
# "<b the result is x=y so it stands alone" used to come back with the
# harmless "<b" deleted, purely because an unrelated "x=y" appeared many
# words later on the same line, nowhere near the tag token.
#
# Fixed by requiring the "=" to belong to the FIRST word immediately
# after the tag name (after optional whitespace) - i.e. the shape of a
# real attribute, "<a href=...", "<a   data-x=...", not just "some '='
# exists somewhere further down the line". A genuine broken tag's first
# attribute is adjacent to the tag name; a comparison/code false positive
# has ordinary words in between with no "=" directly attached to any of
# them.
_ORPHAN_ATTR_PATTERN = re.compile(r"\s+[^\s<>=]+\s*=")
_ORPHAN_TAG_PATTERN = re.compile(
    r"</?(?:" + "|".join(_HTML_TAG_NAMES) + r")\b",
    re.IGNORECASE,
)

# Round 8 review also found the backtick-code-span guard below is
# per-line only and has no notion of a TRIPLE-backtick fenced block
# spanning multiple lines (``` ... ```), as opposed to a single-line
# `inline code span`. Confirmed empirically: an "<a href=..." example
# deliberately placed on an interior line of a fenced code block (that
# specific line itself has zero backticks on it, so the per-line odd/even
# count saw it as "not inside a span") still got its "<a" stripped,
# destroying a documentation example the fence exists to preserve
# verbatim. Fixed by tracking fence state across the whole text first:
# any line between an opening and closing ``` fence line is left
# completely untouched by this pass, regardless of what it contains.
_FENCE_LINE = re.compile(r"^ {0,3}`{3,}")


def _strip_orphan_tag_openers(text: str) -> str:
    lines = text.split("\n")
    out_lines = []
    in_fence = False
    for line in lines:
        if _FENCE_LINE.match(line):
            in_fence = not in_fence
            out_lines.append(line)
            continue
        if in_fence:
            out_lines.append(line)
            continue

        def _repl(m: "re.Match") -> str:
            # Odd number of backticks before the match = inside an open
            # inline code span on this line = leave it alone, it's
            # literal text.
            if line[: m.start()].count("`") % 2 == 1:
                return m.group(0)
            # The first word right after the tag name must itself be
            # followed by "=" to read as a genuine attribute. Anything
            # else (plain words, or an "=" only appearing later and
            # unattached to the immediately-following word) reads as
            # plain prose/code that merely starts with a tag-name-shaped
            # word, not a broken tag.
            tail = line[m.end():]
            if _ORPHAN_ATTR_PATTERN.match(tail):
                return ""
            return m.group(0)

        out_lines.append(_ORPHAN_TAG_PATTERN.sub(_repl, line))
    return "\n".join(out_lines)



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

    # 1b. ZWJ (U+200D) needs context: strip it everywhere except where it
    #     joins two emoji into one real compound glyph (see
    #     _strip_watermark_zwj's docstring - round 7 fix).
    cleaned = _strip_watermark_zwj(cleaned)

    # 2. Unicode-normalize to NFC so combining-character tricks collapse to
    #    their canonical form (defeats some steganographic combining-mark use).
    cleaned = unicodedata.normalize("NFC", cleaned)

    # 3. Strip any stray HTML tags that sometimes leak through clipboard/RTF
    #    paste paths (defensive; the model's plain-text output shouldn't have
    #    these, but a tool-result echo sometimes does). Rejects bare-word
    #    "attributes" with no "=" at all - see _strip_html_tags's docstring
    #    (round 7 fix for the <a>/<b>/<li>-as-variable-name false positive).
    cleaned = _strip_html_tags(cleaned)

    # 3b. Second pass: remove any orphaned tag-opener fragment left behind
    #     by an unterminated-quote attribute that couldn't match the
    #     quote-balanced pattern above (see _ORPHAN_TAG_PATTERN docstring).
    #     Runs after 3 so it only ever sees openers that genuinely failed
    #     to close, never a well-formed tag (those are already gone).
    #     Skips matches inside an open backtick code span - see
    #     _strip_orphan_tag_openers's docstring (round 7 fix).
    cleaned = _strip_orphan_tag_openers(cleaned)

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
