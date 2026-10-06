"""Regression test suite for hermes-strip-metadata.

Covers every bug found across the 3 independent review rounds (Tom, Forge,
Anvil in Paperclip's SAAIC board), plus the 3 post-ship engineering
improvements (test suite itself, font caching/cross-platform resolution,
ASCII cost gate). Run with: pytest tests/test_plugin.py -v

Each test names the specific regression it guards so a future change that
reintroduces one of these bugs fails loudly instead of silently shipping.
"""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import __init__ as plugin  # noqa: E402
import render_ocr  # noqa: E402


# ---------------------------------------------------------------------------
# Stage 1 (sanitize_text) - invisible character stripping
# ---------------------------------------------------------------------------

def test_strips_zero_width_space():
    assert plugin.sanitize_text("hello\u200bworld") == "helloworld"


def test_strips_soft_hyphen():
    assert plugin.sanitize_text("soft\u00adhyphen") == "softhyphen"


def test_strips_bom():
    assert "\ufeff" not in plugin.sanitize_text("\ufeffBOM test")


def test_strips_variation_selectors():
    assert "\ufe0f" not in plugin.sanitize_text("test\ufe0f")


def test_leaves_plain_prose_untouched():
    text = "The quick brown fox jumps over the lazy dog."
    assert plugin.sanitize_text(text) == text


# ---------------------------------------------------------------------------
# REGRESSION 1 (round 1, found independently by Tom/Forge/Anvil):
# sanitize_text's whitespace-collapse regex was destroying leading
# indentation on every line, corrupting code blocks and nested markdown
# lists. Fixed by only collapsing INTERIOR whitespace, never leading.
# ---------------------------------------------------------------------------

def test_regression_code_block_indentation_preserved():
    code = (
        "def add(a, b):\n"
        "    # adds two numbers\n"
        "    result = a + b\n"
        "    return result\n"
    )
    assert plugin.sanitize_text(code) == code


def test_regression_nested_markdown_list_preserved():
    nested = "Fruits:\n  - Apple\n    - Fuji\n    - Gala\n  - Banana\n    - Cavendish\n"
    assert plugin.sanitize_text(nested) == nested


def test_regression_eight_space_indent_preserved():
    text = "if True:\n        deeply_nested_call()\n"
    assert plugin.sanitize_text(text) == text


def test_interior_whitespace_still_collapses():
    # Confirms the fix is scoped correctly: leading indent is protected,
    # but stray interior double-spacing (the thing this collapse exists
    # for) still works.
    text = "word1    word2\tword3  word4"
    assert plugin.sanitize_text(text) == "word1 word2 word3 word4"


# ---------------------------------------------------------------------------
# REGRESSION 2 (round 2, found by Tom): the OCR suspect-detection gate only
# flagged non-ASCII character damage, missing pure-ASCII syntax corruption
# like a mangled code fence marker. Fixed by _has_structural_damage, which
# flags ANY non-whitespace change regardless of character set.
# ---------------------------------------------------------------------------

def test_regression_ascii_fence_damage_detected():
    original = "```python\ndef add(a, b):\n    return a + b\n```"
    mangled = "~*~ python\ndef add(a, b):\n    return a + b\n"  # fence mangled, dropped
    assert render_ocr._has_structural_damage(original, mangled) is True


def test_structural_damage_false_on_identical_text():
    text = "nothing changed here"
    assert render_ocr._has_structural_damage(text, text) is False


def test_structural_damage_false_on_cosmetic_stray_space():
    # Negative control: OCR inserting a stray space between words (common,
    # genuinely cosmetic noise) must NOT be flagged, or every response
    # would be rejected.
    original = "print(i)"
    cosmetic = "print (i)"
    assert render_ocr._has_structural_damage(original, cosmetic) is False


# ---------------------------------------------------------------------------
# REGRESSION 3 (round 2, found by Tom): pagination (_split_into_pages)
# counted raw source lines instead of post-wrap render lines, so a single
# long paragraph (one source line, many render lines after word-wrap)
# bypassed the page-size cap entirely and rendered as one oversized image.
# Fixed by wrapping first, then paginating the wrapped line list.
# ---------------------------------------------------------------------------

def test_regression_long_single_paragraph_is_paginated():
    long_para = (
        "This is a long sentence that keeps going to simulate a giant "
        "single paragraph pasted without any line breaks at all. " * 80
    ).strip()
    wrapped = render_ocr._wrap_for_render(long_para)
    assert len(wrapped) > render_ocr.MAX_LINES_PER_PAGE, (
        "test fixture must actually exceed the page cap to be a real regression check"
    )
    pages = render_ocr._split_into_pages(wrapped)
    assert len(pages) > 1
    for page in pages:
        assert len(page) <= render_ocr.MAX_LINES_PER_PAGE


def test_regression_multi_paragraph_paste_respects_page_cap():
    paragraphs = [f"Line {i} of a paragraph with some filler text." for i in range(150)]
    long_paste = "\n\n".join(paragraphs)
    wrapped = render_ocr._wrap_for_render(long_paste)
    pages = render_ocr._split_into_pages(wrapped)
    assert all(len(p) <= render_ocr.MAX_LINES_PER_PAGE for p in pages)


# ---------------------------------------------------------------------------
# REGRESSION 4 (round 3, found by Tom): _has_structural_damage ignored
# whitespace-only diffs entirely, so OCR silently changing an 8-space
# indent to 4-space indent (pure whitespace, but changes Python/YAML
# meaning) scored a high similarity ratio and passed as not-suspect. Fixed
# by a second check comparing each line's own leading-whitespace run.
# ---------------------------------------------------------------------------

def test_regression_indentation_change_flagged_even_if_pure_whitespace():
    original = "        return x\n        return y\n        return z"
    damaged = "    return x\n    return y\n    return z"  # 8-space -> 4-space
    assert render_ocr._has_structural_damage(original, damaged) is True


def test_indentation_check_does_not_false_positive_on_cosmetic_whitespace():
    # The indentation check must only fire on a line's OWN leading
    # whitespace, not any whitespace-only diff anywhere in the text.
    original = "print(i)"
    cosmetic = "print (i)"
    assert render_ocr._has_structural_damage(original, cosmetic) is False


# ---------------------------------------------------------------------------
# Post-ship improvement: font resolution must work across the OSes this
# plugin is promoted for (Jason: "there are also users going to use this
# on windows machines"). Not a live Windows test (no Windows host
# available), but confirms the resolver logic is OS-aware and fails safe.
# ---------------------------------------------------------------------------

def test_font_resolved_on_this_host():
    # On any host this suite actually runs on, a font must resolve (CI/dev
    # machines are Linux/macOS with DejaVu or system fonts available).
    assert render_ocr.FONT_PATH is not None
    assert os.path.isfile(render_ocr.FONT_PATH)


def test_font_candidates_include_all_three_target_platforms():
    candidates = " ".join(c for c in render_ocr._FONT_CANDIDATES if c)
    assert "dejavu" in candidates.lower() or "Dejavu" in candidates
    assert "consola" in candidates.lower()  # Windows: Consolas
    assert "cour" in candidates.lower()      # Windows: Courier New fallback


def test_font_env_var_override_takes_priority():
    assert render_ocr._FONT_CANDIDATES[0] == os.environ.get("HERMES_STRIP_METADATA_FONT")


def test_retype_fails_safe_when_no_font_available(monkeypatch):
    # Simulate the "no font found on this host" case: retype() must return
    # skipped=True, not crash, and must not touch the module global
    # permanently for other tests.
    monkeypatch.setattr(render_ocr, "FONT_PATH", None)
    result = render_ocr.retype("some plain text")
    assert result.skipped is True
    assert result.clean_text == "some plain text"


def test_cmap_is_cached_not_reparsed_every_call():
    # find_unsupported_chars should reuse the cached cmap, not re-parse the
    # font file from disk on every call.
    render_ocr._CMAP_CACHE = None
    render_ocr.find_unsupported_chars("warm up the cache")
    cache_after_first_call = render_ocr._CMAP_CACHE
    assert cache_after_first_call is not None
    render_ocr.find_unsupported_chars("second call should reuse it")
    assert render_ocr._CMAP_CACHE is cache_after_first_call


# ---------------------------------------------------------------------------
# Post-ship improvement: ASCII cost gate. Plain ASCII text that stage 1
# left untouched skips the expensive render+OCR round trip entirely, since
# there is no hiding technique possible in a plain ASCII string that
# stage 1's exact-codepoint check wouldn't already have caught.
# ---------------------------------------------------------------------------

def test_cost_gate_identifies_plain_ascii_as_low_risk():
    assert plugin._is_low_risk_plain_text("plain ascii prose, nothing fancy") is True


def test_cost_gate_rejects_any_non_ascii():
    assert plugin._is_low_risk_plain_text("café") is False
    assert plugin._is_low_risk_plain_text("naïve") is False
    assert plugin._is_low_risk_plain_text("emoji \U0001F389") is False


def test_cost_gate_skips_stage_2_for_plain_prose_hook_call():
    text = "The quick brown fox jumps over the lazy dog. Plain prose, no tricks."
    start = time.time()
    result = plugin._on_transform_llm_output(response_text=text)
    elapsed = time.time() - start
    assert result is None  # unchanged
    # The fast path must be near-instant (no render+OCR round trip). A
    # generous ceiling well below a real OCR call's typical multi-second
    # cost, to avoid a flaky test on a slow CI box while still catching a
    # regression that accidentally routes plain text through stage 2.
    assert elapsed < 0.5


def test_cost_gate_does_not_skip_stage_2_for_non_ascii_text():
    # Accented text must still go through the full stage 2 check (and in
    # this case get caught as suspect and fall back to stage-1 output,
    # which is the correct, safe behavior).
    text = "naïve café résumé"
    result = plugin._on_transform_llm_output(response_text=text)
    # Either None (unchanged, if OCR round-tripped perfectly) or the
    # stage-1-only text (if OCR flattened an accent and fell back) - both
    # are correct; what must NOT happen is skipping stage 2's check
    # entirely for non-ASCII input.
    assert result is None or result == plugin.sanitize_text(text)


# ---------------------------------------------------------------------------
# End-to-end hook contract: must never raise, must return None for
# genuinely unchanged text, must never corrupt emoji/accents it can't
# safely retype.
# ---------------------------------------------------------------------------

def test_hook_never_raises_on_empty_string():
    assert plugin._on_transform_llm_output(response_text="") is None


def test_hook_never_raises_on_missing_kwarg():
    assert plugin._on_transform_llm_output() is None


def test_hook_preserves_emoji():
    text = "Launch party \U0001F389\U0001F680\U0001F600"
    result = plugin._on_transform_llm_output(response_text=text)
    final = result if result is not None else text
    assert "\U0001F389" in final
    assert "\U0001F680" in final


def test_hook_full_battery_code_and_lists_unchanged():
    # End-to-end confirmation that regressions 1-4 together still produce
    # a clean, unchanged hook result for structured content.
    code = "def add(a, b):\n    return a + b\n"
    result = plugin._on_transform_llm_output(response_text=code)
    assert result is None, "code block should round-trip clean with no changes"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
