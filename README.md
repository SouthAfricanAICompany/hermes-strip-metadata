# hermes-strip-metadata

Free, open-source. Built and maintained by [SAAIC](https://saaic.co.za) (South African AI Company).

Strips hidden/invisible metadata from Hermes's final response **before it reaches any destination**, chat UI, Telegram, Discord, cron delivery, API, anything. No clipboard, no copy-paste, no per-destination patching.

## Why this exists

AI-generated text can carry hidden passengers you never chose to send: invisible Unicode characters used for watermarking or tracking, clipboard/RTF formatting leftovers, stray metadata that survives a copy-paste. Most people posting AI-assisted text publicly (on socials, in a company blog, in a client email) have no idea this is happening, and no easy way to check. This plugin exists to make "what you see is what gets sent" actually true, for free, for anyone, not just SAAIC's own clients.

**A note on the best setup for public-facing use:** if you're running an agent that posts to socials or anything public-facing, the ideal setup is a dedicated Hermes harness/instance for that purpose, with this plugin enabled on it, rather than bolting it onto a general-purpose assistant you also use for fast back-and-forth chat. Stage 2's render+OCR round trip takes real time (seconds, not instant), even though it's skipped automatically for plain ASCII text and so doesn't fire on every single response. A harness dedicated to public output can afford that latency as the cost of doing it right; a harness you're also using for quick personal Q&A probably shouldn't eat it on every message.

## Why this works (and will keep working across Hermes updates)

Hermes ships a plugin hook called `transform_llm_output`. It fires exactly once per turn, inside the agent's own conversation loop (`AIAgent.run_conversation`), **before** any gateway/platform code decides where the text goes. Whatever this hook returns becomes the final response. Every destination downstream sees only the sanitized text.

This is not a workaround or a hack on internals:
- It's a **documented, shipped plugin hook** (`website/docs/user-guide/features/plugins.md`, "Transform" category).
- It lives entirely in `~/.hermes/plugins/` (**zero changes to Hermes core**).
- Hermes's own contribution rubric explicitly prefers this over any core-tool/core-edit approach ("Keep the core narrow... plugin... before new core tool").
- Because it's a stable, versioned extension point (not an internal function you're monkey-patching), Hermes updates can change internals freely without breaking this. The hook contract (`response_text` in, string or `None` out) is the one thing the Hermes team keeps stable for plugin authors.

## What it strips (two stages)

**Stage 1, Unicode cleanup:**
- Zero-width/invisible Unicode (`U+200B` zero-width space, `U+200C/D` ZWNJ/ZWJ, `U+2060` word joiner, `U+FEFF` BOM, bidi override controls, variation selectors, Unicode tag-block characters), the characters commonly used to invisibly watermark or fingerprint AI-generated text.
- Stray HTML tags that sometimes leak through clipboard/RTF paste paths.
- Unicode-normalizes to NFC (defeats some combining-character tricks).
- Normalizes excess whitespace without touching intentional formatting.

**Stage 2, render and OCR (the "retype"):** the stage-1 output gets rendered to a plain bitmap image and read back with OCR (`render_ocr.py`). This produces a brand new string built from pixels, with zero lineage to the model's original output buffer. It is the API-delivery equivalent of a human physically retyping the text on a keyboard: no clipboard flavor, no invisible Unicode, no HTML/RTF wrapper, nothing that isn't literal ink-shaped text survives, because OCR cannot see anything that was never drawn as a visible glyph.

This stage is skipped automatically, falling back to stage-1 output, when:
- The text contains characters the render font cannot draw (emoji, CJK, most non-Latin scripts). Rendering these as fallback boxes and OCR-guessing them back would silently corrupt the text, confirmed by testing (an emoji-corrupting round trip can still score a 93% similarity ratio against the original, high enough to pass a naive threshold while real content is destroyed).
- The OCR read is "suspect": either the aggregate similarity ratio drops below 90%, any non-whitespace content was changed/deleted/inserted (e.g. a mangled code-fence marker, confirmed to happen on real OCR output even when every changed character is plain ASCII), or a line's leading indentation changed (even when every changed character is whitespace, confirmed to happen and to silently change what indented code/YAML means).
- No usable monospace font exists on this host at all (see Install below) - this skips Stage 2 entirely rather than crashing, with a warning logged at plugin load time.
- The already-cleaned text is plain 7-bit ASCII prose that Stage 1 left completely untouched: there is no clipboard flavor, RTF wrapper, or font-level metadata on a Python string, so the only hiding techniques that can exist at this point are the exact invisible/zero-width ranges Stage 1 already checks by codepoint. Pure ASCII text has no room left for any of those, known or future, so Stage 2 provably cannot add anything for this specific text. This is a real cost gate, not a coverage gap: a full render+OCR round trip measured ~2.6 seconds for a 6800-character paste in testing, and most real responses are plain prose. ANY non-ASCII character at all still routes through the full Stage 2 check, since an unknown future hiding technique could live in any non-ASCII range Stage 1 doesn't yet know about.

**What it does not defeat:** Anthropic's (and other vendors') statistical, word-choice watermark. That mark lives in which words the model picked when multiple options were equally valid, not in any character, so neither Unicode stripping nor an OCR round trip touches it; the words that come back out are the same words that went in. The only thing that breaks it is a genuine paraphrase/rewrite pass. Pair this plugin with a paraphraser skill for that layer; this plugin is not a substitute for one.

## Install

**Linux / macOS:**
```bash
pip install Pillow pytesseract fonttools
sudo apt-get install tesseract-ocr   # or your OS's tesseract package

mkdir -p ~/.hermes/plugins/hermes-strip-metadata
cp plugin.yaml __init__.py render_ocr.py ~/.hermes/plugins/hermes-strip-metadata/
hermes plugins enable strip-metadata
```

**Windows: not officially supported yet.** The code has OS-aware font-path detection for Windows (Consolas/Courier New) written in, but it has never been run on an actual Windows machine, so it is untested and unverified. Treat it as experimental if you try it, and please report back what breaks. Tesseract on Windows needs a separate install (the [UB-Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki)) and PATH/`tesseract_cmd` setup that isn't needed on Linux/macOS, which is the likely first thing to go wrong.

**Font (Stage 2 only):** `render_ocr.py` auto-detects a monospace font: DejaVu Sans Mono on Linux (common distro paths checked) or DejaVu Sans Mono via Homebrew on macOS. If none of the built-in paths match your setup, set the `HERMES_STRIP_METADATA_FONT` environment variable to a `.ttf` path on your machine. If no font is found at all, Stage 2 is skipped automatically (same as any other skip condition, falls back to Stage 1 output) rather than crashing - a `RuntimeWarning` is logged once at plugin load time so this is visible, not silent.

**Important:** the folder name doesn't matter, but `hermes plugins enable` needs the plugin's
declared `name:` from `plugin.yaml` (`strip-metadata`), not the folder name. Using the folder
name here is the #1 way this silently fails to activate.

Restart Hermes (or `/reset` in an active session) for the plugin to load.

Verify it loaded:
```bash
hermes plugins list
# should show: strip-metadata   enabled
```

## Verifying it's working

Run the automated regression suite (covers every bug found across the review process, plus the cost-gate/font-resolution logic):
```bash
pip install pytest
python3 -m pytest tests/test_plugin.py -v
# should show: 52 passed
```

Ask Hermes to output something, then check the actual bytes of what you received (e.g. paste into a hex viewer or run `python3 -c "print([hex(ord(c)) for c in open('out.txt').read()])"`), there should be no characters in the `200b-200f`, `2060-2064`, `feff`, `fe00-fe0f` ranges.

A quick in-session sanity check:
```bash
hermes chat -q "Repeat exactly: hello­world" # (contains a hidden soft-hyphen between hello and world)
```
The returned text should come back as plain `helloworld`/`hello world` with no hidden character.

## Scope / limitations

- Only sanitizes **Hermes's own generated final response**, once per turn. It does not sanitize raw tool output, file content the model echoes verbatim mid-turn tool_result, or content from MCP servers unless that content ends up inside the final text (the normal path for anything the model repeats to the user).
- If you need tool-output-level sanitization too (e.g. the model should never even SEE watermarked text from a web page), pair this with the `transform_tool_result` hook, same pattern, different seam. Not included here, ask if you want it added.
- This is NOT a sandbox or security boundary. It's a text-cleanup pass. A sufficiently adversarial model could still describe metadata in visible text if explicitly instructed to.

## Compatibility

Tracks whatever Hermes version ships `transform_llm_output` (confirmed present as of hermes-agent 0.21.x). If a future Hermes major version removes or renames this hook, `hermes plugins list` will show the plugin failing to load/fire. That's the signal to check `hermes-agent` release notes for the hook's replacement name.

## OpenClaw (same concept, different engine)

OpenClaw ships the equivalent seams natively. This is NOT a Hermes-only trick:

- **`message_sending`**: fires per-channel, right before delivery. `content` is mutable (last-writer-wins), or return `{ cancel: true }`. This is the direct equivalent of `transform_llm_output`: same chokepoint, just JS instead of Python.
- **`reply_payload_sending`**: same seam but for the full normalized reply object (media, presentation, delivery), not just text. Use this if you also need to strip metadata from attachments/media refs, not only text.
- **`before_response_emit`** (newer, as of the PR introducing it), run-scoped, closer to Hermes's `transform_llm_output` in spirit (fires once per run on the assistant's final text, with `allContent`/`content` returns and a `block` option). **Caveat found in the PR review**: as shipped it fails OPEN on a hook crash (unmodified text still gets delivered). If you use this hook for metadata stripping, wrap the sanitizer body in its own try/except and fail closed (block the reply) rather than trusting the host's default error handling.

A minimal OpenClaw plugin doing the same job:

```typescript
export default definePluginEntry({
  id: "strip-metadata",
  name: "Strip Metadata",
  register(api) {
    api.on("message_sending", (event) => {
      event.context.content = sanitize(event.context.content);
    });
  },
});
```

Same sanitize function, same guarantee (runs before every channel send, not model-dependent), ported to the host's native hook instead of Hermes's. Read OpenClaw's `docs.openclaw.ai/plugins/hooks` to implement this directly on that platform. It's documented, stable API, not a reverse-engineered internal.

## About SAAIC

[SAAIC](https://saaic.co.za) (South African AI Company) builds and manages AI systems for businesses that need their data handled with care: privacy by design, client-controlled processing, local-first options where it matters. This plugin is one small open-source piece of that work, released free because the underlying problem (AI output carrying hidden data you didn't choose to attach) affects everyone shipping AI-generated text, not just SAAIC's own clients.

If your business is weighing whether to build this kind of thing in-house or have someone else manage it end to end, [book a call](https://saaic.co.za) or email info@saaic.co.za.

