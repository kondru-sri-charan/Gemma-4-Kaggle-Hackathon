# Known issues

Deliberate tech debt we're carrying so we can sprint. Revisit before the submission video.

## 1. Off-scope documents get analyzed anyway (Day 3, May 3 2026)

**Symptom**

Uploading a health insurance card or other non-employment document produces a full
structured analysis with risk flags, instead of the `"Unsupported / Non-employment
document"` response the prompt demands.

**Observed example**

A health insurance membership card, uploaded via UI in real mode. The card mentions
"employer/HR" and "till such time the member is employed" which trips Gemma 4 into
treating it as an employment document.

**Why it happens**

- Employment-adjacent vocabulary in genuinely off-scope documents (insurance cards,
  benefit brochures, company IDs) matches phrases Gemma 4 interprets as employment
  signals.
- The prompt asks for `"Unsupported / Non-employment document"` but also gives a
  schema that must always be filled, so the model finds a path that fills all
  fields rather than refusing.

**Fix plan (Option A, low effort)**

- Tighten the system prompt with explicit examples of off-scope documents
  (insurance card, benefit brochure, company ID, medical bill, invoice, etc.).
- Add 1-2 few-shot examples of the Unsupported response shape.
- Add the actual failing insurance card PDF as a fixture under
  `tests/fixtures/` so we can regression-test with vision in a follow-up.
- Add one real off-scope sample to `sample_docs/` for the demo — showing the
  app correctly refusing to analyze is a trust signal in the video.

**Fallback plan (Option B, if A doesn't hold)**

- Add a cheap first-pass classifier call before the full analysis. Extra
  latency (~3-5s) but robustly separates scope decisions from content
  analysis.

**Not doing now because**

We're prioritizing the function-calling + labor-law grounding work
(Day 5-6) which is the bigger technical wow for the hackathon. Will revisit
before recording the demo video.
