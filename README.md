# Sahaayak AI — A Gemma 4 powered employment document explainer

Sahaayak AI helps Indian workers understand employment documents in simple, structured language. Upload a PDF or image of an offer letter, contract, or payslip and get a practical explanation with risk flags, suggested next steps, questions to ask, and a local language summary in English, Hindi, Marathi, Tamil, or Telugu.

Built for the [Gemma 4 Good Hackathon](https://kaggle.com/competitions/gemma-4-good-hackathon).

---

## The problem

Every year, millions of workers in India sign employment documents they do not fully understand. Bond clauses, liquidated damages, unexplained salary deductions, and unclear notice-period terms can cost a worker months of wages. Most of these documents are in English, use legal language, and are reviewed on a cheap phone over an unreliable internet connection.

Worker-facing advisors — NGOs, labour unions, and legal aid volunteers — exist, but they cannot be everywhere. A model that runs locally, privately, and in the worker's own language can be.

## The approach

Sahaayak uses **Gemma 4** to act as a patient, plain-language explainer. The design is deliberately offline-capable and privacy-first:

- **Local inference via Ollama** is the default path. No document content leaves the machine.
- **Gemma 4 E4B** is the primary model. It is multimodal, supports 140+ languages out of the box, and runs comfortably on a modern laptop, making "community booth" deployments realistic.
- **Hybrid text + vision extraction.** Clean PDFs go through fast PyMuPDF text extraction. Scanned PDFs, phone photos, and images go through Gemma 4's native vision encoder — no OCR engine needed. Mixed documents get both evidence streams and let Gemma cross-reference.
- **Grounded in Indian labor law via function calling.** A small local SQLite knowledge base ships with curated entries on bond clauses, notice periods, working hours, minimum wage, and more. Gemma 4 uses its native function calling to consult this database during analysis, so risk flags cite actual statutes (Industrial Employment Standing Orders Act, Indian Contract Act Section 27, Factories Act Section 59, etc.) rather than generic advice.
- A **heuristic mock analyser** ships alongside the real model and acts as a safe fallback when anything about the real-mode pipeline fails (network blip, invalid JSON, schema violation, provider offline). The app never shows a broken result.
- **Strict JSON schema validation** on every model response. If the model deviates from the contract, we fall back to the mock analyser and label the source as `fallback_after_model_failure` in the UI.

## Architecture

```
          +------------------------+
          |  Uploaded document     |
          |  (PDF, PNG, JPG)       |
          +-----------+------------+
                      |
                      v
          +-------------------------------+
          |  Hybrid extractor             |
          |  - PyMuPDF text (PDF)         |
          |  - Page images as base64 PNG  |
          |    (for the vision path)      |
          |  - Image uploads -> vision    |
          +-----------+-------------------+
                      |
                      v
          +------------------------+
          |  Prompt renderer       |
          |  services/prompt_loader
          +-----------+------------+
                      |
                      v
     +----------------+----------------+
     |                                 |
     v                                 v
+---------+                   +---------------------+
| Mock    |                   | Gemma 4 client      |
| analyser|                   |                     |
| (always |                   | - Ollama: multi-    |
| works)  |                   |   turn tool loop    |
|         |                   |   with              |
|         |                   |   lookup_labor_law  |
|         |                   | - HF Router: single-|
|         |                   |   shot (no tools)   |
+----+----+                   +----------+----------+
     |                                   |
     |                                   v
     |                          +---------------------+
     |                          | Indian labor-law DB |
     |                          | services/labor_law  |
     |                          | (SQLite, seeded)    |
     |                          +----------+----------+
     |                                   |
     |                                   v
     |                          +---------------------+
     |                          | Strict JSON parse + |
     |                          | schema validation   |
     |                          +----+----------+-----+
     |                               |          | fail
     |                               | pass     v
     |                               |     (fallback)
     v                               v
+-----------------------------------------+
|  Validated DocumentAnalysis + tool_log  |
|  document_type, risks, key points,      |
|  next actions, questions, references,   |
|  local language summary, confidence,    |
|  [law citations Gemma consulted]        |
+---------------------+-------------------+
                      |
                      v
          +-----------------------+
          |  Streamlit card UI    |
          +-----------------------+
```

### Key components

- `app.py` — Streamlit frontend: upload, extraction preview, analysis cards.
- `services/analyzer.py` — orchestrates mock and real modes, validates every response, degrades gracefully.
- `services/gemma_client.py` — talks to Gemma 4 via either local Ollama (`/api/chat`) or an OpenAI-compatible HTTP endpoint.
- `services/prompt_loader.py` — renders the prompt template with the extracted text and target language.
- `prompts/document_analysis_prompt.txt` — the strict-JSON contract Gemma 4 is asked to fulfil.

## Output schema

Every analysis — real, mock, or fallback — returns the same validated shape.

```json
{
  "document_type": "string",
  "simple_explanation": "string",
  "key_points": ["string"],
  "risk_flags": [
    {
      "risk_title": "string",
      "risk_explanation": "string",
      "severity": "High | Medium | Low",
      "source_text": "string",
      "suggested_action": "string"
    }
  ],
  "next_actions": ["string"],
  "questions_to_ask": ["string"],
  "source_references": ["string"],
  "local_language_summary": "string",
  "confidence_level": "High | Medium | Low"
}
```

## Quick start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) (for local Gemma 4 inference)

### Install

```bash
git clone <repo>
cd Gemma-4-Kaggle-Hackathon
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### Run in mock mode (no model required)

```bash
export MODEL_MODE=mock
streamlit run app.py
```

Mock mode runs the deterministic heuristic analyser. It is the safe default and useful for development, CI, and demos where you want to show the UX without running a model.

### Run with local Gemma 4 via Ollama

```bash
ollama pull gemma4:e4b
ollama serve                      # only if not already running

export MODEL_MODE=real            # provider defaults to "ollama"
export GEMMA_MODEL=gemma4:e4b     # default, set only to override
streamlit run app.py
```

The sidebar will show `Model mode: real (ollama: gemma4:e4b)` in green when connected. The analysis source label will read `gemma_real` when a response passes validation, or `fallback_after_model_failure` if anything along the path fails.

### Run with a hosted Gemma 4 endpoint (Hugging Face Router)

Optional. Useful if you want to demonstrate the same code against a cloud-hosted Gemma 4.

```bash
export MODEL_MODE=real
export GEMMA_PROVIDER=api
export GEMMA_MODEL=google/gemma-4-e4b-it
export GEMMA_API_BASE_URL=https://router.huggingface.co/v1
export HF_TOKEN=hf_your_token
streamlit run app.py
```

Never commit `HF_TOKEN` or any other credential.

## Tests

```bash
pytest -q
```

The suite covers:

- Schema validation, including risk-flag object structure.
- Strict JSON parsing (rejects extra text, markdown, or non-object responses).
- Employment vs non-employment classification.
- Statutory deductions (PF, ESI, TDS) handled as key points, not high-risk flags.
- High-risk deduction detection (penalty, bond recovery, damages).
- Long-working-hours, overtime, notice-period, and restrictive-clause detection.
- PDF text extraction, image OCR, unsupported and empty uploads.
- Mock and real model mode switching.
- Real-mode fallback on provider error, malformed JSON, extra text, and schema violation.
- Parametrised corpus tests across positive, negative, gibberish, unreadable, and canary documents (prompt injection, schema subversion attempts).

## Sample documents

Six anonymised sample documents live in `sample_docs/`:

- `employment_agreement_bond_clause.pdf` — contract with a restrictive bond clause (exercises high-risk detection and bond-clause law lookup).
- `salary_slip_deductions.pdf` — payslip with statutory deductions (exercises PF/TDS/ESI as key points, not risk flags).
- `non_employment_aws_sample.pdf` — AWS documentation (exercises unsupported-document classification).
- `hindi_english_offer_letter.pdf` — housekeeping offer letter with a worker-facing Hindi summary. Demonstrates Gemma 4's multilingual output on genuinely mixed-script Indian documents.
- `gig_worker_agreement.pdf` — delivery-platform contractor agreement with penalty schedule, non-compete, and device deposit (exercises the penalty/damages law lookup and independent-contractor analysis).
- `karnataka_factory_offer_letter.pdf` — factory worker appointment letter in Karnataka (triggers the Karnataka-specific labor-law lookups under the Karnataka Shops & Commercial Establishments Act and Factories Act).

Regenerate them with:

```bash
pip install reportlab
python generate_sample_docs.py
```

## Safety posture

Uploaded document text is treated as **untrusted input**. The system prompt instructs Gemma 4 to return strict JSON only, and the analyser:

1. Rejects any response that does not start with `{` and end with `}`.
2. Rejects any response whose JSON does not exactly match the 9-field schema.
3. Falls back to the mock analyser with `confidence_level: Low` and a clearly labelled source whenever validation fails.

Additional protections:

- No PII (employee names, IDs) is echoed into the prompt beyond the extracted text the user themselves uploaded.
- Statutory deductions are handled as key points by design; they never trigger a high-risk flag.
- API keys, base URLs, and provider details are never rendered in the UI.
- A confidence level is surfaced on every analysis; low confidence prompts the user toward manual review.

## Current limitations

- Multi-page PDFs are capped at 4 pages for the vision path to keep latency reasonable; longer documents still extract all page text, but only the first 4 pages are shown to the vision encoder.
- `local_language_summary` quality depends on Gemma 4's coverage of the target language. The ones tested in this repo (English, Hindi, Telugu) come back as fluent native-script output.
- The labor-law knowledge base is **informational, not legal advice**. Entries are paraphrases of well-established statutes maintained in `services/labor_law.py`; they are meant to ground Gemma's analysis, not to replace a qualified employment lawyer.
- Function calling is currently wired only for the local Ollama provider. The HF Router provider ignores the tool loop and runs the single-shot analysis path.
- Handwritten documents are not supported.

## Project structure

```text
.
├── app.py                              # Streamlit frontend
├── generate_sample_docs.py             # Create demo PDFs
├── requirements.txt                    # Dependencies
├── pytest.ini                          # Test configuration
├── README.md                           # This file
├── PROJECT_DESIGN.md                   # Design notes and build status
├── prompts/
│   └── document_analysis_prompt.txt    # Strict-JSON Gemma 4 contract
├── services/
│   ├── __init__.py
│   ├── analyzer.py                     # Mock + real orchestration, validation, fallback
│   ├── gemma_client.py                 # Ollama and HF Router clients
│   └── prompt_loader.py                # Template renderer
├── sample_docs/                        # Anonymised demo documents
└── tests/
    ├── conftest.py
    ├── fixtures/document_cases.json    # Positive / negative / gibberish / canary corpus
    ├── test_analyzer.py
    ├── test_extraction_and_app_helpers.py
    ├── test_gemma_client.py
    └── test_prompt_loader.py
```

## Troubleshooting

**Ollama not found after install on macOS** — open `/Applications/Ollama.app` once to start the background service and register `/usr/local/bin/ollama` on PATH.

**`Model mode` stays grey on `mock` in the sidebar** — export `MODEL_MODE=real` in the same shell before launching Streamlit.

**Streamlit shows `fallback_after_model_failure`** — Gemma returned something the validator rejected. Check the terminal for the underlying error; the mock response is still safe to display.

**Tesseract not found for image upload** — Tesseract is no longer required. Image uploads go straight to Gemma 4's vision path.

## Disclaimer

Sahaayak AI provides informational analysis only and is not a substitute for professional legal or HR advice. Always consult a qualified professional before making employment-related decisions.

## License

MIT.
