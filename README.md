# Sahaayak AI Gemma 4 Document Analysis MVP

Sahaayak AI helps workers understand employment documents in simple, structured language. Upload a PDF or image employment document, and get a practical explanation with risks, next steps, questions to ask, and source references—available in English, Hindi, Marathi, Tamil, or Telugu.

---

## Problem Statement

Workers often struggle to understand employment documents due to legal jargon and complex terms. They need:
- **Simple explanations** of what a document means
- **Risk identification** to spot unfavorable clauses
- **Guidance on next steps** before signing
- **Multi-language support** for accessibility
- **Trust** that the analysis is safe and transparent

---

## Target User

- **Primary:** Workers in India reviewing offer letters, employment contracts, payslips, and termination letters
- **Secondary:** HR professionals, labor consultants, or organizations supporting worker rights

---

## Why Gemma 4

Gemma 4 (31B parameters) is chosen for this task because:
- **Open model** - no vendor lock-in, can be deployed on-premise or via API
- **Multilingual** - supports 40+ languages including Hindi, Tamil, Telugu, Marathi
- **Strong reasoning** - handles nuanced employment document analysis
- **Safety** - structured output validation ensures reliable schema adherence
- **Accessible** - available via Hugging Face Router API with affordable pricing

---

## Architecture

```
User Upload (PDF/Image)
        ↓
    [Text Extraction]
    PyMuPDF (PDF) / pytesseract (OCR)
        ↓
    [Prompt Rendering]
    Target language + extracted text → LLM prompt
        ↓
    [Model Modes]
    ├─ Mock: Local heuristic analyzer (always works)
    └─ Real: Gemma 4 via HF Router API
        ↓
    [Strict Validation]
    Parse JSON → Validate schema → Fall back to mock if invalid
        ↓
    [UI Rendering]
    Streamlit cards: Document Type, Risks, Key Points, etc.
```

### Key Components

- **app.py** – Streamlit frontend with file upload, language selection, extracted text preview
- **services/analyzer.py** – Core analysis engine with mock heuristic fallback
- **services/gemma_client.py** – OpenAI-compatible API client for Gemma 4
- **services/prompt_loader.py** – Template rendering with placeholder substitution
- **prompts/document_analysis_prompt.txt** – System prompt defining output schema and rules

---

## Output Schema

The analyzer returns a structured JSON with 9 fields:

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

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip or conda

### Installation

```bash
git clone https://github.com/yourusername/Sahaayak-AI-Gemma4.git
cd Sahaayak-AI-Gemma4
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

---

## Running the App

### Mock Mode (Default - No API Required)

Mock mode uses a tested local heuristic analyzer. Perfect for testing and development.

**Windows (PowerShell):**
```powershell
$env:MODEL_MODE="mock"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**macOS/Linux:**
```bash
export MODEL_MODE="mock"
streamlit run app.py
```

The app will open at `http://localhost:8501`. Upload a PDF or image, select your language, and click **Analyze Document**.

### Real Mode (Gemma 4 via Hugging Face Router)

Real mode calls Gemma 4 through the OpenAI-compatible Hugging Face Router API. Still uses mock analyzer as a safe fallback if the model fails.

**Prerequisites:**
- Hugging Face API token: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

**Windows (PowerShell):**
```powershell
$env:MODEL_MODE="real"
$env:GEMMA_PROVIDER="api"
$env:GEMMA_MODEL="google/gemma-4-31B-it:novita"
$env:GEMMA_API_BASE_URL="https://router.huggingface.co/v1"
$env:HF_TOKEN="hf_your_token_here"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**macOS/Linux:**
```bash
export MODEL_MODE="real"
export GEMMA_PROVIDER="api"
export GEMMA_MODEL="google/gemma-4-31B-it:novita"
export GEMMA_API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="hf_your_token_here"
streamlit run app.py
```

The sidebar will show `Model mode: real (api: google/gemma-4-31B-it:novita)` in green when connected.

**Note:** Never commit `HF_TOKEN` to version control. Use environment variables or a `.env` file (excluded in `.gitignore`).

---

## Testing

Run the comprehensive test suite:

```bash
.venv\Scripts\python.exe -m pytest -q
```

**Test Coverage:**
- Schema validation and strict JSON parsing
- Employment vs. unsupported document classification
- Statutory deduction handling (PF, TDS, ESI as key points, not risks)
- Risk flag detection (bond clauses, penalties, long hours, etc.)
- PDF and image text extraction
- Mock and real model mode switching
- Safe fallback on model failure
- Parametrized corpus tests (positive, negative, gibberish, canary documents)

---

## Safety Features

### Untrusted Input & Strict Output Validation

**The app treats uploaded document text as untrusted input and validates all model responses through a strict JSON schema before rendering.**

This is the core safety mechanism:
1. **Input** - Document text is extracted but treated as potentially malicious
2. **Prompt** - System prompt instructs Gemma to return only strict JSON (no markdown, code fences, or extra text)
3. **Parsing** - Model response must be valid JSON starting with `{` and ending with `}`
4. **Validation** - Response must match the 9-field schema exactly with correct field types
5. **Fallback** - If validation fails, the app falls back to the tested mock analyzer with low confidence

### Additional Protections

- **No PII in prompts** - Employee names and IDs from extracted text are not added to the prompt
- **Statutory deduction awareness** - PF, ESI, TDS, Professional Tax are never flagged as risks
- **Confidence levels** - Low confidence on fallback, mock mode, or unreadable documents
- **Source transparency** - Sidebar shows which analyzer generated the result (`mock_analyzer`, `gemma_real`, or `fallback_after_model_failure`)
- **No secret exposure** - API keys, base URLs, and provider details are never displayed to the user

---

## Sample Documents

Three anonymized demo documents are included in `sample_docs/`:

- **employment_agreement_bond_clause.pdf** – Employment contract with restrictive bond clause (tests high-risk detection)
- **salary_slip_deductions.pdf** – Payslip with statutory deductions (tests PF/TDS/ESI as key points)
- **non_employment_aws_sample.pdf** – AWS EC2 documentation (tests unsupported document classification)

Generate them with:
```bash
python generate_sample_docs.py
```

---

## Current Limitations

- **Scanned PDFs** - Images embedded in PDFs are not OCR'd; only selectable text is extracted
- **Complex layouts** - Documents with multi-column layouts, headers, footers, or form fields may extract text with formatting issues
- **Non-English originals** - Employment documents originally in Indian languages may not extract cleanly
- **Handwritten documents** - Not supported; must be typed/printed
- **Real mode availability** - Requires active Hugging Face API token and internet connection
- **Language support** - Limited to English, Hindi, Marathi, Tamil, Telugu (can be extended)
- **Tone & nuance** - Gemma 4 analysis reflects model capabilities; human review for legal decisions is always recommended

---

## Project Structure

```
Sahaayak-AI-Gemma4/
├── app.py                              # Streamlit frontend
├── generate_sample_docs.py             # Create demo PDFs
├── requirements.txt                    # Dependencies
├── pytest.ini                          # Test configuration
├── README.md                           # This file
├── PROJECT_DESIGN.md                   # Detailed design document
├── .gitignore                          # Git exclusions
├── prompts/
│   └── document_analysis_prompt.txt    # Gemma system prompt
├── services/
│   ├── __init__.py
│   ├── analyzer.py                     # Mock + real model logic
│   ├── gemma_client.py                 # Hugging Face Router API client
│   └── prompt_loader.py                # Prompt template rendering
├── sample_docs/                        # Anonymized demo documents
├── tests/
│   ├── conftest.py                     # pytest fixtures
│   ├── test_analyzer.py                # Analyzer tests
│   ├── test_gemma_client.py            # Model client tests
│   ├── test_prompt_loader.py           # Prompt rendering tests
│   ├── test_extraction_and_app_helpers.py  # App helper tests
│   └── fixtures/
│       └── document_cases.json         # Test corpus
└── outputs/                            # Analysis results (user-generated)
```

---

## Troubleshooting

**"Connection lost" error on file upload:**
Restart Streamlit to pick up configuration:
```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

**Tesseract OCR not found (for image extraction):**
Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki

**Model timeout in real mode:**
Increase timeout (default 60s):
```powershell
$env:GEMMA_TIMEOUT_SECONDS="120"
```

**Mock analyzer returns low confidence:**
This is expected for complex or non-standard documents. Real mode with Gemma 4 may provide better results.

---

## Features

- ✅ Upload PDF or image employment documents
- ✅ Extract text via PyMuPDF (PDFs) and pytesseract (OCR)
- ✅ Multi-language support (EN, HI, MR, TA, TE)
- ✅ Structured analysis with 9 output fields
- ✅ Risk detection (bond clauses, long hours, penalties, etc.)
- ✅ Statutory deduction awareness
- ✅ Mock mode (always available) + Real mode (Gemma 4)
- ✅ Safe fallback on model failure
- ✅ Strict JSON schema validation
- ✅ Comprehensive test suite
- ✅ Anonymized sample documents

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Add tests for new functionality
4. Commit with clear messages (`git commit -m "Add feature"`)
5. Push to your fork and open a Pull Request

---

## License

This project is open source and available under the MIT License.

---

## Disclaimer

Sahaayak AI provides informational analysis only and is not a substitute for professional legal or HR advice. Always consult qualified professionals before making employment-related decisions. Use at your own risk.

---

## Contact & Feedback

Have questions or suggestions? Please open an issue on GitHub or reach out to the maintainers.

## Analyzer Output Schema

The analyzer returns JSON with these fields:

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

## Project Structure

```text
.
+-- app.py
+-- requirements.txt
+-- README.md
+-- prompts/
+-- services/
+-- sample_docs/
+-- outputs/
+-- tests/
```

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Model Setup

The analyzer supports two modes:

- `MODEL_MODE=mock`: default. Uses the tested local heuristic analyzer.
- `MODEL_MODE=real`: calls Gemma through API/cloud mode, parses strict JSON, validates the schema, and falls back to the mock analyzer if anything fails.

### Mock Mode

```powershell
$env:MODEL_MODE="mock"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### Real Mode With Hugging Face Router

Real mode uses the OpenAI-compatible Hugging Face Router API. Set `HF_TOKEN` and use the Gemma model route:

```powershell
$env:MODEL_MODE="real"
$env:GEMMA_PROVIDER="api"
$env:GEMMA_MODEL="google/gemma-4-31B-it:novita"
$env:GEMMA_API_BASE_URL="https://router.huggingface.co/v1"
$env:HF_TOKEN="your-huggingface-token"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

`GEMMA_API_KEY` can be used instead of `HF_TOKEN` if preferred.

The sidebar shows the active model mode. Real mode still uses the mock analyzer as a safe fallback when the provider call fails, returns malformed JSON, adds extra text, or violates the schema.

This project is now API/cloud-first for real model mode. Local Ollama is not the recommended path.

The analysis output also shows a small debug source label:

- `mock_analyzer`
- `gemma_real`
- `fallback_after_model_failure`

The app does not display raw prompts, API keys, or provider secrets.

### Cleaning Local Model Data

After the API/cloud path is confirmed working and the analysis source shows `gemma_real`, local downloaded model data can be removed separately. Do this only after confirming you no longer need local models.

For Ollama, typical cleanup is:

```powershell
ollama list
ollama rm <model-name>
```

Model files may also live under the user-level Ollama model directory. Deleting those files is destructive, so verify the API mode works first.

## Troubleshooting

If the browser says `Connection lost` during upload, restart Streamlit so it picks up the project config in `.streamlit/config.toml`. That config disables Streamlit usage statistics, which avoids permission errors when Windows blocks writes to the user-level `.streamlit` folder.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Run Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The test suite includes unit tests for extraction, prompt loading, analyzer validation, unreadable/unsupported document handling, statutory deduction handling, and a fixture corpus covering positive, negative, gibberish, and canary documents.

## OCR Setup

PDF text extraction works through the Python dependency `PyMuPDF`.

Image text extraction uses `pytesseract`, which also requires the Tesseract OCR engine to be installed on your machine and available on `PATH`. If the engine is missing, the app will still run and will show a clear OCR setup warning for image uploads.

## Analysis Pipeline

- `services/prompt_loader.py` loads and renders prompt templates from `prompts/`.
- `prompts/document_analysis_prompt.txt` defines the strict Gemma 4 JSON contract.
- `services/gemma_client.py` reads model environment variables and calls the configured API/cloud provider.
- `services/analyzer.py` switches between mock and real model mode, parses strict JSON, validates schema, and falls back safely.
- `app.py` renders the validated analyzer JSON as Streamlit cards.

## Design Doc

See `PROJECT_DESIGN.md` for the project design, architecture, current build status, limitations, and next steps.

## Extending Later

The `generate_gemma_response()` function in `services/gemma_client.py` is the provider boundary for real Gemma calls. A production implementation can add:

- Provider-specific authentication or SDK support.
- Real translation/local language output.
- Saved analysis outputs in `outputs/`.
- Example files in `sample_docs/`.
