# Sahaayak AI Project Design

## 1. Purpose

Sahaayak AI helps workers understand employment documents in simple language.

The MVP lets a user upload a PDF or image, extracts document text, analyzes the text through a structured pipeline, and displays a practical explanation with risks, next steps, questions, source references, and a local language summary.

## 2. Target User

- Workers reviewing offer letters, contracts, payslips, termination letters, or similar employment documents.
- Users who may not understand legal or HR language.
- Users who need quick guidance before signing, accepting, or questioning a document.

## 3. Core User Flow

1. User opens the Streamlit app.
2. User uploads a PDF or image.
3. User selects target language:
   - English
   - Hindi
   - Marathi
   - Tamil
   - Telugu
4. User clicks `Analyze Document`.
5. App extracts text from the uploaded file.
6. App displays extracted text in `Extracted Document Text`.
7. Analyzer produces structured JSON.
8. App renders clean Streamlit cards for the analysis.

## 4. What Is Built

### Streamlit App

File: `app.py`

Built features:

- File upload for PDFs and images.
- Language dropdown.
- Analyze button.
- Uploaded document preview for images.
- PDF upload state message.
- Extracted text expander.
- Structured analysis cards.
- Rich risk flag cards.
- Error handling for unsupported files, empty files, unreadable images, missing packages, missing Tesseract, and low-confidence extraction.

### Document Extraction

Implemented in `app.py`.

PDF extraction:

- Uses `PyMuPDF`.
- Reads selectable text page by page.
- Prefixes extracted text with page numbers.
- Handles PDFs with no selectable text.

Image extraction:

- Uses `pytesseract`.
- Uses `Pillow` to read uploaded images.
- Handles invalid image files.
- Handles missing Tesseract OCR engine.

Supported upload types:

- `pdf`
- `png`
- `jpg`
- `jpeg`

### Prompt Template

File: `prompts/document_analysis_prompt.txt`

Purpose:

- Defines the future Gemma 4 analysis instruction.
- Requires strict JSON output.
- Defines required fields.
- Defines structured risk flags.
- Prevents Markdown, code fences, extra text, or invented facts.

### Prompt Loader

File: `services/prompt_loader.py`

Built features:

- Loads prompt templates from `prompts/`.
- Renders placeholders:
  - `{{target_language}}`
  - `{{extracted_document_text}}`
- Raises clear errors if a prompt file or required value is missing.

### Analyzer Service

File: `services/analyzer.py`

Built features:

- Public entrypoint: `analyze_document(extracted_text, target_language)`.
- Returns validated JSON.
- Supports `MODEL_MODE=mock` and `MODEL_MODE=real`.
- Keeps the tested mock analyzer as the fallback path.
- Mock output is realistic and based on extracted text.
- Uses simple heuristics to detect:
  - document type
  - role/designation
  - salary/pay
  - working hours
  - overtime
  - leave
  - notice period
  - deductions
  - penalties
  - probation
  - start date
  - bonds or restrictive clauses
- Adds risk flags with severity and suggested actions.
- Falls back to a safe low-confidence response if analysis fails validation.

### Gemma Client

File: `services/gemma_client.py`

Built features:

- Reads model environment variables:
  - `MODEL_MODE`
  - `GEMMA_PROVIDER`
  - `GEMMA_MODEL`
  - `GEMMA_API_BASE_URL`
  - `GEMMA_OLLAMA_URL`
  - `GEMMA_API_URL`
  - `GEMMA_API_KEY`
  - `HF_TOKEN`
  - `GEMMA_TIMEOUT_SECONDS`
- Supports OpenAI-compatible Hugging Face Router API mode.
- Keeps legacy Ollama provider support in code, but API/cloud mode is the recommended path.
- Returns raw model text to the analyzer.
- Raises controlled client errors so the analyzer can fall back safely.

### Folders

Built folders:

- `prompts/`
- `services/`
- `sample_docs/`
- `outputs/`
- `tests/`

Current placeholder folders:

- `sample_docs/` is ready for example documents.
- `outputs/` is ready for saved analysis results.

### Test Suite

Files:

- `tests/test_analyzer.py`
- `tests/test_prompt_loader.py`
- `tests/test_extraction_and_app_helpers.py`
- `tests/fixtures/document_cases.json`
- `pytest.ini`

Built coverage:

- Analyzer schema validation.
- Employment and unsupported document classification.
- Risk flag object structure.
- Long-hours, deduction, penalty, overtime, and restrictive-clause detection.
- Prompt template loading and rendering.
- Prompt loader failure fallback.
- Mock and real model mode switching.
- Real model JSON acceptance.
- Real model fallback on provider failure, malformed JSON, extra text, and schema violations.
- PDF extraction.
- Upload file type classification.
- Unsupported and empty upload handling.
- Image OCR path with mocked `pytesseract`.
- Missing Tesseract handling.
- Invalid image handling.
- Corpus tests across positive, negative, gibberish, and canary documents.

## 5. JSON Output Schema

The analyzer returns this structure:

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

## 6. Data Flow

```text
Uploaded file
  -> file type detection
  -> PDF extraction or image OCR
  -> extracted text
  -> prompt rendering
  -> mock analyzer or real Gemma provider
  -> strict JSON parsing when real mode is active
  -> validation
  -> fallback if needed
  -> Streamlit cards
```

## 7. Design Decisions

### Keep Extraction Separate From Analysis

Reason:

- PDF/image extraction can evolve independently from the model pipeline.
- Later we can add scanned PDF OCR without changing the analyzer contract.

### Use a Strict JSON Contract

Reason:

- Streamlit rendering becomes predictable.
- Future Gemma 4 integration has a clear response shape.
- Validation can catch malformed output before the UI breaks.

### Keep Mock Analyzer Realistic

Reason:

- The MVP can be demoed without a model key or inference server.
- The app already behaves like the real product flow.
- Risk flags are grounded in extracted text instead of generic placeholders.

### Risk Flags Are Objects, Not Strings

Reason:

- Users need actionable risk information, not just labels.
- Each risk contains title, explanation, severity, source text, and suggested action.
- This supports better UI cards and future filtering by severity.

### Add Fallback Handling

Reason:

- OCR and model responses are inherently noisy.
- A safe low-confidence response is better than a broken app.
- Users are told when the system is uncertain.

### Use Streamlit For MVP

Reason:

- Fast to build.
- Good enough for upload, preview, analysis, and iteration.
- Easy for judges, reviewers, or teammates to run locally.

## 8. Current Limitations

- Real Gemma calls are wired, but provider quality is not tuned yet.
- Translation is mocked.
- Local language summary is currently English text labeled for the selected language.
- Image OCR requires the external Tesseract engine installed on the machine.
- Scanned PDFs without selectable text are not OCR-processed yet.
- The mock analyzer uses heuristics, not legal reasoning.
- No authentication.
- No persistent output saving yet.
- No real-provider integration test suite yet.

## 9. Safety Position

The app should be treated as an explanation assistant, not a legal authority.

Current safety behavior:

- Avoids claiming certainty when extraction is weak.
- Shows confidence level.
- Shows source references.
- Advises users to ask clarifying questions.
- Suggests trusted advisor/legal aid review for high-risk terms.

## 10. Dependencies

File: `requirements.txt`

Current dependencies:

- `streamlit`
- `Pillow`
- `PyMuPDF`
- `pytesseract`
- `pytest`
- `requests`

External dependency:

- Tesseract OCR engine is required for image OCR.

## 11. How To Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Default local URL:

```text
http://localhost:8501
```

## 12. Next Build Steps

Priority order:

1. Test real Gemma provider quality with sample employment documents.
2. Tune the prompt and schema based on real model behavior.
3. Add scanned PDF OCR.
4. Add real translation/local language output.
5. Save analysis JSON files to `outputs/`.
6. Add sample documents to `sample_docs/`.
7. Improve UI severity styling for risk flags.
8. Add downloadable analysis report.
9. Add privacy notice and delete uploaded files after processing.
10. Add end-to-end integration tests for the chosen Gemma provider.

## 13. Current Status

MVP status: working local prototype.

Ready for:

- PDF text extraction demos.
- Image OCR demos if Tesseract is installed.
- Structured mock analysis demos.
- Real Gemma provider trials with safe fallback enabled.
