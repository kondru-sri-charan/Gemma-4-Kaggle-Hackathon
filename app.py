from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Iterable

import streamlit as st
from PIL import Image, UnidentifiedImageError

from services.analyzer import DocumentAnalysis, RiskFlag, analyze_document_with_status
from services.gemma_client import MODEL_MODE_REAL, describe_model_mode, get_gemma_config


SUPPORTED_LANGUAGES = ["English", "Hindi", "Marathi", "Tamil", "Telugu"]
SUPPORTED_FILE_TYPES = ["pdf", "png", "jpg", "jpeg"]
MAX_VISION_PAGES = 4
VISION_DPI = 144


@dataclass(frozen=True)
class ExtractionResult:
    text: str = ""
    # Base64-encoded PNG bytes of each page/image. We pass these to Gemma 4
    # for the hybrid vision path. In mock mode they are ignored.
    images: list[str] = field(default_factory=list)
    # Human-readable notes: extraction warnings, empty-text explanations, etc.
    # Rendered under the extracted-text expander when present.
    error: str | None = None
    # When False, the UI should not proceed to analysis (truly broken upload).
    can_analyze: bool = True
    # Label the dominant evidence source so we can tag the UI accordingly.
    # "text" - selectable text was found
    # "vision" - no text, model will read the image(s)
    # "hybrid" - both text and image(s) available
    evidence: str = "text"


def get_file_extension(file_name: str) -> str:
    return Path(file_name).suffix.lower().lstrip(".")


def classify_uploaded_file(file_name: str, file_type: str) -> str | None:
    extension = get_file_extension(file_name)

    if extension == "pdf" or file_type == "application/pdf":
        return "pdf"

    if extension in {"png", "jpg", "jpeg"} or file_type.startswith("image/"):
        return "image"

    return None


def _render_pdf_pages_to_png_b64(
    pdf_document, max_pages: int = MAX_VISION_PAGES, dpi: int = VISION_DPI
) -> list[str]:
    """Rasterize up to ``max_pages`` PDF pages and return them as base64 PNGs.

    PyMuPDF's 72 DPI baseline is tweaked to ``dpi`` for readability at the
    cost of token budget. 144 DPI gives Gemma 4's vision encoder plenty of
    resolution on typical offer letters without blowing up latency.
    """
    import fitz  # local alias for type checkers

    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    images: list[str] = []
    for page in pdf_document[:max_pages]:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        images.append(base64.b64encode(pix.tobytes("png")).decode("ascii"))
    return images


def extract_pdf_text(file_bytes: bytes) -> ExtractionResult:
    """Hybrid PDF extraction: pull selectable text AND render page images.

    The text path is fast and works for clean, digitally-generated PDFs. The
    image path gives Gemma 4 a direct look at the document, which recovers
    scanned and image-only PDFs without needing Tesseract. Both evidence
    streams are passed on to the analyzer when real mode is active.
    """
    try:
        import fitz
    except ImportError:
        return ExtractionResult(
            error="PyMuPDF is not installed. Run `pip install -r requirements.txt`.",
        )

    try:
        page_texts: list[str] = []
        images: list[str] = []
        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                if text:
                    page_texts.append(f"Page {page_number}\n{text}")

            # Render page images for the vision path regardless of whether
            # text extraction found anything. This keeps latency bounded via
            # MAX_VISION_PAGES while still giving the model something to
            # look at on scanned or stamped pages.
            try:
                images = _render_pdf_pages_to_png_b64(document)
            except Exception:
                images = []

        extracted_text = "\n\n".join(page_texts).strip()

        if not extracted_text and not images:
            return ExtractionResult(
                error=(
                    "No selectable text and no renderable pages were found. "
                    "The file may be empty or corrupted."
                ),
            )

        if not extracted_text:
            return ExtractionResult(
                text="",
                images=images,
                error=(
                    "No selectable text was found in this PDF. The AI will read "
                    "the page images directly (vision mode)."
                ),
                evidence="vision",
            )

        return ExtractionResult(
            text=extracted_text,
            images=images,
            evidence="hybrid" if images else "text",
        )
    except Exception as exc:
        return ExtractionResult(error=f"Could not extract text from the PDF: {exc}")


def _png_b64_from_image_bytes(file_bytes: bytes) -> str | None:
    """Normalize an uploaded image to a base64-encoded PNG.

    Gemma 4 accepts JPG and PNG, but standardizing on PNG avoids any
    ambiguity about data URL prefixes and keeps the downstream code simple.
    """
    try:
        with Image.open(BytesIO(file_bytes)) as image:
            if image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            buf = BytesIO()
            image.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")
    except (UnidentifiedImageError, OSError):
        return None


def extract_image(file_bytes: bytes) -> ExtractionResult:
    """Prepare an image upload for the vision path.

    We no longer OCR on the Python side. Gemma 4's vision encoder reads
    images directly, so we just normalize the upload to PNG and forward it
    to the analyzer. This drops the hard Tesseract dependency and works on
    phone photos, scans, and stamped documents.
    """
    image_b64 = _png_b64_from_image_bytes(file_bytes)
    if image_b64 is None:
        return ExtractionResult(
            error="The uploaded image could not be read. Please upload a valid PNG or JPG file.",
            can_analyze=False,
        )

    return ExtractionResult(
        text="",
        images=[image_b64],
        evidence="vision",
    )


def extract_document_text(uploaded_file) -> ExtractionResult:
    file_type = uploaded_file.type or ""
    document_kind = classify_uploaded_file(uploaded_file.name, file_type)

    if document_kind is None:
        return ExtractionResult(
            error="Unsupported file type. Please upload a PDF, PNG, JPG, or JPEG file.",
            can_analyze=False,
        )

    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        return ExtractionResult(error="The uploaded file is empty.", can_analyze=False)

    if document_kind == "pdf":
        return extract_pdf_text(file_bytes)

    return extract_image(file_bytes)


def render_list(items: Iterable[str]) -> None:
    for item in items:
        st.markdown(f"- {item}")


def render_card(title: str, body: str | Iterable[str]) -> None:
    with st.container(border=True):
        st.subheader(title)
        if isinstance(body, str):
            st.write(body)
        else:
            render_list(body)


def render_risk_flags(risk_flags: list[RiskFlag]) -> None:
    st.subheader("Risk Flags")

    if not risk_flags:
        st.info("No risk flags were returned.")
        return

    for risk in risk_flags:
        severity = risk.get("severity", "Low")
        with st.container(border=True):
            title_column, severity_column = st.columns([3, 1])
            with title_column:
                st.markdown(f"**{risk.get('risk_title', 'Risk flag')}**")
            with severity_column:
                st.caption(f"Severity: {severity}")

            st.write(risk.get("risk_explanation", "No explanation available."))
            st.caption(f"Source: {risk.get('source_text', 'Not found in extracted text')}")
            st.markdown(f"**Suggested action:** {risk.get('suggested_action', 'Review this clause carefully.')}")


def render_analysis_cards(analysis: DocumentAnalysis) -> None:
    top_row = st.columns([2, 1])
    with top_row[0]:
        render_card("Document Type", str(analysis["document_type"]))
    with top_row[1]:
        render_card("Confidence Level", str(analysis["confidence_level"]))

    render_card("Simple Explanation", str(analysis["simple_explanation"]))
    render_card("Local Language Summary", str(analysis["local_language_summary"]))
    render_risk_flags(analysis["risk_flags"])

    first_row = st.columns(2)
    with first_row[0]:
        render_card("Key Points", analysis["key_points"])
    with first_row[1]:
        render_card("Next Actions", analysis["next_actions"])

    render_card("Questions to Ask", analysis["questions_to_ask"])

    render_card("Source References", analysis["source_references"])


def render_analysis_source(source: str) -> None:
    st.caption(f"Analysis source: `{source}`")


def render_tool_log(tool_log: list[dict]) -> None:
    """Surface the labor-law citations Gemma pulled in during analysis.

    This is the "grounded" story: every time the model called
    lookup_labor_law, we show the statute reference so the user can see
    their analysis is backed by real law rather than generic guidance.
    """
    if not tool_log:
        return

    with st.container(border=True):
        st.subheader("Grounded in Indian labor law")
        st.caption(
            f"Gemma 4 consulted the labor-law knowledge base {len(tool_log)} "
            "time(s) to ground this analysis."
        )
        for call in tool_log:
            name = call.get("name", "tool")
            arguments = call.get("arguments") or {}
            result = call.get("result")
            with st.expander(
                f"{name}({_format_tool_args(arguments)})",
                expanded=False,
            ):
                if isinstance(result, list):
                    for row in result:
                        if not isinstance(row, dict):
                            st.write(row)
                            continue
                        st.markdown(
                            f"**{row.get('title', '')}**  \n"
                            f"_{row.get('statute_reference', '')}_\n\n"
                            f"{row.get('summary', '')}"
                        )
                elif isinstance(result, dict) and "error" in result:
                    st.warning(result["error"])
                else:
                    st.write(result)


def _format_tool_args(arguments: dict) -> str:
    parts = []
    for key, value in arguments.items():
        parts.append(f"{key}={value!r}")
    return ", ".join(parts)


def render_evidence_source(extraction: ExtractionResult, model_config) -> None:
    """Show a small badge explaining whether text, vision, or both were used."""
    if model_config.model_mode != MODEL_MODE_REAL:
        return

    if extraction.evidence == "vision":
        st.caption("Evidence: vision — Gemma 4 reads the page images directly.")
    elif extraction.evidence == "hybrid":
        st.caption("Evidence: text + vision — extracted text plus page images.")
    else:
        st.caption("Evidence: text — extracted directly from the document.")


def render_extracted_text(result: ExtractionResult) -> None:
    with st.expander("Extracted Document Text", expanded=False):
        if result.error:
            st.warning(result.error)

        if result.text:
            st.text_area(
                "Extracted Document Text",
                value=result.text,
                height=260,
                disabled=True,
                label_visibility="collapsed",
            )
        elif result.images and not result.error:
            st.info(
                "No selectable text in this document. The analysis will use the "
                "page image(s) directly."
            )
        elif not result.error:
            st.info("No text was extracted from the document.")


def render_upload_preview(uploaded_file) -> None:
    if uploaded_file is None:
        return

    file_type = uploaded_file.type or "unknown type"
    st.caption(
        f"Uploaded: {uploaded_file.name} | "
        f"{file_type} | "
        f"{uploaded_file.size / 1024:.1f} KB"
    )

    document_kind = classify_uploaded_file(uploaded_file.name, file_type)

    if document_kind == "image":
        try:
            image = Image.open(BytesIO(uploaded_file.getvalue()))
            st.image(image, caption="Document preview", use_container_width=True)
        except UnidentifiedImageError:
            st.warning("Image preview is unavailable for this file.")
    elif document_kind == "pdf":
        st.info("PDF uploaded. Text will be extracted when you analyze the document.")
    else:
        st.warning("Unsupported file type. Please upload a PDF, PNG, JPG, or JPEG file.")


def main() -> None:
    st.set_page_config(
        page_title="Sahaayak AI",
        layout="wide",
    )

    st.title("Sahaayak AI")
    st.write("Upload an employment document and get a simple, structured explanation.")

    with st.sidebar:
        st.header("Document")
        model_config = get_gemma_config()
        model_status = f"Model mode: {describe_model_mode(model_config)}"
        if model_config.model_mode == MODEL_MODE_REAL:
            st.success(model_status)
        else:
            st.info(model_status)

        uploaded_file = st.file_uploader(
            "Upload document",
            type=SUPPORTED_FILE_TYPES,
            accept_multiple_files=False,
        )
        target_language = st.selectbox("Target language", SUPPORTED_LANGUAGES)
        analyze_clicked = st.button("Analyze Document", type="primary", use_container_width=True)

    left_column, right_column = st.columns([1, 2], gap="large")

    with left_column:
        st.subheader("Uploaded Document")
        if uploaded_file is None:
            st.info("Upload a PDF or image to begin.")
        else:
            render_upload_preview(uploaded_file)

    with right_column:
        st.subheader("Structured Explanation")
        if analyze_clicked and uploaded_file is None:
            st.warning("Please upload a document before analyzing.")
        elif analyze_clicked and uploaded_file is not None:
            extraction = extract_document_text(uploaded_file)
            render_extracted_text(extraction)
            render_evidence_source(extraction, model_config)

            if extraction.can_analyze:
                # In mock mode we ignore images because the heuristic analyzer
                # cannot see them. In real mode we forward them so Gemma 4 can
                # read the document directly.
                images_to_use = (
                    extraction.images
                    if model_config.model_mode == MODEL_MODE_REAL
                    else None
                )
                with st.spinner("Analyzing with Gemma 4..."):
                    analysis_result = analyze_document_with_status(
                        extraction.text,
                        target_language,
                        images=images_to_use,
                    )
                render_analysis_source(analysis_result.source)
                render_tool_log(analysis_result.tool_log)
                render_analysis_cards(analysis_result.analysis)
        else:
            st.info("Click Analyze Document to generate an explanation.")


if __name__ == "__main__":
    main()
