from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import streamlit as st
from PIL import Image, UnidentifiedImageError

from services.analyzer import DocumentAnalysis, RiskFlag, analyze_document_with_status
from services.gemma_client import MODEL_MODE_REAL, describe_model_mode, get_gemma_config


SUPPORTED_LANGUAGES = ["English", "Hindi", "Marathi", "Tamil", "Telugu"]
SUPPORTED_FILE_TYPES = ["pdf", "png", "jpg", "jpeg"]


@dataclass(frozen=True)
class ExtractionResult:
    text: str = ""
    error: str | None = None
    can_analyze: bool = True


def get_file_extension(file_name: str) -> str:
    return Path(file_name).suffix.lower().lstrip(".")


def classify_uploaded_file(file_name: str, file_type: str) -> str | None:
    extension = get_file_extension(file_name)

    if extension == "pdf" or file_type == "application/pdf":
        return "pdf"

    if extension in {"png", "jpg", "jpeg"} or file_type.startswith("image/"):
        return "image"

    return None


def extract_pdf_text(file_bytes: bytes) -> ExtractionResult:
    try:
        import fitz
    except ImportError:
        return ExtractionResult(
            error="PyMuPDF is not installed. Run `pip install -r requirements.txt`.",
        )

    try:
        page_texts = []
        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                if text:
                    page_texts.append(f"Page {page_number}\n{text}")

        extracted_text = "\n\n".join(page_texts).strip()
        if not extracted_text:
            return ExtractionResult(
                error=(
                    "No selectable text was found in this PDF. It may be scanned; "
                    "image-based PDF OCR can be added in a later version."
                ),
            )

        return ExtractionResult(text=extracted_text)
    except Exception as exc:
        return ExtractionResult(error=f"Could not extract text from the PDF: {exc}")


def extract_image_text(file_bytes: bytes) -> ExtractionResult:
    try:
        import pytesseract
    except ImportError:
        return ExtractionResult(
            error="pytesseract is not installed. Run `pip install -r requirements.txt`.",
        )

    try:
        image = Image.open(BytesIO(file_bytes))
        extracted_text = pytesseract.image_to_string(image).strip()

        if not extracted_text:
            return ExtractionResult(
                error="No text was detected in this image. Try a clearer scan or photo.",
            )

        return ExtractionResult(text=extracted_text)
    except UnidentifiedImageError:
        return ExtractionResult(
            error="The uploaded image could not be read. Please upload a valid PNG or JPG file.",
            can_analyze=False,
        )
    except pytesseract.TesseractNotFoundError:
        return ExtractionResult(
            error=(
                "Tesseract OCR is not installed or is not on PATH. "
                "Install the Tesseract engine to enable image text extraction."
            ),
        )
    except Exception as exc:
        return ExtractionResult(error=f"Could not extract text from the image: {exc}")


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

    return extract_image_text(file_bytes)


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

            if extraction.can_analyze:
                analysis_result = analyze_document_with_status(extraction.text, target_language)
                render_analysis_source(analysis_result.source)
                render_analysis_cards(analysis_result.analysis)
        else:
            st.info("Click Analyze Document to generate a mock explanation.")


if __name__ == "__main__":
    main()
