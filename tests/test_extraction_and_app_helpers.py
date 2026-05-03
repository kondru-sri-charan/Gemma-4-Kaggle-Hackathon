from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import pytest
from PIL import Image

import app
from services.analyzer import UNREADABLE_DOCUMENT_TYPE, analyze_document


@dataclass
class FakeUpload:
    name: str
    type: str
    data: bytes

    @property
    def size(self) -> int:
        return len(self.data)

    def getvalue(self) -> bytes:
        return self.data


def make_png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (20, 20), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def make_pdf_bytes(text: str) -> bytes:
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    return document.write()


@pytest.mark.parametrize(
    ("file_name", "expected"),
    [
        ("offer.pdf", "pdf"),
        ("scan.PNG", "png"),
        ("photo.jpeg", "jpeg"),
        ("archive", ""),
    ],
)
def test_get_file_extension(file_name: str, expected: str) -> None:
    assert app.get_file_extension(file_name) == expected


@pytest.mark.parametrize(
    ("file_name", "mime_type", "expected"),
    [
        ("offer.pdf", "", "pdf"),
        ("anything.bin", "application/pdf", "pdf"),
        ("scan.png", "", "image"),
        ("photo.unknown", "image/jpeg", "image"),
        ("notes.txt", "text/plain", None),
    ],
)
def test_classify_uploaded_file(file_name: str, mime_type: str, expected: str | None) -> None:
    assert app.classify_uploaded_file(file_name, mime_type) == expected


def test_extract_document_text_rejects_unsupported_file() -> None:
    result = app.extract_document_text(
        FakeUpload("notes.txt", "text/plain", b"hello")
    )

    assert result.can_analyze is False
    assert result.text == ""
    assert "Unsupported file type" in result.error


def test_extract_document_text_rejects_empty_file() -> None:
    result = app.extract_document_text(
        FakeUpload("empty.pdf", "application/pdf", b"")
    )

    assert result.can_analyze is False
    assert "empty" in result.error.lower()


def test_extract_pdf_text_returns_page_text() -> None:
    result = app.extract_pdf_text(make_pdf_bytes("Employment Offer Letter"))

    assert result.error is None
    assert "Page 1" in result.text
    assert "Employment Offer Letter" in result.text


def test_extract_pdf_text_handles_invalid_pdf_bytes() -> None:
    result = app.extract_pdf_text(b"not a pdf")

    assert result.text == ""
    assert result.error.startswith("Could not extract text from the PDF")


def test_blank_pdf_extracts_no_text_and_analyzes_as_unreadable() -> None:
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    document.new_page()

    extraction = app.extract_pdf_text(document.write())
    analysis = analyze_document(extraction.text, "English")

    assert extraction.text == ""
    # Blank PDFs still render a page image (which shows nothing), so the
    # extractor reports vision mode rather than hard-failing. In mock mode
    # the analyzer still falls back to the "unreadable" response because
    # the heuristic analyzer cannot see images.
    assert extraction.evidence in {"vision", "text"}
    assert analysis["document_type"] == UNREADABLE_DOCUMENT_TYPE
    assert analysis["risk_flags"] == []


def test_extract_document_text_routes_pdf_upload() -> None:
    upload = FakeUpload("offer.pdf", "application/pdf", make_pdf_bytes("Salary: Rs 18000"))
    result = app.extract_document_text(upload)

    assert result.error is None
    assert "Salary: Rs 18000" in result.text


def test_extract_pdf_text_also_returns_page_images_for_vision_path() -> None:
    """Hybrid extraction: text AND page images both flow through for PDFs."""
    result = app.extract_pdf_text(make_pdf_bytes("Employment Offer Letter"))

    assert result.error is None
    assert "Employment Offer Letter" in result.text
    assert result.images, "page images should be rendered for the vision path"
    assert all(isinstance(img, str) and img for img in result.images)
    assert result.evidence == "hybrid"


def test_extract_image_normalizes_to_png_b64_for_vision() -> None:
    """Image uploads go straight to the vision path: no OCR, no Tesseract."""
    result = app.extract_image(make_png_bytes())

    assert result.error is None
    assert result.can_analyze is True
    assert result.text == ""
    assert len(result.images) == 1
    # The base64 is plain (no data URL prefix) so Ollama can accept it as-is.
    assert not result.images[0].startswith("data:")
    assert result.evidence == "vision"


def test_extract_image_rejects_invalid_bytes() -> None:
    """Broken image bytes must not crash the pipeline and must be flagged."""
    result = app.extract_image(b"not an image")

    assert result.can_analyze is False
    assert result.text == ""
    assert result.images == []
    assert "could not be read" in result.error


def test_extract_image_converts_non_rgb_modes_to_png() -> None:
    """Upload flows may hand us non-RGB images (palette, RGBA); normalize them."""
    buf = BytesIO()
    Image.new("RGBA", (20, 20), color=(255, 255, 255, 255)).save(buf, format="PNG")
    result = app.extract_image(buf.getvalue())

    assert result.error is None
    assert result.images
    assert result.evidence == "vision"
