from __future__ import annotations

import sys
import types
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
    assert "No selectable text was found" in extraction.error
    assert analysis["document_type"] == UNREADABLE_DOCUMENT_TYPE
    assert analysis["risk_flags"] == []


def test_extract_document_text_routes_pdf_upload() -> None:
    upload = FakeUpload("offer.pdf", "application/pdf", make_pdf_bytes("Salary: Rs 18000"))
    result = app.extract_document_text(upload)

    assert result.error is None
    assert "Salary: Rs 18000" in result.text


def test_extract_image_text_uses_pytesseract_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pytesseract = types.SimpleNamespace(
        TesseractNotFoundError=RuntimeError,
        image_to_string=lambda image: "Mock OCR employment text",
    )
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)

    result = app.extract_image_text(make_png_bytes())

    assert result.error is None
    assert result.text == "Mock OCR employment text"


def test_extract_image_text_handles_tesseract_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTesseractNotFoundError(Exception):
        pass

    def raise_missing_engine(image: Image.Image) -> str:
        raise FakeTesseractNotFoundError()

    fake_pytesseract = types.SimpleNamespace(
        TesseractNotFoundError=FakeTesseractNotFoundError,
        image_to_string=raise_missing_engine,
    )
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)

    result = app.extract_image_text(make_png_bytes())

    assert result.text == ""
    assert "Tesseract OCR is not installed" in result.error


def test_extract_image_text_handles_invalid_image() -> None:
    result = app.extract_image_text(b"not an image")

    assert result.can_analyze is False
    assert "could not be read" in result.error
