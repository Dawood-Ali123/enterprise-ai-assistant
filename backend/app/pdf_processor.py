import fitz
import pytesseract
from PIL import Image
from io import BytesIO


def extract_text(pdf_path: str) -> str:
    """
    Extract text from a PDF.

    First tries normal PDF text extraction.
    If the extracted text is empty or meaningless,
    falls back to OCR.
    """

    doc = fitz.open(pdf_path)

    extracted_pages = []

    for page in doc:
        text = page.get_text("text").strip()
        extracted_pages.append(text)

    normal_text = "\n\n".join(extracted_pages).strip()

    # If enough meaningful text was extracted,
    # use normal extraction.
    if _has_meaningful_text(normal_text):
        doc.close()
        return normal_text

    # Otherwise use OCR
    ocr_text = []

    for page_number, page in enumerate(doc):
        pix = page.get_pixmap(
            matrix=fitz.Matrix(2, 2),
            alpha=False
        )

        image = Image.open(BytesIO(pix.tobytes("png")))

        text = pytesseract.image_to_string(
            image,
            lang="eng"
        ).strip()

        if text:
            ocr_text.append(
                f"Page {page_number + 1}\n{text}"
            )

    doc.close()

    return "\n\n".join(ocr_text).strip()


def _has_meaningful_text(text: str) -> bool:
    """
    Detect whether normal PDF extraction produced
    useful text instead of page numbers / PDF metadata.
    """

    if not text:
        return False

    # Remove whitespace
    cleaned = " ".join(text.split())

    # Very small extraction is probably not useful
    if len(cleaned) < 100:
        return False

    # Count alphabetic characters
    alphabetic_chars = sum(
        char.isalpha()
        for char in cleaned
    )

    # If almost no real alphabetic content exists,
    # OCR is needed.
    if alphabetic_chars < 50:
        return False

    return True