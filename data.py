"""
PDF Text Preprocessing for 'Attention Is All You Need'
Extracts and cleans raw text from the paper, separating sections and references.
"""

import os
import re
from pathlib import Path
import pymupdf


def extract_raw_text(pdf_path: str) -> str:
    """Extract raw text from PDF document."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages)


def preprocess_paper(raw_text: str):
    """
    Cleans and structures the research paper text:
    1. Removes OCR tags, page numbers & conference footers
    2. Separates body text and references
    3. Fixes OCR errors, ligatures & broken hyphens
    4. Removes author notes and footnotes
    5. Removes bracketed citations [12]
    6. Extracts sections into a structured dictionary
    """
    # -------------------------------------------------------------
    # 1. REMOVE PAGE MARKERS & CONFERENCE FOOTERS
    # -------------------------------------------------------------
    # Remove OCR tags
    text = re.sub(r'==(?:Start|End) of (?:OCR|PDF).*?==', '', raw_text)

    # Normalize standalone section numbers that appear on their own lines (e.g. \n1\nIntroduction -> \n1 Introduction)
    text = re.sub(r'\n([1-7])\s*\n([A-Z][^\n]+)', r'\n\1 \2', text)

    # Remove running conference header/footer
    text = re.sub(r'31st Conference on Neural Information Processing Systems.*?\n', '', text, flags=re.IGNORECASE)

    # Remove standalone page numbers
    text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)

    # -------------------------------------------------------------
    # 2. SEPARATE BODY AND REFERENCES
    # -------------------------------------------------------------
    split_ref = re.split(r'\nReferences\s*\n', text, maxsplit=1)
    body_text = split_ref[0]
    references_text = split_ref[1] if len(split_ref) > 1 else ""

    # -------------------------------------------------------------
    # 3. FIX OCR ERRORS & HYPHENATIONS
    # -------------------------------------------------------------
    # Replace common PDF ligatures
    ligatures = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
    for search, replace in ligatures.items():
        body_text = body_text.replace(search, replace)
        references_text = references_text.replace(search, replace)

    # Fix line-break hyphens across newlines (e.g., "feed-\nforward" -> "feed-forward")
    body_text = re.sub(r'(\w+)-\n\s*(\w+)', r'\1\2', body_text)
    references_text = re.sub(r'(\w+)-\n\s*(\w+)', r'\1\2', references_text)

    # -------------------------------------------------------------
    # 4. REMOVE FOOTNOTES
    # -------------------------------------------------------------
    # Page 1 author notes
    body_text = re.sub(r'∗Equal contribution.*?(?=\n\s*1\s+Introduction)', '', body_text, flags=re.DOTALL)
    body_text = re.sub(r'†Work performed while at Google Brain\..*?(?=\n\s*1\s+Introduction)', '', body_text, flags=re.DOTALL)
    body_text = re.sub(r'‡Work performed while at Google Research\..*?(?=\n\s*1\s+Introduction)', '', body_text, flags=re.DOTALL)
    # Footnote 4 on Page 4
    body_text = re.sub(r'4To illustrate why the dot products.*?variance dk\.', '', body_text, flags=re.DOTALL)
    # Footnote 5 on Page 8
    body_text = re.sub(r'5We used values of 2\.8.*?respectively\.', '', body_text)

    # -------------------------------------------------------------
    # 5. REMOVE BRACKETED CITATIONS (e.g., [12], [29, 2, 5])
    # -------------------------------------------------------------
    body_text = re.sub(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]', '', body_text)

    # -------------------------------------------------------------
    # 6. EXTRACT SECTIONS INTO A STRUCTURED DICTIONARY
    # -------------------------------------------------------------
    sections = {}

    # Extract Abstract
    abstract_match = re.search(r'Abstract\s*\n(.*?)(?=\n1\s+Introduction)', body_text, re.DOTALL)
    sections['Abstract'] = re.sub(r'\s+', ' ', abstract_match.group(1)).strip() if abstract_match else ""

    # Define standard headings
    headings = [
        "1 Introduction",
        "2 Background",
        "3 Model Architecture",
        "4 Why Self-Attention",
        "5 Training",
        "6 Results",
        "7 Conclusion",
        "Acknowledgements"
    ]

    # Split text by section numbers
    pattern = r'(?=\n(?:' + '|'.join([re.escape(h) for h in headings]) + r')\b)'
    parts = re.split(pattern, body_text)

    for part in parts:
        for heading in headings:
            if part.strip().startswith(heading):
                clean_section_content = re.sub(r'\s+', ' ', part.replace(heading, '', 1)).strip()
                sections[heading] = clean_section_content

    # Clean references text as well
    references_clean = re.sub(r'\s+', ' ', references_text).strip()

    return sections, references_clean


def locate_pdf() -> str:
    """Finds the paper PDF in ./data or root."""
    paths = [
        os.path.join("data", "NIPS-2017-attention-is-all-you-need-Paper.pdf"),
        "NIPS-2017-attention-is-all-you-need-Paper.pdf",
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not find NIPS-2017-attention-is-all-you-need-Paper.pdf")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else locate_pdf()
    print(f"Reading and extracting: {pdf_path}")

    # 1. Extract raw text from PDF
    raw_text = extract_raw_text(pdf_path)

    # 2. Preprocess paper
    sections, references = preprocess_paper(raw_text)

    # 3. Save clean text to file
    out_dir = "data" if os.path.exists("data") else "."
    output_file = os.path.join(out_dir, "clean_paper_text.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        for heading, content in sections.items():
            f.write(f"=== {heading} ===\n\n")
            f.write(content + "\n\n")

        if references:
            f.write("=== References ===\n\n")
            f.write(references + "\n\n")

    print(f"[✓] Successfully preprocessed and saved clean text to: {output_file}")
    print("\nExtracted Sections:")
    for heading, content in sections.items():
        print(f"  • {heading:<20}: {len(content)} characters")
    if references:
        print(f"  • {'References':<20}: {len(references)} characters")
