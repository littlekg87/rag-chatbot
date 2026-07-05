from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pdfplumber


ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "text" / "Marcus-Aurelius-Meditations.pdf"
DATA_DIR = ROOT / "data"
XML_PATH = DATA_DIR / "meditations.xml"
CHUNKS_PATH = DATA_DIR / "meditations_chunks.jsonl"

BOOK_WORDS = {
    "FIRST": 1,
    "SECOND": 2,
    "THIRD": 3,
    "FOURTH": 4,
    "FIFTH": 5,
    "SIXTH": 6,
    "SEVENTH": 7,
    "EIGHTH": 8,
    "NINTH": 9,
    "TENTH": 10,
    "ELEVENTH": 11,
    "TWELFTH": 12,
}

BOOK_RE = re.compile(
    r"^THE (?P<word>FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH) BOOK$",
    re.MULTILINE,
)

# Some section headings are printed as "IV Why..." without a period. Require a
# period for one-letter numerals to avoid matching ordinary sentences beginning
# with "I ".
SECTION_RE = re.compile(
    r"(?m)^(?:(?P<roman_dotted>[IVXLCDM]{1,8})\.\s+|(?P<roman_multi>[IVXLCDM]{2,8})\s+)"
)


def roman_to_int(value: str) -> int:
    numerals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    previous = 0
    for char in reversed(value):
        current = numerals[char]
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total


def int_to_roman(value: int) -> str:
    pairs = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    result = []
    remaining = value
    for amount, symbol in pairs:
        while remaining >= amount:
            result.append(symbol)
            remaining -= amount
    return "".join(result)


def clean_page_text(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "MEDITATIONS OF MARCUS AURELIUS":
            continue
        if re.fullmatch(r"(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH) BOOK", line):
            continue
        if line.startswith("Marcus Aurelius' Meditations"):
            continue
        if re.fullmatch(r"Page \d+ of \d+", line):
            continue
        if re.fullmatch(r"\d+", line):
            continue
        lines.append(line)

    page = "\n".join(lines)
    # Repair drop-cap extraction, e.g. "O\nF MY..." -> "OF MY...".
    page = re.sub(r"(^|\n)([A-Z])\n([A-Z]{2,})", r"\1\2\3", page)
    page = re.sub(r"(^|\n)([A-Z])\n([A-Z])(\s+)", r"\1\2\3\4", page)
    # Repair an OCR-like extraction issue observed in Book 7: "III." -> "Ill."
    page = re.sub(r"(?m)^Ill\.", "III.", page)
    # Repair hyphenated line endings while preserving real hyphenated compounds.
    page = re.sub(r"([A-Za-z])-[\n\r]+([a-z])", r"\1\2", page)
    return page


def extract_body_text() -> str:
    pages: list[str] = []
    with pdfplumber.open(PDF_PATH) as pdf:
        start_index: int | None = None
        stop_index: int | None = None

        page_texts = [page.extract_text() or "" for page in pdf.pages]
        for index, text in enumerate(page_texts):
            if index > 5 and re.search(r"^THE FIRST BOOK$", text, re.MULTILINE):
                start_index = index
                break
        if start_index is None:
            raise RuntimeError("Could not find the first page of THE FIRST BOOK.")

        for index in range(start_index, len(page_texts)):
            if re.search(r"^APPENDIX$|^Appendix$", page_texts[index], re.MULTILINE):
                stop_index = index
                break
        if stop_index is None:
            raise RuntimeError("Could not find the Appendix boundary.")

        for text in page_texts[start_index:stop_index]:
            pages.append(clean_page_text(text))

    return "\n".join(pages)


def normalize_section_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    # The source sometimes leaves spaces before punctuation after line extraction.
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    return text


def parse_sections(book_number: int, raw_book_text: str) -> list[dict[str, object]]:
    matches = list(SECTION_RE.finditer(raw_book_text))
    sections: list[dict[str, object]] = []

    first_start = matches[0].start() if matches else len(raw_book_text)
    first_text = normalize_section_text(raw_book_text[:first_start])
    if first_text:
        sections.append({"number": 1, "roman": "I", "text": first_text})

    for index, match in enumerate(matches):
        roman = match.group("roman_dotted") or match.group("roman_multi")
        if not roman:
            continue
        section_number = roman_to_int(roman)
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(raw_book_text)
        section_text = normalize_section_text(raw_book_text[match.end() : next_start])
        if not section_text:
            continue
        sections.append({"number": section_number, "roman": roman, "text": section_text})

    sections.sort(key=lambda section: int(section["number"]))
    return sections


def parse_books(body_text: str) -> list[dict[str, object]]:
    matches = list(BOOK_RE.finditer(body_text))
    books: list[dict[str, object]] = []

    for index, match in enumerate(matches):
        word = match.group("word")
        book_number = BOOK_WORDS[word]
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(body_text)
        raw_book_text = body_text[match.end() : next_start].strip()
        sections = parse_sections(book_number, raw_book_text)
        books.append(
            {
                "number": book_number,
                "title": f"THE {word} BOOK",
                "sections": sections,
            }
        )

    return books


def build_xml(books: list[dict[str, object]]) -> ET.ElementTree:
    root = ET.Element(
        "meditations",
        {
            "source_pdf": str(PDF_PATH.relative_to(ROOT)).replace("\\", "/"),
            "translator": "Meric Casaubon",
            "edition_note": "Philaletheians PDF, v. 8.16, uploaded 14 July 2013",
        },
    )

    for book in books:
        book_el = ET.SubElement(
            root,
            "book",
            {
                "number": str(book["number"]),
                "title": str(book["title"]),
            },
        )
        for section in book["sections"]:  # type: ignore[index]
            section_number = int(section["number"])
            section_el = ET.SubElement(
                book_el,
                "section",
                {
                    "number": str(section_number),
                    "roman": str(section["roman"]),
                    "id": f"book_{int(book['number']):02d}_section_{section_number:03d}",
                    "source": f"Meditations, Book {int(book['number'])}, Section {section_number}",
                },
            )
            section_el.text = str(section["text"])

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    return tree


def write_chunks_from_xml(xml_path: Path, chunks_path: Path) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    with chunks_path.open("w", encoding="utf-8", newline="\n") as handle:
        for book_el in root.findall("book"):
            book_number = int(book_el.attrib["number"])
            for section_el in book_el.findall("section"):
                text = (section_el.text or "").strip()
                section_number = int(section_el.attrib["number"])
                record = {
                    "id": section_el.attrib["id"],
                    "book": book_number,
                    "section": section_number,
                    "section_roman": section_el.attrib["roman"],
                    "source": section_el.attrib["source"],
                    "text": text,
                    "char_count": len(text),
                    "word_count": len(text.split()),
                    "themes": [],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_summary(books: list[dict[str, object]]) -> None:
    total_sections = sum(len(book["sections"]) for book in books)  # type: ignore[arg-type]
    print(f"books: {len(books)}")
    print(f"sections: {total_sections}")
    for book in books:
        sections = book["sections"]  # type: ignore[index]
        numbers = [section["number"] for section in sections]
        missing = [n for n in range(1, max(numbers) + 1) if n not in numbers] if numbers else []
        status = "ok" if not missing else f"missing {missing}"
        print(f"book {book['number']:>2}: {len(sections):>2} sections ({status})")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    body_text = extract_body_text()
    books = parse_books(body_text)
    if len(books) != 12:
        raise RuntimeError(f"Expected 12 books, found {len(books)}.")

    tree = build_xml(books)
    tree.write(XML_PATH, encoding="utf-8", xml_declaration=True)
    write_chunks_from_xml(XML_PATH, CHUNKS_PATH)
    print_summary(books)
    print(f"wrote: {XML_PATH.relative_to(ROOT)}")
    print(f"wrote: {CHUNKS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
