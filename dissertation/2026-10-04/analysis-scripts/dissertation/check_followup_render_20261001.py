"""Check final native document pagination and produce bounded visual QA sheets."""

import hashlib
import json
import re
from pathlib import Path

import fitz
from docx import Document
from docx.oxml.ns import qn
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent / "followup-20261001/manuscript-work"
REVIEW = HERE
PDF = REVIEW / "Tallam_Krti_Praxis_Engineering_Working_2026-10-01.pdf"


def normalize(value):
    return re.sub(r"\s+", "", value)


def check_fonts(document):
    fonts = {}
    for page in document:
        for reference, _, _, name, *_ in page.get_fonts():
            embedded = bool(document.extract_font(reference)[3])
            if "TimesNewRoman" not in normalize(name) or not embedded:
                raise ValueError(f"Unexpected or unembedded manuscript font: {name}")
            fonts[reference] = {"name": name, "embedded": embedded}
    if not fonts:
        raise ValueError("No embedded manuscript fonts found")
    return list(fonts.values())


def check_cached_navigation(document, page_map):
    observed = {}
    for instruction in document._element.xpath(".//w:instrText"):
        match = re.search(r"\bPAGEREF\s+(\w+)", instruction.text or "")
        if match:
            name = match.group(1)
            if name in observed:
                raise ValueError(f"Duplicate navigation cache: {name}")
            observed[name] = "".join(instruction.getparent().xpath("./w:t/text()"))
    for hyperlink in document._element.xpath(".//w:hyperlink[@w:anchor]"):
        name = hyperlink.get(qn("w:anchor"))
        if name in page_map:
            if name in observed:
                raise ValueError(f"Duplicate navigation cache: {name}")
            observed[name] = "".join(hyperlink.xpath(".//w:t/text()"))
    if observed != page_map:
        raise ValueError("Stale or missing DOCX navigation caches; rebuild from the updated pagination map")
    return len(observed)


def main():
    document = fitz.open(PDF)
    fonts = check_fonts(document)
    entries = json.loads((HERE / "gwu-navigation-20261001.json").read_text())
    page_text = [normalize(page.get_text(clip=fitz.Rect(85, 65, 528, 725))) for page in document]
    numbering = []
    for page in document:
        footer = page.get_text(clip=fitz.Rect(270, 729, 342, 766)).strip()
        numbering.append(footer)
    page_map, locations = {}, []
    for entry in entries:
        matches = [index for index, text in enumerate(page_text) if normalize(entry["text"]) in text]
        if not matches:
            raise ValueError(f'Navigation target missing: {entry["text"]}')
        index = matches[0] if entry["kind"] == "front" and entry["bookmark"] in {"dedication", "acknowledgements", "abstract"} else matches[-1]
        page_map[entry["bookmark"]] = numbering[index]
        locations.append({**entry, "pdf_page": index + 1, "printed_page": numbering[index]})
    if page_map["heading_1"] != "1":
        raise ValueError("Body numbering does not restart at one")
    if numbering[0]:
        raise ValueError("Title page number must be suppressed")
    body_start = next(item["pdf_page"] - 1 for item in locations if item["bookmark"] == "heading_1")
    if numbering[body_start:] != [str(index) for index in range(1, len(document) - body_start + 1)]:
        raise ValueError("Body page numbers are discontinuous")
    outside = []
    empty = []
    for index, page in enumerate(document):
        if not page_text[index]:
            empty.append(index + 1)
        if page.rect.width != 612 or page.rect.height != 792:
            raise ValueError("Non-Letter page")
        for word in page.get_text("words"):
            if word[0] < 89 or word[2] > 523 or word[1] < 70 or (word[3] > 725 and word[1] < 730):
                outside.append({"page": index + 1, "word": word[:5]})
    if outside or empty:
        raise ValueError({"out_of_bounds": outside, "empty_pages": empty})
    (HERE / "gwu-pagination-map-20261001.json").write_text(json.dumps(page_map, indent=2) + "\n")
    cached_targets = check_cached_navigation(Document(PDF.with_suffix(".docx")), page_map)
    (REVIEW / "navigation-targets.json").write_text(json.dumps(locations, indent=2) + "\n")
    for start in range(0, len(document), 9):
        sheet = Image.new("RGB", (1104, 1488), "#dddddd")
        drawing = ImageDraw.Draw(sheet)
        for offset, index in enumerate(range(start, min(start + 9, len(document)))):
            pixmap = document[index].get_pixmap(matrix=fitz.Matrix(.56, .56), alpha=False)
            picture = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            left, top = (offset % 3) * 368 + 12, (offset // 3) * 496 + 30
            sheet.paste(picture, (left, top))
            drawing.text((left, top - 20), f"PDF {index + 1} / printed {numbering[index] or 'title'}", fill="black")
        sheet.save(REVIEW / f"contact-{start // 9 + 1:02}.png")
    report = {
        "status": "mechanical_checks_passed",
        "pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
        "pages": len(document), "front_pages": body_start,
        "body_pages_including_references": len(document) - body_start,
        "navigation_targets": len(entries),
        "verified_cached_navigation_targets": cached_targets,
        "embedded_fonts": fonts,
        "table_targets": sum(entry["kind"] == "table" for entry in entries),
        "out_of_bounds_words": outside, "empty_pages": empty,
        "visual_review": "Contact sheets generated; visual inspection is a separate check.",
    }
    (REVIEW / "render-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
