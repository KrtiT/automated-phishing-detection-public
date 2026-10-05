"""Build the dated v3 manuscript while preserving the June DOCX shell."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import tempfile
import zipfile
from copy import deepcopy
from pathlib import Path

from lxml import etree

SOURCE_SHA256 = "96c056e6becf2bab7335adf4ed850707ca049749233b819fde7fb80a060daf3d"
FRONT_MATTER_SHA256 = "38cae1fa8dbbff6b091409f0c07082f37a666e2596903f85d84ab14f3cbf7147"
CHAPTER_BOUNDARY = 214
DOCUMENT_XML = "word/document.xml"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{W_NS}}}"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _paths_alias(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    try:
        return left.samefile(right)
    except FileNotFoundError:
        return False


def _canonical_hash(elements: list[etree._Element]) -> str:
    data = b"".join(
        etree.tostring(element, method="c14n", exclusive=False, with_comments=False)
        for element in elements
    )
    return _sha256(data)


def _clean_inline(text: str) -> str:
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
    return text.replace("**", "").replace("`", "").strip()


def _run(text: str, *, bold: bool = False, size: int | None = None) -> etree._Element:
    run = etree.Element(f"{W}r")
    properties = etree.SubElement(run, f"{W}rPr")
    fonts = etree.SubElement(properties, f"{W}rFonts")
    for name in ("ascii", "hAnsi", "cs"):
        fonts.set(f"{W}{name}", "Times New Roman")
    if bold:
        etree.SubElement(properties, f"{W}b")
        etree.SubElement(properties, f"{W}bCs")
    if size is not None:
        etree.SubElement(properties, f"{W}sz").set(f"{W}val", str(size))
        etree.SubElement(properties, f"{W}szCs").set(f"{W}val", str(size))
    node = etree.SubElement(run, f"{W}t")
    node.text = text
    return run


def _paragraph(
    text: str,
    *,
    style: str | None = None,
    heading_level: int | None = None,
    bullet: bool = False,
    reference: bool = False,
    page_break_before: bool = False,
) -> etree._Element:
    paragraph = etree.Element(f"{W}p")
    properties = etree.SubElement(paragraph, f"{W}pPr")
    if style:
        etree.SubElement(properties, f"{W}pStyle").set(f"{W}val", style)

    spacing = etree.SubElement(properties, f"{W}spacing")
    spacing.set(f"{W}line", "480")
    spacing.set(f"{W}lineRule", "auto")
    if heading_level:
        spacing.set(f"{W}before", "160" if heading_level > 1 else "360")
        spacing.set(f"{W}after", "80")
        etree.SubElement(properties, f"{W}keepNext")
        etree.SubElement(properties, f"{W}keepLines")
        if heading_level == 1:
            etree.SubElement(properties, f"{W}pageBreakBefore")
            etree.SubElement(properties, f"{W}jc").set(f"{W}val", "center")
        elif page_break_before:
            etree.SubElement(properties, f"{W}pageBreakBefore")
    elif bullet:
        indentation = etree.SubElement(properties, f"{W}ind")
        indentation.set(f"{W}left", "720")
        indentation.set(f"{W}hanging", "360")
    elif reference:
        indentation = etree.SubElement(properties, f"{W}ind")
        indentation.set(f"{W}left", "720")
        indentation.set(f"{W}hanging", "720")
        etree.SubElement(properties, f"{W}keepLines")
    else:
        etree.SubElement(properties, f"{W}ind").set(f"{W}firstLine", "720")

    display = f"• {text}" if bullet else text
    paragraph.append(
        _run(display, bold=bool(heading_level), size=24 if heading_level else None)
    )
    return paragraph


def _table(rows: list[list[str]]) -> etree._Element:
    table = etree.Element(f"{W}tbl")
    properties = etree.SubElement(table, f"{W}tblPr")
    width = etree.SubElement(properties, f"{W}tblW")
    width.set(f"{W}w", "0")
    width.set(f"{W}type", "auto")
    borders = etree.SubElement(properties, f"{W}tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = etree.SubElement(borders, f"{W}{edge}")
        border.set(f"{W}val", "single")
        border.set(f"{W}sz", "4")
        border.set(f"{W}space", "0")
        border.set(f"{W}color", "808080")

    for row_index, values in enumerate(rows):
        row = etree.SubElement(table, f"{W}tr")
        tr_properties = etree.SubElement(row, f"{W}trPr")
        etree.SubElement(tr_properties, f"{W}cantSplit")
        if row_index == 0:
            etree.SubElement(tr_properties, f"{W}tblHeader")
        for value in values:
            cell = etree.SubElement(row, f"{W}tc")
            cell_properties = etree.SubElement(cell, f"{W}tcPr")
            margin = etree.SubElement(cell_properties, f"{W}tcMar")
            for edge in ("top", "left", "bottom", "right"):
                item = etree.SubElement(margin, f"{W}{edge}")
                item.set(f"{W}w", "80")
                item.set(f"{W}type", "dxa")
            if row_index == 0:
                shading = etree.SubElement(cell_properties, f"{W}shd")
                shading.set(f"{W}val", "clear")
                shading.set(f"{W}fill", "D9EAF2")
            paragraph = etree.SubElement(cell, f"{W}p")
            p_properties = etree.SubElement(paragraph, f"{W}pPr")
            if row_index == 0 or (len(rows) <= 8 and row_index < len(rows) - 1):
                etree.SubElement(p_properties, f"{W}keepNext")
            spacing = etree.SubElement(p_properties, f"{W}spacing")
            spacing.set(f"{W}after", "60")
            spacing.set(f"{W}line", "240")
            spacing.set(f"{W}lineRule", "auto")
            paragraph.append(_run(value, bold=row_index == 0, size=20))
    return table


def _parse_table(lines: list[str]) -> list[list[str]]:
    parsed = [
        [_clean_inline(cell) for cell in line.strip().strip("|").split("|")]
        for line in lines
    ]
    if len(parsed) > 1 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in parsed[1]):
        del parsed[1]
    width = len(parsed[0])
    if any(len(row) != width for row in parsed):
        raise ValueError("Markdown table has inconsistent column count")
    return parsed


def _body_elements(markdown: str) -> list[etree._Element]:
    lines = markdown.splitlines()
    elements: list[etree._Element] = []
    paragraph_lines: list[str] = []
    in_references = False

    def flush_paragraph() -> None:
        if paragraph_lines:
            text = _clean_inline(" ".join(line.strip() for line in paragraph_lines))
            elements.append(_paragraph(text, reference=in_references))
            paragraph_lines.clear()

    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            flush_paragraph()
            index += 1
            continue
        if line.startswith("# "):
            flush_paragraph()
            text = _clean_inline(line[2:])
            elements.append(_paragraph(text, style="Heading1", heading_level=1))
            in_references = False
            index += 1
            continue
        if line.startswith("## "):
            flush_paragraph()
            text = _clean_inline(line[3:])
            elements.append(
                _paragraph(
                    text,
                    style="Heading2",
                    heading_level=2,
                )
            )
            in_references = text == "References"
            index += 1
            continue
        if line.startswith("### "):
            flush_paragraph()
            text = _clean_inline(line[4:])
            elements.append(_paragraph(text, style="Heading3", heading_level=3))
            index += 1
            continue
        if line.startswith("| "):
            flush_paragraph()
            table_lines: list[str] = []
            while index < len(lines) and lines[index].rstrip().startswith("|"):
                table_lines.append(lines[index].rstrip())
                index += 1
            elements.append(_table(_parse_table(table_lines)))
            continue
        if line.startswith("- "):
            flush_paragraph()
            elements.append(
                _paragraph(_clean_inline(line[2:]), style="ListParagraph", bullet=True)
            )
            index += 1
            continue
        paragraph_lines.append(line)
        index += 1
    flush_paragraph()
    return elements


def _validate_source(
    source_bytes: bytes, root: etree._Element
) -> tuple[etree._Element, list[etree._Element]]:
    if _sha256(source_bytes) != SOURCE_SHA256:
        raise ValueError("source DOCX SHA-256 does not match the immutable June shell")
    body = root.find(f".//{W}body")
    if body is None:
        raise ValueError("source word/document.xml has no body")
    children = list(body)
    boundaries = [
        index
        for index, element in enumerate(children)
        if "".join(element.itertext()).strip() == "Chapter 1—Introduction"
    ]
    if boundaries != [CHAPTER_BOUNDARY]:
        raise ValueError(f"unexpected Chapter 1 boundary: {boundaries}")
    if _canonical_hash(children[:CHAPTER_BOUNDARY]) != FRONT_MATTER_SHA256:
        raise ValueError("protected front-matter hash does not match")
    if children[-1].tag != f"{W}sectPr":
        raise ValueError("source body does not end in section properties")
    return body, children


def _validate_output(source_parts: dict[str, bytes], output_path: Path) -> None:
    with zipfile.ZipFile(output_path) as package:
        if set(package.namelist()) != set(source_parts):
            raise ValueError("output DOCX package entry set changed")
        for name, source_data in source_parts.items():
            if name != DOCUMENT_XML and package.read(name) != source_data:
                raise ValueError(f"output changed protected package part: {name}")
        root = etree.fromstring(package.read(DOCUMENT_XML))
        body = root.find(f".//{W}body")
        if body is None:
            raise ValueError("output word/document.xml has no body")
        children = list(body)
        if _canonical_hash(children[:CHAPTER_BOUNDARY]) != FRONT_MATTER_SHA256:
            raise ValueError("output changed protected front matter")
        page_numbering = children[-1].find(f"./{W}pgNumType")
        if page_numbering is None or page_numbering.get(f"{W}start") != "1":
            raise ValueError("output body page numbering does not restart at 1")
        forbidden = {
            f"{W}ins",
            f"{W}del",
            f"{W}moveFrom",
            f"{W}moveTo",
            f"{W}commentRangeStart",
            f"{W}commentRangeEnd",
            f"{W}commentReference",
        }
        if forbidden & {element.tag for element in root.iter()}:
            raise ValueError("output contains tracked-revision or comment markup")


def build(source_path: Path, body_source_path: Path, output_path: Path) -> None:
    source_path = Path(source_path)
    body_source_path = Path(body_source_path)
    output_path = Path(output_path)
    if any(
        _paths_alias(output_path, input_path)
        for input_path in (source_path, body_source_path)
    ):
        raise ValueError("output path must differ from input paths")
    source_bytes = source_path.read_bytes()
    body_markdown = body_source_path.read_text(encoding="utf-8")
    replacement = _body_elements(body_markdown)
    if (
        len(
            [
                node
                for node in replacement
                if node.tag == f"{W}p"
                and "".join(node.itertext()).startswith("Chapter ")
            ]
        )
        != 5
    ):
        raise ValueError("body source must contain exactly five chapter headings")

    with zipfile.ZipFile(source_path) as source_package:
        source_parts = {
            name: source_package.read(name) for name in source_package.namelist()
        }
        root = etree.fromstring(source_parts[DOCUMENT_XML])
        body, children = _validate_source(source_bytes, root)
        section_properties = deepcopy(children[-1])
        page_number_resets = [
            reset
            for node in children[CHAPTER_BOUNDARY:-1]
            for reset in node.findall(f".//{W}pgNumType")
            if reset.get(f"{W}start") == "1"
        ]
        if len(page_number_resets) != 1:
            raise ValueError("source shell must contain one body page-number reset")
        page_margin = section_properties.find(f"./{W}pgMar")
        if page_margin is None:
            raise ValueError("source final section has no page margins")
        section_properties.insert(
            section_properties.index(page_margin) + 1,
            deepcopy(page_number_resets[0]),
        )
        for node in children[CHAPTER_BOUNDARY:]:
            body.remove(node)
        for node in replacement:
            body.append(node)
        body.append(section_properties)
        document_xml = etree.tostring(
            root, xml_declaration=True, encoding="UTF-8", standalone=True
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=output_path.parent,
            prefix=f".{output_path.stem}.",
            suffix=".tmp",
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            with zipfile.ZipFile(temporary_path, "w") as output_package:
                for info in source_package.infolist():
                    data = (
                        document_xml
                        if info.filename == DOCUMENT_XML
                        else source_parts[info.filename]
                    )
                    output_package.writestr(info, data)
            _validate_output(source_parts, temporary_path)
            os.replace(temporary_path, output_path)
            os.chmod(output_path, source_path.stat().st_mode & 0o777)
        finally:
            temporary_path.unlink(missing_ok=True)

    if _sha256(source_path.read_bytes()) != SOURCE_SHA256:
        raise RuntimeError("source DOCX changed during build")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=here.parent
        / "attachments"
        / "foUeB5"
        / "Tallam_Krti_Praxis_2026-06-15.docx",
    )
    parser.add_argument(
        "--body",
        type=Path,
        default=here / "Tallam_Krti_Praxis_v3_Body_2026-09-17.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "Tallam_Krti_Praxis_v3_Working_2026-09-17.docx",
    )
    arguments = parser.parse_args()
    build(arguments.source, arguments.body, arguments.output)
    print(arguments.output)


if __name__ == "__main__":
    main()
