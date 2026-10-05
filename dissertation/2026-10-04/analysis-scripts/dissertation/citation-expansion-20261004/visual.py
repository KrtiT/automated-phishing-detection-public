"""Prepare targeted visual checks and compare unchanged scientific pages."""

import hashlib
import json
import re

import fitz
from PIL import Image, ImageDraw

import expand


def compact(text):
    return re.sub(r"\s+", "", text)


def main():
    destination = expand.ROOT / "visual-review"
    destination.mkdir(exist_ok=True)
    original = fitz.open(expand.SOURCE.with_suffix(".pdf"))
    current = fitz.open(expand.OUTPUT.with_suffix(".pdf"))
    original_pages = {}
    for index, page in enumerate(original):
        original_pages.setdefault(compact(page.get_text(clip=fitz.Rect(85, 65, 528, 725))), []).append(index)
    mapping = []
    for index, page in enumerate(current):
        content = compact(page.get_text(clip=fitz.Rect(85, 65, 528, 725)))
        candidates = original_pages.get(content, [])
        matched = None
        if candidates:
            pixels = page.get_pixmap(matrix=fitz.Matrix(1, 1), clip=fitz.Rect(0, 0, 612, 729), alpha=False).samples
            for candidate in candidates:
                prior_pixels = original[candidate].get_pixmap(matrix=fitz.Matrix(1, 1), clip=fitz.Rect(0, 0, 612, 729), alpha=False).samples
                if hashlib.sha256(pixels).digest() == hashlib.sha256(prior_pixels).digest():
                    matched = candidate + 1
                    break
        mapping.append({"current_page": index + 1, "identical_body_to_prior_page": matched})
    needles = ["Axelsson's (2000)", "Le Pochat et al. (2019)", "Prevalence projections require",
               "conditional membership weights", "Dean and Barroso (2013)", "The workload generator also",
               "Optimization uses AdamW", "BIC; Schwarz, 1978", "closed workload in the sense",
               "Le Pochat et al., 2019", "Axelsson, S.", "Dean, J.", "Dempster, A. P.",
               "Le Pochat, V.", "Lipton, Z.", "Loshchilov, I.", "Schroeder, B."]
    targeted = sorted({index for index, page in enumerate(current)
                       if any(compact(needle) in compact(page.get_text()) for needle in needles)})
    figures = [index for index, page in enumerate(current) if page.get_image_info()]
    for index in figures:
        if mapping[index]["identical_body_to_prior_page"] is None:
            targeted.append(index)
    targeted = sorted(set(targeted))
    sheets = []
    for start in range(0, len(targeted), 4):
        selected = targeted[start:start + 4]
        sheet = Image.new("RGB", (1248, 1664), "#dddddd")
        drawing = ImageDraw.Draw(sheet)
        for offset, index in enumerate(selected):
            pixmap = current[index].get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
            picture = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            left, top = (offset % 2) * 624 + 6, (offset // 2) * 832 + 28
            sheet.paste(picture, (left, top))
            drawing.text((left, top - 20), f"PDF page {index + 1}", fill="black")
        path = destination / f"citations-{start // 4 + 1:02}.png"
        sheet.save(path)
        sheets.append(str(path))
    report = {"pdf_pages": len(current), "prior_pdf_pages": len(original),
              "identical_body_pages_excluding_footer": sum(item["identical_body_to_prior_page"] is not None for item in mapping),
              "figure_pages": [{"current_page": index + 1, "prior_identical_body_page": mapping[index]["identical_body_to_prior_page"]} for index in figures],
              "targeted_pages": [index + 1 for index in targeted], "contact_sheets": sheets, "page_mapping": mapping}
    (destination / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "page_mapping"}, indent=2))


if __name__ == "__main__":
    main()
