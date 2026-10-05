"""Make a bounded author-review revision without touching sealed research."""

import hashlib
import json
import re
import shutil
import sys
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph


HERE = Path(__file__).resolve().parent
CONTEXT = HERE.parent.parent
BASE = CONTEXT / "deliverables/Tallam_Dissertation_Layout_Reviewed_2026-10-02"
AUTHOR = CONTEXT / "deliverables/Tallam_Dissertation_Author_Review_2026-10-04"
OPEN_COPY = AUTHOR / "Tallam_Krti_Praxis_Author_Review_2026-10-04.docx"
ROOT = AUTHOR / "Consistency_Reviewed_2026-10-04"
MANUSCRIPT = ROOT / "manuscript"
SOURCE = BASE / "manuscript/Tallam_Krti_Praxis_Layout_Reviewed_2026-10-02.docx"
OUTPUT = MANUSCRIPT / "Tallam_Krti_Praxis_Consistency_Reviewed_2026-10-04.docx"
RFC = (
    "Berners-Lee, T., Fielding, R., & Masinter, L. (2005). Uniform Resource Identifier (URI): "
    "Generic syntax (RFC 3986). RFC Editor. https://doi.org/10.17487/RFC3986"
)
BODY_EDITS = [
    ("reputation, email context, or the URL itself.",
     "reputation, email context, or the URL itself (Basit et al., 2021; Khonji et al., 2013).",
     "Add point-of-use survey attribution in Section 2.1."),
    ("Request for Comments (RFC) 3986 reserved characters are preserved.",
     "Request for Comments (RFC) 3986 reserved characters are preserved (Berners-Lee et al., 2005).",
     "Cite the primary URI syntax standard in Section 3.6."),
    ("BMC Genomics, 21, Article 6.", "BMC Genomics, 21(1), Article 6.",
     "Complete the issue field verified in Crossref publisher metadata."),
]
FRONT_EDITS = [
    ("Praxis directed by\n\nMazen Mheish\nProfessorial Lecturer in Engineering and Applied Science",
     "Praxis directed by\n\nAmir Etemadi\nAssociate Professor of Engineering and Applied Science\n\n"
     "With prior direction by\nMazen Mheish\nProfessorial Lecturer in Engineering and Applied Science",
     "Credit both advisors; distinguish the current appointment from prior direction."),
    ("Mazen Mheish, Professorial Lecturer in Engineering and Applied Science, Praxis Director",
     "Amir Etemadi, Associate Professor of Engineering and Applied Science, Current Praxis Advisor",
     "Reflect the July 27, 2026 GWU advisor appointment without assigning a new chair."),
    ("Amir Etemadi, Associate Professor of Engineering and Applied Science, Praxis Chair",
     "Mazen Mheish, Professorial Lecturer in Engineering and Applied Science, Prior Praxis Advisor",
     "Retain Mheish's direction credit without asserting a joint formal appointment."),
    ("Committee information is carried forward from the author-supplied manuscript. University certification "
     "of the final examination and institutional approval is not asserted in this copy.",
     "The advisor credits reflect the July 2026 appointment of Dr. Etemadi and Dr. Mheish's prior direction. "
     "Vijay Raghavan's listing is retained from the author-supplied manuscript. The final examination "
     "committee and its formal roles require University confirmation; this author-review copy does not "
     "assert that confirmation or institutional approval.",
     "Make the remaining roster uncertainty explicit rather than certify outdated roles."),
    ("I would also like to thank Professor Mazen Mheish for the amount of time that he put into helping me "
     "complete my work, as well as for the time he spent to thoroughly review my work and provide accurate "
     "and detailed feedback. Not only did he assist me in formulating a clearer argument and presenting my "
     "work in a more complete fashion, he also worked with me to make sure that I completed the D.Eng. "
     "document in the proper format for a D.Eng. document, as required by George Washington University "
     "(GWU). Dr. Amir Etemadi and Dr. Timothy Blackburn: Your guidance during the D.Eng. process helped me "
     "a lot to stay on track during my research and to complete it successfully.",
     "I thank Professor Mazen Mheish for the time he devoted to reviewing my work, his detailed feedback, "
     "and his help in clarifying the argument and preparing the D.Eng. manuscript. I also thank Dr. Amir "
     "Etemadi for his guidance as my current advisor during the revised Praxis. I am grateful to Dr. "
     "Timothy Blackburn for his guidance throughout the D.Eng. process.",
     "Thank both advisors explicitly while retaining Blackburn's acknowledgment; AI-assisted editing."),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_span(paragraph, old, new):
    if paragraph.text.count(old) != 1:
        raise ValueError("Nonunique paragraph span: " + old)
    start = paragraph.text.index(old)
    end = start + len(old)
    offset = 0
    inserted = False
    for run in paragraph.runs:
        original = run.text
        run_end = offset + len(original)
        if offset < end and run_end > start:
            prefix = original[:max(0, start - offset)]
            suffix = original[max(0, end - offset):]
            run.text = prefix + (new if not inserted else "") + suffix
            inserted = True
        offset = run_end
    if not inserted:
        raise ValueError("No editable run for span")


def build():
    if OUTPUT.exists():
        raise FileExistsError("Refusing to overwrite an existing review copy")
    if digest(SOURCE) != "05e3c3d5ca792ee853af5c80e92426130c5e57fdc401ed44eee0d5644eac4fad":
        raise ValueError("Sealed manuscript changed")
    open_hash = digest(OPEN_COPY)
    if open_hash != digest(SOURCE):
        raise ValueError("Author copy contains saved edits; reconcile before building a competing copy")
    MANUSCRIPT.mkdir(parents=True, exist_ok=True)
    document = Document(SOURCE)
    markdown = SOURCE.with_suffix(".md").read_text()
    ledger = []
    for old, new, reason in BODY_EDITS + FRONT_EDITS:
        matches = [paragraph for paragraph in document.paragraphs if old in paragraph.text]
        if len(matches) != 1:
            raise ValueError("Nonunique DOCX target: " + old)
        replace_span(matches[0], old, new)
        if (old, new, reason) in BODY_EDITS:
            if markdown.count(old) != 1:
                raise ValueError("Nonunique Markdown target: " + old)
            markdown = markdown.replace(old, new, 1)
        ledger.append({"old": old, "new": new, "reason": reason})
    anchor = next(paragraph for paragraph in document.paragraphs
                  if paragraph.text.startswith("Breiman, L. (2001)."))
    element = deepcopy(anchor._p)
    anchor._p.addprevious(element)
    paragraph = Paragraph(element, anchor._parent)
    paragraph.clear()
    paragraph.add_run("Berners-Lee, T., Fielding, R., & Masinter, L. (2005). ")
    paragraph.add_run("Uniform Resource Identifier (URI): Generic syntax").italic = True
    paragraph.add_run(" (RFC 3986). RFC Editor. https://doi.org/10.17487/RFC3986")
    assert paragraph.text == RFC
    markdown = markdown.replace("\n\nBreiman, L. (2001).", "\n\n" + RFC + "\n\nBreiman, L. (2001).", 1)
    ledger.append({"old": None, "new": RFC, "reason": "Add the missing primary-standard reference in alphabetical order."})
    document.save(OUTPUT)
    OUTPUT.with_suffix(".md").write_text(markdown)
    for stem in ["gwu-system-dataflow-20261001", "followup-source-partitions",
                 "followup-calibration", "followup-paired-recall"]:
        for suffix in [".png", ".pdf"]:
            shutil.copy2(BASE / "manuscript" / (stem + suffix), MANUSCRIPT / (stem + suffix))
    navigation = HERE.parent / "editorial-revision-20261002/manuscript/gwu-navigation-20261001.json"
    shutil.copy2(navigation, MANUSCRIPT / navigation.name)
    shutil.copytree(BASE / "advisor", ROOT / "advisor")
    (ROOT / "changes.json").write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n")
    (HERE / "preservation-before.json").write_text(json.dumps({
        "open_author_copy": str(OPEN_COPY), "sha256": open_hash,
        "sealed_source": str(SOURCE), "source_sha256": digest(SOURCE),
    }, indent=2) + "\n")
    (ROOT / "Front_Matter_Reviewed.txt").write_text("\n\n".join(
        paragraph.text for paragraph in document.paragraphs[:24]) + "\n")
    print(json.dumps({"output": str(OUTPUT), "logged_changes": len(ledger),
                      "references": 51, "author_copy_preserved": digest(OPEN_COPY) == open_hash}))


def refresh():
    pages = json.loads((MANUSCRIPT / "gwu-pagination-map-20261001.json").read_text())
    document = Document(OUTPUT)
    for instruction in document._element.xpath(".//w:instrText"):
        match = re.search(r"\bPAGEREF\s+(\w+)", instruction.text or "")
        if match:
            instruction.getparent().find(qn("w:t")).text = pages[match.group(1)]
    for hyperlink in document._element.xpath(".//w:hyperlink[@w:anchor]"):
        name = hyperlink.get(qn("w:anchor"))
        if name in pages:
            hyperlink.xpath(".//w:t")[0].text = pages[name]
    document.save(OUTPUT)
    print("Refreshed navigation caches in the new copy only.")


def format_reference():
    document = Document(OUTPUT)
    paragraph = next(paragraph for paragraph in document.paragraphs
                     if paragraph.text.startswith("Chicco, D., & Jurman, G. (2020)."))
    before, after = paragraph.text.split("BMC Genomics, 21", 1)
    paragraph.clear()
    paragraph.add_run(before)
    paragraph.add_run("BMC Genomics, 21").italic = True
    paragraph.add_run(after)
    paragraph = next(paragraph for paragraph in document.paragraphs
                     if paragraph.text.startswith("Hannousse, A., & Yahiouche, S. (2020)."))
    original = paragraph.text
    before, after = original.split(" (2020). ", 1)
    title, locator = after.split(" [Preprint].", 1)
    paragraph.clear()
    paragraph.add_run(before + " (2020). ")
    paragraph.add_run(title).italic = True
    paragraph.add_run(" [Preprint]." + locator)
    require_text = paragraph.text
    if require_text != original:
        raise ValueError("Reference formatting changed its wording")
    document.save(OUTPUT)
    ledger_path = ROOT / "changes.json"
    ledger = json.loads(ledger_path.read_text())
    reason = "Restore missing italic title formatting for the Hannousse and Yahiouche (2020) preprint; wording unchanged."
    if not any(entry["reason"] == reason for entry in ledger):
        ledger.append({"old": original, "new": original, "reason": reason, "format_only": True})
        ledger_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n")
    print("Verified journal/volume emphasis and restored the preprint's italic title.")


if __name__ == "__main__":
    {"build": build, "refresh": refresh, "format": format_reference}[sys.argv[1]]()
