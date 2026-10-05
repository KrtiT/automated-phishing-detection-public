"""Render the explanatory figure from the unchanged study dataflow."""

from pathlib import Path

import fitz
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

HERE = Path(__file__).resolve().parent
PDF = HERE / "gwu-system-dataflow-20261001.pdf"


def main():
    drawing = canvas.Canvas(str(PDF), pagesize=(432, 462))
    drawing.setTitle("Frozen URL inference and evaluation dataflow")

    def box(left, bottom, width, height, title, lines):
        drawing.setStrokeColor(HexColor("#374b60"))
        drawing.setFillColor(HexColor("#f0f4f7"))
        drawing.roundRect(left, bottom, width, height, 5, fill=1)
        drawing.setFillColor(HexColor("#172a3b"))
        drawing.setFont("Helvetica-Bold", 10)
        drawing.drawCentredString(left + width / 2, bottom + height - 17, title)
        drawing.setFont("Helvetica", 9)
        for index, line in enumerate(lines):
            drawing.drawCentredString(left + width / 2, bottom + height - 33 - index * 13, line)

    def arrow(points, dashed=False):
        drawing.setStrokeColor(HexColor("#374b60"))
        drawing.setLineWidth(1)
        drawing.setDash(3, 2) if dashed else drawing.setDash()
        path = drawing.beginPath()
        path.moveTo(*points[0])
        for point in points[1:]:
            path.lineTo(*point)
        drawing.drawPath(path)
        drawing.setDash()
        previous, last = points[-2:]
        horizontal = abs(last[0] - previous[0]) > abs(last[1] - previous[1])
        if horizontal:
            direction = 1 if last[0] > previous[0] else -1
            drawing.line(last[0], last[1], last[0] - 5 * direction, last[1] + 3)
            drawing.line(last[0], last[1], last[0] - 5 * direction, last[1] - 3)
        else:
            direction = 1 if last[1] > previous[1] else -1
            drawing.line(last[0], last[1], last[0] - 3, last[1] - 5 * direction)
            drawing.line(last[0], last[1], last[0] + 3, last[1] - 5 * direction)

    box(51, 400, 330, 55, "1. Permitted URL input", ["Preserved source identity; declared parsing and exclusions", "External input: publisher url_norm under disclosed amendment"])
    box(51, 327, 330, 55, "2. Structural inference", ["25 structural features + fixed Logistic-L1 model", "Validation-selected threshold; no test-time refit"])
    arrow([(216, 400), (216, 382)])
    box(8, 209, 194, 93, "3a. Fixed cascade", ["Outside fixed band: Logistic-L1 decision", "Inside fixed band: character transformer", "Separate fixed transformer threshold", "Allow/alert output; no automatic blocking"])
    box(230, 236, 194, 66, "3b. GMM monitor", ["25 features + first-stage score", "256-row windows; stride 64", "Strict frozen alert boundary"])
    box(230, 147, 194, 66, "4. Future-only policy comparator", ["An alert ending at request t", "escalates t+1 through t+256", "Never revises the triggering window"])
    arrow([(216, 327), (216, 316), (105, 316), (105, 302)])
    arrow([(216, 316), (327, 316), (327, 302)])
    arrow([(327, 236), (327, 213)])
    arrow([(230, 180), (215, 180), (215, 239), (202, 239)], dashed=True)
    drawing.setFont("Helvetica", 8)
    drawing.setFillColor(HexColor("#172a3b"))
    drawing.drawString(9, 185, "Dashed path: override for future requests only.")
    box(8, 19, 194, 101, "5a. Detection and policy evaluation", ["Save paired scores, decisions and routes", "Then join source/tier outcome strata", "Domain-clustered recall contrasts", "Low-FPR gates and secondary metrics", "Outcome labels do not drive routing"])
    box(230, 19, 194, 101, "5b. Separate real-HTTP measurement", ["125 cells / 25 five-repeat groups", "Actual physical transformer attempts", "Terminal latencies and request errors", "Client-phase throughput and drain", "No new model or threshold selection"])
    arrow([(104, 209), (104, 195), (3, 195), (3, 134), (327, 134), (327, 120)])
    arrow([(327, 147), (327, 134), (105, 134), (105, 120)])
    drawing.save()
    with fitz.open(PDF) as document:
        document[0].get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False).save(HERE / "gwu-system-dataflow-20261001.png")
    print(PDF)


if __name__ == "__main__":
    main()
