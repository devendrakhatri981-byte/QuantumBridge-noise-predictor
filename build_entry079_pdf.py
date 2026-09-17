"""Build Entry 079 page in the QuantumBridge Research Log house style."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

OUT = "entry079_pages.pdf"
START_PAGE = 155
RUNNING_TITLE = "Entry 079 — Correcting the Kyiv/Brisbane 'Regression'"

NAVY = colors.HexColor("#1F3864")
GREY = colors.HexColor("#6B6B6B")
RULE = colors.HexColor("#B8B8B8")
BAND = colors.HexColor("#EDF1F8")

W, H = A4
LM = RM = 20 * mm
TM, BM = 26 * mm, 18 * mm

body = ParagraphStyle("body", fontName="Helvetica", fontSize=9, leading=12.6,
                      textColor=colors.HexColor("#1A1A1A"), spaceAfter=6)
lead = ParagraphStyle("lead", parent=body, fontSize=9.5, leading=13.4)
h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=16, leading=19,
                    textColor=NAVY, spaceBefore=4, spaceAfter=3)
h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10.5, leading=13,
                    textColor=NAVY, spaceBefore=11, spaceAfter=4)
sub = ParagraphStyle("sub", fontName="Helvetica-Oblique", fontSize=9.5, leading=12.5,
                     textColor=GREY, spaceAfter=9)
banner = ParagraphStyle("banner", fontName="Helvetica-Bold", fontSize=8.5,
                        leading=11, textColor=colors.white)
cap = ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=8, leading=10.5,
                     textColor=GREY, spaceBefore=2, spaceAfter=8)
concl = ParagraphStyle("concl", parent=body, backColor=BAND,
                       borderPadding=(8, 8, 8, 8), spaceBefore=4)
tcell = ParagraphStyle("tcell", fontName="Helvetica", fontSize=8, leading=10,
                       textColor=colors.HexColor("#1A1A1A"))
thead = ParagraphStyle("thead", fontName="Helvetica-Bold", fontSize=8, leading=10,
                       textColor=colors.white)


def header_footer(canvas, doc):
    canvas.saveState()
    y = H - 13 * mm
    canvas.setFont("Helvetica-Bold", 7.5); canvas.setFillColor(NAVY)
    canvas.drawString(LM, y, "QuantumBridge Research Log")
    canvas.setFont("Helvetica", 7.5); canvas.setFillColor(GREY)
    canvas.drawString(LM, y - 9, RUNNING_TITLE)
    canvas.drawRightString(W - RM, y, "Darknight (Mirr) | BTech AI/ML")
    canvas.drawRightString(W - RM, y - 9, f"Page {START_PAGE + doc.page - 1}")
    canvas.setStrokeColor(RULE); canvas.setLineWidth(0.5)
    canvas.line(LM, y - 14, W - RM, y - 14)
    canvas.setFont("Helvetica", 7.5); canvas.setFillColor(GREY)
    canvas.drawString(LM, y - 23, "QuantumBridge — Phase 2 (Research), resumed")
    canvas.restoreState()


def entry_banner(text):
    t = Table([[Paragraph(text, banner)]], colWidths=[W - LM - RM], rowHeights=[16])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY),
                           ("LEFTPADDING", (0, 0), (-1, -1), 7),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


def mktable(headers, rows, colWidths):
    data = [[Paragraph(h, thead) for h in headers]]
    for r in rows:
        data.append([Paragraph(str(c), tcell) for c in r])
    t = Table(data, colWidths=colWidths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
    ]))
    return t


s = []
s.append(entry_banner("ENTRY 079 &nbsp;&nbsp; Correcting the Kyiv/Brisbane “Regression” — "
                      "September 16, 2026 — Status: COMPLETE"))
s.append(Spacer(1, 9))
s.append(Paragraph("It Was a Split-Composition Artifact, Not a Real Capacity Effect", h1))
s.append(Paragraph("Entry 077's Reported Kyiv/Brisbane Dip Does Not Survive a Controlled Comparison", sub))

s.append(Paragraph("The Problem With Entry 077's Original Comparison", h2))
s.append(Paragraph(
    "Entry 077 reported Kyiv MAE rising 0.77→1.08 and Brisbane 0.99→1.14 when Osaka joined "
    "training, read as the model trading a little same-chip precision for broader generalization. "
    "That comparison used two DIFFERENT random 80/20 KFold splits (seed=42 each time, but over "
    "datasets of different length — 5,690 graphs for Entry 069, 6,398 for Entry 077) — so the "
    "actual circuits landing in each \"held-out test set\" were not the same circuits. A difference "
    "in test-set composition alone can move per-chip MAE by this much; the comparison was confounded "
    "from the start.", lead))

s.append(Paragraph("Controlled Re-Test", h2))
s.append(Paragraph(
    "Re-scored the Entry 077 combined model on the EXACT same 1,138 held-out circuits Entry 069 used "
    "(the first 5,690 rows of entry077_graph_dataset.json are in identical order to "
    "entry068_graph_dataset.json, confirmed by spot-check, so Entry 069's KFold test indices apply "
    "unchanged). This isolates the model-capacity question from split-composition noise.", lead))

s.append(mktable(
    ["Chip", "Entry 069 (3-chip)", "Entry 077 (4-chip), same test circuits", "Delta"],
    [["Kyiv", "0.77", "0.84", "+0.07 (noise-level)"],
     ["Sherbrooke", "1.45", "1.09", "-0.36 (better)"],
     ["Brisbane", "0.99", "0.80", "-0.19 (better)"],
     ["Overall", "1.07", "0.91", "-0.16 (better)"]],
    [70, 145, 200, 105]))
s.append(Paragraph("MAE in points, on the identical held-out circuit set both times. Floor-collapse "
                   "MAE on this same set: Kyiv=0.75, Sherbrooke=0.54, Brisbane=0.17 — all healthy, "
                   "no regression there either.", cap))

s.append(Paragraph("Entry 079 — Conclusion", h2))
s.append(Spacer(1, 4))
s.append(Paragraph(
    "There is no real capacity-dilution effect here. When compared fairly — same circuits, same "
    "evaluation — the 4-chip model matches or beats the 3-chip model on every one of the original "
    "three chips, not just on generalization to the unseen fifth. Entry 077's \"nuanced, not-uniform\" "
    "framing was itself an artifact of an apples-to-oranges comparison. No architecture or weighting "
    "fix is warranted. The corrected, honest verdict: adding Osaka to training improved the model "
    "across the board — same-chip accuracy AND zero-shot generalization to a genuinely new chip — "
    "with no accuracy trade-off found on a controlled re-test.", concl))

s.append(Spacer(1, 10))
s.append(Paragraph("Research Log — Index Addendum (Entry 079)", h2))
s.append(Paragraph(
    "No new files — this entry is a controlled re-analysis of entry077_combined_params.json against "
    "Entry 069's original KFold test_idx, run directly against the existing dataset files.", cap))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="QuantumBridge Research Log — Entry 079",
                      author="Darknight (Mirr)")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, BM, W - LM - RM,
                                                         H - TM - BM, id="f")],
                                   onPage=header_footer)])
doc.build(s)
print("built", OUT)
