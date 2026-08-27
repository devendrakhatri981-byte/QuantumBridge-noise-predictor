"""Build Entry 048 pages in the QuantumBridge Research Log house style."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

OUT = "entry048_pages.pdf"
START_PAGE = 119
RUNNING_TITLE = "Entry 048 — Cross-Chip Generalization Fixed"

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
cell = ParagraphStyle("cell", fontName="Helvetica", fontSize=8.2, leading=10.5)
cellb = ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=8.2, leading=10.5)
cellh = ParagraphStyle("cellh", fontName="Helvetica-Bold", fontSize=8.2, leading=10.5,
                       textColor=colors.white)
concl = ParagraphStyle("concl", parent=body, backColor=BAND,
                       borderPadding=(8, 8, 8, 8), spaceBefore=4)


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
    canvas.drawString(LM, y - 23, "QuantumBridge — Phase 2 Research (ML)")
    canvas.restoreState()


def entry_banner(text):
    t = Table([[Paragraph(text, banner)]], colWidths=[W - LM - RM], rowHeights=[16])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY),
                           ("LEFTPADDING", (0, 0), (-1, -1), 7),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


def table(rows, widths, highlight=(), align_right=()):
    data = [[Paragraph(c, cellh) for c in rows[0]]]
    for r in rows[1:]:
        data.append([Paragraph(c, cellb if i in highlight else cell)
                     for i, c in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    st = [("BACKGROUND", (0, 0), (-1, 0), NAVY),
          ("GRID", (0, 0), (-1, -1), 0.4, RULE),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
          ("LEFTPADDING", (0, 0), (-1, -1), 5),
          ("ROWBACKGROUNDS", (0, 1), (-1, -1),
           [colors.white, colors.HexColor("#F5F7FB")])]
    for c in align_right:
        st.append(("ALIGN", (c, 1), (c, -1), "RIGHT"))
    t.setStyle(TableStyle(st))
    return t


s = []
s.append(entry_banner("ENTRY 048 &nbsp;&nbsp; Cross-Chip Generalization Fixed — "
                      "August 20, 2026 — Status: COMPLETE"))
s.append(Spacer(1, 9))
s.append(Paragraph("Entry 048 — Two Bugs, Not One Deep Flaw: Cross-Chip R&sup2; "
                   "Recovers From 0.04-0.37 to 0.75-0.87", h1))
s.append(Paragraph("Removing a Chip-Identity Input the Network Never Saw at Test Time, "
                   "Plus Per-Chip Feature Normalization, Fixes Most of Entry 047's Collapse", sub))

s.append(Paragraph("1. Diagnosis", h2))
s.append(Paragraph(
    "Two concrete, fixable causes were identified for Entry 047's cross-chip collapse, "
    "not a fundamental failure of the approach. First: the global feature vector included "
    "a chip_kyiv/chip_sherbrooke one-hot flag. When trained on Kyiv only, that flag is "
    "CONSTANT ([1,0]) across every training example -- at test time on Sherbrooke it flips "
    "to a value ([0,1]) the network never saw during training, a distribution shift baked "
    "directly into the input. Second: node/edge calibration features (T1, T2, readout, gate "
    "error) were normalized using mean/std computed across both chips combined, so a model "
    "trained only on one chip's normalized range saw out-of-range values for the other chip "
    "at test time, before any physics was even involved.", lead))

s.append(Paragraph("2. Fix and Result", h2))
s.append(table(
    [["train &rarr; test", "MAE (Entry 047)", "MAE (fixed)", "R&sup2; (047)", "R&sup2; (fixed)"],
     ["Kyiv &rarr; Sherbrooke (cold)", "12.87 pts", "4.40 pts", "0.037", "0.747"],
     ["Sherbrooke &rarr; Kyiv (cold)", "10.31 pts", "3.20 pts", "0.373", "0.866"]],
    [150, 100, 90, 90, 90], highlight=(0,), align_right=(1, 2, 3, 4)))
s.append(Paragraph(
    "Dropping the chip one-hot and switching to per-chip-relative normalization (each "
    "chip's calibration features normalized against that chip's own distribution, known in "
    "advance at deployment time without needing labels) recovered most of the lost accuracy "
    "in both directions -- MAE fell roughly 3x, R&sup2; rose from near-zero/weak to 0.75-0.87. "
    "This is still meaningfully worse than the same-chip CV baseline (R&sup2;=0.971, "
    "MAE=1.21), so genuine cross-chip generalization is not yet fully solved -- but the gap "
    "shrank from catastrophic to a real, workable starting point. One caveat: floor-collapse "
    "MAE on the Kyiv-to-Sherbrooke direction got WORSE (8.20 to 12.57 points) even as overall "
    "MAE improved, suggesting the fix helped general calibration but may have traded away "
    "some of whatever weak floor-collapse signal Entry 047's version retained -- worth "
    "watching, not yet understood.", body))

s.append(Paragraph("Entry 048 — Conclusion", h2))
s.append(Spacer(1, 4))
s.append(Paragraph(
    "Entry 047's cross-chip collapse was diagnosable and largely fixable with two targeted "
    "changes, not a sign the whole approach is unsound -- an important distinction for how "
    "this project should be described going forward. The honest current claim is: the model "
    "generalizes reasonably (R&sup2; 0.75-0.87) to an unseen chip of the same hardware family, "
    "up from essentially not at all, but still falls short of its same-chip accuracy and has "
    "an unexplained floor-collapse regression in one direction. This is real, evidence-backed "
    "progress directly responding to the AI reviews' central critique -- not yet a closed "
    "case.", concl))

s.append(Paragraph("3. Next Step", h2))
s.append(Paragraph(
    "Investigate the floor-collapse regression on the Kyiv-to-Sherbrooke direction "
    "specifically before declaring victory. Consider joint training across both chips with "
    "a proper held-out split (rather than pure leave-one-chip-out) as a more realistic "
    "deployment scenario. Real-hardware validation remains the next major open item, "
    "independent of this fix.", body))

s.append(Spacer(1, 10))
s.append(Paragraph("Research Log — Index Addendum (Entry 048)", h2))
s.append(table(
    [["Entry", "Title", "Date", "Status"],
     ["Entry 048", "Cross-Chip Generalization Fix (chip-identity leakage + per-chip normalization)",
      "Aug 20, 2026", "Complete"]],
    [46, 240, 62, 78], highlight=(0,)))
s.append(Paragraph(
    "Scripts: entry048_generalization_fix.py. Data: quantumbridge_data/entry048_results.json. "
    "Cross-chip R&sup2; recovered from 0.04-0.37 (Entry 047) to 0.75-0.87 by removing "
    "chip-identity leakage from the input and normalizing calibration features per-chip "
    "instead of globally.", cap))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="QuantumBridge Research Log — Entry 048",
                      author="Darknight (Mirr)")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, BM, W - LM - RM,
                                                         H - TM - BM, id="f")],
                                   onPage=header_footer)])
doc.build(s)
print("built", OUT)
