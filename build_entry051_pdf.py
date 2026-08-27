"""Build Entry 051 pages in the QuantumBridge Research Log house style."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

OUT = "entry051_pages.pdf"
START_PAGE = 121
RUNNING_TITLE = "Entry 051 — Floor-Collapse Up-Weighting"

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
s.append(entry_banner("ENTRY 051 &nbsp;&nbsp; Floor-Collapse Up-Weighting — "
                      "August 21, 2026 — Status: COMPLETE"))
s.append(Spacer(1, 9))
s.append(Paragraph("Entry 051 — Up-Weighting Floor-Collapse Cases 5x in the Loss "
                   "Improves Cross-Chip Robustness on the Failure Mode That Matters Most", h1))
s.append(Paragraph("A Direct, Successful Response to Entry 050's Diagnosis", sub))

s.append(Paragraph("1. The Fix", h2))
s.append(Paragraph(
    "Entry 050 found that the GNN's floor-collapse override behavior -- correctly ignoring "
    "v4.1's prediction and outputting the true ~50% floor -- degrades on an unseen chip, "
    "likely because floor-collapse cases are only ~5% of training data and ordinary MAE loss "
    "gives the model little pressure to make that specific behavior robust rather than "
    "curve-fit to the training chip. This entry up-weights floor-collapse training examples "
    "5x in the loss, keeping everything else (architecture, per-chip normalization, no-chip-"
    "identity fix from Entry 048) unchanged.", lead))

s.append(Paragraph("2. Result: Real Improvement, One Direction Fully, One Partially", h2))
s.append(table(
    [["train &rarr; test", "MAE (Entry 048)", "MAE (weighted)", "fc MAE (048)", "fc MAE (weighted)"],
     ["Kyiv &rarr; Sherbrooke (cold)", "4.40 pts", "4.99 pts", "12.57 pts", "8.69 pts"],
     ["Sherbrooke &rarr; Kyiv (cold)", "3.20 pts", "2.76 pts", "4.47 pts", "3.48 pts"]],
    [150, 100, 100, 90, 90], highlight=(0,), align_right=(1, 2, 3, 4)))
s.append(Paragraph(
    "Sherbrooke-to-Kyiv improved on every metric simultaneously (overall MAE 3.20 to 2.76, "
    "R&sup2; 0.866 to 0.884, floor-collapse MAE 4.47 to 3.48) -- up-weighting cost nothing "
    "here. Kyiv-to-Sherbrooke shows the expected tradeoff: floor-collapse MAE improved "
    "substantially (12.57 to 8.69, a 31% reduction) at a small cost to overall accuracy "
    "(MAE 4.40 to 4.99, R&sup2; 0.747 to 0.708). Given this project exists specifically to "
    "catch floor-collapse -- the exact failure mode v4.1 cannot see -- this is a trade worth "
    "making: a small overall accuracy cost for meaningfully more robust behavior on the "
    "cases that matter most.", body));

s.append(Paragraph("Entry 051 — Conclusion", h2))
s.append(Spacer(1, 4))
s.append(Paragraph(
    "The up-weighting fix works and is adopted going forward. This closes the loop opened by "
    "Entry 047's AI-review-motivated skepticism, through Entry 048's generalization fix and "
    "Entry 050's precise diagnosis, to a targeted and validated fix here -- a complete, "
    "evidence-driven cycle from external critique to concrete improvement.", concl))

s.append(Paragraph("3. Next Step", h2))
s.append(Paragraph(
    "Per updated project direction: push real physical qubit capacity from 48 to roughly "
    "60-70 nodes via larger star-GHZ circuits, and grow the dataset toward 3,000 circuits "
    "(revised down from the earlier 5,000 target), then retrain and re-verify both the "
    "capacity increase and this floor-collapse fix hold up together at the new scale.", body))

s.append(Spacer(1, 10))
s.append(Paragraph("Research Log — Index Addendum (Entry 051)", h2))
s.append(table(
    [["Entry", "Title", "Date", "Status"],
     ["Entry 051", "Floor-Collapse Up-Weighted Training (5x loss weight)",
      "Aug 21, 2026", "Complete"]],
    [46, 240, 62, 78], highlight=(0,)))
s.append(Paragraph(
    "Scripts: entry051_floorcollapse_weighted.py. Data: quantumbridge_data/entry051_results.json. "
    "Floor-collapse MAE improved 22-31% in both cross-chip directions; Sherbrooke-to-Kyiv "
    "improved on every metric simultaneously.", cap))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="QuantumBridge Research Log — Entry 051",
                      author="Darknight (Mirr)")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, BM, W - LM - RM,
                                                         H - TM - BM, id="f")],
                                   onPage=header_footer)])
doc.build(s)
print("built", OUT)
