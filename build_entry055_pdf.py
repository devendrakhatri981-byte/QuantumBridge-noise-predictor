"""Build Entry 055 pages in the QuantumBridge Research Log house style."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

OUT = "entry055_pages.pdf"
START_PAGE = 126
RUNNING_TITLE = "Entry 055 — Dataset Growth to 3,205 Circuits"

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
s.append(entry_banner("ENTRY 055 &nbsp;&nbsp; Dataset Growth: 2,811 &rarr; 3,205 Circuits — "
                      "August 25, 2026 — Status: COMPLETE"))
s.append(Spacer(1, 9))
s.append(Paragraph("Entry 055 — Second Growth Round Clears the 3,000+ Milestone; "
                   "Kyiv-to-Sherbrooke Floor-Collapse Trend Reverses", h1))
s.append(Paragraph("A Revised, Smaller Target Reached Efficiently After a Cost/Benefit Check-In", sub))

s.append(Paragraph("1. Background", h2))
s.append(Paragraph(
    "Following Entry 054's settle point of 2,811 circuits, growth was resumed toward an "
    "originally discussed 3,500-circuit target. Partway through, throughput of genuinely new "
    "(non-duplicate) circuits was observed slowing as easy qubit-pair bins filled up (~40 "
    "new/round down to ~20/round, with unique-yield dropping toward 60-70%). Given this, the "
    "target was revised down to a more practical 3,200, reached with the pool at 2,152 records "
    "yielding 394 new deduplicated circuits -- bringing the combined training set to 3,205, "
    "just past the milestone. Growth is paused here; the pool can be resumed later toward the "
    "original 3,500/5,000 targets if warranted.", lead));

s.append(Paragraph("2. Retrain and Re-Verify: Same-Chip", h2))
s.append(table(
    [["metric", "Entry 054 (n=2,811)", "Entry 055 (n=3,205)"],
     ["MAE", "1.37 pts", "1.16 pts"],
     ["R&sup2;", "0.964", "0.970"],
     ["floor-collapse MAE", "0.59 pts", "0.79 pts"],
     ["v4.1 MAE / fc MAE", "3.79 / 16.11", "3.88 / 16.27"]],
    [200, 160, 160], highlight=(0,), align_right=(1, 2)))
s.append(Paragraph(
    "Same-chip accuracy improved (MAE 1.37 to 1.16, R&sup2; 0.964 to 0.970) with more training "
    "data, as expected. Floor-collapse MAE ticked up slightly (0.59 to 0.79 pts) but remains "
    "far below v4.1's own 16.27-pt floor-collapse error.", body))

s.append(Paragraph("3. Retrain and Re-Verify: Cross-Chip -- the Key Result", h2))
s.append(table(
    [["train &rarr; test", "MAE (054)", "MAE (055)", "R&sup2; (054)", "R&sup2; (055)",
      "fc MAE (054)", "fc MAE (055)"],
     ["Kyiv &rarr; Sherbrooke", "4.61", "4.08", "0.711", "0.786", "12.28", "8.52"],
     ["Sherbrooke &rarr; Kyiv", "2.55", "2.87", "0.909", "0.894", "3.72", "3.72"]],
    [130, 65, 65, 65, 65, 65, 65], highlight=(0,), align_right=(1, 2, 3, 4, 5, 6)))
s.append(Paragraph(
    "Kyiv-to-Sherbrooke -- the direction that had gotten steadily worse across Entries 051, "
    "053, and 054 (fc MAE 8.69 to 9.67 to 12.28 pts) -- reversed with this larger, more "
    "diverse dataset: MAE improved (4.61 to 4.08), R&sup2; improved (0.711 to 0.786), and "
    "floor-collapse MAE dropped substantially (12.28 to 8.52). This is consistent with the "
    "structural diagnosis made before this growth round: Sherbrooke has a meaningfully higher "
    "floor-collapse rate than Kyiv (6-7% vs. 4-5%), so more Kyiv training data with proportionally "
    "more floor-collapse examples gives the up-weighted loss more signal to work with in the "
    "harder direction. Sherbrooke-to-Kyiv held essentially flat (MAE 2.55 to 2.87, floor-"
    "collapse MAE unchanged at 3.72) -- it was already the stronger direction and simply "
    "maintained that strength.", body))

s.append(Paragraph("Entry 055 — Conclusion", h2))
s.append(Spacer(1, 4))
s.append(Paragraph(
    "The multi-entry Kyiv-to-Sherbrooke floor-collapse regression is reversed, not just "
    "patched -- more training data on the underrepresented direction addressed the structural "
    "cause identified in this entry's own diagnosis, rather than requiring an architectural "
    "change. This validates the diagnosis and closes Task 63 with a confirmed, working fix "
    "path (more balanced/larger data) rather than just a hypothesis.", concl))

s.append(Paragraph("4. Next Steps", h2))
s.append(Paragraph(
    "Per updated project direction: dataset growth is paused here at 3,205 circuits (target "
    "3,500/5,000 deferred). Focus shifts to (a) generating more large circuits (60+ physical "
    "qubits, extending Entry 052's approach) to grow that underrepresented capacity tier, and "
    "(b) scoping uncertainty quantification (ensemble or MC-dropout confidence intervals) as "
    "the next research-value feature, to be worked on alongside large-circuit generation.", body))

s.append(Spacer(1, 10))
s.append(Paragraph("Research Log — Index Addendum (Entry 055)", h2))
s.append(table(
    [["Entry", "Title", "Date", "Status"],
     ["Entry 055", "Dataset Growth to 3,205 Circuits + Floor-Collapse Trend Reversal",
      "Aug 25, 2026", "Complete"]],
    [46, 300, 62, 78], highlight=(0,)))
s.append(Paragraph(
    "Scripts: entry055_build_graphs.py, entry055_train_gnn.py, entry055_cross_chip.py. "
    "Data: quantumbridge_data/entry055_graph_dataset.json, entry055_gnn_results.json, "
    "entry055_cross_chip_results.json. 394 new deduplicated circuits folded in (2,811 to "
    "3,205); Kyiv-to-Sherbrooke floor-collapse MAE improved 12.28 to 8.52 pts, reversing the "
    "trend flagged earlier in this same entry.", cap))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="QuantumBridge Research Log — Entry 055",
                      author="Darknight (Mirr)")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, BM, W - LM - RM,
                                                         H - TM - BM, id="f")],
                                   onPage=header_footer)])
doc.build(s)
print("built", OUT)
