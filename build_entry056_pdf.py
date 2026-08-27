"""Build Entry 056 pages in the QuantumBridge Research Log house style."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

OUT = "entry056_pages.pdf"
START_PAGE = 128
RUNNING_TITLE = "Entry 056 — Large-Circuit Tier Doubled, 65 to 67 Nodes"

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
s.append(entry_banner("ENTRY 056 &nbsp;&nbsp; Large-Circuit Tier Doubled (12 &rarr; 24) — "
                      "August 25, 2026 — Status: COMPLETE"))
s.append(Spacer(1, 9))
s.append(Paragraph("Entry 056 — Doubling the 60+ Physical Qubit Tier and "
                   "Widening Capacity to 90/130", h1))
s.append(Paragraph("Deepening the Underrepresented Large-Circuit Slice, Per Updated Project Direction", sub))

s.append(Paragraph("1. Motivation", h2))
s.append(Paragraph(
    "Following Entry 055's dataset-growth pause, focus shifted to the large-circuit tier: only "
    "12 star-GHZ circuits at k=10/12/14 existed (from Entry 052), a thin slice relative to the "
    "3,205-circuit bell-pair-dominated set. entry056_grow_bigger.py topped up PER_SIZE from 2 "
    "to 4 per size per chip using the same resumable have_per_size logic as Entry 052, adding "
    "12 new circuits (doubling the tier to 24) and pushing the observed maximum from 65 to 67 "
    "physical nodes.", lead))

s.append(Paragraph("2. A New Padding Ceiling Was Hit and Fixed Before Training", h2))
s.append(Paragraph(
    "The new circuits pushed max edges to 106, exceeding the Entry 053/054/055 padding "
    "ceiling of MAX_E=104. This was caught before training rather than silently truncating "
    "data: MAX_N/MAX_E were widened to 90/130 (headroom above the new 67/106 observed max, "
    "same margin convention as every prior capacity bump).", body))

s.append(Paragraph("3. Results: Held or Improved Everywhere", h2))
s.append(table(
    [["metric", "Entry 055 (n=3,205)", "Entry 056 (n=3,217)"],
     ["same-chip MAE", "1.16 pts", "1.13 pts"],
     ["same-chip R&sup2;", "0.970", "0.976"],
     ["Kyiv&rarr;Sherbrooke MAE / R&sup2; / fc", "4.08 / 0.786 / 8.52", "4.10 / 0.814 / 8.60"],
     ["Sherbrooke&rarr;Kyiv MAE / R&sup2; / fc", "2.87 / 0.894 / 3.72", "2.65 / 0.905 / 2.87"]],
    [190, 165, 165], highlight=(0,)))
s.append(Paragraph(
    "Adding 12 large, structurally different circuits (deep routed star topologies vs. mostly "
    "short bell pairs) did not destabilize the model at the wider 90/130 capacity -- same-chip "
    "accuracy improved slightly, Kyiv-to-Sherbrooke R&sup2; improved (0.786 to 0.814) with MAE "
    "and floor-collapse essentially flat, and Sherbrooke-to-Kyiv improved on every metric (MAE "
    "2.87 to 2.65, R&sup2; 0.894 to 0.905, floor-collapse MAE 3.72 to 2.87 -- a new best across "
    "all entries in this direction).", body))

s.append(Paragraph("Entry 056 — Conclusion", h2))
s.append(Spacer(1, 4))
s.append(Paragraph(
    "Growing the large-circuit tier and widening capacity a second time (80/104 to 90/130) "
    "composed cleanly with every prior fix, matching the pattern already seen across Entries "
    "053-055: capacity growth and dataset growth are largely independent axes that do not "
    "trade off against each other in this architecture.", concl))

s.append(Paragraph("4. Next Steps", h2))
s.append(Paragraph(
    "Uncertainty quantification is scoped next (Entry 057) as the primary research-value "
    "addition, run alongside continued large-circuit and dataset growth as capacity allows.", body))

s.append(Spacer(1, 10))
s.append(Paragraph("Research Log — Index Addendum (Entry 056)", h2))
s.append(table(
    [["Entry", "Title", "Date", "Status"],
     ["Entry 056", "Large-Circuit Tier Doubled (12 to 24) + Capacity Bump to 90/130",
      "Aug 25, 2026", "Complete"]],
    [46, 300, 62, 78], highlight=(0,)))
s.append(Paragraph(
    "Scripts: entry056_grow_bigger.py, entry056_build_graphs.py, entry056_train_gnn.py, "
    "entry056_cross_chip.py. Data: quantumbridge_data/entry052_biggerghz_dataset.json (24 "
    "records), entry056_graph_dataset.json (3,217 total), entry056_gnn_results.json, "
    "entry056_cross_chip_results.json. Max circuit size grew 65 to 67 nodes; MAX_N/MAX_E "
    "widened 80/104 to 90/130.", cap))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="QuantumBridge Research Log — Entry 056",
                      author="Darknight (Mirr)")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, BM, W - LM - RM,
                                                         H - TM - BM, id="f")],
                                   onPage=header_footer)])
doc.build(s)
print("built", OUT)
