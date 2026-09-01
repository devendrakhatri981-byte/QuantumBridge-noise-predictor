"""Build Entry 058 pages in the QuantumBridge Research Log house style."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

OUT = "entry058_pages.pdf"
START_PAGE = 132
RUNNING_TITLE = "Entry 058 — Chain-Topology Circuit Diversity"

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
s.append(entry_banner("ENTRY 058 &nbsp;&nbsp; Chain-Topology Circuit Diversity — "
                      "August 31, 2026 — Status: COMPLETE"))
s.append(Spacer(1, 9))
s.append(Paragraph("Entry 058 — A Second Large-Circuit Topology Improves Kyiv-to-Sherbrooke, "
                   "at a Small Cost to Sherbrooke-to-Kyiv", h1))
s.append(Paragraph("Every Large Circuit Before This Entry Was Star-Topology Only", sub))

s.append(Paragraph("1. Motivation", h2))
s.append(Paragraph(
    "All 24 large circuits from Entries 052/056 were star-topology GHZ (one control qubit "
    "entangled to k-1 targets via a hub-and-spoke pattern). This entry adds a structurally "
    "different generator: a chain (q0-q1-q2-...-qk-1, entangled sequentially via H then a "
    "run of CX gates). Both produce the same GHZ state, but the chain routes as a long path "
    "rather than a hub, giving the GNN a genuinely different graph shape to learn from at "
    "large scale rather than only ever seeing star-shaped large circuits.", lead))

s.append(Paragraph("2. A Harder Generation Run Than Star Circuits", h2))
s.append(Paragraph(
    "Chain circuits at k=14 proved too expensive to reliably generate within the sandbox's "
    "time budget -- Sherbrooke chain generation stalled entirely across two full-budget "
    "attempts and was cut rather than continuing to burn compute on it. The final set is 4 "
    "circuits, Kyiv only, at k=10 and k=12 (62-63 physical nodes) -- a small but structurally "
    "meaningful addition, not a full-scale replacement for star circuits.", body))

s.append(Paragraph("3. Results: Same-Chip and Cross-Chip", h2))
s.append(table(
    [["metric", "Entry 056 (n=3,217, star only)", "Entry 058 (n=3,221, +chain)"],
     ["same-chip MAE", "1.13 pts", "1.09 pts"],
     ["same-chip R&sup2;", "0.976", "0.977"],
     ["Kyiv&rarr;Sherbrooke MAE / R&sup2; / fc", "4.10 / 0.814 / 8.60", "3.55 / 0.859 / 7.02"],
     ["Sherbrooke&rarr;Kyiv MAE / R&sup2; / fc", "2.65 / 0.905 / 2.87", "2.88 / 0.901 / 4.69"]],
    [190, 175, 140], highlight=(0,)))
s.append(Paragraph(
    "Kyiv-to-Sherbrooke -- the persistently harder direction across this whole project -- "
    "improved meaningfully on every metric (MAE 4.10 to 3.55, R&sup2; 0.814 to 0.859, "
    "floor-collapse MAE 8.60 to 7.02). This makes sense: the 4 new chain circuits are all on "
    "the Kyiv side, so Kyiv's training set gained topological diversity that Sherbrooke's "
    "training data (when training the other direction) did not. Sherbrooke-to-Kyiv regressed "
    "slightly on floor-collapse (2.87 to 4.69 pts) while staying flat on MAE/R&sup2; -- a small, "
    "honest cost, plausibly because the Kyiv-trained-then-tested-on-itself calibration shifted "
    "slightly with the added chain examples changing the per-chip normalization statistics.", body))

s.append(Paragraph("Entry 058 — Conclusion", h2))
s.append(Spacer(1, 4))
s.append(Paragraph(
    "Topological diversity, even a small dose of it, helped the harder cross-chip direction "
    "more than it hurt the easier one -- a positive but imperfect result, reported here with "
    "both the gain and the regression rather than only the favorable number.", concl))

s.append(Paragraph("4. Next Steps", h2))
s.append(Paragraph(
    "A fresh 3D visualization of the network showing both star and chain circuit topologies "
    "is next, followed by live demo polish (floor-collapse override explanation, cross-chip "
    "confidence context in the UI).", body))

s.append(Spacer(1, 10))
s.append(Paragraph("Research Log — Index Addendum (Entry 058)", h2))
s.append(table(
    [["Entry", "Title", "Date", "Status"],
     ["Entry 058", "Chain-Topology Circuit Diversity (4 circuits, Kyiv, k=10/12)",
      "Aug 31, 2026", "Complete"]],
    [46, 300, 62, 78], highlight=(0,)))
s.append(Paragraph(
    "Scripts: entry058_chain_circuits.py, entry058_build_graphs.py, entry058_train_gnn.py, "
    "entry058_cross_chip.py. Data: quantumbridge_data/entry058_chain_dataset.json, "
    "entry058_graph_dataset.json (3,221 total), entry058_gnn_results.json, "
    "entry058_cross_chip_results.json. Sherbrooke chain generation stalled and was skipped; "
    "Kyiv-only chain circuits still improved Kyiv-to-Sherbrooke transfer meaningfully.", cap))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="QuantumBridge Research Log — Entry 058",
                      author="Darknight (Mirr)")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, BM, W - LM - RM,
                                                         H - TM - BM, id="f")],
                                   onPage=header_footer)])
doc.build(s)
print("built", OUT)
