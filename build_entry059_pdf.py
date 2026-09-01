"""Build Entry 059 pages in the QuantumBridge Research Log house style."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

OUT = "entry059_pages.pdf"
START_PAGE = 134
RUNNING_TITLE = "Entry 059 — 3D Visualization + Demo Polish"

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


s = []
s.append(entry_banner("ENTRY 059 &nbsp;&nbsp; 3D Star-vs-Chain Visualization + Demo Polish — "
                      "August 31, 2026 — Status: COMPLETE"))
s.append(Spacer(1, 9))
s.append(Paragraph("Entry 059 — Closing Out This Phase: Visualization and Demo UX", h1))
s.append(Paragraph("Two Small, Concrete Deliverables Rather Than New Modeling Work", sub))

s.append(Paragraph("1. New 3D Visualization", h2))
s.append(Paragraph(
    "quantumbridge_3d_star_vs_chain.html: a fresh Three.js 3D force-directed view of the real "
    "127-qubit Kyiv network (reusing the existing spring-layout node positions for continuity "
    "with the earlier full-network visualization), with two real, actually-routed large "
    "circuits highlighted in distinct colors -- the largest star-topology GHZ circuit (k=14, "
    "Entry 052/056) in orange, and the largest chain-topology GHZ circuit (k=12, Entry 058) in "
    "cyan. This makes the topological difference between the two large-circuit generators "
    "visually concrete rather than only described in text.", lead))

s.append(Paragraph("2. Live Demo Polish", h2))
s.append(Paragraph(
    "Two additions to quantumbridge_live_demo.html's per-pair explanation: (a) a new "
    "\"confident override\" note that fires whenever the GNN disagrees with v4.1 by more than "
    "15 points while reporting tight MC-Dropout uncertainty (&lt;3%) -- surfacing the model's "
    "override behavior even for pairs without real ground truth, not just the previously-shown "
    "floor-collapse cases; (b) a standing validation disclosure appended to every result noting "
    "that cross-chip transfer was genuinely tested (not just same-chip splits) and that one "
    "direction is measurably harder than the other, pointing to the research log for the full "
    "numbers rather than overstating uniform reliability.", body))

s.append(Paragraph("Entry 059 — Conclusion", h2))
s.append(Spacer(1, 4))
s.append(Paragraph(
    "This closes the current work phase: dataset at 3,221 circuits (star + chain large "
    "circuits), capacity at 90/130, generalization and floor-collapse fixes intact, "
    "uncertainty quantification built and validated in both directions, wired into the live "
    "demo, and the demo now explains its own override and reliability behavior rather than "
    "presenting bare numbers.", concl))

s.append(Spacer(1, 10))
s.append(Paragraph("Research Log — Index Addendum (Entry 059)", h2))
s.append(Paragraph(
    "Files: quantumbridge_3d_star_vs_chain.html, entry059_extract_routes.py. Demo changes: "
    "quantumbridge_live_demo.html (confident-override note, validation disclosure).", cap))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="QuantumBridge Research Log — Entry 059",
                      author="Darknight (Mirr)")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, BM, W - LM - RM,
                                                         H - TM - BM, id="f")],
                                   onPage=header_footer)])
doc.build(s)
print("built", OUT)
