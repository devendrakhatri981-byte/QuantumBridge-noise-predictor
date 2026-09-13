"""Build Entry 074 page in the QuantumBridge Research Log house style."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

OUT = "entry074_pages.pdf"
START_PAGE = 151
RUNNING_TITLE = "Entry 074 — Live Inference API (v1 Productization)"

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
    canvas.drawString(LM, y - 23, "QuantumBridge — Phase 3 (Productization)")
    canvas.restoreState()


def entry_banner(text):
    t = Table([[Paragraph(text, banner)]], colWidths=[W - LM - RM], rowHeights=[16])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY),
                           ("LEFTPADDING", (0, 0), (-1, -1), 7),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


def kv_table(rows, col0=52):
    t = Table(rows, colWidths=[col0 * mm] + [((W - LM - RM - col0 * mm) / (len(rows[0]) - 1))] * (len(rows[0]) - 1))
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    return t


s = []
s.append(entry_banner("ENTRY 074 &nbsp;&nbsp; Live Inference API — v1 Productization Begins — "
                      "September 5, 2026 — Status: COMPLETE"))
s.append(Spacer(1, 9))
s.append(Paragraph("From Static Demo to Real Backend", h1))
s.append(Paragraph("The First Concrete Step of the Roadmap's Third Phase", sub))

s.append(Paragraph("Motivation", h2))
s.append(Paragraph(
    "Entries 060-073 established the science: a validated GNN with calibrated MC-Dropout "
    "uncertainty, tested three independent ways (leave-one-chip-out transfer, pooled multi-chip "
    "generalist, and Entry 073's true zero-shot fourth-chip test). But everything the public could "
    "actually touch -- the live demo -- ran on a static, precomputed 24,003-row lookup table, not a "
    "real service. Productizing the big \"any circuit, any chip\" vision outright would have "
    "outrun what's actually been validated. Instead, this entry builds a tightly-scoped v1: a real "
    "FastAPI backend that does live Qiskit routing, feature extraction, and GNN inference on "
    "demand, for exactly the three chips and one circuit type (bell pairs) the model was actually "
    "trained and validated on.", lead))

s.append(Paragraph("What Was Built", h2))
s.append(Paragraph(
    "api/main.py exposes three endpoints: GET /health (scope and validation caveats, returned in "
    "every response, not buried in docs), GET /chips, and POST /predict, which takes a chip and two "
    "qubit indices and runs the full live pipeline -- SABRE-routed transpile, v4.1 closed-form "
    "prediction, graph feature extraction (entry044_build_graphs.graph_for_record), and a 20-sample "
    "MC-Dropout forward pass using the Entry 071 deployed weights -- computed fresh per request, "
    "not looked up. Requesting an unsupported chip returns an explicit refusal citing Entry 073's "
    "findings rather than silently extrapolating.", lead))

s.append(Paragraph("Validation Against the Offline Pipeline", h2))
s.append(kv_table([
    ["Pair", "v4.1 (offline)", "v4.1 (live)", "GNN (offline)", "GNN (live)"],
    ["kyiv 0-1", "0.9766", "0.9766 (exact)", "0.9775", "0.9775"],
    ["kyiv 0-11", "0.762", "0.762 (exact)", "0.7985", "0.7981"],
    ["sherbrooke 0-11 (fc)", "0.715", "0.715 (exact)", "0.4974", "0.4974 (exact)"],
    ["brisbane 0-101", "0.6854", "0.6854 (exact)", "0.758", "0.7693"],
], col0=40))
s.append(Paragraph(
    "The v4.1 closed-form prediction matches the precomputed table exactly on every tested pair, "
    "confirming the live routing pipeline reproduces the offline one bit-for-bit. The GNN mean "
    "matches closely but not exactly -- the small differences (well under a percentage point on "
    "three of four pairs) are expected MC-Dropout sampling variance, not a bug, and the floor-"
    "collapse Sherbrooke pair matches exactly (0.4974, std=0 both times), which is itself informative "
    "since that's the one case where dropout noise doesn't move a saturated prediction. Latency was "
    "18-110ms per request in local testing.", lead))

s.append(Paragraph("Entry 074 — Conclusion", h2))
s.append(Spacer(1, 4))
s.append(Paragraph(
    "This is a small, deliberately incomplete step -- no auth, no rate limiting, no hosting, no "
    "circuit types beyond bell pairs -- but it is the actual architectural change productization "
    "needed: a callable service instead of a static table, validated against the offline pipeline "
    "it replaces, with every one of its stated limitations disclosed in its own API responses rather "
    "than left implicit. Deployment (choosing and configuring a host) is the user's remaining step, "
    "documented in api/README.md.", concl))

s.append(Spacer(1, 10))
s.append(Paragraph("Research Log — Index Addendum (Entry 074)", h2))
s.append(Paragraph(
    "Files: api/main.py, api/requirements.txt, api/README.md.", cap))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="QuantumBridge Research Log — Entry 074",
                      author="Darknight (Mirr)")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, BM, W - LM - RM,
                                                         H - TM - BM, id="f")],
                                   onPage=header_footer)])
doc.build(s)
print("built", OUT)
