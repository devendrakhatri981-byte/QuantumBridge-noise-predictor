"""Build Entries 089-091 pages in the QuantumBridge Research Log house style."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

OUT = "entry089_091_pages.pdf"
START_PAGE = 158
RUNNING_TITLE = "Entries 089-091 — First Step Past One Gate That Matters"

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
    canvas.drawString(LM, y - 23, "QuantumBridge — Phase 2 (Research), new template track")
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
s.append(entry_banner("ENTRIES 089-091 &nbsp;&nbsp; First Step Past “One Gate That Matters” — "
                      "September 22, 2026 — Status: FIRST PASS COMPLETE"))
s.append(Spacer(1, 9))
s.append(Paragraph("Parallel Multi-Bell-Pair Circuits: the First Template With More Than One Thing That Matters", h1))
s.append(Paragraph("A Deliberate First Step Toward Arbitrary-Circuit Support, Not a Leap to It", sub))

s.append(Paragraph("Why This Template", h2))
s.append(Paragraph(
    "Every template so far (bell/GHZ/star/chain) has exactly one entangling event that determines "
    "success, which is what made cheap ground truth and simple feature engineering possible. "
    "Arbitrary circuits break both assumptions at once. Parallel multi-bell-pair circuits -- N "
    "independent bell pairs on disjoint qubits, run simultaneously -- are the smallest step past "
    "\"one thing matters\" while keeping ground truth cheap (still just correlation-checking, now "
    "joint across N pairs) and success unambiguous (all N pairs correlated).", lead))

s.append(Paragraph("Entry 089 — Sanity Check: Is There Even a Signal?", h2))
s.append(Paragraph(
    "Before investing in feature-schema work, checked whether parallel pairs behave differently from "
    "independent single pairs at all. Compared real joint success (Aer ground truth) against the "
    "product of each pair's own isolated v4.1 prediction -- what you'd expect under zero crosstalk. "
    "Across 20 test circuits (N=2 or 3 pairs, Kyiv): mean gap 8.2 points, max 15.6, min 2.5, in BOTH "
    "directions (joint success sometimes higher, sometimes lower than independence predicts). Real, "
    "substantial, non-trivial signal -- confirmed the template is worth building for.", lead))
s.append(Paragraph(
    "A throughput problem surfaced during growth: simulation cost does not stay flat with hop distance "
    "the way single-pair circuits do -- a single hop~20 multi-pair circuit took 36 seconds vs under a "
    "second at hop~1, since N simultaneous entangled clusters push MPS simulation cost up sharply. "
    "Capped hop distance at 8 (vs 26 for single-pair templates) to keep growth throughput practical; "
    "grew 310 circuits (N=2/3, Kyiv only) for this first pass.", lead))

s.append(Paragraph("Entry 090 — Feature Schema Extension", h2))
s.append(Paragraph(
    "Extended the graph builder with two new dimensions no prior template needed: a 7th node feature "
    "(how many DIFFERENT pairs' SWAP routing ever touched this physical qubit -- a direct proxy for "
    "the shared-resource channel crosstalk travels through), and a 5th edge feature (which of the N "
    "pairs this gate belongs to). The existing is_final logic (real entangling gate vs SWAP) already "
    "generalized correctly to multiple non-swap gates without modification. Confirmed on the built "
    "dataset: 190 of 4,975 total nodes were touched by more than one pair's routing -- the exact "
    "physical-sharing signal this schema exists to represent.", lead))

s.append(Paragraph("Entry 091 — First Model, First Result", h2))
s.append(mktable(
    ["Model", "MAE (pts)", "R2"],
    [["GNN (Entry 091, first pass)", "6.21", "0.752"],
     ["product-of-marginals (no-crosstalk baseline)", "11.56", "0.188"]],
    [260, 90, 90]))
s.append(Paragraph("Single 80/20 split, n=310 (Kyiv only, N=2/3 pairs, hop≤8). 46.3% lower MAE than "
                   "the naive independence assumption.", cap))

s.append(Paragraph("Entries 089-091 — Conclusion", h2))
s.append(Spacer(1, 4))
s.append(Paragraph(
    "This is a first-pass proof of concept, playing the same role Entry 037 played for single bell-"
    "pairs: small dataset, one chip, not yet production quality -- but it answers the question this "
    "round of work existed to answer. Parallel multi-bell circuits carry real, learnable structure "
    "that a naive independence assumption misses by double digits, and the feature-schema extension "
    "needed to capture it (shared-qubit tracking, per-pair gate tagging) works as designed. The next "
    "step, when resumed, is the obvious one: grow the dataset properly, extend across chips, and run "
    "this template through the same fold-in/holdout/zero-shot discipline used for chip generalization "
    "-- but the open question of whether this direction was worth pursuing at all is now answered.", concl))

s.append(Spacer(1, 10))
s.append(Paragraph("Research Log — Index Addendum (Entries 089-091)", h2))
s.append(Paragraph(
    "Files: entry089_grow_multibell.py (entry089_multibell_sanity.json, 310 circuits), "
    "entry090_build_multibell_graphs.py (entry090_multibell_graph_dataset.json), "
    "entry091_train_multibell.py (entry091_multibell_params.json, entry091_multibell_results.json).", cap))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="QuantumBridge Research Log — Entries 089-091",
                      author="Darknight (Mirr)")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, BM, W - LM - RM,
                                                         H - TM - BM, id="f")],
                                   onPage=header_footer)])
doc.build(s)
print("built", OUT)
