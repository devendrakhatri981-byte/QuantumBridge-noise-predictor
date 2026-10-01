"""Build Entries 092-093 pages in the QuantumBridge Research Log house style."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

OUT = "entry092_093_pages.pdf"
START_PAGE = 160
RUNNING_TITLE = "Entries 092-093 — Beyond a Single Number"

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
    canvas.drawString(LM, y - 23, "QuantumBridge — Phase 3 (Model advancement)")
    canvas.restoreState()


def entry_banner(text):
    t = Table([[Paragraph(text, banner)]], colWidths=[W - LM - RM], rowHeights=[16])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY),
                           ("LEFTPADDING", (0, 0), (-1, -1), 7),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


s = []
s.append(entry_banner("ENTRIES 092-093 &nbsp;&nbsp; Beyond a Single Number — "
                      "September 25, 2026 — Status: COMPLETE"))
s.append(Spacer(1, 9))
s.append(Paragraph("Failure-Mode Breakdown, Weak-Link Identification, Chip Recommender", h1))
s.append(Paragraph("Three New Prediction Features, Built on Existing Validated Physics — No Retraining Needed", sub))

s.append(Paragraph("Why This Was Cheap", h2))
s.append(Paragraph(
    "exact_dwell_cost (Entries 024-032) already computes success as a product of independent "
    "survival terms -- final-gate error, decoherence accumulated during SWAP dwell time, single-"
    "qubit gate cost, readout error -- it just never returned them separately, only the combined "
    "number. entry092_decomposed_prediction.py duplicates the SAME validated logic (same bug fixes, "
    "same closed-form derivations) but tracks each channel independently instead of folding them "
    "into one accumulator. Checked against the original exact_dwell_cost on 8 test pairs across two "
    "chips: every decomposed total matched the original to floating-point precision. This is "
    "exposition of physics already in the codebase, not a new model with new failure modes of its "
    "own to validate.", lead))

s.append(Paragraph("Entry 092a — Failure-Mode Breakdown", h2))
s.append(Paragraph(
    "Given a circuit, reports not just \"73% success\" but which channel is costing the most: "
    "gate_error_survival, decoherence_survival, sq_gate_survival, readout_survival, ranked by loss. "
    "Example (Kyiv, qubits 77-82): total 60.4%, dominant failure mode decoherence (31.1 points of "
    "loss) vs gate error (0.6 points) -- on a long route, decoherence during the SWAP chain "
    "overwhelmingly outweighs the entangling gate's own error, exactly the kind of actionable "
    "distinction a single aggregate number can't offer.", lead))

s.append(Paragraph("Entry 092b — Weak-Link Identification", h2))
s.append(Paragraph(
    "The per-SWAP-leg dwell log the original formula already builds (physical qubits, duration, raw "
    "gate error, dephasing) directly identifies the single worst leg by combined gate-error + "
    "dephasing score. For the same 77-82 example: weak link is edge (80,79), consistent with the "
    "decoherence-dominant verdict above. No new ground truth needed -- this was always computable "
    "from data the formula already touches, just never surfaced.", lead))

s.append(Paragraph("Entry 093 — Best Chip/Route Recommender", h2))
s.append(Paragraph(
    "Turns the model from \"grade this circuit\" into \"tell me where to run it.\" Built entirely on "
    "the existing 5-chip precomputed lookup (entry081) -- a ranking/filtering layer, not a new model. "
    "recommend(min_success, max_hop, max_uncertainty, chips, top_k) returns ranked candidates; the "
    "top result gets a live failure-mode breakdown attached automatically. Verified against three "
    "realistic queries: best 5 pairs with no filters (returned a mix of Kyiv and zero-shot Quebec "
    "candidates near 98%), best trained-chip pair under hop≤5 with explanation, and a "
    "zero-shot-only query (\"chip I haven't tried yet\") -- all returned sane, ranked results.", lead))

s.append(Paragraph("Shipped", h2))
s.append(Paragraph(
    "Both new capabilities are live in the API, not just standalone scripts: POST /predict/breakdown "
    "and GET /recommend, tested end-to-end against a running server (both returned correct 200 "
    "responses with the expected structure). README updated with usage examples.", lead))

s.append(Paragraph("Entries 092-093 — Conclusion", h2))
s.append(Spacer(1, 4))
s.append(Paragraph(
    "All three requested prediction features (failure-mode breakdown, weak-link ID, chip "
    "recommender) are built, validated, and wired into the live API in one pass -- made possible "
    "because the physics-informed v4.1 formula already carried this information internally from "
    "Entries 024-032, it just needed to be exposed rather than re-derived. No new training data or "
    "model retraining was required for any of the three. Next up per the agreed plan: multi-bell-"
    "pair template scaling (Entries 089-091's track).", concl))

s.append(Spacer(1, 10))
s.append(Paragraph("Research Log — Index Addendum (Entries 092-093)", h2))
s.append(Paragraph(
    "Files: entry092_decomposed_prediction.py, entry093_recommend.py, api/main.py "
    "(/predict/breakdown, /recommend), api/README.md.", cap))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="QuantumBridge Research Log — Entries 092-093",
                      author="Darknight (Mirr)")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, BM, W - LM - RM,
                                                         H - TM - BM, id="f")],
                                   onPage=header_footer)])
doc.build(s)
print("built", OUT)
