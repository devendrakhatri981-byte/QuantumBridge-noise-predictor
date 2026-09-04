"""Build Entry 067-070 pages in the QuantumBridge Research Log house style."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

OUT = "entry067_pages.pdf"
START_PAGE = 145
RUNNING_TITLE = "Entry 067-070 — Balancing Sherbrooke"

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


def kv_table(rows, col0=45):
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
s.append(entry_banner("ENTRIES 067-070 &nbsp;&nbsp; Balancing Sherbrooke: Growth, Retrain, Re-Score — "
                      "September 4, 2026 — Status: COMPLETE"))
s.append(Spacer(1, 9))
s.append(Paragraph("Balancing the Dataset Across All Three Chips", h1))
s.append(Paragraph("Sherbrooke Was the Thinnest Chip -- This Closes That Gap", sub))

s.append(Paragraph("Motivation", h2))
s.append(Paragraph(
    "After Entries 060-066, chip dataset sizes were uneven: Kyiv ~1,510 bell circuits, Brisbane "
    "~1,621 (after its own dedicated growth push), but Sherbrooke only 723 -- the thinnest of the "
    "three despite being one of the two original chips. Entry 067 grows Sherbrooke specifically, "
    "reusing the same methodology as Brisbane's growth (Entry 061): Aer/MPS ground truth, six "
    "hop-distance bins, atomic checkpointed writes, shuffled task order to avoid starving long-hop "
    "bins. 804 new circuits were added (723 -&gt; 1,527 bell, plus the existing 24 chain circuits), "
    "bringing Sherbrooke roughly to parity with the other two chips.", lead))

s.append(Paragraph("Entry 068 — Fold and Retrain", h2))
s.append(Paragraph(
    "The 804 new circuits were folded into the graph dataset (4,886 -&gt; 5,690 graphs, still well "
    "within the 90/130 node/edge capacity -- max 67 nodes, 106 edges). Same-chip 80/20 baseline was "
    "retrained: MAE=1.07, R&sup2;=0.965, essentially unchanged from Entry 063's 1.17/0.973 at the "
    "smaller scale, confirming the added Sherbrooke data didn't destabilize the well-established "
    "same-chip fit.", lead))

s.append(Paragraph("Leave-One-Chip-Out Cross-Chip Results — Before / After Sherbrooke Growth", h2))
s.append(kv_table([
    ["Direction", "R² (Entry 063)", "R² (Entry 068)", "Change"],
    ["kyiv+sherbrooke → brisbane", "0.275", "0.407", "+0.132 (better)"],
    ["brisbane → kyiv+sherbrooke", "0.811", "0.733", "−0.078 (worse)"],
]))
s.append(Paragraph(
    "The two directions moved in opposite ways, and both are explainable rather than contradictory. "
    "Training on a richer, more balanced kyiv+sherbrooke pool gave the model more diverse noise "
    "structure to learn from, which transferred better to the untouched Brisbane test set -- a real "
    "generalization gain. The reverse direction got harder for a structural reason, not a regression: "
    "the kyiv+sherbrooke test set itself grew by 804 new, harder examples (many long-hop, "
    "previously-unsampled pairs) that a Brisbane-only-trained model had no chance to prepare for. "
    "This is the same pattern seen in Entries 061-064's non-monotonic Brisbane R&sup2; trajectory: "
    "a harder, more representative test set can look like a worse score even when nothing about the "
    "model's underlying competence changed.", lead))

s.append(Paragraph("Entry 069 — Combined Three-Chip Generalist, Retrained", h2))
s.append(kv_table([
    ["Chip", "MAE (Entry 065)", "MAE (Entry 069)", "floor-collapse MAE (069)"],
    ["Kyiv", "0.96", "0.77", "0.56 (n=26)"],
    ["Sherbrooke", "1.74", "1.45", "0.56 (n=40)"],
    ["Brisbane", "1.05", "0.99", "0.32 (n=4)"],
    ["Overall", "1.17", "1.07", "—"],
]))
s.append(Paragraph(
    "Every chip's held-out MAE improved, and Sherbrooke's floor-collapse MAE -- the hardest failure "
    "mode this project has chased since Entry 036 -- fell from 3.13 to 0.56, a direct benefit of "
    "Sherbrooke finally having enough training volume to let the floor-collapse up-weighting (Entry "
    "051) do its job properly. The deployable weights were saved (entry069_combined_params.json) and "
    "used to re-score all 24,003 precomputed qubit pairs (Entry 070), replacing the live demo's "
    "Brisbane predictions and refreshing ground truth for the 804 newly-simulated Sherbrooke pairs.", lead))

s.append(Paragraph("Entries 067-070 — Conclusion", h2))
s.append(Spacer(1, 4))
s.append(Paragraph(
    "Balancing the thinnest chip's dataset produced a strictly better combined generalist model "
    "across all three chips, and a real (if partial) improvement in the hardest cross-chip "
    "generalization direction. It also produced a harder, more honest test in the other direction -- "
    "exactly the kind of result this project has learned to report as informative rather than "
    "alarming. The live demo now reflects the retrained combined model and the wider ground-truth "
    "coverage for Sherbrooke.", concl))

s.append(Spacer(1, 10))
s.append(Paragraph("Research Log — Index Addendum (Entries 067-070)", h2))
s.append(Paragraph(
    "Files: entry067_grow_sherbrooke.py, entry068_build_graphs.py, entry068_train_gnn.py, "
    "entry068_cross_chip.py, entry069_combined_train.py, entry070_score_all_pairs.py. Data: "
    "quantumbridge_data/entry067_sherbrooke_bell_dataset.json (804 new circuits), "
    "entry068_graph_dataset.json (5,690 graphs), entry069_combined_params.json (deployable weights), "
    "entry070_combined_lookup.json (rescored demo lookup).", cap))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="QuantumBridge Research Log — Entries 067-070",
                      author="Darknight (Mirr)")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, BM, W - LM - RM,
                                                         H - TM - BM, id="f")],
                                   onPage=header_footer)])
doc.build(s)
print("built", OUT)
