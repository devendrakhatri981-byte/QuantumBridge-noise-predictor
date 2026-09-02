"""Build Entry 065 pages in the QuantumBridge Research Log house style."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

OUT = "entry065_pages.pdf"
START_PAGE = 143
RUNNING_TITLE = "Entry 065 — The Combined Three-Chip Generalist Model"

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


def kv_table(rows):
    t = Table(rows, colWidths=[45 * mm] + [((W - LM - RM - 45 * mm) / 4)] * 4)
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
s.append(entry_banner("ENTRY 065 &nbsp;&nbsp; The Combined Three-Chip Generalist Model — "
                      "September 3, 2026 — Status: COMPLETE, RESULT STRONG"))
s.append(Spacer(1, 9))
s.append(Paragraph("Entry 065 — A Different Question From Entries 060-064", h1))
s.append(Paragraph("Not \"Does It Transfer Cold\" But \"Does It Work Well If It's Seen a Little of Everything\"", sub))

s.append(Paragraph("The Question, Asked Directly by the User", h2))
s.append(Paragraph(
    "Entries 060-064 all tested strict leave-one-chip-out generalization: train on some chips, "
    "evaluate cold on a chip that contributed zero training examples. That's the right test for "
    "\"does this model understand physics that transfers to a totally novel chip,\" and the answer "
    "so far has been a real, only partially explained gap for Brisbane. But it is a different "
    "question from \"if I pool data from every chip I have and train one model on all of it, does "
    "that model work well across all of them\" -- which is what a practical deployed tool would "
    "actually do, and what the user asked to test directly rather than assume.", lead))

s.append(Paragraph("Setup", h2))
s.append(Paragraph(
    "All 4,886 circuits across Kyiv, Sherbrooke, and Brisbane were pooled into one training pool, "
    "with a random stratified 80/20 split (the same split used for entry063's same-chip baseline, "
    "so the two numbers are directly comparable) -- meaning every chip contributes both training "
    "examples and held-out test examples, unlike Entries 060-064 where an entire chip was held out "
    "of training completely. Architecture, per-chip relative normalization (Entry 048), and "
    "floor-collapse up-weighting (Entry 051) are all unchanged. This time the trained weights were "
    "kept as a deployable artifact rather than discarded after cross-validation, since this is the "
    "first genuinely usable combined model the project has produced.", lead))

s.append(Paragraph("Results — Per-Chip Breakdown of the Held-Out Test Set", h2))
s.append(kv_table([
    ["Chip", "n (test)", "MAE", "R²", "floor-collapse MAE"],
    ["Kyiv", "432", "0.96", "0.989", "0.65 (n=28)"],
    ["Sherbrooke", "222", "1.74", "0.937", "3.13 (n=19)"],
    ["Brisbane", "324", "1.05", "0.981", "0.39 (n=8)"],
    ["Overall", "978", "1.17", "0.973", "—"],
]))

s.append(Paragraph("What This Means", h2))
s.append(Paragraph(
    "Brisbane's numbers here (MAE 1.05, R&sup2;=0.981) are dramatically better than any "
    "leave-one-chip-out result this project has produced for Brisbane (best zero-shot R&sup2; so "
    "far: 0.354, Entry 062) -- close to Kyiv's own held-out performance, and better than "
    "Sherbrooke's in this run. That is a clean, important distinction to draw: the poor zero-shot "
    "transfer found in Entries 061-064 is not evidence that the model fundamentally cannot handle "
    "Brisbane's noise structure. It is evidence that the model cannot handle Brisbane's noise "
    "structure with *zero* examples from Brisbane. Give it even a partial, imperfect slice of "
    "Brisbane's own data during training -- which is exactly what a real deployed tool would have "
    "access to -- and it performs comparably to the chips it has always seen far more of. This "
    "reframes the earlier finding rather than contradicting it: the two experiments answer "
    "genuinely different questions, and both answers are now known.", lead))

s.append(Paragraph("What This Does Not Show", h2))
s.append(Paragraph(
    "This result says nothing about a fourth, still-unseen chip -- that is exactly the "
    "leave-one-out test this entry deliberately did not run. A model that has seen a slice of "
    "every chip it is evaluated on cannot demonstrate transfer to a chip it has never touched; "
    "only Entries 061-064's harder protocol can test that, and did. The two results together are "
    "more informative than either alone: broad multi-chip training produces a strong practical "
    "generalist (this entry), while true zero-shot transfer to a genuinely novel chip remains an "
    "open, partially-understood problem (Entries 061-064).", lead))

s.append(Paragraph("Entry 065 — Conclusion", h2))
s.append(Spacer(1, 4))
s.append(Paragraph(
    "The combined three-chip model is the first version of QuantumBridge's GNN trained on the full "
    "breadth of available chip data at once, and its weights are saved as a deployable artifact "
    "(entry065_combined_params.json) rather than discarded. It performs well across all three "
    "chips including Brisbane, closing the practical gap that Entries 061-064 left open -- while "
    "leaving the harder scientific question (true zero-shot transfer to an unseen fourth chip) "
    "explicitly unanswered, exactly as it should, since this experiment was never designed to "
    "answer it.", concl))

s.append(Spacer(1, 10))
s.append(Paragraph("Research Log — Index Addendum (Entry 065)", h2))
s.append(Paragraph(
    "Files: entry065_combined_train.py. Data: quantumbridge_data/entry065_combined_params.json "
    "(deployable weights), entry065_combined_results.json (per-chip breakdown).", cap))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="QuantumBridge Research Log — Entry 065",
                      author="Darknight (Mirr)")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, BM, W - LM - RM,
                                                         H - TM - BM, id="f")],
                                   onPage=header_footer)])
doc.build(s)
print("built", OUT)
