"""Build Entry 073 pages in the QuantumBridge Research Log house style."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

OUT = "entry073_pages.pdf"
START_PAGE = 149
RUNNING_TITLE = "Entry 073 — The True Zero-Shot Fourth-Chip Test"

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
s.append(entry_banner("ENTRY 073 &nbsp;&nbsp; The True Zero-Shot Fourth-Chip Test — "
                      "September 5, 2026 — Status: COMPLETE, HONEST MIXED RESULT"))
s.append(Spacer(1, 9))
s.append(Paragraph("Osaka: A Chip the Model Has Never Touched", h1))
s.append(Paragraph("The Cleanest Cold-Transfer Test This Project Has Run", sub))

s.append(Paragraph("Why This Test Is Different From Every Prior One", h2))
s.append(Paragraph(
    "Brisbane (Entry 061) was added specifically to test cross-chip generalization, but became a "
    "training chip almost immediately (Entry 065's combined model) -- so no result since has "
    "reflected a chip the DEPLOYED model has literally never seen. FakeOsaka (127 qubits, same "
    "heavy-hex class as the other three, 144 real calibrated edges) was set up and a 708-circuit "
    "eval-only bell-pair dataset was grown for it, but -- critically -- never folded into any "
    "training pipeline. The Entry 071 deployed model, trained on Kyiv+Sherbrooke+Brisbane, was "
    "scored cold against it. Osaka has no per-chip normalization statistics in the model at all; "
    "the average of the three training chips' stats was used as the best available substitute, "
    "since a real deployed tool facing an unfamiliar chip would need some answer here.", lead))

s.append(Paragraph("Results — Osaka, Zero Training Exposure", h2))
s.append(kv_table([
    ["Metric", "GNN (Entry 071)", "v4.1 closed-form", "GNN on trained chips (ref.)"],
    ["MAE", "5.82", "5.57", "1.38"],
    ["R²", "0.623", "0.686", "0.964"],
    ["floor-collapse MAE (n=49)", "15.77", "22.38", "0.50-0.56"],
    ["mean predicted std", "3.27 pts", "n/a", "0.81 pts"],
]))
s.append(Paragraph(
    "Two honest findings, not one. First: on aggregate accuracy, the GNN does NOT clearly beat the "
    "hand-derived v4.1 closed-form formula on a genuinely novel chip -- they are roughly tied, with "
    "v4.1 very slightly ahead on both MAE and R&sup2;. This is worth stating plainly rather than "
    "downplaying: the GNN's advantage over the closed-form model, demonstrated repeatedly on chips "
    "it has trained on, does not automatically transfer to a chip with zero exposure. Second: on "
    "floor-collapse specifically -- the failure mode v4.1 structurally cannot see at all, and the "
    "entire reason the GNN was built (Entry 036 onward) -- the GNN still meaningfully outperforms "
    "v4.1 even zero-shot (15.77 vs 22.38 MAE), suggesting the floor-collapse detection capability "
    "generalizes better than raw point-accuracy does.", lead))

s.append(Paragraph("The Uncertainty Mechanism Passes Its Hardest Test", h2))
s.append(Paragraph(
    "Mean predicted uncertainty on Osaka (3.27 pts) is roughly 4x wider than on the model's home "
    "chips (0.81 pts) -- and the actual MAE degradation is also roughly 4x (5.82 vs 1.38). The "
    "MC-Dropout uncertainty band, calibrated in Entry 071 on held-out data from the training chips, "
    "correctly and proportionally widens when the model is genuinely out of its depth on a chip it "
    "has never seen at all. This is the most important validation this uncertainty mechanism has "
    "had: it is not just calibrated on interpolation within known chips, it also degrades gracefully "
    "and honestly under true extrapolation, which is exactly the property a deployed tool needs to "
    "avoid confidently wrong answers on unfamiliar hardware.", lead))

s.append(Paragraph("Entry 073 — Conclusion", h2))
s.append(Spacer(1, 4))
s.append(Paragraph(
    "The true zero-shot test gives a mixed, credible answer rather than a clean win: the GNN's "
    "raw accuracy advantage over the closed-form baseline does not fully survive a genuinely unseen "
    "chip, but its floor-collapse detection and its uncertainty calibration both do. For a project "
    "whose stated long-term goal is a deployable tool anyone can query on any chip, this is the "
    "single most useful result to have on record before productization begins: it tells a future "
    "user exactly what to expect (roughly v4.1-level point accuracy, better floor-collapse catching, "
    "and honestly wide error bars) on a chip the model was never trained on, rather than an "
    "optimistic claim that would not hold up.", concl))

s.append(Spacer(1, 10))
s.append(Paragraph("Research Log — Index Addendum (Entry 073)", h2))
s.append(Paragraph(
    "Files: setup_osaka.py, entry073_grow_osaka.py, entry073_zero_shot_test.py. Data: "
    "quantumbridge_data/entry073_osaka_bell_dataset.json (708 eval-only circuits, never trained on), "
    "entry073_osaka_graph_dataset.json, entry073_zero_shot_results.json.", cap))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="QuantumBridge Research Log — Entry 073",
                      author="Darknight (Mirr)")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, BM, W - LM - RM,
                                                         H - TM - BM, id="f")],
                                   onPage=header_footer)])
doc.build(s)
print("built", OUT)
