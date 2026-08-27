"""Build Entry 050 pages in the QuantumBridge Research Log house style."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

OUT = "entry050_pages.pdf"
START_PAGE = 120
RUNNING_TITLE = "Entry 050 — Diagnosing the Floor-Collapse Regression"

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
s.append(entry_banner("ENTRY 050 &nbsp;&nbsp; Diagnosing the Floor-Collapse Regression — "
                      "August 21, 2026 — Status: COMPLETE"))
s.append(Spacer(1, 9))
s.append(Paragraph("Entry 050 — On an Unseen Chip, the GNN Partly Reverts to "
                   "Trusting the Formula It Was Built to Override", h1))
s.append(Paragraph("The Floor-Collapse Override Mechanism Is Itself Less "
                   "Transferable Than the Model's General Accuracy", sub))

s.append(Paragraph("1. The Question", h2))
s.append(Paragraph(
    "Entry 048's generalization fix (removing chip-identity leakage, per-chip feature "
    "normalization) improved overall Kyiv-to-Sherbrooke cross-chip accuracy substantially "
    "(R&sup2; 0.037 to 0.747), but floor-collapse MAE specifically got WORSE in that direction "
    "(8.20 to 12.57 points) even as everything else improved. This entry isolates why by "
    "comparing the model's actual predictions, not just aggregate error, on floor-collapse "
    "cases it was trained on (Kyiv) versus cases it had never seen (Sherbrooke).", lead))

s.append(Paragraph("2. Diagnosis: the Override Behavior Doesn't Transfer", h2))
s.append(table(
    [["", "Kyiv (trained, in-sample)", "Sherbrooke (cold, unseen)"],
     ["n floor-collapse cases", "67", "50"],
     ["mean GNN prediction", "0.497", "0.615"],
     ["mean real (Aer) result", "0.498", "0.498"],
     ["correlation(GNN pred, v4.1 pred)", "0.08", "0.46"]],
    [200, 165, 165], highlight=(0,), align_right=(1, 2)))
s.append(Paragraph(
    "On Kyiv, the chip it trained on, the model has essentially learned to ignore the v4.1 "
    "formula's prediction entirely for floor-collapse cases (correlation 0.08, near-zero) and "
    "correctly outputs the true ~50% floor (mean prediction 0.497 against a true mean of "
    "0.498 -- functionally perfect). On Sherbrooke, a chip it never trained on, that override "
    "behavior partially breaks down: predictions correlate moderately with v4.1 (0.46) and "
    "are biased upward (mean 0.615 vs true 0.498), meaning the model is partly falling back "
    "toward trusting the closed-form formula it was specifically built to catch failing. "
    "Individual cases show this directly -- several Sherbrooke floor-collapse circuits get "
    "predictions that land close to v4.1's (wrong) number rather than the true floor.", body))

s.append(Paragraph("Entry 050 — Conclusion", h2))
s.append(Spacer(1, 4))
s.append(Paragraph(
    "This resolves Entry 048's open question with a specific, mechanistic answer rather than "
    "a vague one: the GNN's general fidelity estimation transfers reasonably well across "
    "chips (per Entry 048's fix), but its floor-collapse OVERRIDE mechanism -- arguably the "
    "single most valuable thing this project's GNN does, since it's the exact failure mode "
    "v4.1 cannot see -- is itself less transferable and partially collapses back toward "
    "trusting v4.1 on an unseen chip. This is a more precise and more useful finding than "
    "'cross-chip generalization needs work': the general regression task generalizes; the "
    "specific override behavior this project cares about most does not, yet.", concl))

s.append(Paragraph("3. Next Step", h2))
s.append(Paragraph(
    "Consider training with floor-collapse cases explicitly up-weighted or with a loss term "
    "that penalizes over-trusting v4.1's input feature specifically on ambiguous cases, so "
    "the override behavior is pushed to depend more on graph/calibration structure and less "
    "on the v4.1 input feature, which may be an easier shortcut the model leans on when "
    "out-of-distribution. Joint training across both chips (rather than strict "
    "leave-one-chip-out) remains untried and may also help, since the model would see "
    "floor-collapse examples from both chips during training.", body))

s.append(Spacer(1, 10))
s.append(Paragraph("Research Log — Index Addendum (Entry 050)", h2))
s.append(table(
    [["Entry", "Title", "Date", "Status"],
     ["Entry 050", "Diagnosing the Kyiv-to-Sherbrooke Floor-Collapse Regression",
      "Aug 21, 2026", "Complete"]],
    [46, 240, 62, 78], highlight=(0,)))
s.append(Paragraph(
    "Data: quantumbridge_data/entry050_floorcollapse_diagnosis.json. Real-hardware "
    "validation (Entry 049 scaffolding) is paused for now due to IBM Cloud account "
    "verification friction -- resuming simulation-based work on existing resources per "
    "project direction.", cap))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="QuantumBridge Research Log — Entry 050",
                      author="Darknight (Mirr)")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, BM, W - LM - RM,
                                                         H - TM - BM, id="f")],
                                   onPage=header_footer)])
doc.build(s)
print("built", OUT)
