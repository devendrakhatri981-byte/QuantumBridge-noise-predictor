"""Build Entry 057 pages in the QuantumBridge Research Log house style."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

OUT = "entry057_pages.pdf"
START_PAGE = 129
RUNNING_TITLE = "Entry 057 — MC-Dropout Uncertainty Quantification"

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
warn = ParagraphStyle("warn", parent=body, backColor=colors.HexColor("#FBEEEC"),
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
s.append(entry_banner("ENTRY 057 &nbsp;&nbsp; MC-Dropout Uncertainty Quantification — "
                      "August 25, 2026 — Status: COMPLETE"))
s.append(Spacer(1, 9))
s.append(Paragraph("Entry 057 — The Model Now Knows When It's Out of Its Depth", h1))
s.append(Paragraph("First Research-Value Feature Beyond Point Prediction: Calibrated Confidence Intervals", sub))

s.append(Paragraph("1. Motivation and Method", h2))
s.append(Paragraph(
    "Every AI reviewer of the public demo (see the pasted Gemini/ChatGPT/Grok reviews earlier "
    "this project phase) converged on the same critique: a single point-estimate prediction "
    "gives no reason to prefer this model over just re-running Aer. MC-Dropout (Gal &amp; "
    "Ghahramani, 2016) is the cheapest fix that directly answers this -- dropout (rate=0.15) "
    "is added after node embedding and after each message-passing round, kept active at both "
    "train and inference time. At inference, T=20 stochastic forward passes are run per "
    "circuit; their mean is the point prediction and their std is the uncertainty. Architecture, "
    "per-chip normalization (Entry 048), and floor-collapse up-weighting (Entry 051) are "
    "otherwise unchanged from Entry 056, so any behavior change is attributable to dropout "
    "alone. Cost: one training run instead of a 5-model ensemble.", lead))

s.append(Paragraph("2. A Methodological Error Caught and Fixed Mid-Entry", h2))
s.append(Paragraph(
    "The first evaluation run used a random 80/20 KFold split of the full mixed-chip dataset, "
    "then labeled the chip of one held-out example as the \"cold\" chip. This is invalid: with "
    "a random split, the model had already seen plenty of that chip's circuits during training, "
    "so it was not testing genuine distribution shift -- it produced a misleadingly weak "
    "widening ratio (1.08x). This was caught before being reported as a result. The evaluation "
    "was rebuilt to match the proper cross-chip protocol used in Entries 048/051/053-056: train "
    "on ALL Kyiv circuits only, with Sherbrooke held out completely and never seen during "
    "training.", warn))

s.append(Paragraph("3. Results: Corrected Cross-Chip Protocol", h2))
s.append(table(
    [["test set", "MAE", "mean predicted std", "corr(std, |error|)"],
     ["held-out Kyiv (warm, in-distribution)", "0.90 pts", "0.57 pts", "0.640"],
     ["ALL Sherbrooke (cold, never seen)", "4.63 pts", "1.57 pts", "0.489"]],
    [220, 90, 130, 110], highlight=(0,), align_right=(1, 2, 3)))
s.append(Paragraph(
    "Both calibration checks the plan called for came back positive. First, within the trained "
    "chip, higher predicted uncertainty correlates with higher actual error (r=0.64) -- the "
    "model's confidence is meaningfully informative, not noise. Second, and more importantly, "
    "mean predicted uncertainty nearly triples on the truly unseen chip (0.57 to 1.57 pts, a "
    "2.77x widening ratio), tracking the real jump in error (MAE 0.90 to 4.63 pts) on that same "
    "transfer. The model is not just making worse predictions on new hardware -- it is "
    "correctly reporting that it should be trusted less there.", body))

s.append(Paragraph("Entry 057 — Conclusion", h2))
s.append(Spacer(1, 4))
s.append(Paragraph(
    "This is the first capability in the project that Aer's point-estimate output cannot "
    "provide by construction: a self-reported, empirically calibrated confidence signal that "
    "widens correctly under distribution shift. It directly answers the external reviewers' "
    "\"why not just use Aer\" critique with a concrete, measured result rather than an "
    "architectural claim.", concl))

s.append(Paragraph("4. Next Steps", h2))
s.append(Paragraph(
    "Expose the uncertainty band in the live demo alongside the point prediction (e.g. "
    "\"72% &plusmn; 4%\"). Continue large-circuit and dataset growth as capacity allows, and "
    "consider running MC-Dropout on the Sherbrooke-to-Kyiv direction too for a complete "
    "picture, since only Kyiv-to-Sherbrooke was tested here.", body))

s.append(Spacer(1, 10))
s.append(Paragraph("Research Log — Index Addendum (Entry 057)", h2))
s.append(table(
    [["Entry", "Title", "Date", "Status"],
     ["Entry 057", "MC-Dropout Uncertainty Quantification (Kyiv&rarr;Sherbrooke)",
      "Aug 25, 2026", "Complete"]],
    [46, 300, 62, 78], highlight=(0,)))
s.append(Paragraph(
    "Scripts: entry057_mc_dropout.py (modes: train/eval for same-chip, train_cc/eval_cc for "
    "proper cross-chip). Data: quantumbridge_data/entry057_mcdropout_results.json (same-chip, "
    "flawed cross-chip check retained for transparency), entry057_mcdropout_cc_results.json "
    "(corrected cross-chip result). Uncertainty widens 2.77x on truly unseen chip, correlating "
    "with a real 5x increase in error (0.90 to 4.63 pts MAE).", cap))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="QuantumBridge Research Log — Entry 057",
                      author="Darknight (Mirr)")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, BM, W - LM - RM,
                                                         H - TM - BM, id="f")],
                                   onPage=header_footer)])
doc.build(s)
print("built", OUT)
