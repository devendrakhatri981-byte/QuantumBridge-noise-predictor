"""Build Entry 057b addendum page: Sherbrooke-to-Kyiv MC-Dropout confirmation."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

OUT = "entry057b_pages.pdf"
START_PAGE = 130
RUNNING_TITLE = "Entry 057b — Reverse-Direction Confirmation"

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
s.append(entry_banner("ENTRY 057b &nbsp;&nbsp; Reverse-Direction Confirmation "
                      "(Sherbrooke &rarr; Kyiv) — August 25, 2026 — Status: COMPLETE"))
s.append(Spacer(1, 9))
s.append(Paragraph("Entry 057b — Uncertainty Widening Holds in Both Cross-Chip Directions", h1))
s.append(Paragraph("Confirms the Asymmetry Seen Across Entries 051-056 is Reflected in Uncertainty, Not Just Error", sub))

s.append(Paragraph("Results: Both Directions", h2))
s.append(table(
    [["direction", "warm MAE", "warm std", "cold MAE", "cold std", "widening ratio"],
     ["Kyiv &rarr; Sherbrooke", "0.90", "0.57", "4.63", "1.57", "2.77x"],
     ["Sherbrooke &rarr; Kyiv", "1.23", "0.97", "2.20", "1.56", "1.61x"]],
    [130, 65, 60, 60, 60, 90], highlight=(0,), align_right=(1, 2, 3, 4, 5)))
s.append(Paragraph(
    "A second independently-trained MC-Dropout model, trained on Sherbrooke only with Kyiv "
    "held out completely, reproduces the same qualitative pattern: uncertainty widens on the "
    "unseen chip (1.61x), correlating with a real error increase (MAE 1.23 to 2.20 pts). The "
    "widening is smaller than Kyiv-to-Sherbrooke's 2.77x -- consistent with every prior entry "
    "in this project finding Kyiv-to-Sherbrooke to be the harder transfer direction (Entries "
    "047, 048, 051, 053-056). The uncertainty signal is not just detecting error, it is "
    "detecting the same directional asymmetry the point-prediction error metrics have shown "
    "all along, which is strong evidence the confidence estimate is tracking something real "
    "about the underlying transfer difficulty rather than being an artifact of one run.", body))

s.append(Paragraph("Conclusion", h2))
s.append(Spacer(1, 4))
s.append(Paragraph(
    "Uncertainty quantification is validated in both directions and ready to expose in the "
    "live demo. The remaining work -- re-precomputing predictions with uncertainty for the "
    "full 16,002-pair lookup table used by the live demo -- is scoped separately given its "
    "size (comparable to Entry 046's original full-coverage precomputation).", concl))

s.append(Paragraph("Index Addendum (Entry 057b)", h2))
s.append(Paragraph(
    "Scripts: entry057_mc_dropout.py (train_cc/eval_cc, direction argument added). Data: "
    "quantumbridge_data/entry057_mcdropout_cc_rev_results.json. Confirms 2.77x/1.61x "
    "asymmetric widening matches the point-prediction error asymmetry across the whole project.",
    cap))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="QuantumBridge Research Log — Entry 057b",
                      author="Darknight (Mirr)")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, BM, W - LM - RM,
                                                         H - TM - BM, id="f")],
                                   onPage=header_footer)])
doc.build(s)
print("built", OUT)
