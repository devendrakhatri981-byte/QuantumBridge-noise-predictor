"""Build Entries 084-088 pages in the QuantumBridge Research Log house style."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

OUT = "entry084_088_pages.pdf"
START_PAGE = 156
RUNNING_TITLE = "Entries 084-088 — Does the Generalization Trend Hold at 5 Chips?"

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
    canvas.drawString(LM, y - 23, "QuantumBridge — Phase 2 (Research), resumed")
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
s.append(entry_banner("ENTRIES 084-088 &nbsp;&nbsp; Does the Generalization Trend Hold at 5 Chips? — "
                      "September 16-17, 2026 — Status: COMPLETE"))
s.append(Spacer(1, 9))
s.append(Paragraph("A Third Data Point, and a Broken Fake-Backend Snapshot Along the Way", h1))
s.append(Paragraph("Quebec Graduates to Training, Cusco Becomes the New Holdout", sub))

s.append(Paragraph("Plan", h2))
s.append(Paragraph(
    "Entry 077/078 found the jump from 3 to 4 training chips turned a rough tie against v4.1 "
    "(Osaka) into a clear win (Quebec). One data point is not a trend. This round repeats the "
    "cycle once more -- fold Quebec into training, stand up a 6th chip as the new holdout, retrain, "
    "and see whether the zero-shot advantage keeps growing or was a one-off.", lead))

s.append(Paragraph("Entry 084 — A Broken Fake-Backend Snapshot (Kyoto, Abandoned)", h2))
s.append(Paragraph(
    "FakeKyoto was the first candidate for the new holdout chip. Its topology and coherence exported "
    "cleanly (127 qubits, 144 edges), but every one of its 144 two-qubit gate-error calibration "
    "values was exactly 1.0 (100% error) -- compare Kyiv/Sherbrooke/Brisbane/Osaka/Quebec, which all "
    "have realistic 0.003-0.02 typical ranges. Every single Kyoto bell-pair circuit floor-collapsed, "
    "even trivial adjacent-qubit pairs (hop=1 success rate ~47%, essentially random). v4.1's own MAE "
    "on this data was 31.5 points with R2=-805 -- not a model failure, a broken calibration snapshot "
    "in this Qiskit version. Abandoned before any training time was spent on it; FakeCusco (also 127 "
    "qubits/144 edges, realistic error range 0.005-1.0 with a healthy mean of 0.16) replaced it.", lead))

s.append(Paragraph("Entry 085 — Fold Quebec Into Training (5 Chips)", h2))
s.append(Paragraph(
    "Quebec's 710 circuits (Entry 076) joined Kyiv/Sherbrooke/Brisbane/Osaka, taking the training "
    "pool to 7,108 graphs. Retrained the combined generalist and the deployable MC-Dropout model on "
    "all five. Held-out CV: overall MAE=1.18, R2=0.960. Per chip: Kyiv=0.86, Sherbrooke=1.19, "
    "Brisbane=1.09, Osaka=1.28, Quebec=2.14 (Quebec's own floor-collapse MAE=8.45, n_fc=14, visibly "
    "the hardest chip on its first exposure to training -- consistent with Entry 079's finding that a "
    "newly folded chip needs to be judged on the SAME held-out circuits across model versions before "
    "drawing conclusions, not taken as a standalone number).", lead))

s.append(Paragraph("Entry 086/087/088 — Cusco as the New Zero-Shot Holdout", h2))
s.append(Paragraph(
    "Cusco's eval-only growth stalled well short of the usual ~700-circuit target -- throughput "
    "dropped from the typical 60-90 circuits/round to under 10/round with no single bin starving "
    "(hop-distance spread stayed reasonable throughout). Rather than burn further rounds chasing a "
    "stalled generator, stopped at 228 circuits: still large enough for a meaningful zero-shot read, "
    "smaller than Osaka's 708 or Quebec's 710. This is disclosed, not hidden.", lead))

s.append(mktable(
    ["Model", "MAE (pts)", "R2", "floor-collapse MAE", "n_fc"],
    [["GNN (Entry 085, 5-chip)", "4.01", "0.784", "3.79", "63"],
     ["v4.1 closed-form", "9.32", "0.364", "17.50", "63"]],
    [180, 70, 55, 95, 45]))
s.append(Paragraph("Zero-shot comparison on Cusco (n=228, mean predicted std=1.17pts). Cusco's mean "
                   "two-qubit gate error (16.3%) is noticeably higher than Quebec's (7.4%) or Osaka's "
                   "(5.9%) -- it is simply a noisier chip, which is why both models' raw MAE are "
                   "higher here than on Quebec. The relative comparison is the meaningful one.", cap))

s.append(Paragraph("The Trend Across Three Zero-Shot Tests", h2))
s.append(mktable(
    ["Training chips", "Holdout", "GNN MAE", "v4.1 MAE", "GNN/v4.1 ratio"],
    [["3 (Kyiv/Sherbrooke/Brisbane)", "Osaka (Entry 073)", "5.82", "5.57", "1.04x (roughly tied)"],
     ["4 (+ Osaka)", "Quebec (Entry 078)", "3.02", "5.35", "0.56x (44% better)"],
     ["5 (+ Quebec)", "Cusco (Entry 088)", "4.01", "9.32", "0.43x (57% better)"]],
    [175, 105, 60, 60, 105]))
s.append(Paragraph("MAE in points. The ratio column is the fair comparison across chips of different "
                   "intrinsic difficulty: it answers \"how much better is the GNN than the plain "
                   "physics formula on a chip it has never seen\", independent of how hard that "
                   "particular chip happens to be.", cap))

s.append(Paragraph("Entries 084-088 — Conclusion", h2))
s.append(Spacer(1, 4))
s.append(Paragraph(
    "The trend holds, and holds cleanly: going from 3 to 4 to 5 training chips has monotonically "
    "widened the GNN's advantage over the closed-form baseline on chips it has never trained on -- "
    "tied, then 44% better, then 57% better. This is now three independent zero-shot tests pointing "
    "the same direction, not a single lucky result. The honest caveat: Cusco's holdout sample (228) "
    "is smaller than the other two due to a generation throughput stall, so its exact numbers carry "
    "more sampling noise than Osaka's or Quebec's -- but the direction of the result (another clear "
    "win, wider than Quebec's) is consistent with the trend rather than an outlier reversing it.", concl))

s.append(Spacer(1, 10))
s.append(Paragraph("Research Log — Index Addendum (Entries 084-088)", h2))
s.append(Paragraph(
    "Files: setup_kyoto.py (abandoned, broken calibration), setup_cusco.py, "
    "entry087_grow_cusco.py (entry087_cusco_bell_dataset.json, 228 circuits, eval-only), "
    "entry085_build_graphs.py (entry085_graph_dataset.json, 7108 graphs), entry085_train_gnn.py, "
    "entry085_combined_train.py (entry085_combined_params.json), entry085_mc_dropout.py, "
    "entry085_deploy_train.py (entry085_deploy_params.json, the new deployed model), "
    "entry088_zero_shot_test.py (entry088_zero_shot_results.json).", cap))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="QuantumBridge Research Log — Entries 084-088",
                      author="Darknight (Mirr)")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, BM, W - LM - RM,
                                                         H - TM - BM, id="f")],
                                   onPage=header_footer)])
doc.build(s)
print("built", OUT)
