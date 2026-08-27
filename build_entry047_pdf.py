"""Build Entry 047 pages in the QuantumBridge Research Log house style."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

OUT = "entry047_pages.pdf"
START_PAGE = 117
RUNNING_TITLE = "Entry 047 — Cross-Chip Generalization and Feature Ablation"

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
s.append(entry_banner("ENTRY 047 &nbsp;&nbsp; Cross-Chip Generalization and Feature Ablation — "
                      "August 20, 2026 — Status: COMPLETE"))
s.append(Spacer(1, 9))
s.append(Paragraph("Entry 047 — Three Independent AI Reviews Converged on One Critique. "
                   "We Tested It.", h1))
s.append(Paragraph("Cross-Chip Accuracy Collapses 8-10x Out of Distribution; Message Passing "
                   "Is Confirmed to Be the Feature Doing the Real Work", sub))

s.append(Paragraph("1. Why This Entry", h2))
s.append(Paragraph(
    "After publishing the public demo, three different AI models (independently prompted, "
    "not shown each other's answers) were asked to critique it. All three converged on the "
    "same core objection: the GNN is trained and evaluated against Qiskit Aer's simulated "
    "noise model, so a win over the v4.1 formula could just mean the GNN approximates the "
    "simulator better -- not that it captures anything about real hardware or generalizes "
    "beyond this exact dataset. Two of three also specifically asked whether the model's "
    "features (control/target role, T1/T2 ratio, route position) and its graph structure "
    "(message passing) are actually earning their keep, or whether a simpler model would do "
    "just as well. This entry runs both checks honestly, on data already in hand -- no new "
    "circuits were generated.", lead))

s.append(Paragraph("2. Cross-Chip Generalization: the Model Does Not Transfer", h2))
s.append(table(
    [["train &rarr; test", "MAE", "R&sup2;", "floor-collapse MAE", "n floor-collapse"],
     ["same-chip CV baseline (Entry 045)", "1.21 pts", "0.971", "1.79 pts", "117 (pooled)"],
     ["Kyiv &rarr; Sherbrooke (cold)", "12.87 pts", "0.037", "8.20 pts", "50"],
     ["Sherbrooke &rarr; Kyiv (cold)", "10.31 pts", "0.373", "2.81 pts", "67"]],
    [155, 80, 70, 115, 90], highlight=(0,), align_right=(1, 2, 3)))
s.append(Paragraph(
    "Training on one chip and testing cold on the other collapses R&sup2; from 0.97 to "
    "0.04-0.37 and multiplies MAE by 8-10x. This confirms the reviewers' concern directly: "
    "the model has not been shown to learn transferable route/calibration physics -- it "
    "learns chip-specific patterns that do not carry over even to a structurally similar "
    "127-qubit device. Notably, floor-collapse MAE degrades far less than overall MAE in "
    "both directions (roughly comparable to or better than v4.1's 14.67 baseline even out "
    "of distribution), suggesting the model retains SOME of the structural signal behind "
    "the floor-collapse pattern specifically, even while its general calibration is off -- "
    "but this is not the same as validated cross-chip generalization.", body))

s.append(Paragraph("3. Feature Ablation: Message Passing Is the Real Contributor", h2))
s.append(table(
    [["variant (single 80/20 split)", "MAE", "R&sup2;", "floor-collapse MAE"],
     ["full features + message passing (baseline)", "1.23 pts", "0.977", "1.09 pts"],
     ["no control/target role flag", "1.93 pts", "0.933", "3.18 pts"],
     ["no T1/T2 ratio", "1.17 pts", "0.973", "0.95 pts"],
     ["no route-position / is-final flags", "1.20 pts", "0.974", "1.02 pts"],
     ["no message passing (features pooled directly, no graph propagation)", "2.28 pts", "0.920", "8.43 pts"]],
    [230, 70, 60, 130], highlight=(0,), align_right=(1, 2, 3)))
s.append(Paragraph(
    "Removing the control/target role flag clearly hurts (floor-collapse MAE roughly "
    "triples, 1.09 to 3.18) -- this is real signal, consistent with Entry 032's proven "
    "causal finding that the flag encodes. Removing the T1/T2 ratio or the route-position "
    "features made no meaningful difference (both variants performed within noise of the "
    "full baseline, one even fractionally better) -- these two features from Entry 044 are "
    "not carrying their claimed weight and are candidates for removal or replacement. The "
    "single most important result: disabling message passing entirely (keeping every "
    "feature but removing the graph propagation itself) is catastrophic for floor-collapse "
    "cases specifically -- MAE rises 7.7x, from 1.09 to 8.43 points, while overall MAE only "
    "roughly doubles. This directly answers 'is the GNN doing anything a flat feature set "
    "couldn't' for the exact failure mode this project was built around: yes, decisively.", body))

s.append(Paragraph("Entry 047 — Conclusion", h2))
s.append(Spacer(1, 4))
s.append(Paragraph(
    "The AI reviews were right to be skeptical, and the skepticism was actionable rather "
    "than just noise: the cross-chip test proves the model does not yet generalize beyond "
    "its training chip, which means claims of predicting \"how real IBM chips fail\" need to "
    "be walked back to \"how this specific trained model reproduces Aer's noise model on the "
    "chip it was trained on\" until cross-chip (and eventually cross-hardware) generalization "
    "is fixed. At the same time, the ablation results are a genuine positive finding: message "
    "passing over the real routed graph is doing real, measurable work on exactly the "
    "floor-collapse cases this whole project exists to catch, and the control/target role "
    "feature is validated as useful. The T1/T2 ratio and route-position features are not "
    "currently justified by the evidence and are the first things to simplify or replace.", concl))

s.append(Paragraph("4. Next Step", h2))
s.append(Paragraph(
    "Highest priority: understand and fix the cross-chip generalization failure -- likely "
    "requires either training on both chips jointly with held-out qubit ranges (not just "
    "held-out circuits), or normalizing calibration features relative to each chip's own "
    "distribution rather than raw absolute values, so the model learns relative noise "
    "patterns instead of chip-specific absolute scales. Second priority: drop or replace "
    "the T1/T2 ratio and route-position features given this entry's evidence they don't "
    "help. Real-hardware validation (actual IBM QPU jobs, not just Aer) remains the most "
    "convincing but most expensive fix and is still not attempted.", body))

s.append(Spacer(1, 10))
s.append(Paragraph("Research Log — Index Addendum (Entry 047)", h2))
s.append(table(
    [["Entry", "Title", "Date", "Status"],
     ["Entry 047", "Cross-Chip Generalization Test + Feature Ablation Study",
      "Aug 20, 2026", "Complete"]],
    [46, 240, 62, 78], highlight=(0,)))
s.append(Paragraph(
    "Scripts: entry047_experiments.py. Data: quantumbridge_data/entry047_results.json. "
    "Motivated by independent critiques from three AI models (Gemini, ChatGPT, Grok) asked "
    "to review the public demo, all of which converged on the same simulator-circularity "
    "concern. Cross-chip R&sup2; drops from 0.97 to 0.04-0.37 out of distribution; disabling "
    "message passing raises floor-collapse MAE 7.7x (1.09 to 8.43 points), confirming graph "
    "structure -- not just the feature set -- is responsible for the project's core result.", cap))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="QuantumBridge Research Log — Entry 047",
                      author="Darknight (Mirr)")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, BM, W - LM - RM,
                                                         H - TM - BM, id="f")],
                                   onPage=header_footer)])
doc.build(s)
print("built", OUT)
