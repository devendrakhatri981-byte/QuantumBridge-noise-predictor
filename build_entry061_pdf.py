"""Build Entry 061 pages in the QuantumBridge Research Log house style."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

OUT = "entry061_pages.pdf"
START_PAGE = 137
RUNNING_TITLE = "Entry 061 — Third Chip: Generalization Does Not Trivially Extend"

NAVY = colors.HexColor("#1F3864")
GREY = colors.HexColor("#6B6B6B")
RULE = colors.HexColor("#B8B8B8")
BAND = colors.HexColor("#EDF1F8")
WARN = colors.HexColor("#8A3B1F")

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
warnbox = ParagraphStyle("warnbox", parent=body, backColor=colors.HexColor("#FBEFEA"),
                         borderPadding=(8, 8, 8, 8), spaceBefore=4, textColor=WARN)


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
    t = Table(rows, colWidths=[75 * mm, W - LM - RM - 75 * mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


s = []
s.append(entry_banner("ENTRY 061 &nbsp;&nbsp; Third Chip (Brisbane): Generalization Does Not "
                      "Trivially Extend — September 1, 2026 — Status: COMPLETE, RESULT SOBERING"))
s.append(Spacer(1, 9))
s.append(Paragraph("Entry 061 — Adding a Third Chip to Test Whether Cross-Chip Transfer Was Real", h1))
s.append(Paragraph("Setup, Dataset, and a Result That Does Not Confirm the Hoped-For Story", sub))

s.append(Paragraph("Motivation", h2))
s.append(Paragraph(
    "Every cross-chip generalization result so far (Entries 047, 048, 050, 051, 058, 060) tested "
    "exactly one pair: Kyiv and Sherbrooke. A model that transfers well between two chips could "
    "genuinely have learned physical noise structure that generalizes &mdash; or it could be a "
    "coincidence of those two specific calibration snapshots. The only way to tell the difference "
    "is a third, independently-calibrated chip the model has never touched in any prior entry. "
    "FakeBrisbane was added for exactly this reason: same 127-qubit heavy-hex class as Kyiv and "
    "Sherbrooke (to avoid also confounding chip scale with chip identity), but its own real "
    "calibration data.", lead))

s.append(Paragraph("Dataset Built for Brisbane", h2))
s.append(Paragraph(
    "740 bell-pair circuits across the same six hop-distance bins used throughout the project, "
    "plus 12 chain-topology large circuits (k=10/12/14, using the spatially-local chain-walk fix "
    "from Entry 060). Star-topology large circuits were attempted and abandoned this round: a "
    "single k=10 star circuit did not finish Aer/MPS simulation within 170 seconds on Brisbane, "
    "for a structural reason different from Entry 058's chain-generation stall &mdash; a "
    "hub-and-spoke star's entangling structure is not spatially local in Hilbert space regardless "
    "of qubit ordering (unlike a chain, which can be reordered into a local walk), so Brisbane's "
    "large-circuit tier is chain-only this round. This is logged honestly rather than silently "
    "worked around, and is flagged as follow-up work.", lead))

s.append(Paragraph("A Real Infrastructure Bug Found and Fixed Along the Way", h2))
s.append(Paragraph(
    "The bell-pair generator initially produced zero circuits beyond hop-distance 7 across five "
    "consecutive generation rounds, despite the sampler correctly queuing plenty of longer-hop "
    "tasks each time. Root cause: short-hop circuits simulate fast and long-hop circuits simulate "
    "slow (more SWAPs, deeper routing), and the task list was unshuffled, so every timed run "
    "burned its entire budget on the fast short-hop tasks at the front of the list and never "
    "reached the slow long-hop tasks queued behind them. Shuffling the task order before "
    "submission fixed it immediately. Separately, the GNN training script needed its own fix "
    "once Brisbane's data pushed the combined dataset to 3,993 records: some training calls were "
    "silently hanging past the sandbox's time limit without saving a checkpoint, traced to the "
    "shell's default SIGTERM not reliably stopping a JAX process in this environment &mdash; "
    "switching to SIGKILL (<font face='Courier'>timeout -s KILL</font>) resolved it, and per-epoch "
    "timing confirmed the actual training compute was fast (roughly 1.5-3s/epoch) the whole time.", lead))

s.append(Paragraph("The Generalization Test", h2))
s.append(Paragraph(
    "Same-chip baseline across all three chips combined: MAE 1.11, R&sup2;=0.971, floor-collapse "
    "MAE 1.06 &mdash; consistent with every prior same-chip result, confirming the model architecture "
    "and training procedure are working normally on the combined data. The real test is the two "
    "genuinely cold cross-chip directions:", lead))

s.append(kv_table([
    ["Direction", "MAE / R² / floor-collapse MAE", "n_train / n_test"],
    ["Kyiv+Sherbrooke → Brisbane (cold)", "4.91 / 0.141 / 22.58", "3,241 / 752"],
    ["Brisbane → Kyiv+Sherbrooke (cold)", "3.98 / 0.794 / 18.73", "752 / 3,241"],
    ["(for comparison) Kyiv → Sherbrooke", "3.69 / 0.839 / 7.40", "2,489 / 752"],
    ["(for comparison) Sherbrooke → Kyiv", "2.98 / 0.879 / 2.45", "752 / 2,489"],
]))

s.append(Paragraph(
    "Both directions involving Brisbane are markedly worse than the established Kyiv&harr;Sherbrooke "
    "pair, and one of them is not a small degradation: R&sup2;=0.141 for "
    "Kyiv+Sherbrooke&#8594;Brisbane is close to what a model predicting the training-set mean for "
    "every test example would score. Floor-collapse error roughly tripled in both directions "
    "relative to the two-chip baseline.", lead))

s.append(Paragraph("Reading This Honestly", h2))
s.append(Paragraph(
    "This does not confirm the hoped-for story that cross-chip generalization is a general "
    "property of the model. It shows the opposite, at least so far: transfer that looked solid "
    "between Kyiv and Sherbrooke did not carry over cleanly to a third chip. Two explanations are "
    "plausible and not yet distinguished from each other. First, a genuine physical-noise-structure "
    "limit &mdash; Brisbane's calibration profile may simply differ from Kyiv/Sherbrooke's shared "
    "structure more than they differ from each other, and per-chip relative normalization (Entry "
    "048) may not be sufficient to bridge a bigger gap. Second, a distributional confound this "
    "entry introduced: Brisbane's dataset has no star-topology large circuits (the star generator "
    "stalled and was skipped this round), while both training sets it's compared against are full "
    "of them, and the floor-collapse test population is small and unevenly sized between "
    "directions (13 examples one way, 190 the other) &mdash; not nearly the statistical footing "
    "the two-chip comparisons stand on. The honest conclusion at this scale is: the original "
    "two-chip result should not be assumed to generalize to a third chip without direct testing, "
    "which is exactly what this entry did, and what it found was a real gap that needs more data "
    "and a cleaner distributional match before it can be called resolved either way.", lead))

s.append(Paragraph("Entry 061 — Conclusion", h2))
s.append(Spacer(1, 4))
s.append(Paragraph(
    "A third, independently-calibrated chip was added to the pipeline for the first time (setup, "
    "740 bell-pair circuits, 12 chain-topology large circuits, 3,993-graph combined dataset), and "
    "two real bugs were found and fixed along the way (unshuffled long-hop task starvation; a "
    "training hang traced to signal handling, not compute speed). The generalization test itself "
    "did not confirm what Entries 047-060 might have suggested to hope for: Kyiv&harr;Sherbrooke "
    "transfer does not currently extend cleanly to Brisbane. This is reported as the primary "
    "finding of this entry, not buried under the infrastructure wins, and is the clearest open "
    "item for the next round of work &mdash; likely requiring a matched-distribution Brisbane "
    "dataset (including star-topology circuits) before the question can be answered cleanly.", concl))

s.append(Spacer(1, 10))
s.append(Paragraph("Research Log — Index Addendum (Entry 061)", h2))
s.append(Paragraph(
    "Files: setup_brisbane.py, entry061_grow_brisbane.py, entry061_brisbane_large_circuits.py, "
    "entry061_build_graphs.py, entry061_train_gnn.py, entry061_cross_chip.py. Data: "
    "quantumbridge_data/entry061_brisbane_bell_dataset.json (740), "
    "entry061_brisbane_chain_dataset.json (12), entry061_graph_dataset.json (3,993 graphs), "
    "entry061_gnn_results.json, entry061_cross_chip_results.json.", cap))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="QuantumBridge Research Log — Entry 061",
                      author="Darknight (Mirr)")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, BM, W - LM - RM,
                                                         H - TM - BM, id="f")],
                                   onPage=header_footer)])
doc.build(s)
print("built", OUT)
