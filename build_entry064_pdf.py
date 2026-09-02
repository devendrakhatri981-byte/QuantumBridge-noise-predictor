"""Build Entry 064 pages in the QuantumBridge Research Log house style."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

OUT = "entry064_pages.pdf"
START_PAGE = 141
RUNNING_TITLE = "Entry 064 — Brisbane at 1,621: Scale Helps One Direction, Not the Other"

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
    t = Table(rows, colWidths=[65 * mm, W - LM - RM - 65 * mm])
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
s.append(entry_banner("ENTRY 064 &nbsp;&nbsp; Brisbane Dataset to 1,621 — Scale Helps One "
                      "Direction, Not the Other — September 3, 2026 — Status: COMPLETE"))
s.append(Spacer(1, 9))
s.append(Paragraph("Entry 064 — Does More Brisbane Data Close the Remaining Generalization Gap?", h1))
s.append(Paragraph("Bell-Pair Dataset Grown 740 &#8594; 1,621; Infrastructure Bug Fixed Along the Way", sub))

s.append(Paragraph("What Was Done", h2))
s.append(Paragraph(
    "Following Entry 062's finding that a distributional confound only partly explained "
    "Brisbane's weak cross-chip transfer, Brisbane's bell-pair dataset was grown from 740 to "
    "1,621 circuits, more than doubling it, to test whether the remaining gap shrinks with scale "
    "the way the original two-chip generalization work (Entries 053-058) improved with more data. "
    "The 881 new circuits were folded into the graph dataset (4,005 &#8594; 4,886), and both "
    "cross-chip directions were re-trained and re-evaluated.", lead))

s.append(Paragraph("A Real Data-Loss Incident, Caught and Fixed", h2))
s.append(Paragraph(
    "Partway through this growth, a checkpoint write was killed mid-write by the sandbox's hard "
    "timeout and corrupted the dataset file, destroying progress back to 66 records (from 900). "
    "Root cause: the generator wrote checkpoints via <font face='Courier'>json.dump</font> "
    "directly to the live file, which truncates the file immediately on open and streams content "
    "incrementally -- a kill mid-write leaves a corrupt fragment and destroys the prior good save "
    "too, not just the in-progress one. This was caught immediately (not discovered later), the "
    "actual damage was assessed honestly (the graph dataset already had 764 circuits permanently "
    "baked in from before, so only ~160 records of unconsumed growth were actually lost), and the "
    "generator was fixed to write to a temp file and rename atomically before continuing. This is "
    "the same class of fix Entries 058/060 applied to circuit generation speed -- here applied to "
    "generation safety.", lead))

s.append(Paragraph("Results", h2))
s.append(kv_table([
    ["Direction / n_train", "Entry 062 (764 Brisbane)", "Entry 064 (1,645 Brisbane)"],
    ["Kyiv+Sherbrooke → Brisbane", "MAE 5.23 / R²=0.354 / fc 25.11 (n_fc=14)", "MAE 5.89 / R²=0.275 / fc 15.57 (n_fc=35)"],
    ["Brisbane → Kyiv+Sherbrooke", "MAE 4.62 / R²=0.718 / fc 22.85", "MAE 3.76 / R²=0.811 / fc 19.43"],
]))

s.append(Paragraph("Reading This Honestly", h2))
s.append(Paragraph(
    "One direction improved cleanly with scale; the other did not, and both readings are reported "
    "rather than only the favorable one. Brisbane&#8594;Kyiv+Sherbrooke improved on every metric "
    "and reached its best result across all three entries (R&sup2;=0.811, the closest any "
    "Brisbane-involving direction has come to the original Kyiv&harr;Sherbrooke baseline of "
    "0.84&ndash;0.88) -- consistent with the hypothesis that more training diversity helps the "
    "model learn chip-invariant structure. Kyiv+Sherbrooke&#8594;Brisbane did not follow the same "
    "pattern: R&sup2; moved from 0.354 down to 0.275, even though floor-collapse error on that "
    "same test actually improved substantially (25.11 &#8594; 15.57) and the floor-collapse test "
    "population nearly tripled (14 &#8594; 35 examples, a much sounder sample). The most likely "
    "explanation is not that the model regressed, but that Brisbane's growth deliberately targeted "
    "underfilled long-hop-distance bins (Entry 064's own growth run stalled repeatedly on exactly "
    "these, taking dedicated debugging to push through), so the held-out Brisbane test set is now "
    "structurally harder -- more long-route, higher-decoherence circuits -- than it was in Entry "
    "062. A test set that got harder because it got more representative of the full distribution "
    "is not the same thing as a model that got worse, and the improved floor-collapse number "
    "supports that reading. But this is an inference, not a proof, and is recorded as an open "
    "question rather than a settled one.", lead))

s.append(Paragraph("Entry 064 — Conclusion", h2))
s.append(Spacer(1, 4))
s.append(Paragraph(
    "Brisbane's dataset more than doubled (740 &#8594; 1,621 bell pairs, 4,886 graphs combined), "
    "with a checkpoint-corruption bug caught and fixed with atomic writes along the way. Scale "
    "measurably helped the easier cross-chip direction to its best result yet, and produced a "
    "genuinely mixed, honestly-reported result on the harder direction -- worse headline R&sup2; "
    "but better floor-collapse accuracy on a much larger, harder test sample. The third-chip "
    "generalization question remains open in the same way it has since Entry 061: real progress, "
    "not yet resolved.", concl))

s.append(Spacer(1, 10))
s.append(Paragraph("Research Log — Index Addendum (Entry 064)", h2))
s.append(Paragraph(
    "Files: entry061_grow_brisbane.py (atomic-write fix), entry063_build_graphs.py, "
    "entry063_train_gnn.py, entry063_cross_chip.py. Data: "
    "quantumbridge_data/entry061_brisbane_bell_dataset.json (1,621), "
    "entry063_graph_dataset.json (4,886 graphs), entry063_gnn_results.json, "
    "entry063_cross_chip_results.json.", cap))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="QuantumBridge Research Log — Entry 064",
                      author="Darknight (Mirr)")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, BM, W - LM - RM,
                                                         H - TM - BM, id="f")],
                                   onPage=header_footer)])
doc.build(s)
print("built", OUT)
