"""Build Entry 062 pages in the QuantumBridge Research Log house style."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

OUT = "entry062_pages.pdf"
START_PAGE = 139
RUNNING_TITLE = "Entry 062 — Star Gap Closed for Brisbane, Confound Partly Confirmed"

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
s.append(entry_banner("ENTRY 062 &nbsp;&nbsp; Brisbane Star-Topology Gap Closed — "
                      "Distributional Confound Partly Confirmed, Real Gap Remains — "
                      "September 1, 2026 — Status: COMPLETE"))
s.append(Spacer(1, 9))
s.append(Paragraph("Entry 062 — Closing the Star-Topology Gap Flagged in Entry 061", h1))
s.append(Paragraph("Does a Matched Circuit-Topology Mix Close the Brisbane Generalization Gap?", sub))

s.append(Paragraph("Background", h2))
s.append(Paragraph(
    "Entry 061 found that Kyiv&harr;Sherbrooke cross-chip transfer did not extend cleanly to "
    "Brisbane, and flagged two live explanations without distinguishing them: a genuine physical "
    "difference in Brisbane's noise structure, or a distributional confound &mdash; Brisbane's "
    "dataset had zero star-topology large circuits (that generator stalled) while both chips it "
    "was compared against are full of them. This entry closes that confound directly by fixing "
    "star-topology generation for Brisbane and re-running the exact same test.", lead))

s.append(Paragraph("The Star-Topology Fix", h2))
s.append(Paragraph(
    "Entry 061 diagnosed the star stall as different in kind from Entry 058/060's chain stall: a "
    "chain's entanglement structure can be made spatially local by walking the coupling graph in "
    "order (Entry 060's fix), but a star's hub-and-spoke structure is non-local in Hilbert space "
    "regardless of qubit ordering &mdash; the hub is entangled with every spoke, and reordering "
    "the spokes doesn't change that. What can be controlled is physical distance: if the k-1 "
    "target qubits are chosen near the control qubit on the real coupling map (nearest by BFS "
    "distance, with light randomization among near-tied candidates) rather than uniformly at "
    "random across the whole 127-qubit chip, the *routed* circuit needs few or no SWAPs, keeping "
    "it shallow enough for Aer's MPS simulator. Measured effect on a single Brisbane k=10 star: "
    "did not finish in 170s with fully random targets; 0.5s with local targets. All three sizes "
    "(k=10/12/14, 4 circuits each) generated successfully, in under 90 seconds total.", lead))

s.append(Paragraph("Re-Running the Generalization Test", h2))
s.append(Paragraph(
    "The 12 new star circuits were folded into the graph dataset (3,993 &#8594; 4,005), and both "
    "cross-chip directions from Entry 061 were re-run unchanged otherwise:", lead))

s.append(kv_table([
    ["Direction", "Entry 061 (chain-only Brisbane)", "Entry 062 (chain+star Brisbane)"],
    ["Kyiv+Sherbrooke → Brisbane", "MAE 4.91 / R²=0.141 / fc 22.58", "MAE 5.23 / R²=0.354 / fc 25.11"],
    ["Brisbane → Kyiv+Sherbrooke", "MAE 3.98 / R²=0.794 / fc 18.73", "MAE 4.62 / R²=0.718 / fc 22.85"],
    ["(reference) Kyiv → Sherbrooke", "MAE 3.69 / R²=0.839 / fc 7.40", "unchanged"],
]))

s.append(Paragraph("Reading This Honestly", h2))
s.append(Paragraph(
    "The distributional-confound hypothesis is partly confirmed, not fully. Closing the "
    "star-topology gap measurably improved the harder direction &mdash; R&sup2; for "
    "Kyiv+Sherbrooke&#8594;Brisbane more than doubled, from 0.141 to 0.354. That is real evidence "
    "that at least part of Entry 061's poor result was an artifact of comparing mismatched circuit "
    "mixes, not pure physics. But 0.354 is still far below the 0.84&ndash;0.88 range the "
    "Kyiv&harr;Sherbrooke pair achieves, and floor-collapse error actually got slightly worse "
    "(22.58 &#8594; 25.11), not better. The reverse direction moved the wrong way too (0.794 "
    "&#8594; 0.718), though on a small 764-circuit training set that's plausibly within run-to-run "
    "noise rather than a real regression. Taken together: matching the circuit-topology mix "
    "closes part of the gap but not most of it. There is a real, substantial generalization "
    "difficulty specific to Brisbane that a distributional confound does not fully explain, and "
    "that remains open. The most likely next lever, per the master plan already agreed with the "
    "user, is scale: Brisbane's dataset is still roughly a quarter the size of Kyiv's or "
    "Sherbrooke's individually, and growing it further before drawing firmer conclusions is the "
    "planned next step rather than treating this result as final.", lead))

s.append(Paragraph("Entry 062 — Conclusion", h2))
s.append(Spacer(1, 4))
s.append(Paragraph(
    "Star-topology generation for Brisbane is fixed (local-target selection instead of uniform "
    "random, closing a real generator gap the same way Entry 060 closed Sherbrooke's chain gap), "
    "and the fix was used immediately to test rather than assumed to help: it measurably improved "
    "one direction of three-chip transfer and did not resolve the other. The honest state of the "
    "third-chip generalization question is unchanged in kind from Entry 061 -- a real gap exists "
    "that isn't fully explained by dataset composition -- but the evidence is now cleaner, since "
    "circuit-topology mix is no longer a confound sitting on top of it.", concl))

s.append(Spacer(1, 10))
s.append(Paragraph("Research Log — Index Addendum (Entry 062)", h2))
s.append(Paragraph(
    "Files: entry061_brisbane_large_circuits.py (local_star_targets fix), entry061b_build_graphs.py, "
    "entry061b_train_gnn.py, entry061b_cross_chip.py. Data: "
    "quantumbridge_data/entry061_brisbane_star_dataset.json (12), "
    "entry061b_graph_dataset.json (4,005 graphs), entry061b_gnn_results.json, "
    "entry061b_cross_chip_results.json.", cap))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="QuantumBridge Research Log — Entry 062",
                      author="Darknight (Mirr)")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, BM, W - LM - RM,
                                                         H - TM - BM, id="f")],
                                   onPage=header_footer)])
doc.build(s)
print("built", OUT)
