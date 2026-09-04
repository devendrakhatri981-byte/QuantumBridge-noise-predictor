"""Build Entry 071-072 pages in the QuantumBridge Research Log house style."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

OUT = "entry071_pages.pdf"
START_PAGE = 147
RUNNING_TITLE = "Entry 071-072 — One Unified Uncertainty Model"

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


def kv_table(rows, col0=45):
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
s.append(entry_banner("ENTRIES 071-072 &nbsp;&nbsp; One Unified Uncertainty Model for All Three Chips — "
                      "September 4, 2026 — Status: COMPLETE"))
s.append(Spacer(1, 9))
s.append(Paragraph("Retiring the Two-Lineage Split", h1))
s.append(Paragraph("Kyiv/Sherbrooke Had Calibrated Uncertainty, Brisbane Didn't -- Now All Three Do", sub))

s.append(Paragraph("Why This Was Needed", h2))
s.append(Paragraph(
    "Since Brisbane was added (Entry 061), the deployed demo ran two separate model lineages: "
    "Kyiv/Sherbrooke used Entry 057's MC-Dropout model, trained before Brisbane existed on a "
    "2-chip, 2,329-circuit dataset, with a validated calibrated uncertainty band. Brisbane used "
    "Entry 065/069's combined generalist model, trained without dropout (so any inference-time-only "
    "dropout applied to it would have produced an uncertainty-shaped number that was never "
    "validated -- the project has been careful throughout not to do this). The result was an honest "
    "but visible asymmetry: two different models, two different guarantees, patched around with "
    "explicit labeling in the demo rather than actually resolved.", lead))

s.append(Paragraph("Entry 071 — One Model, Trained With Dropout, On All Current Data", h2))
s.append(Paragraph(
    "A single MC-Dropout-enabled model (same architecture, same per-chip normalization, same "
    "floor-collapse up-weighting as every GNN since Entry 048/051) was trained on the full current "
    "entry068_graph_dataset.json -- 5,690 circuits across Kyiv, Sherbrooke, and Brisbane together, "
    "the balanced post-Entry-067 dataset. Dropout (rate=0.15) is active both during training and at "
    "inference (T=20 stochastic passes), matching Entry 057's original methodology exactly -- only "
    "the underlying data and chip count changed.", lead))

s.append(Paragraph("Calibration Check — Genuine Held-Out Validation", h2))
s.append(kv_table([
    ["", "MAE", "R²", "mean predicted std", "corr(std, |error|)"],
    ["Held-out test (80/20 CV)", "1.38", "0.964", "0.81 pts", "0.553"],
], col0=52))
s.append(Paragraph(
    "This calibration run used a separate model trained on only 80% of the data (never seeing the "
    "held-out 20%), exactly mirroring Entry 057's original validation protocol -- distinct from the "
    "deployed model itself, which is trained on all data for maximum production accuracy. The "
    "positive, meaningfully-sized correlation (0.553) between predicted uncertainty and actual error "
    "confirms the model knows when it's less confident, not just that it produces a number. Per-chip "
    "breakdown on the held-out set: Sherbrooke showed both the highest mean predicted std (1.05pts) "
    "and the highest MAE (1.56) -- the model correctly flags its own weakest chip.", lead))

s.append(Paragraph("Entry 072 — Re-Score and Deploy", h2))
s.append(Paragraph(
    "All 24,003 precomputed qubit pairs across all three chips were re-scored with the deployed "
    "unified model, replacing both the old Entry 057 lookup (Kyiv/Sherbrooke, 2-chip-era) and the "
    "Entry 070 lookup (Brisbane, point-estimate-only) with one consistent table. The live demo's "
    "special-cased \"no uncertainty band yet\" messaging for Brisbane was removed -- there is no "
    "longer a code path where any chip lacks an uncertainty estimate. The validation-note footer "
    "was rewritten to honestly describe what this deployment model is (a generalist trained on data "
    "pooled from all three chips) versus what the separate leave-one-chip-out experiments (Entries "
    "047-068) tested (true cold transfer to an entirely unseen chip) -- the two are not the same "
    "claim, and the demo now says so for every chip rather than singling out Brisbane.", lead))

s.append(Paragraph("Entries 071-072 — Conclusion", h2))
s.append(Spacer(1, 4))
s.append(Paragraph(
    "The project now has one deployed model, one validated uncertainty mechanism, and one honest "
    "validation note that applies uniformly across Kyiv, Sherbrooke, and Brisbane -- replacing two "
    "parallel model lineages with different guarantees. This closes the first item of the three-part "
    "roadmap agreed for this phase (uncertainty parity, a fourth never-seen-chip test, then "
    "productization); the fourth-chip generalization test is next.", concl))

s.append(Spacer(1, 10))
s.append(Paragraph("Research Log — Index Addendum (Entries 071-072)", h2))
s.append(Paragraph(
    "Files: entry071_mc_dropout.py, entry071_deploy_train.py, entry072_score_all_pairs.py. Data: "
    "quantumbridge_data/entry071_deploy_params.json (deployable weights, dropout_rate=0.15), "
    "entry072_lookup_with_uncertainty.json (rescored demo lookup, all 3 chips with calibrated "
    "uncertainty).", cap))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="QuantumBridge Research Log — Entries 071-072",
                      author="Darknight (Mirr)")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, BM, W - LM - RM,
                                                         H - TM - BM, id="f")],
                                   onPage=header_footer)])
doc.build(s)
print("built", OUT)
