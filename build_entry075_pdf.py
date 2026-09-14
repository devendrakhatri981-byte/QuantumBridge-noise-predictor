"""Build Entry 075 page in the QuantumBridge Research Log house style."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

OUT = "entry075_pages.pdf"
START_PAGE = 152
RUNNING_TITLE = "Entry 075 — Wiring the Demo to the Live API"

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
    canvas.drawString(LM, y - 23, "QuantumBridge — Phase 3 (Productization)")
    canvas.restoreState()


def entry_banner(text):
    t = Table([[Paragraph(text, banner)]], colWidths=[W - LM - RM], rowHeights=[16])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY),
                           ("LEFTPADDING", (0, 0), (-1, -1), 7),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


s = []
s.append(entry_banner("ENTRY 075 &nbsp;&nbsp; Wiring the Demo to the Live API — "
                      "September 5, 2026 — Status: COMPLETE"))
s.append(Spacer(1, 9))
s.append(Paragraph("A Progressive Upgrade, Not a Hard Cutover", h1))
s.append(Paragraph("The Demo Works Identically Today and Upgrades Automatically Once Deployed", sub))

s.append(Paragraph("What Changed", h2))
s.append(Paragraph(
    "quantumbridge_live_demo.html now tries a live request to the Entry 074 API first (4-second "
    "timeout via AbortController), and only falls back to the precomputed 24,003-row table if the "
    "API is unreachable or a base URL hasn't been configured. Every result is now labeled with an "
    "explicit badge -- green \"live prediction\" or grey \"cached / precomputed\" -- so a viewer "
    "always knows which path served their answer, rather than the two being silently "
    "indistinguishable. Because API_BASE_URL defaults to an empty string, the demo's behavior is "
    "completely unchanged until someone deploys the API and sets that one constant -- this is a "
    "progressive upgrade wired in ahead of time, not a dependency that had to ship together with a "
    "live deployment.", lead))

s.append(Paragraph("CORS", h2))
s.append(Paragraph(
    "The API (Entry 074) added CORSMiddleware with a permissive allow_origins=[\"*\"] policy, "
    "verified locally with a real preflight OPTIONS request carrying the GitHub Pages origin header "
    "-- confirmed the server responds with the correct access-control-allow-origin and "
    "access-control-allow-methods headers. This is deliberately permissive since /predict is a "
    "public, read-only, no-auth endpoint with no user data; it should be tightened before adding "
    "anything that isn't purely read-only.", lead))

s.append(Paragraph("What Ground Truth Display Still Relies On", h2))
s.append(Paragraph(
    "Live requests don't run a fresh Aer simulation per pair (too slow for interactive use), so the "
    "\"Simulated hardware result\" field still comes from the local precomputed table when "
    "available, merged in alongside the live v4.1/GNN predictions -- a live prediction can still "
    "show real ground truth if that specific pair happened to be simulated during training or "
    "evaluation, it just won't trigger a brand-new simulation on demand.", lead))

s.append(Paragraph("Entry 075 — Conclusion", h2))
s.append(Spacer(1, 4))
s.append(Paragraph(
    "The frontend is ready for a live backend the moment one exists at a real URL -- deployment "
    "itself (choosing and configuring a host) remains the user's step, documented in "
    "api/README.md's new \"Wiring up the live demo\" section. Nothing about the current, already-"
    "deployed demo changes until that URL is set.", concl))

s.append(Spacer(1, 10))
s.append(Paragraph("Research Log — Index Addendum (Entry 075)", h2))
s.append(Paragraph(
    "Files: quantumbridge_live_demo.html (API_BASE_URL, fetchLive(), badge system), "
    "api/main.py (CORS middleware), api/README.md (wiring instructions).", cap))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="QuantumBridge Research Log — Entry 075",
                      author="Darknight (Mirr)")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, BM, W - LM - RM,
                                                         H - TM - BM, id="f")],
                                   onPage=header_footer)])
doc.build(s)
print("built", OUT)
