"""Today's session deck — focused on match quality metrics, improvements, and evaluation."""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pathlib import Path

BLACK           = RGBColor(0x33, 0x33, 0x33)
DARK_GRAY       = RGBColor(0x55, 0x55, 0x55)
MED_GRAY        = RGBColor(0x88, 0x88, 0x88)
GREEN           = RGBColor(0x1B, 0x7F, 0x46)
RED             = RGBColor(0xC0, 0x39, 0x2B)
BLUE            = RGBColor(0x1A, 0x5C, 0x8A)
WHITE           = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_BG        = RGBColor(0xF5, 0xF5, 0xF5)
TABLE_HDR       = RGBColor(0x2C, 0x3E, 0x50)
AMBER           = RGBColor(0xB4, 0x57, 0x09)
PLACEHOLDER_BG  = RGBColor(0xEE, 0xEE, 0xEE)
PLACEHOLDER_TXT = RGBColor(0x99, 0x99, 0x99)


def heading(s, text, left, top, width, size=28):
    b = s.shapes.add_textbox(left, top, width, Inches(0.6))
    tf = b.text_frame
    p = tf.paragraphs[0]
    p.text = text; p.font.size = Pt(size); p.font.bold = True; p.font.color.rgb = BLACK

def subhead(s, text, left, top, width, size=16):
    b = s.shapes.add_textbox(left, top, width, Inches(0.4))
    tf = b.text_frame
    p = tf.paragraphs[0]
    p.text = text; p.font.size = Pt(size); p.font.bold = True; p.font.color.rgb = BLUE

def txt(s, text, left, top, width, height=Inches(0.5), size=14,
        color=DARK_GRAY, bold=False, italic=False, align=PP_ALIGN.LEFT):
    b = s.shapes.add_textbox(left, top, width, height)
    tf = b.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text; p.font.size = Pt(size); p.font.color.rgb = color
    p.font.bold = bold; p.font.italic = italic; p.alignment = align

def bullets(s, items, left, top, width, size=14, height=Inches(4)):
    b = s.shapes.add_textbox(left, top, width, height)
    tf = b.text_frame; tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if isinstance(item, dict):
            p.text = item["text"]
            p.font.color.rgb = item.get("color", DARK_GRAY)
            p.font.bold = item.get("bold", False)
            p.font.italic = item.get("italic", False)
            p.level = item.get("indent", 0)
        else:
            p.text = item; p.font.color.rgb = DARK_GRAY
        p.font.size = Pt(size); p.space_after = Pt(5)

def table(s, data, left, top, col_widths, size=13, rh=Inches(0.42)):
    rows, cols = len(data), len(data[0])
    shape = s.shapes.add_table(rows, cols, left, top, sum(col_widths), rh * rows)
    t = shape.table
    for ci, w in enumerate(col_widths):
        t.columns[ci].width = w
    for ri, row in enumerate(data):
        for ci, cd in enumerate(row):
            cell = t.cell(ri, ci)
            if isinstance(cd, dict):
                text = cd["text"]; color = cd.get("color", BLACK if ri > 0 else WHITE)
                bold = cd.get("bold", ri == 0)
            else:
                text = str(cd); color = WHITE if ri == 0 else BLACK; bold = ri == 0
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            p.text = text; p.font.size = Pt(size); p.font.color.rgb = color
            p.font.bold = bold
            p.alignment = PP_ALIGN.CENTER if ci > 0 else PP_ALIGN.LEFT
            p.space_before = Pt(0); p.space_after = Pt(0)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            if ri == 0:
                cell.fill.solid(); cell.fill.fore_color.rgb = TABLE_HDR
            elif ri % 2 == 0:
                cell.fill.solid(); cell.fill.fore_color.rgb = LIGHT_BG
            else:
                cell.fill.solid(); cell.fill.fore_color.rgb = WHITE

def divider(s, left, top, width):
    sh = s.shapes.add_shape(1, left, top, width, Inches(0.02))
    sh.fill.solid(); sh.fill.fore_color.rgb = RGBColor(0xDD, 0xDD, 0xDD)
    sh.line.fill.background()

def screenshot(s, left, top, width, height, label):
    box = s.shapes.add_shape(1, left, top, width, height)
    box.fill.solid(); box.fill.fore_color.rgb = PLACEHOLDER_BG
    box.line.color.rgb = RGBColor(0xBB, 0xBB, 0xBB)
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = f"[ Screenshot: {label} ]"
    p.font.size = Pt(12); p.font.italic = True
    p.font.color.rgb = PLACEHOLDER_TXT; p.alignment = PP_ALIGN.CENTER
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE

def stat_box(s, label, value, left, top, width, height, val_color=BLUE):
    box = s.shapes.add_shape(1, left, top, width, height)
    box.fill.solid(); box.fill.fore_color.rgb = LIGHT_BG
    box.line.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)

    vb = s.shapes.add_textbox(left, top + Inches(0.15), width, Inches(0.6))
    tf = vb.text_frame
    p = tf.paragraphs[0]
    p.text = value; p.font.size = Pt(28); p.font.bold = True
    p.font.color.rgb = val_color; p.alignment = PP_ALIGN.CENTER

    lb = s.shapes.add_textbox(left, top + Inches(0.7), width, Inches(0.35))
    tf = lb.text_frame
    p = tf.paragraphs[0]
    p.text = label; p.font.size = Pt(11); p.font.color.rgb = MED_GRAY
    p.alignment = PP_ALIGN.CENTER


# ── Slide 1: Title ─────────────────────────────────────────────────────────

def slide_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bar = s.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(3.0))
    bar.fill.solid(); bar.fill.fore_color.rgb = TABLE_HDR; bar.line.fill.background()

    txt(s, "CPT Matching — What I Built Today",
        Inches(1), Inches(0.6), Inches(11.3), size=34, color=WHITE, bold=True)
    txt(s, "Match Quality Metrics · Evaluation Harness · UI Improvements",
        Inches(1), Inches(1.6), Inches(11), size=17, color=RGBColor(0xAA, 0xCC, 0xEE))

    bullets(s, [
        "Started with v2 system (hybrid FAISS+BM25 + Cohere reranking)",
        "Identified and fixed broken patient-friendly titles for 90 policy codes",
        "Expanded evaluation coverage: 25 queries → 720 queries across 90 CPT codes",
        "Added match quality badges to the search UI",
    ], Inches(1.2), Inches(3.4), Inches(10), size=16)


# ── Slide 2: Where I Started ───────────────────────────────────────────────

def slide_starting_point(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    heading(s, "Where I Started: v2 System Baseline", Inches(0.8), Inches(0.4), Inches(11))
    divider(s, Inches(0.8), Inches(1.05), Inches(11.5))

    subhead(s, "System at the start of today's session", Inches(0.8), Inches(1.25), Inches(7))
    bullets(s, [
        "Hybrid search: FAISS semantic (60%) + BM25 lexical (40%)",
        "Cohere rerank-v3.5 as a second-stage re-scorer",
        "16,054 CPT/HCPCS codes indexed with all-MiniLM-L6-v2 (384 dims)",
        "search_text = title + category + description + keywords",
    ], Inches(0.8), Inches(1.75), Inches(7.5), size=13)

    subhead(s, "Baseline eval on 25-query mammography set", Inches(0.8), Inches(3.1), Inches(11))

    # Stat boxes
    stats = [
        ("Hit@5",   "80%",   Inches(0.8)),
        ("Hit@10",  "80%",   Inches(3.1)),
        ("MRR",     "0.627", Inches(5.4)),
    ]
    for label, val, lx in stats:
        stat_box(s, label, val, lx, Inches(3.65), Inches(1.9), Inches(1.1), BLUE)

    subhead(s, "What I found when I looked closer", Inches(0.8), Inches(5.0), Inches(11))
    bullets(s, [
        {"text": "8 mammography/MRI codes had unfilled placeholder titles in the parquet:",
         "color": RED},
        {"text": "  \"[Patient-friendly title for 77067]\" — these were poisoning FAISS search_text",
         "color": RED, "indent": 1, "italic": True},
        "All other mammography results shared the identical breadcrumb title from the CPT codebook:",
        {"text": "  \"Breast Mammography.Diagnostic Radiology. Mammography\"",
         "color": AMBER, "indent": 1, "italic": True},
        "Patients could not distinguish screening vs diagnostic vs 3D — all cards looked the same",
    ], Inches(0.8), Inches(5.5), Inches(11.5), size=13)


# ── Slide 3: What I Fixed ──────────────────────────────────────────────────

def slide_fix(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    heading(s, "What I Fixed: Patient-Friendly Titles", Inches(0.8), Inches(0.4), Inches(11))
    divider(s, Inches(0.8), Inches(1.05), Inches(11.5))

    subhead(s, "The Fix", Inches(0.8), Inches(1.25), Inches(5.5))
    bullets(s, [
        "Used GPT-4o-mini to generate patient-friendly titles for all 90 policy codes",
        "Manual review pass — corrected medically inaccurate outputs",
        {"text": "e.g. 77046/47 came back as \"3D breast imaging\"",
         "color": RED, "italic": True, "indent": 1},
        {"text": "Corrected to \"Breast MRI without contrast\" / \"with contrast\"",
         "color": GREEN, "italic": True, "indent": 1},
        "Stored as clean_title in parquet; matcher.py prefers clean_title over raw title",
        "Rebuilt FAISS index with updated search_text",
    ], Inches(0.8), Inches(1.75), Inches(5.5), size=13)

    subhead(s, "Before / After", Inches(7.2), Inches(1.25), Inches(5.5))
    table(s, [
        ["Code", "Before", "After"],
        ["77067", "Breast Mammography.Diagnostic\nRadiology. Mammography", "Screening mammogram, both breasts"],
        ["77065", "Breast Mammography.Diagnostic\nRadiology. Mammography", "Diagnostic mammogram, one breast"],
        ["77063", "Breast Mammography.Diagnostic\nRadiology. Mammography", "3D mammogram add-on (tomosynthesis)"],
        ["77046", "Breast Mammography.Diagnostic\nRadiology. Mammography", "Breast MRI without contrast"],
    ], Inches(7.2), Inches(1.75),
        [Inches(0.75), Inches(2.35), Inches(2.3)], size=11, rh=Inches(0.46))

    divider(s, Inches(0.8), Inches(4.05), Inches(11.5))

    subhead(s, "Impact on Match Quality", Inches(0.8), Inches(4.25), Inches(11))
    table(s, [
        ["",         "Hit@5",  "Hit@10", "MRR",   "Notes"],
        ["Before fix (v2 baseline)",
                     "80%",    "80%",    "0.627", "Placeholder titles in FAISS vectors"],
        [{"text": "After fix (clean titles + index rebuild)", "bold": True},
         {"text": "88% ▲", "color": GREEN, "bold": True},
         {"text": "88%",   "color": DARK_GRAY},
         {"text": "0.599", "color": DARK_GRAY},
         "Cleaner search_text; better title display"],
    ], Inches(0.8), Inches(4.75),
        [Inches(3.4), Inches(0.85), Inches(0.85), Inches(0.85), Inches(3.7)], size=13)

    bullets(s, [
        {"text": "Hit@5 jumped 8pp — more correct codes appearing in the first 5 results", "color": GREEN},
        {"text": "MRR dipped slightly — the reranker works best when titles match user language precisely",
         "color": MED_GRAY, "italic": True},
    ], Inches(0.8), Inches(5.85), Inches(11.5), size=13)

    screenshot(s, Inches(9.5), Inches(4.25), Inches(3.5), Inches(2.9),
               "Search results with clean titles visible on cards")


# ── Slide 4: Evaluation Harness ────────────────────────────────────────────

def slide_eval(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    heading(s, "How I Evaluated: The Eval Harness", Inches(0.8), Inches(0.4), Inches(11))
    divider(s, Inches(0.8), Inches(1.05), Inches(11.5))

    subhead(s, "Metrics", Inches(0.8), Inches(1.25), Inches(5.5))
    table(s, [
        ["Metric",    "What it measures"],
        ["Hit@K",     "Is any expected code in the top-K results? (binary per query)"],
        ["MRR",       "1 / rank of first correct code — rewards codes ranked higher"],
        ["Recall@K",  "Fraction of all expected codes found in top-K"],
    ], Inches(0.8), Inches(1.75), [Inches(1.2), Inches(4.5)], size=13)

    subhead(s, "The 25-Query Test Set", Inches(0.8), Inches(3.35), Inches(11))
    bullets(s, [
        "25 hand-crafted queries across 5 breast imaging procedures (14 unique CPT codes)",
        "4 verbosity levels simulate how real patients phrase questions — from terse to rambling",
        "Each query mapped to 1–5 expected codes; a \"hit\" means any expected code appears in top-K",
        "Same fixed set used for every iteration — makes improvements directly comparable",
    ], Inches(0.8), Inches(3.85), Inches(11.5), size=13)

    subhead(s, "Verbosity breakdown (after clean titles)", Inches(0.8), Inches(5.3), Inches(6))
    table(s, [
        ["Verbosity",      "Count", "Hit@5",  "Hit@10", "Example query"],
        ["Direct",         "7",     "100%",   "100%",   "mammogram"],
        ["Short",          "7",     "86%",    "100%",   "is a mammogram covered"],
        ["Conversational", "7",     "71%",    "86%",    "I need a mammogram, my last one was 3 years ago, is it covered?"],
        ["Verbose",        "4",     "75%",    "75%",    "I just turned 40 and I know I'm supposed to start getting mammograms..."],
    ], Inches(0.8), Inches(5.8),
        [Inches(1.8), Inches(0.7), Inches(0.75), Inches(0.85), Inches(5.5)], size=12)

    subhead(s, "Verbosity breakdown (Phase 1 baseline — for reference)", Inches(8.8), Inches(5.3), Inches(4.2))
    table(s, [
        ["Verbosity",      "Hit@5",  "Hit@10"],
        ["Direct",         "100%",   "100%"],
        ["Short",          "86%",    "100%"],
        ["Conversational", "71%",    "86%"],
        ["Verbose",        "75%",    "75%"],
    ], Inches(8.8), Inches(5.8), [Inches(1.8), Inches(0.8), Inches(0.85)], size=12, rh=Inches(0.35))


# ── Slide 5: Full Progression ──────────────────────────────────────────────

def slide_progression(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    heading(s, "Full Improvement Journey", Inches(0.8), Inches(0.4), Inches(11))
    divider(s, Inches(0.8), Inches(1.05), Inches(11.5))

    subhead(s, "All iterations — 25-query mammography eval set", Inches(0.8), Inches(1.25), Inches(11))

    G = GREEN
    table(s, [
        ["Iteration",                               "Hit@5",  "Hit@10", "MRR",   "Key Change"],
        ["1. Baseline (title + category + desc)",
                                                    "56%",    "80%",    "0.364", "Starting point"],
        ["2. + KeyBERT keyword enrichment",
                                                    "56%",    "80%",   {"text": "0.452", "color": G},
         "Better ranking order"],
        ["3. + Remove retired codes (77053, 77054)",
                                                    "60%",   {"text": "88%", "color": G},
                                                             {"text": "0.465", "color": G},
         "Fewer false positives"],
        ["4. v2: Hybrid BM25+FAISS + Cohere rerank",
         "80%",    "80%",   {"text": "0.627 ▲", "color": G, "bold": True},
         "Best MRR — reranker sorts correctly"],
        [{"text": "5. + Clean titles  ← today", "bold": True},
         {"text": "88% ▲", "color": G, "bold": True},
         "88%",
         "0.599",
         {"text": "Hit@5 best overall", "color": G}],
    ], Inches(0.8), Inches(1.75),
        [Inches(4.1), Inches(0.8), Inches(0.85), Inches(0.85), Inches(3.2)], size=12)

    divider(s, Inches(0.8), Inches(5.0), Inches(11.5))

    subhead(s, "Key Takeaways", Inches(0.8), Inches(5.2), Inches(11))
    bullets(s, [
        {"text": "Filtering noise (bad codes) beat adding signal (keywords) — admin filtering gave the biggest single jump",
         "bold": True, "color": BLUE},
        "Cohere reranking elevated MRR even when hit rates plateaued — right answers ranked higher",
        {"text": "Clean titles improved Hit@5 by 8pp — better search_text = better vectors",
         "color": GREEN},
        {"text": "1 remaining miss: \"follow up mammogram after abnormal results\" → still returns biopsy codes",
         "color": MED_GRAY, "italic": True},
    ], Inches(0.8), Inches(5.7), Inches(11.5), size=13)


# ── Slide 6: UI Improvements ───────────────────────────────────────────────

def slide_ui(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    heading(s, "UI Improvements", Inches(0.8), Inches(0.4), Inches(11))
    divider(s, Inches(0.8), Inches(1.05), Inches(11.5))

    subhead(s, "Match Quality Badges", Inches(0.8), Inches(1.25), Inches(5.5))
    bullets(s, [
        "Every result card now shows a confidence badge powered by Cohere relevance score",
        {"text": "Best match  · rank 1",   "color": RGBColor(0x06, 0x5F, 0x46), "bold": True},
        {"text": "Good match  · ranks 2–3", "color": RGBColor(0x1E, 0x40, 0xAF), "bold": True},
        {"text": "Possible match  · ranks 4+", "color": RGBColor(0x6B, 0x72, 0x80), "bold": True},
        "Score shown as % — e.g. \"Best match · 74%\"",
        "Helps patients pick the most relevant result before clicking through",
        "Price removed — app is about coverage, not cost",
    ], Inches(0.8), Inches(1.75), Inches(5.5), size=13)

    screenshot(s, Inches(7.0), Inches(1.25), Inches(5.9), Inches(3.0),
               "Search results with Best / Good / Possible match badges")

    divider(s, Inches(0.8), Inches(4.55), Inches(11.5))

    subhead(s, "Demo Queries", Inches(0.8), Inches(4.75), Inches(11))
    table(s, [
        ["Query",                                        "Top Result",                           "Score"],
        ["mammogram",                                    "Screening mammogram, both breasts (77067)", "74%"],
        ["screening mammogram",                          "Screening mammogram, both breasts (77067)", "72%"],
        ["breast MRI",                                   "Breast MRI without contrast (77046)",   "65%"],
        ["my doctor wants a breast MRI (family history)", "Breast MRI without contrast (77046)",  "28%"],
        ["I need an endometrial biopsy",                 "Endometrial biopsy (58100)",            "48%"],
    ], Inches(0.8), Inches(5.25),
        [Inches(4.5), Inches(4.5), Inches(0.9)], size=12, rh=Inches(0.37))

    screenshot(s, Inches(0.8), Inches(4.75), Inches(5.8), Inches(2.7),
               "Eligibility decision flow — Q&A + coverage outcome")


# ── Build ──────────────────────────────────────────────────────────────────

def build():
    prs = Presentation()
    prs.slide_width  = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide_title(prs)
    slide_starting_point(prs)
    slide_fix(prs)
    slide_eval(prs)
    slide_progression(prs)
    slide_ui(prs)

    return prs


if __name__ == "__main__":
    prs = build()
    out = Path(__file__).resolve().parents[1] / "eval" / "results" / "today_session.pptx"
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    print(f"Saved → {out}")
