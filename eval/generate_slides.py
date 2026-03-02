"""Generate presentation slides for the Medical Cost Coverage Estimator project."""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pathlib import Path


# ── Palette ────────────────────────────────────────────────────────────────
BLACK       = RGBColor(0x33, 0x33, 0x33)
DARK_GRAY   = RGBColor(0x55, 0x55, 0x55)
MED_GRAY    = RGBColor(0x88, 0x88, 0x88)
GREEN       = RGBColor(0x1B, 0x7F, 0x46)
RED         = RGBColor(0xC0, 0x39, 0x2B)
BLUE        = RGBColor(0x1A, 0x5C, 0x8A)
LIGHT_BLUE  = RGBColor(0xEB, 0xF3, 0xFB)
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_BG    = RGBColor(0xF5, 0xF5, 0xF5)
TABLE_HDR   = RGBColor(0x2C, 0x3E, 0x50)
ACCENT      = RGBColor(0x1A, 0x5C, 0x8A)
PLACEHOLDER_BG  = RGBColor(0xEE, 0xEE, 0xEE)
PLACEHOLDER_TXT = RGBColor(0x99, 0x99, 0x99)


# ── Helpers ─────────────────────────────────────────────────────────────────

def add_heading(slide, text, left, top, width, font_size=28):
    txBox = slide.shapes.add_textbox(left, top, width, Inches(0.6))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = True
    p.font.color.rgb = BLACK


def add_subheading(slide, text, left, top, width, font_size=17):
    txBox = slide.shapes.add_textbox(left, top, width, Inches(0.4))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = True
    p.font.color.rgb = BLUE


def add_text(slide, text, left, top, width, height=Inches(4),
             font_size=14, color=DARK_GRAY, bold=False, italic=False, align=PP_ALIGN.LEFT):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.italic = italic
    p.alignment = align


def add_bullets(slide, items, left, top, width, font_size=14, height=Inches(4)):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True

    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()

        if isinstance(item, dict):
            text   = item["text"]
            color  = item.get("color", DARK_GRAY)
            bold   = item.get("bold", False)
            indent = item.get("indent", 0)
            italic = item.get("italic", False)
        else:
            text, color, bold, indent, italic = item, DARK_GRAY, False, 0, False

        p.text = text
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.bold = bold
        p.font.italic = italic
        p.space_after = Pt(5)
        p.level = indent


def make_table(slide, data, left, top, col_widths, font_size=13, row_height=Inches(0.42)):
    rows, cols = len(data), len(data[0])
    width = sum(col_widths)

    shape = slide.shapes.add_table(rows, cols, left, top, width, row_height * rows)
    table = shape.table

    for ci, w in enumerate(col_widths):
        table.columns[ci].width = w

    for ri, row_data in enumerate(data):
        for ci, cell_data in enumerate(row_data):
            cell = table.cell(ri, ci)

            if isinstance(cell_data, dict):
                text  = cell_data["text"]
                color = cell_data.get("color", BLACK if ri > 0 else WHITE)
                bold  = cell_data.get("bold", ri == 0)
            else:
                text  = str(cell_data)
                color = WHITE if ri == 0 else BLACK
                bold  = ri == 0

            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            p.text = text
            p.font.size = Pt(font_size)
            p.font.color.rgb = color
            p.font.bold = bold
            p.alignment = PP_ALIGN.CENTER if ci > 0 else PP_ALIGN.LEFT
            p.space_before = Pt(0)
            p.space_after  = Pt(0)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE

            if ri == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = TABLE_HDR
            elif ri % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT_BG
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE


def add_divider(slide, left, top, width):
    shape = slide.shapes.add_shape(1, left, top, width, Inches(0.02))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0xDD, 0xDD, 0xDD)
    shape.line.fill.background()


def add_screenshot_placeholder(slide, left, top, width, height, label):
    """Grey box with a label — replace with actual screenshot before presenting."""
    box = slide.shapes.add_shape(1, left, top, width, height)
    box.fill.solid()
    box.fill.fore_color.rgb = PLACEHOLDER_BG
    box.line.color.rgb = RGBColor(0xBB, 0xBB, 0xBB)

    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = f"[ Screenshot: {label} ]"
    p.font.size = Pt(13)
    p.font.italic = True
    p.font.color.rgb = PLACEHOLDER_TXT
    p.alignment = PP_ALIGN.CENTER
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE


# ── Slides ───────────────────────────────────────────────────────────────────

def slide_title(prs):
    """Slide 1 — Title."""
    s = prs.slides.add_slide(prs.slide_layouts[6])

    # Background accent bar
    bar = s.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(2.8))
    bar.fill.solid()
    bar.fill.fore_color.rgb = TABLE_HDR
    bar.line.fill.background()

    add_text(s, "Medical Cost Coverage Estimator",
             Inches(1), Inches(0.55), Inches(11.3),
             font_size=38, color=WHITE, bold=True)
    add_text(s, "Medi-Cal Coverage Intelligence — DATASCI 210",
             Inches(1), Inches(1.45), Inches(11),
             font_size=18, color=RGBColor(0xAA, 0xCC, 0xEE))

    add_bullets(s, [
        "Semantic procedure search over 16,000+ CPT/HCPCS codes",
        "Policy-grounded eligibility decisions from Medi-Cal guidelines",
        "90 procedures with full coverage decision trees",
        "Progressively improved matching quality across multiple iterations",
    ], Inches(1.2), Inches(3.2), Inches(10), font_size=16)


def slide_problem(prs):
    """Slide 2 — Problem statement."""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_heading(s, "The Problem", Inches(0.8), Inches(0.4), Inches(11))
    add_divider(s, Inches(0.8), Inches(1.05), Inches(11.5))

    add_subheading(s, "Patient Need", Inches(0.8), Inches(1.25), Inches(5.5))
    add_bullets(s, [
        "Medi-Cal patients must navigate dense policy PDFs to find coverage answers",
        "Plain-English questions like \"is a mammogram covered?\" don't match clinical CPT terminology",
        "Patients have no way to verify coverage before scheduling a procedure",
    ], Inches(0.8), Inches(1.75), Inches(5.5), font_size=14)

    add_subheading(s, "Technical Gap", Inches(0.8), Inches(3.4), Inches(5.5))
    add_bullets(s, [
        {"text": "Baseline semantic search: 56% Hit@5, 80% Hit@10, MRR 0.364", "color": RED},
        "Admin codes (G9900, M1285) outranked actual procedure codes",
        "Retired codes (77053, 77054) polluted search results",
        "\"follow up mammogram\" returned biopsy codes instead of mammogram codes",
    ], Inches(0.8), Inches(3.9), Inches(5.5), font_size=14)

    add_subheading(s, "Our Answer", Inches(7.2), Inches(1.25), Inches(5.5))
    add_bullets(s, [
        "Hybrid semantic + lexical search (FAISS + BM25)",
        "Cohere cross-encoder reranking",
        "Patient-friendly clean titles generated via LLM",
        "Policy-grounded eligibility decision trees for 90 codes",
        "Iterative eval harness to measure every improvement",
    ], Inches(7.2), Inches(1.75), Inches(5.5), font_size=14)

    add_screenshot_placeholder(s, Inches(7.2), Inches(4.0), Inches(5.5), Inches(3.0),
                                "App home screen / search box")


def slide_architecture(prs):
    """Slide 3 — System architecture."""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_heading(s, "System Architecture", Inches(0.8), Inches(0.4), Inches(11))
    add_divider(s, Inches(0.8), Inches(1.05), Inches(11.5))

    # Pipeline flow as labelled boxes
    boxes = [
        ("User Query", Inches(0.3)),
        ("Spell\nCorrect", Inches(1.85)),
        ("FAISS\nSemantic", Inches(3.4)),
        ("BM25\nLexical", Inches(4.95)),
        ("Hybrid\nBlend", Inches(6.5)),
        ("Cohere\nRerank", Inches(8.05)),
        ("Eligibility\nTree + RAG", Inches(9.6)),
        ("Coverage\nDecision", Inches(11.15)),
    ]
    bw, bh = Inches(1.35), Inches(0.85)
    by = Inches(1.55)
    colors = [TABLE_HDR, BLUE, BLUE, BLUE, BLUE, BLUE, RGBColor(0x1B, 0x7F, 0x46), RGBColor(0x1B, 0x7F, 0x46)]

    for (label, bx), col in zip(boxes, colors):
        box = s.shapes.add_shape(1, bx, by, bw, bh)
        box.fill.solid()
        box.fill.fore_color.rgb = col
        box.line.fill.background()
        tf = box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = label
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE

        # Arrow
        if bx < Inches(11.15):
            arr = s.shapes.add_shape(1, bx + bw, by + Inches(0.3), Inches(0.5), Inches(0.2))
            arr.fill.solid()
            arr.fill.fore_color.rgb = MED_GRAY
            arr.line.fill.background()

    add_divider(s, Inches(0.8), Inches(2.65), Inches(11.5))

    # Two-column component details
    add_subheading(s, "Search Layer", Inches(0.8), Inches(2.85), Inches(5.5))
    add_bullets(s, [
        "FAISS index: 16,054 vectors, 384 dims (all-MiniLM-L6-v2)",
        "search_text = clean_title + description + KeyBERT keywords",
        "BM25 over same corpus for exact keyword matching",
        "Hybrid blend: 60% semantic + 40% lexical",
        "Cohere rerank-v3.5 re-scores top candidates",
    ], Inches(0.8), Inches(3.35), Inches(5.5), font_size=13)

    add_subheading(s, "Coverage Layer", Inches(7.2), Inches(2.85), Inches(5.5))
    add_bullets(s, [
        "LanceDB vector store: 144 chunks from 2 Medi-Cal policy PDFs",
        "RAG retrieves relevant policy text per selected code",
        "100 codes with pre-built eligibility decision trees",
        "Multi-turn Q&A narrows down coverage pathway",
        "Final decision: Covered / Not Covered / Uncertain",
    ], Inches(7.2), Inches(3.35), Inches(5.5), font_size=13)

    add_screenshot_placeholder(s, Inches(0.8), Inches(5.5), Inches(11.5), Inches(1.7),
                                "End-to-end query → search results → eligibility flow")


def slide_matching_improvements(prs):
    """Slide 4 — Incremental matching improvements."""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_heading(s, "Matching Quality: Iterative Improvements", Inches(0.8), Inches(0.4), Inches(11))
    add_divider(s, Inches(0.8), Inches(1.05), Inches(11.5))

    add_subheading(s, "Incremental Impact on 25-Query Mammography Eval Set",
                   Inches(0.8), Inches(1.25), Inches(11))

    G = GREEN
    make_table(s, [
        ["Change", "Hit@5", "Hit@10", "MRR", "Recall@10", "Key Win"],
        ["1. Baseline (title + category + description)",
         "56%", "80%", "0.364", "67%", "—"],
        ["2. + KeyBERT keyword enrichment",
         "56%", "80%",
         {"text": "0.452 ▲", "color": G}, "63%", "Better ranking order"],
        ["3. + Remove retired codes (77053, 77054)",
         "60%",
         {"text": "88% ▲", "color": G},
         {"text": "0.465 ▲", "color": G},
         {"text": "72% ▲", "color": G}, "Fewer false positives"],
        [{"text": "4. + Admin code filtering (G/M/Q prefixes)", "bold": True},
         {"text": "72% ▲", "color": G, "bold": True},
         {"text": "96% ▲", "color": G, "bold": True},
         {"text": "0.554 ▲", "color": G, "bold": True},
         {"text": "87% ▲", "color": G, "bold": True},
         {"text": "Largest single gain", "color": G, "bold": True}],
        ["5. v2: Hybrid BM25+FAISS + Cohere rerank + clean titles",
         {"text": "80–88%", "color": G},
         {"text": "80–88%", "color": G},
         {"text": "0.627 ▲", "color": G, "bold": True},
         "—", "Better ranking, cleaner titles"],
    ], Inches(0.8), Inches(1.75),
        [Inches(4.3), Inches(0.75), Inches(0.85), Inches(0.85), Inches(1.0), Inches(2.2)],
        font_size=12)

    add_divider(s, Inches(0.8), Inches(4.9), Inches(11.5))

    add_subheading(s, "What Moved the Needle Most", Inches(0.8), Inches(5.1), Inches(6))
    add_bullets(s, [
        {"text": "Admin code filtering: +16pp Hit@5, +8pp Hit@10 in one step", "color": GREEN, "bold": True},
        "Keyword enrichment improved ranking quality (MRR) more than hit rates",
        "Hybrid search + Cohere reranking boosted MRR to 0.627 — best overall",
        {"text": "Generalist MiniLM-L6-v2 outperformed medical-specific models (PubMedBERT)", "color": MED_GRAY},
    ], Inches(0.8), Inches(5.55), Inches(6), font_size=13)

    add_subheading(s, "Remaining Gap", Inches(7.5), Inches(5.1), Inches(5))
    add_bullets(s, [
        {"text": "Conversational queries still score lower than direct queries", "color": RED},
        "\"I think I might be pregnant\" → 16% match confidence",
        "\"pregnancy test\" → 46% match confidence",
        {"text": "Root cause: MiniLM is a symmetric model, not QA-optimised",
         "color": MED_GRAY, "italic": True},
        {"text": "Fix: multi-qa-MiniLM-L6-cos-v1 (next step)", "color": BLUE},
    ], Inches(7.5), Inches(5.55), Inches(5), font_size=13)


def slide_eval_harness(prs):
    """Slide 5 — Evaluation harness."""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_heading(s, "Evaluation Harness", Inches(0.8), Inches(0.4), Inches(11))
    add_divider(s, Inches(0.8), Inches(1.05), Inches(11.5))

    add_subheading(s, "Hand-crafted Set (Phase 1)", Inches(0.8), Inches(1.25), Inches(5.5))
    add_bullets(s, [
        "25 queries across 5 breast imaging procedures",
        "Each query mapped to 1–5 expected CPT codes",
        "4 verbosity levels: direct, short, conversational, verbose",
        "Used to measure all incremental improvements",
    ], Inches(0.8), Inches(1.75), Inches(5.5), font_size=13)

    add_subheading(s, "Policy-wide Set (Phase 2)", Inches(7.2), Inches(1.25), Inches(5.5))
    add_bullets(s, [
        {"text": "720 queries: 90 policy codes × 8 queries each", "bold": True},
        "GPT-4o-mini generated queries per code using policy context",
        "Covers all codes in both Medi-Cal policy PDFs",
        {"text": "Phase 2 results: Hit@5 75.8%, Hit@10 84.3%, MRR 0.534", "color": BLUE},
    ], Inches(7.2), Inches(1.75), Inches(5.5), font_size=13)

    add_divider(s, Inches(0.8), Inches(3.2), Inches(11.5))

    add_subheading(s, "Metrics Used", Inches(0.8), Inches(3.4), Inches(11))
    make_table(s, [
        ["Metric", "Formula", "What It Captures"],
        ["Hit@K", "1 if any expected code in top-K else 0", "Can the patient find the right procedure?"],
        ["MRR", "1 / rank of first correct code", "How high does the correct code rank?"],
        ["Recall@K", "# expected found in top-K / # expected", "How many of the valid codes appear?"],
    ], Inches(0.8), Inches(3.85), [Inches(1.3), Inches(4.5), Inches(5.5)], font_size=13)

    add_subheading(s, "Query Verbosity Breakdown", Inches(0.8), Inches(5.4), Inches(11))
    make_table(s, [
        ["Level", "Example Query"],
        ["Direct", "mammogram"],
        ["Short", "is a mammogram covered by medi-cal?"],
        ["Conversational", "I need a mammogram, my last one was 3 years ago, is it covered?"],
        ["Verbose", "I just turned 40 and I know I'm supposed to start getting mammograms, I'm on medi-cal..."],
    ], Inches(0.8), Inches(5.85), [Inches(1.8), Inches(9.8)], font_size=12, row_height=Inches(0.35))


def slide_clean_titles(prs):
    """Slide 6 — Clean title generation."""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_heading(s, "Patient-Friendly Titles", Inches(0.8), Inches(0.4), Inches(11))
    add_divider(s, Inches(0.8), Inches(1.05), Inches(11.5))

    add_subheading(s, "The Problem", Inches(0.8), Inches(1.25), Inches(5.5))
    add_bullets(s, [
        "CPT title column = section hierarchy from the CPT codebook",
        {"text": "All 8 mammography codes showed identical title:", "color": DARK_GRAY},
        {"text": "\"Breast Mammography.Diagnostic Radiology. Mammography\"",
         "color": RED, "italic": True, "indent": 1},
        "Patients can't distinguish screening vs diagnostic vs 3D",
        "Bad titles also polluted FAISS search_text embeddings",
    ], Inches(0.8), Inches(1.75), Inches(5.5), font_size=13)

    add_subheading(s, "The Fix", Inches(7.2), Inches(1.25), Inches(5.5))
    add_bullets(s, [
        "GPT-4o-mini generated patient-friendly titles for 90 policy codes",
        "Manual review + override for medically inaccurate ones",
        {"text": "e.g. 77046/77047 came back as \"3D breast imaging\" → corrected to",
         "color": MED_GRAY},
        {"text": "\"Breast MRI without contrast\" / \"Breast MRI with contrast\"",
         "color": GREEN, "italic": True, "indent": 1},
        "clean_title stored in parquet, preferred over raw title at query time",
        "FAISS index rebuilt with clean_title in search_text",
    ], Inches(7.2), Inches(1.75), Inches(5.5), font_size=13)

    add_divider(s, Inches(0.8), Inches(3.85), Inches(11.5))

    add_subheading(s, "Before / After (same 8 codes)", Inches(0.8), Inches(4.05), Inches(11))

    make_table(s, [
        ["Code", "Before (CPT breadcrumb)", "After (patient-friendly)"],
        ["77065", "Breast Mammography.Diagnostic Radiology. Mammography",
         "Diagnostic mammogram, one breast"],
        ["77066", "Breast Mammography.Diagnostic Radiology. Mammography",
         "Diagnostic mammogram, both breasts"],
        ["77067", "Breast Mammography.Diagnostic Radiology. Mammography",
         "Screening mammogram, both breasts"],
        ["77063", "Breast Mammography.Diagnostic Radiology. Mammography",
         "3D mammogram add-on (tomosynthesis)"],
        ["77046", "Breast Mammography.Diagnostic Radiology. Mammography",
         "Breast MRI without contrast"],
    ], Inches(0.8), Inches(4.5),
        [Inches(0.85), Inches(5.0), Inches(5.0)], font_size=12, row_height=Inches(0.38))


def slide_search_ui(prs):
    """Slide 7 — Search UI with match quality badges."""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_heading(s, "Search UI — Match Quality Indicators", Inches(0.8), Inches(0.4), Inches(11))
    add_divider(s, Inches(0.8), Inches(1.05), Inches(11.5))

    add_subheading(s, "What Was Added", Inches(0.8), Inches(1.25), Inches(5.5))
    add_bullets(s, [
        "Match quality badge on every result card",
        {"text": "Best match", "color": RGBColor(0x06, 0x5F, 0x46), "bold": True},
        {"text": "  rank 1 — highest confidence", "color": MED_GRAY, "indent": 1},
        {"text": "Good match", "color": RGBColor(0x1E, 0x40, 0xAF), "bold": True},
        {"text": "  ranks 2–3", "color": MED_GRAY, "indent": 1},
        {"text": "Possible match", "color": RGBColor(0x6B, 0x72, 0x80), "bold": True},
        {"text": "  ranks 4+", "color": MED_GRAY, "indent": 1},
        "Cohere relevance score shown as % (e.g. \"Best match · 74%\")",
        "Price removed — focus is coverage, not cost",
    ], Inches(0.8), Inches(1.75), Inches(5.5), font_size=13)

    add_subheading(s, "Demo Queries — Direct vs Conversational", Inches(0.8), Inches(4.6), Inches(11))
    make_table(s, [
        ["Query", "Top Result", "Score", "Note"],
        ["\"mammogram\"", "Screening mammogram, both breasts (77067)", "74%", "Direct → high confidence"],
        ["\"breast MRI\"", "Breast MRI without contrast (77046)", "65%", "Direct → good match"],
        ["\"my doctor wants a breast MRI\"",
         "Breast MRI without contrast (77046)", "28%",
         "Conversational → lower score (known gap)"],
        ["\"I need an endometrial biopsy\"",
         "Endometrial biopsy (58100)", "48%", "Non-mammogram code works too"],
    ], Inches(0.8), Inches(5.1),
        [Inches(3.5), Inches(3.8), Inches(0.8), Inches(2.8)], font_size=12, row_height=Inches(0.38))

    add_screenshot_placeholder(s, Inches(7.2), Inches(1.25), Inches(5.5), Inches(3.1),
                                "Search results showing Best/Good/Possible match badges")


def slide_eligibility_flow(prs):
    """Slide 8 — Eligibility decision flow."""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_heading(s, "Coverage Decision Flow", Inches(0.8), Inches(0.4), Inches(11))
    add_divider(s, Inches(0.8), Inches(1.05), Inches(11.5))

    add_subheading(s, "How It Works", Inches(0.8), Inches(1.25), Inches(5.5))
    add_bullets(s, [
        "Patient selects a procedure from search results",
        "System retrieves relevant Medi-Cal policy chunks (RAG)",
        "LLM identifies 1–2 coverage pathways for that code",
        "Multi-turn Q&A gathers patient-specific criteria",
        {"text": "e.g. \"Is this for screening or diagnostic purposes?\"", "color": MED_GRAY, "italic": True},
        {"text": "e.g. \"Has your doctor confirmed a clinical indication?\"", "color": MED_GRAY, "italic": True},
        "Decision rendered: Covered / Not Covered / Need more info",
        "Reason grounded in policy text",
    ], Inches(0.8), Inches(1.75), Inches(5.5), font_size=13)

    add_subheading(s, "Coverage Data", Inches(7.2), Inches(1.25), Inches(5.5))
    add_bullets(s, [
        "2 Medi-Cal policy PDFs indexed (144 chunks in LanceDB)",
        "90 CPT codes with pre-built decision trees",
        "Each tree has 2 pathways + pathway-specific questions",
        {"text": "100 codes total in eligibility_trees.json", "color": MED_GRAY},
        "Codes without policy coverage return no_pathways flag",
    ], Inches(7.2), Inches(1.75), Inches(5.5), font_size=13)

    add_screenshot_placeholder(s, Inches(0.8), Inches(3.8), Inches(5.5), Inches(3.3),
                                "Eligibility question flow (Q&A with patient)")
    add_screenshot_placeholder(s, Inches(7.2), Inches(3.8), Inches(5.5), Inches(3.3),
                                "Coverage outcome — Covered / Not Covered screen")


def slide_next_steps(prs):
    """Slide 9 — What's next."""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_heading(s, "What's Next", Inches(0.8), Inches(0.4), Inches(11))
    add_divider(s, Inches(0.8), Inches(1.05), Inches(11.5))

    cols = [Inches(0.8), Inches(4.7), Inches(8.6)]
    col_w = Inches(3.5)

    add_subheading(s, "Matching Quality", cols[0], Inches(1.3), col_w)
    add_bullets(s, [
        {"text": "Switch to multi-qa-MiniLM-L6-cos-v1", "bold": True},
        {"text": "Asymmetric model trained for QA retrieval", "color": MED_GRAY, "indent": 1},
        {"text": "Expected: +10-20pp on conversational queries", "color": GREEN, "indent": 1},
        "Wire admin code filtering (G/M/Q) into production matcher",
        "LLM query expansion — rewrite before embedding",
    ], cols[0], Inches(1.8), col_w, font_size=13)

    add_subheading(s, "Coverage Intelligence", cols[1], Inches(1.3), col_w)
    add_bullets(s, [
        {"text": "Expand policy PDF coverage", "bold": True},
        "Add more Medi-Cal policy sections",
        "Ground remaining 10 HCPCS codes in policy text",
        "Confidence scoring calibration",
        "Handle \"uncertain\" gracefully — suggest next steps for patient",
    ], cols[1], Inches(1.8), col_w, font_size=13)

    add_subheading(s, "User Experience", cols[2], Inches(1.3), col_w)
    add_bullets(s, [
        {"text": "\"No policy doc\" indicator for uncovered codes", "bold": True},
        "History panel — past searches and outcomes",
        "Mobile-friendly layout",
        "Accessibility improvements (ARIA, keyboard nav)",
    ], cols[2], Inches(1.8), col_w, font_size=13)

    add_divider(s, Inches(0.8), Inches(4.5), Inches(11.5))

    add_subheading(s, "Key Insight from This Project", Inches(0.8), Inches(4.7), Inches(11))
    add_bullets(s, [
        {"text": "Filtering bad candidates (admin codes, retired codes) had more impact than adding more signal (KeyBERT)",
         "bold": True, "color": BLUE},
        "Generalist embedding models beat domain-specific models on short clinical queries",
        "Eval set breadth matters: 25 mammography queries missed issues visible in 720-query policy eval",
    ], Inches(0.8), Inches(5.2), Inches(11.5), font_size=14)


# ── Main ─────────────────────────────────────────────────────────────────────

def build_presentation():
    prs = Presentation()
    prs.slide_width  = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide_title(prs)
    slide_problem(prs)
    slide_architecture(prs)
    slide_matching_improvements(prs)
    slide_eval_harness(prs)
    slide_clean_titles(prs)
    slide_search_ui(prs)
    slide_eligibility_flow(prs)
    slide_next_steps(prs)

    return prs


if __name__ == "__main__":
    prs = build_presentation()
    out_path = Path(__file__).resolve().parents[1] / "eval" / "results" / "coverage_estimator_deck.pptx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    print(f"Saved to {out_path}")
