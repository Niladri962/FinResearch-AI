"""Layouts found in real annual reports that the synthetic samples do not exercise:
two-page spreads, multi-column text, wrapped statement rows, embedded rupee glyphs,
formula-tagged labels, and standalone + consolidated statements in one document."""
from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from app.documents.parser import _Span, _split_columns, is_heading, Line, parse_document
from app.documents.table_extractor import (
    detect_currency,
    detect_text_tables,
    detect_unit,
    extract_structured_table,
    is_header_fragment,
    parse_number,
)
from app.financial.metrics import match_metric, normalize_label
from app.financial.periods import annual
from app.rag.chunking import FinancialChunker, is_primary_statement
from tests.conftest import upload


def _span(text: str, x0: float, y: float, width: float = 200.0) -> _Span:
    return _Span(text, x0, y, x0 + width, y + 9, 8.5, False)


class TestSpreadsAndColumns:
    def test_landscape_spread_is_split_even_when_both_halves_are_tables(self):
        # Balance sheet on the left printed page, P&L on the right: labels + figures on each side.
        spans = []
        for row in range(20):
            y = 60 + row * 12
            spans += [_span(f"Left item {row}", 50, y, 150), _span("1,234", 480, y, 40),
                      _span(f"Right item {row}", 650, y, 150), _span("5,678", 1080, y, 40)]
        columns = _split_columns(spans, width=1191, height=842)
        assert len(columns) == 2
        assert all(s.x1 < 600 for s in columns[0]) and all(s.x0 > 600 for s in columns[1])

    def test_spread_with_two_text_columns_per_page_gives_four_columns(self):
        spans = []
        for row in range(40):
            y = 60 + row * 11
            for x in (50, 320, 650, 920):
                spans.append(_span("management believes the outlook remains positive", x, y, 240))
        assert len(_split_columns(spans, width=1191, height=842)) == 4

    def test_portrait_table_is_not_split(self):
        spans = []
        for row in range(30):
            y = 60 + row * 12
            spans += [_span(f"Line item {row}", 50, y, 180), _span("1,234", 400, y, 40), _span("987", 480, y, 30)]
        assert len(_split_columns(spans, width=595, height=842)) == 1

    def test_spread_pdf_end_to_end(self, tmp_path: Path):
        doc = pymupdf.open()
        page = doc.new_page(width=1191, height=842)
        page.insert_text((50, 60), "Consolidated Balance Sheet", fontsize=18, fontname="hebo")
        for i, (label, a, b) in enumerate([("Total current assets", "3,900", "3,440"), ("Total assets", "11,200", "9,400"), ("Total equity", "5,500", "4,950")]):
            y = 130 + i * 14
            page.insert_text((50, y), label, fontsize=9)
            page.insert_text((400, y), a, fontsize=9)
            page.insert_text((480, y), b, fontsize=9)
        page.insert_text((50, 100), "Particulars", fontsize=9)
        page.insert_text((400, 100), "FY2025", fontsize=9)
        page.insert_text((480, 100), "FY2024", fontsize=9)
        page.insert_text((650, 60), "Consolidated Statement of Profit and Loss", fontsize=18, fontname="hebo")
        page.insert_text((650, 100), "Particulars", fontsize=9)
        page.insert_text((1000, 100), "FY2025", fontsize=9)
        page.insert_text((1080, 100), "FY2024", fontsize=9)
        for i, (label, a, b) in enumerate([("Revenue from operations", "10,600", "9,400"), ("Finance costs", "240", "170"), ("Profit for the year", "720", "816")]):
            y = 130 + i * 14
            page.insert_text((650, y), label, fontsize=9)
            page.insert_text((1000, y), a, fontsize=9)
            page.insert_text((1080, y), b, fontsize=9)
        path = tmp_path / "spread.pdf"
        doc.save(str(path))
        doc.close()

        parsed = parse_document(path)
        tables = [e for e in parsed.pages[0].elements if e.kind == "table"]
        assert len(tables) == 2                                   # not one table with merged rows
        facts = {}
        for table in tables:
            for fact in extract_structured_table(table.rows).facts:
                facts[(fact.metric, fact.period.label)] = fact.value
        assert facts[("total_assets", "FY2025")] == 11200.0 and facts[("revenue", "FY2025")] == 10600.0
        assert facts[("net_income", "FY2024")] == 816.0 and facts[("equity", "FY2024")] == 4950.0


class TestStatementRows:
    LINES = [
        "Year ended    Year ended",
        "Particulars   Note",
        "31st March, 2026   31st March, 2025",
        "CONTINUING OPERATIONS",
        "INCOME",
        "Revenue from operations    25    61,975    59,676",
        "Changes in inventories of finished goods, work-in-progress and     29",
        " (395)    (78)",
        "stock-in-trade",
        "TOTAL EXPENSES    49,060    47,004",
        "PROFIT BEFORE EXCEPTIONAL ITEMS AND TAX FROM CONTINUING",
        " 13,874    13,849",
        "OPERATIONS",
        "Liabilities",
        "Non-current liabilities",
        "Financial liabilities",
        "Finance costs    31    363    350",
        "PROFIT FOR THE YEAR (C=A+B)    15,427    10,644",
        "The accompanying notes are an integral part of these financial statements.",
    ]

    def test_wrapped_rows_and_stacked_labels_do_not_end_the_table(self):
        tables = detect_text_tables(self.LINES)
        assert len(tables) == 1
        table = tables[0]
        assert (table.start, table.end) == (2, 18)
        assert ["Changes in inventories of finished goods, work-in-progress and", "29", "(395)", "(78)"] in table.rows
        assert ["PROFIT BEFORE EXCEPTIONAL ITEMS AND TAX FROM CONTINUING", "13,874", "13,849"] in table.rows

    def test_facts_from_a_real_style_statement(self):
        table = detect_text_tables(self.LINES)[0]
        facts = {(f.metric, f.period.label): f.value for f in extract_structured_table(table.rows).facts}
        assert facts[("revenue", "FY2026")] == 61975.0 and facts[("revenue", "FY2025")] == 59676.0
        assert facts[("net_income", "FY2026")] == 15427.0          # formula tag "(C=A+B)" ignored
        assert facts[("finance_costs", "FY2025")] == 350.0          # note number column skipped
        assert facts[("total_expenses", "FY2026")] == 49060.0

    def test_header_with_most_periods_wins_over_a_caption(self):
        rows = [["for the year ended 31st March, 2026"], ["Particulars", "31st March, 2026", "31st March, 2025"],
                ["Revenue from operations", "100", "90"]]
        table = extract_structured_table(rows)
        assert table.periods == ["FY2026", "FY2025"]

    def test_continuation_table_inherits_periods_only_when_told_to(self):
        rows = [["Total - Current liabilities (C)", "15,549", "16,537"], ["Total liabilities", "30,744", "30,271"]]
        assert extract_structured_table(rows) is None
        table = extract_structured_table(rows, fallback_periods=[annual(2026), annual(2025)])
        facts = {(f.metric, f.period.label): f.value for f in table.facts}
        assert facts[("current_liabilities", "FY2026")] == 15549.0 and facts[("total_liabilities", "FY2025")] == 30271.0

    def test_eps_split_by_continuing_and_discontinued_operations(self):
        rows = [
            ["Particulars", "FY2026", "FY2025"],
            ["Earnings per equity share", "35"],
            ["For Continuing operations"], ["Basic (in H)", "H46.90", "H45.34"],
            ["For Discontinued operations"], ["Basic (in H)", "H18.76", "H0.04"],
            ["For Continuing and Discontinued operations"], ["Basic (in H)", "H65.66", "H45.30"],
        ]
        facts = {(f.metric, f.period.label): f.value for f in extract_structured_table(rows).facts}
        assert facts == {("eps", "FY2026"): 65.66, ("eps", "FY2025"): 45.30}


class TestLabelsAndGlyphs:
    @pytest.mark.parametrize(
        ("label", "metric"),
        [
            ("PROFIT FOR THE YEAR (C=A+B)", "net_income"),
            ("TOTAL ASSETS (A+B)", "total_assets"),
            ("Total - Equity (A)", "equity"),
            ("Total - Current liabilities (C)", "current_liabilities"),
            ("Net cash flows generated from operating activities - [A]", "cfo"),
            ("Net cash flows (used in) / generated from investing activities - [B]", "cfi"),
            ("Net cash flows used in financing activities - [C]", "cff"),
            ("Net cash from/ (used in) operating activities", "cfo"),
            ("Profit/ (Loss) for the year", "net_income"),
        ],
    )
    def test_real_statement_labels(self, label, metric):
        assert match_metric(label)[0] == metric, normalize_label(label)

    def test_continuing_operations_subtotal_is_not_total_profit(self):
        assert match_metric("PROFIT FOR THE YEAR FROM CONTINUING OPERATIONS (A)") is None

    def test_rupee_glyph_extracted_as_letter(self):
        assert parse_number("H17,265") == 17265.0 and parse_number("H (0.04)") == -0.04 and parse_number("`250") == 250.0
        assert parse_number("H") is None and parse_number("HDFC") is None
        assert detect_unit("(All amounts in H crores, unless otherwise stated)") == ("crore", 1e7)
        assert detect_currency("(All amounts in H crores, unless otherwise stated)") == "INR"
        assert detect_currency("Hindustan Unilever reported growth") is None

    def test_column_header_fragments_are_not_section_headings(self):
        for text in ("Particulars   Note", "As at    As at", "Year ended    Year ended"):
            assert is_header_fragment(text)
            assert is_heading(Line(text=text, y0=0, y1=8, size=8.0, bold=True), body_size=8.5) == 0
        assert not is_header_fragment("Risk Factors")
        assert is_heading(Line(text="Risk Factors", y0=0, y1=8, size=8.0, bold=True), body_size=8.5) == 2

    def test_primary_statement_detection(self):
        assert is_primary_statement("Consolidated Balance Sheet")
        assert is_primary_statement("Standalone Statement of Profit and Loss")
        assert not is_primary_statement("Notes to the consolidated financial statements")


@pytest.mark.integration
class TestStandaloneAndConsolidated:
    def test_consolidated_statement_is_preferred_and_notes_do_not_override(self, empty_client, tmp_path: Path):
        text = (
            "# Meridian Foods Limited\n\nAnnual Report FY2025. (INR in crore)\n\n"
            "# Standalone Statement of Profit and Loss\n\n"
            "Particulars   FY2025   FY2024\n"
            "Revenue from operations   900   800\nFinance costs   20   18\nProfit for the year   90   80\n\n"
            "# Consolidated Statement of Profit and Loss\n\n"
            "Particulars   FY2025   FY2024\n"
            "Revenue from operations   1,500   1,300\nFinance costs   35   30\nProfit for the year   150   120\n\n"
            "# Notes to the consolidated financial statements\n\n"
            "Summarised financial information of a subsidiary is given below for reference by readers.\n\n"
            "Particulars   FY2025   FY2024\n"
            "Revenue from operations   40   35\nFinance costs   2   1\nProfit for the year   4   3\n"
        )
        path = tmp_path / "meridian_foods_annual_report_fy2025.md"
        path.write_text(text, encoding="utf-8")
        document = upload(empty_client, path)
        assert document["status"] == "ready" and document["fact_count"] == 18

        company = empty_client.get("/api/companies").json()[0]
        body = empty_client.post("/api/analyze", json={"company_id": company["id"]}).json()
        latest = body["periods"][-1]["metrics"]
        assert latest["revenue"] == 1500.0 and latest["net_income"] == 150.0     # consolidated, not standalone or note


class TestChunkerSections:
    def test_notes_heading_starts_a_new_part(self):
        from app.documents.parser import Element, ParsedDocument, ParsedPage

        body = "This paragraph explains the accounting treatment in enough words to be kept as a passage for retrieval."
        page = ParsedPage(number=1, elements=[
            Element(kind="heading", text="Consolidated Statement of Cash Flows", level=1),
            Element(kind="text", text=body),
            Element(kind="heading", text="Notes to the consolidated financial statements", level=2),
            Element(kind="text", text=body),
        ])
        chunks = FinancialChunker().chunk(ParsedDocument("text", [page]))
        assert [c.section for c in chunks] == ["Consolidated Statement of Cash Flows", "Notes to the consolidated financial statements"]
