"""Unit tests for the pure-Python financial engine."""
from __future__ import annotations

import math

import pytest

from app.financial.calculations import average, cagr, difference, pct_change, percentage, safe_div
from app.financial.forecasting import project_cagr, project_linear
from app.financial.metrics import detect_metrics, match_metric, normalize_label
from app.financial.periods import find_periods, parse_period
from app.financial.ratios import (
    RATIOS,
    compute_ratio,
    compute_ratios,
    derive_metrics,
    detect_ratios,
    query_metric_keys,
)
from app.financial.risk_signals import detect_risk_signals
from app.financial.trends import analyze_series
from app.utils.errors import CalculationError

VALUES = {
    "revenue": 1000.0, "cogs": 600.0, "ebit": 150.0, "ebitda": 200.0, "net_income": 100.0,
    "finance_costs": 25.0, "total_assets": 2000.0, "equity": 800.0, "total_debt": 400.0,
    "current_assets": 500.0, "current_liabilities": 250.0, "inventory": 100.0, "cash": 50.0,
    "receivables": 125.0, "gross_profit": 400.0, "eps": 5.0,
}


class TestCalculations:
    def test_safe_div(self):
        assert safe_div(10, 4) == 2.5
        assert safe_div(10, 0) is None
        assert safe_div(None, 4) is None
        assert safe_div(10, None) is None

    def test_percentage_and_change(self):
        assert percentage(1, 4) == 25.0
        assert pct_change(125, 100) == 25.0
        assert pct_change(80, 100) == -20.0
        assert pct_change(100, 0) is None
        assert pct_change(None, 100) is None
        # Change is measured relative to the magnitude of the base, so a loss narrowing is positive.
        assert pct_change(-50, -100) == 50.0
        assert difference(12.5, 10.0) == 2.5

    def test_cagr(self):
        assert cagr(100, 121, 2) == pytest.approx(10.0)
        assert cagr(6300, 10600, 4) == pytest.approx(13.891, abs=1e-3)
        assert cagr(100, 121, 0) is None
        assert cagr(-100, 121, 2) is None
        assert cagr(100, None, 2) is None

    def test_average(self):
        assert average([1, 2, None, 3]) == 2.0
        assert average([None]) is None

    @pytest.mark.parametrize("bad", [math.nan, math.inf, "12", True])
    def test_rejects_corrupt_input(self, bad):
        with pytest.raises(CalculationError):
            safe_div(bad, 2)


class TestRatios:
    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("current_ratio", 2.0), ("quick_ratio", 1.6), ("cash_ratio", 0.2),
            ("debt_to_equity", 0.5), ("debt_ratio", 0.2), ("interest_coverage", 6.0),
            ("gross_margin", 40.0), ("operating_margin", 15.0), ("net_margin", 10.0),
            ("ebitda_margin", 20.0), ("roa", 5.0), ("roe", 12.5),
            ("asset_turnover", 0.5), ("inventory_turnover", 6.0), ("receivables_turnover", 8.0),
        ],
    )
    def test_ratio_values(self, key, expected):
        result = compute_ratio(key, VALUES)
        assert result.value == pytest.approx(expected)
        assert result.missing == []

    def test_growth_ratios_need_previous_period(self):
        previous = {"revenue": 800.0, "net_income": 125.0, "eps": 4.0}
        assert compute_ratio("revenue_growth", VALUES, previous).value == pytest.approx(25.0)
        assert compute_ratio("profit_growth", VALUES, previous).value == pytest.approx(-20.0)
        assert compute_ratio("eps_growth", VALUES, previous).value == pytest.approx(25.0)
        missing = compute_ratio("revenue_growth", VALUES)
        assert missing.value is None and missing.missing == ["prev_revenue"]

    def test_missing_inputs_are_reported_not_guessed(self):
        result = compute_ratio("debt_to_equity", {"total_debt": 400.0})
        assert result.value is None
        assert result.missing == ["equity"]

    def test_zero_denominator_is_undefined(self):
        assert compute_ratio("current_ratio", {"current_assets": 5.0, "current_liabilities": 0.0}).value is None

    def test_every_spec_ratio_is_registered(self):
        keys = {r.key for r in RATIOS}
        assert {
            "current_ratio", "quick_ratio", "cash_ratio", "debt_to_equity", "debt_ratio", "interest_coverage",
            "gross_margin", "operating_margin", "net_margin", "roa", "roe", "ebitda_margin", "asset_turnover",
            "inventory_turnover", "receivables_turnover", "revenue_growth", "profit_growth", "eps_growth",
        } <= keys
        assert len(compute_ratios(VALUES)) == len(RATIOS)

    def test_derive_metrics(self):
        values, derived = derive_metrics({
            "short_term_debt": 100.0, "long_term_debt": 300.0, "revenue": 1000.0, "cogs": 600.0,
            "pbt": 125.0, "finance_costs": 25.0, "depreciation": 50.0, "cfo": 180.0, "capex": -220.0,
            "total_assets": 2000.0, "equity": 800.0,
        })
        assert values["total_debt"] == 400.0
        assert values["gross_profit"] == 400.0
        assert values["ebit"] == 150.0
        assert values["ebitda"] == 200.0
        assert values["free_cash_flow"] == -40.0       # capex is an outflow whatever its sign
        assert values["total_liabilities"] == 1200.0
        assert set(derived) == {"total_debt", "gross_profit", "ebit", "ebitda", "free_cash_flow", "total_liabilities"}

    def test_ind_as_statement_format(self):
        """Cost of goods sold split over three lines, and no single profit-before-tax line."""
        values, derived = derive_metrics({
            "revenue": 1000.0, "materials_consumed": 300.0, "purchases_stock_in_trade": 150.0,
            "inventory_change": -20.0, "total_income": 1040.0, "total_expenses": 840.0,
            "finance_costs": 10.0, "depreciation": 30.0,
        })
        assert values["cogs"] == 430.0 and values["gross_profit"] == 570.0
        assert values["ebit"] == 210.0 and values["ebitda"] == 240.0
        assert "before exceptional items" in derived["ebit"]
        only_materials, _ = derive_metrics({"revenue": 100.0, "materials_consumed": 60.0})
        assert only_materials["cogs"] == 60.0
        # A reported profit before tax takes precedence over the fallback.
        reported, formulas = derive_metrics({"pbt": 100.0, "finance_costs": 10.0, "total_income": 500.0, "total_expenses": 380.0})
        assert reported["ebit"] == 110.0 and formulas["ebit"] == "Profit Before Tax + Finance Costs"

    def test_reported_values_are_never_overwritten(self):
        values, derived = derive_metrics({"total_debt": 999.0, "short_term_debt": 1.0, "long_term_debt": 2.0})
        assert values["total_debt"] == 999.0 and "total_debt" not in derived

    def test_query_metric_keys_expand_ratios_to_reported_line_items(self):
        assert query_metric_keys("What is the debt-to-equity ratio?") == [
            "total_debt", "equity", "short_term_debt", "long_term_debt",
        ]
        assert query_metric_keys("Why did net profit fall?") == ["net_income"]
        assert set(query_metric_keys("revenue growth and free cash flow")) == {"revenue", "free_cash_flow", "cfo", "capex"}
        assert query_metric_keys("What did management say about strategy?") == []

    def test_detect_ratios(self):
        assert detect_ratios("What is the debt-to-equity ratio?") == ["debt_to_equity"]
        assert set(detect_ratios("show ROE and the current ratio")) == {"roe", "current_ratio"}
        assert "net_margin" in detect_ratios("How is profitability?")
        assert detect_ratios("How is profitability?", include_topics=False) == []


class TestTrends:
    def test_increasing_series(self):
        trend = analyze_series("revenue", "Revenue", [("FY2023", 2023, 100.0), ("FY2024", 2024, 120.0), ("FY2025", 2025, 150.0)])
        assert trend.direction == "increasing"
        assert [round(p.change, 2) if p.change is not None else None for p in trend.points] == [None, 20.0, 25.0]
        assert trend.cagr == pytest.approx(22.474, abs=1e-3)
        assert len(trend.significant) == 2

    def test_percentage_series_changes_are_in_points(self):
        trend = analyze_series("net_margin", "Net margin", [("FY2024", 2024, 8.68), ("FY2025", 2025, 6.79)], unit="%")
        assert trend.points[1].change == pytest.approx(-1.89)
        assert trend.cagr is None and trend.change_unit == "pp"
        assert trend.direction == "decreasing"

    def test_gap_years_are_not_compared(self):
        trend = analyze_series("revenue", "Revenue", [("FY2021", 2021, 100.0), ("FY2024", 2024, 200.0)])
        assert trend.points[1].change is None
        assert trend.cagr == pytest.approx(25.992, abs=1e-3)

    def test_insufficient_data(self):
        trend = analyze_series("revenue", "Revenue", [("FY2025", 2025, 100.0), ("FY2024", 2024, None)])
        assert not trend.has_data and trend.direction == "insufficient_data"

    def test_projections_are_mechanical(self):
        projection = project_cagr([(2023, 100.0), (2025, 121.0)], periods=2)
        assert projection.labels == ["FY2026E", "FY2027E"]
        assert projection.values == pytest.approx([133.1, 146.41])
        assert project_cagr([(2025, 100.0)]) is None
        linear = project_linear([(2023, 10.0), (2024, 20.0), (2025, 30.0)], periods=1)
        assert linear.values == pytest.approx([40.0])


class TestRiskSignals:
    def test_detects_leverage_cash_flow_and_margin_pressure(self):
        current = {
            "total_debt": 3500.0, "equity": 5500.0, "ebit": 1200.0, "finance_costs": 240.0, "cfo": 980.0,
            "free_cash_flow": -750.0, "revenue": 10600.0, "net_income": 720.0,
            "current_assets": 3900.0, "current_liabilities": 2720.0, "inventory": 1620.0, "receivables": 1420.0,
        }
        previous = {
            "total_debt": 2300.0, "equity": 4950.0, "ebit": 1258.0, "finance_costs": 170.0, "cfo": 1240.0,
            "revenue": 9400.0, "net_income": 816.0, "inventory": 1300.0, "receivables": 1180.0,
        }
        keys = {s.key for s in detect_risk_signals("FY2025", current, "FY2024", previous)}
        assert {
            "rising_leverage", "declining_interest_coverage", "declining_operating_cash_flow",
            "negative_free_cash_flow", "operating_margin_compression", "net_margin_compression",
            "profit_decline_despite_growth", "inventory_build_up",
        } <= keys

    def test_healthy_company_raises_no_signals(self):
        current = {"total_debt": 100.0, "equity": 1000.0, "ebit": 300.0, "finance_costs": 10.0, "cfo": 250.0,
                   "free_cash_flow": 150.0, "revenue": 1100.0, "net_income": 220.0,
                   "current_assets": 600.0, "current_liabilities": 300.0}
        previous = {"total_debt": 110.0, "equity": 900.0, "ebit": 260.0, "finance_costs": 11.0, "cfo": 220.0,
                    "revenue": 1000.0, "net_income": 190.0}
        assert detect_risk_signals("FY2025", current, "FY2024", previous) == []

    def test_no_data_no_signals(self):
        assert detect_risk_signals("FY2025", {}, None, None) == []


class TestPeriods:
    @pytest.mark.parametrize(
        ("text", "label"),
        [
            ("FY2025", "FY2025"), ("FY 25", "FY2025"), ("FY'24", "FY2024"), ("FY2024-25", "FY2025"),
            ("2024-25", "FY2025"), ("Year ended March 31, 2025", "FY2025"), ("As at 31 March 2024", "FY2024"),
            ("December 31, 2023", "FY2023"), ("Q1 FY2025", "Q1 FY2025"), ("Q3 FY25", "Q3 FY2025"),
            ("2022", "FY2022"),
        ],
    )
    def test_parse(self, text, label):
        assert parse_period(text).label == label

    def test_multiple_periods_in_reading_order(self):
        assert [p.label for p in find_periods("Particulars FY2025 FY2024 FY2023")] == ["FY2025", "FY2024", "FY2023"]

    @pytest.mark.parametrize("text", ["revenue of 2,025 crore", "margin 2024.5", "grew 12%", "market share", "decline in costs"])
    def test_non_periods(self, text):
        assert find_periods(text) == []

    def test_quarter_metadata(self):
        period = parse_period("Q2 FY2025")
        assert (period.fiscal_year, period.quarter, period.kind) == (2025, 2, "quarter")


class TestMetricMatching:
    @pytest.mark.parametrize(
        ("label", "metric"),
        [
            ("Revenue from operations", "revenue"), ("Total revenue", "revenue"), ("Net sales", "revenue"),
            ("Profit for the year", "net_income"), ("Net income", "net_income"),
            ("Profit/(loss) for the year", "net_income"),
            ("Total current assets", "current_assets"), ("Total Assets", "total_assets"),
            ("Total stockholders' equity", "equity"), ("Finance costs", "finance_costs"),
            ("Net cash generated from operating activities", "cfo"),
            ("Purchase of property, plant and equipment", "capex"),
            ("Depreciation and amortisation expense", "depreciation"),
            ("(a) Trade receivables", "receivables"), ("Inventories (Note 12)", "inventory"),
            ("Profit before tax", "pbt"),
        ],
    )
    def test_known_labels(self, label, metric):
        assert match_metric(label)[0] == metric

    @pytest.mark.parametrize(
        "label",
        [
            "Total equity and liabilities", "Net profit margin", "Revenue growth (%)", "Other current assets",
            "Deferred tax expense", "Total comprehensive income for the year", "Other expenses",
            "Weighted average number of shares", "Employee benefits expense",
        ],
    )
    def test_ambiguous_labels_are_not_mapped(self, label):
        assert match_metric(label) is None

    def test_normalize(self):
        assert normalize_label("  (ii) Trade Receivables (Note 7) ") == "trade receivables"

    def test_detect_metrics_in_query(self):
        assert detect_metrics("Why did net profit fall while revenue grew?") == ["net_income", "revenue"]
        assert detect_metrics("Is free cash flow negative?") == ["free_cash_flow"]
        assert detect_metrics("hello there") == []
