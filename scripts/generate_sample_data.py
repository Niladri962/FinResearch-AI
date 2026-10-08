"""Generate synthetic sample filings and the evaluation dataset.

The companies, people and figures below are entirely FICTITIOUS. They exist so
the pipeline can be demonstrated, tested and evaluated without real filings and
without ever presenting invented numbers as real ones. Every generated document
states this on its cover.

Usage (from the repo root):
    python scripts/generate_sample_data.py            # writes data/samples + evaluation/dataset.jsonl
    python scripts/generate_sample_data.py --out DIR
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTICE = "SYNTHETIC SAMPLE - fictitious company and figures, generated for software testing only."

PAGE_W, PAGE_H = 595, 842
MARGIN = 60
BODY_W = PAGE_W - 2 * MARGIN


def n(value: float, decimals: int = 0) -> str:
    """Format like a filing: thousands separators, negatives in parentheses."""
    body = f"{abs(value):,.{decimals}f}"
    return f"({body})" if value < 0 else body


class PdfBuilder:
    def __init__(self, running_header: str) -> None:
        self.doc = fitz.open()
        self.running_header = running_header
        self.marks: dict[str, int] = {}
        self.page = None
        self.y = 0.0
        self._new_page()

    @property
    def page_number(self) -> int:
        return self.doc.page_count

    def mark(self, name: str) -> None:
        self.marks[name] = self.page_number

    def _new_page(self) -> None:
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self.page.insert_text((MARGIN, 34), self.running_header, fontsize=8, fontname="helv", color=(0.4, 0.4, 0.4))
        self.page.insert_text((MARGIN, PAGE_H - 28), NOTICE, fontsize=7, fontname="helv", color=(0.4, 0.4, 0.4))
        self.page.insert_text((PAGE_W - MARGIN - 10, PAGE_H - 28), str(self.page_number), fontsize=8, fontname="helv")
        self.y = 62.0

    def _ensure(self, height: float) -> None:
        if self.y + height > PAGE_H - 60:
            self._new_page()

    def new_page(self) -> None:
        self._new_page()

    def _wrap(self, text: str, size: float, font: str, width: float) -> list[str]:
        lines, current = [], ""
        for word in text.split():
            trial = f"{current} {word}".strip()
            if fitz.get_text_length(trial, fontname=font, fontsize=size) <= width:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def title(self, text: str) -> None:
        self._ensure(40)
        for line in self._wrap(text, 22, "hebo", BODY_W):
            self.page.insert_text((MARGIN, self.y + 22), line, fontsize=22, fontname="hebo")
            self.y += 30
        self.y += 8

    def heading(self, text: str) -> None:
        self._ensure(60)
        self.y += 6
        self.page.insert_text((MARGIN, self.y + 14), text, fontsize=14, fontname="hebo")
        self.y += 26

    def subheading(self, text: str) -> None:
        self._ensure(44)
        self.page.insert_text((MARGIN, self.y + 11), text, fontsize=11, fontname="hebo")
        self.y += 20

    def para(self, text: str) -> None:
        lines = self._wrap(text, 10, "helv", BODY_W)
        self._ensure(min(len(lines), 3) * 14)
        for line in lines:
            self._ensure(14)
            self.page.insert_text((MARGIN, self.y + 10), line, fontsize=10, fontname="helv")
            self.y += 14
        self.y += 9

    def table(self, header: list[str], rows: list[list[str]], *, ruled: bool = True, label_width: float = 235) -> None:
        """Draw a table. Rows with a single cell are section sub-labels."""
        row_h = 18
        self._ensure(row_h * (len(rows) + 1) + 6)
        cols = len(header)
        col_w = (BODY_W - label_width) / (cols - 1)
        edges = [MARGIN, MARGIN + label_width] + [MARGIN + label_width + col_w * i for i in range(1, cols)]

        def draw_row(cells: list[str], bold: bool) -> None:
            font = "hebo" if bold else "helv"
            padded = cells + [""] * (cols - len(cells))
            for idx, cell in enumerate(padded):
                x0, x1 = edges[idx], edges[idx + 1]
                if ruled:
                    self.page.draw_rect(fitz.Rect(x0, self.y, x1, self.y + row_h), color=(0.55, 0.55, 0.55), width=0.6)
                if not cell:
                    continue
                if idx == 0:
                    x = x0 + 5
                else:
                    x = x1 - 6 - fitz.get_text_length(cell, fontname=font, fontsize=9)
                self.page.insert_text((x, self.y + 12.5), cell, fontsize=9, fontname=font)
            self.y += row_h

        draw_row(header, bold=True)
        for row in rows:
            draw_row(row, bold=len(row) == 1 or row[0].lower().startswith("total"))
        self.y += 12

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(path), deflate=True)
        self.doc.close()


# ── Fictitious financial data ────────────────────────────────────────────────
@dataclass
class Year:
    revenue: float
    cogs: float
    opex: float            # operating expenses excluding COGS (and excl. D&A for Aurora)
    depreciation: float
    finance_costs: float
    tax: float
    eps: float
    cash: float
    inventory: float
    receivables: float
    other_current_assets: float
    non_current_assets: float
    equity: float
    lt_debt: float
    st_debt: float
    payables: float
    other_current_liabilities: float
    other_non_current_liabilities: float
    cfo: float
    capex: float
    cfi: float
    cff: float
    opex_includes_da: bool = False

    @property
    def gross_profit(self) -> float:
        return self.revenue - self.cogs

    @property
    def ebit(self) -> float:
        base = self.gross_profit - self.opex
        return base if self.opex_includes_da else base - self.depreciation

    @property
    def ebitda(self) -> float:
        return self.ebit + self.depreciation

    @property
    def pbt(self) -> float:
        return self.ebit - self.finance_costs

    @property
    def net_income(self) -> float:
        return self.pbt - self.tax

    @property
    def current_assets(self) -> float:
        return self.cash + self.inventory + self.receivables + self.other_current_assets

    @property
    def total_assets(self) -> float:
        return self.current_assets + self.non_current_assets

    @property
    def current_liabilities(self) -> float:
        return self.st_debt + self.payables + self.other_current_liabilities

    @property
    def non_current_liabilities(self) -> float:
        return self.lt_debt + self.other_non_current_liabilities

    @property
    def total_liabilities(self) -> float:
        return self.current_liabilities + self.non_current_liabilities

    @property
    def total_debt(self) -> float:
        return self.lt_debt + self.st_debt


AURORA = "Aurora Industries Limited"
BOREALIS = "Borealis Technologies Inc."

AURORA_YEARS: dict[int, Year] = {
    2023: Year(8200, 5330, 1558, 300, 150, 216, 12.92, 520, 1150, 1020, 310, 5200, 4300, 1500, 500, 1200, 400, 300,
               1050, -780, -800, -190),
    2024: Year(9400, 6063, 1739, 340, 170, 272, 16.32, 610, 1300, 1180, 350, 5960, 4950, 1700, 600, 1350, 450, 350,
               1240, -1100, -1120, -30),
    2025: Year(10600, 7049, 1961, 390, 240, 240, 14.40, 480, 1620, 1420, 380, 7300, 5500, 2600, 900, 1400, 420, 380,
               980, -1730, -1800, 690),
}
AURORA_HISTORY = {  # five-year highlights: revenue, EBITDA, net profit, EPS
    2021: (6300, 945, 470, 9.40),
    2022: (7100, 1100, 560, 11.20),
}

BOREALIS_YEARS: dict[int, Year] = {
    2023: Year(2400, 720, 1200, 120, 20, 92, 1.84, 900, 0, 420, 180, 2100, 2300, 400, 50, 0, 600, 250,
               560, -150, -210, -180, True),
    2024: Year(2900, 841, 1421, 140, 18, 124, 2.48, 1150, 0, 510, 190, 2350, 2750, 350, 50, 0, 750, 300,
               720, -180, -290, -180, True),
    2025: Year(3500, 980, 1680, 165, 15, 165, 3.30, 1500, 0, 610, 240, 2650, 3350, 300, 50, 0, 950, 350,
               930, -220, -330, -250, True),
}


def _self_check() -> None:
    """The statements must balance, otherwise the samples would teach the wrong thing."""
    for name, years in (("Aurora", AURORA_YEARS), ("Borealis", BOREALIS_YEARS)):
        for fy, y in years.items():
            assert abs(y.total_assets - (y.equity + y.total_liabilities)) < 1e-6, f"{name} FY{fy} does not balance"
        ordered = sorted(years)
        for prev, cur in zip(ordered, ordered[1:]):
            delta = years[cur].cfo + years[cur].cfi + years[cur].cff
            assert abs(years[cur].cash - years[prev].cash - delta) < 1e-6, f"{name} FY{cur} cash does not reconcile"


# ── Document builders ────────────────────────────────────────────────────────
def _cover(b: PdfBuilder, company: str, doc_title: str, period_line: str) -> None:
    b.title(company)
    b.heading(doc_title)
    b.para(period_line)
    b.para(
        "IMPORTANT NOTICE: This document is a synthetic sample. The company, its management, and every figure, "
        "statement and event described here are fictitious and were generated solely to test and demonstrate "
        "document-analysis software. Nothing in it describes a real business or security."
    )


def build_aurora_annual(out: Path, fy: int, evals: list[dict]) -> Path:
    cur, prev = AURORA_YEARS[fy], AURORA_YEARS[fy - 1]
    filename = f"aurora_industries_annual_report_fy{fy}.pdf"
    b = PdfBuilder(f"{AURORA} | Annual Report FY{fy}")
    _cover(b, AURORA, f"Annual Report FY{fy}", f"For the financial year ended March 31, {fy}. All amounts in INR crore unless stated otherwise.")

    growth = (cur.revenue / prev.revenue - 1) * 100
    b.heading("Chairman's Letter")
    b.mark("chairman")
    if fy == 2025:
        b.para(
            f"Dear Shareholders, FY{fy} was a year of deliberate investment for {AURORA}. Revenue from operations grew "
            f"{growth:.1f}% to INR {n(cur.revenue)} crore, led by our industrial components and specialty chemicals "
            "businesses. Profitability, however, came under pressure. Net profit declined to INR "
            f"{n(cur.net_income)} crore from INR {n(prev.net_income)} crore as raw material inflation and higher "
            "finance costs more than offset the benefit of volume growth."
        )
        b.para(
            "We chose to continue our capacity expansion through the cycle. The new Pune components plant was "
            "commissioned in the third quarter and the Gujarat specialty chemicals complex is on schedule. These "
            "projects were funded largely through borrowings, which is why our debt rose during the year. The Board "
            "believes these investments position the company well for the next phase of demand."
        )
    else:
        b.para(
            f"Dear Shareholders, FY{fy} was a strong year for {AURORA}. Revenue from operations grew {growth:.1f}% to "
            f"INR {n(cur.revenue)} crore and net profit rose to INR {n(cur.net_income)} crore from INR "
            f"{n(prev.net_income)} crore, supported by operating leverage and stable input costs."
        )
        b.para(
            "During the year the Board approved a multi-year capacity expansion programme covering a new components "
            "plant in Pune and a specialty chemicals complex in Gujarat. Construction began in the fourth quarter."
        )

    b.heading("Management Discussion and Analysis")
    b.subheading("Revenue")
    b.mark("mda_revenue")
    if fy == 2025:
        b.para(
            f"Revenue from operations increased {growth:.1f}% year-on-year to INR {n(cur.revenue)} crore. Industrial "
            "components contributed 58% of revenue and grew on the back of automotive and railway orders. Specialty "
            "chemicals contributed 31% and benefited from new export customers in Southeast Asia. The remaining 11% "
            "came from engineering services. Volume growth accounted for roughly nine percentage points of the "
            "increase and price increases for the remainder."
        )
        b.subheading("Profitability")
        b.mark("mda_profitability")
        b.para(
            "Operating margin declined during the year. Three factors drove the decline. First, the cost of steel "
            "and key chemical intermediates rose sharply in the first half and could only be partly passed on to "
            f"customers, so cost of materials consumed increased to INR {n(cur.cogs)} crore. Second, employee and "
            "start-up costs at the new Pune plant were incurred before the plant reached meaningful utilisation. "
            f"Third, depreciation increased to INR {n(cur.depreciation)} crore as new capacity was capitalised."
        )
        b.para(
            f"Finance costs rose to INR {n(cur.finance_costs)} crore from INR {n(prev.finance_costs)} crore because "
            "of the additional borrowings taken to fund capital expenditure and higher interest rates on floating-rate "
            f"loans. As a result, profit for the year fell to INR {n(cur.net_income)} crore."
        )
        b.subheading("Capital Expenditure")
        b.mark("mda_capex")
        b.para(
            f"Capital expenditure during FY{fy} was INR {n(-cur.capex)} crore, compared with INR {n(-prev.capex)} "
            "crore in the previous year. The Pune components plant accounted for INR 820 crore and the Gujarat "
            "specialty chemicals complex for INR 640 crore, with the balance spent on maintenance and digital "
            "systems. Management plans capital expenditure of approximately INR 1,200 crore in FY2026, the majority "
            "of it to complete the Gujarat complex, after which spending is expected to moderate."
        )
        b.subheading("Outlook")
        b.mark("mda_outlook")
        b.para(
            "Management expects revenue growth of 10% to 12% in FY2026 as the Pune plant ramps up. We expect "
            "operating margin to recover gradually, supported by easing input costs and better utilisation, although "
            "the first half is likely to remain subdued. Reducing leverage is a priority: management intends to bring "
            "the debt-to-equity ratio below 0.5 by FY2027 using operating cash flows once the current investment "
            "cycle is complete. These statements are expectations, not guarantees, and actual results may differ."
        )
    else:
        b.para(
            f"Revenue from operations increased {growth:.1f}% year-on-year to INR {n(cur.revenue)} crore, driven by "
            "strong demand for industrial components from automotive customers and steady growth in specialty "
            "chemicals."
        )
        b.subheading("Profitability")
        b.mark("mda_profitability")
        b.para(
            "Operating margin improved as input costs were stable and fixed costs were spread over higher volumes. "
            f"Finance costs were INR {n(cur.finance_costs)} crore and profit for the year rose to INR "
            f"{n(cur.net_income)} crore."
        )
        b.subheading("Outlook")
        b.mark("mda_outlook")
        b.para(
            "Management expects double-digit revenue growth in FY2025 and plans to step up capital expenditure "
            "significantly to add capacity in Pune and Gujarat, funded through a mix of internal accruals and debt."
        )

    b.heading("Risk Factors")
    b.mark("risks")
    b.para(
        "Raw material price volatility: steel and chemical intermediates represent the majority of our cost base. "
        "Sharp price increases that cannot be passed on to customers adversely affect margins, as experienced in "
        "the first half of the year."
    )
    b.para(
        "Leverage and interest rate risk: borrowings have increased to fund the expansion programme, and a "
        "significant portion carries floating interest rates. A further rise in rates, or a delay in the ramp-up of "
        "new capacity, would reduce interest coverage and could constrain future investment."
    )
    b.para(
        "Project execution risk: the Gujarat specialty chemicals complex is a large greenfield project. Delays or "
        "cost overruns would defer the expected returns and increase capital employed."
    )
    b.para(
        "Customer concentration: the five largest customers account for about 38% of revenue. The loss of, or a "
        "significant reduction in orders from, any of them would adversely affect revenue and cash flows."
    )
    b.para(
        "Regulatory and environmental risk: our chemicals operations are subject to evolving environmental "
        "regulation. Non-compliance could result in penalties or temporary suspension of operations."
    )

    if fy == 2025:
        b.new_page()
        b.heading("Five-Year Financial Highlights")
        b.mark("highlights")
        b.para("(INR in crore, except per share data)")
        years = [2021, 2022, 2023, 2024, 2025]
        series = {y: AURORA_HISTORY[y] for y in (2021, 2022)}
        for y in (2023, 2024, 2025):
            d = AURORA_YEARS[y]
            series[y] = (d.revenue, d.ebitda, d.net_income, d.eps)
        b.table(
            ["Particulars", *[f"FY{y}" for y in years]],
            [
                ["Revenue from operations", *[n(series[y][0]) for y in years]],
                ["EBITDA", *[n(series[y][1]) for y in years]],
                ["Profit for the year", *[n(series[y][2]) for y in years]],
                ["Earnings per share (INR)", *[n(series[y][3], 2) for y in years]],
            ],
            label_width=160,
        )

    b.new_page()
    b.heading("Consolidated Statement of Profit and Loss")
    b.mark("pnl")
    b.para("(INR in crore, except per share data)")
    cols = ["Particulars", f"Year ended March 31, {fy}", f"Year ended March 31, {fy - 1}"]

    def pair(getter) -> list[str]:  # noqa: ANN001
        return [n(getter(cur)), n(getter(prev))]

    b.table(cols, [
        ["Revenue from operations", *pair(lambda y: y.revenue)],
        ["Cost of materials consumed", *pair(lambda y: y.cogs)],
        ["Gross profit", *pair(lambda y: y.gross_profit)],
        ["Employee benefits and other expenses", *pair(lambda y: y.opex)],
        ["EBITDA", *pair(lambda y: y.ebitda)],
        ["Depreciation and amortisation expense", *pair(lambda y: y.depreciation)],
        ["Operating profit", *pair(lambda y: y.ebit)],
        ["Finance costs", *pair(lambda y: y.finance_costs)],
        ["Profit before tax", *pair(lambda y: y.pbt)],
        ["Tax expense", *pair(lambda y: y.tax)],
        ["Profit for the year", *pair(lambda y: y.net_income)],
        ["Earnings per equity share"],
        ["Basic (INR)", n(cur.eps, 2), n(prev.eps, 2)],
    ])

    b.new_page()
    b.heading("Consolidated Balance Sheet")
    b.mark("balance_sheet")
    b.para("(INR in crore)")
    bs_cols = ["Particulars", f"As at March 31, {fy}", f"As at March 31, {fy - 1}"]
    # Deliberately unruled: exercises text-table recovery and section-aware "Borrowings" mapping.
    b.table(bs_cols, [
        ["ASSETS"],
        ["Non-current assets"],
        ["Property, plant and equipment and other assets", *pair(lambda y: y.non_current_assets)],
        ["Current assets"],
        ["Inventories", *pair(lambda y: y.inventory)],
        ["Trade receivables", *pair(lambda y: y.receivables)],
        ["Cash and cash equivalents", *pair(lambda y: y.cash)],
        ["Other current assets", *pair(lambda y: y.other_current_assets)],
        ["Total current assets", *pair(lambda y: y.current_assets)],
        ["Total assets", *pair(lambda y: y.total_assets)],
        ["EQUITY AND LIABILITIES"],
        ["Total equity", *pair(lambda y: y.equity)],
        ["Non-current liabilities"],
        ["Borrowings", *pair(lambda y: y.lt_debt)],
        ["Other non-current liabilities", *pair(lambda y: y.other_non_current_liabilities)],
        ["Total non-current liabilities", *pair(lambda y: y.non_current_liabilities)],
        ["Current liabilities"],
        ["Borrowings", *pair(lambda y: y.st_debt)],
        ["Trade payables", *pair(lambda y: y.payables)],
        ["Other current liabilities", *pair(lambda y: y.other_current_liabilities)],
        ["Total current liabilities", *pair(lambda y: y.current_liabilities)],
        ["Total liabilities", *pair(lambda y: y.total_liabilities)],
        ["Total equity and liabilities", *pair(lambda y: y.total_assets)],
    ], ruled=False)

    b.new_page()
    b.heading("Consolidated Statement of Cash Flows")
    b.mark("cash_flow")
    b.para("(INR in crore)")
    b.table(cols, [
        ["Net cash generated from operating activities", *pair(lambda y: y.cfo)],
        ["Purchase of property, plant and equipment", *pair(lambda y: y.capex)],
        ["Net cash used in investing activities", *pair(lambda y: y.cfi)],
        ["Net cash from financing activities", *pair(lambda y: y.cff)],
    ])
    if fy == 2025:
        b.para(
            f"Net cash generated from operating activities decreased to INR {n(cur.cfo)} crore from INR "
            f"{n(prev.cfo)} crore, mainly because inventories and trade receivables increased as the new plant "
            "built up working capital. After capital expenditure, free cash flow was negative for the year and the "
            "shortfall was met through additional borrowings."
        )

    b.heading("Significant Accounting Policies")
    b.mark("policies")
    b.para(
        "Revenue is recognised when control of goods is transferred to the customer, generally on dispatch, at the "
        "transaction price net of returns and discounts. Property, plant and equipment is stated at cost less "
        "accumulated depreciation and is depreciated on a straight-line basis over its estimated useful life. "
        "Inventories are valued at the lower of cost, determined on a weighted average basis, and net realisable value."
    )
    path = out / filename
    marks = dict(b.marks)
    b.save(path)

    if fy == 2025:
        de = cur.total_debt / cur.equity
        evals.extend([
            _eval("What was Aurora Industries' revenue from operations in FY2025?",
                  f"Revenue from operations was INR {n(cur.revenue)} crore in FY2025.", filename, marks["pnl"],
                  [n(cur.revenue)]),
            _eval("Why did Aurora Industries' operating margin decline in FY2025?",
                  "Operating margin declined because raw material (steel and chemical intermediates) costs rose and "
                  "could only be partly passed on, start-up and employee costs were incurred at the new Pune plant "
                  "before it was fully utilised, and depreciation increased as new capacity was capitalised.",
                  filename, marks["mda_profitability"], ["steel", "Pune", "depreciation"]),
            _eval("What are Aurora Industries' biggest risks?",
                  "Key risks are raw material price volatility, leverage and interest rate risk, project execution "
                  "risk at the Gujarat complex, customer concentration, and regulatory and environmental risk.",
                  filename, marks["risks"], ["raw material", "interest rate", "customer concentration"]),
            _eval("What are Aurora Industries' capital expenditure plans?",
                  f"Capital expenditure was INR {n(-cur.capex)} crore in FY2025; management plans about INR 1,200 "
                  "crore in FY2026, mostly to complete the Gujarat specialty chemicals complex.",
                  filename, marks["mda_capex"], ["1,200", "Gujarat"]),
            _eval("What is Aurora Industries' debt-to-equity ratio for FY2025?",
                  f"Total debt of INR {n(cur.total_debt)} crore against equity of INR {n(cur.equity)} crore gives a "
                  f"debt-to-equity ratio of {de:.2f}x.", filename, marks["balance_sheet"], [f"{de:.2f}"]),
            _eval("What does Aurora Industries' management expect for next year?",
                  "Management expects revenue growth of 10% to 12% in FY2026, a gradual recovery in operating "
                  "margin, and aims to bring debt-to-equity below 0.5 by FY2027.",
                  filename, marks["mda_outlook"], ["10%", "12%"]),
            _eval("Is there evidence of declining cash flows at Aurora Industries?",
                  f"Yes. Net cash generated from operating activities fell to INR {n(cur.cfo)} crore from INR "
                  f"{n(prev.cfo)} crore and free cash flow was negative after capital expenditure.",
                  filename, marks["cash_flow"], [n(cur.cfo), n(prev.cfo)]),
            _eval("How has Aurora Industries' revenue grown over the last five years?",
                  "Revenue grew from INR 6,300 crore in FY2021 to INR 10,600 crore in FY2025.",
                  filename, marks["highlights"], ["6,300", "10,600"]),
        ])
    return path


def build_borealis_annual(out: Path, fy: int, evals: list[dict]) -> Path:
    cur, prev = BOREALIS_YEARS[fy], BOREALIS_YEARS[fy - 1]
    filename = f"borealis_technologies_annual_report_fy{fy}.pdf"
    b = PdfBuilder(f"{BOREALIS} | Annual Report FY{fy}")
    _cover(b, BOREALIS, f"Annual Report FY{fy}", f"For the fiscal year ended December 31, {fy}. All amounts in USD millions unless stated otherwise.")
    growth = (cur.revenue / prev.revenue - 1) * 100

    b.heading("Letter to Shareholders")
    b.mark("letter")
    b.para(
        f"Fiscal {fy} was another year of profitable growth for {BOREALIS}. Total revenue increased {growth:.1f}% to "
        f"USD {n(cur.revenue)} million, driven by subscription revenue from our cloud data platform, which now "
        f"represents 82% of total revenue. Net income rose to USD {n(cur.net_income)} million from USD "
        f"{n(prev.net_income)} million as we scaled efficiently."
    )
    b.para(
        "We continued to reduce debt and ended the year with a strong net cash position. Our capital-light model "
        "means most of our operating cash flow converts into free cash flow."
    )

    b.heading("Management Discussion and Analysis")
    b.subheading("Revenue")
    b.mark("mda_revenue")
    b.para(
        f"Total revenue was USD {n(cur.revenue)} million, an increase of {growth:.1f}%. Growth came from expansion "
        "within existing enterprise customers, with net revenue retention of 118%, and from 310 new enterprise "
        "customers added during the year. Professional services revenue was broadly flat."
    )
    b.subheading("Profitability")
    b.mark("mda_profitability")
    b.para(
        "Operating margin expanded as hosting costs grew more slowly than revenue following the migration to a "
        "multi-tenant architecture, and as sales and marketing expenses declined as a percentage of revenue. "
        f"Operating income increased to USD {n(cur.ebit)} million. Interest expense declined to USD "
        f"{n(cur.finance_costs)} million as term debt was repaid."
    )
    b.subheading("Outlook")
    b.mark("mda_outlook")
    b.para(
        f"For fiscal {fy + 1}, management expects revenue growth of 17% to 19% and a broadly stable operating "
        "margin as the company increases investment in artificial intelligence features and international "
        "expansion. This outlook reflects current expectations and is subject to the risks described below."
    )

    b.heading("Risk Factors")
    b.mark("risks")
    b.para(
        "Competition: the market for cloud data platforms is intensely competitive and includes much larger "
        "companies with greater resources. Pricing pressure could adversely affect growth and margins."
    )
    b.para(
        "Cybersecurity and data privacy: a security breach or service outage could damage our reputation, expose "
        "us to liability and cause customers to reduce or terminate their subscriptions."
    )
    b.para(
        "Dependence on third-party cloud infrastructure: we rely on a small number of hyperscale providers. "
        "Price increases or disruptions at these providers would adversely affect our cost of revenue and service "
        "availability."
    )
    b.para(
        "Talent: our success depends on attracting and retaining highly skilled engineers. Rising compensation "
        "costs could put pressure on operating margin."
    )

    b.new_page()
    b.heading("Consolidated Statements of Operations")
    b.mark("pnl")
    b.para("(USD in millions, except per share data)")
    cols = ["Particulars", f"FY{fy}", f"FY{fy - 1}"]

    def pair(getter) -> list[str]:  # noqa: ANN001
        return [n(getter(cur)), n(getter(prev))]

    b.table(cols, [
        ["Total revenue", *pair(lambda y: y.revenue)],
        ["Cost of revenue", *pair(lambda y: y.cogs)],
        ["Gross profit", *pair(lambda y: y.gross_profit)],
        ["Total operating expenses", *pair(lambda y: y.opex)],
        ["Operating income", *pair(lambda y: y.ebit)],
        ["Depreciation and amortization", *pair(lambda y: y.depreciation)],
        ["Interest expense", *pair(lambda y: y.finance_costs)],
        ["Income before income taxes", *pair(lambda y: y.pbt)],
        ["Provision for income taxes", *pair(lambda y: y.tax)],
        ["Net income", *pair(lambda y: y.net_income)],
        ["Basic earnings per share", n(cur.eps, 2), n(prev.eps, 2)],
    ])

    b.heading("Consolidated Balance Sheets")
    b.mark("balance_sheet")
    b.para("(USD in millions)")
    b.table(cols, [
        ["Cash and cash equivalents", *pair(lambda y: y.cash)],
        ["Accounts receivable", *pair(lambda y: y.receivables)],
        ["Total current assets", *pair(lambda y: y.current_assets)],
        ["Total assets", *pair(lambda y: y.total_assets)],
        ["Short-term debt", *pair(lambda y: y.st_debt)],
        ["Total current liabilities", *pair(lambda y: y.current_liabilities)],
        ["Long-term debt", *pair(lambda y: y.lt_debt)],
        ["Total liabilities", *pair(lambda y: y.total_liabilities)],
        ["Total stockholders' equity", *pair(lambda y: y.equity)],
    ])

    b.new_page()
    b.heading("Consolidated Statements of Cash Flows")
    b.mark("cash_flow")
    b.para("(USD in millions)")
    b.table(cols, [
        ["Net cash provided by operating activities", *pair(lambda y: y.cfo)],
        ["Purchases of property and equipment", *pair(lambda y: y.capex)],
        ["Net cash used in investing activities", *pair(lambda y: y.cfi)],
        ["Net cash used in financing activities", *pair(lambda y: y.cff)],
    ], ruled=False)
    b.para(
        f"Net cash provided by operating activities increased to USD {n(cur.cfo)} million. Capital expenditures "
        f"were USD {n(-cur.capex)} million, primarily for data centre equipment and office facilities."
    )
    path = out / filename
    marks = dict(b.marks)
    b.save(path)

    if fy == 2025:
        evals.extend([
            _eval("What was Borealis Technologies' net income in FY2025?",
                  f"Net income was USD {n(cur.net_income)} million in FY2025.", filename, marks["pnl"],
                  [n(cur.net_income)]),
            _eval("What drove Borealis Technologies' revenue growth?",
                  "Growth came from expansion within existing enterprise customers (net revenue retention of 118%) "
                  "and 310 new enterprise customers.", filename, marks["mda_revenue"], ["118%", "310"]),
            _eval("What risks does Borealis Technologies face?",
                  "Competition, cybersecurity and data privacy, dependence on third-party cloud infrastructure, "
                  "and talent retention.", filename, marks["risks"], ["competition", "cybersecurity"]),
        ])
    return path


def build_aurora_transcript(out: Path, evals: list[dict]) -> Path:
    filename = "aurora_industries_q4_fy2025_earnings_call_transcript.pdf"
    b = PdfBuilder(f"{AURORA} | Q4 FY2025 Earnings Call Transcript")
    _cover(b, AURORA, "Q4 FY2025 Earnings Conference Call Transcript", "Call held in May 2025 to discuss results for the quarter and year ended March 31, 2025.")

    b.heading("Prepared Remarks")
    b.mark("remarks")
    b.para(
        "Moderator: Good afternoon and welcome to the Aurora Industries Q4 FY2025 earnings conference call. "
        "We have with us Ms. Kavya Raman, Managing Director, and Mr. Dev Malhotra, Chief Financial Officer."
    )
    b.para(
        "Kavya Raman, Managing Director: Thank you. We closed the year with revenue of INR 10,600 crore, up about "
        "13%. I am pleased with the order book, which stands at a record INR 4,200 crore and gives us good "
        "visibility for the coming year. The Pune plant reached 55% utilisation in March and we expect it to "
        "cross 80% by the third quarter of FY2026. I want to be candid that margins this year were disappointing. "
        "Input costs moved against us faster than we could reprice contracts."
    )
    b.para(
        "Dev Malhotra, Chief Financial Officer: On the numbers, net profit for the year was INR 720 crore against "
        "INR 816 crore last year. The decline is explained by three items: lower gross margin on account of raw "
        "material inflation, higher depreciation on the new plant, and finance costs which rose by INR 70 crore. "
        "Net debt increased because we funded the capex programme largely through borrowings. We have hedged "
        "roughly 60% of our floating-rate exposure for the next two years."
    )

    b.heading("Question-and-Answer Session")
    b.mark("qa")
    b.para(
        "Analyst: Can you help us understand when margins recover? Dev Malhotra: Steel prices have softened about "
        "8% from the peak and roughly two thirds of our contracts now carry quarterly price pass-through clauses. "
        "We expect operating margin to improve by 100 to 150 basis points in FY2026, weighted to the second half."
    )
    b.para(
        "Analyst: Leverage has gone up meaningfully. Are you comfortable? Kavya Raman: We are comfortable but not "
        "complacent. This is peak debt. Capital expenditure falls to about INR 1,200 crore next year and we do "
        "not plan any new large projects until debt-to-equity is back below 0.5. We are not considering an equity "
        "raise."
    )
    b.para(
        "Analyst: Any concerns on working capital? Dev Malhotra: Inventory days went up as we stocked the new "
        "plant, and receivables rose with the railway orders, which have longer payment cycles. That is why "
        "operating cash flow dipped. We are targeting a reduction of ten inventory days in FY2026."
    )
    b.para(
        "Analyst: What is the biggest thing that could go wrong? Kavya Raman: Execution at Gujarat. It is a complex "
        "project and a delay of even two quarters would push out returns. We have appointed a dedicated project "
        "director and the project is currently on schedule and within budget."
    )
    path = out / filename
    marks = dict(b.marks)
    b.save(path)
    evals.extend([
        _eval("What did Aurora Industries' management say about margin recovery on the earnings call?",
              "The CFO expects operating margin to improve by 100 to 150 basis points in FY2026, weighted to the "
              "second half, helped by softer steel prices and price pass-through clauses.",
              filename, marks["qa"], ["100 to 150 basis points"]),
        _eval("What is Aurora Industries' order book?",
              "The order book stands at a record INR 4,200 crore.", filename, marks["remarks"], ["4,200"]),
    ])
    return path


def _eval(question: str, answer: str, document: str, page: int, key_facts: list[str]) -> dict:
    return {
        "question": question,
        "expected_answer": answer,
        "relevant_document": document,
        "relevant_page": page,
        "key_facts": key_facts,
    }


def generate(out: Path, dataset_path: Path | None = None) -> list[Path]:
    """Write all sample documents to ``out`` and (optionally) the evaluation dataset."""
    _self_check()
    out.mkdir(parents=True, exist_ok=True)
    evals: list[dict] = []
    paths = [
        build_aurora_annual(out, 2025, evals),
        build_aurora_annual(out, 2024, evals),
        build_aurora_transcript(out, evals),
        build_borealis_annual(out, 2025, evals),
    ]
    if dataset_path is not None:
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with dataset_path.open("w", encoding="utf-8") as handle:
            for row in evals:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "samples")
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "evaluation" / "dataset.jsonl")
    args = parser.parse_args()
    for path in generate(args.out, args.dataset):
        print(f"wrote {path}")
    print(f"wrote {args.dataset}")


if __name__ == "__main__":
    main()
