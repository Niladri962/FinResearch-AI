"""Reporting-period recognition.

Normalises the many ways filings label a period ("FY25", "FY 2024-25",
"Year ended March 31, 2025", "Q1 FY2025", "2024") into a canonical ``Period``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_MONTH_ABBR = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# Whole month names only — a loose "mar[a-z]*" would also match "margin 2024".
_MONTH_RE = (
    r"(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b\.?"
)


@dataclass(frozen=True)
class Period:
    label: str
    fiscal_year: int
    quarter: int | None = None
    kind: str = "annual"  # "annual" | "quarter" | "interim"
    month: int | None = field(default=None, compare=False)  # set when parsed from a date

    def as_interim(self) -> Period:
        """Relabel a date-derived period that heads a quarterly/half-year column."""
        if self.kind != "annual":
            return self
        month = _MONTH_ABBR[self.month] if self.month else "Period"
        return Period(f"Interim {month} {self.fiscal_year}", self.fiscal_year, None, "interim", self.month)

    @property
    def sort_key(self) -> tuple[int, int]:
        return (self.fiscal_year, self.quarter or 5)


def _year(raw: str) -> int | None:
    year = int(raw)
    if len(raw) == 2:
        year += 2000 if year < 80 else 1900
    return year if 1980 <= year <= 2100 else None


def annual(year: int) -> Period:
    return Period(label=f"FY{year}", fiscal_year=year)


def quarterly(quarter: int, year: int) -> Period:
    return Period(label=f"Q{quarter} FY{year}", fiscal_year=year, quarter=quarter, kind="quarter")


def _from_quarter(m: re.Match[str]) -> Period | None:
    year = _year(m.group("y"))
    return quarterly(int(m.group("q")), year) if year else None


def _from_fy_range(m: re.Match[str]) -> Period | None:
    start = _year(m.group(1))
    if start is None:
        return None
    end_raw = m.group(2)
    end = int(end_raw) if len(end_raw) == 4 else (start // 100) * 100 + int(end_raw)
    # "FY2024-25" means the fiscal year ending in 2025; reject things like "2024-12".
    return annual(end) if end == start + 1 else None


def _from_fy(m: re.Match[str]) -> Period | None:
    year = _year(m.group(1))
    return annual(year) if year else None


def _from_date(m: re.Match[str]) -> Period | None:
    year = _year(m.group("y"))
    if year is None:
        return None
    month = _MONTH_ABBR.index(m.group("mon")[:3].title())
    return Period(label=f"FY{year}", fiscal_year=year, month=month)


# (pattern, builder) in priority order — earlier patterns win on overlapping matches.
_PATTERNS: list[tuple[re.Pattern[str], object]] = [
    (re.compile(r"\bQ(?P<q>[1-4])\s*[-/ ]?\s*(?:FY)?\s*'?(?P<y>\d{4}|\d{2})\b", re.I), _from_quarter),
    (re.compile(r"\bFY\s*'?(?P<y>\d{4}|\d{2})\s*[-/ ]?\s*Q(?P<q>[1-4])\b", re.I), _from_quarter),
    (re.compile(r"\bFY\s*'?(\d{4})\s*[-–/]\s*(\d{4}|\d{2})\b", re.I), _from_fy_range),
    (re.compile(r"\bFY\s*'?(\d{4}|\d{2})\b", re.I), _from_fy),
    (re.compile(r"(?<![\d,.])((?:19|20)\d{2})\s*[-–/]\s*(\d{2})(?![\d,.])"), _from_fy_range),
    (
        re.compile(
            rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)?\s+)?{_MONTH_RE}\s*(?:\d{{1,2}}(?:st|nd|rd|th)?)?,?\s*(?P<y>(?:19|20)\d{{2}})\b",
            re.I,
        ),
        _from_date,
    ),
    (re.compile(r"(?<![\d,.\-/])((?:19|20)\d{2})(?![\d,%/\-]|\.\d)"), _from_fy),
]


def find_periods(text: str) -> list[Period]:
    """Return every period mentioned in ``text``, in reading order."""
    candidates: list[tuple[int, int, int, Period]] = []
    for priority, (pattern, builder) in enumerate(_PATTERNS):
        for match in pattern.finditer(text):
            period = builder(match)  # type: ignore[operator]
            if period is not None:
                candidates.append((match.start(), priority, match.end(), period))
    candidates.sort(key=lambda c: (c[1], c[0]))
    taken: list[tuple[int, int]] = []
    accepted: list[tuple[int, Period]] = []
    for start, _priority, end, period in candidates:
        if any(start < t_end and end > t_start for t_start, t_end in taken):
            continue
        taken.append((start, end))
        accepted.append((start, period))
    accepted.sort(key=lambda a: a[0])
    return [p for _, p in accepted]


def parse_period(text: str) -> Period | None:
    periods = find_periods(text)
    return periods[0] if periods else None


def strip_periods(text: str) -> str:
    """Remove period expressions — used to decide whether a line is a pure header row."""
    for pattern, _ in _PATTERNS:
        text = pattern.sub(" ", text)
    return text


def period_from_label(label: str) -> Period | None:
    """Parse a canonical label ("FY2025", "Q2 FY2025") or anything ``find_periods`` accepts."""
    return parse_period(label.strip())
