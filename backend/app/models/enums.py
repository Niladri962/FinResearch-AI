from __future__ import annotations

from enum import Enum


class DocumentStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DocumentType(str, Enum):
    ANNUAL_REPORT = "annual_report"
    QUARTERLY_REPORT = "quarterly_report"
    EARNINGS_CALL = "earnings_call_transcript"
    INVESTOR_PRESENTATION = "investor_presentation"
    FINANCIAL_STATEMENT = "financial_statement"
    MDA = "management_discussion"
    RESEARCH_REPORT = "research_report"
    COMPANY_FILING = "company_filing"
    OTHER = "other"


DOCUMENT_TYPE_LABELS: dict[str, str] = {
    DocumentType.ANNUAL_REPORT.value: "Annual Report",
    DocumentType.QUARTERLY_REPORT.value: "Quarterly Report",
    DocumentType.EARNINGS_CALL.value: "Earnings Call Transcript",
    DocumentType.INVESTOR_PRESENTATION.value: "Investor Presentation",
    DocumentType.FINANCIAL_STATEMENT.value: "Financial Statements",
    DocumentType.MDA.value: "Management Discussion & Analysis",
    DocumentType.RESEARCH_REPORT.value: "Research Report",
    DocumentType.COMPANY_FILING.value: "Company Filing",
    DocumentType.OTHER.value: "Document",
}


class ChunkType(str, Enum):
    FINANCIAL_STATEMENT = "financial_statement"
    TABLE = "table"
    RISK_FACTOR = "risk_factor"
    MANAGEMENT_COMMENTARY = "management_commentary"
    EARNINGS_COMMENTARY = "earnings_commentary"
    ACCOUNTING_POLICY = "accounting_policy"
    NOTES = "notes"
    NARRATIVE = "narrative"


class Intent(str, Enum):
    DOCUMENT_QA = "DOCUMENT_QA"
    FINANCIAL_CALCULATION = "FINANCIAL_CALCULATION"
    COMPANY_COMPARISON = "COMPANY_COMPARISON"
    PERIOD_COMPARISON = "PERIOD_COMPARISON"
    RISK_ANALYSIS = "RISK_ANALYSIS"
    MANAGEMENT_ANALYSIS = "MANAGEMENT_ANALYSIS"
    SUMMARY = "SUMMARY"
    TREND_ANALYSIS = "TREND_ANALYSIS"
    GENERAL_FINANCE = "GENERAL_FINANCE"
