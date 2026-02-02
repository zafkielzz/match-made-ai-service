"""
Types and data structures for job parser
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from pydantic import BaseModel


class ParseDefaults(BaseModel):
    countryCode: str = "VN"
    currency: str = "VND"
    salaryPeriod: str = "MONTH"  # MONTH/YEAR/DAY/HOUR
    salaryType: str = "GROSS"    # GROSS/NET


@dataclass
class ParseResult:
    detected_source: str
    suggested: Dict[str, Any]
    confidence: Dict[str, float]
    warnings: List[str]
