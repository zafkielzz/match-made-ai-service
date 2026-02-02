"""
Salary extraction and parsing
"""
import re
from typing import Any, Dict, Tuple

from .parser_types import ParseDefaults


def extract_salary(cleaned_text: str, defaults: ParseDefaults) -> Tuple[Dict[str, Any], float]:
    """
    Extract salary information from text.
    Supports:
    - Negotiable (thương lượng, thỏa thuận)
    - Range (10-20tr, 10 - 20 triệu)
    - Single value (15tr, 15 triệu)
    """
    lower = cleaned_text.lower()
    salary = {
        "min": 0,
        "max": 0,
        "currency": defaults.currency,
        "period": defaults.salaryPeriod,
        "type": defaults.salaryType,
        "negotiable": False
    }

    # Check negotiable
    if "thương lượng" in lower or "thoả thuận" in lower or "thỏa thuận" in lower:
        salary["negotiable"] = True
        return salary, 0.9

    # Range: 10-20tr, 10 - 20 triệu
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*[-–]\s*(\d+(?:[.,]\d+)?)\s*(tr|triệu)\b", lower)
    if m:
        a = float(m.group(1).replace(",", "."))
        b = float(m.group(2).replace(",", "."))
        salary["min"] = int(a * 1_000_000)
        salary["max"] = int(b * 1_000_000)
        return salary, 0.8

    # Single value: 15tr, 15 triệu
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(tr|triệu)\b", lower)
    if m:
        a = float(m.group(1).replace(",", "."))
        v = int(a * 1_000_000)
        salary["min"] = v
        salary["max"] = v
        return salary, 0.6

    return salary, 0.1
