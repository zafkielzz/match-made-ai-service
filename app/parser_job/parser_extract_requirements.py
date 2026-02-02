"""
Requirements extraction with constraint parsing
"""
import re
from typing import Dict, List, Optional, Tuple


def extract_numeric_constraints(text: str) -> Dict[str, any]:
    """
    Extract numeric constraints from requirements text.
    Examples:
    - GPA: >= 3.2/4.0
    - TOEIC: >= 600
    - Time: >= 28 hours/week or >= 70% sessions/week
    """
    constraints = {}
    
    # GPA pattern: >= 3.2/4.0, > 3.0, GPA 3.5+
    gpa_match = re.search(
        r"(?:gpa|điểm trung bình)[\s:]*(?:>=|>|≥)?\s*(\d+\.?\d*)\s*(?:/\s*(\d+\.?\d*))?",
        text.lower()
    )
    if gpa_match:
        gpa_value = float(gpa_match.group(1))
        gpa_scale = float(gpa_match.group(2)) if gpa_match.group(2) else 4.0
        constraints["gpa"] = {
            "min": gpa_value,
            "scale": gpa_scale
        }
    
    # TOEIC pattern: >= 600, TOEIC 650+
    toeic_match = re.search(
        r"toeic[\s:]*(?:>=|>|≥)?\s*(\d{3,4})",
        text.lower()
    )
    if toeic_match:
        constraints["toeic"] = {
            "min": int(toeic_match.group(1))
        }
    
    # IELTS pattern: >= 6.5
    ielts_match = re.search(
        r"ielts[\s:]*(?:>=|>|≥)?\s*(\d+\.?\d*)",
        text.lower()
    )
    if ielts_match:
        constraints["ielts"] = {
            "min": float(ielts_match.group(1))
        }
    
    # Time commitment: >= 28 hours/week, >= 70% sessions
    time_match = re.search(
        r"(?:>=|>|≥|ít nhất|at least)\s*(\d+)\s*(hours?|giờ|%)\s*(?:/|per)?\s*(week|tuần|sessions?|buổi)?",
        text.lower()
    )
    if time_match:
        value = int(time_match.group(1))
        unit = time_match.group(2)
        period = time_match.group(3) or "week"
        
        if "%" in unit:
            constraints["timeCommitment"] = {
                "min": value,
                "unit": "percent",
                "period": period
            }
        else:
            constraints["timeCommitment"] = {
                "min": value,
                "unit": "hours",
                "period": period
            }
    
    return constraints


def extract_requirements_with_constraints(
    block: str,
    extract_bullets_fn
) -> Tuple[List[str], List[str], Dict[str, any]]:
    """
    Extract requirements and parse numeric constraints.
    Returns: (required_bullets, preferred_bullets, constraints)
    """
    from app.parser_job.parser_sections import REQUIREMENTS_STOP_KEYWORDS
    
    bullets = extract_bullets_fn(block, limit=60, stop_on_keywords=REQUIREMENTS_STOP_KEYWORDS)
    
    required = []
    preferred = []
    
    for item in bullets:
        item_lower = item.lower()
        if item_lower.startswith(("ưu tiên", "preferred", "lợi thế", "nice to have", "plus")):
            preferred.append(item)
        else:
            required.append(item)
    
    # Extract constraints from full block
    constraints = extract_numeric_constraints(block)
    
    return required, preferred, constraints
