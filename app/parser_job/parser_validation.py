"""
Validation and quality checks for parsed results
"""
import re
from typing import Dict, List, Any


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple word-based similarity between two texts"""
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def validate_overview_uniqueness(overview: str, responsibilities: List[str]) -> bool:
    """
    Check that overview is not copied from responsibilities.
    Returns True if overview is unique (< 70% similarity with any responsibility)
    """
    if not overview or not responsibilities:
        return True
    
    for resp in responsibilities:
        similarity = calculate_text_similarity(overview, resp)
        if similarity > 0.7:
            return False
    
    return True


def validate_parsed_result(suggested: Dict[str, Any], confidence: Dict[str, float]) -> List[str]:
    """
    Validate parsed result against acceptance criteria.
    Returns list of validation errors (empty if all pass).
    """
    errors = []
    
    # 1. Responsibilities length >= 4
    responsibilities = suggested.get("responsibilities", [])
    if len(responsibilities) < 4:
        errors.append(f"responsibilities.length={len(responsibilities)} < 4 (minimum required)")
    
    # 2. Requirements.required length >= 4
    requirements = suggested.get("requirements", {})
    required = requirements.get("required", [])
    if len(required) < 4:
        errors.append(f"requirements.required.length={len(required)} < 4 (minimum required)")
    
    # 3. No location duplicates (check by city code)
    locations = suggested.get("locations", [])
    city_codes = [loc.get("city", {}).get("code") for loc in locations if loc.get("city")]
    if len(city_codes) != len(set(city_codes)):
        errors.append("Location duplicates detected (same city.code)")
    
    # 4. Overview not copied from responsibilities
    overview = suggested.get("overview", "")
    if overview and responsibilities:
        if not validate_overview_uniqueness(overview, responsibilities):
            errors.append("Overview is too similar to responsibilities (>70% match)")
    
    # 5. Benefits not empty if section detected
    benefits = suggested.get("benefits", {}).get("custom", [])
    # This check is done in parser with warnings
    
    # 6. Employment type handling for INTERN
    job_level = suggested.get("jobLevel")
    employment_type = suggested.get("employmentType")
    if job_level == "INTERN" and employment_type in ["FULL_TIME", "PART_TIME"]:
        # This is allowed if explicitly stated in JD
        pass
    
    # 7. Confidence reflects output quality
    if not responsibilities and confidence.get("responsibilities", 0) > 0.6:
        errors.append("responsibilities empty but confidence > 0.6")
    
    if not benefits and confidence.get("benefits.custom", 0) > 0.6:
        errors.append("benefits.custom empty but confidence > 0.6")
    
    return errors


def adjust_confidence_based_on_quality(
    suggested: Dict[str, Any],
    confidence: Dict[str, float]
) -> Dict[str, float]:
    """
    Adjust confidence scores based on actual output quality.
    """
    adjusted = confidence.copy()
    
    # Lower confidence if field is empty
    if not suggested.get("responsibilities"):
        adjusted["responsibilities"] = 0.0
    
    if not suggested.get("requirements", {}).get("required"):
        adjusted["requirements.required"] = 0.0
    
    if not suggested.get("benefits", {}).get("custom"):
        adjusted["benefits.custom"] = 0.0
    
    # Lower confidence if overview is too short
    overview = suggested.get("overview", "")
    if overview and len(overview) < 50:
        adjusted["overview"] = min(adjusted.get("overview", 0), 0.4)
    
    return adjusted
