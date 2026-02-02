"""
Benefits extraction and normalization
"""
import re
from typing import List


BENEFIT_KEYWORDS_MAP = {
    # Mentoring & Training
    "mentor": ["mentor 1:1", "mentor 1-1", "hướng dẫn 1:1"],
    "đào tạo": ["đào tạo", "training", "học tập"],
    "review": ["review định kỳ", "đánh giá định kỳ", "performance review"],
    
    # Career development
    "dự án": ["tham gia dự án", "làm việc với dự án", "project"],
    "lên chính thức": ["cơ hội lên chính thức", "chuyển chính thức", "full-time opportunity"],
    
    # Compensation & Benefits
    "ăn trưa": ["ăn trưa miễn phí", "free lunch", "canteen", "suất ăn"],
    "bảo hiểm": ["bảo hiểm", "insurance", "bhxh", "bhyt"],
    "thưởng": ["thưởng", "bonus", "lương tháng 13"],
    
    # Work environment
    "văn hoá": ["hoạt động clb", "văn hoá", "team building", "sự kiện"],
    "thiết bị": ["laptop", "máy tính", "thiết bị làm việc"],
    "flexible": ["linh hoạt", "flexible", "remote"],
}


def normalize_benefit_item(item: str) -> str:
    """
    Normalize benefit text to shorter, cleaner format.
    Examples:
    - "Được hướng dẫn 1:1 bởi mentor" → "Mentor 1:1"
    - "Tham gia các dự án AI/Big Data thực tế" → "Tham gia dự án AI/Big Data thực tế"
    """
    item = item.strip()
    item_lower = item.lower()
    
    # Check for known patterns and normalize
    for keyword, patterns in BENEFIT_KEYWORDS_MAP.items():
        for pattern in patterns:
            if pattern in item_lower:
                # Keep the original casing but shorten
                if "mentor" in pattern:
                    return "Mentor 1:1"
                elif "review" in pattern:
                    return "Review định kỳ"
                elif "đào tạo" in pattern:
                    if "soft" in item_lower or "technical" in item_lower:
                        return "Đào tạo soft + technical skills"
                    return "Đào tạo và phát triển"
                elif "dự án" in pattern:
                    # Keep project type if mentioned
                    if "ai" in item_lower or "big data" in item_lower:
                        return "Tham gia dự án AI/Big Data thực tế"
                    return "Tham gia dự án thực tế"
                elif "lên chính thức" in pattern:
                    return "Cơ hội lên chính thức"
                elif "ăn trưa" in pattern:
                    return "Ăn trưa miễn phí/canteen"
                elif "văn hoá" in pattern or "clb" in item_lower:
                    return "Hoạt động CLB/văn hoá"
    
    # If no pattern matched, return cleaned version
    # Remove common prefixes
    item = re.sub(r"^(được|có|có thể|được hưởng)\s+", "", item, flags=re.IGNORECASE)
    
    # Capitalize first letter
    if item:
        item = item[0].upper() + item[1:]
    
    return item


def extract_benefits_from_section(block: str, extract_bullets_fn) -> List[str]:
    """
    Extract and normalize benefits from section.
    Target: 4-12 items, normalized to short bullets.
    """
    if not block:
        return []
    
    # Extract bullets
    bullets = extract_bullets_fn(block, limit=40)
    
    # Normalize each benefit
    normalized = []
    seen = set()
    
    for bullet in bullets:
        normalized_item = normalize_benefit_item(bullet)
        
        # Deduplicate
        key = normalized_item.lower()
        if key not in seen and len(normalized_item) > 3:
            normalized.append(normalized_item)
            seen.add(key)
    
    return normalized[:12]  # Limit to 12 items
