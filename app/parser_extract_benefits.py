"""Benefits extraction with robust section detection"""
import re
from typing import List, Tuple

# A. Section detection markers
BENEFITS_MARKERS = {
    "quyền lợi",
    "phúc lợi",
    "benefits",
    "chế độ",
    "what we offer"
}

STOP_MARKERS = {
    "địa điểm làm việc",
    "thời gian làm việc",
    "cách thức ứng tuyển",
    "thông tin chung",
    "yêu cầu",
    "mô tả công việc",
    "work location",
    "working time",
    "how to apply"
}

# C. Normalization - Keywords to preserve
PRESERVE_KEYWORDS = {
    "mentor", "đào tạo", "training",
    "dự án thực tế", "project",
    "thưởng", "bonus",
    "ăn trưa", "lunch",
    "clb", "club",
    "nhận chính thức", "lên chính thức", "chính thức"
}

# C. Prefixes to remove
REMOVE_PREFIXES = {
    "thực tập sinh",
    "các bạn",
    "ứng viên",
    "nhân viên"
}


def detect_benefits_section(text: str) -> Tuple[str, bool]:
    """
    A. Robust section detection.
    Returns: (benefits_text, found)
    """
    text_lower = text.lower()
    
    # Find benefits marker
    marker_pos = -1
    marker_found = None
    
    for marker in BENEFITS_MARKERS:
        pos = text_lower.find(marker)
        if pos != -1:
            if marker_pos == -1 or pos < marker_pos:
                marker_pos = pos
                marker_found = marker
    
    if marker_pos == -1:
        return "", False
    
    # Find stop marker (next section)
    stop_pos = len(text)
    for stop_marker in STOP_MARKERS:
        pos = text_lower.find(stop_marker, marker_pos + len(marker_found))
        if pos != -1 and pos < stop_pos:
            stop_pos = pos
    
    # Extract text between markers
    benefits_text = text[marker_pos + len(marker_found):stop_pos].strip()
    
    return benefits_text, True


def extract_bullets_from_benefits(text: str) -> List[str]:
    """
    B. Extraction rule - split by newline, semicolon, or sentences.
    """
    if not text:
        return []
    
    items = []
    
    # Step 1: Split by newlines first
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    
    for line in lines:
        # Remove bullet markers
        line = re.sub(r"^[-•*]\s+", "", line)
        line = re.sub(r"^\d+\.\s+", "", line)
        
        if not line or len(line) < 5:
            continue
        
        # If line has semicolons, split by them
        if ";" in line:
            parts = [p.strip() for p in line.split(";") if p.strip()]
            items.extend(parts)
        # If line is very long (> 200 chars), try splitting by periods
        elif len(line) > 200 and "." in line:
            sentences = [s.strip() for s in line.split(".") if s.strip()]
            items.extend(sentences)
        else:
            items.append(line)
    
    return items


def normalize_benefit_item(item: str) -> str:
    """
    C. Normalization rule:
    - Max 160 chars
    - Remove prefixes
    - Preserve keywords
    """
    item = item.strip()
    
    # Remove prefixes
    for prefix in REMOVE_PREFIXES:
        pattern = rf"^{prefix}\s+"
        item = re.sub(pattern, "", item, flags=re.IGNORECASE)
    
    # Remove generic prefixes
    item = re.sub(r"^(được|có|có thể|được hưởng)\s+", "", item, flags=re.IGNORECASE)
    
    # Limit to 160 chars
    if len(item) > 160:
        item = item[:160].rsplit(" ", 1)[0] + "..."
    
    # Capitalize first letter
    if item:
        item = item[0].upper() + item[1:]
    
    return item


def extract_benefits_from_section(block: str, extract_bullets_fn=None) -> List[str]:
    """
    Main benefits extraction with robust detection.
    D. Minimum pass threshold: >= 3 items
    """
    if not block:
        return []
    
    # B. Extract bullets
    bullets = extract_bullets_from_benefits(block)
    
    # C. Normalize each benefit
    normalized = []
    seen = set()
    
    for bullet in bullets:
        normalized_item = normalize_benefit_item(bullet)
        
        # Skip very short items
        if len(normalized_item) < 5:
            continue
        
        # Deduplicate
        key = normalized_item.lower()
        if key not in seen:
            normalized.append(normalized_item)
            seen.add(key)
    
    # D. Return up to 10 items
    return normalized[:10]


def extract_major_benefits(text: str) -> List[str]:
    """
    Fallback: Extract major benefits using keyword matching.
    """
    text_lower = text.lower()
    found_benefits = []
    seen = set()
    
    # Major benefit patterns
    patterns = {
        "mentor": ["mentor", "hướng dẫn 1:1", "mentor 1:1"],
        "training": ["đào tạo", "training", "phát triển kỹ năng"],
        "project": ["dự án thực tế", "tham gia dự án", "project"],
        "fulltime": ["cơ hội chính thức", "lên chính thức", "nhận chính thức"],
        "lunch": ["ăn trưa", "free lunch", "canteen"],
        "insurance": ["bảo hiểm", "insurance"],
        "bonus": ["thưởng", "bonus"],
        "club": ["clb", "club", "hoạt động ngoại khóa"]
    }
    
    for benefit_type, keywords in patterns.items():
        for keyword in keywords:
            if keyword in text_lower and benefit_type not in seen:
                if benefit_type == "mentor":
                    found_benefits.append("Mentor 1:1")
                elif benefit_type == "training":
                    found_benefits.append("Đào tạo và phát triển kỹ năng")
                elif benefit_type == "project":
                    found_benefits.append("Tham gia dự án thực tế")
                elif benefit_type == "fulltime":
                    found_benefits.append("Cơ hội lên chính thức")
                elif benefit_type == "lunch":
                    found_benefits.append("Ăn trưa miễn phí")
                elif benefit_type == "insurance":
                    found_benefits.append("Bảo hiểm đầy đủ")
                elif benefit_type == "bonus":
                    found_benefits.append("Thưởng hiệu suất")
                elif benefit_type == "club":
                    found_benefits.append("Hoạt động CLB")
                
                seen.add(benefit_type)
                break
    
    return found_benefits

