"""
Section splitting and heading detection
"""
import re
from typing import Dict, List, Tuple, Set


SECTION_NAMES = {
    "responsibilities": {
        "mô tả công việc", "mô tả", "job description", "description"
    },
    "requirements": {
        "yêu cầu công việc", "yêu cầu ứng viên", "requirements", "qualifications"
    },
    "benefits": {
        "quyền lợi", "các phúc lợi dành cho bạn", "phúc lợi", "benefits", "quyền lợi được hưởng"
    },
    "work_location": {
        "địa điểm làm việc", "work location", "location"
    },
    "working_time": {
        "thời gian làm việc", "working time"
    },
    "job_info": {
        "thông tin việc làm", "job information"
    }
}

# Stop keywords for requirements section
REQUIREMENTS_STOP_KEYWORDS = {
    "quyền lợi", "phúc lợi", "benefits", "thời gian làm việc", "working time"
}


def get_all_section_headings() -> Set[str]:
    """Get all section heading keywords"""
    return set().union(*SECTION_NAMES.values())


def split_sections(text: str) -> Dict[str, str]:
    """Split text into sections based on headings"""
    lines = text.split("\n")

    anchors: List[Tuple[int, str]] = []
    for i, ln in enumerate(lines):
        # Remove trailing colon and normalize
        key = ln.strip().lower().rstrip(":")
        
        for sec_key, names in SECTION_NAMES.items():
            if key in names:
                anchors.append((i, sec_key))
                break

    anchors.sort(key=lambda x: x[0])
    if not anchors:
        return {}

    sections: Dict[str, str] = {}
    for j, (start_idx, sec_key) in enumerate(anchors):
        end_idx = anchors[j + 1][0] if j + 1 < len(anchors) else len(lines)
        body = "\n".join(lines[start_idx + 1 : end_idx]).strip()
        sections[sec_key] = body

    return sections


def split_text_into_bullets(text: str, min_bullets: int = 3) -> List[str]:
    """
    Split text into bullet list with smart normalization.
    Target: 5-10 bullets, each 8-25 words.
    
    Rules:
    - Split by newline, period, semicolon, or imperative sentence patterns
    - Normalize: trim, remove empty lines, merge very short items (< 5 chars)
    - If >= 3 sentences or >= 3 lines → force split
    """
    if not text:
        return []
    
    items = []
    
    # Step 1: Try splitting by newlines first
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    
    # If we have multiple lines, use them
    if len(lines) >= min_bullets:
        for ln in lines:
            # Remove bullet markers if present
            cleaned = re.sub(r"^[-•*]\s+", "", ln)
            cleaned = re.sub(r"^\d+\.\s+", "", cleaned)
            if len(cleaned) > 5:
                items.append(cleaned)
    
    # Step 2: If not enough items, try splitting by semicolon
    if len(items) < min_bullets:
        items = []
        for part in re.split(r"[;]\s*", text):
            p = part.strip()
            if len(p) > 5:
                items.append(p)
    
    # Step 3: If still not enough, try splitting by period (sentence-based)
    if len(items) < min_bullets:
        items = []
        # Split by period but keep periods that are part of abbreviations
        sentences = re.split(r'\.(?=\s+[A-ZĐÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ])', text)
        for sent in sentences:
            s = sent.strip()
            if len(s) > 5:
                items.append(s)
    
    # Step 4: Normalize - merge very short items with previous
    normalized = []
    for i, item in enumerate(items):
        if len(item) < 5 and normalized:
            # Merge with previous
            normalized[-1] = normalized[-1] + " " + item
        else:
            normalized.append(item)
    
    # Step 5: Filter by word count (8-25 words ideal, but allow 3-40)
    final = []
    for item in normalized:
        word_count = len(item.split())
        if 3 <= word_count <= 40:
            final.append(item.strip())
    
    return final


def extract_bullets(block: str, limit: int = 50, stop_on_keywords: Set[str] = None) -> List[str]:
    """
    Extract bullet points from a text block.
    Supports:
    - Bullet markers (-, •, *)
    - Numbered lists (1., 2., etc.)
    - Smart text splitting for paragraphs
    """
    if not block:
        return []

    items = []
    all_headings = get_all_section_headings()

    # Step 1: Try extracting explicit bullets/numbers
    for ln in block.split("\n"):
        s = ln.strip()
        if not s:
            continue

        # Stop on section headings
        if s.lower() in all_headings:
            break

        # Stop on specific keywords
        if stop_on_keywords and any(kw in s.lower() for kw in stop_on_keywords):
            break

        # Bullet markers
        if re.match(r"^[-•*]\s+", s):
            cleaned = re.sub(r"^[-•*]\s+", "", s)
            if len(cleaned) > 3:
                items.append(cleaned)
        # Numbered lists
        elif re.match(r"^\d+\.\s+", s):
            cleaned = re.sub(r"^\d+\.\s+", "", s)
            if len(cleaned) > 3:
                items.append(cleaned)

    # Step 2: If no explicit bullets, use smart splitting
    if not items:
        items = split_text_into_bullets(block, min_bullets=3)

    return items[:limit]
