"""
Common field extractors: title, experience, education, job level, deadline, working time
"""
import re
from typing import Dict, Optional, Tuple

from .parser_metadata import first_meaningful_line


def extract_title(cleaned_text: str) -> Tuple[Optional[str], float]:
    """Extract job title (first meaningful line)"""
    title = first_meaningful_line(cleaned_text)
    if title and 3 <= len(title) <= 90:
        return title, 0.95
    return None, 0.0


def extract_experience_years(cleaned_text: str) -> Tuple[Optional[Dict[str, int]], float]:
    """Extract experience requirements"""
    lower = cleaned_text.lower()

    # No experience required
    if "không yêu cầu kinh nghiệm" in lower or "no experience" in lower:
        return {"min": 0, "max": 0}, 0.9

    # Range: "có từ 3 - 5 năm"
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*(năm|years?)", lower)
    if m:
        return {"min": int(m.group(1)), "max": int(m.group(2))}, 0.85

    # VietnamWorks format: "SỐ NĂM KINH NGHIỆM TỐI THIỂU\n3"
    m = re.search(r"số năm kinh nghiệm tối thiểu\s*\n\s*(\d+)", lower)
    if m:
        n = int(m.group(1))
        return {"min": n, "max": n}, 0.8

    # "ít nhất 2 năm", "từ 2 năm"
    m = re.search(r"(từ|ít nhất|at least)\s*(\d+)\s*(năm|years?)", lower)
    if m:
        n = int(m.group(2))
        return {"min": n, "max": n}, 0.6

    return None, 0.0


def extract_job_level(cleaned_text: str) -> Tuple[Optional[str], float]:
    """Extract job level (INTERN, STAFF, MANAGER, etc.)"""
    lower = cleaned_text.lower()

    # Intern keywords
    if re.search(r"\bintern\b", lower) or "thực tập" in lower:
        return "INTERN", 0.85

    # VietnamWorks: CẤP BẬC -> Nhân viên
    m = re.search(r"cấp bậc\s*\n\s*([^\n]+)", lower)
    if m:
        val = m.group(1).strip()
        if "nhân viên" in val:
            return "STAFF", 0.8
        if "trưởng" in val or "quản lý" in val:
            return "MANAGER", 0.7

    return None, 0.0


def extract_education(cleaned_text: str) -> Tuple[Optional[Dict[str, str]], float]:
    """Extract education requirements"""
    lower = cleaned_text.lower()
    
    if "đại học" in lower or "bachelor" in lower or "university" in lower:
        return {"minLevel": "BACHELOR"}, 0.6
    if "cao đẳng" in lower or "associate" in lower:
        return {"minLevel": "ASSOCIATE"}, 0.6
    
    return None, 0.0


def extract_deadline(cleaned_text: str) -> Tuple[Optional[str], float]:
    """
    Extract application deadline.
    Format: "Hạn nộp hồ sơ: 01/03/2026"
    Note: "Ngày đăng" (posted date) should NOT be treated as deadline.
    """
    # Only match "hạn nộp" patterns, not "ngày đăng"
    m = re.search(r"hạn nộp.*?:\s*(\d{2})/(\d{2})/(\d{4})", cleaned_text.lower())
    if m:
        dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
        return f"{yyyy}-{mm}-{dd}T00:00:00Z", 0.9
    
    return None, 0.0


def extract_working_time(sections: Dict[str, str], cleaned_text: str) -> Tuple[Optional[str], float]:
    """
    Extract working time/hours.
    Format: "Thứ 2 - Thứ 6 (từ 08:00 đến 17:00)"
    """
    # Prefer working_time section
    block = sections.get("working_time") or ""
    combined = (block + "\n" + cleaned_text).lower()

    # Pattern: "Thứ 2 - Thứ 6 (từ 08:00 đến 17:00)"
    m = re.search(
        r"(thứ\s*\d\s*[-–]\s*thứ\s*\d).*?(\d{1,2}:\d{2}).*?(\d{1,2}:\d{2})",
        combined
    )
    if m:
        return f"{m.group(1)} {m.group(2)}-{m.group(3)}", 0.75

    return None, 0.0


def extract_vietnamworks_benefits(cleaned_text: str) -> list[str]:
    """
    Extract benefits from VietnamWorks-specific format.
    Block: "Các phúc lợi dành cho bạn\nĐào tạo\n...\nMáy tính xách tay\n..."
    """
    lower = cleaned_text.lower()
    if "các phúc lợi dành cho bạn" not in lower:
        return []

    # Extract block between "Các phúc lợi..." and "Thông tin việc làm"
    m = re.search(
        r"các phúc lợi dành cho bạn\s*(.*?)\s*thông tin việc làm",
        lower,
        flags=re.S
    )
    block = m.group(1) if m else ""

    lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
    if not lines:
        return []

    benefits: list[str] = []
    i = 0
    while i < len(lines):
        title = lines[i]
        # Short title (likely benefit name)
        if len(title) <= 40:
            desc = ""
            if i + 1 < len(lines) and len(lines[i + 1]) > 40:
                desc = lines[i + 1]
                i += 1
            benefits.append(f"{title}: {desc}".strip(": ").strip())
        i += 1

    # Deduplicate
    out, seen = [], set()
    for b in benefits:
        k = b.lower()
        if k not in seen:
            out.append(b)
            seen.add(k)
    
    return out[:20]
