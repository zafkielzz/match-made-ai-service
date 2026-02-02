"""
Text normalization and noise removal utilities
"""
import re
from typing import List, Tuple


NOISE_EXACT = {
    "nộp đơn",
    "lưu công việc này",
    "ứng tuyển ngay",
    "lưu tin",
    "xem thêm",
    "mức lương phổ biến trên thị trường là bao nhiêu?",
    "mức độ phù hợp và xếp hạng của bạn so với ứng viên khác như thế nào?",
}


def normalize_text(raw: str) -> str:
    """Normalize line endings, remove zero-width chars, clean whitespace"""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = "\n".join(ln.strip() for ln in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_noise_line(line: str) -> bool:
    """Check if line is noise (ads, view counts, etc.)"""
    l = line.strip().lower()
    if not l:
        return False
    if l in NOISE_EXACT:
        return True
    if re.fullmatch(r"\d+\s*lượt xem", l):
        return True
    if "lượt xem" in l:
        return True
    if "hết hạn trong" in l:
        return True
    return False


def remove_noise_lines(text: str) -> Tuple[str, List[str]]:
    """Remove noise lines from text, return cleaned text and removed lines"""
    kept, removed = [], []
    for ln in text.split("\n"):
        if is_noise_line(ln):
            removed.append(ln)
        else:
            kept.append(ln)
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, removed


def detect_source(text: str) -> str:
    """Detect job board source from text patterns"""
    lower = text.lower()
    if "thông tin việc làm" in lower and "ngày đăng" in lower:
        return "VIETNAMWORKS"
    if "chi tiết tin tuyển dụng" in lower or "gửi tôi việc làm tương tự" in lower:
        return "TOPCV"
    return "OTHER"


def looks_like_street_address(s: str) -> bool:
    """Check if string looks like a street address (has number + street keywords)"""
    s = s.strip()
    has_number = bool(re.search(r"\b\d{1,4}\b", s))
    has_street_keyword = any(
        x in s.lower() 
        for x in ["đường", "street", "st.", "ave", "quận", "ward", "phường", "ngõ", "số"]
    )
    return has_number and has_street_keyword
