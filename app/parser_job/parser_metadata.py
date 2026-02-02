"""
Key-value metadata extraction and normalization
"""
import re
from typing import Dict


METADATA_LABELS = {
    "ngày đăng",
    "số lượng tuyển",
    "số lượng tuyển dụng",
    "chức vụ",
    "hình thức làm việc",
    "yêu cầu giới tính",
    "kinh nghiệm",
    "bằng cấp",
    "ngôn ngữ",
    "ngành nghề",
    "địa điểm làm việc",
}

DATE_VALUE_REGEX = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
NUMBER_ONLY_REGEX = re.compile(r"^\d+$")


def extract_key_value_metadata(text: str) -> Dict[str, str]:
    """
    Extract key-value pairs from metadata sections.
    Format: Label on one line, value on next line.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    meta = {}

    i = 0
    while i < len(lines) - 1:
        key_raw = lines[i].lower()
        val = lines[i + 1]

        key = None
        # Normalize "số lượng tuyển dụng" -> "số lượng tuyển"
        if key_raw.startswith("số lượng tuyển"):
            key = "số lượng tuyển"
        elif key_raw.startswith("ngày đăng"):
            key = "ngày đăng"
        elif key_raw.startswith("chức vụ"):
            key = "chức vụ"
        elif key_raw.startswith("hình thức làm việc"):
            key = "hình thức làm việc"
        elif key_raw.startswith("kinh nghiệm"):
            key = "kinh nghiệm"
        elif key_raw.startswith("bằng cấp"):
            key = "bằng cấp"

        if key:
            meta[key] = val
            i += 2
        else:
            i += 1

    return meta


def first_meaningful_line(text: str) -> str | None:
    """
    Extract first meaningful line (likely job title).
    Skip metadata labels, noise, dates, numbers, section headings.
    """
    from app.parser_job.parser_sections import get_all_section_headings
    from app.parser_job.parser_utils_text import is_noise_line
    
    prev_was_metadata = False
    all_headings = get_all_section_headings()

    for ln in [l.strip() for l in text.split("\n")]:
        if not ln:
            continue

        lnl = ln.lower()

        # Skip value after metadata label
        if prev_was_metadata:
            prev_was_metadata = False
            continue

        if is_noise_line(ln):
            continue

        if lnl in METADATA_LABELS:
            prev_was_metadata = True
            continue

        if lnl in all_headings:
            continue

        # Skip date values
        if DATE_VALUE_REGEX.match(ln):
            continue
        # Skip pure numbers
        if NUMBER_ONLY_REGEX.match(ln):
            continue

        # Skip short uppercase labels
        if len(ln) <= 25 and ln.isupper():
            continue

        return ln

    return None
