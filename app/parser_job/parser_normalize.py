"""
Normalization utilities for various fields
"""
import re
from typing import Dict, Optional


def normalize_working_time(raw_time: str) -> Dict[str, any]:
    """
    Normalize working time to consistent format.
    Input: "thứ 2 - thứ 6 08:00-17:00"
    Output: {
        "text": "Thứ 2–Thứ 6, 08:00–17:00",
        "days": ["MON", "TUE", "WED", "THU", "FRI"],  # optional
        "startTime": "08:00",  # optional
        "endTime": "17:00"  # optional
    }
    """
    if not raw_time:
        return {"text": None}
    
    result = {"text": raw_time}
    
    # Normalize text format
    text = raw_time.strip()
    
    # Capitalize "Thứ"
    text = re.sub(r"\bthứ\b", "Thứ", text, flags=re.IGNORECASE)
    
    # Normalize dash to en-dash
    text = re.sub(r"\s*-\s*", "–", text)
    
    # Add comma after days if not present
    text = re.sub(r"(\d)\s+(\d{1,2}:\d{2})", r"\1, \2", text)
    
    result["text"] = text
    
    # Extract days (optional)
    days_match = re.search(r"thứ\s*(\d)\s*[-–]\s*thứ\s*(\d)", text, re.IGNORECASE)
    if days_match:
        start_day = int(days_match.group(1))
        end_day = int(days_match.group(2))
        
        day_map = {
            2: "MON", 3: "TUE", 4: "WED", 5: "THU", 6: "FRI", 7: "SAT", 8: "SUN"
        }
        
        days = []
        for d in range(start_day, end_day + 1):
            if d in day_map:
                days.append(day_map[d])
        
        if days:
            result["days"] = days
    
    # Extract time range (optional)
    time_match = re.search(r"(\d{1,2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2})", text)
    if time_match:
        result["startTime"] = time_match.group(1)
        result["endTime"] = time_match.group(2)
    
    return result
