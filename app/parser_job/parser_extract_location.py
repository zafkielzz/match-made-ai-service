"""
Location extraction: cities, addresses, city-address pairs
"""
import re
from typing import Dict, List, Optional, Tuple

from .parser_utils_text import looks_like_street_address


CITY_MAP = {
    "hà nội": ("VN-HN", "Hà Nội"),
    "ha noi": ("VN-HN", "Hà Nội"),
    "hồ chí minh": ("VN-SG", "Hồ Chí Minh"),
    "ho chi minh": ("VN-SG", "Hồ Chí Minh"),
    "tp hcm": ("VN-SG", "Hồ Chí Minh"),
    "tp.hcm": ("VN-SG", "Hồ Chí Minh"),
    "hcm": ("VN-SG", "Hồ Chí Minh"),
    "đà nẵng": ("VN-DN", "Đà Nẵng"),
    "da nang": ("VN-DN", "Đà Nẵng"),
}


def extract_cities(cleaned_text: str) -> Tuple[List[Dict[str, str]], float]:
    """Extract city mentions from text"""
    lower = cleaned_text.lower()
    found = []
    seen = set()

    for k, (code, name) in CITY_MAP.items():
        if k in lower:
            if code not in seen:
                found.append({"code": code, "name": name})
                seen.add(code)

    if not found:
        return [], 0.0

    # Higher confidence for single city
    conf = 0.9 if len(found) == 1 else 0.75
    return found, conf


def extract_city_address_pairs_from_location_section(
    sections: Dict[str, str]
) -> List[Tuple[Dict[str, str], str]]:
    """
    Parse city-address pairs from location section.
    Format:
        Hà Nội: Số 182 Đường ...
        TP.HCM: 98C Bạch Đằng ...
        HCM : Address here
    """
    block = sections.get("work_location") or ""
    if not block:
        return []

    results: List[Tuple[Dict[str, str], str]] = []

    for ln in block.split("\n"):
        line = ln.strip()
        if not line or ":" not in line:
            continue

        parts = line.split(":", 1)
        if len(parts) != 2:
            continue

        city_part = parts[0].strip()
        addr_part = parts[1].strip()

        # Normalize city key
        city_key = city_part.lower().replace(".", "").strip()

        # Map to city object
        city_obj = None
        if "ha noi" in city_key or "hà nội" in city_key:
            city_obj = {"code": "VN-HN", "name": "Hà Nội"}
        elif "tp hcm" in city_key or "hcm" in city_key or "ho chi minh" in city_key or "hồ chí minh" in city_key:
            city_obj = {"code": "VN-SG", "name": "Hồ Chí Minh"}
        elif "da nang" in city_key or "đà nẵng" in city_key:
            city_obj = {"code": "VN-DN", "name": "Đà Nẵng"}

        # Only add if address looks valid
        if city_obj and addr_part and looks_like_street_address(addr_part):
            results.append((city_obj, addr_part))

    return results


def deduplicate_locations(locations: List[Dict]) -> List[Dict]:
    """
    Deduplicate locations with STABILITY rule.
    
    Rules:
    - If same city.code + country.code and one has address=null → ALWAYS keep the one with address
    - If both have different addresses → keep both
    - NEVER "downgrade" from address to null unless input lacks address
    """
    if not locations:
        return []
    
    # Group by (country.code, city.code)
    groups = {}
    for loc in locations:
        country_code = loc.get("country", {}).get("code", "")
        city_code = loc.get("city", {}).get("code", "")
        key = (country_code, city_code)
        
        if key not in groups:
            groups[key] = []
        groups[key].append(loc)
    
    # Process each group with STABILITY priority
    deduped = []
    for key, locs in groups.items():
        if len(locs) == 1:
            deduped.append(locs[0])
        else:
            # Multiple locations for same city
            # PRIORITY: Keep ones with address (NEVER downgrade to null)
            with_address = [loc for loc in locs if loc.get("address")]
            without_address = [loc for loc in locs if not loc.get("address")]
            
            if with_address:
                # Dedupe by normalized address
                seen_addresses = set()
                for loc in with_address:
                    addr_key = loc["address"].lower().strip()
                    if addr_key not in seen_addresses:
                        deduped.append(loc)
                        seen_addresses.add(addr_key)
                # NEVER add null address if we have address
            elif without_address:
                # All have no address, keep only one
                deduped.append(without_address[0])
    
    return deduped


def filter_benefits_that_are_addresses(benefits: List[str]) -> Tuple[List[str], List[Tuple[Dict, str]]]:
    """
    Filter out benefits that are actually addresses.
    Format: "Hà Nội: Ngõ 15..." or "HCM: 123 Street..."
    Returns: (filtered_benefits, extracted_city_address_pairs)
    """
    filtered = []
    extracted_pairs = []

    for item in benefits:
        # Check if starts with city name followed by colon
        match = re.match(r"^(Hà Nội|Ha Noi|TP\.HCM|HCM|Ho Chi Minh|Hồ Chí Minh)\s*:\s*(.+)$", item, re.IGNORECASE)
        
        if match:
            city_part = match.group(1).strip()
            addr_part = match.group(2).strip()
            
            # Map city
            city_key = city_part.lower().replace(".", "").strip()
            city_obj = None
            if "ha noi" in city_key or "hà nội" in city_key:
                city_obj = {"code": "VN-HN", "name": "Hà Nội"}
            elif "hcm" in city_key or "ho chi minh" in city_key or "hồ chí minh" in city_key:
                city_obj = {"code": "VN-SG", "name": "Hồ Chí Minh"}
            
            # If looks like address, extract it
            if city_obj and looks_like_street_address(addr_part):
                extracted_pairs.append((city_obj, addr_part))
            else:
                # Not a valid address, keep in benefits
                filtered.append(item)
        else:
            filtered.append(item)

    return filtered, extracted_pairs
