"""
Main job parser orchestration
"""
from typing import Any, Dict

from app.parser_job.parser_extract_common import (
    extract_deadline,
    extract_education,
    extract_experience_years,
    extract_job_level,
    extract_title,
    extract_vietnamworks_benefits,
    extract_working_time,
)
from app.parser_job.parser_extract_location import (
    deduplicate_locations,
    extract_cities,
    extract_city_address_pairs_from_location_section,
    filter_benefits_that_are_addresses,
)
from app.parser_job.parser_extract_salary import extract_salary
from app.parser_job.parser_extract_benefits import extract_benefits_from_section
from app.parser_job.parser_extract_requirements import extract_requirements_with_constraints
from app.parser_job.parser_metadata import extract_key_value_metadata
from app.parser_job.parser_sections import extract_bullets, split_sections, REQUIREMENTS_STOP_KEYWORDS
from app.parser_job.parser_normalize import normalize_working_time
from app.parser_job.parser_overview import synthesize_overview
from app.parser_job.parser_validation import validate_parsed_result, adjust_confidence_based_on_quality
from app.parser_job.parser_types import ParseDefaults, ParseResult
from app.parser_job.parser_utils_text import detect_source, normalize_text, remove_noise_lines


def parse_job_text(raw_text: str, defaults: ParseDefaults) -> ParseResult:
    """
    Main parser function.
    Returns ParseResult with detected_source, suggested fields, confidence scores, warnings.
    """
    # Step 1: Normalize and clean
    normalized = normalize_text(raw_text)
    cleaned, removed_noise = remove_noise_lines(normalized)
    detected = detect_source(cleaned)

    # Step 2: Initialize output structure
    suggested: Dict[str, Any] = {
        "title": None,
        "companyName": None,
        "locations": [],  # ✅ Only use "locations" (plural)
        "taxonomy": None,
        "jobLevel": None,
        "employmentType": None,
        "workMode": None,
        "experienceYears": None,
        "education": None,
        "languageRequirements": [],
        "salary": None,
        "overview": None,
        "responsibilities": [],
        "requirements": {"required": [], "preferred": []},
        "benefits": {"predefined": [], "custom": []},
        "workingTime": None,
        "applicationDeadline": None,
        "hireNumber": None,
        "status": "DRAFT",
    }

    confidence: Dict[str, float] = {}
    warnings: list[str] = []

    # Step 3: Extract metadata (key-value pairs)
    meta_kv = extract_key_value_metadata(cleaned)

    # hireNumber from metadata
    if "số lượng tuyển" in meta_kv:
        import re
        raw = meta_kv["số lượng tuyển"]
        m = re.search(r"\d+", raw)
        if m:
            suggested["hireNumber"] = int(m.group())
            confidence["hireNumber"] = 0.85
        else:
            warnings.append("Could not parse hireNumber")

    # jobLevel from metadata
    if meta_kv.get("chức vụ") == "Nhân viên":
        suggested["jobLevel"] = "STAFF"
        confidence["jobLevel"] = 0.9

    # employmentType from metadata
    if "toàn thời gian" in meta_kv.get("hình thức làm việc", "").lower():
        suggested["employmentType"] = "FULL_TIME"
        confidence["employmentType"] = 0.9

    # experienceYears from metadata
    if "không yêu cầu" in meta_kv.get("kinh nghiệm", "").lower():
        suggested["experienceYears"] = {"min": 0, "max": 0}
        confidence["experienceYears"] = 0.95

    # education from metadata
    if "cao đẳng" in meta_kv.get("bằng cấp", "").lower():
        suggested["education"] = {"minLevel": "ASSOCIATE"}
        confidence["education.minLevel"] = 0.9

    # Step 4: Split sections
    sections = split_sections(cleaned)

    # Step 5: Extract title
    title, c = extract_title(cleaned)
    if title:
        suggested["title"] = title
        confidence["title"] = c
    else:
        warnings.append("Could not detect title")

    # Step 6: Extract locations (city-address pairs preferred)
    city_address_pairs = extract_city_address_pairs_from_location_section(sections)

    if city_address_pairs:
        # High confidence: city-specific addresses found
        for city_obj, addr in city_address_pairs:
            suggested["locations"].append({
                "country": {
                    "code": defaults.countryCode,
                    "name": "Vietnam" if defaults.countryCode == "VN" else ""
                },
                "city": city_obj,
                "address": addr,
                "workModeHint": None
            })
        confidence["locations"] = 0.95
    else:
        # Fallback: city-only detection
        cities, c_conf = extract_cities(cleaned)
        for city in cities:
            suggested["locations"].append({
                "country": {
                    "code": defaults.countryCode,
                    "name": "Vietnam" if defaults.countryCode == "VN" else ""
                },
                "city": city,
                "address": None,
                "workModeHint": None
            })
        if cities:
            confidence["locations"] = c_conf

    # Deduplicate locations (fix HN, HN duplicate bug)
    suggested["locations"] = deduplicate_locations(suggested["locations"])

    if not suggested["locations"]:
        warnings.append("Could not detect any location")

    # Step 7: Extract salary
    salary, c = extract_salary(cleaned, defaults)
    suggested["salary"] = salary
    confidence["salary"] = c
    if salary.get("negotiable") and salary.get("min", 0) == 0 and salary.get("max", 0) == 0:
        warnings.append("Salary is negotiable; min/max unknown")

    # Step 8: Extract experience
    exp, c = extract_experience_years(cleaned)
    if exp:
        suggested["experienceYears"] = exp
        confidence["experienceYears"] = c

    # Step 9: Extract job level
    jl, c = extract_job_level(cleaned)
    if jl:
        suggested["jobLevel"] = jl
        confidence["jobLevel"] = c

    # Step 10: Extract education
    edu, c = extract_education(cleaned)
    if edu:
        suggested["education"] = edu
        confidence["education.minLevel"] = c

    # Step 11: Extract deadline
    ddl, c = extract_deadline(cleaned)
    if ddl:
        suggested["applicationDeadline"] = ddl
        confidence["applicationDeadline"] = c

    # Step 12: Extract working time (with normalization)
    wt, c = extract_working_time(sections, cleaned)
    if wt:
        # Normalize working time format
        normalized_wt = normalize_working_time(wt)
        suggested["workingTime"] = normalized_wt["text"]
        confidence["workingTime"] = c
        
        # Store structured data in meta if available
        if "days" in normalized_wt or "startTime" in normalized_wt:
            confidence["workingTime.structured"] = {
                k: v for k, v in normalized_wt.items() if k != "text"
            }

    # Step 13: Extract responsibilities (with smart splitting)
    resp_items = extract_bullets(sections.get("responsibilities", ""), limit=50)
    if resp_items:
        suggested["responsibilities"] = resp_items[:10]  # Limit to 10 bullets
        confidence["responsibilities"] = 0.75

    # Step 14: Extract requirements (with constraints extraction)
    req_block = sections.get("requirements", "")
    if req_block:
        required, preferred, constraints = extract_requirements_with_constraints(
            req_block,
            extract_bullets
        )
        suggested["requirements"] = {"required": required[:12], "preferred": preferred[:12]}
        confidence["requirements.required"] = 0.75 if required else 0.0
        confidence["requirements.preferred"] = 0.6 if preferred else 0.0
        
        # Store constraints in meta if any found
        if constraints:
            confidence["requirements.constraints"] = constraints

    # Step 15: Extract benefits (ROBUST section detection + >= 3 items minimum)
    from app.parser_extract_benefits import detect_benefits_section, extract_major_benefits
    
    # A. Robust section detection
    benefits_text, section_found = detect_benefits_section(cleaned)
    
    if section_found and benefits_text:
        # Extract from detected section
        ben_items = extract_benefits_from_section(benefits_text, extract_bullets)
        
        # D. Minimum pass threshold: >= 3 items
        if ben_items and len(ben_items) >= 3:
            suggested["benefits"]["custom"] = ben_items[:10]
            confidence["benefits.custom"] = 0.7
        else:
            # Try fallback: major keywords
            major_benefits = extract_major_benefits(benefits_text)
            if len(major_benefits) >= 3:
                suggested["benefits"]["custom"] = major_benefits
                confidence["benefits.custom"] = 0.6
            else:
                # Section detected but < 3 items
                warnings.append("Benefits section detected but extraction failed (< 3 items)")
                confidence["benefits.custom"] = 0.2
    else:
        # Fallback: Try VietnamWorks format or section-based
        vw_benefits = extract_vietnamworks_benefits(cleaned)
        if vw_benefits and len(vw_benefits) >= 3:
            suggested["benefits"]["custom"] = vw_benefits[:10]
            confidence["benefits.custom"] = 0.8
        else:
            ben_block = sections.get("benefits", "")
            if ben_block:
                ben_items = extract_benefits_from_section(ben_block, extract_bullets)
                if ben_items and len(ben_items) >= 3:
                    suggested["benefits"]["custom"] = ben_items[:10]
                    confidence["benefits.custom"] = 0.7
                else:
                    warnings.append("Benefits section detected but extraction failed (< 3 items)")
                    confidence["benefits.custom"] = 0.2

    # Step 16: Filter benefits that are actually addresses
    filtered_benefits, addr_pairs = filter_benefits_that_are_addresses(
        suggested["benefits"]["custom"]
    )
    suggested["benefits"]["custom"] = filtered_benefits

    # Add extracted addresses to locations
    for city_obj, addr in addr_pairs:
        suggested["locations"].append({
            "country": {
                "code": defaults.countryCode,
                "name": "Vietnam" if defaults.countryCode == "VN" else ""
            },
            "city": city_obj,
            "address": addr,
            "workModeHint": None
        })

    # Deduplicate again after adding from benefits
    suggested["locations"] = deduplicate_locations(suggested["locations"])

    # Step 17: Extract workMode (strict rules - only set ONSITE if clear evidence)
    lower = cleaned.lower()
    
    # Check for explicit remote/hybrid keywords
    if "remote" in lower or "từ xa" in lower or "làm việc từ xa" in lower:
        suggested["workMode"] = "REMOTE"
        confidence["workMode"] = 0.75
    elif "hybrid" in lower or "kết hợp" in lower or "linh hoạt" in lower:
        suggested["workMode"] = "HYBRID"
        confidence["workMode"] = 0.65
    else:
        # Only set ONSITE if ALL conditions met:
        # 1. Has specific address
        # 2. Has fixed working hours
        # 3. No remote/hybrid keywords
        has_address = any(loc.get("address") for loc in suggested["locations"])
        has_working_hours = bool(suggested.get("workingTime"))
        no_remote_keywords = "remote" not in lower and "từ xa" not in lower
        
        if has_address and has_working_hours and no_remote_keywords:
            suggested["workMode"] = "ONSITE"
            confidence["workMode"] = 0.7
        else:
            # Not enough evidence - leave as null
            suggested["workMode"] = None
            warnings.append("Could not confidently detect workMode (insufficient evidence)")

    # Step 18: Extract employmentType (with INTERNSHIP support)
    # Patch C: Support INTERNSHIP enum if available
    if "part time" in lower or "part-time" in lower or "bán thời gian" in lower:
        suggested["employmentType"] = "PART_TIME"
        confidence["employmentType"] = 0.6
    elif "full time" in lower or "full-time" in lower or "toàn thời gian" in lower:
        suggested["employmentType"] = "FULL_TIME"
        confidence["employmentType"] = 0.6
    elif suggested["jobLevel"] == "INTERN":
        # Try to set INTERNSHIP if schema supports it
        # Otherwise keep null + hint
        suggested["employmentType"] = "INTERNSHIP"  # Schema should support this
        confidence["employmentType"] = 0.8
        confidence["employmentTypeHint"] = "INTERNSHIP"
        warnings.append("INTERN role detected, employmentType set to INTERNSHIP")
    elif not suggested["employmentType"]:
        warnings.append("Could not detect employmentType")

    # Step 19: Generate overview (HARD RULES - NO abbreviated verb lists)
    if not suggested["overview"]:
        overview = synthesize_overview(
            title=suggested["title"],
            responsibilities=suggested["responsibilities"],
            requirements=suggested["requirements"]["required"],
            job_level=suggested["jobLevel"],
            max_length=450
        )
        if overview:
            # Final check: NO abbreviated verb lists (HARD RULE)
            from app.parser_overview import has_abbreviated_verb_list
            if has_abbreviated_verb_list(overview):
                suggested["overview"] = None
                warnings.append("Overview synthesis failed due to invalid abbreviation")
            else:
                suggested["overview"] = overview
                confidence["overview"] = 0.7
        else:
            # Cannot synthesize quality overview
            suggested["overview"] = None
            warnings.append("Overview synthesis skipped due to low confidence")

    # Step 20: Validate and adjust confidence
    adjusted_confidence = adjust_confidence_based_on_quality(suggested, confidence)
    
    # Step 21: Run validation checks
    validation_errors = validate_parsed_result(suggested, adjusted_confidence)
    if validation_errors:
        for error in validation_errors:
            warnings.append(f"Validation: {error}")

    return ParseResult(
        detected_source=detected,
        suggested=suggested,
        confidence=adjusted_confidence,
        warnings=warnings
    )
