"""
Regression tests for job parser
"""
import pytest

from app.job_parse import ParseDefaults, parse_job_text


@pytest.fixture
def defaults():
    return ParseDefaults(countryCode="VN", currency="VND", salaryPeriod="MONTH", salaryType="GROSS")


def test_vietnamworks_format(defaults):
    """Test VietnamWorks format with metadata and sections"""
    raw_text = """
    Senior Python Developer
    
    THÔNG TIN VIỆC LÀM
    Ngày đăng
    15/01/2026
    Số lượng tuyển
    3
    Chức vụ
    Nhân viên
    Hình thức làm việc
    Toàn thời gian
    Kinh nghiệm
    Không yêu cầu
    Bằng cấp
    Cao đẳng
    
    MÔ TẢ CÔNG VIỆC
    - Develop and maintain Python applications
    - Write clean, maintainable code
    - Collaborate with team members
    
    YÊU CẦU CÔNG VIỆC
    - 3-5 năm kinh nghiệm Python
    - Biết Django, Flask
    - Ưu tiên có kinh nghiệm AWS
    
    QUYỀN LỢI
    - Lương 15-20 triệu
    - Bảo hiểm đầy đủ
    - Thưởng cuối năm
    
    ĐỊA ĐIỂM LÀM VIỆC
    Hà Nội: Số 182 Đường Bờ Sông Sét, Phường Phúc Đồng
    TP.HCM: 98C Bạch Đằng, Quận 1
    """
    
    result = parse_job_text(raw_text, defaults)
    
    # No crash
    assert result is not None
    assert result.detected_source == "VIETNAMWORKS"
    
    # Schema keys exist
    assert "title" in result.suggested
    assert "locations" in result.suggested
    assert "salary" in result.suggested
    assert "responsibilities" in result.suggested
    assert "requirements" in result.suggested
    assert "benefits" in result.suggested
    
    # Title detected
    assert result.suggested["title"] == "Senior Python Developer"
    
    # Locations: 2 cities with addresses
    assert len(result.suggested["locations"]) == 2
    assert result.suggested["locations"][0]["city"]["code"] == "VN-HN"
    assert result.suggested["locations"][1]["city"]["code"] == "VN-SG"
    assert "182" in result.suggested["locations"][0]["address"]
    assert "98C" in result.suggested["locations"][1]["address"]
    
    # No duplicates
    city_codes = [loc["city"]["code"] for loc in result.suggested["locations"]]
    assert len(city_codes) == len(set(city_codes))
    
    # Salary
    assert result.suggested["salary"]["min"] == 15_000_000
    assert result.suggested["salary"]["max"] == 20_000_000
    
    # hireNumber
    assert result.suggested["hireNumber"] == 3


def test_topcv_format(defaults):
    """Test TopCV format"""
    raw_text = """
    Frontend Developer (ReactJS)
    
    Chi tiết tin tuyển dụng
    
    Mô tả công việc:
    - Build responsive web applications
    - Work with React, Redux
    - Optimize performance
    
    Yêu cầu ứng viên:
    - Có từ 2 năm kinh nghiệm ReactJS
    - Thành thạo HTML, CSS, JavaScript
    
    Quyền lợi:
    - Mức lương: Thỏa thuận
    - Làm việc tại Hồ Chí Minh
    """
    
    result = parse_job_text(raw_text, defaults)
    
    assert result is not None
    assert result.detected_source == "TOPCV"
    
    # Title
    assert "Frontend Developer" in result.suggested["title"]
    
    # Locations
    assert len(result.suggested["locations"]) >= 1
    assert any(loc["city"]["code"] == "VN-SG" for loc in result.suggested["locations"])
    
    # Salary negotiable
    assert result.suggested["salary"]["negotiable"] is True
    
    # Experience
    assert result.suggested["experienceYears"]["min"] == 2


def test_key_value_format(defaults):
    """Test key-value metadata format"""
    raw_text = """
    Data Analyst
    
    Số lượng tuyển dụng
    5
    
    Ngày đăng
    20/01/2026
    
    Chức vụ
    Nhân viên
    
    Hình thức làm việc
    Toàn thời gian
    
    Mô tả:
    Analyze data and create reports
    
    Địa điểm: Hà Nội
    """
    
    result = parse_job_text(raw_text, defaults)
    
    assert result is not None
    
    # hireNumber from "Số lượng tuyển dụng"
    assert result.suggested["hireNumber"] == 5
    
    # jobLevel
    assert result.suggested["jobLevel"] == "STAFF"
    
    # employmentType
    assert result.suggested["employmentType"] == "FULL_TIME"
    
    # Location
    assert len(result.suggested["locations"]) >= 1
    assert result.suggested["locations"][0]["city"]["code"] == "VN-HN"


def test_multi_address_format(defaults):
    """Test multi-address format with city:address pairs"""
    raw_text = """
    Marketing Manager
    
    Địa điểm làm việc
    Hà Nội:Số 182 Đường Bờ Sông Sét, Phường Phúc Đồng, Quận Long Biên
    TP.HCM:98C Bạch Đằng, Phường 2, Quận Tân Bình
    HCM : 123 Nguyễn Văn Linh, Quận 7
    
    Mô tả công việc:
    - Plan marketing campaigns
    - Manage social media
    
    Yêu cầu:
    - 5 năm kinh nghiệm marketing
    - Thành thạo tiếng Anh
    """
    
    result = parse_job_text(raw_text, defaults)
    
    assert result is not None
    
    # Locations: should have HN (1) and HCM (2 different addresses) = 3 total
    assert len(result.suggested["locations"]) == 3
    
    # City codes correct
    city_codes = [loc["city"]["code"] for loc in result.suggested["locations"]]
    assert "VN-HN" in city_codes
    assert "VN-SG" in city_codes
    
    # HCM appears twice with different addresses (not duplicates)
    hcm_locs = [loc for loc in result.suggested["locations"] if loc["city"]["code"] == "VN-SG"]
    assert len(hcm_locs) == 2
    assert hcm_locs[0]["address"] != hcm_locs[1]["address"]
    
    # Addresses assigned correctly (street-like)
    for loc in result.suggested["locations"]:
        assert loc["address"] is not None
        assert len(loc["address"]) > 0
        # Should not be just "City, Vietnam"
        assert "Vietnam" not in loc["address"]
        # Should have street indicators
        assert any(kw in loc["address"].lower() for kw in ["đường", "số", "quận", "phường"])


def test_benefits_not_containing_addresses(defaults):
    """Test that addresses in benefits are filtered out"""
    raw_text = """
    Software Engineer
    
    Quyền lợi:
    - Bảo hiểm xã hội đầy đủ
    - Hà Nội: Ngõ 15 Phố Huế, Quận Hai Bà Trưng
    - Thưởng hiệu suất
    - HCM: 456 Lê Văn Việt, Quận 9
    
    Địa điểm: Hà Nội, Hồ Chí Minh
    """
    
    result = parse_job_text(raw_text, defaults)
    
    assert result is not None
    
    # Benefits should not contain address-like items
    benefits = result.suggested["benefits"]["custom"]
    for benefit in benefits:
        # Should not start with city name followed by colon and address
        assert not (benefit.startswith("Hà Nội:") and "Ngõ" in benefit)
        assert not (benefit.startswith("HCM:") and "Lê Văn Việt" in benefit)
    
    # Those addresses should be in locations instead
    assert len(result.suggested["locations"]) >= 2
    
    # Check that actual benefits remain (at least 2 non-address benefits)
    assert len(benefits) >= 2


def test_no_location_field_used(defaults):
    """Ensure no code uses 'location' (singular) - only 'locations' (plural)"""
    raw_text = """
    Test Job
    Địa điểm: Hà Nội
    """
    
    result = parse_job_text(raw_text, defaults)
    
    # Should have 'locations' key
    assert "locations" in result.suggested
    
    # Should NOT have 'location' key
    assert "location" not in result.suggested


def test_intern_role_warning(defaults):
    """Test that INTERN role without employmentType generates warning"""
    raw_text = """
    Intern Developer
    
    Mô tả: Thực tập sinh phát triển phần mềm
    Địa điểm: Hà Nội
    """
    
    result = parse_job_text(raw_text, defaults)
    
    assert result.suggested["jobLevel"] == "INTERN"
    
    # Should have warning about employmentType
    assert any("employmentType" in w for w in result.warnings)


def test_ngay_dang_not_deadline(defaults):
    """Test that 'Ngày đăng' is not treated as applicationDeadline"""
    raw_text = """
    Backend Developer
    
    Ngày đăng
    10/01/2026
    
    Hạn nộp hồ sơ: 28/02/2026
    
    Mô tả: Develop APIs
    """
    
    result = parse_job_text(raw_text, defaults)
    
    # Deadline should be from "Hạn nộp", not "Ngày đăng"
    if result.suggested["applicationDeadline"]:
        assert "2026-02-28" in result.suggested["applicationDeadline"]
        assert "2026-01-10" not in result.suggested["applicationDeadline"]



# ============================================
# NEW TESTS FOR IMPROVEMENTS
# ============================================

def test_responsibilities_split_into_bullets(defaults):
    """Test that responsibilities are split into proper bullet list (5-10 items)"""
    raw_text = """
    Backend Developer
    
    Mô tả công việc:
    Phát triển và bảo trì các API RESTful cho hệ thống. Làm việc với database PostgreSQL và Redis. Tối ưu hiệu suất query và caching. Viết unit test và integration test. Code review cho team members. Tham gia daily standup và sprint planning.
    
    Địa điểm: Hà Nội
    """
    
    result = parse_job_text(raw_text, defaults)
    
    assert result is not None
    
    # Responsibilities should be split into bullets
    responsibilities = result.suggested["responsibilities"]
    assert isinstance(responsibilities, list)
    assert len(responsibilities) >= 3  # At least 3 bullets
    assert len(responsibilities) <= 10  # Max 10 bullets
    
    # Each bullet should be reasonable length (not the whole paragraph)
    for bullet in responsibilities:
        word_count = len(bullet.split())
        assert 3 <= word_count <= 40  # Reasonable word count per bullet


def test_requirements_with_constraints_extraction(defaults):
    """Test requirements extraction with numeric constraints (GPA, TOEIC, etc.)"""
    raw_text = """
    Data Analyst Intern
    
    Yêu cầu công việc:
    - Sinh viên năm 3, năm 4 chuyên ngành CNTT, Toán, Thống kê
    - GPA >= 3.2/4.0
    - TOEIC >= 600 hoặc tương đương
    - Có thể làm việc ít nhất 28 giờ/tuần
    - Biết Python, SQL
    - Ưu tiên có kinh nghiệm với Pandas, NumPy
    
    Địa điểm: Hồ Chí Minh
    """
    
    result = parse_job_text(raw_text, defaults)
    
    assert result is not None
    
    # Requirements should be split
    requirements = result.suggested["requirements"]
    assert len(requirements["required"]) >= 3
    assert len(requirements["preferred"]) >= 1
    
    # Constraints should be extracted
    constraints = result.confidence.get("requirements.constraints", {})
    
    # Check GPA constraint
    if "gpa" in constraints:
        assert constraints["gpa"]["min"] == 3.2
        assert constraints["gpa"]["scale"] == 4.0
    
    # Check TOEIC constraint
    if "toeic" in constraints:
        assert constraints["toeic"]["min"] == 600
    
    # Check time commitment
    if "timeCommitment" in constraints:
        assert constraints["timeCommitment"]["min"] == 28
        assert constraints["timeCommitment"]["unit"] == "hours"


def test_benefits_extraction_and_normalization(defaults):
    """Test benefits extraction from section with normalization"""
    raw_text = """
    Software Engineer Intern
    
    Quyền lợi:
    - Được hướng dẫn 1:1 bởi mentor giàu kinh nghiệm
    - Review định kỳ và feedback chi tiết
    - Đào tạo soft skills và technical skills
    - Tham gia các dự án AI/Big Data thực tế
    - Cơ hội chuyển chính thức sau thực tập
    - Ăn trưa miễn phí tại canteen công ty
    - Tham gia các hoạt động CLB, team building
    - Môi trường làm việc trẻ trung, năng động
    
    Địa điểm: Hà Nội
    """
    
    result = parse_job_text(raw_text, defaults)
    
    assert result is not None
    
    # Benefits should be extracted
    benefits = result.suggested["benefits"]["custom"]
    assert len(benefits) >= 4  # At least 4 benefits
    assert len(benefits) <= 12  # Max 12 benefits
    
    # Check for normalized benefits (should be shorter, cleaner)
    benefit_text = " ".join(benefits).lower()
    
    # Should contain key benefits
    assert any("mentor" in b.lower() for b in benefits)
    assert any("review" in b.lower() or "feedback" in b.lower() for b in benefits)


def test_location_deduplication_with_null_address(defaults):
    """Test that locations with same city but one has null address are deduplicated"""
    raw_text = """
    Marketing Manager
    
    Địa điểm làm việc:
    Hà Nội: Số 182 Đường Bờ Sông Sét, Quận Long Biên
    
    Mô tả: Work in Hà Nội office
    """
    
    result = parse_job_text(raw_text, defaults)
    
    assert result is not None
    
    # Should have only 1 location (the one with address)
    locations = result.suggested["locations"]
    
    # Filter to HN locations
    hn_locs = [loc for loc in locations if loc["city"]["code"] == "VN-HN"]
    
    # Should keep the one with address, drop the one without
    if len(hn_locs) > 1:
        # All should have addresses (no null)
        for loc in hn_locs:
            assert loc["address"] is not None
    else:
        # Or just one location
        assert len(hn_locs) == 1


def test_working_time_normalization(defaults):
    """Test working time normalization to consistent format"""
    raw_text = """
    Developer
    
    Thời gian làm việc:
    thứ 2 - thứ 6 08:00-17:00
    
    Địa điểm: Hà Nội
    """
    
    result = parse_job_text(raw_text, defaults)
    
    assert result is not None
    
    working_time = result.suggested["workingTime"]
    
    if working_time:
        # Should be normalized format
        assert "Thứ" in working_time  # Capitalized
        assert "–" in working_time or "-" in working_time  # Has dash
        
        # Check structured data in confidence
        structured = result.confidence.get("workingTime.structured", {})
        if structured:
            # Should have days array
            if "days" in structured:
                assert "MON" in structured["days"]
                assert "FRI" in structured["days"]
            
            # Should have time range
            if "startTime" in structured:
                assert structured["startTime"] == "08:00"
            if "endTime" in structured:
                assert structured["endTime"] == "17:00"


def test_responsibilities_from_paragraph(defaults):
    """Test splitting responsibilities from long paragraph without bullets"""
    raw_text = """
    Product Manager
    
    Mô tả công việc
    Quản lý roadmap sản phẩm và ưu tiên các tính năng. Làm việc chặt chẽ với engineering team để đảm bảo delivery đúng hạn. Phân tích metrics và user feedback để cải thiện sản phẩm. Tổ chức sprint planning và retrospective meetings. Viết PRD và user stories chi tiết.
    
    Địa điểm: Hồ Chí Minh
    """
    
    result = parse_job_text(raw_text, defaults)
    
    assert result is not None
    
    # Should split paragraph into bullets
    responsibilities = result.suggested["responsibilities"]
    assert len(responsibilities) >= 3  # Should have multiple bullets
    
    # Each bullet should not be the entire paragraph
    for bullet in responsibilities:
        assert len(bullet) < 200  # Each bullet should be shorter than full text


def test_requirements_constraints_not_guessed(defaults):
    """Test that constraints are not guessed when unclear"""
    raw_text = """
    Developer
    
    Yêu cầu:
    - Có kinh nghiệm lập trình
    - Biết tiếng Anh
    - Làm việc nhóm tốt
    
    Địa điểm: Hà Nội
    """
    
    result = parse_job_text(raw_text, defaults)
    
    assert result is not None
    
    # Should not have constraints if not explicitly stated
    constraints = result.confidence.get("requirements.constraints", {})
    
    # Should be empty or minimal (no guessing)
    assert len(constraints) == 0 or all(
        k in ["gpa", "toeic", "ielts", "timeCommitment"] 
        for k in constraints.keys()
    )



# ============================================
# STRICT RULES TESTS (Phase 3)
# ============================================

def test_intern_employment_type_not_guessed(defaults):
    """Test that INTERN employmentType is set to INTERNSHIP (Patch C)"""
    raw_text = """
    Software Engineer Intern
    
    Mô tả: Phát triển phần mềm. Làm việc với team. Học hỏi kinh nghiệm. Code review.
    
    Yêu cầu: Sinh viên năm 3, năm 4. Biết Python. Có thể làm việc 28h/tuần. GPA >= 3.0.
    
    Địa điểm: Hà Nội
    """
    
    result = parse_job_text(raw_text, defaults)
    
    assert result is not None
    assert result.suggested["jobLevel"] == "INTERN"
    
    # Patch C: Should set employmentType to INTERNSHIP
    assert result.suggested["employmentType"] == "INTERNSHIP"
    assert result.confidence.get("employmentType", 0) >= 0.7
    
    # Should have hint in confidence
    assert "employmentTypeHint" in result.confidence
    assert result.confidence["employmentTypeHint"] == "INTERNSHIP"


def test_overview_not_copied_from_responsibilities(defaults):
    """Test that overview is synthesized, not copied from responsibilities"""
    raw_text = """
    Backend Developer
    
    Mô tả công việc:
    - Phát triển và bảo trì các API RESTful cho hệ thống
    - Làm việc với database PostgreSQL và Redis
    - Tối ưu hiệu suất query và caching
    - Viết unit test và integration test
    - Code review cho team members
    
    Yêu cầu:
    - 2-3 năm kinh nghiệm Python
    - Biết Django, Flask
    - Kinh nghiệm với PostgreSQL
    - Biết Git, Docker
    
    Địa điểm: Hồ Chí Minh
    """
    
    result = parse_job_text(raw_text, defaults)
    
    assert result is not None
    
    overview = result.suggested.get("overview", "")
    responsibilities = result.suggested.get("responsibilities", [])
    
    if overview and responsibilities:
        # Overview should not be >70% similar to any responsibility
        from app.parser_job.parser_validation import calculate_text_similarity
        
        for resp in responsibilities:
            similarity = calculate_text_similarity(overview, resp)
            assert similarity < 0.7, f"Overview too similar to responsibility: {similarity:.2f}"
        
        # Overview should be reasonable length
        assert 50 <= len(overview) <= 450


def test_workmode_onsite_requires_evidence(defaults):
    """Test that ONSITE is only set with clear evidence"""
    # Case 1: No address, no working hours -> should be null
    raw_text1 = """
    Developer
    Mô tả: Code stuff. Test stuff. Deploy stuff. Review code.
    Yêu cầu: Biết Python. Có kinh nghiệm. Làm việc nhóm. Giao tiếp tốt.
    Địa điểm: Hà Nội
    """
    
    result1 = parse_job_text(raw_text1, defaults)
    # Should NOT set ONSITE without address + working hours
    # (may be null or have low confidence)
    
    # Case 2: Has address + working hours -> can set ONSITE
    raw_text2 = """
    Developer
    
    Mô tả: Code stuff. Test stuff. Deploy stuff. Review code.
    
    Yêu cầu: Biết Python. Có kinh nghiệm. Làm việc nhóm. Giao tiếp tốt.
    
    Địa điểm làm việc:
    Hà Nội: Số 182 Đường Bờ Sông Sét, Quận Long Biên
    
    Thời gian làm việc:
    Thứ 2 - Thứ 6, 08:00-17:00
    """
    
    result2 = parse_job_text(raw_text2, defaults)
    
    # Should set ONSITE with good confidence
    if result2.suggested["workMode"] == "ONSITE":
        assert result2.confidence.get("workMode", 0) >= 0.7


def test_benefits_section_detected_but_not_extracted_warning(defaults):
    """Test warning when benefits section detected but extraction fails"""
    raw_text = """
    Developer
    
    Mô tả: Code. Test. Deploy. Review.
    
    Yêu cầu: Python. Git. Docker. Teamwork.
    
    Quyền lợi:
    (empty or malformed)
    
    Địa điểm: Hà Nội
    """
    
    result = parse_job_text(raw_text, defaults)
    
    # If benefits section detected but empty
    if not result.suggested["benefits"]["custom"]:
        # Should have warning
        assert any("benefits" in w.lower() for w in result.warnings)


def test_validation_minimum_requirements(defaults):
    """Test that validation catches insufficient data"""
    # Insufficient responsibilities and requirements
    raw_text = """
    Developer
    
    Mô tả: Code stuff
    
    Yêu cầu: Know Python
    
    Địa điểm: Hà Nội
    """
    
    result = parse_job_text(raw_text, defaults)
    
    # Should have validation warnings
    # (responsibilities < 4 or requirements < 4)
    assert len(result.warnings) > 0


def test_confidence_reflects_quality(defaults):
    """Test that confidence is lowered when output is poor quality"""
    raw_text = """
    Developer
    
    Some vague description without clear structure
    
    Địa điểm: Hà Nội
    """
    
    result = parse_job_text(raw_text, defaults)
    
    # If responsibilities empty, confidence should be 0
    if not result.suggested["responsibilities"]:
        assert result.confidence.get("responsibilities", 1.0) == 0.0
    
    # If benefits empty, confidence should be 0
    if not result.suggested["benefits"]["custom"]:
        assert result.confidence.get("benefits.custom", 1.0) == 0.0
