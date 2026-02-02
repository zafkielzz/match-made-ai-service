import re
import json
from datetime import datetime, timezone
import requests
from bs4 import BeautifulSoup
from bs4 import FeatureNotFound

URL = "https://careerviet.vn/en/search-job/chuyen-vien-phat-trien-san-pham-nha-may-so-doanh-nghiep-product-development-specialist-digital-enterprise-factory-khoi-ngan-hang-so-ho25-455.35C5CD41.html"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Upgrade-Insecure-Requests": "1",
    "Connection": "keep-alive",
}

# ---------------------------
# Text helpers
# ---------------------------
JOB_LEVEL_ENUM = {
    "intern", "junior", "mid", "senior", "manager", "lead", "unknown"
}
SCHEDULE_TAGS = {"full_time", "part_time", "shift"}

def apply_default_full_time(job_types: list[str]) -> list[str]:
    if not any(t in SCHEDULE_TAGS for t in job_types):
        job_types.append("full_time")
    return job_types


def norm_space(x) -> str | None:
    """
    Normalize whitespace for strings.
    Accepts str/int/float/list/dict; returns a clean string or None.
    - list/tuple/set -> join items with ", "
    - dict -> json dump
    """
    if x is None:
        return None

    # If already string-like
    if isinstance(x, str):
        s = x
    elif isinstance(x, (int, float, bool)):
        s = str(x)
    elif isinstance(x, (list, tuple, set)):
        parts = []
        for it in x:
            if it is None:
                continue
            if isinstance(it, str):
                parts.append(it)
            else:
                parts.append(str(it))
        s = ", ".join([p for p in parts if p.strip()])
    elif isinstance(x, dict):
        s = json.dumps(x, ensure_ascii=False)
    else:
        s = str(x)

    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def clean_text(x) -> str | None:
    """
    Clean multi-line text.
    Accepts str/list; list will be joined by newlines.
    """
    if x is None:
        return None

    if isinstance(x, (list, tuple)):
        x = "\n".join([str(i) for i in x if i is not None])

    if not isinstance(x, str):
        x = str(x)

    s = x.replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = s.strip()
    return s or None

def pick_first_text(soup: BeautifulSoup, selectors: list[str]) -> str | None:
    for sel in selectors:
        node = soup.select_one(sel)
        if node:
            t = node.get_text(" ", strip=True)
            if t:
                return norm_space(t)
    return None

def norm_key(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip().lower())
    s = s.replace(":", "")
    return s

# ---------------------------
# JSON-LD JobPosting parser (preferred)
# ---------------------------
def parse_jobposting_jsonld(soup: BeautifulSoup) -> dict:
    """
    Try to parse schema.org JobPosting from JSON-LD.
    Returns partial normalized fields if found.
    """
    out: dict[str, str] = {}
    for sc in soup.find_all("script", type="application/ld+json"):
        raw = sc.get_text(strip=True)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue

        items = data if isinstance(data, list) else [data]
        for it in items:
            if not isinstance(it, dict):
                continue

            t = it.get("@type")
            is_job = (t == "JobPosting") or (isinstance(t, list) and "JobPosting" in t)
            if not is_job:
                continue

            out["job_title"] = norm_space(it.get("title") or it.get("name"))

            org = it.get("hiringOrganization") or it.get("hiring_organization") or {}
            if isinstance(org, dict):
                out["company_name"] = norm_space(org.get("name"))

            # location
            loc = it.get("jobLocation")
            locs = loc if isinstance(loc, list) else ([loc] if isinstance(loc, dict) else [])
            if locs and isinstance(locs[0], dict):
                addr = locs[0].get("address") or {}
                if isinstance(addr, dict):
                    out["location_city"] = norm_space(
                        addr.get("addressLocality") or addr.get("addressRegion")
                    )

            out["job_type"] = norm_space(it.get("employmentType"))
            out["industry"] = norm_space(it.get("industry"))
            out["deadline"] = norm_space(it.get("validThrough"))
            out["updated"] = norm_space(it.get("datePosted"))

            # salary (optional, many sites omit or use complex structure)
            base = it.get("baseSalary")
            salary_txt = None
            if isinstance(base, dict):
                val = base.get("value")
                if isinstance(val, dict):
                    if "value" in val and val.get("value") is not None:
                        salary_txt = str(val.get("value"))
                    else:
                        mn = val.get("minValue")
                        mx = val.get("maxValue")
                        if mn is not None or mx is not None:
                            salary_txt = f"{mn or ''} - {mx or ''}".strip(" -")
            if salary_txt:
                out["salary"] = norm_space(salary_txt)

            return {k: v for k, v in out.items() if v}

    return {}

# ---------------------------
# Section helpers
# ---------------------------
def find_section_container_by_heading(soup: BeautifulSoup, heading: str):
    h = heading.strip().lower()
    for tag in soup.find_all(["h2", "h3", "h4", "strong", "div", "p"]):
        t = tag.get_text(" ", strip=True)
        if not t:
            continue
        tt = t.strip().lower()
        # allow exact or substring match (WORK LOCATION / Work location / etc.)
        if tt == h or h in tt:
            return tag.find_parent(["section", "div"]) or tag.parent
    return None

def parse_section_text(soup: BeautifulSoup, heading: str) -> str | None:
    """
    Extract section text by locating heading node and reading content that follows it,
    instead of taking the whole parent container (avoids pulling meta blocks).
    """
    h = heading.strip().lower()

    # Find the heading tag whose text matches/contains heading
    heading_tag = None
    for tag in soup.find_all(["h2", "h3", "h4"]):
        t = tag.get_text(" ", strip=True)
        if t and (t.strip().lower() == h or h in t.strip().lower()):
            heading_tag = tag
            break

    if not heading_tag:
        return None

    # Collect text from following siblings until next heading of same level group
    parts = []
    for sib in heading_tag.find_all_next():
        # stop if we hit another major heading
        if sib.name in ["h2", "h3", "h4"] and sib.get_text(" ", strip=True):
            t2 = sib.get_text(" ", strip=True).strip().lower()
            if t2 != h:
                break

        # Skip scripts/styles/nav
        if sib.name in ["script", "style", "nav"]:
            continue

        # Prefer paragraphs/lists/div blocks
        if sib.name in ["p", "ul", "ol", "li", "div"]:
            txt = sib.get_text("\n", strip=True)
            txt = clean_text(txt)
            if txt:
                parts.append(txt)

        # Safety stop: don't let it run too far
        if len(parts) > 80:
            break

    out = clean_text("\n\n".join(parts))
    return out


def parse_work_location(soup: BeautifulSoup) -> dict:
    """
    Returns {"city":..., "address":...}
    """
    container = find_section_container_by_heading(soup, "Work location")
    if not container:
        return {}

    lines = [ln.strip() for ln in container.get_text("\n", strip=True).split("\n") if ln.strip()]

    # find the line that contains "work location"
    idx = None
    for i, ln in enumerate(lines):
        if "work location" in ln.lower():
            idx = i
            break
    if idx is None:
        return {}

    city = lines[idx + 1] if idx + 1 < len(lines) else None
    address = lines[idx + 2] if idx + 2 < len(lines) else None
    return {"city": norm_space(city), "address": norm_space(address)}

# ---------------------------
# Company name extraction (robust with JSON-LD fallback)
# ---------------------------
def parse_company_name(soup: BeautifulSoup) -> str | None:
    # 1) common anchors to company page
    for sel in [
        "a[href*='/company/']",
        "a[href*='/cong-ty/']",
        "a[href*='company-overview']",
        ".company-name a",
        ".company-name",
    ]:
        node = soup.select_one(sel)
        if node:
            t = node.get_text(" ", strip=True)
            if t and len(t) <= 120:
                return norm_space(t)

    # 2) JSON-LD structured data (HiringOrganization)
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.get_text(strip=True))
        except Exception:
            continue

        items = data if isinstance(data, list) else [data]
        for it in items:
            if not isinstance(it, dict):
                continue
            org = it.get("hiringOrganization") or it.get("hiring_organization")
            if isinstance(org, dict):
                name = org.get("name")
                if name:
                    return norm_space(name)

    return None

# ---------------------------
# Employment Information meta parser (layout-robust)
# ---------------------------
LABEL_ALIASES = {
    # EN
    "industry": {"industry"},
    "salary": {"salary"},
    "job type": {"job type"},
    "job level": {"job level"},
    "experience": {"experience"},
    "deadline to apply": {"deadline to apply"},
    "updated": {"updated"},
    "location": {"location"},
    # VI (in case template is Vietnamese)
    "industry": {"ngành nghề", "lĩnh vực"},
    "salary": {"mức lương", "thu nhập"},
    "job type": {"hình thức", "loại hình"},
    "job level": {"cấp bậc", "cấp bậc công việc"},
    "experience": {"kinh nghiệm"},
    "deadline to apply": {"hạn nộp", "hạn ứng tuyển"},
    "updated": {"cập nhật", "ngày đăng"},
    "location": {"địa điểm", "nơi làm việc"},
}

def canonical_label(s: str) -> str | None:
    k = norm_key(s)
    for canon, aliases in LABEL_ALIASES.items():
        if k in aliases:
            return canon
    return None

def find_employment_info_container(soup: BeautifulSoup):
    patterns = [r"employment information", r"thông tin việc làm", r"thông tin tuyển dụng"]
    for pat in patterns:
        node = soup.find(string=re.compile(pat, re.I))
        if node:
            return node.find_parent(["section", "div"])
    return None
def extract_value_by_label_anywhere(soup: BeautifulSoup, label_variants: list[str]) -> str | None:
    """
    Find a label text anywhere in the page and return the nearest value text.
    Works for layouts where meta is shown in a top summary card (not in Employment Information).
    """
    # Build regex like r"^(Experience|Kinh nghiệm)$"
    pat = re.compile(r"^\s*(%s)\s*$" % "|".join([re.escape(x) for x in label_variants]), re.I)

    # 1) Look for exact text nodes equal to label
    for node in soup.find_all(string=True):
        txt = (node or "").strip()
        if not txt:
            continue
        if not pat.match(txt):
            continue

        # Candidate containers: immediate parent, then parent of parent
        for container in [node.parent, node.parent.parent if node.parent else None]:
            if not container:
                continue

            # Strategy A: value is in the same container's text, after label
            full = container.get_text("\n", strip=True)
            lines = [ln.strip() for ln in full.split("\n") if ln.strip()]
            # find label line
            for i, ln in enumerate(lines):
                if pat.match(ln):
                    # next non-label line is value
                    if i + 1 < len(lines):
                        val = norm_space(lines[i + 1])
                        if val and not pat.match(val):
                            return val

            # Strategy B: value is in next siblings (common in summary cards)
            sib = container.next_sibling
            steps = 0
            while sib is not None and steps < 6:
                if getattr(sib, "get_text", None):
                    v = norm_space(sib.get_text(" ", strip=True))
                    if v and not pat.match(v):
                        return v
                sib = sib.next_sibling
                steps += 1

        # Strategy C: last resort — check next elements in DOM order
        nxt = node.parent.find_next(string=True)
        steps = 0
        while nxt is not None and steps < 12:
            v = norm_space(str(nxt))
            if v and not pat.match(v):
                return v
            nxt = nxt.find_next(string=True) if hasattr(nxt, "find_next") else None
            steps += 1

    return None
def normalize_job_levels(raw_level: str | None, exp_raw: str | None) -> list[str]:
    """
    Return list of job levels a JD accepts.
    Example:
    - "Junior / Senior" -> ["junior", "senior"]
    - "Experienced (Non-Manager)" + 2-5 years -> ["junior", "mid"]
    """
    if not raw_level:
        return []

    s = raw_level.lower()

    levels = set()

    # explicit keywords
    if re.search(r"\bintern|internship|trainee|thực tập\b", s):
        levels.add("intern")
    if re.search(r"\bjunior|fresher|entry\b", s):
        levels.add("junior")
    if re.search(r"\bmid|middle\b", s):
        levels.add("mid")
    if re.search(r"\bsenior|lead|principal\b", s):
        levels.add("senior")
    if re.search(r"\bmanager|head|director|trưởng phòng|quản lý\b", s):
        levels.add("manager")

    # CareerViet specific
    if "experienced" in s and "non-manager" in s:
        min_y, max_y = parse_experience_years(exp_raw)
        levels.add("junior")
        if min_y is not None and min_y >= 2:
            levels.add("mid")
        if min_y is not None and min_y >= 4:
            levels.add("senior")

    return sorted(levels) if levels else ["unknown"]
def normalize_job_types(job_type_raw: str | None) -> list[str]:
    s = norm_space(job_type_raw)
    if not s:
        return []

    s = s.lower().replace("_", " ")

    # split nhẹ nhàng: phẩy, /, |, + ; KHÔNG split bằng dấu '-' để tránh làm vỡ cụm từ
    # (nếu site dùng "Contract - Freelance", ta vẫn bắt được cả 2 trong cùng token)
    parts = re.split(r"\s*[,/|]\s*|\s*\+\s*", s)
    parts = [p.strip() for p in parts if p.strip()]

    tags: list[str] = []
    def add(t: str):
        if t not in tags:
            tags.append(t)

    for p in parts:
        # contract nature (có thể đồng thời xuất hiện)
        if re.search(r"\bpermanent\b|\bchính thức\b|\bkhông thời hạn\b", p):
            add("permanent")
        if re.search(r"\btemporary\b|\bthời vụ\b|\bseasonal\b", p):
            add("temporary")
        if re.search(r"\bproject\b|\bdự án\b", p):
            add("project")
        if re.search(r"\bcontract\b|\bhợp đồng\b", p):
            add("contract")
        if re.search(r"\bfreelance\b|\btự do\b", p):
            add("freelance")
        if re.search(r"\b(intern|internship|trainee)\b|\bthực tập\b", p):
            add("internship")

        # schedule (explicit)
        if re.search(r"\bpart\s*time\b|\bbán thời gian\b", p):
            add("part_time")

        # shift = xoay ca/ca kíp
        if re.search(r"\b(xoay ca|luân phiên|rotating|night shift|3 ca|2 ca)\b", p):
            add("shift")
        if re.search(r"\bfull\s*time\b|\bfulltime\b|\bfull_time\b", p):
            add("full_time")
        if re.search(r"\bpart\s*time\b|\bparttime\b|\bpart_time\b|\bbán thời gian\b", p):
            add("part_time")

    return tags

def parse_experience_and_level_fallback(soup: BeautifulSoup) -> dict[str, str]:
    """
    Fallback for layouts where Experience/Job level appear in the top job summary.
    """
    out = {}

    exp = extract_value_by_label_anywhere(
        soup, ["Experience", "Kinh nghiệm", "Years of experience", "Số năm kinh nghiệm"]
    )
    if exp:
        out["experience"] = exp

    lvl = extract_value_by_label_anywhere(
        soup, ["Job level", "Cấp bậc", "Cấp bậc công việc", "Level"]
    )
    if lvl:
        out["job level"] = lvl

    return out

def parse_employment_kv_strict(soup: BeautifulSoup) -> dict[str, str]:
    """
    Layout-agnostic: scan the EMPLOYMENT INFORMATION container text and extract values
    by label anchors (Industry / Salary / Job type / Job level / Experience / Deadline / Updated / Location).
    """
    container = find_employment_info_container(soup)
    if not container:
        return {}

    # Special handling for Industry: keep links (best signal)
    industry_links = [a.get_text(" ", strip=True) for a in container.find_all("a") if a.get_text(strip=True)]
    industry_links = [norm_space(x) for x in industry_links if x]
    # We'll still parse others via text anchors.

    full = container.get_text("\n", strip=True)
    lines = [norm_space(ln) for ln in full.split("\n") if norm_space(ln)]

    # Normalize label variants into canonical keys
    label_map = {
        "industry": ["industry", "ngành nghề", "lĩnh vực"],
        "salary": ["salary", "mức lương", "thu nhập"],
        "job type": ["job type", "hình thức", "loại hình"],
        "job level": ["job level", "cấp bậc", "cấp bậc công việc"],
        "experience": ["experience", "kinh nghiệm"],
        "deadline to apply": ["deadline to apply", "hạn nộp", "hạn ứng tuyển"],
        "updated": ["updated", "cập nhật", "ngày đăng"],
        "location": ["location", "địa điểm", "nơi làm việc"],
    }

    def canon_of(line: str) -> str | None:
        k = norm_key(line)
        for canon, aliases in label_map.items():
            if k in [norm_key(a) for a in aliases]:
                return canon
        return None

    kv: dict[str, str] = {}
    i = 0
    while i < len(lines):
        canon = canon_of(lines[i])
        if not canon:
            i += 1
            continue

        # value is usually next line (or next few lines until next label)
        j = i + 1
        vals = []
        while j < len(lines):
            if canon_of(lines[j]):
                break
            vals.append(lines[j])
            j += 1

        val = norm_space(" ".join([v for v in vals if v])) if vals else None
        if val:
            kv[canon] = val
        i = j

    # Override industry with link list if available (more precise)
    if industry_links:
        # dedup keep order
        seen, out = set(), []
        for x in industry_links:
            if x not in seen:
                out.append(x); seen.add(x)
        kv["industry"] = ", ".join(out)

    return kv


# ---------------------------
# Experience -> years parser (EN+VI)
# ---------------------------
def normalize_industries(industry_raw: str | list | None) -> list[str]:
    if not industry_raw:
        return []

    if isinstance(industry_raw, list):
        items = industry_raw
    else:
        items = re.split(r"\s*,\s*|\s*/\s*", industry_raw)

    out = []
    for it in items:
        it = norm_space(it)
        if it and it not in out:
            out.append(it)

    return out

def parse_experience_years(exp_raw: str | None) -> tuple[int | None, int | None]:
    """
    Parse experience string into (min_years, max_years).
    Supports: "No experience", "Over 1 Years", "2 - 5 Years", "3 to 5 years", "5 years", "Trên 2 năm", "2 năm", ...
    """
    if not exp_raw:
        return (None, None)

    s = exp_raw.strip().lower()

    # normalize
    s = s.replace("years", "year").replace("yrs", "year")
    s = s.replace("năm", "year")

    # no experience patterns
    if re.search(r"\b(no experience|không yêu cầu kinh nghiệm|không cần kinh nghiệm|0\s*year|0\s*năm)\b", s):
        return (0, 0)

    # range: 2 - 5 year
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*year", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))

    # "3 to 5 year" / "3 đến 5 năm"
    m = re.search(r"(\d+)\s*(to|đến)\s*(\d+)\s*year", s)
    if m:
        return (int(m.group(1)), int(m.group(3)))

    # "over 2 year" / "more than 2 year" / "trên 2 năm"
    m = re.search(r"(over|more than|trên)\s*(\d+)\s*year", s)
    if m:
        return (int(m.group(2)), None)

    # single: "5 year"
    m = re.search(r"\b(\d+)\s*year\b", s)
    if m:
        y = int(m.group(1))
        return (y, y)

    return (None, None)


def infer_level(
    job_title: str | None,
    jd_text: str | None,
    raw_job_level: str | None,
    exp_raw: str | None
) -> str:
    """
    CareerViet-aware normalization:
    - Student/Internship -> intern
    - Entry Level -> junior
    - Experienced (Non-Manager) can be junior OR mid depending on years
    - Team Leader/Supervisor, Manager, Senior Mgmt, Executive Mgmt -> senior
    - If no reliable signal -> unknown (avoid guessing)
    """
    title = (job_title or "").lower()
    level_raw = (raw_job_level or "").strip().lower()
    check_text = f"{title} {level_raw}"

    # 0) internship
    if re.search(r"\b(student|intern|internship|thực tập|trainee)\b", check_text):
        return "intern"

    # years
    min_y, _ = parse_experience_years(exp_raw)

    # 1) Strong signals from CareerViet categorical job level
    # IMPORTANT: These rules override generic IT assumptions.
    if level_raw:
        # Entry level
        if re.search(r"\b(entry level)\b", level_raw):
            return "junior"

        # Experienced (Non-Manager)
        if re.search(r"\bexperienced\b", level_raw) and re.search(r"\bnon\s*-\s*manager\b|\bnon-manager\b", level_raw):
            # If years unknown or low, treat as junior (CareerViet "experienced" != IT "mid")
            if min_y is None:
                return "junior"
            if min_y < 2:
                return "junior"
            if min_y < 4:
                return "mid"
            return "senior"

        # Team Leader / Supervisor
        if re.search(r"\b(team leader|supervisor)\b", level_raw):
            # If explicitly "team leader/supervisor" but years very low, keep mid; else senior
            if min_y is not None and min_y < 2:
                return "mid"
            return "senior"

        # Managerial tiers
        if re.search(r"\b(manager|senior management|executive management)\b", level_raw):
            return "senior"

    # 2) If raw_job_level missing, infer from title keywords (weak, but useful)
    # managerial keywords in Vietnamese/English
    if re.search(r"\bmanager\b|\bhead\b|\bdirector\b|\bchief\b", title) or re.search(
        r"trưởng phòng|giám đốc|quản lý|trưởng bộ phận|head of|director", title
    ):
        return "senior"

    # explicit junior/senior in title
    if re.search(r"\b(senior|principal|lead)\b", title) or re.search(r"chuyên gia|lead", title):
        return "senior"
    if re.search(r"\b(junior|fresher)\b", title) or re.search(r"mới ra trường|thực tập", title):
        return "junior"

    # 3) Experience-only fallback (if you have it)
    if min_y is not None:
        if min_y == 0:
            return "junior"
        if min_y < 2:
            return "junior"
        if min_y < 4:
            return "mid"
        return "senior"

    # 4) If nothing reliable, don't guess
    return "unknown"

# ---------------------------
# Main parse
# ---------------------------
def make_soup(html: str) -> BeautifulSoup:
    # Prefer lxml, fallback to built-in html.parser if lxml not installed
    try:
        return BeautifulSoup(html, "lxml")
    except FeatureNotFound:
        return BeautifulSoup(html, "html.parser")
def parse_careerviet_job(url: str) -> dict:
    r = requests.get(url, headers=HEADERS, timeout=30)
    # DEBUG: nếu bị chặn sẽ thấy ngay ở đây
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {url}\nResponse head: {r.text[:500]}")

    soup = make_soup(r.text)
    # 1) JSON-LD (preferred)
    structured = parse_jobposting_jsonld(soup)

    # 2) DOM fallbacks
    job_title = structured.get("job_title") or pick_first_text(soup, ["h1"])
    company_name = structured.get("company_name") or parse_company_name(soup)

    meta = parse_employment_kv_strict(soup)
    if not meta.get("job type"):
      jt = extract_value_by_label_anywhere(
          soup, ["Job type", "Hình thức", "Loại hình"]
      )
      if jt:
          meta["job type"] = jt
    work_loc = parse_work_location(soup)
    if not meta.get("experience") or not meta.get("job level"):
      fb = parse_experience_and_level_fallback(soup)
      meta.setdefault("experience", fb.get("experience"))
      meta.setdefault("job level", fb.get("job level"))
    industry = structured.get("industry") or meta.get("industry")
    if not meta.get("industry"):
      ind = extract_value_by_label_anywhere(
          soup, ["Industry", "Ngành nghề", "Lĩnh vực"]
      )
      if ind:
          meta["industry"] = ind
    salary = structured.get("salary") or meta.get("salary")
    job_type_struct = structured.get("job_type")
    job_type_meta = meta.get("job type")

    tags_struct = normalize_job_types(job_type_struct)
    tags_meta = normalize_job_types(job_type_meta)

    job_types = []

    # contract nature + others: union
    for t in tags_meta + tags_struct:
        if t not in job_types and t not in SCHEDULE_TAGS:
            job_types.append(t)

    # schedule: meta > struct > default
    if any(t in SCHEDULE_TAGS for t in tags_meta):
        job_types += [t for t in tags_meta if t in SCHEDULE_TAGS]
    elif any(t in SCHEDULE_TAGS for t in tags_struct):
        job_types += [t for t in tags_struct if t in SCHEDULE_TAGS]
    else:
        job_types.append("full_time")


    raw_job_level = meta.get("job level")
    exp_raw = meta.get("experience")

    job_levels = normalize_job_levels(raw_job_level, exp_raw)
    industries = normalize_industries(industry)
    deadline = structured.get("deadline") or meta.get("deadline to apply")

    # location city: prefer JSON-LD -> Work location -> meta location
    location_city = (
        structured.get("location_city")
        or work_loc.get("city")
        or meta.get("location")
    )

    # JD text
    jd_desc = parse_section_text(soup, "Job Description")
    jd_req = parse_section_text(soup, "Job Requirement")
    jd_text = clean_text("\n\n".join([x for x in [jd_desc, jd_req] if x]))

    # infer job level
    level = infer_level(job_title, jd_text, raw_job_level, exp_raw)

    now_iso = datetime.now(timezone.utc).isoformat()
   

    return {
        "source": "careerviet",
        "source_url": url,
        "last_seen_at": now_iso,
        "deadline": deadline,
        "job_title": job_title,
        "company_name": company_name,
        "location_city": location_city,
        "industry": industry,
        "salary": salary,
        "job_types": job_types,
        "experience_raw": exp_raw,
        "job_level": level,
        "jd_text": jd_text,

        # debugging
        "raw_job_level": raw_job_level,
        "work_location": work_loc,
        "meta_debug": meta,
        "jsonld_debug": structured,
    }

if __name__ == "__main__":
    job = parse_careerviet_job(URL)

    for k in [
        "job_title", "company_name", "location_city", "job_level",
        "industry", "salary","job_types","source_url", "deadline"
    ]:
        print(f"{k}: {job.get(k)}")

    print("\n--- experience_raw / raw_job_level ---")
    print("experience_raw:", job.get("experience_raw"))
    print("raw_job_level:", job.get("raw_job_level"))

    print("\n--- jd_text preview ---")
    print((job.get("jd_text") or "")[:700] + "...")

