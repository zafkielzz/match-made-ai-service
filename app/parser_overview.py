"""Overview synthesis with HARD anti-abbreviation rules"""
import re
from typing import List, Optional, Set

DOMAIN_KEYWORDS = {
    "machine learning", "ml", "deep learning", "dl",
    "computer vision", "cv", "ai camera",
    "generative ai", "gen ai", "multimodal", "edge ai",
    "artificial intelligence", "ai", "nlp",
    "data science", "backend", "frontend", "fullstack"
}

def extract_domain_keywords(text: str) -> Set[str]:
    text_lower = text.lower()
    return {kw for kw in DOMAIN_KEYWORDS if kw in text_lower}

def has_abbreviated_verb_list(text: str) -> bool:
    """
    HARD RULE: Check for abbreviated verb lists.
    Pattern: 1-4 letter words separated by commas (e.g., "nắm, tham, ứng")
    Regex: \\b(\\w{1,4})(,\\s*\\w{1,4}){1,}\\b
    """
    # Match pattern: short_word, short_word, short_word
    pattern = r'\b([a-zA-ZÀ-ỹ]{1,4})(,\s*[a-zA-ZÀ-ỹ]{1,4}){1,}\b'
    return bool(re.search(pattern, text))

def synthesize_overview(
    title: Optional[str],
    responsibilities: List[str],
    requirements: List[str],
    job_level: Optional[str],
    max_length: int = 450
) -> Optional[str]:
    """
    Synthesize overview with HARD rules.
    Returns None if cannot meet quality standards.
    
    Minimum requirements:
    - >= 2 responsibilities
    - >= 2 domain keywords
    - NO abbreviated verb lists
    """
    if not title:
        return None
    
    # Requirement 1: >= 2 responsibilities
    if len(responsibilities) < 2:
        return None
    
    # Extract domain keywords
    all_text = f"{title} {' '.join(requirements[:5])}"
    domain_keywords = extract_domain_keywords(all_text)
    
    # Requirement 2: >= 2 domain keywords
    if len(domain_keywords) < 2:
        return None
    
    # Build overview using safe template
    sentences = []
    
    # Sentence 1: Role + domain context
    role_part = f"{title}"
    if job_level == "INTERN":
        role_part = f"{title}"
    
    # Add domain context
    domain_context = ""
    if "machine learning" in domain_keywords or "deep learning" in domain_keywords:
        if "computer vision" in domain_keywords or "cv" in domain_keywords:
            domain_context = "(Computer Vision)"
        else:
            domain_context = "(Machine Learning)"
    elif "computer vision" in domain_keywords:
        domain_context = "(Computer Vision)"
    elif "generative ai" in domain_keywords:
        domain_context = "(Generative AI)"
    
    # Safe template (Vietnamese)
    if domain_context:
        sentence1 = f"{role_part} {domain_context} tham gia nghiên cứu và triển khai các dự án AI"
    else:
        sentence1 = f"{role_part} tham gia phát triển và triển khai các giải pháp kỹ thuật"
    
    # Add specific domain
    if "computer vision" in domain_keywords:
        sentence1 += "/Computer Vision"
    if "machine learning" in domain_keywords or "deep learning" in domain_keywords:
        sentence1 += ", ứng dụng Machine Learning và Deep Learning trong bối cảnh sản phẩm thực tế"
    else:
        sentence1 += " trong bối cảnh sản phẩm thực tế"
    
    sentences.append(sentence1 + ".")
    
    # Sentence 2: Role focus
    sentence2 = "Vai trò tập trung vào xây dựng và tối ưu giải pháp"
    if "ai" in domain_keywords or "machine learning" in domain_keywords:
        sentence2 += " AI"
    else:
        sentence2 += " kỹ thuật"
    
    # Add emerging tech if present
    emerging_tech = []
    if "generative ai" in domain_keywords:
        emerging_tech.append("Generative AI")
    if "multimodal" in domain_keywords:
        emerging_tech.append("Multimodal AI")
    if "edge ai" in domain_keywords:
        emerging_tech.append("Edge AI")
    
    if emerging_tech:
        sentence2 += f", đồng thời tiếp cận các công nghệ mới như {' và '.join(emerging_tech)}"
    
    sentences.append(sentence2 + ".")
    
    # Combine
    overview = " ".join(sentences)
    
    # HARD RULE VALIDATION
    # Check 1: NO abbreviated verb lists
    if has_abbreviated_verb_list(overview):
        return None
    
    # Check 2: Minimum 20 words
    if len(overview.split()) < 20:
        return None
    
    # Check 3: Length constraints
    if len(overview) > max_length:
        overview = overview[:max_length].rsplit(".", 1)[0] + "."
    
    # Check 4: Uniqueness
    from app.parser_validation import validate_overview_uniqueness
    if not validate_overview_uniqueness(overview, responsibilities):
        return None
    
    return overview if len(overview) >= 50 else None

