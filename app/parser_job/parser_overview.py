"""
Overview synthesis - generate unique overview without copying responsibilities
"""
import re
from typing import List, Optional


def synthesize_overview(
    title: Optional[str],
    responsibilities: List[str],
    requirements: List[str],
    job_level: Optional[str],
    max_length: int = 450
) -> Optional[str]:
    """
    Synthesize overview from job information.
    Rules:
    - 2-3 sentences, 240-450 chars
    - Sentence 1: role + domain
    - Sentence 2: main responsibilities
    - Sentence 3 (optional): key tech/skills
    - Must NOT copy >70% from any responsibility
    """
    if not title:
        return None
    
    sentences = []
    
    # Sentence 1: Role + level
    role_sentence = f"Vị trí {title}"
    if job_level:
        level_map = {
            "INTERN": "thực tập sinh",
            "JUNIOR": "junior",
            "MID": "middle",
            "SENIOR": "senior",
            "STAFF": "nhân viên"
        }
        level_text = level_map.get(job_level, "")
        if level_text and level_text not in title.lower():
            role_sentence = f"Vị trí {level_text} {title}"
    
    # Extract domain/tech from title
    if any(kw in title.lower() for kw in ["python", "java", "javascript", "react", "node"]):
        role_sentence += " với focus vào phát triển phần mềm"
    elif any(kw in title.lower() for kw in ["data", "analyst", "bi"]):
        role_sentence += " với focus vào phân tích dữ liệu"
    elif any(kw in title.lower() for kw in ["marketing", "content", "seo"]):
        role_sentence += " với focus vào marketing"
    
    sentences.append(role_sentence + ".")
    
    # Sentence 2: Main responsibilities (summarized, not copied)
    if responsibilities:
        # Extract key verbs/actions
        actions = []
        for resp in responsibilities[:3]:  # Only first 3
            # Extract first verb
            words = resp.split()
            if words:
                first_word = words[0].lower()
                if first_word not in actions:
                    actions.append(first_word)
        
        if actions:
            action_text = ", ".join(actions[:3])
            sentences.append(f"Nhiệm vụ chính bao gồm {action_text} và các công việc liên quan.")
    
    # Sentence 3: Key tech/skills from requirements
    if requirements:
        tech_keywords = []
        tech_patterns = [
            r"\b(python|java|javascript|react|node|vue|angular|django|flask|spring)\b",
            r"\b(sql|postgresql|mysql|mongodb|redis)\b",
            r"\b(aws|azure|gcp|docker|kubernetes)\b",
            r"\b(git|agile|scrum)\b"
        ]
        
        for req in requirements[:5]:
            req_lower = req.lower()
            for pattern in tech_patterns:
                matches = re.findall(pattern, req_lower)
                tech_keywords.extend(matches)
        
        # Dedupe and limit
        tech_keywords = list(dict.fromkeys(tech_keywords))[:4]
        
        if tech_keywords:
            tech_text = ", ".join(tech_keywords)
            sentences.append(f"Yêu cầu kinh nghiệm với {tech_text}.")
    
    # Combine sentences
    overview = " ".join(sentences)
    
    # Ensure length constraints
    if len(overview) < 100:
        # Too short, add more context
        if responsibilities and len(responsibilities) > 3:
            overview += f" Vị trí này đòi hỏi {len(responsibilities)} kỹ năng và trách nhiệm chính."
    
    if len(overview) > max_length:
        overview = overview[:max_length].rsplit(".", 1)[0] + "."
    
    # Validate uniqueness
    from app.parser_job.parser_validation import validate_overview_uniqueness
    if not validate_overview_uniqueness(overview, responsibilities):
        # Too similar, return None
        return None
    
    return overview if len(overview) >= 50 else None
