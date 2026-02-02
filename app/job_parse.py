"""
Job parser - backward compatibility wrapper.
Main implementation moved to app.parser module.
"""
from __future__ import annotations

# Re-export types for backward compatibility
from app.parser_job.parser_types import ParseDefaults, ParseResult

# Re-export main function
from app.parser import parse_job_text


__all__ = ["ParseDefaults", "ParseResult", "parse_job_text"]


