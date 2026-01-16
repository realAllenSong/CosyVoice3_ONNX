"""
Frontend module for text processing.

Includes text normalization, language detection, and prosody control.
"""

from .normalizer import TextNormalizer, normalize_text
from .language import LanguageDetector, SUPPORTED_LANGUAGES, SUPPORTED_DIALECTS
from .prosody import ProsodyParser, parse_prosody_tags

__all__ = [
    "TextNormalizer",
    "normalize_text",
    "LanguageDetector", 
    "SUPPORTED_LANGUAGES",
    "SUPPORTED_DIALECTS",
    "ProsodyParser",
    "parse_prosody_tags",
]
