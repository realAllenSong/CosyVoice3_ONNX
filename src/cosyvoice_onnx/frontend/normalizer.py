"""
Text Normalization using WeTextProcessing.

Converts numbers, dates, currencies, etc. to spoken form.
"""

import re
from typing import Optional
from ..utils.logger import get_logger

# Try to import WeTextProcessing
_HAS_WETEXT = False
try:
    from tn.chinese.normalizer import Normalizer as ChineseNormalizer
    from tn.english.normalizer import Normalizer as EnglishNormalizer
    _HAS_WETEXT = True
except ImportError:
    ChineseNormalizer = None
    EnglishNormalizer = None


class TextNormalizer:
    """Text normalizer with WeTextProcessing support.
    
    Converts written text to spoken form:
    - Numbers: "123" → "一百二十三" (zh) / "one hundred twenty three" (en)
    - Dates: "2024-01-15" → "二零二四年一月十五日"
    - Currency: "$100" → "一百美元"
    - Time: "3:30" → "三点三十分"
    """
    
    def __init__(self, default_language: str = "zh"):
        """Initialize normalizer.
        
        Args:
            default_language: Default language for normalization ("zh" or "en")
        """
        self.logger = get_logger()
        self.default_language = default_language
        
        self._zh_normalizer = None
        self._en_normalizer = None
        
        if _HAS_WETEXT:
            try:
                self._zh_normalizer = ChineseNormalizer()
                self.logger.info("Chinese text normalizer loaded")
            except Exception as e:
                self.logger.warning(f"Failed to load Chinese normalizer: {e}")
            
            try:
                self._en_normalizer = EnglishNormalizer()
                self.logger.info("English text normalizer loaded")
            except Exception as e:
                self.logger.warning(f"Failed to load English normalizer: {e}")
        else:
            self.logger.warning(
                "WeTextProcessing not installed. Using fallback normalization. "
                "Install with: pip install WeTextProcessing"
            )
    
    def normalize(self, text: str, language: Optional[str] = None) -> str:
        """Normalize text to spoken form.
        
        Args:
            text: Input text
            language: Language code ("zh", "en", etc.)
            
        Returns:
            Normalized text
        """
        lang = language or self.default_language
        
        if lang == "zh" and self._zh_normalizer:
            try:
                return self._zh_normalizer.normalize(text)
            except Exception as e:
                self.logger.warning(f"Chinese normalization failed: {e}")
                return self._fallback_normalize_zh(text)
        elif lang == "en" and self._en_normalizer:
            try:
                return self._en_normalizer.normalize(text)
            except Exception as e:
                self.logger.warning(f"English normalization failed: {e}")
                return self._fallback_normalize_en(text)
        else:
            # Fallback for other languages or when WeTextProcessing unavailable
            if lang == "zh":
                return self._fallback_normalize_zh(text)
            else:
                return self._fallback_normalize_en(text)
    
    def _fallback_normalize_zh(self, text: str) -> str:
        """Fallback Chinese normalization using regex."""
        # Number mapping
        num_map = {
            '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
        }
        
        # Convert standalone digits to Chinese
        def replace_digits(match):
            digits = match.group(0)
            return ''.join(num_map.get(d, d) for d in digits)
        
        # Replace sequences of digits (simple version)
        result = re.sub(r'\d+', replace_digits, text)
        
        # Common symbol replacements
        replacements = {
            '%': '百分之',
            '$': '美元',
            '¥': '人民币',
            '€': '欧元',
            '£': '英镑',
            '@': '艾特',
            '&': '和',
        }
        
        for symbol, word in replacements.items():
            result = result.replace(symbol, word)
        
        return result
    
    def _fallback_normalize_en(self, text: str) -> str:
        """Fallback English normalization using regex."""
        # Simple digit to word mapping for basic cases
        num_words = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
            '10': 'ten', '11': 'eleven', '12': 'twelve'
        }
        
        result = text
        
        # Replace common abbreviations
        abbreviations = {
            'Mr.': 'Mister',
            'Mrs.': 'Missus', 
            'Dr.': 'Doctor',
            'Jr.': 'Junior',
            'Sr.': 'Senior',
            'vs.': 'versus',
            'etc.': 'et cetera',
        }
        
        for abbr, full in abbreviations.items():
            result = result.replace(abbr, full)
        
        # Replace small numbers (0-12)
        for digit, word in num_words.items():
            result = re.sub(rf'\b{digit}\b', word, result)
        
        return result
    
    @property
    def has_full_support(self) -> bool:
        """Check if full WeTextProcessing is available."""
        return _HAS_WETEXT and (self._zh_normalizer is not None or self._en_normalizer is not None)


# Convenience function
def normalize_text(text: str, language: str = "zh") -> str:
    """Normalize text using default normalizer.
    
    Args:
        text: Input text
        language: Language code
        
    Returns:
        Normalized text
    """
    normalizer = TextNormalizer(default_language=language)
    return normalizer.normalize(text, language)
