"""
Language detection and dialect support.

Supports 9 languages and 18+ Chinese dialects.
"""

import re
from typing import Optional, Tuple
from ..utils.logger import get_logger


# Supported languages
SUPPORTED_LANGUAGES = {
    "zh": "Chinese (Mandarin)",
    "en": "English", 
    "jp": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "es": "Spanish",
    "ru": "Russian",
}

# Supported Chinese dialects
SUPPORTED_DIALECTS = {
    "canton": "粤语 (Cantonese)",
    "minnan": "闽南话 (Minnan)",
    "sichuan": "四川话 (Sichuanese)",
    "dongbei": "东北话 (Northeastern)",
    "shaanxi": "陕西话 (Shaanxi)",
    "shanxi": "山西话 (Shanxi)",
    "shanghai": "上海话 (Shanghainese)",
    "tianjin": "天津话 (Tianjin)",
    "shandong": "山东话 (Shandong)",
    "ningxia": "宁夏话 (Ningxia)",
    "gansu": "甘肃话 (Gansu)",
    "henan": "河南话 (Henan)",
    "hubei": "湖北话 (Hubei)",
    "hunan": "湖南话 (Hunan)",
    "jiangxi": "江西话 (Jiangxi)",
    "anhui": "安徽话 (Anhui)",
    "zhejiang": "浙江话 (Zhejiang)",
    "fujian": "福建话 (Fujian)",
}


class LanguageDetector:
    """Detect language from text or use explicit specification."""
    
    def __init__(self):
        self.logger = get_logger()
        
        # Unicode ranges for language detection
        self._chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        self._japanese_pattern = re.compile(r'[\u3040-\u309f\u30a0-\u30ff]')
        self._korean_pattern = re.compile(r'[\uac00-\ud7af]')
        self._cyrillic_pattern = re.compile(r'[\u0400-\u04ff]')
    
    def detect(self, text: str) -> str:
        """Detect language from text.
        
        Args:
            text: Input text
            
        Returns:
            Detected language code
        """
        # Count character types
        chinese_count = len(self._chinese_pattern.findall(text))
        japanese_count = len(self._japanese_pattern.findall(text))
        korean_count = len(self._korean_pattern.findall(text))
        cyrillic_count = len(self._cyrillic_pattern.findall(text))
        
        total_len = len(text)
        if total_len == 0:
            return "en"
        
        # Japanese has hiragana/katakana
        if japanese_count > 0:
            return "jp"
        
        # Korean
        if korean_count > chinese_count:
            return "ko"
        
        # Cyrillic = Russian
        if cyrillic_count > total_len * 0.3:
            return "ru"
        
        # Chinese
        if chinese_count > total_len * 0.3:
            return "zh"
        
        # Default to English
        return "en"
    
    def parse_dialect_tags(self, text: str) -> Tuple[str, Optional[str]]:
        """Parse dialect tags from text.
        
        Args:
            text: Input text with optional dialect tags
            
        Returns:
            Tuple of (cleaned_text, dialect_code)
            
        Example:
            >>> parse_dialect_tags("<dialect:canton>你好</dialect>")
            ("你好", "canton")
        """
        # Pattern: <dialect:xxx>text</dialect>
        pattern = r'<dialect:(\w+)>(.*?)</dialect>'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            dialect = match.group(1).lower()
            inner_text = match.group(2)
            
            if dialect in SUPPORTED_DIALECTS:
                # Remove the dialect tag from text
                cleaned = re.sub(pattern, inner_text, text, flags=re.DOTALL)
                return cleaned, dialect
            else:
                self.logger.warning(f"Unknown dialect: {dialect}. Supported: {list(SUPPORTED_DIALECTS.keys())}")
        
        return text, None
    
    def get_language_prefix(self, language: str, dialect: Optional[str] = None) -> str:
        """Get the language/dialect prefix for TTS prompt.
        
        Args:
            language: Language code
            dialect: Optional dialect code
            
        Returns:
            Prefix string to prepend to prompt
        """
        if dialect and dialect in SUPPORTED_DIALECTS:
            # Dialect-specific prefix (in Chinese)
            dialect_name = SUPPORTED_DIALECTS[dialect].split()[0]
            return f"[用{dialect_name}说]"
        
        # Language-specific prefixes
        lang_prefixes = {
            "zh": "",  # Default, no prefix needed
            "en": "",  # English also default
            "jp": "[用日语说]",
            "ko": "[用韩语说]",
            "de": "[用德语说]",
            "fr": "[用法语说]",
            "it": "[用意大利语说]",
            "es": "[用西班牙语说]",
            "ru": "[用俄语说]",
        }
        
        return lang_prefixes.get(language, "")
