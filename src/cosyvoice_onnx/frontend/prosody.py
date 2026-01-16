"""
Prosody control: pinyin, phonemes, emotion tags.

Parses and processes prosody control tags in text.
"""

import re
from typing import Optional, List, Tuple, NamedTuple
from dataclasses import dataclass
from ..utils.logger import get_logger


@dataclass
class ProsodyTag:
    """A prosody control tag."""
    tag_type: str  # 'pinyin', 'phoneme', 'emotion', 'speed', 'pause'
    value: str
    text: str  # The text this tag applies to
    start: int  # Start position in original text
    end: int  # End position in original text


class ProsodyParser:
    """Parse prosody control tags from text.
    
    Supported tags:
    - Pinyin: <pinyin:ce4 shi4>测试</pinyin>
    - Phoneme: <phoneme:T EH1 S T>test</phoneme>
    - Emotion: [happy], [sad], [angry], [calm], [excited]
    - Speed: <speed:1.2>快速</speed>
    - Pause: <pause:500ms/>
    """
    
    # Supported emotions
    EMOTIONS = ['happy', 'sad', 'angry', 'calm', 'excited', 'neutral', 'serious', 'gentle']
    
    def __init__(self):
        self.logger = get_logger()
        
        # Tag patterns
        self._pinyin_pattern = re.compile(
            r'<pinyin:([^>]+)>([^<]*)</pinyin>',
            re.IGNORECASE
        )
        self._phoneme_pattern = re.compile(
            r'<phoneme:([^>]+)>([^<]*)</phoneme>',
            re.IGNORECASE
        )
        self._emotion_pattern = re.compile(
            r'\[(\w+)\]',
            re.IGNORECASE
        )
        self._speed_pattern = re.compile(
            r'<speed:([0-9.]+)>([^<]*)</speed>',
            re.IGNORECASE
        )
        self._pause_pattern = re.compile(
            r'<pause:(\d+)ms\s*/?>',
            re.IGNORECASE
        )
    
    def parse(self, text: str) -> Tuple[str, List[ProsodyTag]]:
        """Parse all prosody tags from text.
        
        Args:
            text: Input text with prosody tags
            
        Returns:
            Tuple of (cleaned_text, list_of_tags)
        """
        tags = []
        
        # Parse pinyin tags
        for match in self._pinyin_pattern.finditer(text):
            tags.append(ProsodyTag(
                tag_type='pinyin',
                value=match.group(1),
                text=match.group(2),
                start=match.start(),
                end=match.end()
            ))
        
        # Parse phoneme tags
        for match in self._phoneme_pattern.finditer(text):
            tags.append(ProsodyTag(
                tag_type='phoneme',
                value=match.group(1),
                text=match.group(2),
                start=match.start(),
                end=match.end()
            ))
        
        # Parse emotion tags
        for match in self._emotion_pattern.finditer(text):
            emotion = match.group(1).lower()
            if emotion in self.EMOTIONS:
                tags.append(ProsodyTag(
                    tag_type='emotion',
                    value=emotion,
                    text='',
                    start=match.start(),
                    end=match.end()
                ))
        
        # Parse speed tags
        for match in self._speed_pattern.finditer(text):
            tags.append(ProsodyTag(
                tag_type='speed',
                value=match.group(1),
                text=match.group(2),
                start=match.start(),
                end=match.end()
            ))
        
        # Parse pause tags
        for match in self._pause_pattern.finditer(text):
            tags.append(ProsodyTag(
                tag_type='pause',
                value=match.group(1),
                text='',
                start=match.start(),
                end=match.end()
            ))
        
        # Clean text by removing all tags
        cleaned = self._clean_text(text)
        
        return cleaned, tags
    
    def _clean_text(self, text: str) -> str:
        """Remove all prosody tags, keeping only the text content."""
        # Remove pinyin tags, keep inner text
        result = self._pinyin_pattern.sub(r'\2', text)
        # Remove phoneme tags, keep inner text
        result = self._phoneme_pattern.sub(r'\2', result)
        # Remove emotion tags
        result = self._emotion_pattern.sub('', result)
        # Remove speed tags, keep inner text
        result = self._speed_pattern.sub(r'\2', result)
        # Remove pause tags
        result = self._pause_pattern.sub('', result)
        # Clean up extra whitespace
        result = re.sub(r'\s+', ' ', result).strip()
        return result
    
    def get_emotion(self, tags: List[ProsodyTag]) -> Optional[str]:
        """Extract emotion from tags."""
        for tag in tags:
            if tag.tag_type == 'emotion':
                return tag.value
        return None
    
    def get_emotion_prefix(self, emotion: Optional[str]) -> str:
        """Get emotion instruction prefix for TTS.
        
        Args:
            emotion: Emotion name
            
        Returns:
            Instruction prefix string
        """
        if not emotion:
            return ""
        
        emotion_prefixes = {
            'happy': '[用快乐的语气说]',
            'sad': '[用悲伤的语气说]',
            'angry': '[用愤怒的语气说]',
            'calm': '[用平静的语气说]',
            'excited': '[用兴奋的语气说]',
            'neutral': '',
            'serious': '[用严肃的语气说]',
            'gentle': '[用温柔的语气说]',
        }
        
        return emotion_prefixes.get(emotion.lower(), "")


def parse_prosody_tags(text: str) -> Tuple[str, List[ProsodyTag]]:
    """Convenience function to parse prosody tags.
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (cleaned_text, tags)
    """
    parser = ProsodyParser()
    return parser.parse(text)
