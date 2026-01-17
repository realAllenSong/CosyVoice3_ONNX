"""
CosyVoice3 ONNX - Simple CPU-based TTS with Voice Cloning

Based on ayousanz/cosy-voice3-onnx implementation.
Provides easy-to-use API for text-to-speech synthesis with zero-shot voice cloning.

Example:
    >>> from cosyvoice_onnx import CosyVoiceTTS
    >>> tts = CosyVoiceTTS()
    >>> audio = tts.clone_voice(
    ...     prompt_audio="speaker.wav",
    ...     prompt_text="Hello, my name is Alice.",
    ...     target_text="Nice to meet you!"
    ... )
    >>> audio.save("output.wav")
"""

__version__ = "0.3.0"  # Added FastAPI server, quality fixes for CosyVoice3

from .api import CosyVoiceTTS, create_tts
from .config import CosyVoiceConfig
from .types import AudioData, PresetVoice, ProgressInfo, AudioChunk
from .model_manager import ModelManager, ModelNotFoundError, ModelDownloadError
from .streaming import StreamingEngine, StreamingConfig
from .audio_processor import AudioProcessor, normalize_volume, concat_audio
from .audio_validator import AudioValidator, AudioValidationResult, validate_reference_audio
from .utils.preset_downloader import download_presets
from .frontend import (
    TextNormalizer, 
    normalize_text,
    LanguageDetector,
    SUPPORTED_LANGUAGES,
    SUPPORTED_DIALECTS,
    ProsodyParser,
    parse_prosody_tags,
)

__all__ = [
    # Main API
    "CosyVoiceTTS",
    "create_tts",
    # Configuration
    "CosyVoiceConfig",
    # Streaming
    "StreamingEngine",
    "StreamingConfig",
    # Audio Processing
    "AudioProcessor",
    "AudioValidator",
    "AudioValidationResult",
    "validate_reference_audio",
    "download_presets",
    "normalize_volume",
    "concat_audio",
    # Text Frontend
    "TextNormalizer",
    "normalize_text",
    "LanguageDetector",
    "SUPPORTED_LANGUAGES",
    "SUPPORTED_DIALECTS",
    "ProsodyParser",
    "parse_prosody_tags",
    # Data types
    "AudioData",
    "PresetVoice", 
    "ProgressInfo",
    "AudioChunk",
    # Model management
    "ModelManager",
    "ModelNotFoundError",
    "ModelDownloadError",
]

