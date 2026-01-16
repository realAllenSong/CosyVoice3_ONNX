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

__version__ = "0.1.0"

from .api import CosyVoiceTTS, create_tts
from .config import CosyVoiceConfig
from .types import AudioData, PresetVoice, ProgressInfo, AudioChunk
from .model_manager import ModelManager, ModelNotFoundError, ModelDownloadError

__all__ = [
    # Main API
    "CosyVoiceTTS",
    "create_tts",
    # Configuration
    "CosyVoiceConfig",
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
