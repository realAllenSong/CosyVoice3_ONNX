"""
Audio validation and preprocessing for CosyVoice3.

CosyVoice3 has specific requirements for reference audio:
- Recommended sample rate: 24kHz (auto-resampled if different)
- Recommended duration: 3-15 seconds
- Format: mono audio
"""

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional, Tuple, List
import numpy as np

from .utils.logger import get_logger


@dataclass
class AudioValidationResult:
    """Result of audio validation."""
    valid: bool
    sample_rate: int
    duration_seconds: float
    channels: int
    format: str
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Processed audio (resampled to 24kHz mono if needed)
    processed_audio: Optional[np.ndarray] = None
    processed_sample_rate: int = 24000


class AudioValidator:
    """Validates and preprocesses reference audio for CosyVoice3.

    CosyVoice3 requires specific audio characteristics for optimal quality:
    - Sample rate: 24kHz (for mel extraction, 16kHz for speaker embedding)
    - Duration: 3-15 seconds recommended
    - Channels: mono

    This validator checks audio and auto-resamples when necessary.
    """

    # CosyVoice3 target sample rate
    TARGET_SAMPLE_RATE = 24000

    # Duration limits
    MIN_DURATION_SECONDS = 1.0
    MAX_DURATION_SECONDS = 30.0
    RECOMMENDED_MIN_DURATION = 3.0
    RECOMMENDED_MAX_DURATION = 15.0

    # Supported formats
    SUPPORTED_FORMATS = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac'}

    def __init__(self, auto_resample: bool = True):
        """Initialize the audio validator.

        Args:
            auto_resample: If True, automatically resample non-24kHz audio
        """
        self.auto_resample = auto_resample
        self.logger = get_logger()

    def validate(
        self,
        audio_input: Union[str, bytes, np.ndarray, Path],
        sample_rate: Optional[int] = None
    ) -> AudioValidationResult:
        """Validate audio input.

        Args:
            audio_input: Audio file path, bytes, or numpy array
            sample_rate: Sample rate (required if audio_input is numpy array)

        Returns:
            AudioValidationResult with validation status and processed audio
        """
        warnings = []
        errors = []

        # Load audio based on input type
        try:
            audio, sr, fmt = self._load_audio(audio_input, sample_rate)
        except Exception as e:
            return AudioValidationResult(
                valid=False,
                sample_rate=0,
                duration_seconds=0.0,
                channels=0,
                format="unknown",
                errors=[f"Failed to load audio: {str(e)}"]
            )

        # Calculate duration
        duration = len(audio) / sr

        # Check format
        if fmt.lower() not in self.SUPPORTED_FORMATS and fmt != "numpy":
            warnings.append(f"Format '{fmt}' may not be fully supported")

        # Check channels (convert to mono if needed)
        channels = 1
        if audio.ndim > 1:
            channels = audio.shape[1] if audio.shape[1] < audio.shape[0] else audio.shape[0]
            if channels > 1:
                warnings.append(f"Converting {channels}-channel audio to mono")
                audio = audio.mean(axis=-1) if audio.shape[-1] == channels else audio.mean(axis=0)

        # Check duration
        if duration < self.MIN_DURATION_SECONDS:
            errors.append(
                f"Audio too short: {duration:.2f}s (minimum: {self.MIN_DURATION_SECONDS}s)"
            )
        elif duration > self.MAX_DURATION_SECONDS:
            errors.append(
                f"Audio too long: {duration:.2f}s (maximum: {self.MAX_DURATION_SECONDS}s)"
            )
        elif duration < self.RECOMMENDED_MIN_DURATION:
            warnings.append(
                f"Audio shorter than recommended: {duration:.2f}s "
                f"(recommended: {self.RECOMMENDED_MIN_DURATION}-{self.RECOMMENDED_MAX_DURATION}s)"
            )
        elif duration > self.RECOMMENDED_MAX_DURATION:
            warnings.append(
                f"Audio longer than recommended: {duration:.2f}s "
                f"(recommended: {self.RECOMMENDED_MIN_DURATION}-{self.RECOMMENDED_MAX_DURATION}s)"
            )

        # Check sample rate and resample if needed
        processed_audio = audio
        processed_sr = sr

        if sr != self.TARGET_SAMPLE_RATE:
            if self.auto_resample:
                warnings.append(
                    f"Resampling audio from {sr}Hz to {self.TARGET_SAMPLE_RATE}Hz"
                )
                processed_audio = self._resample(audio, sr, self.TARGET_SAMPLE_RATE)
                processed_sr = self.TARGET_SAMPLE_RATE
            else:
                warnings.append(
                    f"Sample rate {sr}Hz differs from optimal {self.TARGET_SAMPLE_RATE}Hz. "
                    f"Consider resampling for better quality."
                )

        # Normalize audio to float32 in range [-1, 1]
        processed_audio = self._normalize(processed_audio)

        # Determine if valid
        valid = len(errors) == 0

        # Log validation result
        if valid:
            if warnings:
                self.logger.info(f"Audio validated with warnings: {warnings}")
            else:
                self.logger.info(f"Audio validated successfully: {duration:.2f}s at {sr}Hz")
        else:
            self.logger.error(f"Audio validation failed: {errors}")

        return AudioValidationResult(
            valid=valid,
            sample_rate=sr,
            duration_seconds=duration,
            channels=channels,
            format=fmt,
            warnings=warnings,
            errors=errors,
            processed_audio=processed_audio if valid else None,
            processed_sample_rate=processed_sr
        )

    def _load_audio(
        self,
        audio_input: Union[str, bytes, np.ndarray, Path],
        sample_rate: Optional[int]
    ) -> Tuple[np.ndarray, int, str]:
        """Load audio from various input types.

        Returns:
            Tuple of (audio_array, sample_rate, format)
        """
        import librosa

        if isinstance(audio_input, np.ndarray):
            if sample_rate is None:
                raise ValueError("sample_rate is required when audio_input is numpy array")
            return audio_input.astype(np.float32), sample_rate, "numpy"

        elif isinstance(audio_input, bytes):
            import soundfile as sf
            audio, sr = sf.read(io.BytesIO(audio_input))
            return audio.astype(np.float32), sr, "bytes"

        elif isinstance(audio_input, (str, Path)):
            path = Path(audio_input)
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {path}")

            fmt = path.suffix.lstrip('.').lower()
            audio, sr = librosa.load(str(path), sr=None)
            return audio.astype(np.float32), sr, fmt

        else:
            raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to float32 in range [-1, 1]."""
        audio = audio.astype(np.float32)

        # Check if already normalized
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            # Likely int16 or similar, normalize
            audio = audio / max_val

        return audio

    def validate_and_prepare(
        self,
        audio_input: Union[str, bytes, np.ndarray, Path],
        sample_rate: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """Validate audio and return processed audio ready for use.

        Args:
            audio_input: Audio file path, bytes, or numpy array
            sample_rate: Sample rate (required if audio_input is numpy array)

        Returns:
            Tuple of (processed_audio, sample_rate)

        Raises:
            ValueError: If audio validation fails
        """
        result = self.validate(audio_input, sample_rate)

        if not result.valid:
            error_msg = "; ".join(result.errors)
            raise ValueError(f"Audio validation failed: {error_msg}")

        return result.processed_audio, result.processed_sample_rate


# Convenience function
def validate_reference_audio(
    audio_input: Union[str, bytes, np.ndarray, Path],
    sample_rate: Optional[int] = None,
    auto_resample: bool = True
) -> AudioValidationResult:
    """Validate reference audio for CosyVoice3.

    Args:
        audio_input: Audio file path, bytes, or numpy array
        sample_rate: Sample rate (required if audio_input is numpy array)
        auto_resample: If True, automatically resample to 24kHz

    Returns:
        AudioValidationResult with validation status and processed audio
    """
    validator = AudioValidator(auto_resample=auto_resample)
    return validator.validate(audio_input, sample_rate)
