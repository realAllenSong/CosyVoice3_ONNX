"""
Audio post-processing utilities.

Provides volume normalization, audio concatenation, and format conversion.
"""

import io
from typing import List, Optional, Union
import numpy as np
from .types import AudioData
from .utils.logger import get_logger


class AudioProcessor:
    """Audio post-processing utilities.
    
    Features:
    - Volume normalization (peak/RMS)
    - Audio concatenation
    - Format conversion (WAV, MP3)
    """
    
    def __init__(self, sample_rate: int = 24000):
        """Initialize audio processor.
        
        Args:
            sample_rate: Default sample rate
        """
        self.sample_rate = sample_rate
        self.logger = get_logger()
    
    def normalize_volume(
        self,
        audio: Union[AudioData, np.ndarray],
        target_db: float = -20.0,
        method: str = "peak"
    ) -> Union[AudioData, np.ndarray]:
        """Normalize audio volume.
        
        Args:
            audio: Input audio (AudioData or numpy array)
            target_db: Target level in dB (default -20 dB)
            method: "peak" or "rms"
            
        Returns:
            Normalized audio (same type as input)
        """
        # Extract array
        if isinstance(audio, AudioData):
            arr = audio.to_numpy()
            return_audio_data = True
            sample_rate = audio.sample_rate
            channels = audio.channels
            fmt = audio.format
        else:
            arr = audio
            return_audio_data = False
        
        # Calculate current level
        if method == "peak":
            current_level = np.max(np.abs(arr))
            if current_level == 0:
                return audio
            target_linear = 10 ** (target_db / 20)
            gain = target_linear / current_level
        else:  # RMS
            rms = np.sqrt(np.mean(arr ** 2))
            if rms == 0:
                return audio
            target_rms = 10 ** (target_db / 20)
            gain = target_rms / rms
        
        # Apply gain
        normalized = arr * gain
        
        # Clip to prevent clipping
        normalized = np.clip(normalized, -1.0, 1.0)
        
        if return_audio_data:
            return AudioData(
                data=normalized.astype(np.float32).tobytes(),
                sample_rate=sample_rate,
                channels=channels,
                format=fmt,
                duration_ms=int(len(normalized) / sample_rate * 1000)
            )
        return normalized
    
    def concat_audio(
        self,
        audio_list: List[Union[AudioData, np.ndarray]],
        gap_ms: int = 0
    ) -> AudioData:
        """Concatenate multiple audio segments.
        
        Args:
            audio_list: List of AudioData or numpy arrays
            gap_ms: Gap between segments in milliseconds
            
        Returns:
            Concatenated AudioData
        """
        if not audio_list:
            raise ValueError("audio_list cannot be empty")
        
        arrays = []
        sample_rate = self.sample_rate
        
        for i, audio in enumerate(audio_list):
            if isinstance(audio, AudioData):
                arr = audio.to_numpy()
                if i == 0:
                    sample_rate = audio.sample_rate
            else:
                arr = audio
            
            arrays.append(arr)
            
            # Add silence gap
            if gap_ms > 0 and i < len(audio_list) - 1:
                gap_samples = int(sample_rate * gap_ms / 1000)
                arrays.append(np.zeros(gap_samples, dtype=np.float32))
        
        # Concatenate
        combined = np.concatenate(arrays)
        
        return AudioData(
            data=combined.astype(np.float32).tobytes(),
            sample_rate=sample_rate,
            channels=1,
            format="wav",
            duration_ms=int(len(combined) / sample_rate * 1000)
        )
    
    def convert_format(
        self,
        audio: AudioData,
        target_format: str = "wav",
        bitrate: str = "128k"
    ) -> bytes:
        """Convert audio to different format.
        
        Args:
            audio: Input AudioData
            target_format: Target format ("wav" or "mp3")
            bitrate: MP3 bitrate (default "128k")
            
        Returns:
            Audio bytes in target format
        """
        if target_format == "wav":
            return audio.to_bytes("wav")
        
        elif target_format == "mp3":
            try:
                from pydub import AudioSegment
                
                arr = audio.to_numpy()
                # Convert to int16 for MP3
                audio_int16 = (arr * 32767).astype(np.int16)
                
                segment = AudioSegment(
                    audio_int16.tobytes(),
                    frame_rate=audio.sample_rate,
                    sample_width=2,
                    channels=audio.channels
                )
                
                buffer = io.BytesIO()
                segment.export(buffer, format="mp3", bitrate=bitrate)
                return buffer.getvalue()
                
            except ImportError:
                self.logger.error(
                    "MP3 export requires pydub and ffmpeg. "
                    "Install with: pip install pydub && brew install ffmpeg"
                )
                raise ImportError("pydub not installed for MP3 export")
        else:
            raise ValueError(f"Unsupported format: {target_format}")
    
    def trim_silence(
        self,
        audio: Union[AudioData, np.ndarray],
        threshold_db: float = -40.0,
        min_silence_ms: int = 100
    ) -> Union[AudioData, np.ndarray]:
        """Trim silence from start and end of audio.
        
        Args:
            audio: Input audio
            threshold_db: Silence threshold in dB
            min_silence_ms: Minimum silence duration to trim
            
        Returns:
            Trimmed audio
        """
        if isinstance(audio, AudioData):
            arr = audio.to_numpy()
            sample_rate = audio.sample_rate
            return_audio_data = True
        else:
            arr = audio
            sample_rate = self.sample_rate
            return_audio_data = False
        
        threshold = 10 ** (threshold_db / 20)
        min_samples = int(sample_rate * min_silence_ms / 1000)
        
        # Find non-silent regions
        above_threshold = np.abs(arr) > threshold
        
        # Find first and last non-silent sample
        nonzero_indices = np.where(above_threshold)[0]
        
        if len(nonzero_indices) == 0:
            # All silence
            if return_audio_data:
                return AudioData(
                    data=np.array([], dtype=np.float32).tobytes(),
                    sample_rate=sample_rate,
                    channels=1,
                    format="wav",
                    duration_ms=0
                )
            return np.array([], dtype=np.float32)
        
        start = max(0, nonzero_indices[0] - min_samples)
        end = min(len(arr), nonzero_indices[-1] + min_samples)
        
        trimmed = arr[start:end]
        
        if return_audio_data:
            return AudioData(
                data=trimmed.astype(np.float32).tobytes(),
                sample_rate=sample_rate,
                channels=1,
                format="wav",
                duration_ms=int(len(trimmed) / sample_rate * 1000)
            )
        return trimmed


# Convenience functions
def normalize_volume(audio: AudioData, target_db: float = -20.0) -> AudioData:
    """Normalize audio volume."""
    processor = AudioProcessor(audio.sample_rate)
    return processor.normalize_volume(audio, target_db)


def concat_audio(audio_list: List[AudioData], gap_ms: int = 0) -> AudioData:
    """Concatenate audio segments."""
    processor = AudioProcessor()
    return processor.concat_audio(audio_list, gap_ms)
