"""
Data types for CosyVoice3 ONNX.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class AudioData:
    """Container for generated audio data."""
    
    data: bytes
    sample_rate: int
    channels: int
    format: str  # "wav" or "mp3"
    duration_ms: int
    
    def save(self, path: str) -> None:
        """Save audio to file.
        
        Args:
            path: Output file path
        """
        import soundfile as sf
        
        # Convert bytes back to numpy array
        audio_array = np.frombuffer(self.data, dtype=np.float32)
        
        if self.format == "mp3":
            # MP3 requires pydub
            try:
                from pydub import AudioSegment
                import io
                
                # Convert to int16 for MP3
                audio_int16 = (audio_array * 32767).astype(np.int16)
                
                # Create AudioSegment
                segment = AudioSegment(
                    audio_int16.tobytes(),
                    frame_rate=self.sample_rate,
                    sample_width=2,  # 16-bit
                    channels=self.channels
                )
                segment.export(path, format="mp3")
            except ImportError:
                # Fall back to WAV if pydub not available
                sf.write(path.replace(".mp3", ".wav"), audio_array, self.sample_rate)
        else:
            # WAV format
            sf.write(path, audio_array, self.sample_rate)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array.
        
        Returns:
            Audio samples as float32 numpy array
        """
        return np.frombuffer(self.data, dtype=np.float32).copy()
    
    def to_bytes(self, format: str = "wav") -> bytes:
        """Convert to bytes in specified format.
        
        Args:
            format: Output format ("wav" or "mp3")
            
        Returns:
            Audio data as bytes
        """
        import io
        import soundfile as sf
        
        audio_array = np.frombuffer(self.data, dtype=np.float32)
        
        if format == "wav":
            buffer = io.BytesIO()
            sf.write(buffer, audio_array, self.sample_rate, format='WAV')
            return buffer.getvalue()
        elif format == "mp3":
            try:
                from pydub import AudioSegment
                
                audio_int16 = (audio_array * 32767).astype(np.int16)
                segment = AudioSegment(
                    audio_int16.tobytes(),
                    frame_rate=self.sample_rate,
                    sample_width=2,
                    channels=self.channels
                )
                buffer = io.BytesIO()
                segment.export(buffer, format="mp3")
                return buffer.getvalue()
            except ImportError:
                raise ImportError("MP3 export requires pydub. Install with: pip install pydub")
        else:
            raise ValueError(f"Unsupported format: {format}")


@dataclass
class PresetVoice:
    """A preset voice configuration."""
    
    name: str
    audio_path: str
    transcript: str
    language: str
    gender: Optional[str] = None
    description: Optional[str] = None
    
    # Cached embeddings
    _embedding: Optional[np.ndarray] = None
    _speech_tokens: Optional[np.ndarray] = None
    _mel: Optional[np.ndarray] = None


@dataclass
class ProgressInfo:
    """Progress information for synthesis."""
    
    task_id: str
    total_chars: int
    processed_chars: int
    progress_percent: float
    estimated_remaining_ms: int


@dataclass
class AudioChunk:
    """A chunk of streaming audio."""
    
    data: bytes
    sample_rate: int
    channels: int
    format: str  # 'pcm16' or 'float32'
    is_final: bool
    chunk_index: int = 0
