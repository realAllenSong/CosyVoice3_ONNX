"""
High-level API for CosyVoice3 ONNX TTS.
"""

import asyncio
import time
from pathlib import Path
from typing import Optional, Callable, Union, AsyncIterator
import numpy as np

from .config import CosyVoiceConfig
from .model_manager import ModelManager
from .engine import CosyVoiceEngine
from .types import AudioData, PresetVoice, ProgressInfo, AudioChunk
from .utils.logger import setup_logger, get_logger


class CosyVoiceTTS:
    """High-level TTS API for CosyVoice3.
    
    This is the main entry point for text-to-speech synthesis.
    
    Example:
        ```python
        tts = CosyVoiceTTS()
        
        # Basic synthesis
        audio = await tts.synthesize_async("Hello, world!")
        audio.save("output.wav")
        
        # Voice cloning
        audio = await tts.clone_voice_async(
            prompt_audio="speaker.wav",
            prompt_text="Hello, my name is Alice.",
            target_text="Nice to meet you!"
        )
        ```
    """
    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        precision: str = "auto",
        preload: bool = False,
        num_threads: int = 0,
        config: Optional[CosyVoiceConfig] = None,
        log_level: str = "INFO"
    ):
        """Initialize CosyVoice TTS.
        
        Args:
            model_dir: Path to model directory. Default: ~/.cosyvoice3/models
            precision: Precision mode: "fp16", "fp32", or "auto"
            preload: If True, load models immediately
            num_threads: Number of CPU threads (0 = auto)
            config: Full configuration object (overrides other params)
            log_level: Logging level
        """
        # Setup logging
        self.logger = setup_logger(level=log_level)
        
        # Create config
        if config is not None:
            self.config = config
        else:
            self.config = CosyVoiceConfig(
                model_dir=model_dir or str(CosyVoiceConfig().model_dir),
                precision=precision,
                preload=preload,
                num_threads=num_threads,
                log_level=log_level
            )
        
        # Initialize managers
        self.model_manager = ModelManager(self.config)
        self._engine: Optional[CosyVoiceEngine] = None
        self._presets: dict[str, PresetVoice] = {}
        
        # Preload if requested
        if preload:
            self.model_manager.load_models()
            self._engine = CosyVoiceEngine(self.model_manager, self.config)
    
    def _ensure_engine(self) -> CosyVoiceEngine:
        """Ensure engine is ready."""
        if self._engine is None:
            self.model_manager.load_models()
            self._engine = CosyVoiceEngine(self.model_manager, self.config)
        return self._engine
    
    async def synthesize_async(
        self,
        text: str,
        prompt_audio: Optional[Union[str, bytes, np.ndarray]] = None,
        prompt_text: Optional[str] = None,
        speed: float = 1.0,
        volume: float = 1.0,
        emotion: str = "neutral",
        output_format: str = "wav",
        on_progress: Optional[Callable[[ProgressInfo], None]] = None
    ) -> AudioData:
        """Synthesize speech from text asynchronously.
        
        Args:
            text: Text to synthesize
            prompt_audio: Reference audio for voice cloning (path, bytes, or array)
            prompt_text: Transcript of prompt audio (required for voice cloning)
            speed: Speech speed (0.5-2.0, default 1.0)
            volume: Output volume (0.0-2.0, default 1.0)
            emotion: Emotion tag (not fully implemented yet)
            output_format: Output format ("wav" or "mp3")
            on_progress: Progress callback
            
        Returns:
            AudioData containing the generated audio
        """
        # Run synthesis in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._synthesize_sync(
                text, prompt_audio, prompt_text,
                speed, volume, emotion, output_format, on_progress
            )
        )
    
    def synthesize(
        self,
        text: str,
        prompt_audio: Optional[Union[str, bytes, np.ndarray]] = None,
        prompt_text: Optional[str] = None,
        speed: float = 1.0,
        volume: float = 1.0,
        emotion: str = "neutral",
        output_format: str = "wav"
    ) -> AudioData:
        """Synthesize speech from text (synchronous version).
        
        Args:
            text: Text to synthesize
            prompt_audio: Reference audio for voice cloning
            prompt_text: Transcript of prompt audio
            speed: Speech speed
            volume: Output volume
            emotion: Emotion tag
            output_format: Output format
            
        Returns:
            AudioData containing the generated audio
        """
        return self._synthesize_sync(
            text, prompt_audio, prompt_text,
            speed, volume, emotion, output_format, None
        )
    
    def _synthesize_sync(
        self,
        text: str,
        prompt_audio: Optional[Union[str, bytes, np.ndarray]],
        prompt_text: Optional[str],
        speed: float,
        volume: float,
        emotion: str,
        output_format: str,
        on_progress: Optional[Callable]
    ) -> AudioData:
        """Internal synchronous synthesis."""
        import librosa
        
        engine = self._ensure_engine()
        start_time = time.time()
        
        self.logger.info(f"Synthesizing: '{text[:50]}...'")
        
        # Load prompt audio if provided
        if prompt_audio is not None:
            if isinstance(prompt_audio, str):
                audio, sr = librosa.load(prompt_audio, sr=None)
            elif isinstance(prompt_audio, bytes):
                import io
                import soundfile as sf
                audio, sr = sf.read(io.BytesIO(prompt_audio))
            else:
                audio = prompt_audio
                sr = 16000  # Assume 16kHz for numpy arrays
            
            if prompt_text is None:
                raise ValueError("prompt_text is required when using prompt_audio")
            
            # Extract embeddings
            embedding = engine.extract_speaker_embedding(audio, sr)
            prompt_tokens = engine.extract_speech_tokens(audio, sr)
            prompt_mel = engine.extract_speech_mel(audio, sr)
        else:
            # Use first available preset if no prompt
            if self._presets:
                preset = list(self._presets.values())[0]
                return self.synthesize_with_preset(text, preset, speed, volume, output_format)
            else:
                raise ValueError(
                    "Either prompt_audio or a preset voice is required. "
                    "CosyVoice is a voice cloning TTS that needs a reference voice."
                )
        
        # LLM inference
        speech_tokens = engine.llm_inference(
            text,
            prompt_text=prompt_text,
            prompt_speech_tokens=prompt_tokens,
            sampling_k=self.config.sampling_k,
            max_len=self.config.max_tokens,
            min_len=self.config.min_tokens
        )
        
        if on_progress:
            on_progress(ProgressInfo(
                task_id="synth",
                total_chars=len(text),
                processed_chars=len(text) // 2,
                progress_percent=50.0,
                estimated_remaining_ms=1000
            ))
        
        # Flow inference
        mel = engine.flow_inference(
            speech_tokens,
            embedding,
            prompt_tokens=prompt_tokens,
            prompt_mel=prompt_mel,
            n_timesteps=self.config.n_timesteps
        )
        
        # HiFT vocoder
        audio = engine.hift_inference(mel)
        
        # Apply speed (simple time stretching)
        if speed != 1.0:
            import scipy.signal
            audio = scipy.signal.resample(audio, int(len(audio) / speed))
        
        # Apply volume
        if volume != 1.0:
            audio = audio * volume
            audio = np.clip(audio, -1.0, 1.0)
        
        elapsed = time.time() - start_time
        duration_ms = int(len(audio) / CosyVoiceEngine.SAMPLE_RATE * 1000)
        
        self.logger.info(f"Synthesis complete: {duration_ms}ms audio in {elapsed:.2f}s")
        
        if on_progress:
            on_progress(ProgressInfo(
                task_id="synth",
                total_chars=len(text),
                processed_chars=len(text),
                progress_percent=100.0,
                estimated_remaining_ms=0
            ))
        
        return AudioData(
            data=audio.astype(np.float32).tobytes(),
            sample_rate=CosyVoiceEngine.SAMPLE_RATE,
            channels=1,
            format=output_format,
            duration_ms=duration_ms
        )
    
    async def clone_voice_async(
        self,
        prompt_audio: Union[str, bytes, np.ndarray],
        prompt_text: str,
        target_text: str,
        speed: float = 1.0,
        volume: float = 1.0,
        output_format: str = "wav"
    ) -> AudioData:
        """Clone a voice and synthesize target text.
        
        Args:
            prompt_audio: Reference audio (3-15 seconds recommended)
            prompt_text: Transcript of the reference audio
            target_text: Text to synthesize in the cloned voice
            speed: Speech speed
            volume: Output volume
            output_format: Output format
            
        Returns:
            AudioData with synthesized audio
        """
        return await self.synthesize_async(
            text=target_text,
            prompt_audio=prompt_audio,
            prompt_text=prompt_text,
            speed=speed,
            volume=volume,
            output_format=output_format
        )
    
    def clone_voice(
        self,
        prompt_audio: Union[str, bytes, np.ndarray],
        prompt_text: str,
        target_text: str,
        speed: float = 1.0,
        volume: float = 1.0,
        output_format: str = "wav"
    ) -> AudioData:
        """Clone a voice and synthesize target text (synchronous).
        
        Args:
            prompt_audio: Reference audio
            prompt_text: Transcript of reference
            target_text: Text to synthesize
            speed: Speech speed
            volume: Output volume
            output_format: Output format
            
        Returns:
            AudioData with synthesized audio
        """
        return self.synthesize(
            text=target_text,
            prompt_audio=prompt_audio,
            prompt_text=prompt_text,
            speed=speed,
            volume=volume,
            output_format=output_format
        )
    
    def load_preset(self, name: str) -> PresetVoice:
        """Load a preset voice by name.
        
        Args:
            name: Preset name
            
        Returns:
            PresetVoice object
        """
        if name in self._presets:
            return self._presets[name]
        
        # Load from presets directory
        presets_dir = Path(self.config.model_dir).parent / "presets" / "voices"
        metadata_path = Path(self.config.model_dir).parent / "presets" / "metadata.json"
        
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            if name in metadata:
                info = metadata[name]
                preset = PresetVoice(
                    name=name,
                    audio_path=str(presets_dir / info.get("audio", f"{name}.wav")),
                    transcript=info.get("transcript", ""),
                    language=info.get("language", "en"),
                    gender=info.get("gender"),
                    description=info.get("description")
                )
                self._presets[name] = preset
                return preset
        
        raise ValueError(f"Preset '{name}' not found")
    
    def list_presets(self) -> list[str]:
        """List available preset voice names."""
        presets_dir = Path(self.config.model_dir).parent / "presets"
        metadata_path = presets_dir / "metadata.json"
        
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                return list(json.load(f).keys())
        
        return []
    
    def synthesize_with_preset(
        self,
        text: str,
        preset: PresetVoice,
        speed: float = 1.0,
        volume: float = 1.0,
        output_format: str = "wav"
    ) -> AudioData:
        """Synthesize using a preset voice.
        
        Args:
            text: Text to synthesize
            preset: PresetVoice object
            speed: Speech speed
            volume: Output volume
            output_format: Output format
            
        Returns:
            AudioData with synthesized audio
        """
        return self.synthesize(
            text=text,
            prompt_audio=preset.audio_path,
            prompt_text=preset.transcript,
            speed=speed,
            volume=volume,
            output_format=output_format
        )
    
    async def preload_models(self) -> None:
        """Preload all models into memory."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.model_manager.load_models)
        self._engine = CosyVoiceEngine(self.model_manager, self.config)
    
    def unload_models(self) -> None:
        """Unload models to free memory."""
        self.model_manager.unload_models()
        self._engine = None
    
    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self.model_manager.is_loaded


# Convenience function
def create_tts(
    model_dir: Optional[str] = None,
    precision: str = "auto",
    preload: bool = False
) -> CosyVoiceTTS:
    """Create a CosyVoiceTTS instance.
    
    Args:
        model_dir: Model directory path
        precision: Precision mode
        preload: Whether to preload models
        
    Returns:
        CosyVoiceTTS instance
    """
    return CosyVoiceTTS(
        model_dir=model_dir,
        precision=precision,
        preload=preload
    )
