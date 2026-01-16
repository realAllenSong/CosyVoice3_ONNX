"""
Streaming output implementation for CosyVoice3 ONNX.

Provides chunked audio generation for real-time playback.
"""

import asyncio
import threading
import time
from typing import Optional, Callable, AsyncIterator, Tuple
from dataclasses import dataclass
import numpy as np

from .engine import CosyVoiceEngine
from .config import CosyVoiceConfig
from .types import AudioChunk, ProgressInfo
from .utils.logger import get_logger


@dataclass
class StreamingConfig:
    """Configuration for streaming output."""
    
    chunk_size_tokens: int = 30  # Tokens per chunk
    min_tokens: int = 10         # Minimum tokens before yielding
    max_tokens: int = 500        # Maximum total tokens
    sampling_k: int = 25         # Top-k sampling
    n_timesteps: int = 10        # Flow steps


class StreamingEngine:
    """Streaming inference engine.
    
    Implements incremental batch processing:
    - LLM generates tokens in batches
    - Each batch triggers Flow + HiFT
    - Audio chunks are yielded immediately
    """
    
    def __init__(
        self,
        engine: CosyVoiceEngine,
        config: Optional[StreamingConfig] = None
    ):
        """Initialize streaming engine.
        
        Args:
            engine: Base CosyVoice engine
            config: Streaming configuration
        """
        self.engine = engine
        self.config = config or StreamingConfig()
        self.logger = get_logger()
        
        self._cancel_flag = threading.Event()
        self._current_task_id: Optional[str] = None
    
    def cancel(self) -> None:
        """Cancel current streaming synthesis."""
        self._cancel_flag.set()
    
    def _is_cancelled(self) -> bool:
        """Check if synthesis was cancelled."""
        return self._cancel_flag.is_set()
    
    async def stream_inference(
        self,
        text: str,
        prompt_audio: np.ndarray,
        prompt_audio_sr: int,
        prompt_text: str,
        on_progress: Optional[Callable[[ProgressInfo], None]] = None,
        task_id: Optional[str] = None
    ) -> AsyncIterator[AudioChunk]:
        """Stream audio generation.
        
        Args:
            text: Text to synthesize
            prompt_audio: Reference audio array
            prompt_audio_sr: Reference audio sample rate
            prompt_text: Transcript of reference audio
            on_progress: Progress callback
            task_id: Task identifier for cancellation
            
        Yields:
            AudioChunk instances
        """
        self._cancel_flag.clear()
        self._current_task_id = task_id or f"stream_{int(time.time())}"
        
        loop = asyncio.get_event_loop()
        
        # Run synchronous generation in executor
        generator = self._stream_sync_generator(
            text, prompt_audio, prompt_audio_sr, prompt_text, on_progress
        )
        
        chunk_index = 0
        try:
            while True:
                if self._is_cancelled():
                    self.logger.info("Streaming cancelled")
                    break
                
                # Get next chunk from sync generator
                chunk = await loop.run_in_executor(
                    None, lambda: next(generator, None)
                )
                
                if chunk is None:
                    break
                
                chunk.chunk_index = chunk_index
                chunk_index += 1
                
                yield chunk
                
                if chunk.is_final:
                    break
                    
        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
            raise
    
    def _stream_sync_generator(
        self,
        text: str,
        prompt_audio: np.ndarray,
        prompt_audio_sr: int,
        prompt_text: str,
        on_progress: Optional[Callable[[ProgressInfo], None]] = None
    ):
        """Synchronous streaming generator.
        
        This runs in an executor thread.
        """
        engine = self.engine
        config = self.config
        
        start_time = time.time()
        self.logger.info(f"Streaming synthesis: '{text[:50]}...'")
        
        # ===== Phase 1: Prepare prompt (one-time) =====
        embedding = engine.extract_speaker_embedding(prompt_audio, prompt_audio_sr)
        prompt_tokens = engine.extract_speech_tokens(prompt_audio, prompt_audio_sr)
        prompt_mel = engine.extract_speech_mel(prompt_audio, prompt_audio_sr)
        
        prompt_prep_time = time.time() - start_time
        self.logger.debug(f"Prompt preparation: {prompt_prep_time:.2f}s")
        
        # ===== Phase 2: LLM token generation =====
        # Tokenize texts
        prompt_text_tokens = engine.tokenize_text(prompt_text)
        tts_text_tokens = engine.tokenize_text(text)
        
        prompt_text_len = prompt_text_tokens.shape[1]
        tts_text_len = tts_text_tokens.shape[1]
        
        # Combined tokens
        combined_text_tokens = np.concatenate([prompt_text_tokens, tts_text_tokens], axis=1)
        
        # Get embeddings
        text_emb = engine.get_text_embedding(combined_text_tokens)
        sos_emb = engine.get_speech_embedding(np.array([[engine.SOS_TOKEN]], dtype=np.int64))
        task_id_emb = engine.get_speech_embedding(np.array([[engine.TASK_ID]], dtype=np.int64))
        prompt_speech_emb = engine.get_speech_embedding(prompt_tokens)
        
        # Build initial input
        lm_input = np.concatenate([sos_emb, text_emb, task_id_emb, prompt_speech_emb], axis=1).astype(np.float32)
        seq_len = lm_input.shape[1]
        attention_mask = np.ones((1, seq_len), dtype=np.float32)
        
        # Initial forward pass
        llm_initial = engine.model_manager.get_session("llm_backbone_initial")
        initial_outputs = llm_initial.run(None, {
            'inputs_embeds': lm_input,
            'attention_mask': attention_mask
        })
        hidden_states = initial_outputs[0]
        past_key_values = initial_outputs[1] if len(initial_outputs) > 1 else np.zeros((48, 1, 2, seq_len, 64), dtype=np.float32)
        
        # Get initial logits
        llm_decoder = engine.model_manager.get_session("llm_decoder")
        logits = llm_decoder.run(None, {'hidden_state': hidden_states[:, -1:, :]})[0]
        
        # Token generation limits
        min_len = max(config.min_tokens, int(tts_text_len * 2))
        max_len = min(config.max_tokens, int(tts_text_len * 20))
        
        llm_decode = engine.model_manager.get_session("llm_backbone_decode")
        
        # ===== Phase 3: Incremental batch processing =====
        all_tokens = []
        processed_mel_len = 0
        chunk_tokens = []
        
        for i in range(max_len):
            if self._is_cancelled():
                break
            
            # Top-k sampling
            logp = engine._log_softmax(logits.squeeze())
            top_k_idx = np.argsort(logp)[-config.sampling_k:]
            top_k_logp = logp[top_k_idx]
            top_k_probs = engine._softmax(top_k_logp)
            sampled_idx = np.random.choice(len(top_k_idx), p=top_k_probs)
            token_id = top_k_idx[sampled_idx]
            
            # Check EOS
            if token_id == engine.EOS_TOKEN and i >= min_len:
                break
            
            all_tokens.append(token_id)
            chunk_tokens.append(token_id)
            
            # Check if chunk is ready
            if len(chunk_tokens) >= config.chunk_size_tokens:
                # Generate audio for this chunk
                chunk = self._process_token_chunk(
                    all_tokens,
                    prompt_tokens,
                    prompt_mel,
                    embedding,
                    processed_mel_len,
                    is_final=False,
                    on_progress=on_progress,
                    total_expected_tokens=max_len,
                    current_token=i
                )
                
                if chunk is not None:
                    processed_mel_len = chunk._mel_end
                    yield chunk
                
                chunk_tokens = []
            
            # Next token embedding
            next_emb = engine.get_speech_embedding(np.array([[token_id]], dtype=np.int64))
            
            # Update attention mask
            total_len = seq_len + len(all_tokens)
            attention_mask = np.ones((1, total_len), dtype=np.float32)
            
            # Decode step
            outputs = llm_decode.run(None, {
                'inputs_embeds': next_emb.astype(np.float32),
                'attention_mask': attention_mask,
                'past_key_values': past_key_values
            })
            hidden_states = outputs[0]
            past_key_values = outputs[1]
            
            # Next logits
            logits = llm_decoder.run(None, {'hidden_state': hidden_states})[0]
        
        # ===== Final chunk with remaining tokens =====
        if chunk_tokens or (all_tokens and processed_mel_len == 0):
            final_chunk = self._process_token_chunk(
                all_tokens,
                prompt_tokens,
                prompt_mel,
                embedding,
                processed_mel_len,
                is_final=True,
                on_progress=on_progress,
                total_expected_tokens=len(all_tokens),
                current_token=len(all_tokens)
            )
            
            if final_chunk is not None:
                yield final_chunk
        
        total_time = time.time() - start_time
        self.logger.info(f"Streaming complete: {len(all_tokens)} tokens in {total_time:.2f}s")
    
    def _process_token_chunk(
        self,
        all_tokens: list,
        prompt_tokens: np.ndarray,
        prompt_mel: np.ndarray,
        embedding: np.ndarray,
        processed_mel_len: int,
        is_final: bool,
        on_progress: Optional[Callable],
        total_expected_tokens: int,
        current_token: int
    ) -> Optional[AudioChunk]:
        """Process a chunk of tokens through Flow and HiFT.
        
        Args:
            all_tokens: All generated tokens so far
            prompt_tokens: Reference speech tokens
            prompt_mel: Reference mel spectrogram
            embedding: Speaker embedding
            processed_mel_len: Already processed mel frames
            is_final: Whether this is the final chunk
            on_progress: Progress callback
            total_expected_tokens: Estimated total tokens
            current_token: Current token index
            
        Returns:
            AudioChunk or None if no audio generated
        """
        from scipy.ndimage import zoom
        
        engine = self.engine
        config = self.config
        
        if not all_tokens:
            return None
        
        speech_tokens = np.array([all_tokens], dtype=np.int64)
        
        # Flow inference
        mel = engine.flow_inference(
            speech_tokens,
            embedding,
            prompt_tokens=prompt_tokens,
            prompt_mel=prompt_mel,
            n_timesteps=config.n_timesteps
        )
        
        # Extract only new mel frames
        current_mel_len = mel.shape[2]
        
        if current_mel_len <= processed_mel_len:
            return None
        
        new_mel = mel[:, :, processed_mel_len:]
        
        # HiFT vocoder
        audio = engine.hift_inference(new_mel)
        
        # Create AudioChunk
        audio_bytes = audio.astype(np.float32).tobytes()
        
        # Report progress
        if on_progress:
            progress_percent = min(100.0, (current_token / total_expected_tokens) * 100)
            on_progress(ProgressInfo(
                task_id=self._current_task_id or "unknown",
                total_chars=total_expected_tokens,
                processed_chars=current_token,
                progress_percent=progress_percent,
                estimated_remaining_ms=0
            ))
        
        chunk = AudioChunk(
            data=audio_bytes,
            sample_rate=engine.SAMPLE_RATE,
            channels=1,
            format='float32',
            is_final=is_final
        )
        
        # Store mel end position for next chunk
        chunk._mel_end = current_mel_len
        
        return chunk
