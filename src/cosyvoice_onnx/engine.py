"""
Core inference engine for CosyVoice3 ONNX.
Based on ayousanz/cosy-voice3-onnx implementation.
"""

import time
from typing import Optional, Tuple
import numpy as np

from .model_manager import ModelManager
from .config import CosyVoiceConfig
from .utils.logger import get_logger


class CosyVoiceEngine:
    """Core TTS inference engine using ONNX models.
    
    This class handles the low-level inference pipeline:
    1. LLM inference - generates speech tokens from text
    2. Flow inference - converts speech tokens to mel spectrograms
    3. HiFT inference - converts mel to audio waveform
    """
    
    # Model parameters for CosyVoice3
    HIDDEN_DIM = 896
    SPEECH_TOKEN_SIZE = 6561
    SOS_TOKEN = 6561  # Start of sequence
    EOS_TOKEN = 6562  # End of sequence
    TASK_ID = 6563    # Task identifier
    SAMPLE_RATE = 24000
    
    def __init__(self, model_manager: ModelManager, config: CosyVoiceConfig):
        """Initialize the inference engine.
        
        Args:
            model_manager: Loaded model manager
            config: Configuration
        """
        self.model_manager = model_manager
        self.config = config
        self.logger = get_logger()
    
    def tokenize_text(self, text: str) -> np.ndarray:
        """Tokenize text using Qwen2 tokenizer.
        
        Args:
            text: Input text
            
        Returns:
            Token IDs as numpy array [1, seq_len]
        """
        tokens = self.model_manager.tokenizer.encode(text, add_special_tokens=False)
        return np.array([tokens], dtype=np.int64)
    
    def get_text_embedding(self, token_ids: np.ndarray) -> np.ndarray:
        """Get text embeddings.
        
        Args:
            token_ids: Token IDs [1, seq_len]
            
        Returns:
            Text embeddings [1, seq_len, hidden_dim]
        """
        session = self.model_manager.get_session("text_embedding")
        return session.run(None, {'input_ids': token_ids})[0]
    
    def get_speech_embedding(self, token_ids: np.ndarray) -> np.ndarray:
        """Get speech token embeddings.
        
        Args:
            token_ids: Speech token IDs [1, seq_len]
            
        Returns:
            Speech embeddings [1, seq_len, hidden_dim]
        """
        session = self.model_manager.get_session("llm_speech_embedding")
        return session.run(None, {'token': token_ids})[0]
    
    def extract_speaker_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract speaker embedding from audio using CAMPPlus.
        
        Args:
            audio: Audio waveform (mono, float32)
            sample_rate: Sample rate (should be 16000)
            
        Returns:
            Speaker embedding [1, 192]
        """
        import librosa
        
        # Ensure 16kHz
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        
        audio = audio.astype(np.float32)
        
        # Compute Kaldi-style fbank features
        mel = librosa.feature.melspectrogram(
            y=audio, sr=16000, n_fft=400, hop_length=160,
            n_mels=80, fmin=20, fmax=7600
        )
        log_mel = np.log(np.maximum(mel, 1e-10))
        log_mel = log_mel.T  # [frames, 80]
        
        # Mean normalization
        log_mel = log_mel - log_mel.mean(axis=0, keepdims=True)
        
        # [1, frames, 80]
        feat = log_mel[np.newaxis, :, :].astype(np.float32)
        
        # Run CAMPPlus
        session = self.model_manager.get_session("campplus")
        input_name = session.get_inputs()[0].name
        embedding = session.run(None, {input_name: feat})[0]
        
        # Flatten to [1, 192]
        embedding = embedding.flatten()[np.newaxis, :]
        return embedding.astype(np.float32)
    
    def extract_speech_tokens(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract speech tokens from audio.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            
        Returns:
            Speech tokens [1, seq_len]
        """
        import librosa
        
        # Ensure 16kHz
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        
        audio = audio.astype(np.float32)
        
        # Whisper-style log mel (128 mels)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=16000, n_fft=400, hop_length=160,
            n_mels=128, fmin=0, fmax=8000
        )
        log_mel = np.log10(np.maximum(mel, 1e-10))
        log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0
        
        # [1, 128, frames]
        feat = log_mel[np.newaxis, :, :].astype(np.float32)
        feat_len = np.array([feat.shape[2]], dtype=np.int32)
        
        # Run speech tokenizer
        session = self.model_manager.get_session("speech_tokenizer")
        input_names = [inp.name for inp in session.get_inputs()]
        speech_token = session.run(None, {
            input_names[0]: feat,
            input_names[1]: feat_len
        })[0]
        
        # [1, seq_len]
        speech_token = speech_token.flatten()[np.newaxis, :]
        return speech_token.astype(np.int64)
    
    def extract_speech_mel(self, audio: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
        """Extract mel spectrogram for flow conditioning.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            
        Returns:
            Mel features [1, frames, 80]
        """
        import librosa
        
        # Ensure 24kHz (CosyVoice native rate)
        if sample_rate != 24000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=24000)
        
        audio = audio.astype(np.float32)
        
        mel = librosa.feature.melspectrogram(
            y=audio, sr=24000, n_fft=1024, hop_length=256,
            n_mels=80, fmin=0, fmax=12000
        )
        log_mel = np.log(np.maximum(mel, 1e-10))
        
        # [1, frames, 80]
        mel_feat = log_mel.T[np.newaxis, :, :].astype(np.float32)
        return mel_feat
    
    def llm_inference(
        self,
        text: str,
        prompt_text: str,
        prompt_speech_tokens: np.ndarray,
        sampling_k: int = 25,
        max_len: int = 500,
        min_len: int = 10
    ) -> np.ndarray:
        """Generate speech tokens using LLM (zero-shot mode).
        
        Args:
            text: Text to synthesize
            prompt_text: Transcript of prompt audio
            prompt_speech_tokens: Speech tokens from prompt [1, seq_len]
            sampling_k: Top-k sampling parameter
            max_len: Maximum output length
            min_len: Minimum output length
            
        Returns:
            Generated speech tokens [1, seq_len]
        """
        self.logger.debug(f"LLM inference: text='{text[:50]}...'")
        
        # Tokenize texts
        prompt_text_tokens = self.tokenize_text(prompt_text)
        tts_text_tokens = self.tokenize_text(text)
        
        prompt_text_len = prompt_text_tokens.shape[1]
        tts_text_len = tts_text_tokens.shape[1]
        
        # Concatenate tokens (zero-shot mode)
        combined_text_tokens = np.concatenate([prompt_text_tokens, tts_text_tokens], axis=1)
        
        # Get embeddings
        text_emb = self.get_text_embedding(combined_text_tokens)
        sos_emb = self.get_speech_embedding(np.array([[self.SOS_TOKEN]], dtype=np.int64))
        task_id_emb = self.get_speech_embedding(np.array([[self.TASK_ID]], dtype=np.int64))
        
        if prompt_speech_tokens is not None and prompt_speech_tokens.shape[1] > 0:
            prompt_speech_emb = self.get_speech_embedding(prompt_speech_tokens)
        else:
            prompt_speech_emb = np.zeros((1, 0, self.HIDDEN_DIM), dtype=np.float32)
        
        # Build initial input: [SOS, text_emb, TASK_ID, prompt_speech_emb]
        lm_input = np.concatenate([sos_emb, text_emb, task_id_emb, prompt_speech_emb], axis=1).astype(np.float32)
        
        # Initial forward pass
        seq_len = lm_input.shape[1]
        attention_mask = np.ones((1, seq_len), dtype=np.float32)
        
        llm_initial = self.model_manager.get_session("llm_backbone_initial")
        initial_outputs = llm_initial.run(None, {
            'inputs_embeds': lm_input, 
            'attention_mask': attention_mask
        })
        hidden_states = initial_outputs[0]
        
        # Get KV cache
        if len(initial_outputs) > 1:
            past_key_values = initial_outputs[1]
        else:
            past_key_values = np.zeros((48, 1, 2, seq_len, 64), dtype=np.float32)
        
        # Get initial logits
        llm_decoder = self.model_manager.get_session("llm_decoder")
        logits = llm_decoder.run(None, {'hidden_state': hidden_states[:, -1:, :]})[0]
        
        # Calculate dynamic length limits
        min_len = max(min_len, int(tts_text_len * 2))
        max_len = min(max_len, int(tts_text_len * 20))
        
        # Autoregressive generation
        llm_decode = self.model_manager.get_session("llm_backbone_decode")
        out_tokens = []
        
        for i in range(max_len):
            # Top-k sampling
            logp = self._log_softmax(logits.squeeze())
            top_k_idx = np.argsort(logp)[-sampling_k:]
            top_k_logp = logp[top_k_idx]
            top_k_probs = self._softmax(top_k_logp)
            sampled_idx = np.random.choice(len(top_k_idx), p=top_k_probs)
            token_id = top_k_idx[sampled_idx]
            
            # Check EOS
            if token_id == self.EOS_TOKEN and i >= min_len:
                break
            
            out_tokens.append(token_id)
            
            # Get next token embedding
            next_emb = self.get_speech_embedding(np.array([[token_id]], dtype=np.int64))
            
            # Update attention mask
            total_len = seq_len + len(out_tokens)
            attention_mask = np.ones((1, total_len), dtype=np.float32)
            
            # Decode step
            outputs = llm_decode.run(None, {
                'inputs_embeds': next_emb.astype(np.float32),
                'attention_mask': attention_mask,
                'past_key_values': past_key_values
            })
            hidden_states = outputs[0]
            past_key_values = outputs[1]
            
            # Get next logits
            logits = llm_decoder.run(None, {'hidden_state': hidden_states})[0]
        
        self.logger.debug(f"Generated {len(out_tokens)} speech tokens")
        return np.array([out_tokens], dtype=np.int64)
    
    def flow_inference(
        self,
        speech_tokens: np.ndarray,
        embedding: np.ndarray,
        prompt_tokens: Optional[np.ndarray] = None,
        prompt_mel: Optional[np.ndarray] = None,
        n_timesteps: int = 10
    ) -> np.ndarray:
        """Convert speech tokens to mel using Flow model.
        
        Args:
            speech_tokens: Generated speech tokens [1, seq_len]
            embedding: Speaker embedding [1, 192]
            prompt_tokens: Prompt speech tokens (optional)
            prompt_mel: Prompt mel features (optional)
            n_timesteps: Number of flow steps
            
        Returns:
            Mel spectrogram [1, 80, mel_len]
        """
        from scipy.ndimage import zoom
        
        # Normalize and project speaker embedding
        embedding_norm = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8)
        flow_speaker = self.model_manager.get_session("flow_speaker_projection")
        spks = flow_speaker.run(None, {'embedding': embedding_norm.astype(np.float32)})[0]
        
        # Concatenate prompt and generated tokens
        if prompt_tokens is not None and prompt_tokens.shape[1] > 0:
            all_tokens = np.concatenate([prompt_tokens, speech_tokens], axis=1)
            prompt_token_len = prompt_tokens.shape[1]
        else:
            all_tokens = speech_tokens
            prompt_token_len = 0
        
        # Embed tokens
        flow_token = self.model_manager.get_session("flow_token_embedding")
        token_embedded = flow_token.run(None, {'token': all_tokens.astype(np.int64)})[0]
        
        # Pre-lookahead (includes repeat_interleave with token_mel_ratio=2)
        flow_pre = self.model_manager.get_session("flow_pre_lookahead")
        h = flow_pre.run(None, {'token_embedded': token_embedded.astype(np.float32)})[0]
        
        token_mel_ratio = 2
        mel_len = h.shape[1]
        
        if prompt_tokens is not None and prompt_token_len > 0:
            mel_len1 = prompt_token_len * token_mel_ratio
            mel_len2 = mel_len - mel_len1
        else:
            mel_len1 = 0
            mel_len2 = mel_len
        
        # Build conditions
        conds = np.zeros((1, 80, mel_len), dtype=np.float32)
        if prompt_mel is not None and prompt_mel.shape[1] > 0 and mel_len1 > 0:
            prompt_mel_t = prompt_mel.transpose(0, 2, 1)
            src_len = prompt_mel_t.shape[2]
            if src_len != mel_len1:
                zoom_factor = mel_len1 / src_len
                prompt_mel_resized = zoom(prompt_mel_t, (1, 1, zoom_factor), order=1)
                conds[:, :, :mel_len1] = prompt_mel_resized[:, :, :mel_len1]
            else:
                conds[:, :, :mel_len1] = prompt_mel_t[:, :, :mel_len1]
        
        # Prepare inputs
        mu = h.transpose(0, 2, 1)
        mask = np.ones((1, 1, mel_len), dtype=np.float32)
        x = np.random.randn(1, 80, mel_len).astype(np.float32)
        
        # Batch for estimator
        x_batch = np.concatenate([x, x], axis=0)
        mask_batch = np.concatenate([mask, mask], axis=0)
        mu_batch = np.concatenate([mu, mu], axis=0)
        spks_batch = np.concatenate([spks, spks], axis=0)
        conds_batch = np.concatenate([conds, conds], axis=0)
        
        # Euler solver
        flow_estimator = self.model_manager.get_session("flow_estimator")
        for step in range(n_timesteps):
            t = np.array([step / n_timesteps, step / n_timesteps], dtype=np.float32)
            
            velocity = flow_estimator.run(None, {
                'x': x_batch,
                'mask': mask_batch,
                'mu': mu_batch,
                't': t,
                'spks': spks_batch,
                'cond': conds_batch
            })[0]
            
            dt = 1.0 / n_timesteps
            x_batch = x_batch + velocity * dt
        
        # Extract generated portion
        mel = x_batch[:1]
        if mel_len1 > 0:
            mel = mel[:, :, mel_len1:]
        
        return mel
    
    def hift_inference(self, mel: np.ndarray) -> np.ndarray:
        """Convert mel spectrogram to audio using HiFT vocoder.
        
        Args:
            mel: Mel spectrogram [1, 80, mel_len]
            
        Returns:
            Audio waveform (float32)
        """
        from scipy.signal import get_window
        
        # Predict F0
        hift_f0 = self.model_manager.get_session("hift_f0_predictor")
        f0 = hift_f0.run(None, {'mel': mel.astype(np.float32)})[0]
        
        # Generate source
        f0_input = f0[:, np.newaxis, :]
        hift_source = self.model_manager.get_session("hift_source_generator")
        source = hift_source.run(None, {'f0': f0_input.astype(np.float32)})[0]
        
        # Compute STFT
        source_squeezed = source.squeeze()
        stft_real, stft_imag = self._stft(source_squeezed, n_fft=16, hop_len=4, center=True)
        source_stft = np.concatenate([stft_real, stft_imag], axis=0)
        source_stft = source_stft[np.newaxis, :, :]
        
        # HiFT decode
        hift_decoder = self.model_manager.get_session("hift_decoder")
        outputs = hift_decoder.run(None, {
            'mel': mel.astype(np.float32),
            'source_stft': source_stft.astype(np.float32)
        })
        magnitude = outputs[0]
        phase = outputs[1]
        
        # ISTFT
        audio = self._istft(magnitude.squeeze(0), phase.squeeze(0), n_fft=16, hop_len=4)
        
        # Clip
        audio = np.clip(audio, -0.99, 0.99)
        
        return audio
    
    def _stft(self, x: np.ndarray, n_fft: int = 16, hop_len: int = 4, center: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Compute STFT."""
        from scipy.signal import get_window
        
        window = get_window("hann", n_fft, fftbins=True).astype(np.float32)
        x = x.astype(np.float32)
        
        if center:
            pad_len = n_fft // 2
            x = np.pad(x, (pad_len, pad_len), mode='reflect')
        
        n_frames = 1 + (len(x) - n_fft) // hop_len
        n_freqs = n_fft // 2 + 1
        
        real = np.zeros((n_freqs, n_frames), dtype=np.float32)
        imag = np.zeros((n_freqs, n_frames), dtype=np.float32)
        
        for i in range(n_frames):
            start = i * hop_len
            frame = x[start:start + n_fft] * window
            spectrum = np.fft.rfft(frame)
            real[:, i] = np.real(spectrum)
            imag[:, i] = np.imag(spectrum)
        
        return real, imag
    
    def _istft(self, magnitude: np.ndarray, phase: np.ndarray, n_fft: int = 16, hop_len: int = 4) -> np.ndarray:
        """Compute ISTFT."""
        from scipy.signal import get_window
        
        window = get_window("hann", n_fft, fftbins=True).astype(np.float32)
        magnitude = np.clip(magnitude, a_min=None, a_max=100.0)
        
        complex_spec = magnitude * np.exp(1j * phase)
        n_frames = complex_spec.shape[1]
        output_length = n_fft + (n_frames - 1) * hop_len
        
        audio = np.zeros(output_length, dtype=np.float32)
        window_sum = np.zeros(output_length, dtype=np.float32)
        
        for i in range(n_frames):
            start = i * hop_len
            frame_spec = complex_spec[:, i]
            frame = np.fft.irfft(frame_spec, n=n_fft).astype(np.float32)
            audio[start:start + n_fft] += frame * window
            window_sum[start:start + n_fft] += window ** 2
        
        window_sum = np.maximum(window_sum, 1e-8)
        audio = audio / window_sum
        
        return audio.astype(np.float32)
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    @staticmethod
    def _log_softmax(x: np.ndarray) -> np.ndarray:
        return x - np.log(np.sum(np.exp(x - np.max(x)))) - np.max(x)
