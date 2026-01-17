# Notes: CosyVoice3 ONNX Project Analysis

## Interview Summary

### User Requirements
- **Platform**: Mac M3 Max, 16-32GB RAM, CPU only (no MPS)
- **Goal**: Local FastAPI HTTP API service for TTS
- **Future**: Integration with PyQt5 desktop pet via HTTP requests

### Current Problems (User Experienced)
1. **乱读/胡说** - Random/nonsense speech output
2. **前后空白很长** - Long silence before/after audio
3. **音质差/电流声** - Poor quality, static/crackling sounds

### API Requirements
- Basic TTS
- Voice cloning (zero-shot)
- Streaming output
- Multi-language support (zh, en, ja, ko, etc.)
- Multiple response formats (WAV/MP3/Base64)
- Detailed logging
- Production-ready stability

### Desktop Pet Integration
- HTTP requests (not WebSocket)
- Interactive dialogue
- Emotion expression
- Multiple voice characters
- Both presets and custom voices

### Technical Constraints
- 16-32GB RAM → Need lazy loading strategy
- CPU only → Optimize for Apple Silicon CPU
- Fix quality issues BEFORE building service

---

## Root Cause Analysis

### Problem 1: Random/Nonsense Speech
**Root Cause**: Missing `You are a helpful assistant.<|endofprompt|>` prefix

**Evidence**:
- User's note explicitly states this is required for CosyVoice3
- Current ONNX codebase (`engine.py`) does NOT implement this prefix
- The LLM expects this system prompt format

**Location**: [engine.py](src/cosyvoice_onnx/engine.py) - `_build_llm_inputs()` method

### Problem 2: Long Silence Before/After
**Root Cause**: FP16 precision issues

**Evidence**:
- User's note: "禁用fp16 (fp16=False)，否则可能导致结果出现前后较长空白"
- Current code defaults to `precision="auto"` which may select FP16

**Location**: [model_manager.py](src/cosyvoice_onnx/model_manager.py) - `_detect_precision()`

### Problem 3: Poor Audio Quality / Static
**Root Cause**: Reference audio sampling rate mismatch

**Evidence**:
- User's note: "参考音频采样率使用24KHz"
- User's reference audio is downloaded (unknown sampling rate)
- Current code resamples to 16kHz for speaker embedding, 24kHz for mel

**Location**: [engine.py](src/cosyvoice_onnx/engine.py) - `_extract_speaker_embedding()`, `_extract_prompt_mel()`

---

## Code Analysis

### Critical Files to Modify

1. **[engine.py](src/cosyvoice_onnx/engine.py)**
   - Add `<|endofprompt|>` prefix handling
   - Lines 200-250: `_build_llm_inputs()` needs modification

2. **[api.py](src/cosyvoice_onnx/api.py)**
   - Force FP32 precision for stability
   - Add endofprompt prefix parameter

3. **[config.py](src/cosyvoice_onnx/config.py)**
   - Add configuration for system prompt prefix
   - Add strict sampling rate validation

### Missing Features (Need Implementation)

1. **FastAPI Server** - Only basic template exists in `examples/server_example.py`
   - Need production-ready implementation
   - Endpoints: /tts, /clone, /stream, /presets, /health

2. **Audio Validation**
   - Validate reference audio is 24kHz (or auto-resample with warning)
   - Reject/warn for very short (<1s) or very long (>30s) reference audio

3. **Error Handling**
   - Detailed error messages for common issues
   - Automatic retry for transient failures

---

## Implementation Priorities

### Phase 1: Quality Fix (Critical)
1. Add `You are a helpful assistant.<|endofprompt|>` prefix to prompt_text
2. Force FP32 precision (disable FP16 auto-detection)
3. Add reference audio validation (24kHz check/resample)

### Phase 2: FastAPI Server
1. Create production-ready FastAPI server
2. Implement all endpoints with streaming support
3. Add comprehensive logging

### Phase 3: Testing & Validation
1. Test scripts for quality verification
2. API endpoint tests
3. Performance benchmarks

---

## Technical Notes

### ONNX Model Files Location
Default: `~/.cosyvoice3/models/`
Source: `ayousanz/cosy-voice3-onnx` on HuggingFace

### Audio Processing Pipeline
```
Reference Audio (any SR)
  → Resample to 16kHz (speaker embedding extraction)
  → Resample to 24kHz (prompt mel extraction)

Text Input
  → Add system prompt prefix
  → Qwen2 tokenization
  → Text embedding

LLM Inference
  → Speech token generation (autoregressive)

Flow + HiFT
  → Mel spectrogram → Waveform (24kHz output)
```

### Memory Estimation
- Full model preload: ~8-10GB
- Lazy loading: ~3-4GB baseline, load models on demand
- Recommended: Lazy loading for 16-32GB systems

### Performance on M3 Max (CPU)
- Expected RTF: 0.3-0.5x (3-5 seconds for 10 seconds of speech)
- Streaming reduces perceived latency significantly
