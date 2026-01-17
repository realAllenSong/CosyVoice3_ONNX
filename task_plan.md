# Task Plan: CosyVoice3 FastAPI HTTP Service Setup

## Goal
Deploy CosyVoice3 0.5B as a local FastAPI HTTP API service on Mac M3 Max (CPU only), with future PyQt5 desktop pet integration.

## Phases
- [x] Phase 1: Requirements Clarification (Interview)
- [x] Phase 2: Research & Code Analysis
- [x] Phase 3: Implementation Planning
- [x] Phase 4: Spec Documentation
- [ ] Phase 5: Final Plan Delivery

## Key Questions (Answered)
1. ~~ONNX vs PyTorch~~ → Continue with ONNX version
2. ~~Is the "endofprompt" prefix needed~~ → YES, must add it
3. ~~What API endpoints needed~~ → /health, /presets, /tts, /clone, /stream, /validate_audio
4. ~~Error handling requirements~~ → Production-ready with detailed logging

## Interview Results

### User Requirements
- **Platform**: Mac M3 Max, 16-32GB RAM, CPU only
- **Current Issue**: Ran existing code, got poor quality (乱读, 空白, 电流声)
- **API Needs**: Full features (TTS + Clone + Streaming)
- **Response Format**: WAV/MP3/Base64 all supported
- **Language**: Multi-language (zh, en, ja, ko, de, fr, it, es, ru)
- **Performance**: Balance speed and quality, streaming output
- **Robustness**: Production-ready with detailed logging
- **Testing**: Provide test scripts

### Desktop Pet Integration
- Call method: HTTP requests (not WebSocket)
- Features: Interactive dialogue, emotion expression, multiple voice characters
- Voice management: Both presets and custom voices

### Technical Constraints
- Memory: 16-32GB → Must use lazy loading
- Fix strategy: Fix quality FIRST, then build service
- Project scope: Production-ready

## Root Cause Analysis

| Problem | Root Cause | Fix |
|---------|------------|-----|
| 乱读/胡说 | Missing `<\|endofprompt\|>` prefix | Modify engine.py |
| 前后空白 | FP16 precision issue | Force FP32 |
| 音质差 | Reference audio SR mismatch | Auto-resample to 24kHz |

## Deliverables Created
- [SPEC.md](SPEC.md) - Complete specification document
- [notes.md](notes.md) - Analysis and research notes
- Implementation plan (in plan file)

## Errors Encountered
- (None)

## Status
**COMPLETED** - All implementation tasks finished

## Implementation Summary

### Files Modified
- [src/cosyvoice_onnx/engine.py](src/cosyvoice_onnx/engine.py) - Added `SYSTEM_PROMPT_PREFIX`
- [src/cosyvoice_onnx/api.py](src/cosyvoice_onnx/api.py) - Forced FP32 precision
- [src/cosyvoice_onnx/__init__.py](src/cosyvoice_onnx/__init__.py) - Version bump, new exports
- [config.json](config.json) - Added server/audio config sections
- [pyproject.toml](pyproject.toml) - Added server dependencies

### Files Created
- [src/cosyvoice_onnx/audio_validator.py](src/cosyvoice_onnx/audio_validator.py) - Audio validation
- [src/cosyvoice_onnx/server.py](src/cosyvoice_onnx/server.py) - FastAPI server
- [run_server.py](run_server.py) - Server launcher
- [tests/test_quality.py](tests/test_quality.py) - Quality tests
- [tests/test_api.py](tests/test_api.py) - API endpoint tests
- [tests/test_streaming.py](tests/test_streaming.py) - Streaming tests

### Next Steps
1. Install server dependencies: `pip install -e ".[server]"`
2. Run quality tests: `python tests/test_quality.py`
3. Start server: `python run_server.py`
4. Run API tests: `python tests/test_api.py`
