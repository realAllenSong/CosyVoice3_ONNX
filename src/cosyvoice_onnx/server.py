"""
Production-ready FastAPI server for CosyVoice3 ONNX TTS.

This server provides HTTP endpoints for:
- Basic text-to-speech
- Voice cloning
- Streaming output
- Preset voice management
- Audio validation

Usage:
    python run_server.py --host 127.0.0.1 --port 8000
"""

import asyncio
import base64
import io
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from .api import CosyVoiceTTS
from .config import CosyVoiceConfig
from .audio_validator import AudioValidator, AudioValidationResult
from .types import AudioData, PresetVoice
from .utils.logger import setup_logger, get_logger


# ============================================================================
# Pydantic Models
# ============================================================================

class TTSRequest(BaseModel):
    """Request model for basic TTS."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    preset: Optional[str] = Field(None, description="Preset voice name")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed")
    volume: float = Field(1.0, ge=0.0, le=2.0, description="Output volume")
    emotion: str = Field("neutral", description="Emotion tag")
    format: str = Field("wav", description="Output format: wav, mp3, or base64")
    language: str = Field("auto", description="Language hint: auto, zh, en, ja, ko, etc.")


class CloneRequest(BaseModel):
    """Request model for voice cloning (JSON body, audio via form)."""
    prompt_text: str = Field(..., min_length=1, description="Transcript of reference audio")
    target_text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    speed: float = Field(1.0, ge=0.5, le=2.0)
    volume: float = Field(1.0, ge=0.0, le=2.0)
    format: str = Field("wav", description="Output format")


class StreamRequest(BaseModel):
    """Request model for streaming synthesis."""
    text: str = Field(..., min_length=1, max_length=5000)
    preset: Optional[str] = None
    prompt_audio_base64: Optional[str] = Field(None, description="Base64 encoded reference audio")
    prompt_text: Optional[str] = None
    speed: float = Field(1.0, ge=0.5, le=2.0)
    volume: float = Field(1.0, ge=0.0, le=2.0)
    chunk_size_tokens: int = Field(30, ge=10, le=100)


class AudioValidationResponse(BaseModel):
    """Response model for audio validation."""
    valid: bool
    sample_rate: int
    duration_seconds: float
    channels: int
    format: str
    warnings: List[str]
    errors: List[str]


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: dict


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str


class PresetInfo(BaseModel):
    """Preset voice information."""
    name: str
    language: str
    gender: Optional[str] = None
    description: Optional[str] = None


class PresetsResponse(BaseModel):
    """Response for preset list."""
    presets: List[PresetInfo]


# ============================================================================
# Server State
# ============================================================================

class ServerState:
    """Global server state."""
    def __init__(self):
        self.tts: Optional[CosyVoiceTTS] = None
        self.validator: Optional[AudioValidator] = None
        self.config: Optional[dict] = None
        self.start_time: float = 0
        self.request_count: int = 0


state = ServerState()


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server startup and shutdown."""
    logger = get_logger()

    # Startup
    logger.info("Starting CosyVoice3 ONNX Server...")
    state.start_time = time.time()

    # Load configuration
    config_path = Path(__file__).parent.parent.parent / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                state.config = json.load(f)
            logger.info(f"Loaded config from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            state.config = {}
    else:
        state.config = {}

    # Initialize TTS engine (lazy loading, models loaded on first request)
    server_config = state.config.get("server", {})
    model_config = state.config.get("model", {})

    state.tts = CosyVoiceTTS(
        precision=model_config.get("precision", "fp32"),  # Will be forced to fp32
        preload=model_config.get("preload", False),
        num_threads=model_config.get("num_threads", 0),
        log_level=server_config.get("log_level", "INFO")
    )

    state.validator = AudioValidator(auto_resample=True)

    logger.info("Server initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down server...")
    if state.tts:
        state.tts.unload_models()
    logger.info("Server shutdown complete")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="CosyVoice3 ONNX TTS API",
    description="Production-ready TTS API with voice cloning and streaming support",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request Logging Middleware
# ============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    request_id = str(uuid.uuid4())[:8]
    logger = get_logger()

    start_time = time.time()
    logger.info(f"[{request_id}] {request.method} {request.url.path}")

    state.request_count += 1

    try:
        response = await call_next(request)
        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] Completed in {elapsed:.3f}s - Status: {response.status_code}")
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Failed after {elapsed:.3f}s - Error: {e}")
        raise


# ============================================================================
# Helper Functions
# ============================================================================

def create_error_response(code: str, message: str, details: dict = None) -> dict:
    """Create standardized error response."""
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {}
        }
    }


def audio_to_response(audio: AudioData, format: str, filename: str = "output") -> Response:
    """Convert AudioData to appropriate HTTP response."""
    if format == "base64":
        # Return JSON with base64 encoded audio
        audio_b64 = base64.b64encode(audio.data).decode("utf-8")
        return Response(
            content=json.dumps({
                "audio": audio_b64,
                "sample_rate": audio.sample_rate,
                "duration_ms": audio.duration_ms,
                "format": "wav"
            }),
            media_type="application/json"
        )
    elif format == "mp3":
        # Convert to MP3 if pydub is available
        try:
            from pydub import AudioSegment
            audio_segment = AudioSegment(
                data=audio.data,
                sample_width=4,  # float32
                frame_rate=audio.sample_rate,
                channels=1
            )
            mp3_buffer = io.BytesIO()
            audio_segment.export(mp3_buffer, format="mp3")
            return Response(
                content=mp3_buffer.getvalue(),
                media_type="audio/mpeg",
                headers={"Content-Disposition": f'attachment; filename="{filename}.mp3"'}
            )
        except ImportError:
            # Fall back to WAV if pydub not available
            pass

    # Default: return WAV
    wav_buffer = io.BytesIO()
    import soundfile as sf
    audio_array = np.frombuffer(audio.data, dtype=np.float32)
    sf.write(wav_buffer, audio_array, audio.sample_rate, format="WAV")
    wav_buffer.seek(0)

    return Response(
        content=wav_buffer.getvalue(),
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{filename}.wav"'}
    )


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=state.tts.is_loaded if state.tts else False,
        version="1.0.0"
    )


@app.get("/presets", response_model=PresetsResponse)
async def list_presets():
    """List available preset voices."""
    if not state.tts:
        raise HTTPException(status_code=503, detail="TTS not initialized")

    try:
        preset_names = state.tts.list_presets()
        presets = []

        for name in preset_names:
            try:
                preset = state.tts.load_preset(name)
                presets.append(PresetInfo(
                    name=preset.name,
                    language=preset.language,
                    gender=preset.gender,
                    description=preset.description
                ))
            except Exception:
                # Include name even if full info not available
                presets.append(PresetInfo(name=name, language="unknown"))

        return PresetsResponse(presets=presets)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=create_error_response("INTERNAL_ERROR", str(e))
        )


@app.post("/tts")
async def synthesize_tts(request: TTSRequest):
    """Basic text-to-speech synthesis.

    Returns audio in requested format (wav, mp3, or base64).
    """
    if not state.tts:
        raise HTTPException(status_code=503, detail="TTS not initialized")

    logger = get_logger()

    try:
        if request.preset:
            # Use preset voice
            try:
                preset = state.tts.load_preset(request.preset)
                audio = state.tts.synthesize_with_preset(
                    text=request.text,
                    preset=preset,
                    speed=request.speed,
                    volume=request.volume,
                    output_format="wav"
                )
            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=create_error_response("PRESET_NOT_FOUND", str(e))
                )
        else:
            # Need a preset or reference audio
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "INVALID_REQUEST",
                    "Either 'preset' must be provided, or use /clone endpoint with reference audio"
                )
            )

        return audio_to_response(audio, request.format)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response("MODEL_ERROR", str(e))
        )


@app.post("/clone")
async def clone_voice(
    prompt_audio: UploadFile = File(..., description="Reference audio file"),
    prompt_text: str = Form(..., description="Transcript of reference audio"),
    target_text: str = Form(..., description="Text to synthesize"),
    speed: float = Form(1.0, ge=0.5, le=2.0),
    volume: float = Form(1.0, ge=0.0, le=2.0),
    format: str = Form("wav", description="Output format: wav, mp3, or base64")
):
    """Voice cloning endpoint.

    Upload a reference audio file and its transcript, then synthesize
    new text in that voice.
    """
    if not state.tts:
        raise HTTPException(status_code=503, detail="TTS not initialized")

    logger = get_logger()

    try:
        # Read and validate uploaded audio
        audio_bytes = await prompt_audio.read()

        validation = state.validator.validate(audio_bytes)
        if not validation.valid:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "INVALID_AUDIO",
                    "Reference audio validation failed",
                    {"errors": validation.errors, "warnings": validation.warnings}
                )
            )

        # Log warnings
        if validation.warnings:
            logger.warning(f"Audio validation warnings: {validation.warnings}")

        # Perform voice cloning
        audio = state.tts.clone_voice(
            prompt_audio=validation.processed_audio,
            prompt_text=prompt_text,
            target_text=target_text,
            speed=speed,
            volume=volume,
            output_format="wav"
        )

        return audio_to_response(audio, format)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice cloning failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response("MODEL_ERROR", str(e))
        )


@app.post("/stream")
async def stream_synthesis(request: StreamRequest):
    """Streaming synthesis endpoint using Server-Sent Events (SSE).

    Returns audio chunks as they are generated for real-time playback.
    """
    if not state.tts:
        raise HTTPException(status_code=503, detail="TTS not initialized")

    logger = get_logger()

    # Prepare prompt audio
    prompt_audio = None
    prompt_text = request.prompt_text

    if request.prompt_audio_base64:
        try:
            prompt_audio = base64.b64decode(request.prompt_audio_base64)
            validation = state.validator.validate(prompt_audio)
            if not validation.valid:
                raise HTTPException(
                    status_code=400,
                    detail=create_error_response(
                        "INVALID_AUDIO",
                        "Reference audio validation failed",
                        {"errors": validation.errors}
                    )
                )
            prompt_audio = validation.processed_audio
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=create_error_response("INVALID_AUDIO", f"Failed to decode audio: {e}")
            )
    elif request.preset:
        try:
            preset = state.tts.load_preset(request.preset)
            import librosa
            prompt_audio, _ = librosa.load(preset.audio_path, sr=24000)
            prompt_text = preset.transcript
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=create_error_response("PRESET_NOT_FOUND", str(e))
            )
    else:
        raise HTTPException(
            status_code=400,
            detail=create_error_response(
                "INVALID_REQUEST",
                "Either 'preset' or 'prompt_audio_base64' (with 'prompt_text') is required"
            )
        )

    async def event_generator():
        """Generate SSE events for audio chunks."""
        chunk_index = 0
        total_duration_ms = 0

        try:
            async for chunk in state.tts.synthesize_stream(
                text=request.text,
                prompt_audio=prompt_audio,
                prompt_text=prompt_text,
                speed=request.speed,
                volume=request.volume,
                chunk_size_tokens=request.chunk_size_tokens
            ):
                chunk_b64 = base64.b64encode(chunk.data).decode("utf-8")
                duration_ms = len(chunk.data) // 4 * 1000 // 24000  # float32, 24kHz

                yield {
                    "event": "audio_chunk",
                    "data": json.dumps({
                        "chunk": chunk_b64,
                        "index": chunk_index,
                        "is_final": chunk.is_final,
                        "duration_ms": duration_ms
                    })
                }

                chunk_index += 1
                total_duration_ms += duration_ms

            # Send completion event
            yield {
                "event": "done",
                "data": json.dumps({
                    "total_chunks": chunk_index,
                    "duration_ms": total_duration_ms
                })
            }

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }

    return EventSourceResponse(event_generator())


@app.post("/validate_audio", response_model=AudioValidationResponse)
async def validate_audio(
    audio: UploadFile = File(..., description="Audio file to validate")
):
    """Validate reference audio before using it for voice cloning.

    Returns validation result including sample rate, duration, and any
    warnings about the audio quality.
    """
    if not state.validator:
        raise HTTPException(status_code=503, detail="Validator not initialized")

    try:
        audio_bytes = await audio.read()
        result = state.validator.validate(audio_bytes)

        return AudioValidationResponse(
            valid=result.valid,
            sample_rate=result.sample_rate,
            duration_seconds=result.duration_seconds,
            channels=result.channels,
            format=result.format,
            warnings=result.warnings,
            errors=result.errors
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=create_error_response("INTERNAL_ERROR", str(e))
        )


@app.get("/stats")
async def get_stats():
    """Get server statistics."""
    uptime = time.time() - state.start_time if state.start_time else 0

    return {
        "uptime_seconds": uptime,
        "request_count": state.request_count,
        "model_loaded": state.tts.is_loaded if state.tts else False
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Run the server using uvicorn."""
    import uvicorn
    uvicorn.run(
        "cosyvoice_onnx.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
