#!/usr/bin/env python3
"""
CosyVoice3 ONNX TTS Server Launcher

A simple CLI to start the FastAPI TTS server.

Usage:
    python run_server.py                          # Default: localhost:8000
    python run_server.py --port 8080              # Custom port
    python run_server.py --host 0.0.0.0           # Allow external connections
    python run_server.py --reload                 # Development mode with auto-reload

Examples:
    # Local development
    python run_server.py --reload

    # Production (local only)
    python run_server.py --host 127.0.0.1 --port 8000

    # LAN access (use with caution)
    python run_server.py --host 0.0.0.0 --port 8000
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="CosyVoice3 ONNX TTS Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_server.py                    Start server on localhost:8000
  python run_server.py --port 8080        Use custom port
  python run_server.py --reload           Enable auto-reload for development
  python run_server.py --host 0.0.0.0     Allow external connections (LAN)

API Endpoints:
  GET  /health           Health check
  GET  /presets          List available preset voices
  POST /tts              Basic text-to-speech
  POST /clone            Voice cloning with reference audio
  POST /stream           Streaming synthesis (SSE)
  POST /validate_audio   Validate reference audio
  GET  /stats            Server statistics
        """
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1, recommended for TTS)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)"
    )

    args = parser.parse_args()

    # Check dependencies
    try:
        import uvicorn
        import fastapi
        import sse_starlette
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("\nPlease install required packages:")
        print("  pip install fastapi uvicorn sse-starlette python-multipart")
        sys.exit(1)

    # Print startup info
    print("=" * 60)
    print("CosyVoice3 ONNX TTS Server")
    print("=" * 60)
    print(f"  Host:     {args.host}")
    print(f"  Port:     {args.port}")
    print(f"  Workers:  {args.workers}")
    print(f"  Reload:   {args.reload}")
    print(f"  Log:      {args.log_level}")
    print()

    if args.host == "0.0.0.0":
        print("⚠️  WARNING: Server is accessible from external network!")
        print("   For local use only, use --host 127.0.0.1")
        print()

    print(f"  API Docs: http://{args.host}:{args.port}/docs")
    print(f"  Health:   http://{args.host}:{args.port}/health")
    print("=" * 60)
    print()

    # Run server
    uvicorn.run(
        "cosyvoice_onnx.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
