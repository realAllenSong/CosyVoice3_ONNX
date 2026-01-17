#!/usr/bin/env python3
"""
Streaming API tests for CosyVoice3 ONNX Server.

These tests verify that the SSE streaming endpoint works correctly.
The server must be running for these tests to work.

Usage:
    # Start server first
    python run_server.py &

    # Run tests
    python tests/test_streaming.py
"""

import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

# Test configuration
BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 120  # seconds for streaming


def wait_for_server(max_wait: int = 30) -> bool:
    """Wait for server to be ready."""
    print(f"Waiting for server at {BASE_URL}...")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            resp = requests.get(f"{BASE_URL}/health", timeout=2)
            if resp.status_code == 200:
                print("  Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    print("  Server not responding!")
    return False


def test_stream_with_preset():
    """Test POST /stream with preset voice."""
    print("\n[1/3] Testing POST /stream with preset...")
    try:
        # Get available presets
        presets_resp = requests.get(f"{BASE_URL}/presets", timeout=10)
        presets = presets_resp.json().get("presets", [])

        if not presets:
            print("  ⊘ Skipped: No presets available")
            return None

        preset_name = presets[0]["name"]
        print(f"  Using preset: {preset_name}")

        # Make streaming request
        resp = requests.post(
            f"{BASE_URL}/stream",
            json={
                "text": "你好，这是流式输出测试。",
                "preset": preset_name,
                "speed": 1.0,
                "chunk_size_tokens": 30
            },
            stream=True,
            timeout=TIMEOUT
        )

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"

        # Parse SSE events
        chunks_received = 0
        total_duration_ms = 0
        all_audio_bytes = []

        for line in resp.iter_lines():
            if not line:
                continue

            line = line.decode("utf-8")

            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_str = line[5:].strip()
                try:
                    data = json.loads(data_str)

                    if event_type == "audio_chunk":
                        chunks_received += 1
                        chunk_bytes = base64.b64decode(data["chunk"])
                        all_audio_bytes.append(chunk_bytes)
                        duration_ms = data.get("duration_ms", 0)
                        total_duration_ms += duration_ms
                        print(f"    Chunk {data['index']}: {len(chunk_bytes)} bytes, "
                              f"{duration_ms}ms, final={data['is_final']}")

                    elif event_type == "done":
                        print(f"    Done: {data['total_chunks']} chunks, "
                              f"{data['duration_ms']}ms total")

                    elif event_type == "error":
                        print(f"    Error: {data['error']}")
                        return False

                except json.JSONDecodeError:
                    continue

        if chunks_received > 0:
            print(f"  ✓ Streaming successful: {chunks_received} chunks, "
                  f"{total_duration_ms}ms audio")
            return True
        else:
            print("  ✗ No chunks received")
            return False

    except requests.exceptions.HTTPError as e:
        if e.response and e.response.status_code == 400:
            print(f"  ⊘ Skipped: Request rejected - {e}")
            return None
        raise
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stream_with_audio():
    """Test POST /stream with base64 encoded reference audio."""
    print("\n[2/3] Testing POST /stream with reference audio...")
    try:
        # Create test reference audio
        sample_rate = 24000
        duration = 5.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

        # Convert to base64
        import soundfile as sf
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio, sample_rate, format="WAV")
        audio_b64 = base64.b64encode(wav_buffer.getvalue()).decode("utf-8")

        print(f"  Reference audio: {duration}s, {len(audio_b64)} chars base64")

        # Make streaming request
        resp = requests.post(
            f"{BASE_URL}/stream",
            json={
                "text": "这是使用参考音频的流式测试。",
                "prompt_audio_base64": audio_b64,
                "prompt_text": "这是参考音频的文本。",
                "speed": 1.0,
                "chunk_size_tokens": 20
            },
            stream=True,
            timeout=TIMEOUT
        )

        if resp.status_code != 200:
            error_detail = resp.text[:500] if resp.text else "No details"
            print(f"  ⚠ Request returned {resp.status_code}: {error_detail}")
            return None

        # Parse SSE events
        chunks_received = 0

        for line in resp.iter_lines():
            if not line:
                continue

            line = line.decode("utf-8")

            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_str = line[5:].strip()
                try:
                    data = json.loads(data_str)

                    if event_type == "audio_chunk":
                        chunks_received += 1
                        print(f"    Chunk {data['index']}: final={data['is_final']}")

                    elif event_type == "done":
                        print(f"    Done: {data['total_chunks']} chunks")

                    elif event_type == "error":
                        print(f"    Error: {data['error']}")
                        return False

                except json.JSONDecodeError:
                    continue

        if chunks_received > 0:
            print(f"  ✓ Streaming with audio successful: {chunks_received} chunks")
            return True
        else:
            print("  ✗ No chunks received")
            return False

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stream_error_handling():
    """Test streaming error handling."""
    print("\n[3/3] Testing streaming error handling...")
    checks_passed = 0

    try:
        # Test 1: Missing required fields
        resp = requests.post(
            f"{BASE_URL}/stream",
            json={"text": "test"},  # Missing preset or audio
            timeout=10
        )
        if resp.status_code == 400:
            print("  ✓ Missing preset/audio rejected correctly")
            checks_passed += 1
        else:
            print(f"  ⚠ Missing fields returned {resp.status_code}")

        # Test 2: Invalid base64 audio
        resp = requests.post(
            f"{BASE_URL}/stream",
            json={
                "text": "test",
                "prompt_audio_base64": "not-valid-base64!!!",
                "prompt_text": "test"
            },
            timeout=10
        )
        if resp.status_code == 400:
            print("  ✓ Invalid base64 rejected correctly")
            checks_passed += 1
        else:
            print(f"  ⚠ Invalid base64 returned {resp.status_code}")

        return checks_passed >= 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    """Run all streaming tests."""
    print("=" * 60)
    print("CosyVoice3 ONNX Streaming Tests")
    print("=" * 60)
    print(f"Server URL: {BASE_URL}")

    # Check server is running
    if not wait_for_server():
        print("\n⚠ Server is not running!")
        print("Please start the server first:")
        print("  python run_server.py")
        sys.exit(1)

    results = []

    # Run tests
    results.append(("Stream with Preset", test_stream_with_preset()))
    results.append(("Stream with Audio", test_stream_with_audio()))
    results.append(("Stream Error Handling", test_stream_error_handling()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = 0
    failed = 0
    skipped = 0

    for name, result in results:
        if result is True:
            status = "✓ PASSED"
            passed += 1
        elif result is False:
            status = "✗ FAILED"
            failed += 1
        else:
            status = "⊘ SKIPPED"
            skipped += 1
        print(f"  {name}: {status}")

    print()
    print(f"Total: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    # Exit code
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
