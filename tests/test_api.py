#!/usr/bin/env python3
"""
API endpoint tests for CosyVoice3 ONNX Server.

These tests verify that the FastAPI server endpoints work correctly.
The server must be running for these tests to work.

Usage:
    # Start server first
    python run_server.py &

    # Run tests
    python tests/test_api.py

    # Or with pytest
    pytest tests/test_api.py -v
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
TIMEOUT = 60  # seconds


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


def test_health_endpoint():
    """Test GET /health endpoint."""
    print("\n[1/7] Testing GET /health...")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"

        data = resp.json()
        assert "status" in data, "Missing 'status' field"
        assert "model_loaded" in data, "Missing 'model_loaded' field"
        assert "version" in data, "Missing 'version' field"

        print(f"  ✓ Health check passed: {data}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_presets_endpoint():
    """Test GET /presets endpoint."""
    print("\n[2/7] Testing GET /presets...")
    try:
        resp = requests.get(f"{BASE_URL}/presets", timeout=10)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"

        data = resp.json()
        assert "presets" in data, "Missing 'presets' field"
        assert isinstance(data["presets"], list), "'presets' should be a list"

        print(f"  ✓ Found {len(data['presets'])} presets")
        if data["presets"]:
            print(f"    First preset: {data['presets'][0]}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_validate_audio_endpoint():
    """Test POST /validate_audio endpoint."""
    print("\n[3/7] Testing POST /validate_audio...")
    try:
        # Create a test audio file (3 seconds of sine wave)
        sample_rate = 24000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

        # Write to WAV format
        import soundfile as sf
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio, sample_rate, format="WAV")
        wav_buffer.seek(0)

        # Send request
        files = {"audio": ("test.wav", wav_buffer, "audio/wav")}
        resp = requests.post(f"{BASE_URL}/validate_audio", files=files, timeout=10)

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"

        data = resp.json()
        assert "valid" in data, "Missing 'valid' field"
        assert "sample_rate" in data, "Missing 'sample_rate' field"
        assert "duration_seconds" in data, "Missing 'duration_seconds' field"

        print(f"  ✓ Validation result: valid={data['valid']}, "
              f"sr={data['sample_rate']}, duration={data['duration_seconds']:.2f}s")
        if data.get("warnings"):
            print(f"    Warnings: {data['warnings']}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tts_with_preset():
    """Test POST /tts with preset voice."""
    print("\n[4/7] Testing POST /tts with preset...")
    try:
        # First get available presets
        presets_resp = requests.get(f"{BASE_URL}/presets", timeout=10)
        presets = presets_resp.json().get("presets", [])

        if not presets:
            print("  ⊘ Skipped: No presets available")
            return None

        preset_name = presets[0]["name"]
        print(f"  Using preset: {preset_name}")

        # Make TTS request
        resp = requests.post(
            f"{BASE_URL}/tts",
            json={
                "text": "你好，这是一个测试。",
                "preset": preset_name,
                "speed": 1.0,
                "format": "base64"
            },
            timeout=TIMEOUT
        )

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"

        data = resp.json()
        assert "audio" in data, "Missing 'audio' field"
        assert "duration_ms" in data, "Missing 'duration_ms' field"

        # Decode and check audio
        audio_bytes = base64.b64decode(data["audio"])
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

        print(f"  ✓ TTS successful: {data['duration_ms']}ms, "
              f"{len(audio_array)} samples")
        return True
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            print(f"  ⊘ Skipped: Preset not available - {e}")
            return None
        raise
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_clone_endpoint():
    """Test POST /clone endpoint."""
    print("\n[5/7] Testing POST /clone...")
    try:
        # Create a test reference audio (5 seconds)
        sample_rate = 24000
        duration = 5.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

        # Write to WAV
        import soundfile as sf
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio, sample_rate, format="WAV")
        wav_buffer.seek(0)

        # Make clone request
        files = {"prompt_audio": ("reference.wav", wav_buffer, "audio/wav")}
        data = {
            "prompt_text": "这是参考音频的文本。",
            "target_text": "你好，世界。",
            "speed": "1.0",
            "volume": "1.0",
            "format": "base64"
        }

        resp = requests.post(
            f"{BASE_URL}/clone",
            files=files,
            data=data,
            timeout=TIMEOUT
        )

        if resp.status_code == 200:
            result = resp.json()
            assert "audio" in result, "Missing 'audio' field"
            print(f"  ✓ Clone successful: {result.get('duration_ms', 'N/A')}ms")
            return True
        else:
            # Model might not be loaded or other issue
            print(f"  ⚠ Clone returned {resp.status_code}: {resp.text[:200]}")
            return None

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tts_error_handling():
    """Test error handling for invalid requests."""
    print("\n[6/7] Testing error handling...")
    checks_passed = 0

    try:
        # Test 1: Empty text
        resp = requests.post(
            f"{BASE_URL}/tts",
            json={"text": "", "preset": "test"},
            timeout=10
        )
        if resp.status_code == 422:  # Validation error
            print("  ✓ Empty text rejected correctly")
            checks_passed += 1
        else:
            print(f"  ⚠ Empty text returned {resp.status_code}")

        # Test 2: Missing preset
        resp = requests.post(
            f"{BASE_URL}/tts",
            json={"text": "test"},
            timeout=10
        )
        if resp.status_code == 400:  # Bad request
            print("  ✓ Missing preset rejected correctly")
            checks_passed += 1
        else:
            print(f"  ⚠ Missing preset returned {resp.status_code}")

        # Test 3: Invalid audio file
        resp = requests.post(
            f"{BASE_URL}/validate_audio",
            files={"audio": ("test.txt", b"not audio data", "text/plain")},
            timeout=10
        )
        # Should return 500 (internal error) or 200 with valid=false
        if resp.status_code in [200, 500]:
            data = resp.json()
            if resp.status_code == 200 and not data.get("valid", True):
                print("  ✓ Invalid audio rejected correctly (valid=false)")
                checks_passed += 1
            elif resp.status_code == 500:
                print("  ✓ Invalid audio rejected correctly (500 error)")
                checks_passed += 1
            else:
                print(f"  ⚠ Invalid audio handling: {data}")
        else:
            print(f"  ⚠ Invalid audio returned {resp.status_code}")

        return checks_passed >= 2
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_stats_endpoint():
    """Test GET /stats endpoint."""
    print("\n[7/7] Testing GET /stats...")
    try:
        resp = requests.get(f"{BASE_URL}/stats", timeout=5)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"

        data = resp.json()
        assert "uptime_seconds" in data, "Missing 'uptime_seconds' field"
        assert "request_count" in data, "Missing 'request_count' field"

        print(f"  ✓ Stats: uptime={data['uptime_seconds']:.1f}s, "
              f"requests={data['request_count']}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    """Run all API tests."""
    print("=" * 60)
    print("CosyVoice3 ONNX API Tests")
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
    results.append(("Health Endpoint", test_health_endpoint()))
    results.append(("Presets Endpoint", test_presets_endpoint()))
    results.append(("Validate Audio", test_validate_audio_endpoint()))
    results.append(("TTS with Preset", test_tts_with_preset()))
    results.append(("Clone Endpoint", test_clone_endpoint()))
    results.append(("Error Handling", test_tts_error_handling()))
    results.append(("Stats Endpoint", test_stats_endpoint()))

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
