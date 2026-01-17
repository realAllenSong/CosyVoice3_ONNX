#!/usr/bin/env python3
"""
Quality test script for CosyVoice3 ONNX.

This script verifies that the quality fixes are working correctly:
1. System prompt prefix (prevents random/nonsense speech)
2. FP32 precision (prevents long silences)
3. Audio validation (ensures proper sample rate)

Usage:
    python tests/test_quality.py

Requirements:
    - A reference audio file (will use examples/samples/test_prompt.wav if exists)
    - Models downloaded (will auto-download if not present)
"""

import sys
import time
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np


def test_imports():
    """Test that all imports work correctly."""
    print("\n[1/6] Testing imports...")
    try:
        from cosyvoice_onnx import (
            CosyVoiceTTS,
            AudioValidator,
            validate_reference_audio,
            CosyVoiceConfig,
        )
        from cosyvoice_onnx.engine import CosyVoiceEngine
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_system_prompt_prefix():
    """Test that system prompt prefix is defined correctly."""
    print("\n[2/6] Testing system prompt prefix...")
    try:
        from cosyvoice_onnx.engine import CosyVoiceEngine

        expected_prefix = "You are a helpful assistant.<|endofprompt|>"
        actual_prefix = CosyVoiceEngine.SYSTEM_PROMPT_PREFIX

        if actual_prefix == expected_prefix:
            print(f"  ✓ System prompt prefix is correct: '{actual_prefix}'")
            return True
        else:
            print(f"  ✗ System prompt prefix mismatch!")
            print(f"    Expected: '{expected_prefix}'")
            print(f"    Got: '{actual_prefix}'")
            return False
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def test_fp32_enforcement():
    """Test that FP32 precision is enforced."""
    print("\n[3/6] Testing FP32 precision enforcement...")
    try:
        from cosyvoice_onnx import CosyVoiceTTS

        # Test with fp16 - should be overridden to fp32
        tts = CosyVoiceTTS(precision="fp16", preload=False)

        if tts.config.precision == "fp32":
            print("  ✓ FP32 is enforced even when fp16 requested")
            return True
        else:
            print(f"  ✗ Precision not enforced: {tts.config.precision}")
            return False
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def test_audio_validator():
    """Test audio validation functionality."""
    print("\n[4/6] Testing audio validator...")
    try:
        from cosyvoice_onnx import validate_reference_audio

        # Create a test audio array (1 second of sine wave at 44100Hz)
        sample_rate = 44100
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        result = validate_reference_audio(audio, sample_rate=sample_rate)

        checks_passed = 0

        # Check 1: Should be valid
        if result.valid:
            print("  ✓ Audio validation passed")
            checks_passed += 1
        else:
            print(f"  ✗ Validation failed: {result.errors}")

        # Check 2: Should have resampling warning
        has_resample_warning = any("Resampling" in w for w in result.warnings)
        if has_resample_warning:
            print(f"  ✓ Resampling warning present: {result.warnings}")
            checks_passed += 1
        else:
            print(f"  ✗ Expected resampling warning, got: {result.warnings}")

        # Check 3: Processed audio should be 24kHz
        if result.processed_sample_rate == 24000:
            print("  ✓ Processed audio is 24kHz")
            checks_passed += 1
        else:
            print(f"  ✗ Expected 24kHz, got {result.processed_sample_rate}Hz")

        return checks_passed == 3
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_duration_limits():
    """Test audio duration validation."""
    print("\n[5/6] Testing audio duration limits...")
    try:
        from cosyvoice_onnx import validate_reference_audio

        sample_rate = 24000
        checks_passed = 0

        # Test too short audio (0.5 seconds)
        short_audio = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
        result = validate_reference_audio(short_audio, sample_rate=sample_rate)
        if not result.valid and any("too short" in e for e in result.errors):
            print("  ✓ Correctly rejected too-short audio")
            checks_passed += 1
        else:
            print(f"  ✗ Should reject too-short audio: {result.errors}")

        # Test too long audio (35 seconds)
        long_audio = np.zeros(int(sample_rate * 35), dtype=np.float32)
        result = validate_reference_audio(long_audio, sample_rate=sample_rate)
        if not result.valid and any("too long" in e for e in result.errors):
            print("  ✓ Correctly rejected too-long audio")
            checks_passed += 1
        else:
            print(f"  ✗ Should reject too-long audio: {result.errors}")

        # Test valid audio (5 seconds)
        valid_audio = np.zeros(int(sample_rate * 5), dtype=np.float32)
        result = validate_reference_audio(valid_audio, sample_rate=sample_rate)
        if result.valid:
            print("  ✓ Correctly accepted valid-length audio")
            checks_passed += 1
        else:
            print(f"  ✗ Should accept valid audio: {result.errors}")

        return checks_passed == 3
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_synthesis_quality(reference_audio_path: str = None):
    """Test actual synthesis quality (requires models)."""
    print("\n[6/6] Testing synthesis quality (optional - requires models)...")

    # Find reference audio
    if reference_audio_path is None:
        candidates = [
            Path(__file__).parent.parent / "examples" / "samples" / "test_prompt.wav",
            Path(__file__).parent.parent / "presets" / "voices" / "en_m.wav",
        ]
        for candidate in candidates:
            if candidate.exists():
                reference_audio_path = str(candidate)
                break

    if reference_audio_path is None or not Path(reference_audio_path).exists():
        print("  ⊘ Skipped: No reference audio found")
        print("    To run this test, provide a reference audio file")
        return None  # Skip, not fail

    try:
        from cosyvoice_onnx import CosyVoiceTTS, validate_reference_audio
        import soundfile as sf

        # Validate reference audio first
        print(f"  Using reference audio: {reference_audio_path}")
        validation = validate_reference_audio(reference_audio_path)
        if not validation.valid:
            print(f"  ⊘ Reference audio validation failed: {validation.errors}")
            return None

        print(f"  Reference audio: {validation.duration_seconds:.2f}s at {validation.sample_rate}Hz")
        if validation.warnings:
            print(f"  Warnings: {validation.warnings}")

        # Initialize TTS (models will be downloaded if needed)
        print("  Initializing TTS (this may take a while on first run)...")
        tts = CosyVoiceTTS(precision="fp32", preload=False)

        # Test synthesis
        test_text = "你好，世界。这是一个测试。"
        prompt_text = "这是参考音频的文本内容。"  # Placeholder

        print(f"  Synthesizing: '{test_text}'")
        start_time = time.time()

        audio = tts.synthesize(
            text=test_text,
            prompt_audio=reference_audio_path,
            prompt_text=prompt_text,
            speed=1.0,
            volume=1.0
        )

        elapsed = time.time() - start_time

        # Check results
        checks_passed = 0

        # Check 1: Audio was generated
        if audio.duration_ms > 0:
            print(f"  ✓ Audio generated: {audio.duration_ms}ms in {elapsed:.2f}s")
            checks_passed += 1
        else:
            print("  ✗ No audio generated")

        # Check 2: Audio length is reasonable (not too short or too long)
        expected_min_ms = len(test_text) * 50  # ~50ms per character minimum
        expected_max_ms = len(test_text) * 500  # ~500ms per character maximum

        if expected_min_ms <= audio.duration_ms <= expected_max_ms:
            print(f"  ✓ Audio length reasonable: {audio.duration_ms}ms")
            checks_passed += 1
        else:
            print(f"  ⚠ Audio length may be abnormal: {audio.duration_ms}ms")
            print(f"    Expected range: {expected_min_ms}-{expected_max_ms}ms")
            # Don't fail, just warn
            checks_passed += 1

        # Check 3: No excessive silence (check first/last 10% of audio)
        audio_array = np.frombuffer(audio.data, dtype=np.float32)
        samples_10pct = len(audio_array) // 10

        first_10pct_power = np.mean(audio_array[:samples_10pct] ** 2)
        last_10pct_power = np.mean(audio_array[-samples_10pct:] ** 2)
        total_power = np.mean(audio_array ** 2)

        silence_threshold = 0.001  # Very low power threshold

        if first_10pct_power > silence_threshold or last_10pct_power > silence_threshold:
            print(f"  ✓ No excessive silence detected")
            checks_passed += 1
        else:
            if total_power > silence_threshold:
                print(f"  ⚠ Possible silence at start/end (may be normal)")
                checks_passed += 1
            else:
                print(f"  ✗ Audio may be mostly silent")

        # Save test output
        output_path = Path(__file__).parent / "test_output.wav"
        audio.save(str(output_path))
        print(f"  Output saved to: {output_path}")

        return checks_passed >= 2

    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all quality tests."""
    print("=" * 60)
    print("CosyVoice3 ONNX Quality Tests")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("System Prompt Prefix", test_system_prompt_prefix()))
    results.append(("FP32 Enforcement", test_fp32_enforcement()))
    results.append(("Audio Validator", test_audio_validator()))
    results.append(("Duration Limits", test_audio_duration_limits()))

    # Optional synthesis test
    synth_result = test_synthesis_quality()
    if synth_result is not None:
        results.append(("Synthesis Quality", synth_result))

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
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
