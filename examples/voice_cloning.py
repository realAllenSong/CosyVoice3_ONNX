#!/usr/bin/env python3
"""
Voice Cloning Example for CosyVoice3 ONNX

This example demonstrates advanced voice cloning features:
1. Cloning a voice from any audio file
2. Generating speech in the cloned voice
3. Adjusting speed and volume
"""

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cosyvoice_onnx import CosyVoiceTTS


def main():
    print("=" * 60)
    print("CosyVoice3 ONNX - Voice Cloning Example")
    print("=" * 60)
    
    # Initialize
    tts = CosyVoiceTTS(log_level="INFO")
    
    # =========================================
    # Voice Cloning Example
    # =========================================
    
    # Replace these with your actual files
    prompt_audio = "path/to/speaker_sample.wav"
    prompt_text = "This is what the speaker says in the audio file."
    
    if not Path(prompt_audio).exists():
        print(f"\n❌ Please provide a reference audio file: {prompt_audio}")
        print("\nTo use voice cloning:")
        print("1. Record or find a clear audio sample (3-15 seconds)")
        print("2. Write down exactly what is said in the audio")
        print("3. Update the prompt_audio and prompt_text variables")
        return
    
    print(f"\n📢 Reference audio: {prompt_audio}")
    print(f"📝 Transcript: {prompt_text}")
    
    # Basic voice cloning
    print("\n1. Basic voice cloning...")
    audio = tts.clone_voice(
        prompt_audio=prompt_audio,
        prompt_text=prompt_text,
        target_text="Hello! I'm speaking with a cloned voice. Isn't that amazing?"
    )
    audio.save("output_cloned.wav")
    print(f"   ✓ Saved: output_cloned.wav ({audio.duration_ms}ms)")
    
    # Different speeds
    print("\n2. Speed variations...")
    
    for speed, label in [(0.8, "slow"), (1.0, "normal"), (1.3, "fast")]:
        audio = tts.clone_voice(
            prompt_audio=prompt_audio,
            prompt_text=prompt_text,
            target_text="This is a speed test.",
            speed=speed
        )
        filename = f"output_speed_{label}.wav"
        audio.save(filename)
        print(f"   ✓ Speed {speed}x: {filename} ({audio.duration_ms}ms)")
    
    # Multi-language (if supported in prompt)
    print("\n3. Multi-language synthesis...")
    
    texts = {
        "english": "Hello, how are you today?",
        "chinese": "你好，今天过得怎么样？",
        "japanese": "こんにちは、今日はどうですか？",
    }
    
    for lang, text in texts.items():
        try:
            audio = tts.clone_voice(
                prompt_audio=prompt_audio,
                prompt_text=prompt_text,
                target_text=text
            )
            filename = f"output_lang_{lang}.wav"
            audio.save(filename)
            print(f"   ✓ {lang.capitalize()}: {filename}")
        except Exception as e:
            print(f"   ⚠️ {lang.capitalize()}: {e}")
    
    print("\n" + "=" * 60)
    print("Voice cloning complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
