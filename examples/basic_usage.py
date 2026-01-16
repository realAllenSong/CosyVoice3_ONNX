#!/usr/bin/env python3
"""
Basic Usage Example for CosyVoice3 ONNX

This example demonstrates:
1. Basic TTS with voice cloning
2. Using preset voices
3. Saving output in different formats
"""

import asyncio
from pathlib import Path

# Add src to path for development
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cosyvoice_onnx import CosyVoiceTTS, AudioData


async def main():
    print("=" * 60)
    print("CosyVoice3 ONNX - Basic Usage Example")
    print("=" * 60)
    
    # Initialize TTS (models will be downloaded automatically on first use)
    print("\n1. Initializing TTS engine...")
    tts = CosyVoiceTTS(
        precision="auto",  # Auto-select FP16 or FP32 based on system
        preload=False,     # Lazy load models on first synthesis
        log_level="INFO"
    )
    
    # Example with voice cloning
    # You need to provide a reference audio file and its transcript
    prompt_audio = "path/to/your/reference_audio.wav"  # 3-15 seconds of speech
    prompt_text = "The transcript of what is said in the reference audio."
    
    if Path(prompt_audio).exists():
        print("\n2. Synthesizing with voice cloning...")
        
        # Synchronous version
        audio = tts.clone_voice(
            prompt_audio=prompt_audio,
            prompt_text=prompt_text,
            target_text="Hello! This is a test of the CosyVoice3 ONNX TTS system.",
            speed=1.0,
            volume=1.0
        )
        
        # Save as WAV
        output_path = "output_basic.wav"
        audio.save(output_path)
        print(f"   Saved to: {output_path}")
        print(f"   Duration: {audio.duration_ms}ms")
        print(f"   Sample rate: {audio.sample_rate}Hz")
        
        # Async version
        print("\n3. Async synthesis example...")
        audio = await tts.clone_voice_async(
            prompt_audio=prompt_audio,
            prompt_text=prompt_text,
            target_text="This is the async version of voice synthesis.",
        )
        audio.save("output_async.wav")
        print("   Saved to: output_async.wav")
        
    else:
        print(f"\n⚠️  Reference audio not found: {prompt_audio}")
        print("   Please provide a reference audio file for voice cloning.")
        print("\n   CosyVoice is a voice cloning TTS system that requires:")
        print("   1. A reference audio file (3-15 seconds of clear speech)")
        print("   2. The transcript of what is spoken in the audio")
        print("\n   Example:")
        print('   audio = tts.clone_voice(')
        print('       prompt_audio="speaker.wav",')
        print('       prompt_text="Hello, my name is Alice.",')
        print('       target_text="Text you want to synthesize"')
        print('   )')
    
    # Using presets (if available)
    print("\n4. Checking for preset voices...")
    presets = tts.list_presets()
    if presets:
        print(f"   Found {len(presets)} presets: {presets}")
        
        # Use first preset
        preset = tts.load_preset(presets[0])
        audio = tts.synthesize_with_preset(
            text="Hello from preset voice!",
            preset=preset
        )
        audio.save(f"output_preset_{presets[0]}.wav")
        print(f"   Saved preset audio to: output_preset_{presets[0]}.wav")
    else:
        print("   No presets found. You can add presets to:")
        print(f"   ~/.cosyvoice3/presets/voices/")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
