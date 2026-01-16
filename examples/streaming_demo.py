#!/usr/bin/env python3
"""
Streaming Output Demo for CosyVoice3 ONNX

Demonstrates real-time audio generation with streaming API.
Audio chunks are yielded as they are generated.
"""

import sys
import time
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def streaming_demo_async():
    """Async streaming demo."""
    from cosyvoice_onnx import CosyVoiceTTS
    
    print("=" * 60)
    print("CosyVoice3 ONNX - Streaming Demo (Async)")
    print("=" * 60)
    
    # Configuration
    PROMPT_AUDIO = Path(__file__).parent / "samples" / "test_prompt.wav"
    PROMPT_TEXT = "Hello, my name is Sarah. I'm excited to help you with your project today. Let me know if you have any questions."
    TEST_TEXT = "这是一个流式输出的测试。每生成一段音频就会立即输出。这样可以实现边生成边播放，降低整体延迟。"
    
    if not PROMPT_AUDIO.exists():
        print(f"❌ Sample audio not found: {PROMPT_AUDIO}")
        print("   Run ./run.sh first to download samples.")
        return
    
    # Initialize TTS
    print("\n初始化 TTS 引擎...")
    tts = CosyVoiceTTS(precision="fp16", log_level="WARNING")
    
    # Progress callback
    def on_progress(info):
        print(f"   进度: {info.progress_percent:.1f}%")
    
    # Streaming synthesis
    print(f"\n目标文本: {TEST_TEXT}")
    print("\n开始流式生成...\n")
    
    chunks = []
    chunk_times = []
    start_time = time.time()
    first_chunk_time = None
    
    async for chunk in tts.synthesize_stream(
        text=TEST_TEXT,
        prompt_audio=str(PROMPT_AUDIO),
        prompt_text=PROMPT_TEXT,
        chunk_size_tokens=30,
        on_progress=on_progress
    ):
        chunk_time = time.time() - start_time
        
        if first_chunk_time is None:
            first_chunk_time = chunk_time
            print(f"🎵 首个 chunk 到达: {first_chunk_time*1000:.0f}ms")
        
        # Calculate chunk duration
        import numpy as np
        audio_samples = len(chunk.data) // 4  # float32 = 4 bytes
        chunk_duration_ms = audio_samples / chunk.sample_rate * 1000
        
        chunks.append(chunk)
        chunk_times.append(chunk_time)
        
        marker = "🏁" if chunk.is_final else "📦"
        print(f"   {marker} Chunk {chunk.chunk_index}: {chunk_duration_ms:.0f}ms audio @ {chunk_time*1000:.0f}ms")
    
    total_time = time.time() - start_time
    
    # Combine all chunks into final audio
    print("\n合并所有 chunks...")
    import numpy as np
    all_audio = []
    for chunk in chunks:
        audio_array = np.frombuffer(chunk.data, dtype=np.float32)
        all_audio.append(audio_array)
    
    combined_audio = np.concatenate(all_audio)
    total_duration_ms = len(combined_audio) / chunks[0].sample_rate * 1000
    
    # Save combined audio
    output_path = Path(__file__).parent / "outputs" / "streaming_output.wav"
    output_path.parent.mkdir(exist_ok=True)
    
    import soundfile as sf
    sf.write(str(output_path), combined_audio, chunks[0].sample_rate)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Streaming Summary")
    print("=" * 60)
    print(f"  总时长: {total_duration_ms:.0f}ms")
    print(f"  总耗时: {total_time*1000:.0f}ms")
    print(f"  首 chunk 延迟: {first_chunk_time*1000:.0f}ms")
    print(f"  Chunk 数量: {len(chunks)}")
    print(f"  RTF: {total_time / (total_duration_ms/1000):.2f}")
    print(f"  输出文件: {output_path}")
    print("=" * 60)


def streaming_demo_sync():
    """Synchronous streaming demo."""
    from cosyvoice_onnx import CosyVoiceTTS
    
    print("=" * 60)
    print("CosyVoice3 ONNX - Streaming Demo (Sync)")
    print("=" * 60)
    
    PROMPT_AUDIO = Path(__file__).parent / "samples" / "test_prompt.wav"
    PROMPT_TEXT = "Hello, my name is Sarah. I'm excited to help you with your project today."
    TEST_TEXT = "Hello! This is a streaming output test. Each chunk is yielded immediately."
    
    if not PROMPT_AUDIO.exists():
        print(f"❌ Sample audio not found: {PROMPT_AUDIO}")
        return
    
    print("\n初始化...")
    tts = CosyVoiceTTS(precision="fp16", log_level="WARNING")
    
    print(f"\n目标文本: {TEST_TEXT}\n")
    
    start_time = time.time()
    first_chunk_time = None
    
    for chunk in tts.synthesize_stream_sync(
        text=TEST_TEXT,
        prompt_audio=str(PROMPT_AUDIO),
        prompt_text=PROMPT_TEXT,
        chunk_size_tokens=30
    ):
        if first_chunk_time is None:
            first_chunk_time = time.time() - start_time
            print(f"🎵 首个 chunk: {first_chunk_time*1000:.0f}ms")
        
        import numpy as np
        audio_samples = len(chunk.data) // 4
        chunk_duration = audio_samples / chunk.sample_rate * 1000
        
        marker = "🏁" if chunk.is_final else "📦"
        print(f"   {marker} Chunk {chunk.chunk_index}: {chunk_duration:.0f}ms")
    
    print(f"\n总耗时: {(time.time() - start_time)*1000:.0f}ms")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Streaming demo")
    parser.add_argument("--sync", action="store_true", help="Use synchronous API")
    args = parser.parse_args()
    
    if args.sync:
        streaming_demo_sync()
    else:
        asyncio.run(streaming_demo_async())
