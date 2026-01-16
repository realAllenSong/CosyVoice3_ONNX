#!/usr/bin/env python3
"""
V1 Integration Test Script for CosyVoice3 ONNX

This script tests the complete TTS pipeline:
1. Downloads models from HuggingFace (if not present)
2. Downloads a sample prompt audio for testing
3. Generates audio with voice cloning
4. Saves output for verification

配置调整说明:
--------------
1. 全局配置文件: ~/.cosyvoice3/config.yaml
2. 运行时参数: 见下方 CONFIG 部分
3. 模型精度: precision = "fp16" | "fp32" | "auto"
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ==========================================
# 配置区域 - 可以调整以下参数
# ==========================================
CONFIG = {
    # 模型配置
    "precision": "fp16",       # 使用 FP16 (ayousanz repo 只提供 FP16 版本)
    "num_threads": 0,          # CPU 线程数，0=自动

    # 生成参数
    "speed": 1.0,              # 语速: 0.5 (慢) - 2.0 (快)
    "volume": 1.0,             # 音量: 0.0 - 2.0
    
    # 采样参数 (高级)
    "sampling_k": 25,          # Top-k 采样，越大越多样
    "max_tokens": 50000,         # 最大生成 token 数
    "min_tokens": 10,          # 最小生成 token 数
    "n_timesteps": 10,         # Flow 步数，越多质量越好但更慢
    
    # 输出
    "output_format": "wav",    # "wav" 或 "mp3"
}

# 测试文本
TEST_TEXTS = {
    "chinese": "你好！这是使用 CosyVoice3 ONNX 生成的语音测试。今天天气真好！",
    "english": "Hello! This is a test of the CosyVoice3 ONNX text-to-speech system.",
    "mixed": "Hello，你好！这是一个 mixed 中英文混合语音测试。",
}

# ==========================================
# 测试脚本
# ==========================================

def download_sample_audio():
    """Download a sample audio for testing voice cloning."""
    samples_dir = Path(__file__).parent / "samples"
    samples_dir.mkdir(exist_ok=True)
    
    sample_path = samples_dir / "test_prompt.wav"
    transcript_path = samples_dir / "test_prompt.txt"
    
    if sample_path.exists() and transcript_path.exists():
        print(f"✓ Sample audio found: {sample_path}")
        with open(transcript_path, 'r') as f:
            transcript = f.read().strip()
        return str(sample_path), transcript
    
    if sample_path.exists() and not transcript_path.exists():
        # Create default transcript
        transcript = "Hello, my name is Sarah. I'm excited to help you with your project today. Let me know if you have any questions."
        with open(transcript_path, 'w') as f:
            f.write(transcript)
        print(f"✓ Sample audio found (created transcript): {sample_path}")
        return str(sample_path), transcript
    
    print("⚠️ No sample audio found.")
    print("   Please provide your own audio file for testing.")
    print("\n   To test, place a WAV file (3-15 seconds) in:")
    print(f"   {samples_dir}/test_prompt.wav")
    print("\n   And create a transcript file:")
    print(f"   {samples_dir}/test_prompt.txt")
    print("   (containing the exact text spoken in the audio)")
    
    return None, None


def test_model_download(tts):
    """Test model download functionality."""
    print("\n" + "="*60)
    print("Step 1: Checking/Downloading Models")
    print("="*60)
    
    start = time.time()
    try:
        tts.model_manager.ensure_models_exist()
        print(f"✓ Models ready in {time.time() - start:.1f}s")
        return True
    except Exception as e:
        print(f"✗ Model download failed: {e}")
        return False


def test_model_loading(tts):
    """Test model loading."""
    print("\n" + "="*60)
    print("Step 2: Loading Models")
    print("="*60)
    
    start = time.time()
    try:
        tts.model_manager.load_models()
        print(f"✓ Models loaded in {time.time() - start:.1f}s")
        print(f"  - Precision: {tts.config.get_precision_suffix()}")
        print(f"  - Model dir: {tts.config.model_dir}")
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_voice_cloning(tts, prompt_audio, prompt_text):
    """Test voice cloning."""
    print("\n" + "="*60)
    print("Step 3: Testing Voice Cloning")
    print("="*60)
    
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    for lang, text in TEST_TEXTS.items():
        print(f"\n  Testing {lang}...")
        print(f"  Text: {text[:50]}...")
        
        start = time.time()
        try:
            audio = tts.clone_voice(
                prompt_audio=prompt_audio,
                prompt_text=prompt_text,
                target_text=text,
                speed=CONFIG["speed"],
                volume=CONFIG["volume"],
                output_format=CONFIG["output_format"]
            )
            
            elapsed = time.time() - start
            output_path = output_dir / f"test_{lang}.{CONFIG['output_format']}"
            audio.save(str(output_path))
            
            rtf = elapsed / (audio.duration_ms / 1000)
            
            print(f"  ✓ Success!")
            print(f"    - Duration: {audio.duration_ms}ms")
            print(f"    - Time: {elapsed:.2f}s")
            print(f"    - RTF: {rtf:.2f} (lower is better)")
            print(f"    - Output: {output_path}")
            
            results.append({
                "lang": lang,
                "success": True,
                "duration_ms": audio.duration_ms,
                "time_s": elapsed,
                "rtf": rtf,
                "output": str(output_path)
            })
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "lang": lang,
                "success": False,
                "error": str(e)
            })
    
    return results


def print_config_guide():
    """Print configuration guide."""
    print("\n" + "="*60)
    print("📋 配置调整指南")
    print("="*60)
    
    print("""
1. 全局配置文件 (~/.cosyvoice3/config.yaml):
   ----------------------------------------
   model_dir: ~/.cosyvoice3/models   # 模型存储路径
   precision: auto                    # fp16/fp32/auto
   num_threads: 0                     # CPU线程数
   default_speed: 1.0                 # 默认语速
   default_volume: 1.0                # 默认音量
   auto_download: true                # 自动下载模型
   log_level: INFO                    # 日志级别

2. 运行时参数 (在代码中调整):
   ----------------------------------------
   tts = CosyVoiceTTS(
       precision="fp16",              # 精度选择
       num_threads=4,                 # CPU线程
       preload=True                   # 预加载模型
   )
   
   audio = tts.clone_voice(
       speed=1.2,                     # 加速 20%
       volume=0.8,                    # 降低音量
       output_format="mp3"            # 输出格式
   )

3. 高级参数 (修改 CosyVoiceConfig):
   ----------------------------------------
   from cosyvoice_onnx import CosyVoiceConfig
   
   config = CosyVoiceConfig(
       sampling_k=25,                 # Top-k 采样
       max_tokens=500,                # 最大token
       min_tokens=10,                 # 最小token
       n_timesteps=10                 # Flow步数
   )
   tts = CosyVoiceTTS(config=config)

4. 本测试脚本的配置:
   ----------------------------------------
   直接修改脚本顶部的 CONFIG 字典
""")


def main():
    print("="*60)
    print("CosyVoice3 ONNX - V1 Integration Test")
    print("="*60)
    
    # Check for sample audio
    prompt_audio, prompt_text = download_sample_audio()
    
    if prompt_audio is None:
        print("\n❌ Cannot proceed without sample audio.")
        print("   Please add a sample audio file and run again.")
        print_config_guide()
        return
    
    print(f"\n📢 Using prompt audio: {prompt_audio}")
    print(f"📝 Transcript: {prompt_text[:50]}...")
    
    # Import and initialize
    from cosyvoice_onnx import CosyVoiceTTS, CosyVoiceConfig
    
    print("\n🔧 Initializing with config:")
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    
    config = CosyVoiceConfig(
        precision=CONFIG["precision"],
        num_threads=CONFIG["num_threads"],
        default_speed=CONFIG["speed"],
        default_volume=CONFIG["volume"],
        sampling_k=CONFIG["sampling_k"],
        max_tokens=CONFIG["max_tokens"],
        min_tokens=CONFIG["min_tokens"],
        n_timesteps=CONFIG["n_timesteps"],
    )
    
    tts = CosyVoiceTTS(config=config, log_level="INFO")
    
    # Run tests
    if not test_model_download(tts):
        return
    
    if not test_model_loading(tts):
        return
    
    results = test_voice_cloning(tts, prompt_audio, prompt_text)
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    success_count = sum(1 for r in results if r.get("success"))
    print(f"\n  Passed: {success_count}/{len(results)}")
    
    for r in results:
        status = "✓" if r.get("success") else "✗"
        print(f"  {status} {r['lang']}: ", end="")
        if r.get("success"):
            print(f"{r['duration_ms']}ms audio in {r['time_s']:.2f}s (RTF={r['rtf']:.2f})")
        else:
            print(f"Failed - {r.get('error', 'unknown error')}")
    
    if success_count > 0:
        print(f"\n  Output files saved to: {Path(__file__).parent / 'outputs'}")
    
    # Config guide
    print_config_guide()


if __name__ == "__main__":
    main()
