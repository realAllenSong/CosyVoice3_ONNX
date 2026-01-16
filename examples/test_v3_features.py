#!/usr/bin/env python3
"""
V3 Features Test - Text Normalization, Languages, Dialects, Prosody

Tests the new V3 features including:
- Text normalization (numbers, dates, currencies)
- Multi-language detection
- Chinese dialect support
- Prosody control (emotion, pinyin)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_text_normalization():
    """Test text normalization features."""
    from cosyvoice_onnx import TextNormalizer, normalize_text
    
    print("=" * 60)
    print("Text Normalization Test")
    print("=" * 60)
    
    normalizer = TextNormalizer(default_language="zh")
    
    # Test Chinese normalization
    test_cases_zh = [
        ("今天是2024年1月15日", "今天是二零二四年一月一五日 (expected)"),
        ("价格是$100美元", "价格是美元一零零美元 (expected)"),
        ("他跑了123米", "他跑了一二三米 (expected)"),
        ("成功率达到95%", "成功率达到九五百分之 (expected)"),
    ]
    
    print("\n[Chinese Normalization]")
    for text, expected in test_cases_zh:
        result = normalizer.normalize(text, "zh")
        print(f"  Input:  {text}")
        print(f"  Output: {result}")
        print()
    
    # Test English normalization
    normalizer_en = TextNormalizer(default_language="en")
    test_cases_en = [
        ("Dr. Smith has 3 apples", "Doctor Smith has three apples (expected)"),
        ("The price is $10", "The price is ten (expected)"),
    ]
    
    print("[English Normalization]")
    for text, expected in test_cases_en:
        result = normalizer_en.normalize(text, "en")
        print(f"  Input:  {text}")
        print(f"  Output: {result}")
        print()
    
    print(f"Full WeTextProcessing support: {normalizer.has_full_support}")


def test_language_detection():
    """Test language detection and dialect support."""
    from cosyvoice_onnx import LanguageDetector, SUPPORTED_LANGUAGES, SUPPORTED_DIALECTS
    
    print("=" * 60)
    print("Language Detection Test")
    print("=" * 60)
    
    detector = LanguageDetector()
    
    test_texts = [
        ("你好世界", "zh"),
        ("Hello world", "en"),
        ("こんにちは", "jp"),
        ("안녕하세요", "ko"),
        ("Привет мир", "ru"),
        ("Hello 你好 world", "en"),  # Mixed
    ]
    
    print("\n[Language Detection]")
    for text, expected in test_texts:
        detected = detector.detect(text)
        status = "✅" if detected == expected else "❌"
        print(f"  {status} '{text}' -> {detected} (expected: {expected})")
    
    # Test dialect parsing
    print("\n[Dialect Parsing]")
    dialect_texts = [
        "<dialect:canton>你好</dialect>",
        "<dialect:sichuan>巴适</dialect>",
        "Normal text without dialect",
    ]
    
    for text in dialect_texts:
        cleaned, dialect = detector.parse_dialect_tags(text)
        print(f"  Input: {text}")
        print(f"  Output: text='{cleaned}', dialect={dialect}")
        if dialect:
            print(f"  Prefix: {detector.get_language_prefix('zh', dialect)}")
        print()
    
    print(f"\nSupported Languages ({len(SUPPORTED_LANGUAGES)}):")
    for code, name in list(SUPPORTED_LANGUAGES.items())[:5]:
        print(f"  {code}: {name}")
    print("  ...")
    
    print(f"\nSupported Dialects ({len(SUPPORTED_DIALECTS)}):")
    for code, name in list(SUPPORTED_DIALECTS.items())[:5]:
        print(f"  {code}: {name}")
    print("  ...")


def test_prosody_control():
    """Test prosody control tags."""
    from cosyvoice_onnx import ProsodyParser, parse_prosody_tags
    
    print("=" * 60)
    print("Prosody Control Test")
    print("=" * 60)
    
    parser = ProsodyParser()
    
    test_texts = [
        "[happy]今天天气真好！",
        "[sad]我很难过。",
        "<pinyin:ce4 shi4>测试</pinyin>发音。",
        "<speed:1.5>快速说话</speed>，然后<speed:0.8>慢速说话</speed>。",
        "普通文本，<pause:500ms/>停顿一下。",
        "[excited]这太棒了！<pinyin:ni3 hao3>你好</pinyin>！",
    ]
    
    print("\n[Prosody Tag Parsing]")
    for text in test_texts:
        cleaned, tags = parser.parse(text)
        emotion = parser.get_emotion(tags)
        
        print(f"  Input: {text}")
        print(f"  Cleaned: {cleaned}")
        if emotion:
            print(f"  Emotion: {emotion}")
            print(f"  Prefix: {parser.get_emotion_prefix(emotion)}")
        if tags:
            print(f"  Tags: {len(tags)} found")
            for tag in tags:
                print(f"    - {tag.tag_type}: {tag.value}")
        print()


def test_audio_processor():
    """Test audio processing utilities."""
    from cosyvoice_onnx import AudioProcessor, AudioData
    import numpy as np
    
    print("=" * 60)
    print("Audio Processor Test")
    print("=" * 60)
    
    processor = AudioProcessor(sample_rate=24000)
    
    # Create test audio
    duration = 1.0  # 1 second
    sample_rate = 24000
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio1 = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
    audio2 = np.sin(2 * np.pi * 880 * t).astype(np.float32) * 0.3
    
    # Test normalize
    print("\n[Volume Normalization]")
    normalized = processor.normalize_volume(audio1, target_db=-20.0, method="peak")
    print(f"  Original peak: {np.max(np.abs(audio1)):.3f}")
    print(f"  Normalized peak: {np.max(np.abs(normalized)):.3f}")
    
    # Test concatenation
    print("\n[Audio Concatenation]")
    audio_data1 = AudioData(
        data=audio1.tobytes(),
        sample_rate=sample_rate,
        channels=1,
        format="wav",
        duration_ms=int(duration * 1000)
    )
    audio_data2 = AudioData(
        data=audio2.tobytes(),
        sample_rate=sample_rate,
        channels=1,
        format="wav",
        duration_ms=int(duration * 1000)
    )
    
    combined = processor.concat_audio([audio_data1, audio_data2], gap_ms=100)
    print(f"  Audio 1: {audio_data1.duration_ms}ms")
    print(f"  Audio 2: {audio_data2.duration_ms}ms")
    print(f"  Gap: 100ms")
    print(f"  Combined: {combined.duration_ms}ms")
    
    # Test silence trimming
    print("\n[Silence Trimming]")
    # Audio with silence at start and end
    silence = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
    audio_with_silence = np.concatenate([silence, audio1, silence])
    original_len = len(audio_with_silence) / sample_rate * 1000
    
    trimmed = processor.trim_silence(audio_with_silence, threshold_db=-40.0)
    trimmed_len = len(trimmed) / sample_rate * 1000
    
    print(f"  Original duration: {original_len:.0f}ms")
    print(f"  Trimmed duration: {trimmed_len:.0f}ms")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CosyVoice3 ONNX - V3 Features Test")
    print("=" * 60 + "\n")
    
    test_text_normalization()
    print()
    
    test_language_detection()
    print()
    
    test_prosody_control()
    print()
    
    test_audio_processor()
    
    print("\n" + "=" * 60)
    print("All V3 feature tests completed!")
    print("=" * 60)
