#!/usr/bin/env python3
"""
Download preset voices from official demo pages.

This script downloads audio samples from:
- https://funaudiollm.github.io/cosyvoice3/
- https://openbmb.github.io/VoxCPM-demopage/
"""

import os
import json
import re
import requests
from pathlib import Path
from typing import List, Dict
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor


def get_default_presets_dir() -> Path:
    """Get default presets directory."""
    return Path.home() / ".cosyvoice3" / "presets"


def download_file(url: str, save_path: Path) -> bool:
    """Download a file from URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"  ❌ Failed to download {url}: {e}")
        return False


def download_cosyvoice3_demos(output_dir: Path) -> List[Dict]:
    """Download demo audio from CosyVoice3 demo page."""
    base_url = "https://funaudiollm.github.io/cosyvoice3/"
    
    print(f"\n📥 Downloading from CosyVoice3 demos...")
    
    # Common demo patterns (you may need to manually inspect the page)
    # These are placeholder URLs - actual implementation would scrape the page
    demos = [
        {
            "name": "chinese_female_1",
            "audio": "demos/chinese_female_1.wav",
            "transcript": "欢迎使用CosyVoice语音合成系统。",
            "language": "zh",
            "gender": "female"
        },
        {
            "name": "chinese_male_1", 
            "audio": "demos/chinese_male_1.wav",
            "transcript": "这是一个高质量的语音合成示例。",
            "language": "zh",
            "gender": "male"
        },
        {
            "name": "english_female_1",
            "audio": "demos/english_female_1.wav", 
            "transcript": "Hello, welcome to the CosyVoice text to speech system.",
            "language": "en",
            "gender": "female"
        },
        {
            "name": "english_male_1",
            "audio": "demos/english_male_1.wav",
            "transcript": "This is a demonstration of high quality speech synthesis.",
            "language": "en", 
            "gender": "male"
        }
    ]
    
    downloaded = []
    voices_dir = output_dir / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    
    for demo in demos:
        url = urljoin(base_url, demo["audio"])
        save_path = voices_dir / f"{demo['name']}.wav"
        
        print(f"  Downloading {demo['name']}...")
        
        # Note: Actual download may fail if files don't exist at these URLs
        # This is a template - actual demo files need to be obtained manually
        # or by properly scraping the demo page
        
        if download_file(url, save_path):
            demo["audio_path"] = str(save_path)
            downloaded.append(demo)
            print(f"  ✓ {demo['name']}")
        else:
            print(f"  ⚠️ Skipping {demo['name']} - file not available")
    
    return downloaded


def create_metadata(presets: List[Dict], output_dir: Path) -> None:
    """Create metadata.json for presets."""
    metadata = {}
    
    for preset in presets:
        name = preset["name"]
        metadata[name] = {
            "audio": f"{name}.wav",
            "transcript": preset.get("transcript", ""),
            "language": preset.get("language", "en"),
            "gender": preset.get("gender"),
            "description": preset.get("description", "")
        }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Created metadata.json with {len(metadata)} presets")


def main():
    print("=" * 60)
    print("CosyVoice3 ONNX - Preset Voice Downloader")
    print("=" * 60)
    
    output_dir = get_default_presets_dir()
    print(f"Output directory: {output_dir}")
    
    all_presets = []
    
    # Download from CosyVoice3 demos
    presets = download_cosyvoice3_demos(output_dir)
    all_presets.extend(presets)
    
    # Create metadata
    if all_presets:
        create_metadata(all_presets, output_dir)
        print(f"\n✅ Downloaded {len(all_presets)} preset voices")
    else:
        print("\n⚠️ No presets were downloaded.")
        print("   You may need to manually download audio samples from:")
        print("   - https://funaudiollm.github.io/cosyvoice3/")
        print("   - https://openbmb.github.io/VoxCPM-demopage/")
        print(f"\n   Place audio files in: {output_dir / 'voices'}")
        print("   Then create metadata.json with format:")
        print('   {"voice_name": {"audio": "filename.wav", "transcript": "...", "language": "zh"}}')
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
