#!/usr/bin/env python3
"""
Download preset voices with MATCHED transcripts.
Wrapper around cosyvoice_onnx.utils.preset_downloader.
"""
import sys
import shutil
from pathlib import Path
from cosyvoice_onnx.utils.preset_downloader import download_presets

if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "presets/voices"
    
    # Clean existing presets if directory exists
    presets_path = Path(output_dir)
    if presets_path.exists():
        print("🗑️ Cleaning existing presets...")
        shutil.rmtree(presets_path)
    
    download_presets(output_dir)
