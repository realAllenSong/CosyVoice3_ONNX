#!/bin/bash
# ============================================================
# CosyVoice3 ONNX - 一键启动脚本
# 读取 config.json 配置并生成语音或启动服务器
#
# 使用方法:
#   ./tts.sh              # 根据 config.json 的 mode 运行
#   ./tts.sh server       # 直接启动 FastAPI 服务器
#   ./tts.sh clone        # 直接运行语音克隆
#   ./tts.sh preset       # 直接运行预设模式
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 解析参数
MODE_OVERRIDE=""
CONFIG_FILE="config.json"

if [ "$1" = "server" ] || [ "$1" = "clone" ] || [ "$1" = "preset" ] || [ "$1" = "batch" ]; then
    MODE_OVERRIDE="$1"
    CONFIG_FILE="${2:-config.json}"
else
    CONFIG_FILE="${1:-config.json}"
fi

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}   CosyVoice3 ONNX TTS${NC}"
echo -e "${BLUE}============================================================${NC}"

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ 配置文件不存在: $CONFIG_FILE${NC}"
    echo -e "${YELLOW}请先复制模板: cp config.json.example config.json${NC}"
    exit 1
fi

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}创建虚拟环境...${NC}"
    uv venv .venv --python python3.11
fi

# 激活环境
source .venv/bin/activate

# 确保基础依赖已安装
if ! python -c "import cosyvoice_onnx" 2>/dev/null; then
    echo -e "${YELLOW}安装基础依赖...${NC}"
    uv pip install -e .
fi

# 如果是服务器模式，确保服务器依赖已安装
if [ "$MODE_OVERRIDE" = "server" ] || grep -q '"mode": *"server"' "$CONFIG_FILE" 2>/dev/null; then
    if ! python -c "import fastapi; import uvicorn; import sse_starlette" 2>/dev/null; then
        echo -e "${YELLOW}安装服务器依赖...${NC}"
        uv pip install -e ".[server]"
    fi
fi

echo -e "${GREEN}配置文件: $CONFIG_FILE${NC}"
if [ -n "$MODE_OVERRIDE" ]; then
    echo -e "${GREEN}模式覆盖: $MODE_OVERRIDE${NC}"
fi
echo ""

# 设置模式覆盖环境变量
export COSYVOICE_MODE="$MODE_OVERRIDE"

# 运行 Python 脚本
python - "$CONFIG_FILE" << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""
TTS Runner - 读取 config.json 并生成语音
"""

import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def load_config(config_path: str) -> dict:
    """加载配置文件，忽略注释字段"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 移除所有以 // 开头的注释字段
    def remove_comments(obj):
        if isinstance(obj, dict):
            return {k: remove_comments(v) for k, v in obj.items() if not k.startswith('//')}
        elif isinstance(obj, list):
            return [remove_comments(item) for item in obj]
        return obj
    
    return remove_comments(config)


def run_clone_mode(config: dict):
    """语音克隆模式"""
    from cosyvoice_onnx import CosyVoiceTTS, TextNormalizer, ProsodyParser, AudioProcessor
    
    clone_cfg = config.get('clone', {})
    speech_cfg = config.get('speech', {})
    model_cfg = config.get('model', {})
    gen_cfg = config.get('generation', {})
    output_cfg = config.get('output', {})
    text_cfg = config.get('text_processing', {})
    
    # 获取文本
    text = config.get('text', '')
    text_file = config.get('text_file', '')
    if text_file and Path(text_file).exists():
        text = Path(text_file).read_text(encoding='utf-8').strip()
    
    if not text:
        print("❌ 错误: 未指定文本内容")
        return
    
    print(f"📝 文本: {text[:50]}{'...' if len(text) > 50 else ''}")
    print(f"🎤 参考音频: {clone_cfg.get('prompt_audio', 'N/A')}")
    
    # 文本预处理
    if text_cfg.get('enable_normalization', True):
        normalizer = TextNormalizer()
        lang = speech_cfg.get('language', 'auto')
        if lang == 'auto':
            lang = 'zh' if any('\u4e00' <= c <= '\u9fff' for c in text) else 'en'
        text = normalizer.normalize(text, lang)
        print(f"📋 规范化后: {text[:50]}{'...' if len(text) > 50 else ''}")
    
    if text_cfg.get('enable_prosody_tags', True):
        parser = ProsodyParser()
        text, tags = parser.parse(text)
        emotion = parser.get_emotion(tags)
        if emotion:
            print(f"😊 检测到情感: {emotion}")
            # 添加情感前缀
            prefix = parser.get_emotion_prefix(emotion)
            text = prefix + text
    
    # 初始化 TTS
    print("\n🔧 初始化 TTS 引擎...")
    tts = CosyVoiceTTS(
        precision=model_cfg.get('precision', 'fp16'),
        num_threads=model_cfg.get('num_threads', 0),
        preload=model_cfg.get('preload', False),
        log_level="WARNING"
    )
    
    # 更新配置
    tts.config.sampling_k = gen_cfg.get('sampling_k', 25)
    tts.config.max_tokens = gen_cfg.get('max_tokens', 500)
    tts.config.min_tokens = gen_cfg.get('min_tokens', 10)
    tts.config.n_timesteps = gen_cfg.get('n_timesteps', 10)
    
    # 生成
    print("🎵 生成语音...")
    start_time = time.time()
    
    audio = tts.clone_voice(
        prompt_audio=clone_cfg.get('prompt_audio'),
        prompt_text=clone_cfg.get('prompt_text'),
        target_text=text,
        speed=speech_cfg.get('speed', 1.0),
        volume=speech_cfg.get('volume', 1.0)
    )
    
    elapsed = time.time() - start_time
    
    # 后处理
    processor = AudioProcessor(audio.sample_rate)
    
    if output_cfg.get('normalize_volume', True):
        audio = processor.normalize_volume(audio, target_db=output_cfg.get('target_db', -20.0))
    
    if output_cfg.get('trim_silence', False):
        audio = processor.trim_silence(audio)
    
    # 保存
    output_file = output_cfg.get('file', 'output.wav')
    audio.save(output_file)
    
    print(f"\n✅ 完成!")
    print(f"   输出文件: {output_file}")
    print(f"   时长: {audio.duration_ms}ms")
    print(f"   耗时: {elapsed:.2f}s")


def run_batch_mode(config: dict):
    """批量处理模式"""
    from cosyvoice_onnx import CosyVoiceTTS
    
    batch_cfg = config.get('batch', {})
    clone_cfg = config.get('clone', {})
    speech_cfg = config.get('speech', {})
    model_cfg = config.get('model', {})
    output_cfg = config.get('output', {})
    
    text_list = batch_cfg.get('text_list', [])
    output_dir = Path(batch_cfg.get('output_dir', 'batch_output/'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not text_list:
        print("❌ 错误: batch.text_list 为空")
        return
    
    print(f"📦 批量处理: {len(text_list)} 条文本")
    
    # 初始化
    tts = CosyVoiceTTS(
        precision=model_cfg.get('precision', 'fp16'),
        preload=True,
        log_level="WARNING"
    )
    
    for i, text in enumerate(text_list, 1):
        print(f"\n[{i}/{len(text_list)}] {text[:30]}...")
        
        audio = tts.clone_voice(
            prompt_audio=clone_cfg.get('prompt_audio'),
            prompt_text=clone_cfg.get('prompt_text'),
            target_text=text,
            speed=speech_cfg.get('speed', 1.0)
        )
        
        output_file = output_dir / f"output_{i:03d}.wav"
        audio.save(str(output_file))
        print(f"   ✅ {output_file}")
    
    print(f"\n✅ 批量处理完成! 输出目录: {output_dir}")


def run_preset_mode(config: dict):
    """预设音色模式"""
    from cosyvoice_onnx import CosyVoiceTTS, TextNormalizer, ProsodyParser, AudioProcessor
    
    preset_cfg = config.get('preset', {})
    speech_cfg = config.get('speech', {})
    model_cfg = config.get('model', {})
    gen_cfg = config.get('generation', {})
    output_cfg = config.get('output', {})
    text_cfg = config.get('text_processing', {})
    
    voice_name = preset_cfg.get('voice', 'zh_female_1')
    
    # 查找预设
    presets_dir = Path('presets')
    metadata_path = presets_dir / 'metadata.json'
    
    if not metadata_path.exists():
        print("❌ 预设元数据不存在，请先运行: python scripts/download_presets.py")
        return
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    if voice_name not in metadata:
        print(f"❌ 预设 '{voice_name}' 不存在")
        print(f"   可用预设: {', '.join(list(metadata.keys())[:10])}...")
        return
    
    preset = metadata[voice_name]
    prompt_audio = str(presets_dir / 'voices' / preset['audio'])
    prompt_text = preset.get('transcript', '')
    
    # 如果没有 transcript，使用占位符
    if not prompt_text:
        prompt_text = "This is a sample voice for text to speech synthesis."
    
    print(f"🎤 预设音色: {voice_name}")
    print(f"   语言: {preset.get('language', 'unknown')}")
    print(f"   风格: {preset.get('style', 'unknown')}")
    
    # 获取文本
    text = config.get('text', '')
    text_file = config.get('text_file', '')
    if text_file and Path(text_file).exists():
        text = Path(text_file).read_text(encoding='utf-8').strip()
    
    if not text:
        print("❌ 错误: 未指定文本内容")
        return
    
    print(f"📝 文本: {text[:50]}{'...' if len(text) > 50 else ''}")
    
    # 文本预处理
    if text_cfg.get('enable_normalization', True):
        normalizer = TextNormalizer()
        lang = speech_cfg.get('language', 'auto')
        if lang == 'auto':
            lang = preset.get('language', 'zh')
        text = normalizer.normalize(text, lang)
    
    if text_cfg.get('enable_prosody_tags', True):
        parser = ProsodyParser()
        text, tags = parser.parse(text)
        emotion = parser.get_emotion(tags)
        if emotion:
            prefix = parser.get_emotion_prefix(emotion)
            text = prefix + text
    
    # 初始化 TTS
    print("\n🔧 初始化 TTS 引擎...")
    tts = CosyVoiceTTS(
        precision=model_cfg.get('precision', 'fp16'),
        num_threads=model_cfg.get('num_threads', 0),
        preload=model_cfg.get('preload', False),
        log_level="WARNING"
    )
    
    tts.config.sampling_k = gen_cfg.get('sampling_k', 25)
    tts.config.max_tokens = gen_cfg.get('max_tokens', 500)
    tts.config.min_tokens = gen_cfg.get('min_tokens', 10)
    tts.config.n_timesteps = gen_cfg.get('n_timesteps', 10)
    
    # 生成
    print("🎵 生成语音...")
    start_time = time.time()
    
    audio = tts.clone_voice(
        prompt_audio=prompt_audio,
        prompt_text=prompt_text,
        target_text=text,
        speed=speech_cfg.get('speed', 1.0),
        volume=speech_cfg.get('volume', 1.0)
    )
    
    elapsed = time.time() - start_time
    
    # 后处理
    processor = AudioProcessor(audio.sample_rate)
    
    if output_cfg.get('normalize_volume', True):
        audio = processor.normalize_volume(audio, target_db=output_cfg.get('target_db', -20.0))
    
    if output_cfg.get('trim_silence', False):
        audio = processor.trim_silence(audio)
    
    # 保存
    output_file = output_cfg.get('file', 'output.wav')
    audio.save(output_file)
    
    print(f"\n✅ 完成!")
    print(f"   输出文件: {output_file}")
    print(f"   时长: {audio.duration_ms}ms")
    print(f"   耗时: {elapsed:.2f}s")


def run_server_mode(config: dict):
    """启动 FastAPI 服务器"""
    import os

    server_cfg = config.get('server', {})
    model_cfg = config.get('model', {})

    host = server_cfg.get('host', '127.0.0.1')
    port = server_cfg.get('port', 8000)
    log_level = server_cfg.get('log_level', 'INFO').lower()
    workers = server_cfg.get('workers', 1)

    print(f"🌐 启动 FastAPI 服务器...")
    print(f"   地址: http://{host}:{port}")
    print(f"   API 文档: http://{host}:{port}/docs")
    print(f"   日志级别: {log_level}")
    print(f"   精度: {model_cfg.get('precision', 'fp16')}")
    print()
    print("📌 可用端点:")
    print("   GET  /health         - 健康检查")
    print("   GET  /presets        - 预设声音列表")
    print("   POST /tts            - 基础 TTS")
    print("   POST /clone          - 声音克隆")
    print("   POST /stream         - 流式输出")
    print("   POST /validate_audio - 验证音频")
    print()
    print("按 Ctrl+C 停止服务器")
    print("=" * 60)

    # 设置环境变量让服务器读取配置
    os.environ['COSYVOICE_CONFIG'] = sys.argv[1] if len(sys.argv) > 1 else 'config.json'

    import uvicorn
    uvicorn.run(
        "cosyvoice_onnx.server:app",
        host=host,
        port=port,
        log_level=log_level,
        workers=workers,
        reload=False
    )


def main():
    import os

    if len(sys.argv) < 2:
        print("Usage: python tts_runner.py config.json")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)

    # 检查是否有模式覆盖 (从环境变量)
    mode_override = os.environ.get('COSYVOICE_MODE', '')
    mode = mode_override if mode_override else config.get('mode', 'clone')

    print(f"🚀 模式: {mode}")

    if mode == 'server':
        run_server_mode(config)
    elif mode == 'clone':
        run_clone_mode(config)
    elif mode == 'batch':
        run_batch_mode(config)
    elif mode == 'preset':
        run_preset_mode(config)
    else:
        print(f"❌ 未知模式: {mode}")
        print("   可用模式: server | clone | preset | batch")


if __name__ == '__main__':
    main()
PYTHON_SCRIPT

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}   完成!${NC}"
echo -e "${GREEN}============================================================${NC}"
