#!/bin/bash
#
# CosyVoice3 ONNX - 运行脚本
# Run script for testing TTS and Voice Cloning APIs
#
# 使用方法 / Usage:
#   ./run.sh                    # 运行完整测试
#   ./run.sh clone              # 仅测试语音克隆
#   ./run.sh tts                # 仅测试基础TTS (需要预设音色)
#   ./run.sh install            # 仅安装依赖
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}   CosyVoice3 ONNX 测试脚本${NC}"
echo -e "${BLUE}=====================================${NC}"

# 检查虚拟环境
setup_venv() {
    if [ ! -d ".venv" ]; then
        echo -e "${YELLOW}创建虚拟环境 (ARM64 Python)...${NC}"
        
        # 优先使用ARM Python
        if [ -f "/opt/homebrew/opt/python@3.11/bin/python3.11" ]; then
            uv venv --python /opt/homebrew/opt/python@3.11/bin/python3.11
        else
            uv venv --python 3.11
        fi
    fi
    
    source .venv/bin/activate
    
    # 显示Python信息
    echo -e "${GREEN}Python: $(python --version)${NC}"
    echo -e "${GREEN}架构: $(python -c 'import platform; print(platform.machine())')${NC}"
}

# 安装依赖
install_deps() {
    echo -e "${YELLOW}安装依赖...${NC}"
    uv pip install \
        "numpy>=1.24.0,<2.0" \
        scipy \
        soundfile \
        pyyaml \
        transformers \
        huggingface_hub \
        "onnxruntime>=1.17.0,<1.18.0" \
        librosa
    echo -e "${GREEN}依赖安装完成!${NC}"
}

# 测试语音克隆
test_voice_clone() {
    echo ""
    echo -e "${BLUE}=====================================${NC}"
    echo -e "${BLUE}   测试语音克隆 API${NC}"
    echo -e "${BLUE}=====================================${NC}"
    
    # 检查示例音频
    if [ ! -f "examples/samples/test_prompt.wav" ]; then
        echo -e "${YELLOW}下载示例音频...${NC}"
        mkdir -p examples/samples
        curl -sL "https://huggingface.co/ayousanz/cosy-voice3-onnx/resolve/main/prompts/en_female_nova_greeting.wav" \
            -o examples/samples/test_prompt.wav
        echo "Hello, my name is Sarah. I'm excited to help you with your project today. Let me know if you have any questions." \
            > examples/samples/test_prompt.txt
    fi
    
    python -c "
import sys
sys.path.insert(0, 'src')
from cosyvoice_onnx import CosyVoiceTTS

print('正在初始化 TTS 引擎...')
tts = CosyVoiceTTS(precision='fp16', log_level='WARNING')

print('正在加载模型 (首次需要下载约 3GB)...')
tts.model_manager.load_models()

print('正在生成语音...')
audio = tts.clone_voice(
    prompt_audio='examples/samples/test_prompt.wav',
    prompt_text='Hello, my name is Sarah. I am excited to help you with your project today. Let me know if you have any questions.',
    target_text='你好！这是使用 CosyVoice3 ONNX 生成的语音测试。'
)

audio.save('output_clone_test.wav')
print(f'✅ 语音克隆成功! 保存到: output_clone_test.wav')
print(f'   时长: {audio.duration_ms}ms')
"
}

# 测试基础TTS (需要预设)
test_basic_tts() {
    echo ""
    echo -e "${BLUE}=====================================${NC}"
    echo -e "${BLUE}   测试基础 TTS API${NC}"
    echo -e "${BLUE}=====================================${NC}"
    
    python -c "
import sys
sys.path.insert(0, 'src')
from cosyvoice_onnx import CosyVoiceTTS

print('正在初始化 TTS 引擎...')
tts = CosyVoiceTTS(precision='fp16', log_level='WARNING')

# 使用 synthesize API (与 clone_voice 相同，但更通用)
print('正在生成语音...')
audio = tts.synthesize(
    text='Hello! This is a test of the CosyVoice3 ONNX text-to-speech system.',
    prompt_audio='examples/samples/test_prompt.wav',
    prompt_text='Hello, my name is Sarah. I am excited to help you with your project today.',
    speed=1.0,
    volume=1.0
)

audio.save('output_tts_test.wav')
print(f'✅ TTS 成功! 保存到: output_tts_test.wav')
print(f'   时长: {audio.duration_ms}ms')
"
}

# 显示帮助
show_help() {
    echo ""
    echo "使用方法:"
    echo "  ./run.sh           # 运行完整测试 (克隆 + TTS)"
    echo "  ./run.sh clone     # 仅测试语音克隆"
    echo "  ./run.sh tts       # 仅测试基础TTS"
    echo "  ./run.sh install   # 仅安装依赖"
    echo "  ./run.sh help      # 显示此帮助"
    echo ""
    echo "配置调整:"
    echo "  1. 全局配置: ~/.cosyvoice3/config.yaml"
    echo "  2. 脚本配置: 修改 examples/test_v1.py 中的 CONFIG 字典"
    echo "  3. 代码配置: CosyVoiceTTS(precision='fp16', num_threads=4)"
    echo ""
}

# 主函数
main() {
    case "${1:-all}" in
        install)
            setup_venv
            install_deps
            ;;
        clone)
            setup_venv
            test_voice_clone
            ;;
        tts)
            setup_venv
            test_basic_tts
            ;;
        all)
            setup_venv
            # 检查依赖
            if ! python -c "import onnxruntime" 2>/dev/null; then
                install_deps
            fi
            test_voice_clone
            test_basic_tts
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo -e "${RED}未知命令: $1${NC}"
            show_help
            exit 1
            ;;
    esac
    
    echo ""
    echo -e "${GREEN}=====================================${NC}"
    echo -e "${GREEN}   完成!${NC}"
    echo -e "${GREEN}=====================================${NC}"
}

main "$@"
