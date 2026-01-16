# CosyVoice3 ONNX 🎙️

**CosyVoice3 ONNX** - 简单易用的 CPU 语音合成与语音克隆库

基于 [ayousanz/cosy-voice3-onnx](https://huggingface.co/ayousanz/cosy-voice3-onnx) 实现，提供开箱即用的 Python API。

## ✨ 特性

- **零样本语音克隆** - 只需 3-15 秒参考音频即可克隆任意声音
- **CPU 高效运行** - 基于 ONNX Runtime，无需 GPU
- **跨平台支持** - macOS (Intel/Apple Silicon), Windows, Linux
- **自动模型下载** - 首次运行自动从 HuggingFace 下载模型
- **简洁 API** - 同步和异步接口，易于集成

## 🚀 快速开始

### 安装

```bash
pip install cosyvoice-onnx
```

或使用 UV：

```bash
uv pip install cosyvoice-onnx
```

### 基本使用

```python
from cosyvoice_onnx import CosyVoiceTTS

# 初始化（首次运行会自动下载模型，约 3GB）
tts = CosyVoiceTTS()

# 语音克隆
audio = tts.clone_voice(
    prompt_audio="speaker.wav",      # 参考音频（3-15秒）
    prompt_text="这是参考音频的文字内容",  # 参考音频的文字转录
    target_text="你好！这是克隆的声音。"   # 要合成的文本
)

# 保存音频
audio.save("output.wav")
```

### 异步版本

```python
import asyncio
from cosyvoice_onnx import CosyVoiceTTS

async def main():
    tts = CosyVoiceTTS()
    
    audio = await tts.clone_voice_async(
        prompt_audio="speaker.wav",
        prompt_text="Hello, my name is Alice.",
        target_text="Nice to meet you!"
    )
    audio.save("output.wav")

asyncio.run(main())
```

## 📖 API 参考

### CosyVoiceTTS

主要 TTS 类。

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_dir` | str | `~/.cosyvoice3/models` | 模型目录 |
| `precision` | str | `"auto"` | 精度：`"fp16"`, `"fp32"`, `"auto"` |
| `preload` | bool | `False` | 是否预加载模型 |
| `num_threads` | int | `0` | CPU 线程数（0=自动） |
| `log_level` | str | `"INFO"` | 日志级别 |

#### 主要方法

##### `clone_voice()` / `clone_voice_async()`

克隆声音并合成语音。

```python
audio = tts.clone_voice(
    prompt_audio="speaker.wav",  # str, bytes, 或 numpy array
    prompt_text="...",           # 参考音频的文字转录
    target_text="...",           # 要合成的文本
    speed=1.0,                   # 语速 (0.5-2.0)
    volume=1.0,                  # 音量 (0.0-2.0)
    output_format="wav"          # "wav" 或 "mp3"
)
```

##### `synthesize()` / `synthesize_async()`

通用合成方法（需要提供 prompt 或使用预设）。

##### `load_preset(name)` / `list_presets()`

加载和列出预设音色。

### AudioData

音频数据容器。

```python
audio.save("output.wav")           # 保存到文件
audio.to_numpy()                   # 转换为 numpy 数组
audio.to_bytes(format="wav")       # 转换为 bytes
audio.duration_ms                  # 时长（毫秒）
audio.sample_rate                  # 采样率
```
### 预设音色库

包含45+个高质量预设音色，涵盖多语言、情感、方言和官方/名人音色。

```python
from cosyvoice_onnx import download_presets

# 下载预设音色库
download_presets("presets/voices")
```

**可用预设：**

*   **多语言**: `zh_female_1`, `en_female_1`, `ja_female_1`, `ko_female_1`, `de_female_1` 等
*   **情感**: `emotion_happy_zh`, `emotion_angry_en`, `emotion_sad_zh`, `emotion_fearful_en` 等
*   **中国方言**: `dialect_cantonese` (粤语), `dialect_sichuan` (四川), `dialect_dongbei` (东北) 等
*   **VoxCPM 官方**: `ben` (英语男声), `trump` (名人), `dialact_guangxi` (广西普通话) 等

### HTTP 服务集成

提供了 FastAPI 服务示例，轻松集成到微服务架构。

```bash
# 1. 安装依赖
uv pip install fastapi uvicorn

# 2. 运行服务
python examples/server_example.py

# 3. 调用 API
curl -X POST "http://localhost:8000/tts" \
     -H "Content-Type: application/json" \
     -d '{"text": "你好，世界", "voice": "zh_female_1"}'
```

详见 [examples/server_example.py](examples/server_example.py)。
## 🎯 使用场景

### 桌面应用集成（PyQt5）

```python
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from cosyvoice_onnx import CosyVoiceTTS

class TTSWorker(QThread):
    finished = pyqtSignal(bytes)
    error = pyqtSignal(str)
    
    def __init__(self, tts, text, prompt_audio, prompt_text):
        super().__init__()
        self.tts = tts
        self.text = text
        self.prompt_audio = prompt_audio
        self.prompt_text = prompt_text
    
    def run(self):
        try:
            audio = self.tts.clone_voice(
                prompt_audio=self.prompt_audio,
                prompt_text=self.prompt_text,
                target_text=self.text
            )
            self.finished.emit(audio.to_bytes())
        except Exception as e:
            self.error.emit(str(e))
```

### 批量处理

```python
tts = CosyVoiceTTS(preload=True)  # 预加载模型

texts = ["句子一", "句子二", "句子三"]
for i, text in enumerate(texts):
    audio = tts.clone_voice(
        prompt_audio="speaker.wav",
        prompt_text="...",
        target_text=text
    )
    audio.save(f"output_{i}.wav")
```

## ⚙️ 配置

配置文件位于 `~/.cosyvoice3/config.yaml`：

```yaml
model_dir: ~/.cosyvoice3/models
precision: auto
num_threads: 0
default_speed: 1.0
default_volume: 1.0
auto_download: true
log_level: INFO
```

## 📦 模型文件

首次运行时会自动从 HuggingFace 下载模型：
- 下载源：`ayousanz/cosy-voice3-onnx`
- 大小：约 3GB
- 位置：`~/.cosyvoice3/models/`

手动下载：

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('ayousanz/cosy-voice3-onnx', local_dir='~/.cosyvoice3/models')"
```

## 🔧 故障排除

### ONNX Runtime 版本错误

```bash
pip install onnxruntime>=1.18.0
```

### NumPy 2.x 兼容性问题

```bash
pip install "numpy>=1.24.0,<2.0"
```

### 内存不足

尝试使用 FP16 精度：

```python
tts = CosyVoiceTTS(precision="fp16")
```

## 📄 许可证

Apache 2.0 License

## 🙏 致谢

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - 原始模型
- [ayousanz/cosy-voice3-onnx](https://huggingface.co/ayousanz/cosy-voice3-onnx) - ONNX 转换

## 📚 相关链接

- [CosyVoice 官方 Demo](https://funaudiollm.github.io/cosyvoice3/)
- [CosyVoice 论文](https://arxiv.org/pdf/2505.17589)
- [HuggingFace 模型](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
