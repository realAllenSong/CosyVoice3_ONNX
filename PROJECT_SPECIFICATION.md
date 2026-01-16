# CosyVoice 3 0.5B ONNX 项目规格说明
**Project Specification for CosyVoice3-ONNX TTS Service**

**版本**: 1.0  
**日期**: 2026-01-15  
**项目类型**: 开源 Python TTS 服务库  
**目标用户**: Python 开发者（特别是桌面应用开发者）

---

## 🎯 项目目标 (Project Goals)

基于 [ayousanz/cosy-voice3-onnx](https://huggingface.co/ayousanz/cosy-voice3-onnx) 实现，创建一个**生产级、跨平台、易集成**的 CosyVoice 3 0.5B ONNX TTS 服务库，专为桌面宠物等需要实时语音交互的应用设计。

### 核心价值主张
1. **CPU 高效运行** - 量化 ONNX 模型，无需 GPU
2. **即插即用** - 一行代码集成，自动化模型下载
3. **实时流式输出** - 边生成边播放，低延迟体验
4. **全功能支持** - 零样本语音克隆、多语言、中文方言、发音控制等

---

## 🏗️ 系统架构 (System Architecture)

### 技术栈设计
```
┌─────────────────────────────────────────────┐
│         User Application (PyQt5)            │
│         (如：VCat 桌面宠物)                  │
└─────────────────┬───────────────────────────┘
                  │ Python Async API
┌─────────────────▼───────────────────────────┐
│     CosyVoice3-ONNX Service Layer           │
│  ┌─────────────────────────────────────┐    │
│  │  Async API (AsyncIO based)          │    │
│  │  - synthesize_async()               │    │
│  │  - synthesize_stream()              │    │
│  │  - clone_voice_async()              │    │
│  └─────────────────────────────────────┘    │
│  ┌─────────────────────────────────────┐    │
│  │  Model Manager                      │    │
│  │  - Lazy loading / Preloading        │    │
│  │  - Auto download from HuggingFace   │    │
│  │  - Memory management                │    │
│  └─────────────────────────────────────┘    │
│  ┌─────────────────────────────────────┐    │
│  │  Audio Pipeline                     │    │
│  │  - Text normalization               │    │
│  │  - Streaming chunk generator        │    │
│  │  - Format conversion (WAV/MP3)      │    │
│  │  - Post-processing (normalize etc.) │    │
│  └─────────────────────────────────────┘    │
└─────────────────┬───────────────────────────┘
                  │ ONNX Runtime
┌─────────────────▼───────────────────────────┐
│   CosyVoice3 ONNX Models (FP16/FP32)        │
│   - LLM Model                               │
│   - Flow Model                              │
│   - HiFT Decoder                            │
└─────────────────────────────────────────────┘
```

### 项目目录结构
```
CosyVoice3_ONNX/
├── src/
│   ├── cosyvoice_onnx/
│   │   ├── __init__.py                # 公开 API
│   │   ├── api.py                     # AsyncIO API 接口
│   │   ├── model_manager.py           # 模型加载/下载管理
│   │   ├── audio_processor.py         # 音频处理管线
│   │   ├── config.py                  # 配置管理
│   │   ├── voice_cloner.py            # 语音克隆功能
│   │   ├── streaming.py               # 流式输出逻辑
│   │   ├── text_normalizer.py         # 文本规范化
│   │   └── utils/
│   │       ├── logger.py              # 日志系统
│   │       ├── retry.py               # 重试机制
│   │       └── downloader.py          # HuggingFace 下载器
├── models/                            # 模型文件目录（自动下载）
│   └── cosyvoice3-0.5b/
│       ├── llm-*.onnx
│       ├── flow-*.onnx
│       ├── hift-*.onnx
│       └── tokenizer/
├── presets/                           # 预设音色库
│   ├── voices/                        # 从官方 demo 提取
│   │   ├── female_soft.wav
│   │   ├── male_energetic.wav
│   │   └── ...
│   └── metadata.json
├── examples/                          # 示例代码
│   ├── basic_usage.py                 # 基础 TTS
│   ├── voice_cloning.py               # 语音克隆示例
│   ├── streaming_demo.py              # 流式输出
│   ├── pyqt5_integration.py           # PyQt5 集成示例
│   └── advanced_controls.py           # 高级控制（情感、语速等）
├── tests/                             # 测试（暂不要求）
├── docs/                              # 文档
│   ├── API.md                         # 完整 API 文档
│   ├── TUTORIAL.md                    # 使用教程
│   ├── BEST_PRACTICES.md              # 最佳实践
│   └── TROUBLESHOOTING.md             # 故障排查
├── scripts/                           # 工具脚本
│   ├── download_models.py             # 手动下载模型
│   ├── download_presets.py            # 下载预设音色
│   └── build_package.py               # 打包脚本（PyInstaller）
├── pyproject.toml                     # 项目配置（uv 兼容）
├── README.md                          # 项目说明
├── LICENSE                            # Apache 2.0
└── .gitignore
```

---

## 📋 功能需求 (Functional Requirements)

### Phase 1 - 基础 TTS + 语音克隆 (V1)
**MVP 核心功能**

#### F1.1 基础文本转语音
- **输入**: 中英文文本（主要）
- **输出**: WAV/MP3 音频文件或内存 bytes
- **方法**: `async synthesize_async(text: str, **kwargs) -> AudioData`
- **支持参数**:
  - `speed`: 语速控制 (0.5~2.0，默认 1.0)
  - `volume`: 音量控制 (0.0~2.0，默认 1.0)  
  - `emotion`: 情感标签（如 "happy", "sad", 默认 "neutral"）
  - `output_format`: "wav" | "mp3" (默认 "wav")

#### F1.2 零样本语音克隆 (优先级: 4/5)
- **方法**: `async clone_voice_async(prompt_audio: Union[str, bytes], prompt_text: str, target_text: str, **kwargs) -> AudioData`
- **支持格式**: WAV, MP3, FLAC, OGG 等常见格式
- **Prompt 音频要求**:
  - 长度：10-15 秒推荐，3-30 秒可接受
  - 采样率：自动重采样，无限制
  - 质量：清晰人声，低噪音
- **实现**: 基于 ayousanz 实现，使用 prompt_wav + prompt_text 方式

#### F1.3 预设音色库
- **来源**: 从官方 demo 页面下载优质音色
  - [CosyVoice3 Demos](https://funaudiollm.github.io/cosyvoice3/)
  - [VoxCPM Demos](https://openbmb.github.io/VoxCPM-demopage/)
- **数量**: 全部下载下来然后我在自己挑选
- **访问**: `load_preset(name: str) -> PresetVoice`
- **元数据**: JSON 文件记录音色名称、性别、风格、语言等

#### F1.4 模型自动下载
- **首次运行**: 检测模型缺失，自动从 HuggingFace 下载
- **下载源**: `ayousanz/cosy-voice3-onnx`
- **下载内容**:
  - ONNX 模型文件 (FP16 和 FP32 两个版本)
  - Tokenizer 文件
  - 配置文件
- **进度显示**: 回调函数报告下载进度
- **断点续传**: 使用 `huggingface_hub` 的 `snapshot_download`
- **离线模式**: 支持手动下载后配置路径

#### F1.5 基础配置系统
- **配置文件**: `~/.cosyvoice3/config.yaml`
- **可配置项**:
  - `model_dir`: 模型存储路径
  - `precision`: "fp16" | "fp32" (自动选择最快的)
  - `num_threads`: CPU 线程数（默认自动）
  - `default_speed`: 默认语速
  - `default_volume`: 默认音量
  - `cache_size`: 模型缓存大小
- **API 覆盖**: 运行时参数覆盖配置文件

---

### Phase 2 - 流式输出 (V2)
**实时交互关键功能** (优先级: 5/5)

#### F2.1 流式文本转语音
- **方法**: `async synthesize_stream(text: str, **kwargs) -> AsyncIterator[AudioChunk]`
- **返回**: 异步生成器，产出音频 chunk
- **Chunk 格式**:
  ```python
  @dataclass
  class AudioChunk:
      data: bytes          # PCM 音频数据
      sample_rate: int     # 采样率
      channels: int        # 声道数
      format: str          # 'pcm16' | 'float32'
      is_final: bool       # 是否为最后一个 chunk
  ```
- **性能目标**:
  - **首 chunk 延迟**: <500ms (越快越好)
  - **持续输出**: 平滑无明显间隔
  - **生成速度**: Real-time factor < 0.5 (M3 Max 目标)

#### F2.2 中断控制
- **方法**: `cancel_synthesis(task_id: str)`
- **行为**: 立即停止音频生成，释放资源
- **清理**: 清除缓冲区，中止 ONNX 推理

#### F2.3 进度回调
- **方法签名**: `on_progress: Callable[[ProgressInfo], None]`
- **ProgressInfo**:
  ```python
  @dataclass
  class ProgressInfo:
      task_id: str
      total_chars: int
      processed_chars: int
      progress_percent: float
      estimated_remaining_ms: int
  ```

---

### Phase 3 - 完整功能 + 打包 (V3)
**生产就绪功能**

#### F3.1 多语言支持 (优先级: 4/5)
- **支持语言**: 9 种
  - 中文（简体）✅ 核心
  - 英文 ✅ 核心
  - 日语、韩语、德语、西班牙语、法语、意大利语、俄语
- **自动检测**: 文本语言自动识别
- **混合语言**: 支持中英混合等

#### F3.2 中文方言支持 (优先级: 4/5)
- **方言列表**: 广东话、闽南话、四川话、东北话、陕西话、山西话、上海话、天津话、山东话、宁夏、甘肃等 18+
- **使用方式**: 通过 `dialect` 参数或指令文本 `<dialect:canton>你好</dialect>`

#### F3.3 发音修复/控制 (优先级: 5/5)
- **拼音控制**: 支持中文拼音标注
  - 示例: `这是一个<pinyin:ce4 shi4>测试`
- **音素控制**: 支持 CMU 音素（英文）
  - 示例: `This is a <phoneme:T EH1 S T>test`
- **Text Normalization**: 默认启用
  - 数字读法（"123" → "一百二十三"）
  - 符号处理（"$100" → "一百美元"）
  - 日期时间（"2024-01-15" → "二零二四年一月十五日"）

#### F3.4 情感与指令控制 (优先级: 4/5)
- **情感标签**: `emotion` 参数
  - 支持: happy, sad, angry, calm, excited 等
- **指令式**: 文本内嵌指令（如官方支持）
  - 示例: `[用快乐的语气说]今天天气真好！`

#### F3.5 音频后处理 (优先级: 3/5)
- **音量归一化**: 确保输出音量一致
- **降噪**: 可选的后处理降噪
- **音频拼接**: `concat_audio(audio_list: List[AudioData]) -> AudioData`
- **格式转换**: 支持 WAV, MP3 输出

#### F3.6 资源管理与优化
- **懒加载**: 首次调用时加载模型
- **预加载**: `preload_models()` 显式预加载
- **内存监控**: 长时间运行防止内存泄漏
  - 定期释放缓存
  - 限制 KV-cache 大小
- **队列管理**: 单设备单请求（用户要求）
  - 内部队列确保串行处理

#### F3.7 跨平台打包
- **打包工具**: PyInstaller（推荐）或 Nuitka
- **支持平台**:
  - **macOS**: 11+ (Intel x86_64 + Apple Silicon ARM64)
  - **Windows**: 10/11 (x86_64)
  - **Linux**: Ubuntu 20.04+ / Debian 11+ (x86_64)
- **分发形式**: 解压即用，无需安装
- **模型分离**: 可执行文件不包含模型，首次运行自动下载
- **大小目标**: 可执行文件 <500MB（不含模型）

#### F3.8 错误处理与健壮性
- **异常类型**:
  ```python
  class CosyVoiceError(Exception): pass
  class ModelNotFoundError(CosyVoiceError): pass
  class AudioProcessingError(CosyVoiceError): pass
  class TextTooLongError(CosyVoiceError): pass
  class VoiceCloningError(CosyVoiceError): pass
  ```
- **重试机制**: 
  - 网络下载：3 次重试，指数退避
  - 推理失败：1 次重试，记录日志
- **日志系统**:
  - 日志级别: DEBUG, INFO, WARNING, ERROR
  - 输出位置: `~/.cosyvoice3/logs/` + console
  - 格式: 时间戳、级别、模块名、消息
  - 日志轮转: 单文件最大 10MB，保留最近 5 个

---

## 🎨 API 设计 (API Design)

### 核心 API

```python
from cosyvoice_onnx import CosyVoiceTTS, AudioData, PresetVoice

# 初始化（可选预加载）
tts = CosyVoiceTTS(
    model_dir=None,          # 默认 ~/.cosyvoice3/models
    precision="auto",        # "fp16" | "fp32" | "auto"
    preload=False,           # True 立即加载模型
    num_threads=None,        # None=自动
)

# 基础 TTS
audio: AudioData = await tts.synthesize_async(
    text="你好，世界！",
    speed=1.0,
    volume=1.0,
    emotion="neutral",
    output_format="wav",
)
audio.save("output.wav")

# 流式 TTS（异步生成器）
async for chunk in tts.synthesize_stream(
    text="这是流式输出测试。",
    on_progress=lambda p: print(f"Progress: {p.progress_percent:.1f}%")
):
    # 边生成边播放
    play_audio_chunk(chunk.data, chunk.sample_rate)

# 语音克隆
cloned_audio = await tts.clone_voice_async(
    prompt_audio="speaker.wav",
    prompt_text="Hello, my name is Alice.",
    target_text="今天天气很好。",
    speed=1.0,
)

# 使用预设音色
preset = tts.load_preset("female_soft")
audio = await tts.synthesize_with_preset(
    text="使用预设音色",
    preset=preset,
)

# 高级控制
audio = await tts.synthesize_async(
    text="这是一个<pinyin:ce4 shi4>测试",
    language="zh",           # 显式指定语言
    dialect="canton",        # 粤语
    emotion="happy",
    speed=1.2,
    enable_text_norm=True,   # 文本规范化
)

# 音频后处理
normalized = tts.normalize_volume(audio, target_db=-20)
concatenated = tts.concat_audio([audio1, audio2, audio3])

# 资源管理
await tts.preload_models()   # 预加载
tts.unload_models()           # 卸载释放内存
```

### AudioData 类型

```python
@dataclass
class AudioData:
    data: bytes              # 音频二进制数据
    sample_rate: int         # 采样率
    channels: int            # 声道数
    format: str              # "wav" | "mp3"
    duration_ms: int         # 时长（毫秒）
    
    def save(self, path: str) -> None: ...
    def to_numpy(self) -> np.ndarray: ...
    def to_bytes(self, format: str = "wav") -> bytes: ...
```

---

## 🚀 性能目标 (Performance Targets)

### 基准平台
- **M3 Max** (用户开发环境)
- **Intel i5-10400** (中等 x86 CPU)
- **AMD Ryzen 5 5600X** (中等 x86 CPU)

### 性能指标

| 指标 | M3 Max 目标 | Intel/AMD 目标 | 备注 |
|------|-------------|----------------|------|
| 冷启动时间 | <2s | <5s | 模型加载到 ready |
| 首 chunk 延迟 | <300ms | <500ms | 流式模式 |
| RTF (实时因子) | <0.3 | <0.5 | 越小越好 |
| 生成 1s 音频 | <0.5s | <1s | 非流式模式 |
| 内存占用 | <2GB | <3GB | 模型加载后 |
| 长时间运行 | 无泄漏 | 无泄漏 | 8 小时测试 |

### 精度选择策略
- **默认**: 自动检测 ONNX Runtime 版本，优先 FP16（性能更好）
- **兼容性**: FP16 需要 ONNX Runtime ≥1.18，否则回退 FP32
- **用户可覆盖**: 通过 `precision` 参数强制指定

---

## 🔧 技术实现细节 (Technical Implementation Details)

### 基于 ayousanz 实现
- **继承关系**: 完全基于 `ayousanz/cosy-voice3-onnx` 代码进行封装和改进
- **核心保留**:
  - ONNX 模型加载逻辑
  - 推理管线（LLM → Flow → HiFT）
  - Tokenizer 处理
- **改进点**:
  - 包装为 AsyncIO API
  - 添加流式输出支持
  - 模型自动下载
  - 配置管理系统
  - 完善错误处理和日志
  - PyQt5 信号集成示例

### 异步架构
- **框架**: AsyncIO（Python 3.9+）
- **推理线程**: ONNX 推理在线程池执行，避免阻塞事件循环
- **信号系统**: 提供 PyQt5 信号适配器（参考 VCat `SherpaOnnxTTS` 设计）
  ```python
  class CosyVoiceTTSQt(QObject):
      started = pyqtSignal()
      progress = pyqtSignal(float)  # 0.0 ~ 1.0
      chunk_ready = pyqtSignal(bytes, int)  # data, sample_rate
      finished = pyqtSignal()
      error = pyqtSignal(str)
  ```

### 流式输出实现
- **Chunk 大小**: 可配置，默认 1024 samples
- **缓冲策略**: 双缓冲，一个生成一个播放
- **延迟优化**: 首 chunk 快速返回（牺牲少量质量）

### 文本规范化
- **中文**:
  - 数字 → 汉字读法（"123" → "一百二十三"）
  - 符号 → 文字（"%", "$", "€" 等）
  - 日期时间规范化
  - URL 处理（移除或读出）
- **英文**:
  - 数字 → 单词（"123" → "one hundred twenty-three"）
  - 缩写展开（"Dr." → "Doctor"）

### 依赖管理
```toml
[project]
dependencies = [
    "onnxruntime>=1.18.0",   # 或 onnxruntime-gpu
    "numpy>=1.26.0,<2.0",    # 避免 2.x 兼容性问题
    "soundfile>=0.12.1",
    "librosa>=0.10.2",
    "transformers>=4.51.3",  # Tokenizer
    "scipy>=1.13.1",
    "huggingface_hub>=0.30.0",
    "pyyaml>=6.0",           # 配置文件
    "pydantic>=2.0",         # 数据验证
    "aiofiles>=23.0",        # 异步文件 IO
]

[project.optional-dependencies]
gpu = ["onnxruntime-gpu>=1.18.0"]
audio = ["pydub>=0.25.1"]  # MP3 转换
pyqt = ["PyQt5>=5.15.0"]   # PyQt5 集成
dev = ["pytest", "black", "ruff"]
```

---

## 📦 打包与分发 (Packaging & Distribution)

### PyPI 发布
```bash
pip install cosyvoice-onnx
```

### 开发安装（uv 优先）
```bash
uv pip install -e .
```

### 打包可执行文件

#### PyInstaller 配置
```python
# scripts/build_package.py
# 支持 macOS / Windows / Linux 三平台构建
# 模型文件不打包，首次运行自动下载
# 生成单个可执行文件或目录
```

#### 目标产物
```
cosyvoice-onnx-v1.0-macos-arm64.tar.gz        # M1/M2/M3 Mac
cosyvoice-onnx-v1.0-macos-x86_64.tar.gz       # Intel Mac
cosyvoice-onnx-v1.0-windows-x86_64.zip        # Windows
cosyvoice-onnx-v1.0-linux-x86_64.tar.gz       # Linux
```

---

## 📚 文档需求 (Documentation Requirements)

### README.md
- 项目简介与特性
- 快速开始（3 分钟上手）
- 安装方式（pip / uv / 可执行文件）
- 基础示例代码
- 常见问题 FAQ
- 许可证说明
- 致谢与引用

### docs/API.md
- 完整 API 参考
- 所有类、方法、参数的详细说明
- 类型注解与默认值
- 异常说明
- 示例代码片段

### docs/TUTORIAL.md
- 入门教程
  - 安装与配置
  - 第一个 TTS 程序
  - 语音克隆教程
  - 流式输出使用
- 进阶教程
  - PyQt5 集成指南
  - 自定义配置
  - 性能优化技巧
  - 多语言与方言
- 实战案例
  - 桌面宠物集成（参考 VCat）
  - 聊天机器人语音
  - 有声书生成

### docs/BEST_PRACTICES.md
- 性能优化建议
- 内存管理最佳实践
- 音质优化技巧
- 生产环境部署建议
- 安全性考虑

### docs/TROUBLESHOOTING.md
- 常见错误与解决方案
  - ONNX Runtime 版本错误
  - NumPy 2.x 兼容性
  - 模型下载失败
  - 音质问题
- 日志分析指南
- 性能问题诊断

---

## 🧪 验证计划 (Verification Plan)

### V1 验证
1. **基础 TTS 测试**
   - 运行 `examples/basic_usage.py`
   - 验证中英文文本生成音频
   - 检查输出音频质量
2. **语音克隆测试**
   - 运行 `examples/voice_cloning.py`
   - 使用测试音频文件克隆
   - 对比克隆效果
3. **模型自动下载**
   - 清空模型目录
   - 首次运行，验证自动下载
   - 检查下载进度显示

### V2 验证
1. **流式输出测试**
   - 运行 `examples/streaming_demo.py`
   - 测量首 chunk 延迟（<500ms）
   - 验证边生成边播放
2. **性能基准测试**
   - 在 M3 Max 上测试 RTF
   - 记录内存占用
   - 长时间运行测试（1 小时）

### V3 验证
1. **多语言测试**
   - 9 种语言各生成测试音频
2. **PyQt5 集成测试**
   - 运行 `examples/pyqt5_integration.py`
   - 验证信号机制
3. **跨平台打包测试**
   - 在 macOS / Windows / Linux 构建
   - 测试可执行文件运行

### 用户验收测试
- 集成到 VCat 项目
- 实际使用场景测试
- 用户反馈收集

---

## 🎯 迭代计划 (Iteration Plan)

### V1: 基础 TTS + 语音克隆 (Week 1-2)
**目标**: 可运行的核心功能 demo

**交付物**:
- [x] 项目结构搭建
- [ ] 模型加载与推理
- [ ] 基础同步 API
- [ ] 语音克隆功能
- [ ] 模型自动下载
- [ ] 预设音色库（3-5 个）
- [ ] 基础配置系统
- [ ] 示例代码：`basic_usage.py`, `voice_cloning.py`
- [ ] README 初版

**验收标准**:
- ✅ 能生成中英文语音
- ✅ 能克隆自定义音色
- ✅ 首次运行自动下载模型
- ✅ 代码可运行 demo

---

### V2: 流式输出 (Week 3)
**目标**: 实时交互能力

**交付物**:
- [ ] AsyncIO 架构重构
- [ ] 流式生成器 API
- [ ] 进度回调机制
- [ ] 中断控制
- [ ] PyQt5 信号适配器
- [ ] 示例代码：`streaming_demo.py`, `pyqt5_integration.py`
- [ ] 性能优化（首 chunk 延迟）

**验收标准**:
- ✅ 首 chunk 延迟 <500ms（M3 Max <300ms）
- ✅ 可中断生成
- ✅ 进度回调正常工作
- ✅ PyQt5 示例可运行

---

### V3: 完整功能 + 打包 (Week 4-5)
**目标**: 生产就绪

**交付物**:
- [ ] 多语言支持（9 种）
- [ ] 中文方言支持
- [ ] 发音修复（拼音/音素）
- [ ] 情感与指令控制
- [ ] 音频后处理
- [ ] 完善错误处理
- [ ] 日志系统
- [ ] 跨平台打包脚本
- [ ] 完整文档（API, Tutorial, Best Practices）
- [ ] 10+ 预设音色
- [ ] 所有示例代码
- [ ] 性能基准测试报告

**验收标准**:
- ✅ 所有功能正常工作
- ✅ 跨平台可执行文件可用
- ✅ 文档完整
- ✅ 性能达标
- ✅ 用户可一键安装使用

---

## 🔒 许可证与合规 (License & Compliance)

### 本项目许可证
- **开源协议**: Apache 2.0（与 CosyVoice 官方一致）
- **商业友好**: 允许商业使用

### 上游依赖许可证
- **CosyVoice 官方**: Apache 2.0 ✅
- **ayousanz/cosy-voice3-onnx**: Apache 2.0 ✅
- **ONNX Runtime**: MIT ✅
- **Transformers (HuggingFace)**: Apache 2.0 ✅
- **NumPy, SciPy, Librosa**: BSD ✅

### 免责声明
- 模型仅供研究与合法用途
- 语音克隆需获得原音色所有者授权
- 不对滥用负责

---

## 📊 技术风险与缓解策略 (Risk Mitigation)

| 风险 | 影响 | 缓解策略 |
|------|------|----------|
| ONNX Runtime 兼容性问题 | 高 | 提供 FP16/FP32 两种版本，自动降级 |
| 跨平台性能差异 | 中 | 基于 CPU 架构优化编译选项，提供性能建议 |
| 模型下载失败 | 中 | 重试机制 + 手动下载选项 + 离线安装包 |
| 流式输出延迟过高 | 高 | 优化 chunk 策略，参考官方 bi-streaming 实现 |
| 内存泄漏 | 中 | 定期测试，限制 KV-cache，提供卸载 API |
| 打包文件过大 | 低 | 模型外置，首次下载 |
| 音质不如官方 PyTorch 版 | 中 | 使用官方 ONNX 导出，参数调优 |

---

## 🎓 参考资料 (References)

1. **CosyVoice 3 官方资源**
   - [GitHub](https://github.com/FunAudioLLM/CosyVoice)
   - [论文](https://arxiv.org/pdf/2505.17589)
   - [HuggingFace 模型](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
   - [官方 Demo](https://funaudiollm.github.io/cosyvoice3/)

2. **ONNX 实现**
   - [ayousanz/cosy-voice3-onnx](https://huggingface.co/ayousanz/cosy-voice3-onnx)

3. **VCat 项目参考**
   - 位置: `/Users/songallen/Desktop/VCat`
   - 参考点: PyQt5 集成、Sherpa-ONNX TTS 架构、异步信号设计

---

## ✅ 验收标准总结 (Acceptance Criteria Summary)

### V1 验收
- ✅ 基础 TTS 可生成中英文音频
- ✅ 语音克隆功能正常
- ✅ 模型自动下载无错误
- ✅ 至少 3 个预设音色可用
- ✅ 示例代码可运行

### V2 验收
- ✅ 流式输出首 chunk <500ms
- ✅ 可中断、可进度回调
- ✅ PyQt5 集成示例正常

### V3 验收
- ✅ 9 种语言、18+ 方言支持
- ✅ 发音修复、情感控制工作
- ✅ 跨平台打包成功
- ✅ 文档完整（README + API + Tutorial + Best Practices）
- ✅ 性能达标（见性能目标表）

### 最终验收
- ✅ 集成到 VCat 项目成功
- ✅ 用户可一键安装（`pip install` 或解压运行）
- ✅ 用户反馈满意（傻瓜式操作、性能良好）

---

**备注**: 本规格说明基于 2026-01-15 完成的三轮深度访谈编写，涵盖了所有关键技术细节、功能需求和实现计划。如有变更需求，请更新本文档并同步到 `task.md`。
