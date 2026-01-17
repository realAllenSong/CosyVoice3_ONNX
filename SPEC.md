# CosyVoice3 ONNX FastAPI 服务规格说明书

## 项目概述

### 目标
在 Mac M3 Max (CPU only) 上部署 CosyVoice3 0.5B ONNX 版本的本地 FastAPI HTTP API 服务，用于后续集成 PyQt5 桌面宠物应用。

### 当前问题
用户运行现有代码时遇到以下问题：
1. **乱读/胡说** - 输出与输入文本不匹配
2. **前后空白很长** - 音频开头结尾有大段沉默
3. **音质差/电流声** - 有电流音、卡带效果

### 根因分析

| 问题 | 根因 | 解决方案 |
|------|------|----------|
| 乱读/胡说 | 缺少 `You are a helpful assistant.<\|endofprompt\|>` 前缀 | 修改 engine.py 添加前缀 |
| 前后空白 | FP16 精度问题 | 强制使用 FP32 |
| 音质差 | 参考音频采样率不正确 | 验证/自动重采样到 24kHz |

---

## 技术规格

### 硬件环境
- **设备**: Mac M3 Max
- **内存**: 16-32GB
- **计算**: CPU only (不使用 MPS)

### 软件栈
- Python 3.10+
- ONNX Runtime 1.17.x (FP32 mode)
- FastAPI + Uvicorn
- 现有 cosyvoice_onnx 库

### 模型配置
- **来源**: `ayousanz/cosy-voice3-onnx` (HuggingFace)
- **精度**: 强制 FP32 (禁用 FP16 自动检测)
- **加载策略**: 懒加载 (内存优化)
- **线程数**: 自动检测

---

## 功能需求

### 核心功能

#### 1. 基础 TTS
- 文本转语音
- 支持预设声音
- 支持速度/音量调节
- 支持情绪表达 (happy, sad, angry, calm, excited, neutral)

#### 2. 声音克隆 (Zero-shot)
- 上传参考音频 + 参考文本
- 生成目标文本的语音
- 参考音频自动重采样到 24kHz

#### 3. 流式输出
- 边生成边返回音频块
- SSE (Server-Sent Events) 实现
- 支持进度回调

### 语言支持
- 中文 (zh)
- 英文 (en)
- 日语 (ja)
- 韩语 (ko)
- 德语 (de)
- 法语 (fr)
- 意大利语 (it)
- 西班牙语 (es)
- 俄语 (ru)

### 响应格式
- WAV (默认)
- MP3 (可选)
- Base64 编码 JSON

---

## API 设计

### 基础信息
- **Host**: `127.0.0.1` (仅本地)
- **Port**: `8000` (可配置)
- **协议**: HTTP/1.1

### 端点列表

#### `GET /health`
健康检查
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

#### `GET /presets`
获取预设声音列表
```json
{
  "presets": [
    {
      "name": "trump",
      "language": "en",
      "gender": "male",
      "description": "..."
    }
  ]
}
```

#### `POST /tts`
基础文本转语音
```json
// Request
{
  "text": "你好世界",
  "preset": "zh_female_1",  // 可选，使用预设
  "speed": 1.0,             // 0.5-2.0
  "volume": 1.0,            // 0.0-2.0
  "emotion": "neutral",     // happy/sad/angry/calm/excited/neutral
  "format": "wav",          // wav/mp3/base64
  "language": "auto"        // auto/zh/en/ja/ko/...
}

// Response (format=wav)
audio/wav binary

// Response (format=base64)
{
  "audio": "base64_encoded_data",
  "sample_rate": 24000,
  "duration_ms": 1234,
  "format": "wav"
}
```

#### `POST /clone`
声音克隆
```json
// Request (multipart/form-data)
{
  "prompt_audio": <file>,     // 参考音频文件
  "prompt_text": "参考文本",   // 参考音频对应的文本
  "target_text": "目标文本",   // 要合成的文本
  "speed": 1.0,
  "volume": 1.0,
  "format": "wav"
}

// Response
同 /tts
```

#### `POST /stream`
流式输出
```json
// Request
{
  "text": "...",
  "preset": "...",
  // 其他参数同 /tts
}

// Response: SSE stream
event: audio_chunk
data: {"chunk": "base64_audio", "index": 0, "is_final": false}

event: audio_chunk
data: {"chunk": "base64_audio", "index": 1, "is_final": true}

event: done
data: {"total_chunks": 2, "duration_ms": 2345}
```

#### `POST /validate_audio`
验证参考音频
```json
// Request (multipart/form-data)
{
  "audio": <file>
}

// Response
{
  "valid": true,
  "sample_rate": 24000,
  "duration_seconds": 5.2,
  "channels": 1,
  "warnings": ["Audio will be resampled from 44100Hz to 24000Hz"]
}
```

### 错误响应格式
```json
{
  "error": {
    "code": "INVALID_AUDIO",
    "message": "参考音频格式不支持",
    "details": {
      "supported_formats": ["wav", "mp3", "flac"],
      "received_format": "ogg"
    }
  }
}
```

### 错误码定义
| 代码 | 说明 |
|------|------|
| `INVALID_TEXT` | 文本为空或过长 |
| `INVALID_AUDIO` | 音频格式不支持或损坏 |
| `AUDIO_TOO_SHORT` | 参考音频太短 (<1秒) |
| `AUDIO_TOO_LONG` | 参考音频太长 (>30秒) |
| `PRESET_NOT_FOUND` | 预设声音不存在 |
| `MODEL_ERROR` | 模型推理错误 |
| `INTERNAL_ERROR` | 内部错误 |

---

## 代码修改清单

### 1. engine.py - 添加 endofprompt 前缀
```python
# 位置: src/cosyvoice_onnx/engine.py
# 修改: _build_llm_inputs() 方法

# 在 prompt_text 前添加系统提示前缀
SYSTEM_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"

def _build_llm_inputs(self, text, prompt_text, ...):
    # 添加前缀
    full_prompt_text = SYSTEM_PROMPT_PREFIX + prompt_text
    # ... 继续现有逻辑
```

### 2. api.py - 强制 FP32
```python
# 位置: src/cosyvoice_onnx/api.py
# 修改: CosyVoiceTTS.__init__()

def __init__(self, ...):
    # 强制使用 FP32，忽略 auto 检测
    self.precision = "fp32"
    # ...
```

### 3. 新增 server.py - FastAPI 服务
```
位置: src/cosyvoice_onnx/server.py (新文件)
内容: 完整的 FastAPI 应用实现
```

### 4. 新增 audio_validator.py - 音频验证
```
位置: src/cosyvoice_onnx/audio_validator.py (新文件)
内容: 参考音频验证和重采样逻辑
```

---

## 文件结构

```
CosyVoice3_ONNX/
├── src/cosyvoice_onnx/
│   ├── __init__.py
│   ├── api.py              # [修改] 强制 FP32
│   ├── engine.py           # [修改] 添加 endofprompt 前缀
│   ├── server.py           # [新增] FastAPI 服务
│   ├── audio_validator.py  # [新增] 音频验证
│   └── ...
├── tests/
│   ├── test_quality.py     # [新增] 质量测试脚本
│   ├── test_api.py         # [新增] API 端点测试
│   └── test_streaming.py   # [新增] 流式输出测试
├── run_server.py           # [新增] 服务启动脚本
├── config.json             # [修改] 添加服务器配置
├── SPEC.md                 # 本规格说明
├── task_plan.md            # 任务计划
└── notes.md                # 分析笔记
```

---

## 实施计划

### 阶段 1: 质量修复 (优先级: 最高)
1. 修改 engine.py 添加 `<|endofprompt|>` 前缀
2. 修改 api.py 强制 FP32 精度
3. 创建 audio_validator.py 实现参考音频验证
4. 创建 test_quality.py 验证修复效果

**验收标准**: 运行 test_quality.py，输出音频无乱读、无长空白、无电流声

### 阶段 2: FastAPI 服务
1. 创建 server.py 实现所有 API 端点
2. 实现 SSE 流式输出
3. 添加完整的错误处理和日志
4. 创建 run_server.py 启动脚本

**验收标准**: 所有 API 端点可用，返回正确格式

### 阶段 3: 测试脚本
1. 创建 test_api.py 端点测试
2. 创建 test_streaming.py 流式测试
3. 添加性能基准测试

**验收标准**: 所有测试通过

---

## 测试计划

### 质量测试 (test_quality.py)
```python
# 测试用例
1. 中文短句: "你好，世界" - 验证无乱读
2. 中文长句: 100字段落 - 验证无重复
3. 英文句子: "Hello, world" - 验证多语言
4. 情绪测试: 同一句话不同情绪 - 验证情绪变化
5. 声音克隆: 参考音频+文本 - 验证克隆效果
```

### API 测试 (test_api.py)
```python
# 测试用例
1. GET /health - 健康检查
2. GET /presets - 预设列表
3. POST /tts - 基础 TTS
4. POST /clone - 声音克隆
5. POST /stream - 流式输出
6. 错误处理 - 各种错误场景
```

### 性能测试
```python
# 基准指标
- 首次响应延迟 (首块音频返回时间)
- 完整生成时间
- RTF (Real-Time Factor)
- 内存占用峰值
```

---

## 配置示例

### config.json 更新
```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8000,
    "workers": 1,
    "log_level": "INFO"
  },
  "model": {
    "precision": "fp32",
    "preload": false,
    "num_threads": 0
  },
  "audio": {
    "target_sample_rate": 24000,
    "auto_resample": true,
    "min_duration_seconds": 1.0,
    "max_duration_seconds": 30.0
  },
  "generation": {
    "default_speed": 1.0,
    "default_volume": 1.0,
    "sampling_k": 50,
    "max_tokens": 1000
  }
}
```

---

## 启动命令

```bash
# 开发模式
python run_server.py --host 127.0.0.1 --port 8000 --reload

# 生产模式
python run_server.py --host 127.0.0.1 --port 8000 --workers 1

# 运行测试
python -m pytest tests/ -v
```

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| FP32 导致推理变慢 | 中 | 使用流式输出减少感知延迟 |
| 内存不足 | 高 | 严格使用懒加载策略 |
| 参考音频质量差 | 中 | 添加验证和警告提示 |
| 长文本生成不稳定 | 低 | 添加最大 token 限制 |

---

## PyQt5 集成指南 (Future)

### 推荐调用方式
```python
import requests

def speak(text, preset="zh_female_1", emotion="neutral"):
    response = requests.post(
        "http://127.0.0.1:8000/tts",
        json={
            "text": text,
            "preset": preset,
            "emotion": emotion,
            "format": "wav"
        }
    )
    if response.status_code == 200:
        # 播放音频
        play_audio(response.content)
    else:
        # 处理错误
        error = response.json()["error"]
        handle_error(error)
```

### 流式播放
```python
import sseclient

def speak_streaming(text, preset="zh_female_1"):
    response = requests.post(
        "http://127.0.0.1:8000/stream",
        json={"text": text, "preset": preset},
        stream=True
    )
    client = sseclient.SSEClient(response)
    for event in client.events():
        if event.event == "audio_chunk":
            data = json.loads(event.data)
            play_chunk(base64.b64decode(data["chunk"]))
```

---

## 版本信息

- **规格版本**: 1.0.0
- **创建日期**: 2026-01-16
- **状态**: 待实施
