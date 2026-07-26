# MeetOminute

MeetOminute 是一个本地优先的个人会议工作流应用：上传录音，后台顺序完成音频标准化、转写、说话人标记和纪要生成，再由用户校正并导出 Markdown、TXT、Word 或 JSON。

当前为 **v0.1 原型**。网页、SQLite 持久化、单任务队列、FFmpeg 预处理、FunASR 本地转写、Qwen3.5 本地纪要、逐字稿编辑、说话人姓名映射和四种导出格式已落地。本机 `.env` 已启用全本地处理，录音和逐字稿无需发送到第三方。

## 已配置的项目环境

项目使用独立虚拟环境：

```text
E:\meetominute\.venv
Python 3.11.9
PyTorch 2.6.0+cu124
Torchaudio 2.6.0+cu124
FunASR 1.3.29
Ollama 0.30.10
Qwen3.5 9.2B Q4_K_M
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
```

CUDA 版 PyTorch 从同为 CPython 3.11 的 `E:\rec\.venv-rec-eval` 复制到项目环境，源环境未被修改；匹配版本的 Torchaudio 和 FunASR 依赖随后安装到本项目。`pip check` 已通过，CUDA 12.4 与 RTX 3060 均已实际验证。

直接双击 `start.bat`，或在 PowerShell 中运行：

```powershell
.\.venv\Scripts\python.exe -m app.launcher
```

应用只监听 `http://127.0.0.1:8000`，启动后会自动打开浏览器。按 `Ctrl+C` 停止。若本地纪要后端为 Ollama，启动器会在专用端口 `11435` 自动启动服务，并强制屏蔽 Intel Vulkan 核显、优先使用 RTX/CUDA。

## 从零安装

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

要求系统能在命令行调用 `ffmpeg` 和 `ffprobe`。

## 配置真实后端

复制 `.env.example` 为 `.env`，再按服务填写。三种处理模式的映射：

| 页面模式 | 转写后端 | 纪要后端 |
| --- | --- | --- |
| 本地 | `MEETOMINUTE_LOCAL_TRANSCRIBER` | `MEETOMINUTE_LOCAL_LLM` |
| 混合 | `MEETOMINUTE_LOCAL_TRANSCRIBER` | `MEETOMINUTE_CLOUD_LLM` |
| 云端 | `MEETOMINUTE_CLOUD_TRANSCRIBER` | `MEETOMINUTE_CLOUD_LLM` |

当前代码支持 `mock`、FunASR、独立 Ollama 后端和 OpenAI 兼容接口。以云端兼容接口为例：

```dotenv
MEETOMINUTE_CLOUD_TRANSCRIBER=openai
MEETOMINUTE_CLOUD_LLM=openai
MEETOMINUTE_OPENAI_BASE_URL=https://your-provider.example/v1
MEETOMINUTE_OPENAI_API_KEY=...
MEETOMINUTE_TRANSCRIBE_MODEL=your-transcription-model
MEETOMINUTE_LLM_MODEL=your-chat-model
```

原始录音只会发送给所选转写后端；纪要后端只接收校正后的逐字稿。本地模式使用单独的 Ollama 配置，不会覆盖云端接口设置。

### 从界面接入外部 LLM

启动应用后，点击顶部“外部 LLM”或“模型设置”，打开：

```text
http://127.0.0.1:8000/settings/external-llm
```

填写兼容 OpenAI Chat Completions 协议的 API Base URL、模型名称和 API Key，可先点击“测试连接”，确认后保存。配置保存后立即生效，无需重启：

- “全本地”继续使用 Ollama，不调用外部接口。
- “混合模式”使用本地 FunASR 转写，只把校正后的逐字稿发送给外部 LLM 生成纪要。
- “云端模式”使用云端转写配置，并使用这里配置的外部 LLM 生成纪要。

API Key 不会写入会议文件、模板或日志。Windows 下使用当前用户的 DPAPI 加密后保存在 `data/external-llm.json`，设置页和 `GET /api/settings/external-llm` 只返回“是否已配置”，不会返回密钥。未通过界面保存配置时，原有 `MEETOMINUTE_OPENAI_*` 环境变量仍可作为兼容配置来源。

### FunASR 本地转写

本机环境已经配置完成。若在另一台 RTX/NVIDIA 设备上从零重建，先安装匹配的 PyTorch CUDA wheel，再安装项目的本地转写扩展：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -e ".[dev,local-asr]"
```

然后在 `.env` 中启用：

```dotenv
MEETOMINUTE_LOCAL_TRANSCRIBER=funasr
MEETOMINUTE_FUNASR_DEVICE=auto
MEETOMINUTE_FUNASR_ISOLATE_PROCESS=true
```

默认组合为 Paraformer-zh + FSMN-VAD + CT-Punc + CAM++，可同时生成中文转写、标点、时间戳和匿名说话人标签。“预计发言人数”会作为聚类人数传给 CAM++。模型首次使用时下载到 `data/models/`，当前缓存约 2.07 GB。FunASR 默认在独立子进程运行，结束时整个 CUDA 上下文一并退出，避免和随后加载的 Ollama 模型争抢 6 GB 显存。

使用项目附带的 15 分钟 AISHELL-4 样本执行基准：

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_funasr.py `
  samples\aishell4_zh_meeting_15min.wav `
  --textgrid samples\aishell4_zh_meeting_source.TextGrid `
  --clip-offset 120 `
  --expected-speakers 7 `
  --output benchmark-results\funasr-aishell4-15min-final.json
```

本机最终基准结果：

| 指标 | 结果 |
| --- | --- |
| 音频 | AISHELL-4，普通话多人会议，15 分钟 |
| 处理耗时 | 71.775 秒（包含缓存模型加载） |
| 速度 | 12.54 倍实时，RTF 0.0797 |
| 峰值 GPU 显存 | 2534.6 MB |
| 说话人数 | 预测 7 / 标注 7 |
| 转写片段 | 302 |
| 近似字符错误率 | 37.83% |

字符错误率是把带重叠说话的多人会议标注按时间拼接后计算的粗略值，不能与干净、单说话人的标准测试集直接比较。完整指标和逐段结果见 `benchmark-results/funasr-aishell4-15min-final.json`。

### Ollama 本地纪要

本机复用 `E:\OllamaModels` 中已有的 Qwen3.5 9.2B Q4_K_M 权重，并通过 `ollama/Modelfile.meetominute` 创建专用模型名；新模型共享原权重，不会重复占用 5.8 GB：

```powershell
ollama create meetominute-qwen35-9b -f ollama\Modelfile.meetominute
```

本机配置：

```dotenv
MEETOMINUTE_LOCAL_LLM=ollama
MEETOMINUTE_OLLAMA_BASE_URL=http://127.0.0.1:11435/v1
MEETOMINUTE_OLLAMA_MODEL=meetominute-qwen35-9b
MEETOMINUTE_OLLAMA_REASONING_EFFORT=none
MEETOMINUTE_LLM_CHUNK_CHARS=6000
MEETOMINUTE_LLM_MAX_TOKENS=4096
```

15 分钟转写稿的最终纪要基准：

| 指标 | 结果 |
| --- | --- |
| 模型运行大小 | 6.15 GB，其中约 4.15 GB 驻留 RTX 显存 |
| 上下文 | 8192 tokens |
| 处理耗时 | 200.625 秒 |
| 模型调用 | 3 次 |
| 输出 tokens | 2621 |
| 有效输出速度 | 13.06 tokens/s |
| 证据时间戳 | 20 / 20 有效 |

完整纪要和指标见 `benchmark-results/ollama-qwen35-9b-aishell4-15min-minutes-final.json`。专用端口 `11435` 上的完整应用级测试用 122.212 秒处理 60 秒录音，已覆盖上传、FunASR、Ollama、SQLite、模型释放和全部导出格式，见 `benchmark-results/funasr-ollama-app-smoke-dedicated.json`。

## 界面工作流

当前网页已升级为完整的本地会议工作台：

- 首页集中展示本地处理栈、隐私状态、任务创建和可搜索的会议历史。
- 任务页用四阶段轨道展示录音接收、语音转写、纪要生成和人工审核状态。
- 逐字稿支持说话人姓名映射、文本搜索、按说话人筛选、原音定位和未保存修改提示。
- 纪要页按事实类型分区展示，证据时间可直接回听原录音，并提供 Word、Markdown、TXT 和 JSON 导出入口。
- 桌面端和 390 px 手机端均已完成浏览器截图验证，无横向布局溢出或控制台错误。

界面验收截图保存在 `benchmark-results/ui-home-desktop.png`、`benchmark-results/ui-minutes-desktop.png` 和 `benchmark-results/ui-home-mobile.png`。可用 `scripts/visual_check_ui.js` 重复执行浏览器级布局与交互检查。

## 数据与恢复

默认数据目录为 `data/`：

```text
data/
  meetominute.sqlite3
  meetings/
    2026-07-26_课题组周会_ab12cd34/
      original.m4a
      normalized.wav
      meeting.json
      transcript_raw.json
      transcript_edited.json
      speakers.json
      glossary.txt
      transcript.md
      minutes.json
      minutes.md
      minutes.txt
      minutes.docx
      processing.log
```

上传采用 1 MB 分块写入，不会将大文件整体载入内存。浏览器关闭不影响后台队列；若应用进程在处理时退出，重启后任务会标记为失败，可点击“重新处理”。原始录音与 `transcript_raw.json` 不会被人工编辑覆盖。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest
```

硬件评估和当前实施状态见 `docs/hardware-assessment.md`。
