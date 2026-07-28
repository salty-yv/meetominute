# MeetOminute

MeetOminute 是一个本地优先的个人会议工作流应用：上传录音，后台顺序完成音频标准化、转写、说话人标记和纪要生成，再由用户校正并导出 Markdown、TXT、Word 或 JSON。

当前为 **v0.1 原型**。网页、SQLite 持久化阶段队列、任务取消与断点续跑、FFmpeg 预处理、FunASR 本地转写、Qwen3.5 本地纪要、逐字稿编辑、说话人姓名映射和四种导出格式已落地。本机 `.env` 已启用全本地处理，录音和逐字稿无需发送到第三方。

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

启动后可从顶部“运行诊断”进入本机环境检查页，集中查看 Python 虚拟环境、数据目录、磁盘、SQLite、FFmpeg、PyTorch/CUDA、FunASR 和 Ollama 状态。页面提供逐项修复建议，并可复制不含 API Key 和会议内容的诊断 JSON；程序接口为 `GET /api/diagnostics`。即使 Ollama 自动启动失败，网页仍会打开，便于从诊断页查看原因和处理建议。

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
MEETOMINUTE_LLM_INPUT_CHAR_BUDGET=6000
MEETOMINUTE_LLM_MAX_TOKENS=4096
```

纪要输入预算会同时约束系统提示、模板和待处理内容。长会议会先分段抽取，
再按预算进行多层归并，避免一次性把全部分段结果塞进模型上下文。

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
- 任务页用四阶段轨道展示录音接收、语音转写、纪要生成和人工审核状态，并可取消任务、从最近断点继续或从头重跑。
- 逐字稿支持说话人姓名映射、文本搜索、按说话人筛选、原音定位、未保存修改提示和 TXT 导出。
- 纪要页按事实类型分区展示，证据时间可直接回听原录音，并提供 Word、Markdown、TXT 和 JSON 导出入口。
- 纪要模板中心支持新建、编辑、复制和删除自定义模板，可控制章节启用、章节名称、额外章节及模型提取重点。
- 会议日历按实际会议日期展示正常、归档和回收站记录，支持按月切换并直接进入会议详情。
- 资料库支持会议归档、只读查看、回收站恢复和经过二次确认的永久删除。
- 备份中心可创建一致性 ZIP 快照、下载到其他磁盘，并以“不覆盖已有会议”的方式合并恢复。
- 运行诊断页会检查完整本地处理栈，并对缺失命令、CUDA 不可用、模型未缓存和磁盘空间不足给出修复建议。
- 桌面端和 390 px 手机端均已完成浏览器截图验证，无横向布局溢出或控制台错误。

界面验收截图保存在 `benchmark-results/ui-home-desktop.png`、`benchmark-results/ui-minutes-desktop.png` 和 `benchmark-results/ui-home-mobile.png`。可用 `scripts/visual_check_ui.js` 重复执行浏览器级布局与交互检查。

## 数据与恢复

默认数据目录为 `data/`：

```text
data/
  meetominute.sqlite3
  backups/
    meetominute-backup-20260727-120000-a1b2c3d4.zip
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
      transcript.txt
      minutes.json
      minutes.md
      minutes.txt
      minutes.docx
      processing.log
```

上传采用 1 MB 分块写入，不会将大文件整体载入内存。任务及 `uploaded → normalized → transcribed → completed` 检查点保存在 SQLite；浏览器关闭不影响后台队列，应用进程退出后再次启动会自动从最近有效断点继续。单个任务即使遇到数据库或磁盘级异常也不会拖停整个后台队列。用户也可在任务页取消处理，已完成的标准化音频或逐字稿不会被删除；“从断点继续”复用这些成果，“从头重跑”才会清理生成文件并重新处理原始录音。FFmpeg 和默认的隔离 FunASR 子进程可即时取消，外部 LLM 请求会在当前 HTTP 请求结束后停止。原始录音与 `transcript_raw.json` 不会被人工编辑覆盖。

纪要模板入口为 `http://127.0.0.1:8000/minutes-templates`，会议日历入口为 `http://127.0.0.1:8000/calendar`，资料库入口为 `http://127.0.0.1:8000/archive`，备份中心为 `http://127.0.0.1:8000/backups`：

- 创建会议和重新生成纪要时均可选择模板；生成结果会保存模板快照，因此后续修改模板不会改变已有纪要的章节和导出格式。
- 自定义模板可以关闭不需要的内置章节、修改章节名称，并通过“章节名称 | 提取要求”增加最多 8 个带证据时间的列表章节。
- 模板要求会同时进入分段事实抽取和最终合并提示词，但不能覆盖“不推测负责人、期限、决定或实验结论”等事实约束。
- 日历按填写的会议日期归类，支持月份选择和前后月份切换；归档及回收站记录不会从原日期消失。
- 归档会议会从工作台隐藏并进入只读模式，取消归档后可继续编辑。
- 移入回收站不会立即删除文件；只有在回收站中再次确认“永久删除”才会删除数据库记录和对应的精确会议目录。
- 完整备份包含纪要模板、活跃、归档及回收站会议、任务历史和会议目录，不包含模型缓存或外部 LLM API Key。
- 创建或恢复备份时会进入独占维护状态，新的编辑和后台文件写入会等待或被友好拦截；备份文件清单从同一份 SQLite 快照读取。
- 恢复采用合并导入，相同会议 ID 或目录名会跳过，不覆盖当前数据；备份中的未完成任务会恢复为已取消状态，可按现有断点手动继续。
- SQLite 数据库使用显式版本迁移，当前 schema 为 v4；旧版数据库在启动时自动补齐任务表、会议生命周期和纪要模板字段。

保存在 `data/backups/` 的副本仍与项目位于同一磁盘。需要防范硬盘损坏时，请下载 ZIP 并复制到另一块磁盘或可信的加密存储位置。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest
```

仓库中的 GitHub Actions 会在 Windows + Python 3.11 环境自动执行测试、
Python 编译检查和浏览器 JavaScript 语法检查。

硬件评估和当前实施状态见 `docs/hardware-assessment.md`。
