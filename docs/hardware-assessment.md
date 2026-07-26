# 阶段 0：本机硬件与运行环境评估

评估日期：2026-07-26

## 已确认配置

| 项目 | 结果 |
| --- | --- |
| CPU | Intel Core i7-12700H，20 个逻辑线程 |
| 内存 | 15.7 GB |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| 显存 | 6 GB |
| 系统 | 64 位 Windows，系统构建号 10.0.26200 |
| Python | 项目虚拟环境 Python 3.11.9 |
| FFmpeg | 7.1 full build，包含 CUDA/NVENC 相关支持 |

## CUDA 与依赖验证

项目虚拟环境已验证以下组合：

| 组件 | 版本或结果 |
| --- | --- |
| PyTorch | 2.6.0+cu124 |
| Torchaudio | 2.6.0+cu124 |
| CUDA runtime | 12.4 |
| FunASR | 1.3.29 |
| ModelScope | 1.38.1 |
| `torch.cuda.is_available()` | `True` |
| `pip check` | 无依赖冲突 |

CUDA 版 PyTorch 来自同机、同为 CPython 3.11 的 `E:\rec\.venv-rec-eval`，复制时未修改源环境；匹配版本的 Torchaudio 及 FunASR 依赖安装在项目 `.venv` 内。四个 FunASR 模型缓存位于 `data/models/`，共约 2.07 GB。

## AISHELL-4 会议基准

样本为 AISHELL-4 会话 `20200706_L_R001S01C01` 的 15 分钟片段，16 kHz、16-bit、单声道，人工标注含 7 位说话人和 242 个话语片段。模型组合为 Paraformer-zh + FSMN-VAD + CT-Punc + CAM++。

| 指标 | 冷启动（首次下载） | 缓存后最终运行 |
| --- | ---: | ---: |
| 录音时长 | 900 秒 | 900 秒 |
| 处理耗时 | 308.255 秒 | 71.775 秒 |
| 实时倍率 | 2.92× | 12.54× |
| RTF | 0.3425 | 0.0797 |
| 峰值显存 | 2534.6 MB | 2534.6 MB |
| 预测 / 标注说话人数 | 8 / 7 | 7 / 7 |
| 识别片段数 | 302 | 302 |
| 近似字符错误率 | 37.83% | 37.83% |

最终运行把用户填写的预计发言人数作为 CAM++ 的聚类人数，因此与 7 人标注一致。字符错误率按 TextGrid 话语的时间顺序直接拼接计算；样本包含重叠说话，单路 ASR 不能同时还原所有重叠内容，因此该数值只作为当前实现的相对基线。

## Ollama / Qwen3.5 纪要基准

复用现有 `E:\OllamaModels` 中的 Qwen3.5 9.2B Q4_K_M 权重，并创建了 `meetominute-qwen35-9b` 专用模型配置。该模型采用 8192 token 上下文、关闭思考模式、固定温度为 0，并通过 Ollama 的 OpenAI 兼容 JSON 模式生成结构化纪要。

| 指标 | 首轮 | 优化后最终运行 |
| --- | ---: | ---: |
| 输入 | 302 段 / 3296 个识别字符 | 同左 |
| 处理耗时 | 390.669 秒 | 200.625 秒 |
| 模型调用次数 | 4 | 3 |
| 输出 tokens | 5172 | 2621 |
| 有效输出速度 | 13.24 tokens/s | 13.06 tokens/s |
| 证据时间字段 | 8 | 20 |
| 含有效时间戳 | 8 | 20 |
| RTX 驻留显存 | 约 4.15 GB | 约 4.15 GB |
| CPU / GPU 模型分配 | 33% / 67% | 33% / 67% |

最终纪要包含摘要、成员进展、建议、待办、未决问题和后续跟进；对于没有逐字稿依据的实验结果和已确认决定保持空数组。短会议测试也加入了强制主题摘要和纯逐字稿兜底，避免过度保守时只返回“未明确”。

## 应用级验证与显存隔离

首次把 FunASR 与 Ollama 放在同一长期运行进程中时，Windows 上出现 Ollama CUDA 子进程初始化失败。最终实现将 FunASR 放入独立子进程，并在每场任务前后主动卸载 Ollama 模型：

1. 卸载驻留的 Qwen，释放 RTX 显存。
2. 在独立子进程中运行 FunASR；子进程退出时彻底释放 PyTorch CUDA 上下文。
3. 加载 Qwen 生成纪要。
4. 生成结束后卸载 Qwen，显存回落到约 17 MB。

最终专用端口 `11435` 上的 60 秒冒烟测试，从网页上传到全部导出耗时 122.212 秒；FunASR 和 Ollama 后端均为真实模型，Markdown、TXT、DOCX 和 JSON 全部成功。任务完成后 `/api/ps` 返回空模型列表，RTX 上无模型计算进程残留。

结果保存在：

- `benchmark-results/funasr-aishell4-15min-final.json`
- `benchmark-results/ollama-qwen35-9b-aishell4-15min-minutes-final.json`
- `benchmark-results/funasr-ollama-app-smoke-dedicated.json`

## 当前结论

这台设备适合“应用本地运行、模型顺序加载”的方案。6 GB 显存可稳定运行 FunASR，并可通过 CPU/GPU 混合推理运行 Qwen3.5 9B。单任务队列、FunASR 子进程隔离和任务前后卸载 Ollama 模型必须继续保留。当前 `.env` 已启用 FunASR + Ollama 全本地处理。

下一阶段应使用一段用户自己的典型会议录音验证普通话/方言、中英混合、专业词表、说话人分离和纪要事实约束。AISHELL-4 基准中的近似字符错误率仍说明人工校正逐字稿是正式导出前的重要步骤。
