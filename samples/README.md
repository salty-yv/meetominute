# 中文会议测试样本

## `aishell4_zh_meeting_15min.wav`

- 来源：AISHELL-4 普通话多人会议语料
- 原始会话：`20200706_L_R001S01C01.flac`
- 截取范围：原始录音 `00:02:00` 至 `00:17:00`
- 样本时长：15 分钟
- 输出格式：16 kHz、16-bit、单声道 PCM WAV
- 实际说话人数：7 人（依据配套 RTTM 标注）
- 标注片段数：242 段
- 使用目的：MeetOminute 本地转写、时间戳和说话人区分基准
- 官方数据页：https://www.openslr.org/111/
- 文件镜像：https://huggingface.co/datasets/AISHELL/AISHELL-4
- 许可：CC BY-SA 4.0（以 OpenSLR 官方数据页标示为准）

AISHELL-4 是真实录制的普通话会议语料，每场包含 4–8 名说话人，并包含自然停顿、重叠说话、快速轮换和会议室噪声。使用或再分发本样本时，请保留本说明并引用 AISHELL-4：

> Yihui Fu et al. “AISHELL-4: An Open Source Dataset for Speech Enhancement, Separation, Recognition and Speaker Diarization in Conference Scenario.” Interspeech 2021.

配套文件：

- `aishell4_zh_meeting_source.TextGrid`：原始完整会话人工转写与时间标注。
- `aishell4_zh_meeting_source.rttm`：原始完整会话说话人活动标注。
- `aishell4_zh_meeting_15min.json`：样本来源、音频属性和标注统计。

配套标注仍使用原始会话时间轴；本音频的本地时间 `00:00:00` 对应标注中的 `00:02:00`。
