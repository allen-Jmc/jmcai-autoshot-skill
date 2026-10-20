# 工作流契约与验证状态

记录日期：2026-10-14。"已验证"指在本机用真实运行实测过；"未验证"就是没试过或没查到。

算力路径：`scripts/jmcai.py`（本技能自带的 Bridge 客户端）→ JMCAI Comfypet 桌面端 `127.0.0.1:32100` → 桌面端绑定好的算力目标（本机 ComfyUI 或外部 4090 服务器）。

## Bridge 契约（bridge_version 1.2.0）

| 用途 | 接口 |
|---|---|
| 桌面端状态 | `GET /api/v1/health` |
| 已启用工作流 + 暴露的 alias schema | `GET /api/v1/workflows` |
| 提交运行 | `POST /api/v1/workflows/<id>/runs`，体 `{"args": {...}, "target_id": "..."}` → **HTTP 202**，返回 `{"id": "<run_id>", "status": "queued", ...}` |
| 查询单次运行 | `GET /api/v1/runs/<run_id>` |
| 历史运行 | `GET /api/v1/workflows/<id>/history` |
| 上传资产 | `POST /api/v1/uploads`（本机 bridge 用不到） |

**实测要点：**

- **提交成功是 202，不是 200。** 只看 200 会把每次提交都当成失败（踩过这个坑）。
- **本机 bridge（127.0.0.1）时，`image` / `audio` / `video` / `file` 类参数直接传本机绝对路径**，桌面端自己把文件送到算力目标，不需要上传、也不要用 `upload:` 前缀。
- **`outputs[].path` 已经是本机绝对路径**（桌面端自动从算力目标取回来了），直接读即可。
- 运行状态：`queued → running → success | error`，终态是 `success` / `error`。
- Bridge 返回的字段是 **snake_case**（`prompt_id`、`target_id`、`media_kind`、`file_name`）；桌面端本地 run 文件里是 camelCase（`promptId`）。两边都要兼容。

## 四套工作流

### `qwen3-tts-_api` — Qwen3-TTS 声音克隆（默认）

暴露参数：`audio_4`(参考音频，必填，本机路径)、`text`(必填)、`seed`(默认 42)、`language_2`(默认 Auto)、`ref_audio_max_seconds`(默认 30，上限 120)。
**没有 `instruct`** —— 所以方括号情绪标签会被原样念出来，clone 模式必须剥掉。
产物：`_01.txt`（参考音频的识别文字）+ `_02.flac`（**24 kHz 单声道**）。
实测：同一段文字 + 同一参考音频 + `seed=42`，多次运行输出**逐字节相同**（222,098 字节 / 7.6 秒），完全确定。冷启动首次约 57 秒，模型常驻后约 0.3 秒。

### `qwen3-tts-_api-2` — Qwen3-TTS 预设音色

暴露参数：`text`(必填)、`speaker`(必填，9 个：Vivian / Serena / Uncle_Fu / Dylan / Eric / Ryan / Aiden / Ono_Anna / Sohee，默认 Vivian)、`language`(默认 Chinese)、`seed`(默认 42)、`instruct`(默认空)。
`instruct` 接受情绪/语气指令 → 情绪标签在**这一套**里有效。

### `qwen3-tts-voicedesign-_api` — Qwen3-TTS 声音设计

暴露参数：`text`(必填)、`instruct`(必填，音色描述)、`language`(默认 Chinese)、`seed`(必填)。
用一句自然语言描述造音色，例如"低沉、语速慢的中年男声"。

### `qwen3-asr-_api` — Qwen3-ASR 字级时间戳

暴露参数：`audio_1`(必填)、`language`(30 个候选，默认 auto)。
产物：**两个内容相同、只有换行符不同（LF / CRLF）的 `.txt`**。每行一条 `0.48-0.64: 徒`——**字级**、单位秒、**不带标点**。
实测：一段 8.5 秒中文，59931 ms 返回；解析出 28 个字，首条 `{"text":"徒","begin":0.48,"end":0.64}`。
注意：字与字首尾相接，说话中间的停顿被算进前一个字里 → **停顿必须用音频能量检测，不能从字间隙找**。
识别会把中文数字写成阿拉伯数字（"三十秒"→"30秒"），比较文字前要统一写法（`asr.norm_text`）。

## 已知的坑

- **桌面端等远端输出有上限（实测约 180 秒）**，超时会把这次运行标成 `error`（`error_message` 形如"等待 ComfyUI 输出超时。"），但**任务在目标机上还在跑**。数字人这种重任务要 4–5 分钟，所以 `run_avatar.py` 遇到这种超时会用运行记录里的 `promptId` 直接去目标 ComfyUI 取结果（见下）。
- **算力目标上节点校验失败时，桌面端仍报 success。** 例如枚举值不在候选列表里，ComfyUI 打 `Failed to validate prompt … Value not in list`、把输出整个丢掉，但桌面端只看"有没有拿到输出"，于是报成功、`outputs` 只有个别的文本节点。表现是"成功但没有视频"。**防线**：`common.validate_args()` 提交前拿桌面端的实时 schema 核一遍枚举值。
- **工作流的 `aspect_ratio` 候选值必须和远端节点逐字一致。** 桌面端的 schema 是从远端节点探测出来的，远端节点实际候选是 8 个：`1:1 (Square)`、`2:3 (Portrait Photo)`、`3:2 (Photo)`、`3:4 (Portrait Standard)`、`4:3 (Standard)`、`9:16 (Portrait Widescreen)`、`16:9 (Widescreen)`、`21:9 (Ultrawide)`。

## 直接问算力目标（桌面端等不及时的兜底）

目标地址在 `%APPDATA%\JMCAI\workflows\targets.json`（`url` 字段）。用运行记录里的 `prompt_id`：

- `GET <target>/history/<prompt_id>` → `{prompt_id: {status: {status_str: success|error}, outputs: {...}}}`
- 输出在 `outputs.<node>.gifs / images / audio / videos` 里，每项有 `filename` / `subfolder` / `type`
- `GET <target>/view?filename=…&subfolder=…&type=output` 下载文件

`scripts/jmcai.py` 里的 `target_url()` / `comfy_history()` / `comfy_outputs()` / `comfy_download()` 就是这几个。

## 未验证

- 预设音色、声音设计两套的实际听感（没有逐个试听过）。
- `ref_audio_max_seconds` 改成非 30 的效果。
- 桌面端 180 秒上限是否可配置（在应用里没找到对应设置项）。
