# JMCAI AutoShot：全自动数字人口播一条龙

**简体中文** | [English](README.md)

[![Comfypet 桌面端](https://img.shields.io/badge/前置依赖-下载%20JMCAI%20Comfypet%20桌面端-3DDC84)](https://github.com/allen-Jmc/comfypet-jmcai-Dist/releases/latest)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

把一段文案做成一条可以直接发布的数字人口播成片：用你自己的声音配音、让一张照片开口说话并带手势，再配上动态包装、字幕、卡片与音效，全自动一条龙出片。

**纯本地 / 私有化运行**：基于本地 **JMCAI Comfypet 桌面端 Workflow Bridge**（`127.0.0.1:32100`）调度算力（如 RTX 4090），**无需任何线上 API Key，不产生任何云端费用**。

> **⚠️ 必须先装桌面端再装本技能。** 本技能只是调度外壳，配音 / 识别 / 数字人三个模型都跑在 Comfypet 里。
> 👉 **[下载 JMCAI Comfypet 桌面端](https://github.com/allen-Jmc/comfypet-jmcai-Dist/releases/latest)**（Windows 安装版 / 便携版、macOS DMG）

![成片效果](docs/demo_face.jpg)

> 上面四格都出自同一条 27.6 秒的成片：全屏出镜开场 → 右下角小窗配左栏信息 → 纯图文铺满（37 条测试逐个亮起）→ 收尾品牌落点。人物的三种形态轮换，不是全程贴一个小窗。

---

## 三个阶段全流程

| 阶段 | 做什么 | 依赖工作流 / 工具 | 产物 |
|---|---|---|---|
| **1 配音** | 声音克隆/音色合成，字级时间戳对齐 | `qwen3-tts-_api` + `qwen3-asr-_api` | `merged.wav`（配音）、`words.json`（字级时间戳） |
| **2 数字人** | 一张照片开口说话，手势跟着内容自然起伏 | `minimax-h3-copy-json` (MiniMax H3) | `final_avatar.mp4`（1080P 高清口播视频） |
| **3 包装** | 编排版式：右下角人物小窗、字幕、科技卡片、音效 | Remotion 4.x + Chromium | `final/成片_JMCAI口播一条过.mp4` |

---

## 快速安装与使用（在新设备上）

### 1. 安装到技能目录

将本项目复制或克隆到 WorkBuddy / Claude 的用户技能目录中：

```bash
# WorkBuddy AI 用户目录（推荐）
cp -r jmcai-autoshot ~/.workbuddy-ai/skills/jmcai-autoshot

# 或 Claude Code 技能目录
cp -r jmcai-autoshot ~/.claude/skills/jmcai-autoshot
```

### 2. 前置准备

在目标设备上仅需准备两项：

1. **安装并启动 JMCAI Comfypet 桌面端**

   这个技能的所有算力调度都走 Comfypet 的 Workflow Bridge，**没有它跑不起来**。下载：

   | 平台 | 推荐包 | 适合谁 | 下载 |
   |---|---|---|---|
   | Windows | 安装版 | 大多数用户，支持自动更新与系统集成 | [下载安装包](https://github.com/allen-Jmc/comfypet-jmcai-Dist/releases/latest) |
   | Windows | 便携版 | 不想安装、希望放移动盘或独立目录 | [Releases 页](https://github.com/allen-Jmc/comfypet-jmcai-Dist/releases/latest) 选 `portable.exe` |
   | macOS | DMG | 直接拖入 Applications | [Releases 页](https://github.com/allen-Jmc/comfypet-jmcai-Dist/releases/latest) 选 `.dmg` |

   全部版本见 [Releases](https://github.com/allen-Jmc/comfypet-jmcai-Dist/releases)，仓库主页 [comfypet-jmcai-Dist](https://github.com/allen-Jmc/comfypet-jmcai-Dist)，官网 [jmcai.cn](https://jmcai.cn)。

   装好之后确认：
   - 桌面端正常运行，**Workflow Bridge 已开启**（默认端口 `32100`）。
   - 在桌面端里**已配置并启用这 4 套工作流**（缺任何一套，对应阶段就会失败）：
     - `qwen3-tts-_api`（声音克隆，可选备选：`qwen3-tts-_api-2` 预设音色、`qwen3-tts-voicedesign-_api` 声音设计）
     - `qwen3-asr-_api`（字级时间戳识别）
     - `minimax-h3-copy-json`（MiniMax H3 数字人生成）
   - **算力目标已绑好**（本地 ComfyUI，或一台装了 4090 的机器）。

   > 如果你装的桌面端版本较旧、里面找不到上面这几套工作流，先升级到最新版再看。工作流模板是随桌面端版本走的。

2. **基础依赖自检**：
   ```bash
   python scripts/setup_env.py
   ```
   脚本会自动检测 Python 包（numpy, opencv-python, Pillow, av, imageio-ffmpeg, librosa）、Node.js、Remotion 依赖与无头浏览器，缺什么会自动安装，无需手动配置。

### 3. 关于 Python 与音频依赖（重要）

配音阶段的**声学门禁**（自动把关音色像不像、不达标自动重抽）依赖 `librosa` + `scipy`。这里有个已知的坑：

- **Python 3.13 及更新版本，`librosa` / `av` / `imageio-ffmpeg` 常常没有预编译包**，`setup_env.py` 会在 `failed` 里列出它们。这**不代表功能坏了**。
- 脚本有**环境自适应**：当前解释器没有 `librosa` 时，会自动去找本机 ComfyUI 便携版里自带的 `python_embeded`（那个环境通常已经装好 librosa）来跑声学分析。
- 想显式指定用哪个解释器，设一个环境变量即可（最可靠）：
  ```bash
  # Windows
  set COMFYUI_PYTHON=D:\ComfyUI_windows_portable\python_embeded\python.exe
  # macOS / Linux
  export COMFYUI_PYTHON=~/ComfyUI/venv/bin/python
  ```
- 如果两种情况都不满足（本机解释器没 librosa、也找不到 ComfyUI），配音仍然能出声音，只是**跳过音色校验**，不会自动重抽——这时音色漂移只能靠人耳把关。建议顺手把 librosa 装上，或指一下 `COMFYUI_PYTHON`。

**推荐做法**：直接用 Python 3.11 / 3.12（生态最全），或者干脆一直用 ComfyUI 自带的 `python_embeded` 跑配音脚本。

### 4. 检查环境是否就绪
```bash
python scripts/voice/check_key.py      # 桌面端在线 + 配音/识别工作流已启用
python scripts/avatar/check_setup.py   # 桌面端在线 + 数字人工作流已就绪
```
两个都退出码 0，就可以开始做片子了。

### 5. 一键出片

在 AI 对话框中直接输入：
> “用 jmcai-autoshot 帮我把这段文案做成口播视频”

并附带：
- **文案**：准备好的口播文案（建议 30~300 字）
- **声音样本**：3~30 秒清晰纯净的人声录音（用过一次后会自动记住，后续无需重复上传）
- **口播图片**：一张半身正脸正面、双手自然下垂或放在身前的高清照片

技能会自动串联调度配音、数字人与包装，最后在项目的 `final/` 目录下交付高清成片！

---

## 目录结构

```
jmcai-autoshot/
├── SKILL.md                    技能入口与主调度规则
├── README.md                   英文说明
├── README.zh-CN.md             中文说明
├── references/                 各阶段详细指南
│   ├── pipeline.md             项目目录交接与断点续做机制
│   ├── voice/                  配音阶段操作手册与工作流契约
│   ├── avatar/                 数字人分段、提示词规则与质检
│   └── packaging/              Remotion 包装编排与视觉规范
├── scripts/                    核心执行脚本
│   ├── jmcai.py                JMCAI Bridge 客户端（配音/数字人通用通信层）
│   ├── status.py               项目进度查看工具
│   ├── setup_env.py            依赖检测与自动安装
│   ├── voice/                  配音拆段、合成、识别与核对
│   ├── avatar/                 底图处理、提示词校验、调度与拼接
│   └── packaging/              Remotion 包装工程脚手架与 QC 质检
├── tests/                      离线回归测试（改过脚本后跑）
│   └── avatar/                 数字人：重抽种子计数、质检判定
└── assets/
    └── remotion-template/      包装工程基础模板
```

---

## 核心特性优势

- 🔒 **全链路私有安全**：所有文案、声音样本与人像照片均留在私有环境与本地设备中流转，不上传任何第三方云端 SaaS。
- 💰 **零 API 消耗费用**：彻底摆脱云端按秒计费，无 API Key 过期或额度消耗焦虑。
- 🎯 **断点续做保障**：配音与数字人任务号实时落盘，即便中断或超时，重新执行会自动续上进度，绝不重复计算。
- 🎬 **代码级动态包装**：告别千篇一律的死板模板，由代码驱动生成高精度小窗画中画（PIP）、字级对齐字幕与高保真音效。
