# 工作流定义

这 5 套 ComfyUI 工作流是本技能的下游依赖。**JMCAI Comfypet 桌面端不自带任何工作流**（我们验证过：全新安装后工作流列表是空的），所以你需要在桌面端里把它们导入一次。

| 工作流 ID | 用途 | 节点数 | 阶段 |
|---|---|---|---|
| `qwen3-tts-_api` | 声音克隆：参考音频 + 台词 → 克隆音色的配音 | 8 | 1 配音 |
| `qwen3-tts-_api-2` | 预设音色：台词 + speaker + instruct → 配音 | 4 | 1 配音（备选） |
| `qwen3-tts-voicedesign-_api` | 声音设计：文字描述音色 → 配音 | 4 | 1 配音（备选） |
| `qwen3-asr-_api` | 语音识别：音频 → 文字 + 字级时间戳 | 7 | 1 配音 |
| `minimax-h3-copy-json` | 数字人：人物图 + 音频 → 带口型和手势的视频 | 31 | 2 数字人 |

只用 `qwen3-tts-_api` + `qwen3-asr-_api` + `minimax-h3-copy-json` 三套就能跑完全流程，另外两套是配音的备选方案。

每个目录里：

- `workflow.json` — ComfyUI 节点定义（API 格式），**导入用这个**
- `schema.json` — 参数暴露定义，说明这个工作流接受哪些输入、哪些可以调
- `meta.json` — 工作流元信息（名称、说明、输入输出模态）

> 这些文件从本机 Comfypet 导出后做过清洗：**已剥离算力目标绑定（`defaultTargetId` / `bindings`）、导入时间戳和所有备份文件**。不含 API Key、不含服务器地址、不含本机路径。

## 一、导入到 Comfypet

在桌面端的**工作流**页面选择导入，选对应目录下的 `workflow.json`。导入后：

1. **绑定算力目标**（本机 ComfyUI，或一台装了 4090 的机器）——这是必须做的一步，导入后默认没有绑定。
2. **启用**这套工作流（导入后默认可能是禁用状态）。

导入 5 套后，可以用脚本确认链路：

```bash
python scripts/voice/check_key.py      # 配音 + 识别
python scripts/avatar/check_setup.py   # 数字人
```

## 二、依赖的自定义节点

这些工作流用到了 ComfyUI 原生节点之外的扩展节点。**装到算力目标那台机器的 `ComfyUI/custom_nodes/` 下**。

> 下面这份清单是逐个节点核对出来的：用 ComfyUI 的 `/object_info` 接口查每个节点的 `python_module`，再回到 `custom_nodes/` 里定位它由哪个包提供。**只有确实用到的才列。**

### 配音 + 识别（`qwen3-tts-*` / `qwen3-asr-_api`）

| 节点包 | 提供的关键节点 | 仓库 |
|---|---|---|
| **ComfyUI-Qwen3-TTS** | `Qwen3Loader`、`Qwen3VoiceClone`、`Qwen3CustomVoice`、`Qwen3VoiceDesign` | [DarioFT/ComfyUI-Qwen3-TTS](https://github.com/DarioFT/ComfyUI-Qwen3-TTS) |
| **ComfyUI-Qwen3-ASR** | `Qwen3ASRLoader`、`Qwen3ASRTranscribe` | [DarioFT/ComfyUI-Qwen3-ASR](https://github.com/DarioFT/ComfyUI-Qwen3-ASR) |

### 数字人（`minimax-h3-copy-json`）

| 节点包 | 提供的关键节点 | 仓库 |
|---|---|---|
| **ComfyUI-KJNodes** | `MiniMaxChunkFeedForward`、`MiniMaxLowVRAMAttention`、`ModelPreviewOverrideKJ` | [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) |
| **selflift-Avatar** | `SelfLiftAvatarH3Sampler` | [slmonker/selflift-Avatar](https://github.com/slmonker/selflift-Avatar) |
| **ComfyUI-SoundFlow** | `SoundFlow_GetLength` | [huixingyun/ComfyUI-SoundFlow](https://github.com/huixingyun/ComfyUI-SoundFlow) |
| **ComfyUI-VideoHelperSuite** | `VHS_VideoCombine` | [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) |

### 不需要装的（容易装错）

| 包 | 为什么不需要 |
|---|---|
| ~~ComfyUI-MiniMaxH3-Easy~~ | 它提供的是一套独立的 `MiniMaxH3Easy*` 高层封装节点。本工作流用的是 ComfyUI **内置**的 `MiniMaxH3ReferenceToVideo` + KJNodes 的辅助节点，**一个 Easy 节点都没用到**。装了不会出错，但纯属多余。 |
| ~~ComfyUI-LTXVideo~~ | `LTXVConcatAVLatent`、`LTXVSeparateAVLatent`、`LTXVAudioVAEEncode` 这几个节点**已经内置在 ComfyUI 里**（`comfy_extras/nodes_lt.py`），不需要额外的扩展包。 |

### 其余节点都是 ComfyUI 内置

`UNETLoader`、`VAELoader`、`CLIPLoader`、`LoadImage`、`LoadAudio`、`VAEDecode`、`LoraLoaderModelOnly`、`ConditioningZeroOut`、`SetLatentNoiseMask`、`SolidMask`、`BasicScheduler`、`KSamplerSelect`、`ImageScaleToTotalPixels`、`ResolutionSelector`、`ComfyMathExpression`、`ComfyNumberConvert`、`TrimAudioDuration`、`PreviewAny`、`MiniMaxH3ReferenceToVideo`、`ModelAttentionBackend` —— 这些来自 `nodes.py` 和 `comfy_extras/`，**不用装任何东西**。

> **对 ComfyUI 版本的要求**：内置的 MiniMax H3 节点（`comfy_extras/nodes_minimax_h3.py`）和 LTXV 节点（`comfy_extras/nodes_lt.py`）是较新版本才有的。
>
> 这份清单在**两套环境上分别核对过**，结论一致：
>
> | 环境 | ComfyUI 版本 | 节点总数 | 结论 |
> |---|---|---|---|
> | 本地 ComfyUI（RTX 4060 Ti） | 0.42.1 | 988 | 归属与上表一致 |
> | 生产算力目标（RTX 4090） | **0.41.2** | 5139 | 38 种节点全部存在，归属与上表一致 |
>
> 也就是说 **0.41.2 及以上的 ComfyUI 都能直接用**。如果你的算力目标版本更老、缺 `nodes_minimax_h3.py` 或 `nodes_lt.py`，升级 ComfyUI 本体，而不是去装第三方封装。

### 缺节点怎么补

不用一个个比对。在 ComfyUI 里**导入 `workflow.json` 打开画布**，ComfyUI-Manager 会弹出「Install Missing Custom Nodes」，点它按提示装完重启即可。装完再导入到 Comfypet。

## 三、模型文件

工作流本身只有节点连线，**模型不在仓库里**（体积太大）。需要的模型：

- Qwen3-TTS 模型：节点包首次运行时按需下载到 `ComfyUI/models/Qwen3-TTS/`
- MiniMax H3、LTXV 相关模型：见各自节点包的说明

模型下载慢是正常的（几十 GB）。如果 ComfyUI 在国内，建议先配好镜像源再跑第一次。

## 四、参数怎么填

`schema.json` 里定义了每个工作流的输入。本技能的脚本会自动传参，你不用手填——但如果你想在 Comfypet 界面里手动试跑，看这个文件就知道每个字段的含义。

几个容易搞错的：

- `qwen3-tts-_api`（声音克隆）需要**参考音频**：3–30 秒、干净人声、单人、无背景音乐。太短学不像，太长反而慢。
- `minimax-h3-copy-json` 的人物图**建议正面、双手自然放在身前或桌面**。手在画面外的话，动作只能靠身体带，效果会打折。
- `qwen3-asr-_api` 返回的是**字级时间戳**，包装阶段做字幕对齐全靠它；识别有同音错字是正常的，字幕最终用的是原文案。

## 五、维护者：这些文件怎么来的

由 `_export_from_comfypet.py` 从本机 Comfypet 的 `definitions/` 导出并清洗。Comfypet 更新工作流后，重新导出一遍即可：

```bash
python workflows/_export_from_comfypet.py
```

它会剥掉算力绑定和时间戳，并对导出结果跑一遍敏感信息扫描（URL / IP / 绝对路径 / 凭证），扫到任何一处都会报错退出。

> ⚠️ **不要导出 `targets.json`**。那个文件里有真实的算力服务器地址，属于私有配置。

### 改节点清单前先跑这个

节点依赖是**逐个核对**出来的，不是靠仓库名猜的。写清单或改工作流之后：

```bash
# 查本机 ComfyUI（默认 8188）
python workflows/_check_node_deps.py

# 查实际跑片子的算力目标 —— 这一步最该做
python workflows/_check_node_deps.py --target-url https://<你的算力目标>:8188

# 直接输出可贴进文档的表格
python workflows/_check_node_deps.py --print-table
```

它优先查 ComfyUI 的 `/object_info`（能准确区分内置/外部），再扫 `custom_nodes/` 定位包名；
最后拿结果和本文件的表格对账，漏写、多写、写错都会报出来。报告里还会打印 ComfyUI 版本和设备。

**为什么一定要查目标机**：本机和生产目标的版本经常不一样（实测本机 0.42.1、目标机 0.41.2）。
内置节点和外部节点的边界随版本变，只有查目标机才知道这份清单在真正跑片子的环境里成不成立。

**为什么必须有这个脚本**：这个清单靠印象写过两次，两次都错——

1. 曾误以为需要 `ComfyUI-MiniMaxH3-Easy`。实际那个包提供的是一套独立的 `MiniMaxH3Easy*` 高层封装，本工作流一个都没用到；用的是 ComfyUI **内置**的 `MiniMaxH3ReferenceToVideo` 加 KJNodes 的几个辅助节点。
2. 曾误以为需要 `ComfyUI-LTXVideo`。实际 `LTXV*` 节点已经内置在 `comfy_extras/nodes_lt.py`。

这两个包的信息在 ComfyUI 社区里传得很广，光看搜索结果很容易顺手写上。**只有查 `python_module` 才知道真相。**
