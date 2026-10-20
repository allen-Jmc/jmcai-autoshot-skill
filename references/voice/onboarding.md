# 本机前置条件：JMCAI Comfypet 桌面端

只在 `scripts/voice/check_key.py` 或 `scripts/avatar/check_setup.py` 报错（退出码 3）时读。

本技能不连任何线上 API、不需要 API Key、不产生费用。所有生成都通过**本机的 JMCAI Comfypet 桌面端**（它内置 Workflow Bridge，默认监听 `127.0.0.1:32100`）转发到桌面端里绑定好的算力目标。

## 需要用户确认的三件事

1. **JMCAI Comfypet 桌面端已打开**，且里面的 Workflow Bridge 是启动状态。
2. **要用的工作流已在桌面端启用**：
   - 配音：`qwen3-tts-_api`（声音克隆，默认）/ `qwen3-tts-_api-2`（预设音色）/ `qwen3-tts-voicedesign-_api`（声音设计）
   - 识别：`qwen3-asr-_api`
   - 数字人：`minimax-h3-copy-json`
3. **工作流已绑定可用的算力目标**（本机 ComfyUI 或一台外部服务器，例如 4090）。

## 怎么跟用户说（原样，别自己改端口或翻配置）

> 本机没有连上 JMCAI Comfypet 桌面端。请先打开 JMCAI Comfypet 桌面应用，确认里面的 Workflow Bridge 已启动、
> 需要的工作流已启用并绑定好算力目标，然后告诉我，我重试。

用户说端口不是 32100 时，让他把地址告诉我，我设环境变量 `JMCAI_BRIDGE_URL` 指过去（也可以写进
`~/.workbuddy-ai/skills/comfypet-jmcai-skill/config.json` 的 `bridge_url`，本技能会自动读）。

## 不要做的事

- 不要自己去扫端口、猜地址、改桌面端配置文件。
- 不要让用户去申请任何 API Key。
- 不要试图绕过桌面端直接调算力目标（唯一例外：桌面端等远端输出超时后的取结果兜底，见 `api-notes.md`）。
