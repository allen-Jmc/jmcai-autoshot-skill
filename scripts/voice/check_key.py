"""查配音阶段能不能开工：本机的 JMCAI Comfypet 桌面端在不在线、TTS 与 ASR 工作流有没有启用。

  python check_key.py            # 都齐：打印 JSON，退出码 0；桌面端没起或缺工作流：打印引导，退出码 3

文件名保留 `check_key.py` 是为了不改动 SKILL.md 和各处引用；这个阶段现在**不需要任何 API Key**，
原来的百炼密钥检查已经换成 JMCAI 桌面端与工作流的可用性检查。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import jmcai


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    h = jmcai.health()
    if h.get("status") != "success":
        print(jmcai_bridge_guide())
        sys.exit(3)

    ids = [jmcai.WF_TTS_CLONE, jmcai.WF_TTS_PRESET, jmcai.WF_TTS_DESIGN, jmcai.WF_ASR]
    have = {w.get("id") for w in jmcai.registry()}
    missing = [i for i in ids if i not in have]
    # 默认模式（clone）与识别是硬性要求，另外两套 TTS 属于可选增强
    hard_missing = [i for i in missing if i in (jmcai.WF_TTS_CLONE, jmcai.WF_ASR)]
    if hard_missing:
        print("JMCAI 桌面端在线，但缺少配音必需的工作流：%s\n"
              "请在 JMCAI Comfypet 桌面端里导入并启用它们（导入后需要重启一次桌面端）。" % "、".join(hard_missing))
        sys.exit(3)

    print(json.dumps({
        "ok": True,
        "bridge": h.get("bridge_url"),
        "desktop_app": h.get("desktop_app"),
        "bridge_version": h.get("bridge_version"),
        "tts_mode": C.TTS_MODE,
        "tts_workflow": C.tts_workflow(),
        "asr_workflow": C.asr_workflow(),
        "optional_missing": [i for i in missing if i not in hard_missing],
    }, ensure_ascii=False, indent=1))


def jmcai_bridge_guide():
    return ("本机没有连上 JMCAI Comfypet 桌面端（%s）。\n"
            "请先打开 JMCAI Comfypet 桌面应用，确认里面的 Workflow Bridge 已启动，再重试。\n"
            "如果桌面端的端口不是默认的 32100，请把环境变量 JMCAI_BRIDGE_URL 指到正确的地址。"
            % jmcai.bridge_url())


if __name__ == "__main__":
    main()
