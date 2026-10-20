# -*- coding: utf-8 -*-
"""从本机 JMCAI Comfypet 导出工作流，清洗后写进本目录。

维护者工具。Comfypet 里的工作流改过之后，重新跑一遍就能把仓库里的副本更新掉：

    python workflows/_export_from_comfypet.py

只保留"别人导入后能直接用的东西"：
    workflow.json  —— 纯 ComfyUI 节点定义（API 格式）
    schema.json    —— 参数暴露定义
    meta.json      —— 只留 id/name/description/format/输入输出模态

必须剥掉的全部是本机私有信息：
    defaultTargetId / bindings —— 指向某台具体的算力服务器，等于告诉别人我的机器在哪
    importedAt / updatedAt     —— 时间戳
    origin / examplePrompts    —— 本机导入痕迹
    *.orig / *.bak / *.client-rewrote —— 备份文件，可能残留旧内容

绝不导出 targets.json —— 里面有真实的算力服务器域名和上传端点。
"""
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Comfypet 的工作流目录；要换位置就设环境变量 JMCAI_WORKFLOWS
SRC = Path(os.environ.get("JMCAI_WORKFLOWS")
           or Path(os.path.expanduser("~")) / "AppData/Roaming/JMCAI/workflows/definitions")

WORKFLOWS = {
    "qwen3-tts-_api": "声音克隆：参考音频 + 台词 → 克隆音色的配音",
    "qwen3-tts-_api-2": "预设音色：台词 + speaker + instruct → 配音（备选）",
    "qwen3-tts-voicedesign-_api": "声音设计：用文字描述音色 → 配音（备选）",
    "qwen3-asr-_api": "语音识别：音频 → 文字 + 字级时间戳",
    "minimax-h3-copy-json": "数字人：人物图 + 音频 → 带口型和手势的视频",
}

KEEP_META = ("id", "name", "description", "summary", "format",
             "inputModalities", "outputModalities", "enabled")

# 任何文件里都不许出现的模式。
# 注意：必须对 **JSON 解析之后** 的字符串做检查——直接扫原文会把
# `definitions:\n` 这类换行转义误判成 Windows 路径 `s:\`（实测 minimax 的
# schema.json 因此报过 6 个假阳性）。
SENSITIVE = [
    (re.compile(r"https?://(?!127\.0\.0\.1)[^\"'\s]"), "外部 URL"),
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "IP 地址"),
    (re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:\\[A-Za-z0-9_.\-]"), "Windows 绝对路径"),
    (re.compile(r"(?i)(api[_-]?key|token|secret|password|bearer)\s*[:=]\s*[\"'][^\"']{4,}"), "疑似凭证"),
    (re.compile(r"external-server"), "本机算力绑定 ID"),
]


def walk_strings(obj, path=""):
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_strings(v, "%s.%s" % (path, k) if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_strings(v, "%s[%d]" % (path, i))


def scan_text(text):
    hits = []
    for pat, label in SENSITIVE:
        for m in pat.finditer(text):
            hits.append((label, m.group(0)[:60]))
    return hits


def scan_file(p):
    hits = []
    raw = p.read_text(encoding="utf-8", errors="ignore")
    if p.suffix.lower() == ".json":
        try:
            data = json.loads(raw)
        except Exception:
            return [("JSON 解析失败，无法审计", "")]
        for path, s in walk_strings(data):
            for label, m in scan_text(s):
                hits.append(("%s @ %s" % (label, path), m))
    else:
        for label, m in scan_text(raw):
            hits.append((label, m))
    return hits


def clean_meta(p):
    d = json.loads(p.read_text(encoding="utf-8"))
    out = {k: d[k] for k in KEEP_META if k in d}
    tags = [t for t in (d.get("tags") or []) if t != "imported"]
    if tags:
        out["tags"] = tags
    return out


def main():
    if not SRC.is_dir():
        sys.exit("找不到 Comfypet 工作流目录：%s\n可用环境变量 JMCAI_WORKFLOWS 指定。" % SRC)

    for wf, purpose in WORKFLOWS.items():
        src = SRC / wf
        if not src.is_dir():
            print("  跳过 %-28s 本机没有这套工作流" % wf)
            continue
        dst = HERE / wf
        dst.mkdir(parents=True, exist_ok=True)
        for fn in ("workflow.json", "schema.json"):
            f = src / fn
            if f.is_file():
                (dst / fn).write_bytes(f.read_bytes())
        if (src / "meta.json").is_file():
            (dst / "meta.json").write_text(
                json.dumps(clean_meta(src / "meta.json"), ensure_ascii=False, indent=1),
                encoding="utf-8")
        (dst / "README.md").write_text(
            "# %s\n\n%s\n\n由本机 JMCAI Comfypet 导出。已剥离算力目标绑定、时间戳与备份文件，"
            "只保留工作流定义本身。\n" % (wf, purpose), encoding="utf-8")
        print("  已导出 %-28s %s" % (wf, purpose))

    print("\n=== 敏感信息扫描 ===")
    hits = 0
    for p in sorted(HERE.rglob("*")):
        if not p.is_file() or p.name.startswith("_") or p.name == "README.md":
            continue
        for label, snippet in scan_file(p):
            hits += 1
            print("  ✗ %s: %s → %s" % (p.relative_to(HERE), label, snippet))
    if hits:
        print("  发现 %d 处敏感信息，**不要提交**，先处理。" % hits)
        return 1
    print("  干净。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
