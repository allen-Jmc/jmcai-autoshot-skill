"""数字人阶段的公共部分：固定参数、尺寸换算、JMCAI Bridge 接入。

算力来自**本机的 JMCAI Comfypet 桌面端**（它内置 Workflow Bridge，默认 127.0.0.1:32100），
数字人工作流固定用 minimax-h3-copy-json。这里不再有 RunningHub 的 Key / 会员档位 / 应用 ID / 计费。
"""
import json
import math
import os
import sys

# scripts/ 目录加进 sys.path，才能 import 上层的 jmcai 客户端
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import jmcai  # noqa: E402

# ---------------------------------------------------------------- 固定参数
MAX_SEGMENT_SECONDS = 14.0       # 分段只设最长限制（用户定）
RESOLUTION_MEGAPIXELS = 1.0      # 只用于 prepare_image.py 预估目标尺寸（本地工作流按画幅自己出分辨率）
RATIOS = {"16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1), "3:4": (3, 4), "4:3": (4, 3), "21:9": (21, 9)}
SIZE_MIN, SIZE_MAX = 256, 2144   # 边长上限/下限，且必须是 32 的倍数

# 画幅 → MiniMax H3 工作流暴露的 aspect_ratio 枚举值。
# 这些字符串必须和**算力目标上那个节点的真实候选值**逐字一致，否则 ComfyUI 会
# "Failed to validate prompt … Value not in list"，输出被忽略，但桌面端仍报 success。
# 拿不准时用 validate_args() 对着桌面端的实时 schema 核一遍。
ASPECT_RATIO_CHOICES = {
    "1:1": "1:1 (Square)",
    "2:3": "2:3 (Portrait Photo)",
    "3:2": "3:2 (Photo)",
    "3:4": "3:4 (Portrait Standard)",
    "4:3": "4:3 (Standard)",
    "9:16": "9:16 (Portrait Widescreen)",
    "16:9": "16:9 (Widescreen)",
    "21:9": "21:9 (Ultrawide)",
}

WORKSPACE_ROOT = os.environ.get("DH_WORKSPACE", "").strip() or os.path.join(os.path.expanduser("~"), ".digital-human")
CONFIG_FILE = os.path.join(WORKSPACE_ROOT, "config.json")

DEFAULT_SEED = 42


# ---------------------------------------------------------------- 尺寸
def snap32(v):
    return min(SIZE_MAX, max(SIZE_MIN, int(round(v / 32.0)) * 32))


def size_for_ratio(ratio_name, megapixels=RESOLUTION_MEGAPIXELS):
    rw, rh = RATIOS[ratio_name]
    g = math.sqrt(megapixels * 1024 * 1024 / (rw * rh))
    return snap32(rw * g), snap32(rh * g)


def aspect_ratio_value(ratio_name):
    """画幅名（如 "16:9"）→ 工作流 aspect_ratio 的枚举值。认不出来时回落到 9:16 竖屏。"""
    return ASPECT_RATIO_CHOICES.get(ratio_name) or ASPECT_RATIO_CHOICES["9:16"]


def validate_args(workflow_id, args):
    """按桌面端暴露的实时 schema 校验参数，返回问题清单（空列表 = 通过）。

    这是为了拦住一类很隐蔽的失败：枚举值写错时，ComfyUI 会在算力目标上
    "Failed to validate prompt … Value not in list"，把输出整个丢掉，但桌面端
    仍然报 success——只有拿到空的 outputs 才会发现。提前核一遍就能当场说清楚。
    """
    schema = [f for f in (jmcai.workflow_schema(workflow_id) or []) if f.get("enabled", True)]
    by_alias = {f.get("alias"): f for f in schema if f.get("alias")}
    problems = []
    for name, f in by_alias.items():
        if f.get("required") and name not in args:
            problems.append("缺少必填参数 %s" % name)
    for k, v in args.items():
        f = by_alias.get(k)
        if not f:
            problems.append("桌面端没有暴露参数 %s（已暴露：%s）" % (k, "、".join(by_alias) or "无"))
            continue
        ch = f.get("choices") or []
        if f.get("type") == "enum" and ch and v not in ch:
            problems.append("%s=%r 不在允许的取值里：%s" % (k, v, " / ".join(map(str, ch))))
    return problems


# ---------------------------------------------------------------- 配置
def load_config():
    if os.path.isfile(CONFIG_FILE):
        try:
            return json.load(open(CONFIG_FILE, encoding="utf-8"))
        except (OSError, ValueError):
            return {}
    return {}


def save_config(cfg):
    os.makedirs(WORKSPACE_ROOT, exist_ok=True)
    json.dump(cfg, open(CONFIG_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def concurrency(override=None):
    """本地是单台 4090，默认串行 1 路；用户想并行可在 config.json 里写 concurrency。"""
    if override:
        return max(1, int(override))
    return max(1, int(load_config().get("concurrency", 1)))


# ---------------------------------------------------------------- Bridge
def require_bridge():
    """确认桌面端在线；不在线就打印引导并退出（退出码 3）。"""
    return jmcai.require_bridge()


def avatar_workflow():
    """确认数字人工作流在桌面端已启用；没有就退出。"""
    wf = jmcai.pick_workflow([jmcai.WF_AVATAR])
    if not wf:
        print("JMCAI 桌面端里没有启用数字人工作流 %s。\n"
              "请在桌面端导入并启用 MiniMax H3 数字人工作流，然后重试。" % jmcai.WF_AVATAR)
        sys.exit(3)
    return wf


def require_key():                     # 兼容旧调用点：本地没有密钥，改成检查桌面端
    return require_bridge()
