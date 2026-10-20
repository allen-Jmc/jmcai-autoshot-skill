"""配音阶段的公共部分：固定参数、JMCAI Bridge 连接、文本标签、ffmpeg。

配音与识别都跑在**用户本机的 JMCAI Comfypet 桌面端**里（桌面端已配好 Qwen3-TTS / Qwen3-ASR 工作流，
算力可以是本机 ComfyUI，也可以是局域网/外网的一台 4090）。本文件是原「百炼 DashScope」版本的平替：
不再需要任何云端 API Key，也不再向用户索取密钥。

固定参数（用户已定，不再询问）：
  * 合成走 `qwen3-tts-_api`（声音克隆）：把用户本人的参考音频 + 台词送进去，内置 ASR 自动提参考文本。
  * 识别走 `qwen3-asr-_api`：出**字级**时间戳。
  * 输出的 wav 统一 48k 单声道。
"""
import json
import os
import re
import shutil
import subprocess
import sys

# 让 scripts/ 下的 jmcai.py 能被 import（本文件在 scripts/voice/ 下）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jmcai  # noqa: E402

# ---------------------------------------------------------------- 固定参数（用户已定，不再询问）
SAMPLE_RATE = 48000
MAX_CHARS = 1000         # 每段文字上限
WORKERS = int(os.environ.get("DH_TTS_WORKERS", "1"))   # 并发数。本地单卡跑 TTS，默认 1 路最稳（模型常驻显存，串行也很快）
SUBMIT_INTERVAL = 0.5    # 相邻两次提交的最小间隔（秒）
GAP_SECONDS = 0.3        # 段与段之间的停顿
MAX_TRIES = 6

# 合成模式：clone（用用户自己的声音，默认）/ preset（官方预设音色）/ design（自然语言描述造音色）
TTS_MODE = os.environ.get("DH_TTS_MODE", "clone").strip().lower()
PRESET_SPEAKER = os.environ.get("DH_TTS_SPEAKER", "Vivian").strip()
DESIGN_INSTRUCT = os.environ.get("DH_TTS_INSTRUCT", "").strip()

# 情绪标签只在能收自然语言指令的模式（preset / design）里用得上；clone 模式会把标签剥掉（模型会照字念出来）
INSTRUCTION = os.environ.get(
    "DH_TTS_INSTRUCTION",
    "像朋友聊天一样自然随意的口语，不要播音腔，语速较快，句与句之间保留自然的停顿，语气轻松带笑意")
QUESTION_TAG = "[curious]" if TTS_MODE in ("preset", "design") else ""
ASR_LANGUAGE = os.environ.get("DH_ASR_LANGUAGE", "auto")

# 项目缓存根目录：每份口播文案一个目录，配音、逐词时间、数字人等后续环节共用
WORKSPACE_ROOT = os.environ.get("DH_WORKSPACE", "").strip() or os.path.join(os.path.expanduser("~"), ".digital-human")


# ---------------------------------------------------------------- JMCAI Bridge
def require_bridge():
    """确认 JMCAI 桌面端在线；不在线就打印引导并退出（退出码 3）。"""
    return jmcai.require_bridge()


def tts_workflow_id(mode=None):
    """TTS 模式的配置名（如 qwen3-tts-clone）。**纯函数：不连桌面端、不退出进程。**

    缓存键这类纯计算要用它。误用 tts_workflow() 会让"算个哈希"变成"必须桌面端在线"，
    而且那边是 sys.exit（BaseException，except Exception 抓不到），会把整个流程杀掉。
    模式非法时抛 ValueError，由调用方决定怎么处理。
    """
    m = (mode or TTS_MODE).strip().lower()
    want = {"clone": jmcai.WF_TTS_CLONE, "preset": jmcai.WF_TTS_PRESET, "design": jmcai.WF_TTS_DESIGN}.get(m)
    if not want:
        raise ValueError("TTS 模式只能是 clone / preset / design，现在是：%s" % m)
    return want


def tts_workflow(mode=None):
    """本次要用的 TTS 工作流 id。优先取传入的 mode（来自 voice.json），否则取 TTS_MODE。
    会连桌面端确认它已启用；没启用就退出（退出码 1）。只该在真要合成前调用。"""
    try:
        want = tts_workflow_id(mode)
    except ValueError as e:
        sys.exit(str(e))
    have = jmcai.pick_workflow([want])
    if not have:
        sys.exit("JMCAI 桌面端里没有启用工作流「%s」。请在桌面端导入并启用它，或把 TTS 模式换成别的模式。" % want)
    return have


def asr_workflow():
    have = jmcai.pick_workflow([jmcai.WF_ASR])
    if not have:
        sys.exit("JMCAI 桌面端里没有启用工作流「%s」。请在桌面端导入并启用它。" % jmcai.WF_ASR)
    return have


# ---------------------------------------------------------------- 文本标签
_SENT = re.compile(r"[^。！？!?；;…~～\n]+[。！？!?；;…~～]*")
_ANY_TAG = re.compile(r"\[[a-z_]+\]")


def strip_tags(text):
    """剥掉 [excited] 这类情绪标签：Qwen3-TTS 的工作流会把方括号照字念出来。"""
    return _ANY_TAG.sub("", text)


def tag_questions(text):
    """在每个以问号结尾的句子前加 QUESTION_TAG。只改发给合成接口的文字，不改文案和缓存里的原文。"""
    if not QUESTION_TAG:
        return text

    def f(m):
        s = m.group(0)
        if not re.search(r"[？?]", s[-3:]):
            return s
        lead = len(s) - len(s.lstrip())
        return s[:lead] + QUESTION_TAG + s[lead:]
    return _SENT.sub(f, text)


def tag_emotions(text, rules):
    """按项目里 emotions.json 的规则，在指定的句子或分句前加情绪标签（如 [excited]）。
    rules: [{"at": "这句话开头的几个字", "tag": "[excited]", "n": 第几次出现（默认 1）}]。
    只改发给合成接口的文字；找不到 at 的规则跳过（这一段里没有这句话）。那个位置已经有问句标签时不再加。"""
    spots = []
    for r in rules or []:
        at, tag, n = r.get("at", ""), r.get("tag", ""), int(r.get("n", 1))
        if not at or not tag:
            continue
        i = -1
        for _ in range(n):
            i = text.find(at, i + 1)
            if i < 0:
                break
        if i < 0 or (QUESTION_TAG and text[max(0, i - len(QUESTION_TAG)):i] == QUESTION_TAG):
            continue
        spots.append((i, tag))
    for i, tag in sorted(spots, reverse=True):
        text = text[:i] + tag + text[i:]
    return text


def apply_carry(paras, rules, carry="auto", start_tag="[excited]", gap=80):
    """给每一段加好所有标签，返回和 paras 等长的列表。
    先按 rules 在指定的句子前加情绪标签、给问句加问句标签；然后"沿用"：没有自己标签的句子，沿用前面最近一个情绪标签，
    这样信息密集的地方（报档位、念清单）和段与段的交界都不会出现情绪突变。第一句之前没有标签时用 start_tag。
    carry: "auto"（默认）段首必补，段内某个句子没有标签、而且离上一个标签已经超过 gap 个字时才补；
           "sentence" 每个没标签的句子都补；"paragraph" 只在段首补；"off" 不补。
    注意：clone 模式下 QUESTION_TAG 为空、且最终会把标签剥掉，所以这里主要服务于 preset / design 模式。"""
    out, cur = [], start_tag
    for p in paras:
        tagged = tag_emotions(tag_questions(p), rules)
        parts = _SENT.findall(tagged)
        if not rules or carry == "off" or "".join(parts) != tagged:
            out.append(tagged)
            continue
        res, since, after_q = [], 0, False
        for k, sent in enumerate(parts):
            body = sent.lstrip()
            m = re.match(r"\[[a-z_]+\]", body)
            if m:
                after_q = m.group(0) == QUESTION_TAG
                if not after_q:
                    cur = m.group(0)
                res.append(sent)
                since = len(body)
            elif k == 0 or after_q or carry == "sentence" or (carry == "auto" and since >= gap):
                res.append(sent[:len(sent) - len(body)] + cur + body)
                since, after_q = len(body), False
            else:
                res.append(sent)
                since += len(body)
        out.append("".join(res))
    return out


def tags_not_at_sentence_start(tagged):
    """标签不在句首的位置（句中标签容易被念出来，也会让一句话内部变调）。"""
    bad = []
    for m in re.finditer(r"\[[a-z_]+\]", tagged):
        i = m.start()
        before = tagged[:i].rstrip()
        if before and before[-1] not in "。！？!?；;…]":
            bad.append(tagged[max(0, i - 8):i + 12])
    return bad


def emotion_hint(tagged):
    """把句首情绪标签折成一句自然语言指令，给 preset / design 模式当 instruct 用。"""
    seen = []
    for m in _ANY_TAG.finditer(tagged):
        t = m.group(0)[1:-1]
        if t != "curious" and t not in seen:
            seen.append(t)
    return "，".join(seen)


# ---------------------------------------------------------------- 兼容旧调用
def require_key():
    """百炼版本的遗留入口：现在不需要密钥，改为确认桌面端在线。"""
    require_bridge()
    return "jmcai"


# ---------------------------------------------------------------- 项目缓存
def update_meta(workdir, **kw):
    """如果 workdir 是项目目录（有 meta.json），把进度写回去；否则什么也不做。"""
    p = os.path.join(workdir, "meta.json")
    if not os.path.isfile(p):
        return
    m = json.load(open(p, encoding="utf-8"))
    m.update(kw)
    json.dump(m, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- ffmpeg
def ffmpeg():
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        sys.exit("找不到 ffmpeg：请安装 ffmpeg 并加入 PATH，或 pip install imageio-ffmpeg")


def to_pcm_wav(src, dst, sr=SAMPLE_RATE):
    """统一转成 16bit 单声道 wav。"""
    subprocess.run([ffmpeg(), "-y", "-v", "error", "-i", src, "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", dst],
                   check=True)


def media_seconds(path):
    """用 ffmpeg 读时长（秒）。"""
    r = subprocess.run([ffmpeg(), "-hide_banner", "-i", path], capture_output=True)
    for line in r.stderr.decode("utf-8", "replace").splitlines():
        if "Duration:" in line:
            t = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = t.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return None
