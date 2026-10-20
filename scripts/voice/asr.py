"""JMCAI 桌面端的 Qwen3-ASR 工作流：返回文字 + **字级**起止时间（秒）。

被 align_words.py 和 verify_audio.py 共用。这是原「百炼 qwen-audio-3.1-asr-flash」的平替：
不再需要任何云端 API Key，识别跑在用户本机的 JMCAI 桌面端里（算力可以是本机 ComfyUI 或外部 4090）。

工作流 `qwen3-asr-_api` 的输出是一个纯文本文件，每行一个字的起止时间：
    0.48-0.64: 徒
    0.64-0.80: 儿
    ...
时间单位是**秒**。文件里没有标点（标点由包装阶段按 copy.txt 的原文补回来）。

识别结果按"音频内容哈希"缓存在 <cache_dir>/asr_<哈希>.json，同一段音频不会重复跑。
"""
import hashlib
import json
import os
import random
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C
import jmcai

_lock = threading.Lock()
_next = [0.0]
INTERVAL = 0.5
LINE = re.compile(r"^\s*([\d.]+)\s*-\s*([\d.]+)\s*[:：]\s*(.*?)\s*$")


def _pace():
    with _lock:
        now = time.time()
        t = max(now, _next[0])
        _next[0] = t + INTERVAL
    if t > now:
        time.sleep(t - now)


def parse_timestamp_text(text):
    """把工作流输出的「起-止: 字」文本解析成 {"text","words":[{text,word,begin,end}]}。
    同一时间点重复出现（例如 I 的起止都是 4.56）也照收，不丢字。"""
    words, chars = [], []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        m = LINE.match(line)
        if not m:
            continue
        b, e, ch = float(m.group(1)), float(m.group(2)), m.group(3)
        if not ch:
            continue
        words.append({"text": ch, "word": ch, "begin": b, "end": max(e, b)})
        chars.append(ch)
    return {"text": "".join(chars), "words": words}


def _pick_timestamp_file(paths):
    """工作流会输出两个内容相同、行尾不同的 txt；挑解析出行数最多的那个。"""
    best, best_n = None, -1
    for p in paths:
        try:
            raw = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        n = len(parse_timestamp_text(raw)["words"])
        if n > best_n:
            best, best_n = raw, n
    return best


def transcribe(wav_path, key=None, cache_dir=None, language=None):
    """返回 {"text","words":[{text,word,begin,end}]}；失败抛 RuntimeError。

    key 参数是为了兼容原来的调用方式（百炼时代要传密钥），现在忽略。
    """
    h = hashlib.sha256(open(wav_path, "rb").read()).hexdigest()[:16]
    cp = os.path.join(cache_dir, "asr_%s.json" % h) if cache_dir else None
    if cp and os.path.isfile(cp):
        return json.load(open(cp, encoding="utf-8"))

    wf = C.asr_workflow()
    args = {"audio_1": os.path.abspath(wav_path), "language": language or C.ASR_LANGUAGE}
    last = ""
    for attempt in range(1, C.MAX_TRIES + 1):
        _pace()
        r = jmcai.run_and_wait(wf, args, quiet=True, label="ASR",
                               timeout_s=float(os.environ.get("JMCAI_ASR_TIMEOUT", "900")))
        st = r.get("status")
        if st == "success":
            txts = [o["path"] for o in jmcai.outputs_of(r) if o["path"].lower().endswith(".txt")]
            raw = _pick_timestamp_file(txts)
            if raw is None:
                raise RuntimeError("识别完成但没有时间戳文本输出：" + json.dumps(r, ensure_ascii=False)[:300])
            res = parse_timestamp_text(raw)
            if not res["words"]:
                raise RuntimeError("识别返回里没有字级时间戳：" + raw[:200])
            if cp:
                json.dump(res, open(cp, "w", encoding="utf-8"), ensure_ascii=False)
            return res
        last = (jmcai.error_text(r) or str(r.get("message") or ""))[:200]
        if st == "timeout" or attempt < C.MAX_TRIES:
            wait = min(3.0 * attempt, 20.0) + random.random()
            print("[ASR] %s %s，%.1fs 后重试（%d/%d）" % (st, last[:80], wait, attempt, C.MAX_TRIES), flush=True)
            time.sleep(wait)
            continue
        break
    raise RuntimeError("识别失败：" + last)


# ---------------------------------------------------------------- 文字比较用：统一数字写法
_CN = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_UNIT = {"十": 10, "百": 100, "千": 1000}


def _cn_to_int(s):
    if not any(ch in _UNIT for ch in s):          # 没有十百千：逐位读（年份"一九五六"→1956）
        return int("".join(str(_CN[ch]) for ch in s))
    total, cur = 0, 0
    for ch in s:
        if ch in _CN:
            cur = _CN[ch]
        elif ch in _UNIT:
            total += (cur or 1) * _UNIT[ch]
            cur = 0
    return total + cur


def norm_text(s):
    """去标点和空白，并把中文数字统一成阿拉伯数字（识别结果常把"三十"写成"30"）。"""
    s = re.sub(r"[一-鿿]*", lambda m: re.sub(r"[零〇一二两三四五六七八九十百千]+",
                                                     lambda n: str(_cn_to_int(n.group())), m.group()), s)
    return re.sub(r"[\s，。、！？!?；;：:“”\"'（）()\-—…,.]", "", s).lower()
