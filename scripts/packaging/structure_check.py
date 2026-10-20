"""包装工程的结构检查：字幕排得对不对、场景有没有重叠（重影）、章节时间轴乱没乱、字号够不够大。

  python structure_check.py --workdir <项目目录>
  python structure_check.py --workdir <项目目录> --max-chars 22 --sub-band 0.20 --out sc.json

`qc_scan.py` / `av_check.py` / `density_check.py` 都只看"画面糊不糊、空不空、有没有黑帧"，
**查不出内容排布上的错**。这份脚本补的就是这一类——它们和审美无关，是算得出来的硬伤：

  A 字幕数据（src/data/captions.json）
    超长（一行放不下，会换行或溢出）、停留过短（读不完）、时间倒序或重叠、超出片长、空白条
  B 场景重叠（src/scenes/*.tsx）——**这是重影的来源**
    每场用 `const tStart = T('…') ± n` / `const tEnd = T('…') ± n` 划定自己的时间区间。
    上一场的 tEnd 晚于下一场的 tStart，两场就会同时渲染；两场的面板在同一位置，
    于是能看到两层标题和数字压在一起（实测 V1 每处重叠 0.4 秒，12 个切点全中）。
  C 章节时间轴（Film.tsx 的 CHAPTERS）：时间必须递增、不能超出片长
  D 字幕字号：低于画面短边的 5% 在手机上读不清（design.md 的要求，这里把它变成可检查的）

B 和 D 是扫源码的，属于尽力而为：写法不一样就跳过并说明，不会假装检查过。
A 和 C 读的是数据文件，只要文件在就一定查得准。

退出码：0 没有 error；1 有 error；2 读不到工程。
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

# 和 lib/words.ts 的 STRIP 保持一致：只保留真正的字，标点和空白都去掉
STRIP = re.compile(r"[\s，。、！？!?；;：:“”\"'（）()\-—…,.]")
# `T('…')` / `T('…', 2)`（第二次说到） + 可选的 `± n` 偏移，或者干脆是一个数字
T_EXPR = re.compile(r"T\(\s*['\"]([^'\"]+)['\"]\s*(?:,\s*(\d+)\s*)?\)\s*(?:([+-])\s*([\d.]+))?")
NUM_EXPR = re.compile(r"^\s*(-?[\d.]+)\s*$")
RANGE = re.compile(r"(tStart|tEnd)\s*=\s*([^;\n]+)")
SCENE_SPLIT = re.compile(r"export const (\w+)\s*:\s*React\.FC")
CHAPTERS = re.compile(r"\{\s*name\s*:\s*['\"]([^'\"]*)['\"]\s*,\s*at\s*:\s*([^}]+)\}")
FONT_SIZE = re.compile(r"fontSize\s*:\s*([A-Za-z_][\w]*|\d+(?:\.\d+)?)\s*\*\s*u")
# 字幕专用文件里还常写成 `const size = 54 * u` + `fontSize: size`（没有 `* u` 紧跟其后）。
# 不认这种写法就会静默漏检——实测就是这样漏掉了一条 39.5px（4.3%）的字幕。
FONT_SIZE_BARE = re.compile(r"fontSize\s*:\s*([A-Za-z_][\w]*)")
SUB_FILES = ("src/parts/Subs.tsx", "src/core/Subtitles.tsx")
# 另一种常见写法：直接写字面量区间 `if (t < 52.4 || t > 63.4) return null;`（两个比较符的顺序也可能反着写）
BOUNDS = re.compile(r"if\s*\(\s*t\s*([<>])\s*([\d.]+)\s*\|\|\s*t\s*([<>])\s*([\d.]+)\s*\)\s*return\s+null")
TAIL_ONLY = re.compile(r"if\s*\(\s*t\s*>\s*([\d.]+)\s*\)\s*return\s+null")
OVERLAP_TOL = 0.02          # 秒，浮点误差
MIN_CAPTION_MS = 600        # 一条字幕至少停留这么久才读得完
CAPTION_OVERLAP_MS = 150    # 字幕之间允许这么多重叠（captions_from_copy 会留 60ms 做淡接）
FONT_MIN_RATIO = 0.05       # 字号不得小于画面短边的 5%
                            # 原为 4%：928 短边做 37px，缩到 390px 手机宽只剩 8.7px，等于没有字幕。
                            # 46px(5%) 缩下去 10.8px 才是可读下限。用户反馈过"手机端字号偏小"就是照 4% 做的。


def fail(msg):
    """读不到工程 / 读不懂文件：退出码 2，和"查出 error"（1）区分开。"""
    print(msg, file=sys.stderr)
    sys.exit(2)


def load(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as e:
        fail("%s 不是合法 JSON：%s" % (path, e))


class Clock:
    """复刻 lib/words.ts 的 T()：把逐词时间摊成"每个字一个时间点"，按串查第一次出现的时刻。"""

    def __init__(self, words):
        self.chars, self.times = [], []
        for w in words:
            txt = STRIP.sub("", (w.get("text") or w.get("word") or ""))
            n = max(1, len(txt))
            for k, ch in enumerate(txt):
                self.chars.append(ch.lower())
                self.times.append(w["start"] + (w["end"] - w["start"]) * k / n)
        self.joined = "".join(self.chars)

    def at(self, key, occurrence=1):
        k = STRIP.sub("", key).lower()
        i = -1
        for _ in range(occurrence):
            i = self.joined.find(k, i + 1)
            if i < 0:
                return None
        return self.times[i]


def resolve(expr, clock):
    """把 `T('…') - 0.2`、`T('…', 2)`、`3.5` 这类写法解析成秒；解析不了返回 None。"""
    m = T_EXPR.search(expr)
    if m:
        t = clock.at(m.group(1), int(m.group(2) or 1))
        if t is None:
            return None
        off = float(m.group(4)) if m.group(4) else 0.0
        return t + (-off if m.group(3) == "-" else off if m.group(3) == "+" else 0.0)
    m = NUM_EXPR.match(expr)
    return float(m.group(1)) if m else None


def check_captions(caps, duration, max_chars, errors, warns):
    if not caps:
        warns.append("没有 src/data/captions.json —— 这条片子没有字幕？还是没跑 remotion_prepare.py？")
        return
    prev = None
    for i, c in enumerate(caps, 1):
        text = (c.get("text") or "").strip()
        s, e = c.get("startMs"), c.get("endMs")
        if not text:
            warns.append("字幕第 %d 条是空的" % i)
            continue
        if s is None or e is None:
            errors.append("字幕第 %d 条缺 startMs / endMs：%s" % (i, text[:20]))
            continue
        if len(text) > max_chars * 1.15:
            warns.append("字幕第 %d 条 %d 字，超过一行的 %d 字（会换行或溢出）：%s"
                         % (i, len(text), max_chars, text[:24]))
        if e - s < MIN_CAPTION_MS:
            warns.append("字幕第 %d 条只停留 %d ms，读不完：%s" % (i, e - s, text[:24]))
        if e <= s:
            errors.append("字幕第 %d 条 endMs 不大于 startMs：%s" % (i, text[:20]))
        if prev is not None:
            if s < prev["startMs"] - 50:
                errors.append("字幕第 %d 条比上一条开始得还早（时间倒序）：%s" % (i, text[:20]))
            elif s < prev["endMs"] - CAPTION_OVERLAP_MS:
                warns.append("字幕第 %d 条和上一条重叠了 %d ms（超过允许的 %d ms，会同时出现两条）：%s"
                             % (i, prev["endMs"] - s, CAPTION_OVERLAP_MS, text[:20]))
        if e > duration * 1000 + 100:
            errors.append("字幕第 %d 条结束于 %.2fs，超出片长 %.2fs：%s"
                          % (i, e / 1000, duration, text[:20]))
        prev = {"startMs": s, "endMs": e}


def check_scenes(pkg, clock, errors, warns):
    """扫源码里的 tStart / tEnd（或 `if (t < a || t > b) return null`），找相邻两场重叠——重影就是这么来的。

    场景可能写在 `src/scenes/*.tsx` 里，也可能全写在一个 `src/scenes.tsx` 或 `Film.tsx` 里，
    所以扫的是 `src/` 下所有 tsx，而不是只认一个目录。
    """
    d = Path(pkg) / "src"
    if not d.is_dir():
        warns.append("没有 src/ 目录，跳过场景重叠检查")
        return
    # core/ 和 lib/ 是模板给的元件（人物层、字幕、时间轴），不是场景，不参与重叠检查
    files = sorted(p for p in d.rglob("*.tsx")
                   if not any(part in ("core", "lib") for part in p.relative_to(d).parts))
    if not files:
        warns.append("src/ 下没有 tsx 文件，跳过场景重叠检查")
        return
    spans, skipped = [], []
    for f in files:
        src = f.read_text(encoding="utf-8")
        # 一个文件里可能写了好几场：按 `export const X: React.FC` 切开，逐块判断
        marks = [(m.start(), m.group(1)) for m in SCENE_SPLIT.finditer(src)]
        chunks = []
        for k, (pos, name) in enumerate(marks):
            end = marks[k + 1][0] if k + 1 < len(marks) else len(src)
            chunks.append((name, src[pos:end]))
        if not chunks:
            chunks = [(f.stem, src)]
        for name, text in chunks:
            # 只有"按时间 t 提前返回"的块才可能是场景；别的元件（字幕、时间轴）不算
            is_scene = bool(re.search(r"if\s*\(\s*t\s*[<>]", text)) or "tStart" in text
            got = {}
            for kind, expr in RANGE.findall(text):
                t = resolve(expr, clock)
                if t is None:
                    skipped.append("%s:%s 的 %s 解析不出时间（%s）" % (f.name, name, kind, expr.strip()[:30]))
                    continue
                got[kind] = t
            if "tStart" in got and "tEnd" in got:
                spans.append({"file": "%s:%s" % (f.name, name), "start": got["tStart"], "end": got["tEnd"]})
                continue
            m = BOUNDS.search(text)                 # 退而求其次：字面量区间
            if m:
                op1, n1, op2, n2 = m.group(1), float(m.group(2)), m.group(3), float(m.group(4))
                lo, hi = (n1, n2) if op1 == "<" else (n2, n1)
                spans.append({"file": "%s:%s" % (f.name, name), "start": lo, "end": hi})
                continue
            m = TAIL_ONLY.search(text)              # 只有结束条件：从 0 开始
            if m:
                spans.append({"file": "%s:%s" % (f.name, name), "start": 0.0, "end": float(m.group(1))})
                continue
            if is_scene:
                skipped.append("%s:%s 按时间提前返回，但区间不是 tStart/tEnd 也不是字面量" % (f.name, name))
    if not spans:
        warns.append("一个场景的 tStart/tEnd 都没解析出来，**场景重叠没有检查**（%d 个文件写法不同）"
                     "——重影要自己看一遍静帧" % len(skipped))
        return
    spans.sort(key=lambda s: s["start"])
    for a, b in zip(spans, spans[1:]):
        ov = a["end"] - b["start"]
        if ov > OVERLAP_TOL:
            errors.append("场景重叠 %.2fs：%s 到 %.2fs，但 %s 从 %.2fs 就开始——"
                          "两场的面板在同一位置，这 %.2f 秒里会看到两层字压在一起（重影）。"
                          "让上一场的 tEnd 等于下一场的 tStart。"
                          % (ov, a["file"], a["end"], b["file"], b["start"], ov))
    total = len(spans) + len(skipped)
    if skipped:
        # 只报"跳过了哪个文件"太容易被忽略：覆盖率不足时，没覆盖的那几处等于没检查
        warns.append("**场景重叠只覆盖了 %d/%d 个场景**，下面这些写法没解析出来、没参与检查，"
                     "它们的接缝要自己看静帧确认：" % (len(spans), total))
        for s in skipped:
            warns.append("  · " + s)
    else:
        # 全部解析成功时也要报出来：不报的话，"没有发现结构问题"这句话
        # 分不清是真没问题、还是其实一个场景都没扫到（曾经就是这样被误信的）
        print("场景重叠检查：%d 个场景全部解析成功（%s）"
              % (len(spans), "、".join(s["file"].split(":")[-1] for s in spans)))


def check_chapters(pkg, clock, duration, errors, warns):
    f = Path(pkg) / "src" / "Film.tsx"
    if not f.is_file():
        warns.append("没有 src/Film.tsx，跳过章节时间轴检查")
        return
    src = f.read_text(encoding="utf-8")
    block = src[src.find("CHAPTERS"):] if "CHAPTERS" in src else ""
    rows = []
    for name, expr in CHAPTERS.findall(block):
        t = resolve(expr, clock)
        if t is None:
            warns.append("章节「%s」的时间解析不出来（%s）" % (name, expr.strip()[:40]))
            continue
        rows.append((name, t))
    if not rows:
        warns.append("Film.tsx 里没有解析出章节（CHAPTERS 可能写成了别的形式）")
        return
    for (n1, t1), (n2, t2) in zip(rows, rows[1:]):
        if t2 <= t1:
            errors.append("章节「%s」(%.2fs) 不比「%s」(%.2fs) 晚，时间轴会倒着走" % (n2, t2, n1, t1))
    if rows[-1][1] > duration + 0.05:
        errors.append("最后一章「%s」在 %.2fs，超出片长 %.2fs" % (rows[-1][0], rows[-1][1], duration))


def check_font_size(pkg, size, warns):
    """字幕字号不得小于画面短边的 5%。认 `fontSize: 46 * u` 和 `fontSize: size * u`（再看 size 的默认值）。"""
    short = min(size) if size else 0
    if not short:
        return
    need = short * FONT_MIN_RATIO
    found, unknown = [], []
    for rel in ("src/parts/Subs.tsx", "src/core/Subtitles.tsx", "src/Film.tsx"):
        f = Path(pkg) / rel
        if not f.is_file():
            continue
        src = f.read_text(encoding="utf-8")
        toks = FONT_SIZE.findall(src)
        if rel in SUB_FILES:
            # 字幕专用文件里的 fontSize 都算字号，包括 `fontSize: size` 这种变量写法
            for tok in FONT_SIZE_BARE.findall(src):
                if tok not in toks:
                    toks.append(tok)
        for tok in toks:
            try:
                num = float(tok)
            except ValueError:                       # 变量：找它的默认值
                # `const size = 54 * u` 和 `useState(54)` 两种定义方式都要认
                m = (re.search(r"\b%s\b\s*(?::\s*number\s*)?=\s*([\d.]+)\s*\*\s*u" % re.escape(tok), src)
                     or re.search(r"\b%s\b\s*(?::\s*number\s*)?=\s*([\d.]+)" % re.escape(tok), src))
                if not m:
                    unknown.append("%s 的 fontSize 写成 `%s`，找不到 %s 的数值" % (rel, tok, tok))
                    continue
                num = float(m.group(1))
            found.append((rel, num, num * short / 1080.0))   # u = 短边 / 1080
    if not found:
        for u in unknown:
            warns.append(u)
        warns.append("没算出字幕字号，跳过下限检查——自己确认字幕不小于短边的 %d%%（%.0fpx）"
                     % (FONT_MIN_RATIO * 100, need))
        return
    for rel, num, px in found:
        if px < need * 0.95:
            warns.append("%s 的字幕字号 %g×u ≈ %.0fpx，低于短边 5%%（%.0fpx）——手机上读不清"
                         % (rel, num, px, need))


def main():
    ap = argparse.ArgumentParser(description="包装工程结构检查")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--max-chars", type=int, default=None,
                    help="字幕一行最多几个字；不给就按画幅推（横屏 22、竖屏 14）")
    ap.add_argument("--out", help="把结果写成 JSON")
    a = ap.parse_args()

    pkg = os.path.join(a.workdir, "package")
    if not os.path.isdir(pkg):
        fail("找不到 %s：先跑 remotion_new.py 起工程。" % pkg)
    if not os.path.isfile(os.path.join(pkg, "src", "data", "film.json")):
        fail("工程里还没有 src/data/film.json：素材还没放进来，先跑 remotion_new.py"
             "（或对已有工程重跑一次 remotion_prepare.py），再来做结构检查。")
    film = load(os.path.join(pkg, "src", "data", "film.json"), {})
    words = load(os.path.join(pkg, "src", "data", "words.json"), {})
    caps = load(os.path.join(pkg, "src", "data", "captions.json"), [])
    if isinstance(caps, dict):
        caps = caps.get("captions") or caps.get("items") or []
    duration = float(film.get("duration") or 0)
    size = [int(film.get("width") or 0), int(film.get("height") or 0)]
    max_chars = a.max_chars or (14 if size[1] > size[0] else 22)

    errors, warns = [], []
    clock = Clock((words or {}).get("words") or [])
    if not clock.chars:
        warns.append("没有 src/data/words.json 的逐词时间，场景与章节的时间锚点无法解析")

    if not duration:
        errors.append("film.json 里没有 duration，没法判断字幕和章节有没有越界")
    else:
        check_captions(caps, duration, max_chars, errors, warns)
        check_chapters(pkg, clock, duration, errors, warns)
    if clock.chars:
        check_scenes(pkg, clock, errors, warns)
    check_font_size(pkg, size, warns)

    print("结构检查：%s（%dx%d，%.1fs，字幕 %d 条，一行 %d 字）"
          % (pkg, size[0], size[1], duration, len(caps), max_chars))
    for e in errors:
        print(" [错] " + e)
    for w in warns:
        print(" [看] " + w)
    if not errors and not warns:
        print(" 没有发现结构问题。")
    if a.out:
        Path(a.out).write_text(json.dumps({"errors": errors, "warnings": warns,
                                           "duration": duration, "size": size,
                                           "captions": len(caps), "max_chars": max_chars},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
