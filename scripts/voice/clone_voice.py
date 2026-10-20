"""登记本阶段要用的音色。结果存到 <workdir>/voice.json，并记为用户默认音色（<根>/voice.json）。

  python clone_voice.py --sample 我的声音.wav --workdir <项目目录>      # 用你自己的声音（声音克隆，默认）
  python clone_voice.py --workdir <项目目录>                            # 复用默认音色
  python clone_voice.py --sample 新声音.wav --workdir <项目目录> --force  # 换声音
  python clone_voice.py --sample 我的声音.wav --workdir <项目目录> --dry-run  # 只做本地检查，不登记
  python clone_voice.py --speaker Serena --workdir <项目目录>            # 用官方预设音色（不用样本）
  python clone_voice.py --design "低沉磁性的中年男声，语速偏慢" --workdir <项目目录>   # 自然语言造音色
  python clone_voice.py --list                                          # 列出用过的所有音色
  python clone_voice.py --workdir <项目目录> --use 2                     # 换回某个以前的音色，不用重新给样本

平替说明：原版这里会调用百炼的 voice-enrollment 去云端注册音色、拿一个 voice_id。
本地 Qwen3-TTS 的工作流不需要预先注册——**参考音频本身就是音色**，每次合成时随请求一起送进去。
所以这一步现在只做三件事：校验样本、把样本复制进项目目录固定下来、把音色信息写进 voice.json。
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import acoustic_gate

HIST = os.path.join(C.WORKSPACE_ROOT, "voice_history.json")
MIME = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".flac": "audio/flac", ".ogg": "audio/ogg", ".aac": "audio/aac"}
MIN_SECONDS = 3.0        # 低于这个长度克隆质量会明显掉
WARN_SECONDS = 30.0      # 工作流的参考音频截取上限默认 30 秒，超过会被截断


def _read(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def history():
    """用过的所有音色：历史记录文件，加上默认音色和各项目里的 voice.json。按创建时间排序、去重。"""
    cands = []
    h = _read(HIST)
    cands += h if isinstance(h, list) else []
    cands.append(_read(os.path.join(C.WORKSPACE_ROOT, "voice.json")))
    proj = os.path.join(C.WORKSPACE_ROOT, "projects")
    if os.path.isdir(proj):
        cands += [_read(os.path.join(proj, d, "voice.json")) for d in sorted(os.listdir(proj))]
    seen, rows = set(), []
    for c in cands:
        if isinstance(c, dict) and c.get("voice_id") and c["voice_id"] not in seen:
            seen.add(c["voice_id"])
            rows.append(c)
    return sorted(rows, key=lambda r: r.get("created_at", ""))


def label_of(info):
    """给用户看的一行说明。"""
    mode = info.get("mode", "clone")
    if mode == "preset":
        return "预设音色 %s" % info.get("speaker", "?")
    if mode == "design":
        return "声音设计「%s」" % (info.get("instruct", "")[:20] or "?")
    prof = info.get("acoustic_profile")
    f0_str = ("，F0 %.0fHz" % prof["f0_median"]) if (prof and prof.get("f0_median")) else ""
    return "克隆自 %s（%.1f 秒%s）" % (os.path.basename(info.get("sample", "?")), info.get("sample_seconds") or 0, f0_str)


def register(workdir, info):
    """把音色写进项目目录与默认音色文件，并追加到历史。"""
    vj = os.path.join(workdir, "voice.json")
    json.dump(info, open(vj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.makedirs(C.WORKSPACE_ROOT, exist_ok=True)
    json.dump(info, open(os.path.join(C.WORKSPACE_ROOT, "voice.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(history(), open(HIST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    C.update_meta(workdir, voice_id=info["voice_id"], voice_mode=info.get("mode", "clone"))
    return vj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample")
    ap.add_argument("--speaker")
    ap.add_argument("--design")
    ap.add_argument("--workdir")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--use", help="换回以前用过的音色：序号、样本文件名里的字或音色标识")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    default_vj = os.path.join(C.WORKSPACE_ROOT, "voice.json")

    # ---- 列出 / 换回
    if a.list or a.use:
        rows = history()
        if a.list or not rows:
            for i, r in enumerate(rows, 1):
                print("%d  %s  %s  %s" % (i, r.get("created_at", ""), label_of(r),
                                          "  ← 当前默认" if r["voice_id"] == (_read(default_vj) or {}).get("voice_id") else ""))
            if not rows:
                print("还没有登记过音色")
            return
        if not a.workdir:
            sys.exit("--use 需要同时给 --workdir")
        # 序号必须优先精确匹配：a.use="4" 这种单个数字，用 `in voice_id` 去模糊匹配时
        # 几乎必然会命中路径/时间戳里的同一个数字字符，结果 `--use 4` 报"匹配到 2 个"。
        # 实测就是这么坏的——而 --use <序号> 正是文档最推荐的用法。
        if a.use.isdigit() and 1 <= int(a.use) <= len(rows):
            picked, exact = rows[int(a.use) - 1], True
        else:
            # 模糊匹配只看"用户看得见的东西"：原始样本文件名、预设音色名、声音设计描述。
            # 不能拿 voice_id 来匹配——那是项目内的副本路径，样本一律被改成 voice_sample.<ext>，
            # 于是输入 "voice_sample" 会命中所有历史记录，这个功能等于失效。
            hit = [r for r in rows
                   if a.use in os.path.basename(r.get("sample", "") or "")
                   or a.use in (r.get("speaker") or "") or a.use in (r.get("instruct") or "")]
            if len(hit) != 1:
                # 光说"匹配到 N 个"没用，用户还得再跑一遍 --list 去数。
                # 常见情形是多条历史都指向同一个样本文件（同一段声音登记过几次），按文件名永远匹配不上。
                msg = ["没有唯一匹配的音色（匹配到 %d 个）：" % len(hit)]
                for i, r in enumerate(rows, 1):
                    if r in hit:
                        msg.append("  %d  %s  %s" % (i, r.get("created_at", ""), label_of(r)))
                msg.append("用 --use <序号> 指定，序号就是上面行首的数字。")
                sys.exit("\n".join(msg))
            picked, exact = hit[0], False
        os.makedirs(a.workdir, exist_ok=True)
        register(a.workdir, picked)
        print("已换成：%s，并记为默认音色%s" % (label_of(picked), "" if exact else ""))
        return

    if not a.workdir:
        sys.exit("需要 --workdir")
    os.makedirs(a.workdir, exist_ok=True)
    vj = os.path.join(a.workdir, "voice.json")

    # ---- 已有音色：直接复用
    if os.path.isfile(vj) and not a.force and not a.dry_run and not (a.sample or a.speaker or a.design):
        info = json.load(open(vj, encoding="utf-8"))
        if info.get("mode", "clone") == "clone" and not info.get("acoustic_profile"):
            acoustic_gate.ensure_voice_profile(info, a.workdir, save_back=True)
        print("复用已有音色：%s" % label_of(info))
        return

    # ---- 预设音色（不需要样本）
    if a.speaker:
        info = {"voice_id": "preset:%s" % a.speaker, "mode": "preset", "speaker": a.speaker,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        if a.dry_run:
            print("[dry-run] 预设音色 %s，未登记。" % a.speaker)
            return
        register(a.workdir, info)
        print("已登记预设音色：%s（同时记为用户默认音色）" % a.speaker)
        return

    # ---- 声音设计（不需要样本）
    if a.design:
        info = {"voice_id": "design:%s" % a.design[:24], "mode": "design", "instruct": a.design,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        if a.dry_run:
            print("[dry-run] 声音设计「%s」，未登记。" % a.design)
            return
        register(a.workdir, info)
        print("已登记声音设计音色：「%s」（同时记为用户默认音色）" % a.design)
        return

    # ---- 没有样本时：沿用默认音色
    if not a.sample:
        if os.path.isfile(default_vj) and not a.dry_run:
            info = json.load(open(default_vj, encoding="utf-8"))
            if info.get("mode", "clone") == "clone" and not info.get("acoustic_profile"):
                acoustic_gate.ensure_voice_profile(info, a.workdir, save_back=True)
            register(a.workdir, info)
            print("复用用户默认音色：%s" % label_of(info))
            return
        sys.exit("没有可复用的音色，需要提供一段人声样本（--sample），或用 --speaker / --design")

    # ---- 声音克隆：校验样本
    if not os.path.isfile(a.sample):
        sys.exit("找不到音频文件：" + a.sample)
    ext = os.path.splitext(a.sample)[1].lower()
    if ext not in MIME:
        sys.exit("格式不支持：%s（支持 wav / mp3 / m4a / flac / ogg / aac）" % ext)
    secs = C.media_seconds(a.sample)
    if secs is None:
        sys.exit("读不到音频时长，文件可能已损坏")
    if secs < MIN_SECONDS:
        sys.exit("音频只有 %.1f 秒，太短了（建议 5–15 秒的连续朗读），请重新提供" % secs)
    size = os.path.getsize(a.sample)
    print("参考音频：%.1f 秒，%.0f KB，%s" % (secs, size / 1024, ext))
    if secs > WARN_SECONDS:
        print("注意：超过 %.0f 秒的部分会被工作流截掉（参考音频截取上限），建议给 5–15 秒。" % WARN_SECONDS)

    if a.dry_run:
        print("[dry-run] 本地检查通过，未登记。")
        return

    # 把样本复制进项目目录固定下来（原文件被移动/删除后还能重跑）
    kept = os.path.join(a.workdir, "voice_sample" + ext)
    if os.path.abspath(a.sample) != os.path.abspath(kept):
        shutil.copy2(a.sample, kept)
    info = {"voice_id": kept, "mode": "clone", "sample": os.path.abspath(a.sample),
            "sample_seconds": round(secs, 1), "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        prof = acoustic_gate.extract_profile(kept)
        info["acoustic_profile"] = prof
        print("声学基准已提取：F0 中位数 %.1fHz (均值 %.1fHz)，有效发音比 %.1f%%" % (
            prof.get("f0_median", 0.0), prof.get("f0_mean", 0.0), prof.get("f0_voiced_ratio", 0.0) * 100))
    except Exception as e:
        print("注意：提取声学基准特征时出现提示：%s（将在运行时补全）" % e)
    register(a.workdir, info)
    print("音色已登记：%s\n样本已存到：%s\n（同时记为用户默认音色）" % (label_of(info), kept))


if __name__ == "__main__":
    main()
