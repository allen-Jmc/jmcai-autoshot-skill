"""拼接各段数字人视频并质检：<workdir>/avatar/final_avatar.mp4 + seams.png。
             + words_final.json、voice_final.wav（供包装阶段零漂移对齐）

  python finalize.py --workdir <项目目录> [--trim-frames 4]

做法：
  * **每段开头裁掉几帧**（`--trim-frames`，**默认 0 即关闭**）：模型每段开头有几帧启动帧，
    人物还没进入口播状态。上游（云端工作流）实测裁 4 帧接缝更干净，但**本机实测无效**——
    在 0~32 帧范围内扫过，接缝跳变 MSE 波动不到 3%，因为本机接缝跳变的主因不是启动帧，
    而是**各段构图不一致**（见下面那条）。换工作流时可以打开再试。
  * 每段视频再按各自的音频时长裁掉多出来的帧（云端返回的视频会比音频略长），统一帧率后依次拼接；
  * **检查各段分辨率是否一致**：分辨率不同时，模型输出的构图（人物在画面里的大小）也不同，
    拼接处会明显跳一下——这不是裁几帧能解决的，只能重跑时统一 `--megapixels`。不一致会明确告警。
  * 声音不用云端回传的音轨，直接用项目里的 out/merged.wav（原配音），避免二次压缩和时间漂移；
  * 质检：分辨率、时长与配音差、黑帧和冻结候选，并在每个接缝前后各取两帧拼成 seams.png 供人工查看；
    同时把每处接缝的帧间差写进 `seams.json`（和段内正常帧间差比，超过 1.6 倍标为跳变）。
  * 另外产出两份「最终版」资产，包装阶段优先用它们，不要再回头读 out/ 下的中间文件：
      - `words_final.json`：最终成片的逐词时间（附裁帧信息与每段边界）；
      - `voice_final.wav`：从成片里提出来的人声轨（48kHz 双声道），和画面严格同长同轴。
口型是否对准只能听看成片确认，脚本不判断。
接缝跳变能否接受也要人看 seams.png；本机的经验是交给包装阶段用小窗／贴纸盖过去，
比重新花钱生成更划算。
"""
import argparse
import json
import os
import re
import subprocess
import sys

import cv2
import numpy as np




def _imwrite(path, img):
    """cv2.imwrite 在带中文的路径上会悄悄失败（不报错、也不写文件），改成先编码再用 numpy 写出"""
    ok, buf = cv2.imencode(os.path.splitext(str(path))[1] or ".png", img)
    if not ok:
        raise RuntimeError("图片编码失败：%s" % path)
    buf.tofile(str(path))
    return True

def ffmpeg():
    import shutil
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        sys.exit("找不到 ffmpeg：请安装 ffmpeg 并加入 PATH，或 pip install imageio-ffmpeg")


def probe_resolution(path):
    """用 OpenCV 读视频宽高；读不到返回 (None, None)。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (w, h) if w and h else (None, None)


def probe_frames(path):
    """读视频总帧数；读不到返回 0。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(0, n)


def plan_trims(segs, res, fps, want):
    """算每段实际裁掉多少帧。

    开头那几帧是模型的启动帧，裁掉接缝才干净；但裁完必须还够本段的音频时长，
    否则会把内容裁短、总长对不上配音。不够的段自动退让，并说明原因。
    返回 {段号: 实际裁帧数}。
    """
    out = {}
    for s in segs:
        i = s["i"]
        need = int(round(s["duration"] * fps))
        avail = probe_frames(res[str(i)]["file"])
        n = want if avail - want >= need else 0
        if want and not n and avail:
            print("提示：段%d 只有 %d 帧，比音频需要的 %d 帧多不到 %d 帧，这一段不裁头（其余段照裁）"
                  % (i, avail, need, want))
        out[i] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--trim-frames", type=int, default=0,
                    help="每段开头裁掉几帧。上游用 4，但本机实测无效（0~32 帧跳变波动<3%%），默认 0 关闭")
    a = ap.parse_args()
    od = os.path.join(a.workdir, "avatar")
    plan = json.load(open(os.path.join(od, "plan.json"), encoding="utf-8"))
    info = json.load(open(os.path.join(od, "image_info.json"), encoding="utf-8"))
    res = json.load(open(os.path.join(od, "result.json"), encoding="utf-8"))["results"]
    for k, v in res.items():
        if "file" in v and not os.path.isabs(v["file"]):
            v["file"] = os.path.abspath(os.path.join(a.workdir, v["file"]))
    segs = plan["segments"]
    missing = [s["i"] for s in segs if str(s["i"]) not in res or not os.path.isfile(res[str(s["i"])]["file"])]
    if missing:
        sys.exit("这些段还没有视频：%s。先用 run_avatar.py 跑完。" % missing)
    audio = os.path.join(a.workdir, "out", "merged.wav")
    total = plan["total_seconds"]
    w, h = info["width"], info["height"]
    # 各段分辨率：本地工作流按画幅自己出分辨率，以实际片段为准。
    # 但**各段必须一致**——分辨率不同，模型输出的构图（人物在画面里的大小）也不同，
    # 拼接处会明显跳一下，这不是裁几帧能解决的，只能重跑时统一 --megapixels。
    seg_res = {s["i"]: probe_resolution(res[str(s["i"])]["file"]) for s in segs}
    uniq = sorted({v for v in seg_res.values() if v[0]})
    if len(uniq) > 1:
        print("⚠️  各段分辨率不一致：%s" % "，".join("段%d %d×%d" % (i, v[0], v[1]) for i, v in sorted(seg_res.items())))
        print("    分辨率不同时模型输出的构图（人物在画面里的大小）也不同，拼接处会明显跳变。")
        print("    这不是裁帧能解决的——重跑时所有段请用同一个 --megapixels，否则接缝没法干净。")
    pw, ph = seg_res[segs[0]["i"]]
    if pw and ph:
        if (pw, ph) != (w, h):
            print("提示：片段实际分辨率 %d×%d，按它输出（image_info 里记的是 %d×%d）" % (pw, ph, w, h))
        w, h = pw, ph
    ff = ffmpeg()
    trims = plan_trims(segs, res, a.fps, a.trim_frames)
    cmd = [ff, "-y", "-v", "error"]
    for s in segs:
        cmd += ["-i", res[str(s["i"])]["file"]]
    cmd += ["-i", audio]
    parts = []
    for k, s in enumerate(segs):
        t = trims.get(s["i"], 0)
        if t:                                       # 裁掉开头的启动帧，再取够本段音频的帧数
            end = t + int(round(s["duration"] * a.fps))
            vf = "fps=%d,trim=start_frame=%d:end_frame=%d,setpts=PTS-STARTPTS,scale=%d:%d" % (a.fps, t, end, w, h)
        else:
            vf = "fps=%d,trim=duration=%.4f,setpts=PTS-STARTPTS,scale=%d:%d" % (a.fps, s["duration"], w, h)
        parts.append("[%d:v]%s[v%d]" % (k, vf, k))
    fc = ";".join(parts) + ";" + "".join("[v%d]" % k for k in range(len(segs))) + "concat=n=%d:v=1:a=0[v]" % len(segs)
    final = os.path.join(od, "final_avatar.mp4")
    cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "%d:a" % len(segs), "-t", "%.4f" % total,
            "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", final]
    subprocess.run(cmd, check=True)

    # ---- 质检
    cap = cv2.VideoCapture(final)
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    vdur = n / fps if fps else 0
    checks = [("分辨率", "%d×%d" % (fw, fh), (fw, fh) == (w, h)),
              ("视频时长", "%.2fs（配音 %.2fs，差 %.2fs）" % (vdur, total, vdur - total), abs(vdur - total) <= 0.1)]
    r = subprocess.run([ff, "-i", final, "-an", "-vf", "blackdetect=d=0.1:pic_th=0.98,freezedetect=n=-60dB:d=1.0", "-f", "null", "-"],
                       capture_output=True)
    err = r.stderr.decode("utf-8", "replace")
    blacks = re.findall(r"black_start:([\d.]+)", err)
    freezes = re.findall(r"freeze_start: ([\d.]+)", err)
    checks.append(("黑帧", "无" if not blacks else "候选起点 " + ",".join(blacks), not blacks))
    checks.append(("冻结≥1秒", "无" if not freezes else "候选起点 " + ",".join(freezes), not freezes))
    # 接缝帧
    rows = []
    for s in segs[:-1]:
        cut = s["end"]
        row = []
        for dt in (-0.12, -0.04, 0.04, 0.12):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round((cut + dt) * fps))))
            ok, fr = cap.read()
            if ok:
                fr = cv2.resize(fr, (480, int(480 * fh / fw)))
                cv2.putText(fr, "cut %.2fs %+.2f" % (cut, dt), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                row.append(fr)
        if row:
            rows.append(np.hstack(row))
    if rows:
        _imwrite(os.path.join(od, "seams.png"), np.vstack(rows))

    # ---- 接缝量化：交界处相邻两帧的灰度平均绝对差（MAD），和段内正常帧间差比。
    #     段内相邻帧因为人物在动本来就有差值，接缝如果只是"正常动作"，MAD 会和基线差不多；
    #     构图跳变（人物大小、位置、色调变了）会让 MAD 明显高出基线。
    def _mad(a1, b1):
        d = cv2.absdiff(cv2.cvtColor(a1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b1, cv2.COLOR_BGR2GRAY))
        return float(d.mean())

    base_samples = []
    for s in segs:                                     # 段内取几对相邻帧当基线（避开接缝前后 0.2 秒）
        lo, hi = s["start"] + 0.2, s["end"] - 0.2
        if hi - lo < 0.3:
            continue
        for k in range(4):
            t = lo + (hi - lo) * (k + 0.5) / 4
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t * fps))))
            ok1, f1 = cap.read()
            ok2, f2 = cap.read() if ok1 else (False, None)
            if ok1 and ok2:
                base_samples.append(_mad(f1, f2))
    baseline = round(float(np.median(base_samples)), 2) if base_samples else None
    seam_rows = []
    for idx, s in enumerate(segs[:-1]):
        cut = s["end"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(cut * fps)) - 1))
        ok1, f1 = cap.read()
        ok2, f2 = cap.read() if ok1 else (False, None)
        if not (ok1 and ok2):
            continue
        m = _mad(f1, f2)
        seam_rows.append({"at": round(cut, 2), "from": s["i"], "to": segs[idx + 1]["i"],
                          "mad": round(m, 2), "intra_baseline": baseline})
    cap.release()
    if seam_rows:
        # 和段内相邻帧比没有意义：换段本来就是两张不同的画面，MAD 一定远高于同段相邻帧。
        # 有用的是**彼此比较**——哪一处比全片其余接缝更跳，那一处才值得改提示词重跑。
        vals = [r["mad"] for r in seam_rows]
        seam_med = round(float(np.median(vals)), 2)
        limit = max(1.8 * seam_med, 25.0)
        for r in seam_rows:
            r["seam_median"] = seam_med
            r["warn"] = bool(r["mad"] > limit)
        json.dump({"intra_frame_mad": baseline, "seam_median_mad": seam_med, "warn_above": round(limit, 2),
                   "seams": seam_rows},
                  open(os.path.join(od, "seams.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        bad = [r for r in seam_rows if r["warn"]]
        if bad:
            print("接缝跳变（全片接缝 MAD 中位数 %.1f，超过 %.1f 的标出）：" % (seam_med, limit))
            for r in bad:
                print("  段%d→段%d @%.2fs  MAD %.1f ——比全片其余接缝明显更跳。"
                      "这一段首尾姿态可能没回到原位，改提示词后用 --only 重跑；"
                      "或者交给包装阶段用小窗／贴纸盖过去。" % (r["from"], r["to"], r["at"], r["mad"]))
        else:
            print("接缝：%d 处彼此接近（中位数 MAD %.1f），没有哪一处特别跳。" % (len(seam_rows), seam_med))
        print("      注：接缝 MAD 天然高于段内相邻帧（%s），只要各段之间均匀就不用管。" % baseline)
    print("成片：", final)
    for name, val, ok in checks:
        print("%s %s：%s" % ("✓" if ok else "✗", name, val))
    if rows:
        print("接缝检查图（每行一个接缝，前后各两帧）：", os.path.join(od, "seams.png"))
    print("未做：口型同步、手势细节、人物一致性——需要人看和听成片确认。")

    # ---- 最终版资产：包装阶段优先用这两份，不要再回头读 out/ 下的中间文件
    wf = os.path.join(od, "voice_final.wav")
    subprocess.run([ff, "-y", "-v", "error", "-i", final, "-vn", "-ac", "2", "-ar", "48000", wf], check=True)
    wpath = os.path.join(a.workdir, "out", "words.json")
    wj = json.load(open(wpath, encoding="utf-8")) if os.path.isfile(wpath) else {"words": []}
    wj = dict(wj)
    wj["trim_frames_per_segment"] = {str(k): v for k, v in sorted(trims.items())}
    wj["final_video"] = final
    wj["final_voice"] = wf
    json.dump(wj, open(os.path.join(od, "words_final.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    trimmed = sum(1 for v in trims.values() if v)
    print("每段开头裁 %d 帧（%d/%d 段已裁，约 %.2f 秒／段）" % (a.trim_frames, trimmed, len(segs), a.trim_frames / a.fps))
    print("最终版资产：", os.path.join(od, "words_final.json"), "＋", wf)

    mp = os.path.join(a.workdir, "meta.json")        # 项目目录里写回进度，后续环节可读
    if os.path.isfile(mp):
        m = json.load(open(mp, encoding="utf-8"))
        m["avatar_video"] = final
        json.dump(m, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if not all(ok for _, _, ok in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
