"""数字人阶段的自动重拍闭环：质检判定哪几段不行 → 换种子重拍那几段 → 重新拼接 → 再质检，最多两轮。

  python auto_retake.py --workdir <项目目录>              # 预览：跑一遍质检，说要重拍哪几段，不提交
  python auto_retake.py --workdir <项目目录> --yes        # 真跑
  python auto_retake.py --workdir <项目目录> --pick-best  # 不重拍，只在已有的多个版本里挑最好的拼回去

它做的是 qa_video.py 已经算完、但以前没人接的那一步：把 `qa.json` 里的 `retake` 变成实际动作。
判定规则（哪些问题该重拍、哪些不该）见 qa_video.py 开头的说明和 references/avatar/guide.md。

三条边界，都是为了不把时间烧在没用的重拍上：
  * **单段最多重拍 MAX_RETAKES 次**（沿用 run_avatar 的上限），到顶就停，提示改提示词而不是再抽一次；
  * **一轮要重拍的段数超过 `--max-segments`（默认 3）或全片一半**就停——那说明是提示词或底图的问题，
    不是抽卡运气的问题，继续抽只会一直错；
  * **旧版本一律保留**（run_avatar 会归档到 clips/takes/），最后按清晰度和嘴动量挑最好的拼回去，
    抽卡可能越抽越差，不能默认"新的更好"。

退出码：0 正常；2 有段到上限或触发停止条件（需要人决定）；3 找不到可用的质检结果。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qa_video                      # noqa: E402  复用它的清晰度口径和人脸定位

MAX_RETAKES = 2                      # 和 run_avatar.MAX_RETAKES 保持一致
TAKE_SAMPLES = 12                    # 给一个片段打分时抽几帧算清晰度


def run(cmd, quiet=False):
    """跑一个同目录脚本。0/1/2 都算正常返回（1 是 finalize 的质检告警，2 是本脚本自己的「还有段没过」），
    3 以上才当成真的失败。"""
    r = subprocess.run([sys.executable] + cmd, capture_output=quiet, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode >= 3:
        sys.stderr.write((r.stderr or "")[-1500:] + "\n")
        raise SystemExit("命令失败（%d）：%s" % (r.returncode, " ".join(cmd)))
    return r


def load(path, default=None):
    if not os.path.isfile(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def score_clip(path):
    """给一个片段打分：清晰度（拉普拉斯方差中位数）+ 嘴部动作量（帧差中位数）。

    嘴部区域用 qa_video 的人脸定位，定位不到就按人物居中估算——这里只用来**比较几个版本**，
    绝对值和阈值无关，所以估算也够用。
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(3)), int(cap.get(4))
    if n < 2 or not W or not H:
        cap.release()
        return None
    fb = qa_video.face_box(str(path))
    if fb is not None:
        fx, fy, fw, fh = fb
        mb = (fx + int(.25 * fw), fx + int(.75 * fw), fy + int(.68 * fh), fy + int(.98 * fh))
    else:
        mb = (int(.42 * W), int(.58 * W), int(.46 * H), int(.58 * H))
    picks = set(np.linspace(0, n - 1, min(TAKE_SAMPLES, n)).astype(int).tolist())
    sharp, mouth, prev = [], [], None
    for k in range(n):
        ok, f = cap.read()
        if not ok:
            break
        g8 = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)          # Laplacian 要用 uint8，float32 在这个 OpenCV 构建上会报错
        if k in picks:
            sharp.append(float(cv2.Laplacian(g8, cv2.CV_64F).var()))
        g = g8.astype(np.float32)
        if prev is not None:
            x0, x1, y0, y1 = mb
            mouth.append(float(np.abs(g[y0:y1, x0:x1] - prev[y0:y1, x0:x1]).mean()))
        prev = g
    cap.release()
    return {"sharp": round(float(np.median(sharp)), 1) if sharp else 0.0,
            "mouth": round(float(np.median(mouth)), 2) if mouth else 0.0}


def candidates(od, i):
    """这一段的全部版本：当前版 + clips/takes/ 里的历史版。"""
    cur = os.path.join(od, "clips", "seg_%02d.mp4" % i)
    out = [("当前", cur)] if os.path.isfile(cur) else []
    takes = os.path.join(od, "clips", "takes")
    if os.path.isdir(takes):
        for name in sorted(os.listdir(takes)):
            if name.startswith("seg_%02d." % i) and name.lower().endswith(".mp4"):
                out.append((name, os.path.join(takes, name)))
    return out


def pick_best(od, ids, sharp_floor, dry_run):
    """在多个版本里挑最好的，必要时覆盖回 clips/seg_%02d.mp4。返回 [(段, 选了哪个, 原因)]。"""
    picked = []
    for i in ids:
        vers = candidates(od, i)
        if len(vers) < 2:
            continue
        scored = [(tag, p, score_clip(p)) for tag, p in vers]
        scored = [x for x in scored if x[2]]
        if not scored:
            continue
        # 先看清晰度是否过线，过线的里面挑嘴动量最大的；都不过线就挑最清晰的
        ok = [x for x in scored if x[2]["sharp"] >= sharp_floor] or scored
        best = max(ok, key=lambda x: (x[2]["mouth"], x[2]["sharp"]))
        if best[0] == "当前":
            picked.append((i, "当前", "当前版本已经是最好的"))
            continue
        detail = "清晰度 %.0f / 嘴动量 %.2f（共 %d 个版本）" % (best[2]["sharp"], best[2]["mouth"], len(scored))
        picked.append((i, best[0], detail))
        if dry_run:
            continue
        cur = os.path.join(od, "clips", "seg_%02d.mp4" % i)
        if os.path.isfile(cur):                       # 当前版先归档，覆盖掉也不会丢
            dst = os.path.join(od, "clips", "takes",
                               "seg_%02d.take99.%s.mp4" % (i, time.strftime("%Y%m%d-%H%M%S")))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(cur, dst)
        shutil.copy2(best[1], cur)
    return picked


def main():
    ap = argparse.ArgumentParser(description="数字人自动重拍闭环")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--rounds", type=int, default=MAX_RETAKES, help="最多几轮（默认 %d）" % MAX_RETAKES)
    ap.add_argument("--max-segments", type=int, default=3, help="一轮最多重拍几段（默认 3）")
    ap.add_argument("--pick-best", action="store_true", help="不重拍，只在已有版本里挑最好的拼回去")
    ap.add_argument("--yes", action="store_true", help="真跑（不加只预览）")
    a = ap.parse_args()

    od = os.path.join(a.workdir, "avatar")
    plan_p = os.path.join(od, "plan.json")
    if not os.path.isfile(plan_p):
        sys.exit("找不到 %s：先把数字人阶段跑到 build_prompts 那一步。" % plan_p)
    segs = load(plan_p)["segments"]
    total = len(segs)
    qa_p = os.path.join(od, "qa.json")
    final_p = os.path.join(od, "final_avatar.mp4")
    if not a.pick_best and not os.path.isfile(final_p):
        sys.exit("还没有 %s：先跑 finalize.py 拼出成片，再来做闭环。" % final_p)

    if a.pick_best:
        qa = load(qa_p, {})
        if not qa:
            sys.exit("没有 qa.json：先跑 qa_video.py。")
        ids = sorted({int(r["i"]) for r in qa.get("rows", [])
                      if r.get("verdict") == "retake"} | {int(i) for i in qa.get("retake", [])})
        got = pick_best(od, ids, qa.get("sharp_floor") or qa_video.SHARP_ABS_MIN, dry_run=not a.yes)
        print("\n=== 版本挑选 ===" if got else "\n没有多个版本可挑。")
        for i, tag, why in got:
            print("段%d：%s（%s）" % (i, tag, why))
        if got and a.yes:
            run([os.path.join(HERE, "finalize.py"), "--workdir", a.workdir])
            run([os.path.join(HERE, "qa_video.py"), "--workdir", a.workdir], quiet=True)
            print("已重新拼接并质检。")
        elif got:
            print("[预览] 没有改动。确认后加 --yes。")
        return

    stopped = None
    for rnd in range(1, a.rounds + 1):
        print("\n=== 第 %d 轮质检 ===" % rnd)
        run([os.path.join(HERE, "qa_video.py"), "--workdir", a.workdir], quiet=True)
        qa = load(qa_p, {})
        rows = {int(r["i"]): r for r in qa.get("rows", [])}
        if not rows:
            sys.exit(3)
        retakes = {int(k): int(v) for k, v in (load(os.path.join(od, "result.json"), {})
                                               .get("retakes") or {}).items()}
        want = [i for i in qa.get("retake", []) if retakes.get(i, 0) < MAX_RETAKES]
        exhausted = [i for i in qa.get("retake", []) if retakes.get(i, 0) >= MAX_RETAKES]
        for i in sorted(rows):
            r = rows[i]
            print("  段%-2d 清晰度 %-6s 嘴动量 %-6s %s%s" % (
                i,
                "-" if r.get("sharp") is None else "%.0f" % r["sharp"],
                "-" if r.get("mouth_ratio") is None else "%.2f" % r["mouth_ratio"],
                "重拍" if r["verdict"] == "retake" else "通过",
                ("（%s）" % "+".join(r.get("reasons") or [])) if r.get("reasons") else ""))
        if not qa.get("retake"):
            print("  全部通过，不用重拍。")
            stopped = "done"
            break
        if exhausted:
            print("  段%s 已经重拍到上限（%d 次），不再重拍——同样的输入再抽也是类似结果，"
                  "该改 prompt_src.json 里那几段的写法了。" % ("、".join(map(str, exhausted)), MAX_RETAKES))
        if not want:
            stopped = "exhausted"
            break
        if len(want) > a.max_segments or len(want) * 2 > total:
            stopped = "too-many"
            print("  这一轮要重拍 %d/%d 段，超过上限（单轮 %d 段、且不超过全片一半）。"
                  "\n  这么多段一起不达标，通常不是抽卡运气问题——先看看提示词和底图，"
                  "\n  或者把其中几处交给包装阶段遮盖。" % (len(want), total, a.max_segments))
            break
        print("  本轮重拍：段%s（约 %d 分钟）" % ("、".join(map(str, want)), len(want) * 5))
        if not a.yes:
            print("[预览] 还没有提交。确认后加 --yes。")
            return
        r = run([os.path.join(HERE, "run_avatar.py"), "--workdir", a.workdir,
                 "--only"] + [str(i) for i in want] + ["--seed", "random", "--yes"])
        if r.returncode == 2:
            print("  重拍没有完全成功，先看上面的失败原因；已成功的段不会白跑。")
            stopped = "error"
            break
        run([os.path.join(HERE, "finalize.py"), "--workdir", a.workdir])
    else:
        stopped = "rounds"

    if stopped == "rounds":
        print("\n已跑满 %d 轮。剩下的问题需要改提示词或交给包装遮盖。" % a.rounds)

    # ---- 收尾：在多版本里挑最好的
    qa = load(qa_p, {})
    touched = sorted({int(i) for i in (qa.get("retake") or [])} |
                     {int(k) for k, v in (load(os.path.join(od, "result.json"), {})
                                          .get("retakes") or {}).items() if int(v) > 0})
    if touched:
        print("\n=== 在多版本里挑最好的 ===")
        got = pick_best(od, touched, qa.get("sharp_floor") or qa_video.SHARP_ABS_MIN, dry_run=not a.yes)
        for i, tag, why in got:
            print("段%d：%s（%s）" % (i, tag, why))
        if got and a.yes:
            run([os.path.join(HERE, "finalize.py"), "--workdir", a.workdir])
            run([os.path.join(HERE, "qa_video.py"), "--workdir", a.workdir], quiet=True)
            print("已按最优版本重新拼接并质检。")

    qa = load(qa_p, {})
    left = qa.get("retake") or []
    print("\n=== 结果 ===")
    if left:
        print("还有 %d 段没过：段%s。这些请改提示词后重跑，或者交给包装阶段遮盖（anomalies.json 已经写好）。"
              % (len(left), "、".join(map(str, left))))
        sys.exit(2)
    print("全部通过。")


if __name__ == "__main__":
    main()
