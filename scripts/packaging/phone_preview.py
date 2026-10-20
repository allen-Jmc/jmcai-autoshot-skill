"""把成片的关键帧缩到手机宽度拼成一张图：判断字号在手机上到底读不读得清。

  python phone_preview.py --workdir <项目目录>
  python phone_preview.py --video 成片.mp4 --times 5,30,80 --out phone.jpg
  python phone_preview.py --workdir <项目目录> --width 390        # 默认就是 390（iPhone 逻辑宽度）

为什么要在手机宽度上看：成片是在 1664 或 1920 宽的画布上做的，字号也是按那个尺寸定的。
缩到手机上（宽约 390）之后，短边 1080 设计里的 22px 只剩 8px，就糊成一团了；
而 44px 的字还剩 16px，刚好能读。**这个差别在电脑上看原片是看不出来的**——
所以交付前必须按真实观看尺寸看一遍。

输出的图里每帧标了时间点，看到读不清的地方，就回去改 `src/parts/Subs.tsx`（或 `core/Subtitles.tsx`）
的字号，重渲后再看一遍。
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _common  # noqa: F401  统一 stdout 编码
from _common import imwrite, media_info
from contact_sheet import grab

PHONE_WIDTH = 390          # iPhone 12/13/14 的逻辑宽度，安卓主流也在这个量级
GAP = 10


def label(img, text):
    h = 26
    bar = np.full((h, img.shape[1], 3), 18, np.uint8)
    cv2.putText(bar, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    ap = argparse.ArgumentParser(description="手机尺寸预览")
    ap.add_argument("--workdir", help="项目目录（读 package/out/main.mp4）")
    ap.add_argument("--video")
    ap.add_argument("--times", help="逗号分隔的秒数；不给就在全片均匀取 6 帧")
    ap.add_argument("--width", type=int, default=PHONE_WIDTH)
    ap.add_argument("--out", help="输出图片路径")
    a = ap.parse_args()

    video = a.video
    out = a.out
    if not video:
        if not a.workdir:
            sys.exit("给 --workdir 或 --video")
        pkg = os.path.join(a.workdir, "package")
        for cand in ("out/main.mp4", "out/_raw.mp4"):
            p = os.path.join(pkg, cand)
            if os.path.isfile(p):
                video = p
                break
        if not video:
            finals = sorted(f for f in os.listdir(os.path.join(a.workdir, "final"))
                            if f.lower().endswith(".mp4")) if os.path.isdir(os.path.join(a.workdir, "final")) else []
            if finals:
                video = os.path.join(a.workdir, "final", finals[-1])
        if not video:
            sys.exit("找不到成片：先渲出 package/out/main.mp4，或用 --video 指定")
        out = out or os.path.join(pkg, "out", "phone_preview.jpg")
    out = out or os.path.splitext(video)[0] + "_phone.jpg"

    dur = media_info(video)["duration"]
    if not dur:
        sys.exit("读不到 %s 的时长" % video)
    times = ([float(x) for x in a.times.split(",")] if a.times
             else [round(dur * (i + 0.5) / 6, 2) for i in range(6)])

    cells = []
    for t in times:
        f = grab(video, t, 1920)          # 先按高分辨率取，再缩，避免直接从低分辨率放大
        if f is None:
            continue
        h = int(round(f.shape[0] * a.width / f.shape[1]))
        small = cv2.resize(f, (a.width, h), interpolation=cv2.INTER_AREA)
        cells.append(label(small, "%.1fs" % t))

    if not cells:
        sys.exit("一帧都没取到")
    w = max(c.shape[1] for c in cells)
    rows = []
    for i, c in enumerate(cells):
        rows.append(c)
        if i < len(cells) - 1:
            rows.append(np.full((GAP, w, 3), 12, np.uint8))
    sheet = np.vstack(rows)
    imwrite(out, sheet)
    print("手机预览（宽 %dpx，共 %d 帧）：%s" % (a.width, len(cells), out))
    print("  看三件事：字幕读不读得出、画面上的小字认不认得出、人物小窗会不会挡到字幕。")
    print("  读不清就回去改字号（design.md 要求字号不小于短边的 5%；汉字字形只有字号的约 0.78 倍，")
    print("  要手机 390px 宽下字形 ≥11px，字号得按短边的 6.5% 做），重渲后再看一遍。")


if __name__ == "__main__":
    main()
