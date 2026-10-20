"""成片自动质检：嘴部是否随说话动、两只手是否有一只僵着、画面糊不糊、各段色彩是否一致，并给出逐段「重拍 / 通过」判定。

  python qa_video.py --workdir <项目目录>                    # 默认检查 avatar/final_avatar.mp4
  python qa_video.py --workdir <项目目录> --video 其他.mp4

检查内容（都是帧间差分和图像方差得到的粗略指标，只用来**发现该看的地方**，不证明口型对不对）：
  1. 嘴部：用 OpenCV 自带的人脸检测定位脸，取下半脸；按 1 秒窗口，音频说话占比 ≥50% 但嘴部动作量
     低于全片说话时嘴部动作量中位数的 45%，连续 ≥3 个窗口 → 标出"说话时嘴几乎不动"（第一版第 4 段 40–49 秒就是这种）。
  2. 两只手：以脸为中心取画面左右两侧、胸前到画面底部的区域，统计每段里动作量 <0.8 的帧占比；
     任一只手 >60% 标出。
  3. 清晰度：每段抽 8 帧算拉普拉斯方差取中位数。低于**全片中位数的 80%** → 这一段糊了（实测：糊段 128.3，
     正常段 150–172，换种子重拍后升到 156.7）。整片中位数低于 110 时只提示"整片偏软"，不逐段判重拍——
     那是 --megapixels 或底图分辨率的问题。
  4. 色彩：取画面左右边缘的背景条带，按段算平均 Lab 色，与各段中位色的距离 >25 标出。
  5. 逐秒色彩：同一条带每秒的平均色与输入图片比，色相偏转 >20° 或彩度 <55% 的连续区间记为异常时间段。

**逐段判定**（写进 qa.json 的 `retake`，供 auto_retake.py 使用）：
  * `sharp` 不够 → 重拍（糊是遮盖盖不掉的）
  * 段内连续 ≥3 秒嘴不动 → 重拍（先改提示词再重拍，光换种子通常没用）
  * 手整段静止、段间色差大 → 只提示，**不重拍**：前者是提示词问题，后者交给包装遮盖更划算

输出 avatar/anomalies.json（嘴不动和逐秒色彩的异常时间段，供包装环节遮盖用）和 avatar/qa.json（逐段指标与判定）。
手和色彩的阈值按用户看片后的意见放宽过（2026-10-03）：单段手静止 56%、段间色差约 19 的成片用户认为可以接受；
整段明显变成另一种色调（实测色差 54–72）仍会被标出。嘴和清晰度的阈值没有放宽。阈值都是经验值，
只在这类构图（人物居中、双手在身体两侧）上验证过；换构图后先人工看一遍再信。
"""
import argparse
import json
import os
import sys
import wave

import cv2
import numpy as np

STATIC_THR = 0.8
MOUTH_RATIO = 0.45
HAND_STATIC_MAX = 0.60     # 一段里某只手近似静止的帧占比超过它才标出
COLOR_DE_MAX = 25.0        # 某段背景色与各段中位色的 Lab 距离超过它才标出
HUE_SHIFT_MAX = 20.0       # 逐秒：背景色相比图片偏转超过多少度算变色（实测正常 ≤12°，变青/变黄绿 ≥26°）
CHROMA_MIN = 0.55          # 逐秒：背景彩度降到图片的多少倍以下算褪色（实测正常 0.9–1.7，褪成灰白 ≤0.4）
# ---- 逐段判定：决定要不要重拍
SHARP_REL_MIN = 0.80       # 段清晰度低于全片各段中位数的这个比例 → 这一段糊了，重拍
SHARP_ABS_MIN = 110.0      # 全片中位数低于它 → 整片偏软，提示查 --megapixels / 底图（不逐段重拍）
MOUTH_SEG_RATIO = 0.45     # 段内说话时嘴动量低于全片中位数的这个比例算"嘴不动"
MOUTH_MIN_SECONDS = 3      # 段内连续这么多秒嘴不动才判重拍（偶尔一句不动是正常的）
SHARP_SAMPLES = 8          # 每段抽几帧算清晰度
FACE_CX_MAX = 0.05         # 各段脸的横向位置与中位数相差超过画宽的 5% → 人物构图不一致
FACE_SIZE_MAX = 0.15       # 脸的宽度相差超过 15% → 人物大小不一致（重拍换了构图／megapixels 不同）



def _imwrite(path, img):
    """cv2.imwrite 在带中文的路径上会悄悄失败（不报错、也不写文件），改成先编码再用 numpy 写出"""
    ok, buf = cv2.imencode(os.path.splitext(str(path))[1] or ".png", img)
    if not ok:
        raise RuntimeError("图片编码失败：%s" % path)
    buf.tofile(str(path))
    return True

def mouth_low_windows(mouth_motion, speak, fps, med_mouth):
    """找出"这一秒在说话、但嘴几乎不动"的秒号。

    判据：这一秒里音频说话占比 ≥50%，且嘴部动作量低于全片中位数的 MOUTH_RATIO 倍。
    med_mouth 为 0 或 nan（全片没测到说话）时返回空——没有基准就无从比较，
    此时"嘴不动"这件事根本判不出来，交给 mouth_strip.png 让人看。
    """
    if not (med_mouth > 0) or not np.isfinite(med_mouth):
        return []
    out = []
    for sec in range(int(len(mouth_motion) / fps)):
        sl = slice(int(sec * fps), int((sec + 1) * fps))
        if speak[sl].mean() >= 0.5 and mouth_motion[sl].mean() < MOUTH_RATIO * med_mouth:
            out.append(sec)
    return out


def merge_consecutive(seconds, min_len=MOUTH_MIN_SECONDS):
    """把连续的秒号合并成 (起, 止) 区间，只保留长度 ≥ min_len 的。"""
    runs, cur = [], []
    for s in seconds:
        if cur and s != cur[-1] + 1:
            runs.append(cur)
            cur = []
        cur.append(s)
    if cur:
        runs.append(cur)
    return [(r[0], r[-1] + 1) for r in runs if len(r) >= min_len]


def judge_segments(rows, segs, sharp, med_mouth, mouth_runs, region_source, fps, n, mouth_motion=None, speak=None):
    """逐段判定要不要重拍：原地写入 rows 的 sharp / mouth_ratio / mouth_low_seconds / verdict / reasons。

    两条会触发重拍，其余只提示：
      * sharp 低于全片各段中位数的 SHARP_REL_MIN 倍 → 糊，遮盖盖不掉
      * 段内和"嘴不动"区间重叠 ≥ MOUTH_MIN_SECONDS 秒 → 先改提示词再重拍
        但嘴部区域是**估算**出来的就不据此判重拍：区域没框到嘴时动作量天然接近 0。

    返回 (该重拍的段号列表, 清晰度门槛, 全片清晰度中位数)。重拍段号供 auto_retake.py 用。
    这是数字人闭环的分诊台，判错一段的代价是白等 4–5 分钟，所以单独抽出来便于离线回归。
    """
    sharp_vals = [v for v in sharp.values() if v is not None]
    sharp_med = float(np.median(sharp_vals)) if sharp_vals else 0.0
    # 只用"相对全片"判定：整片都偏软时逐段标出来没有意义（那要统一调 --megapixels）
    sharp_floor = SHARP_REL_MIN * sharp_med
    for r in rows:
        i = r["i"]
        seg = next(s for s in segs if s["i"] == i)
        sl = slice(int(seg["start"] * fps), min(n, int(seg["end"] * fps)))
        r["sharp"] = sharp.get(i)
        # 段内说话时嘴动量的中位数，相对全片中位数——低于 1 说明这段嘴比别处更僵
        if mouth_motion is not None and speak is not None and speak[sl].any() and med_mouth > 0:
            r["mouth_ratio"] = round(float(np.median(mouth_motion[sl][speak[sl]])) / med_mouth, 2)
        else:
            r["mouth_ratio"] = None
        # 这一段和"嘴不动"区间重叠了多少秒
        r["mouth_low_seconds"] = round(sum(max(0.0, min(b1, seg["end"]) - max(a1, seg["start"]))
                                          for a1, b1 in mouth_runs), 1)
        reasons = []
        if r["sharp"] is not None and r["sharp"] < sharp_floor:
            reasons.append("sharp")
        if r["mouth_low_seconds"] >= MOUTH_MIN_SECONDS and region_source != "estimated":
            reasons.append("mouth")
        r["verdict"] = "retake" if reasons else "ok"
        r["reasons"] = reasons
    return [r["i"] for r in rows if r["verdict"] == "retake"], sharp_floor, sharp_med


def face_consistency(track, rows):
    """各段脸的横向位置和大小对不对得上。返回提示文案列表，并原地写 rows 的 face_offset。

    换种子重拍最容易在这里出问题：脸偏了、大小变了，接缝处一眼就看出来。
    少于 3 段有检测结果时不下结论（样本太少，中位数没有意义）。
    """
    flags = []
    if len(track) < 3:
        for r in rows:
            r["face_offset"] = None
        return flags
    cxs = np.array([v[0] for v in track.values()])
    sizes = np.array([v[2] for v in track.values()])
    mcx, msz = float(np.median(cxs)), float(np.median(sizes))
    for r in rows:
        v = track.get(r["i"])
        if not v:
            r["face_offset"] = None
            continue
        r["face_offset"] = round(float(v[0] - mcx), 4)
        if abs(v[0] - mcx) > FACE_CX_MAX or abs(v[2] - msz) / max(msz, 1e-6) > FACE_SIZE_MAX:
            flags.append("人物一致性：段%d 的脸比其余段偏了 %.1f%% 画宽、大小差 %.0f%%"
                         "——多半是重拍时换了构图，或各段 --megapixels 不一致"
                         % (r["i"], (v[0] - mcx) * 100, (v[2] / msz - 1) * 100))
    return flags


def color_anomalies(colors, ref_ab, fps):
    """逐秒比较背景色与输入图片：色相偏转或褪色的连续区间，返回 [(起, 止, 说明)]。
    只看色相和彩度，不看亮度和"更鲜艳"——同一色调下变亮、变浓是用户认可的正常波动。"""
    r = ref_ab - 128.0
    rn = float(np.linalg.norm(r))
    bad = []
    for sec in range(int(len(colors) / fps)):
        v = colors[int(sec * fps):int((sec + 1) * fps), 1:].mean(0) - 128.0
        vn = float(np.linalg.norm(v))
        if rn < 6:                                      # 图片背景接近无彩色：没有色相可比，看有没有染上颜色
            if float(np.linalg.norm(v - r)) > 12:
                bad.append((sec, "背景染上了图片里没有的颜色"))
            continue
        hue = float(np.degrees(np.arccos(np.clip(np.dot(v, r) / (vn * rn + 1e-6), -1, 1))))
        if vn / rn < CHROMA_MIN:
            bad.append((sec, "背景褪色（彩度只有图片的 %.0f%%）" % (vn / rn * 100)))
        elif hue > HUE_SHIFT_MAX:
            bad.append((sec, "背景变成另一种色调（色相偏转 %.0f°）" % hue))
    runs = []
    for sec, why in bad:
        if runs and sec <= runs[-1][1] + 1:             # 相邻或只隔 1 秒的并成一个区间
            runs[-1][1] = sec + 1
        else:
            runs.append([sec, sec + 1, why])
    return [(a, b, why) for a, b, why in runs]


def cascade_path():
    """找 haarcascade_frontalface_default.xml。

    优先用 OpenCV 自带的（opencv-python 的 wheel 一般带），其次用**技能里随包的那一份**
    （assets/opencv/）。以前这里写死过一个 ComfyUI 自定义节点的绝对路径，换机器就失效，
    失效后嘴部区域会静默降级成"居中估算"——所以现在随技能带一份，不依赖外部安装。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cands = []
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        cands.append(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    cands.append(os.path.join(here, "..", "..", "assets", "opencv", "haarcascade_frontalface_default.xml"))
    for c in cands:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def face_box(video):
    if not hasattr(cv2, "CascadeClassifier"):
        return None            # 本机的 OpenCV 是精简构建（没有 objdetect），只能退回居中估算
    xml = cascade_path()
    if not xml:
        return None
    det = cv2.CascadeClassifier(xml)
    if det.empty():
        return None
    cap = cv2.VideoCapture(video)
    boxes = []
    for t in (0.5, 2.0, 4.0, 6.0):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        # minSize 从大往小试：大脸先命中，找不到再放宽，避免人物占画面比例不同时直接漏检
        for frac in (10, 16, 24):
            try:
                r = det.detectMultiScale(g, 1.1, 5, minSize=(max(24, f.shape[1] // frac),) * 2)
            except Exception:
                r = []
            if len(r):
                boxes.append(max(r, key=lambda b: b[2] * b[3]))
                break
    cap.release()
    if not boxes:
        return None
    return tuple(int(np.median([b[i] for b in boxes])) for i in range(4))


def mouth_box_from_face(face, W=None, H=None):
    """从人脸框推嘴部框：(fx,fy,fw,fh) → (x0, x1, y0, y1)。

    横向取脸的 25%–75%（避开脸颊两侧），纵向取 68%–98%（下半脸到下巴）。
    抽成纯函数是为了能离线测——这个比例一旦写错，嘴部框就落到鼻子或脖子上，
    "嘴动量"整个指标都会失真，而且在成片里看不出来（数字照样输出）。
    """
    fx, fy, fw, fh = face
    return (fx + int(0.25 * fw), fx + int(0.75 * fw),
            fy + int(0.68 * fh), fy + int(0.98 * fh))


def pick_segment_box(seg_boxes, bounds, t):
    """按当前时刻取所在段的嘴框；没有分段结果或该段没有框时返回 None。

    bounds 是 [(起, 止, 段号)]。用于逐帧循环——人物在各段位置不同，必须逐段换框。
    """
    if not seg_boxes:
        return None
    for a, b, i in bounds:
        if a <= t < b:
            return seg_boxes.get(i)
    return None


def mouth_boxes_per_segment(video, segs, fps, base_box, det=None):
    """为每一段推算它自己的嘴部框（像素），返回 ({段号: (x0, x1, y0, y1)}, {段号: 来源})。

    为什么不能全片用一个固定框：数字人各段的人物位置会漂移。`image_info` 里的 mouth_box 是按
    底图量的，只对"和底图姿态最接近"的那一段成立。实跑踩过——三段里段3 的脸右移了 148px，
    固定框落在脸颊左侧、碰不到嘴，嘴部动作量从 11.6 掉到 3.1，看着像"说话时嘴不动"。
    如果据此判重拍，就是白等 4–5 分钟重跑一段根本没问题的片子（正好是这条片子文案里吐槽的事）。

    做法：每段在**回到原姿态的开头**附近检出脸，按脸推嘴框（横向 25%–75%、纵向 68%–98%）；
    检不到就退回 base_box。每段实际用的框和来源都记在 qa.json 里，便于核对。
    """
    H = W = None
    cap = cv2.VideoCapture(video)
    W, H = int(cap.get(3)), int(cap.get(4))
    out, src = {}, {}
    for s in segs:
        i = s["i"]
        box, why = None, "image_info"
        if det is not None:
            # 段首附近取几帧试：段首是回到原姿态的地方，比段中更稳
            a, b = float(s["start"]), float(s["end"])
            for k in range(4):
                t = a + (b - a) * (0.06 + 0.06 * k)
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t * fps))))
                ok, f = cap.read()
                if not ok:
                    continue
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                found = None
                for frac in (8, 10, 12, 16, 24):
                    try:
                        r = det.detectMultiScale(g, 1.08, 5, minSize=(max(40, W // frac),) * 2)
                    except Exception:
                        r = []
                    if len(r):
                        found = max(r, key=lambda q: q[2] * q[3])
                        break
                if found is not None:
                    fx, fy, fw, fh = found
                    box = mouth_box_from_face(found)
                    why = "face_detect"
                    break
        if box is None and base_box:
            mx0, my0, mx1, my1 = base_box
            box = (int(mx0 * W), int(mx1 * W), int(my0 * H), int(my1 * H))
            why = "image_info"
        out[i], src[i] = box, why
    cap.release()
    return out, src


def sharpness(video, segs, fps):
    """每段抽 SHARP_SAMPLES 帧算拉普拉斯方差取中位数，返回 {段号: 值}。

    这是"画面糊不糊"唯一的量化依据：同一批段里糊的那一段方差会明显低于其余段。
    实跑标定：糊段 128.3，正常段 150–172，换种子重拍后升到 156.7。
    """
    cap = cv2.VideoCapture(video)
    out = {}
    for s in segs:
        a, b = float(s["start"]), float(s["end"])
        vals = []
        for k in range(SHARP_SAMPLES):
            t = a + (b - a) * (k + 0.5) / SHARP_SAMPLES
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t * fps))))
            ok, f = cap.read()
            if not ok:
                continue
            vals.append(float(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()))
        if vals:
            out[s["i"]] = round(float(np.median(vals)), 1)
    cap.release()
    return out


def face_track(video, segs, fps):
    """每段取中点检测一次脸，返回 {段号: (cx, cy, size)}（都按画面宽高归一化）。

    用来查"人物一致性"：换种子重拍后，脸在画面里的位置和大小可能和其余段对不上。
    这不是审美问题——位置/大小跳变在接缝处会直接看出来。检测不到脸就跳过这一段。
    """
    if not hasattr(cv2, "CascadeClassifier"):
        return {}
    xml = cascade_path()
    if not xml:
        return {}
    det = cv2.CascadeClassifier(xml)
    if det.empty():
        return {}
    cap = cv2.VideoCapture(video)
    out = {}
    for s in segs:
        t = (float(s["start"]) + float(s["end"])) / 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t * fps))))
        ok, f = cap.read()
        if not ok:
            continue
        H, W = f.shape[:2]
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        found = None
        for frac in (10, 16, 24):
            try:
                r = det.detectMultiScale(g, 1.1, 5, minSize=(max(24, W // frac),) * 2)
            except Exception:
                r = []
            if len(r):
                found = max(r, key=lambda b: b[2] * b[3])
                break
        if found is not None:
            x, y, w, h = found
            out[s["i"]] = ((x + w / 2) / W, (y + h / 2) / H, w / W)
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--video")
    a = ap.parse_args()
    od = os.path.join(a.workdir, "avatar")
    video = a.video or os.path.join(od, "final_avatar.mp4")
    plan = json.load(open(os.path.join(od, "plan.json"), encoding="utf-8"))
    segs = plan["segments"]
    cap = cv2.VideoCapture(video)
    fps, W, H = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
    info_p = os.path.join(od, "image_info.json")
    info = json.load(open(info_p, encoding="utf-8")) if os.path.isfile(info_p) else {}
    mb = info.get("mouth_box")          # 由看图的人写入：嘴部在画面中的位置（占画面宽高的比例）[x0,y0,x1,y1]
    fb = face_box(video)
    # 每段自己的嘴部框：人物在各段的位置会漂移，全片用一个固定框会脱靶（见 mouth_boxes_per_segment）
    det = None
    if hasattr(cv2, "CascadeClassifier") and cascade_path():
        _d = cv2.CascadeClassifier(cascade_path())
        det = None if _d.empty() else _d
    seg_boxes, seg_src = ({}, {})
    if det is not None:
        seg_boxes, seg_src = mouth_boxes_per_segment(video, segs, fps, mb, det)
    if mb:
        mx0, my0, mx1, my1 = int(mb[0] * W), int(mb[1] * H), int(mb[2] * W), int(mb[3] * H)
        cx = (mx0 + mx1) // 2
        fy, fh = int(my0 - 0.25 * H), int(0.3 * H)
        region_source = "image_info"
        print("嘴部区域：基准位置取自图片信息 %s" % str(mb))
        if seg_boxes:
            drift = []
            for i, bx in sorted(seg_boxes.items()):
                if bx is None:
                    continue
                dx = bx[0] - mx0
                if abs(dx) > 0.5 * max(1, mx1 - mx0):
                    drift.append("段%d 偏移 %+dpx" % (i, dx))
            real = sum(1 for v in seg_src.values() if v == "face_detect")
            print("  按段校正：%d/%d 段用人脸检测重新定位%s" % (
                real, len(segs), ("；" + "、".join(drift) + "（固定框会脱靶，已按各段实际位置测量）") if drift else ""))
    elif fb is not None:
        fx, fy, fw, fh = fb
        cx = fx + fw // 2
        mx0, mx1, my0, my1 = fx + int(0.25 * fw), fx + int(0.75 * fw), fy + int(0.68 * fh), fy + int(0.98 * fh)
        region_source = "face_detect"
        print("嘴部区域：用人脸检测定位")
    else:
        mx0, mx1, my0, my1 = int(0.42 * W), int(0.58 * W), int(0.46 * H), int(0.58 * H)
        cx, fy, fh = W // 2, int(0.12 * H), int(0.3 * H)
        region_source = "estimated"
        print("提示：既没有嘴部位置也没有人脸检测，嘴部区域按人物居中估算，嘴部结果可能不准。"
              "让看图的人用 prepare_image.py --mouth-box 写入嘴的位置。"
              "（估算出来的区域**不会**被判成重拍，只提示。）")
        if not hasattr(cv2, "CascadeClassifier"):
            print("  原因：这个 OpenCV 构建没有 objdetect 模块（精简版）。装 opencv-python-headless 完整版即可。")
        elif not cascade_path():
            print("  原因：找不到 haarcascade_frontalface_default.xml（技能 assets/opencv/ 下应该随包带一份）。")
        else:
            print("  原因：人脸检测在这几帧没找到脸（人物太小、太偏或侧脸）。用 --mouth-box 直接写位置最稳。")
    reg = {"嘴": (mx0, mx1, my0, my1),
           "左手": (max(0, cx - int(0.42 * W)), max(0, cx - int(0.20 * W)), min(H - 2, fy + int(1.2 * fh)), H),
           "右手": (min(W - 2, cx + int(0.26 * W)), min(W, cx + int(0.45 * W)), min(H - 2, fy + int(1.2 * fh)), H)}
    # 逐帧要用的嘴框：有分段结果时按当前时间所在的段取，否则用全片一个
    bounds = [(float(s["start"]), float(s["end"]), s["i"]) for s in segs]
    mot = {k: [] for k in reg}
    edge_cols = list(range(0, int(W * 0.06))) + list(range(int(W * 0.94), W))
    colors = []
    prev = None
    fi = -1
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fi += 1
        t_now = fi / fps
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            cur_reg = reg
            if seg_boxes:
                bx = pick_segment_box(seg_boxes, bounds, t_now)
                if bx:
                    cur_reg = dict(reg)
                    cur_reg["嘴"] = tuple(bx)
            for k, (x0, x1, y0, y1) in cur_reg.items():
                y0 = max(0, min(H - 2, int(y0)))
                y1 = max(y0 + 1, min(H, int(y1)))
                x0 = max(0, min(W - 2, int(x0)))
                x1 = max(x0 + 1, min(W, int(x1)))
                mot[k].append(float(np.abs(g[y0:y1, x0:x1] - prev[y0:y1, x0:x1]).mean()))
        prev = g
        lab = cv2.cvtColor(f[: int(H * 0.5)][:, edge_cols], cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0)
        colors.append(lab)
    cap.release()
    mot = {k: np.array(v) for k, v in mot.items()}
    colors = np.array(colors)
    n = len(mot["嘴"])
    w = wave.open(os.path.join(a.workdir, "out", "merged.wav"))
    sr = w.getframerate()
    audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    w.close()
    act = np.array([np.sqrt(np.mean(audio[int(i / fps * sr):int((i + 1) / fps * sr)] ** 2)) if int((i + 1) / fps * sr) > int(i / fps * sr) else 0 for i in range(n)])
    speak = act > 0.05 * np.percentile(act, 95)
    med_mouth = float(np.median(mot["嘴"][speak]))
    flags = []
    # ---- 嘴
    low_windows = mouth_low_windows(mot["嘴"], speak, fps, med_mouth)
    mouth_runs = merge_consecutive(low_windows)
    for a0, a1 in mouth_runs:
        segno = [s["i"] for s in segs if s["start"] <= (a0 + a1) / 2 < s["end"]]
        flags.append("嘴：%d–%d 秒（段%s）说话时嘴部几乎不动（动作量低于全片中位数 %.1f 的 %d%%）" % (a0, a1, segno[0] if segno else "?", med_mouth, MOUTH_RATIO * 100))
    # ---- 手 & 色彩 按段
    base = np.median(np.array([colors[slice(int(s["start"] * fps), min(n, int(s["end"] * fps)))].mean(0) for s in segs]), axis=0)
    rows = []
    for s in segs:
        sl = slice(int(s["start"] * fps), min(n, int(s["end"] * fps)))
        row = {"i": s["i"], "mouth": float(mot["嘴"][sl][speak[sl]].mean()) if speak[sl].any() else None,
               "left_static": float((mot["左手"][sl] < STATIC_THR).mean()), "right_static": float((mot["右手"][sl] < STATIC_THR).mean())}
        c = colors[sl].mean(0)
        row["color_dE"] = float(np.linalg.norm(c - base))
        rows.append(row)
        for side, key in (("左手", "left_static"), ("右手", "right_static")):
            if row[key] > HAND_STATIC_MAX:
                flags.append("手：段%d 的%s有 %.0f%% 的帧近似静止（>%d%%）" % (s["i"], side + "（画面左侧）" if side == "左手" else side + "（画面右侧）", row[key] * 100, HAND_STATIC_MAX * 100))
        if row["color_dE"] > COLOR_DE_MAX:
            flags.append("色彩：段%d 的背景色与各段中位色相差 %.1f（>%d），整段色调明显不同" % (s["i"], row["color_dE"], COLOR_DE_MAX))
    # ---- 清晰度 + 逐段判定：决定哪几段该重拍
    sharp = sharpness(video, segs, fps)
    retake, sharp_floor, sharp_med = judge_segments(
        rows, segs, sharp, med_mouth, mouth_runs, region_source, fps, n,
        mouth_motion=mot["嘴"], speak=speak)

    # ---- 人物一致性：各段脸的横向位置和大小对不对得上（换种子重拍最容易在这里出问题）
    flags.extend(face_consistency(face_track(video, segs, fps), rows))
    # ---- 逐秒色彩：与输入图片比（按段平均会把段内几秒的变色平均掉）
    anomalies = [{"start": float(a0), "end": float(a1), "type": "mouth", "cover": "content_full",
                  "detail": "说话时嘴部几乎不动"} for a0, a1 in mouth_runs]
    img_p = os.path.join(od, "image.png")
    ref_img = cv2.imdecode(np.fromfile(img_p, dtype=np.uint8), cv2.IMREAD_COLOR) if os.path.isfile(img_p) else None
    if ref_img is not None:
        ref_img = cv2.resize(ref_img, (W, H), interpolation=cv2.INTER_AREA)
        ref_ab = cv2.cvtColor(ref_img[: int(H * 0.5)][:, edge_cols], cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0)[1:]
        for a0, a1, why in color_anomalies(colors, ref_ab, fps):
            anomalies.append({"start": float(a0), "end": float(a1), "type": "color", "cover": "content_pip", "detail": why})
            flags.append("色彩：%d–%d 秒 %s" % (a0, a1, why))
    anomalies.sort(key=lambda x: x["start"])
    json.dump({"video": video, "duration": round(n / fps, 2), "anomalies": anomalies},
              open(os.path.join(od, "anomalies.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("全片说话时嘴部动作量中位数 %.2f；清晰度门槛 %.1f（全片中位数 %.1f）" % (med_mouth, sharp_floor, sharp_med))
    if sharp_med and sharp_med < SHARP_ABS_MIN:
        print("提示：整片清晰度偏低（各段中位数 %.0f，参考线 %.0f）。这不是某一段的问题——"
              "先查底图分辨率和 --megapixels（1.0 快但软、2.0 高清），别逐段重拍。"
              % (sharp_med, SHARP_ABS_MIN))
    print("段   嘴动量比   嘴不动秒   清晰度   左手静止   右手静止   色差ΔE   判定")
    for r in rows:
        print("%-3d  %-9s  %-9s  %-7s  %-9s  %-9s  %-7.1f  %s" % (
            r["i"],
            "-" if r["mouth_ratio"] is None else "%.2f" % r["mouth_ratio"],
            "%.1f" % r["mouth_low_seconds"],
            "-" if r["sharp"] is None else "%.0f" % r["sharp"],
            "%.0f%%" % (r["left_static"] * 100), "%.0f%%" % (r["right_static"] * 100),
            r["color_dE"],
            "重拍" if r["verdict"] == "retake" else "通过"))
    if retake:
        print()
        print("建议重拍：段 %s（%s）" % ("、".join(map(str, retake)),
              "，".join("段%d %s" % (r["i"], "+".join(r["reasons"])) for r in rows if r["verdict"] == "retake")))
        print("  清晰度不够 → 换种子重拍；嘴不动 → 先改提示词再重拍。自动重拍：python scripts/avatar/auto_retake.py --workdir <项目目录>")
    tot_l, tot_r = float((mot["左手"] < STATIC_THR).mean()), float((mot["右手"] < STATIC_THR).mean())
    print("全片：左手静止帧 %.0f%%，右手静止帧 %.0f%%" % (tot_l * 100, tot_r * 100))
    print()
    print("需要看一下的地方：" if flags else "没有触发任何阈值（仍建议抽帧看一遍）。")
    for f in flags:
        print(" -", f)
    if anomalies:
        print("异常时间段已写入 %s（%d 处）。不想重跑时可以交给包装环节遮盖：人物缩到小窗或暂时隐藏，主画面放图形。"
              % (os.path.join(od, "anomalies.json"), len(anomalies)))
    # ---- 嘴部逐秒图：每秒一格，标出这一秒音频是否在说话（红 S = 在说话）。帧间差分会被头部晃动干扰，
    #      "说话时嘴是否闭着"最终要看这张图（用 Read 工具看）。
    cap = cv2.VideoCapture(video)
    tiles = []
    mx0_, mx1_, my0_, my1_ = reg["嘴"]
    padx, pady = int((mx1_ - mx0_) * 0.35), int((my1_ - my0_) * 0.6)
    X0, X1, Y0, Y1 = max(0, mx0_ - padx), min(W, mx1_ + padx), max(0, my0_ - pady), min(H, my1_ + pady)
    for sec in range(int(n / fps)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((sec + 0.5) * fps))
        ok, f = cap.read()
        if not ok:
            break
        t = cv2.resize(f[Y0:Y1, X0:X1], (220, int(220 * (Y1 - Y0) / max(1, X1 - X0))))
        sp = speak[int(sec * fps):int((sec + 1) * fps)].mean() >= 0.5
        cv2.putText(t, "%ds" % sec, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if sp:
            cv2.putText(t, "S", (t.shape[1] - 26, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        tiles.append(t)
    cap.release()
    if tiles:
        cols = 8
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        _imwrite(os.path.join(od, "mouth_strip.png"), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))
        print("嘴部逐秒图（红 S = 这一秒在说话；说话的秒里嘴一直闭着就是问题）：", os.path.join(od, "mouth_strip.png"))
    print("说明：这些是帧间差分的粗略指标，用来发现该看的地方；口型是否对准、手势是否自然仍要人看。")
    json.dump({"rows": rows, "flags": flags, "mouth_region": list(reg["嘴"]), "mouth_median": med_mouth,
               "hand_static_total": [tot_l, tot_r], "mouth_low_ranges": mouth_runs,
               "sharp_floor": round(sharp_floor, 1), "sharp_median": round(sharp_med, 1),
               "mouth_region_source": region_source,
               # 每段实际用的嘴框（按该段人物位置校正过）。核对"嘴动量低"时先看这里：
               # 框脱靶也会让嘴动量变低，那是测量问题，不是表演问题。
               "mouth_region_per_segment": {str(i): {"box": [int(v) for v in b] if b else None, "source": seg_src.get(i)}
                                            for i, b in sorted(seg_boxes.items())} if seg_boxes else None,
               "retake": retake,
               "thresholds": {"sharp_abs_min": SHARP_ABS_MIN, "sharp_rel_min": SHARP_REL_MIN,
                              "mouth_seg_ratio": MOUTH_SEG_RATIO, "mouth_min_seconds": MOUTH_MIN_SECONDS,
                              "hand_static_max": HAND_STATIC_MAX, "color_de_max": COLOR_DE_MAX}},
              open(os.path.join(od, "qa.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
