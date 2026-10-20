# -*- coding: utf-8 -*-
"""qa_video.py 判定逻辑的离线回归测试。

不读视频、不需要桌面端、不需要 OpenCV 的人脸模型——只喂数组，验证"该不该重拍"这个分诊台。

为什么值得单独测：`auto_retake.py` 完全依赖 `qa.json` 里的 `retake` 列表。
判错一段的代价是白等 4–5 分钟（一段人物视频的生成时间），
而这段逻辑以前全塞在 `main()` 里，既测不了、也没人测过。

  python tests/avatar/test_qa_verdict.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts", "avatar"))

import qa_video as QA  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("✓" if ok else "✗"), name, detail if not ok else "")


# ---------------------------------------------------------------- 嘴不动窗口
def test_low_windows():
    fps = 30
    # 10 秒：每秒 30 帧。嘴动量固定 1.0，说话标记单独给
    motion = np.full(300, 1.0)
    speak = np.zeros(300, dtype=bool)
    speak[:] = True
    # 第 4、5、6 秒嘴动量掉到 0.1（低于 0.45 × 1.0）
    motion[4 * fps:7 * fps] = 0.1
    w = QA.mouth_low_windows(motion, speak, fps, med_mouth=1.0)
    check("嘴不动窗口：连续 3 秒低动作 → 全部标出", w == [4, 5, 6], str(w))

    # 不说话的那几秒即使嘴不动也不算
    speak2 = np.zeros(300, dtype=bool)
    speak2[: 2 * fps] = True          # 只有前 2 秒在说话
    w2 = QA.mouth_low_windows(motion, speak2, fps, med_mouth=1.0)
    check("嘴不动窗口：不说话时不动不算（只在前 2 秒说话的片子里）", w2 == [], str(w2))

    # 基准为 0 / nan → 没有比较基准，返回空而不是乱标
    check("嘴不动窗口：med_mouth=0 → 返回空（没有基准）",
          QA.mouth_low_windows(motion, speak, fps, med_mouth=0.0) == [])
    check("嘴不动窗口：med_mouth=nan → 返回空",
          QA.mouth_low_windows(motion, speak, fps, med_mouth=float("nan")) == [])


def test_merge():
    check("合并：连续 3 秒 → 一条 [1,4)", QA.merge_consecutive([1, 2, 3]) == [(1, 4)])
    check("合并：只 2 秒 → 太短，丢掉", QA.merge_consecutive([1, 2]) == [])
    check("合并：不连续的两段各算各的",
          QA.merge_consecutive([0, 1, 2, 7, 8, 9]) == [(0, 3), (7, 10)])
    check("合并：空输入 → 空输出", QA.merge_consecutive([]) == [])
    check("合并：自定义门槛 min_len=2 时 2 秒也保留", QA.merge_consecutive([1, 2], min_len=2) == [(1, 3)])


# ---------------------------------------------------------------- 逐段判定
SEGS = [{"i": 1, "start": 0.0, "end": 10.0},
        {"i": 2, "start": 10.0, "end": 20.0},
        {"i": 3, "start": 20.0, "end": 30.0}]


def rows_for(n=3):
    return [{"i": i, "mouth": 1.0, "left_static": 0.1, "right_static": 0.1, "color_dE": 1.0}
            for i in range(1, n + 1)]


def test_judge_sharp():
    # 全片 200 / 200 / 100：中位数 200，门槛 0.8×200 = 160 → 第 3 段（100）糊
    sharp = {1: 200.0, 2: 200.0, 3: 100.0}
    rows = rows_for()
    retake, floor, med = QA.judge_segments(rows, SEGS, sharp, 1.0, [], "image_info", 30, 900)
    check("判定：清晰度低于全片中位数 80% → 重拍",
          retake == [3] and rows[2]["reasons"] == ["sharp"], str((retake, rows[2]["reasons"])))
    check("判定：门槛算对（中位数 200 × 0.8 = 160）",
          abs(floor - 160.0) < 1e-6 and abs(med - 200.0) < 1e-6, "%s / %s" % (floor, med))

    # 全部一样清晰 → 谁也不该被判糊（相对判定的必然结果）
    rows = rows_for()
    retake, _, _ = QA.judge_segments(rows, SEGS, {1: 150.0, 2: 150.0, 3: 150.0}, 1.0, [], "image_info", 30, 900)
    check("判定：各段一样清晰时不冤枉任何一段", retake == [], str(retake))

    # 整片偏软（全低于 SHARP_ABS_MIN）也不逐段重拍——那是 --megapixels 的问题
    rows = rows_for()
    retake, floor, med = QA.judge_segments(rows, SEGS, {1: 90.0, 2: 90.0, 3: 90.0}, 1.0, [], "image_info", 30, 900)
    check("判定：整片偏软（90）不逐段重拍，只靠主流程提示", retake == [] and med < QA.SHARP_ABS_MIN,
          "%s / med=%s" % (retake, med))

    # 清晰度缺失（None）不参与判定，也不能拉低中位数
    rows = rows_for()
    retake, floor, med = QA.judge_segments(rows, SEGS, {1: 200.0, 2: None, 3: 200.0}, 1.0, [], "image_info", 30, 900)
    check("判定：某段清晰度缺失 → 跳过它，中位数不被 None 污染",
          retake == [] and abs(med - 200.0) < 1e-6, "%s med=%s" % (retake, med))

    rows = rows_for()
    retake, floor, med = QA.judge_segments(rows, SEGS, {}, 1.0, [], "image_info", 30, 900)
    check("判定：完全没有清晰度数据 → 不重拍、门槛为 0、不崩", retake == [] and floor == 0.0, str(floor))


def test_judge_mouth():
    # 第 2 段（10–20 秒）里 12–16 秒嘴不动 = 4 秒 ≥ 3 → 重拍
    runs = [(12, 16)]
    rows = rows_for()
    retake, _, _ = QA.judge_segments(rows, SEGS, {1: 200.0, 2: 200.0, 3: 200.0}, 1.0, runs, "image_info", 30, 900)
    check("判定：段内嘴不动 4 秒 → 只重拍这一段",
          retake == [2] and rows[1]["reasons"] == ["mouth"], str((retake, rows[1]["reasons"])))
    check("判定：重叠秒数算对（12–16 落在 10–20 段内 = 4.0 秒）",
          abs(rows[1]["mouth_low_seconds"] - 4.0) < 1e-6, str(rows[1]["mouth_low_seconds"]))

    # 嘴部区域是估算的 → 不据此判重拍（没框到嘴时动作量天然接近 0）
    rows = rows_for()
    retake, _, _ = QA.judge_segments(rows, SEGS, {1: 200.0, 2: 200.0, 3: 200.0}, 1.0, runs, "estimated", 30, 900)
    check("判定：嘴部区域为估算 → 不因 mouth 判重拍", retake == [], str(retake))
    check("判定：估算时仍会记下 mouth_low_seconds 供人参考",
          abs(rows[1]["mouth_low_seconds"] - 4.0) < 1e-6, str(rows[1]["mouth_low_seconds"]))

    # 只 2 秒不动 → 不重拍（偶尔一句不动是正常的）
    rows = rows_for()
    retake, _, _ = QA.judge_segments(rows, SEGS, {1: 200.0, 2: 200.0, 3: 200.0}, 1.0, [(12, 14)], "image_info", 30, 900)
    check("判定：嘴不动只 2 秒（<3）→ 不重拍", retake == [], str(retake))

    # 跨段的嘴不动区间，两段各算自己那部分
    rows = rows_for()
    QA.judge_segments(rows, SEGS, {1: 200.0, 2: 200.0, 3: 200.0}, 1.0, [(6, 14)], "image_info", 30, 900)
    check("判定：跨段区间按段切开算（6–14 → 段1 得 4 秒、段2 得 4 秒）",
          abs(rows[0]["mouth_low_seconds"] - 4.0) < 1e-6 and abs(rows[1]["mouth_low_seconds"] - 4.0) < 1e-6,
          "%s / %s" % (rows[0]["mouth_low_seconds"], rows[1]["mouth_low_seconds"]))


def test_judge_both():
    """两条同时命中：reasons 要有两个，auto_retake 才能分清是"换种子"还是"改提示词"。"""
    rows = rows_for()
    retake, _, _ = QA.judge_segments(rows, SEGS, {1: 200.0, 2: 50.0, 3: 200.0}, 1.0, [(12, 16)], "image_info", 30, 900)
    check("判定：糊 + 嘴不动同时命中 → reasons 为 ['sharp','mouth']",
          retake == [2] and rows[1]["reasons"] == ["sharp", "mouth"], str(rows[1]["reasons"]))


def test_judge_mouth_ratio():
    """嘴动量比：段内说话时嘴动量 / 全片中位数。小于 1 表示这段嘴比别处更僵。"""
    fps, n = 30, 900
    motion = np.full(n, 2.0)
    speak = np.ones(n, dtype=bool)
    motion[10 * fps:20 * fps] = 1.0          # 第 2 段只有全片一半
    rows = rows_for()
    QA.judge_segments(rows, SEGS, {1: 200.0, 2: 200.0, 3: 200.0}, 2.0, [], "image_info", fps, n,
                      mouth_motion=motion, speak=speak)
    check("嘴动量比：段2 是全片一半 → 0.5", abs(rows[1]["mouth_ratio"] - 0.5) < 0.02,
          str(rows[1]["mouth_ratio"]))
    check("嘴动量比：段1 正常 → 1.0", abs(rows[0]["mouth_ratio"] - 1.0) < 0.02,
          str(rows[0]["mouth_ratio"]))

    rows = rows_for()
    QA.judge_segments(rows, SEGS, {1: 200.0, 2: 200.0, 3: 200.0}, 2.0, [], "image_info", fps, n)
    check("嘴动量比：没传 mouth_motion → 全为 None，不崩", all(r["mouth_ratio"] is None for r in rows))


# ---------------------------------------------------------------- 人物一致性
def test_face():
    rows = rows_for()
    flags = QA.face_consistency({}, rows)
    check("人脸：一段都没检测到 → 不报警、face_offset 全 None",
          flags == [] and all(r["face_offset"] is None for r in rows))

    # 只有 2 段有结果：样本太少，不下结论
    rows = rows_for()
    flags = QA.face_consistency({1: (0.50, 0.3, 0.20), 2: (0.62, 0.3, 0.20)}, rows)
    check("人脸：只有 2 段有结果 → 不下结论（中位数没意义）", flags == [], str(flags))

    # 3 段都正常
    rows = rows_for()
    flags = QA.face_consistency({1: (0.49, 0.3, 0.20), 2: (0.50, 0.3, 0.20), 3: (0.51, 0.3, 0.20)}, rows)
    check("人脸：横向位置接近 → 不报警", flags == [], str(flags))

    # 第 3 段横向偏了 15% 画宽（> 5%）
    rows = rows_for()
    flags = QA.face_consistency({1: (0.42, 0.3, 0.20), 2: (0.43, 0.3, 0.20), 3: (0.72, 0.3, 0.20)}, rows)
    check("人脸：横向偏 29% → 报警并说明是构图/megapixels 问题",
          len(flags) == 1 and "段3" in flags[0] and "megapixels" in flags[0], str(flags))

    # 第 3 段脸变大 40%（> 15%）
    rows = rows_for()
    flags = QA.face_consistency({1: (0.50, 0.3, 0.20), 2: (0.50, 0.3, 0.20), 3: (0.50, 0.3, 0.28)}, rows)
    check("人脸：脸大 40% → 报警", len(flags) == 1 and "段3" in flags[0], str(flags))

    # 某段没有检测结果 → face_offset 为 None，但不影响其余段判定
    rows = rows_for()
    flags = QA.face_consistency({1: (0.50, 0.3, 0.20), 2: (0.50, 0.3, 0.20), 3: (0.50, 0.3, 0.20)}, rows)
    rows[1]["i"] = 99                       # 让它对不上任何 track 键
    flags2 = QA.face_consistency({1: (0.50, 0.3, 0.20), 2: (0.50, 0.3, 0.20), 3: (0.50, 0.3, 0.20)}, rows)
    check("人脸：某段查无结果 → 该段 face_offset=None，不误报", rows[1]["face_offset"] is None,
          str(rows[1]["face_offset"]))


# ---------------------------------------------------------------- 色彩异常
def test_color_anomalies():
    """注意数据形状：color_anomalies 收的是完整 Lab 三通道（它内部取 [: , 1:] 丢掉 L），
    ref_ab 则是已经取过 [1:] 的两通道。写成两通道会直接崩，这里统一用 (N, 3)。"""
    fps = 10

    def lab(n, L=128.0, a=128.0, b=148.0):
        return np.tile(np.array([L, a, b]), (n, 1))

    # 参考背景：Lab 的 a/b 偏移 (0, 20) —— 明显有彩度
    ref = np.array([128.0, 148.0])
    # 10 秒都一样：同色相、同彩度 → 无异常
    check("色彩：全程与图片一致 → 无异常", QA.color_anomalies(lab(100), ref, fps) == [])

    # 第 3–5 秒彩度掉到 20%（< 55%）→ 褪色
    colors = lab(100)
    colors[30:60, 2] = 128.0 + 4.0
    got = QA.color_anomalies(colors, ref, fps)
    check("色彩：彩度掉到 20% → 标出褪色区间 3–6 秒",
          len(got) == 1 and got[0][0] == 3 and got[0][1] == 6 and "褪色" in got[0][2], str(got))

    # 色相偏转 90°（a 从 0 变 20、b 从 20 变 0）→ 变成另一种色调
    colors = lab(100)
    colors[50:70, 1] = 148.0
    colors[50:70, 2] = 128.0
    got = QA.color_anomalies(colors, ref, fps)
    check("色彩：色相偏 90° → 标出变色调区间 5–7 秒",
          len(got) == 1 and got[0][0] == 5 and got[0][1] == 7 and "色调" in got[0][2], str(got))

    # 图片背景接近无彩色（|ref-128| < 6）：只看有没有染上颜色
    ref_gray = np.array([128.0, 130.0])
    check("色彩：图片背景无彩色 + 画面也无彩色 → 无异常",
          QA.color_anomalies(lab(100, b=130.0), ref_gray, fps) == [])
    colors = lab(100, b=130.0)
    colors[20:40, 2] = 160.0
    got = QA.color_anomalies(colors, ref_gray, fps)
    check("色彩：图片背景无彩色但画面染上颜色 → 标出",
          len(got) == 1 and "染上" in got[0][2], str(got))

    # 亮度变化不算异常（用户认可的正常波动）
    check("色彩：只变亮不变色 → 不算异常（亮度波动是正常的）",
          QA.color_anomalies(lab(100, L=180.0), ref, fps) == [])


# ---------------------------------------------------------------- 嘴部框定位
def test_mouth_box():
    """嘴部框从脸框推：横向 25%–75%、纵向 68%–98%。

    这个比例错了不会报错，只会让"嘴动量"这个指标静默失真——所以单独钉住。
    """
    box = QA.mouth_box_from_face((100, 200, 400, 400))
    x0, x1, y0, y1 = box
    check("嘴框：横向取脸的 25%–75%（100+100=200 到 100+300=400）",
          (x0, x1) == (200, 400), str((x0, x1)))
    check("嘴框：纵向取脸的 68%–98%（200+272=472 到 200+392=592）",
          (y0, y1) == (472, 592), str((y0, y1)))
    check("嘴框：宽度是脸的 50%", (x1 - x0) == 200, str(x1 - x0))

    # 脸整体右移 → 嘴框跟着右移同样的量
    a = QA.mouth_box_from_face((100, 200, 400, 400))
    b = QA.mouth_box_from_face((400, 200, 400, 400))
    check("嘴框：脸右移 300px → 嘴框也右移 300px（这正是固定框脱靶的根源）",
          b[0] - a[0] == 300, str(b[0] - a[0]))


def test_pick_segment_box():
    """按时刻选段嘴框：人物各段位置不同，逐帧必须换框。"""
    bounds = [(0.0, 9.0, 1), (9.0, 17.9, 2), (17.9, 27.6, 3)]
    boxes = {1: (800, 950, 320, 400), 2: (890, 1040, 315, 400), 3: (1020, 1170, 305, 395)}
    check("选框：t=4 → 段1 的框", QA.pick_segment_box(boxes, bounds, 4.0) == boxes[1])
    check("选框：t=13 → 段2 的框", QA.pick_segment_box(boxes, bounds, 13.0) == boxes[2])
    check("选框：t=20 → 段3 的框", QA.pick_segment_box(boxes, bounds, 20.0) == boxes[3])
    check("选框：段边界左侧归上一段（t=8.99）", QA.pick_segment_box(boxes, bounds, 8.99) == boxes[1])
    check("选框：段边界右侧归下一段（t=9.0）", QA.pick_segment_box(boxes, bounds, 9.0) == boxes[2])
    check("选框：超出片尾 → None", QA.pick_segment_box(boxes, bounds, 99.0) is None)
    check("选框：没有分段结果 → None（回退到全片一个框）",
          QA.pick_segment_box(None, bounds, 4.0) is None)
    check("选框：该段没有框 → None", QA.pick_segment_box({1: (0, 1, 2, 3)}, [(0.0, 9.0, 9)], 4.0) is None)


def main():
    test_low_windows()
    test_merge()
    test_judge_sharp()
    test_judge_mouth()
    test_judge_both()
    test_judge_mouth_ratio()
    test_face()
    test_color_anomalies()
    test_mouth_box()
    test_pick_segment_box()
    print("\n通过 %d/%d" % (sum(results), len(results)))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
