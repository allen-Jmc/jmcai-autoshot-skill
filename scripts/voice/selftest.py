# -*- coding: utf-8 -*-
"""配音阶段的离线回归测试：声学门禁的判定、抽卡自愈的循环、缓存键、拆段、wav 读写。

不联网、不需要桌面端、不需要真实音频——把 synth_text 和 evaluate_audio 换成假的，
只验证"判定和循环"这两处最容易改坏、又最不容易在实跑里发现的逻辑。

  python scripts/voice/selftest.py
"""
import os
import sys
import tempfile
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import acoustic_gate as AG      # noqa: E402
import synth as S               # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("✓" if ok else "✗"), name, detail if not ok else "")


def profile(mfcc, f0, voiced=0.45, f0_std=None, rms_std=None):
    """造一个声学画像：mfcc_norm 用单位向量的偏移来模拟不同的音色。"""
    import math
    n = len(mfcc)
    norm = math.sqrt(sum(x * x for x in mfcc)) or 1.0
    p = {"mfcc_norm": [x / norm for x in mfcc], "mfcc_mean": list(mfcc),
         "f0_median": f0, "f0_voiced_ratio": voiced}
    if f0_std is not None:
        p["f0_std"] = f0_std
    if rms_std is not None:
        p["rms_std"] = rms_std
    return p


REF = profile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0], 231.0)
# 带动态特征的参考（真实项目实测：f0_std 34.83、rms_std 0.0817）
REF_DYN = profile(REF["mfcc_mean"], 231.0, f0_std=34.83, rms_std=0.0817)


def test_gate():
    same = AG.compare_profiles(REF, REF)
    check("门禁：和基准完全一致 → 通过、满分", same["passed"] and same["score"] > 99, str(same))

    off = profile([3.0, 1.0, 2.0, 5.0, 4.0, 9.0, 7.0, 8.0, 12.0, 10.0, 11.0, 13.0, 6.0], 231.0)
    r = AG.compare_profiles(off, REF)
    check("门禁：音色偏了 → 不通过，理由里有 MFCC", (not r["passed"]) and any("MFCC" in x for x in r["reasons"]), str(r["reasons"]))

    high = profile(REF["mfcc_mean"], 231.0 * 1.4)
    r = AG.compare_profiles(high, REF)
    check("门禁：F0 偏高 40% → 不通过，理由里有 F0", (not r["passed"]) and any("F0" in x for x in r["reasons"]), str(r["reasons"]))

    quiet = profile(REF["mfcc_mean"], 231.0, voiced=0.05)
    r = AG.compare_profiles(quiet, REF)
    check("门禁：有效发音帧太少 → 不通过", (not r["passed"]) and any("发音" in x for x in r["reasons"]), str(r["reasons"]))

    r = AG.compare_profiles(None, REF)
    check("门禁：缺画像 → 判不通过且得 0 分，不是静默放行", r["passed"] is False and r["score"] == 0.0, str(r))

    check("MFCC：同一向量距离为 0", AG.mfcc_euclidean_distance([1, 0, 0], [1, 0, 0]) == 0.0)
    check("MFCC：长度不一致返回 1.0 而不是崩", AG.mfcc_euclidean_distance([1, 0], [1, 0, 0]) == 1.0)


def test_dynamic():
    """语调动态：只提示不拦截——发僵的段往往音色完全对、但基频几乎不动。"""
    flat = profile(REF_DYN["mfcc_mean"], 231.0, f0_std=34.83 * 0.5, rms_std=0.0817 * 0.5)
    r = AG.compare_profiles(flat, REF_DYN)
    check("语调动态：起伏只有参考 50% → 仍然 passed（不拦截）", r["passed"], str(r["reasons"]))
    check("语调动态：起伏只有参考 50% → 给出 f0 和 rms 两条提示",
          len(r["notes"]) == 2 and any("语调偏平" in n for n in r["notes"]) and any("语气偏平" in n for n in r["notes"]),
          str(r["notes"]))
    check("语调动态：比值算对（0.5）", abs(r["f0_dynamic_ratio"] - 0.5) < 0.01, str(r["f0_dynamic_ratio"]))

    normal = profile(REF_DYN["mfcc_mean"], 231.0, f0_std=34.83 * 1.02, rms_std=0.0817 * 1.05)
    r = AG.compare_profiles(normal, REF_DYN)
    check("语调动态：起伏正常 → 没有提示", r["passed"] and r["notes"] == [], str(r["notes"]))

    r = AG.compare_profiles(normal, REF)                 # 参考没有 f0_std（老画像）
    check("语调动态：参考没有 f0_std → 跳过这项，不报错",
          r["passed"] and r["f0_dynamic_ratio"] is None and r["notes"] == [], str(r))


def fake_evals(plan):
    """按顺序返回预设的门禁结果；用完之后一直返回最后一个。"""
    state = {"i": 0}

    def _eval(path, ref, mfcc_thresh=None, f0_thresh=None):
        i = min(state["i"], len(plan) - 1)
        state["i"] += 1
        return plan[i]
    return _eval, state


def bad(penalty):
    return {"passed": False, "mfcc_dist": 0.5, "f0_cand_med": 300.0, "f0_ref_med": 231.0,
            "f0_diff_hz": 69.0, "f0_ratio": 0.3, "penalty": penalty, "score": 10.0,
            "reasons": ["MFCC距离偏高"]}


def good():
    return {"passed": True, "mfcc_dist": 0.12, "f0_cand_med": 235.0, "f0_ref_med": 231.0,
            "f0_diff_hz": 4.0, "f0_ratio": 0.02, "penalty": 0.13, "score": 80.0, "reasons": []}


def test_reroll():
    tmp = tempfile.mkdtemp(prefix="voice_selftest_")
    calls = {"n": 0}

    def fake_synth(text, voice, cache_dir, label, depth=0, seed=42):
        calls["n"] += 1
        p = os.path.join(tmp, "seg_%d.wav" % calls["n"])
        open(p, "wb").close()
        return p, [], False

    orig_synth, orig_eval = S.synth_text, AG.evaluate_audio
    S.synth_text = fake_synth
    try:
        # 1) 第一次就过：只该抽一次
        calls["n"] = 0
        ev, _ = fake_evals([good()])
        AG.evaluate_audio = ev
        path, notes, hit, seed, res = S.synth_with_gate("文本", {}, tmp, "段1", 42, ref_profile=REF)
        check("抽卡：一次就过 → 只合成一次", calls["n"] == 1 and res["passed"], "调用 %d 次" % calls["n"])

        # 2) 前两次不过、第三次过
        calls["n"] = 0
        ev, st = fake_evals([bad(0.5), bad(0.4), good()])
        AG.evaluate_audio = ev
        path, notes, hit, seed, res = S.synth_with_gate("文本", {}, tmp, "段2", 42, max_rerolls=3, ref_profile=REF)
        check("抽卡：第三次才过 → 合成 3 次并采纳第三次", calls["n"] == 3 and res["passed"], "调用 %d 次" % calls["n"])

        # 3) 全不过 → 取惩罚最小的那次
        calls["n"] = 0
        ev, st = fake_evals([bad(0.9), bad(0.3), bad(0.6)])
        AG.evaluate_audio = ev
        path, notes, hit, seed, res = S.synth_with_gate("文本", {}, tmp, "段3", 42, max_rerolls=2, ref_profile=REF)
        check("抽卡：全不过 → 选惩罚最小的候选（0.3 那次）",
              (not res["passed"]) and abs(res["penalty"] - 0.3) < 1e-9 and any("最优候选" in n for n in notes),
              str((res["penalty"], notes)))

        # 4) 关掉重抽 → 只抽一次就返回
        calls["n"] = 0
        ev, _ = fake_evals([bad(0.9)])
        AG.evaluate_audio = ev
        path, notes, hit, seed, res = S.synth_with_gate("文本", {}, tmp, "段4", 42, max_rerolls=3,
                                                        ref_profile=REF, allow_reroll=False)
        check("抽卡：allow_reroll=False → 只抽一次", calls["n"] == 1, "调用 %d 次" % calls["n"])

        # 5) 没有参考画像 → 直接返回，不评估
        calls["n"] = 0
        AG.evaluate_audio = lambda *a, **k: (_ for _ in ()).throw(AssertionError("不该评估"))
        path, notes, hit, seed, res = S.synth_with_gate("文本", {}, tmp, "段5", 42, ref_profile=None)
        check("抽卡：没有参考画像 → 跳过门禁，不报错", calls["n"] == 1 and res is None, "调用 %d 次" % calls["n"])

        # 6) 评估本身抛异常 → 放行，不卡住整条片子
        calls["n"] = 0
        AG.evaluate_audio = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("librosa 挂了"))
        path, notes, hit, seed, res = S.synth_with_gate("文本", {}, tmp, "段6", 42, ref_profile=REF)
        check("抽卡：评估异常 → 放行（不让环境问题挡住出片）", calls["n"] == 1 and res is None, "调用 %d 次" % calls["n"])
    finally:
        S.synth_text, AG.evaluate_audio = orig_synth, orig_eval


def test_cache_adopt():
    """兜底选出的最优候选必须写回"默认种子"的缓存槽位。

    cache_key 有意让默认种子不进 key，所以第 1 次抽卡占的是 K.wav，重抽占 K_<seed>.wav。
    不写回的话，下次重跑命中 K.wav —— 也就是第一次被门禁判掉的那张坏卡。
    实跑踩过：段1 第 1 次 MFCC 0.2924、第 4 次 0.2010，重跑别段时段1 从缓存拿回了 0.2924。
    """
    tmp = tempfile.mkdtemp(prefix="voice_adopt_")
    voice = {"voice_id": "fake_id", "mode": "clone"}
    text = "写回测试用的一句话"
    default_path = os.path.join(tmp, S.cache_key(text, voice["voice_id"], S.DEFAULT_TTS_SEED) + ".wav")
    made = {"n": 0}

    def fake_synth(t, v, cache_dir, label, depth=0, seed=42):
        """模拟真实缓存语义：默认种子写 K.wav，换种子写 K_<seed>.wav，各写不同内容。"""
        made["n"] += 1
        if seed == S.DEFAULT_TTS_SEED:
            p = default_path
        else:
            p = os.path.join(tmp, S.cache_key(t, v.get("voice_id", ""), seed) + ".wav")
        with open(p, "w", encoding="utf-8") as f:
            f.write("v%d" % made["n"])          # 用内容区分是哪一次的作品
        return p, [], False

    orig_synth, orig_eval = S.synth_text, AG.evaluate_audio
    S.synth_text = fake_synth
    try:
        # 三次都不过，惩罚最小的是第 2 次（0.3）
        made["n"] = 0
        ev, _ = fake_evals([bad(0.9), bad(0.3), bad(0.6)])
        AG.evaluate_audio = ev
        path, notes, hit, seed, res = S.synth_with_gate(text, voice, tmp, "段A", 42, max_rerolls=2, ref_profile=REF)
        adopted = open(default_path, encoding="utf-8").read()
        check("写回：兜底选中的最优候选被复制进默认缓存槽位", adopted == "v2", "默认槽位=%s 期望 v2" % adopted)
        check("写回：返回的路径仍指向最优候选本身", os.path.basename(path) != os.path.basename(default_path),
              "%s vs %s" % (path, default_path))

        # 重抽后才通过的，同样要写回
        made["n"] = 0
        ev, _ = fake_evals([bad(0.8), good()])
        AG.evaluate_audio = ev
        path, notes, hit, seed, res = S.synth_with_gate(text, voice, tmp, "段B", 42, max_rerolls=2, ref_profile=REF)
        adopted = open(default_path, encoding="utf-8").read()
        check("写回：重抽后才通过 → 通过的版本也写回默认槽位", adopted == "v2" and res["passed"],
              "默认槽位=%s passed=%s" % (adopted, res["passed"]))

        # 第一次就过：本来就在默认槽位，不该多余复制（内容仍是 v1）
        made["n"] = 0
        ev, _ = fake_evals([good()])
        AG.evaluate_audio = ev
        path, notes, hit, seed, res = S.synth_with_gate(text, voice, tmp, "段C", 42, ref_profile=REF)
        adopted = open(default_path, encoding="utf-8").read()
        check("写回：第一次就过 → 默认槽位就是它本身，不变", adopted == "v1", adopted)
    finally:
        S.synth_text, AG.evaluate_audio = orig_synth, orig_eval


def test_utils():
    k1 = S.cache_key("同一段话", {"mode": "clone"})
    k2 = S.cache_key("同一段话", {"mode": "clone"})
    check("缓存键：同样的输入得到同一个键", k1 == k2, k1)
    check("缓存键：换种子就换键", S.cache_key("同一段话", {"mode": "clone"}, 99) != k1)
    check("缓存键：换文本就换键", S.cache_key("另一段话", {"mode": "clone"}) != k1)

    long_text = "第一句话在这里，稍微长一点点。" * 4
    half = S.split_half(long_text)
    check("拆段：在句末标点处对半拆，拼回来还是原文",
          half is not None and (half[0] + half[1]) == long_text, str(half and [len(half[0]), len(half[1])]))
    check("拆段：没有句末标点时返回 None", S.split_half("一二三四五六七八九十" * 4) is None)

    tmp = tempfile.mkdtemp(prefix="voice_wav_")
    p = os.path.join(tmp, "a.wav")
    S.write_wav(p, [0.0] * 100 + [0.5] * 100 + [0.0] * 100, 16000)
    x, sr = S.read_wav(p)
    check("wav 读写：写 300 个采样读回来还是 300 个", len(x) == 300 and sr == 16000, "%d 个" % len(x))
    y = S.trim(x, 16000)
    check("trim：去掉了首尾静音，比原来短", len(y) < len(x), "%d → %d" % (len(x), len(y)))

    q = os.path.join(tmp, "b.wav")
    S.write_wav(q, [0.3] * 50, 16000)
    dst = os.path.join(tmp, "c.wav")
    S.concat_wavs([p, q], dst, gap=0.1)
    z, _ = S.read_wav(dst)
    check("concat：两段相加 + 0.1 秒间隔（1600 采样）",
          abs(len(z) - (300 + 50 + 1600)) <= 2, "%d 个" % len(z))


def main():
    test_gate()
    test_dynamic()
    test_reroll()
    test_cache_adopt()
    test_utils()
    print("\n通过 %d/%d" % (sum(results), len(results)))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
