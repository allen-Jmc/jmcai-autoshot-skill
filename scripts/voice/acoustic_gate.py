"""配音声学质量门禁（Acoustic Quality Gate）与音色特征比对模块。

提供：
1. 声学特征提取（MFCC 13维、F0 pyin 基频中位数/均值/标准差/有声比例、RMS 能量）
2. 环境自适应嗅探（当前 Python 若无 librosa，自动调度 ComfyUI 嵌入式 Python 进行提取）
3. 声学特征画像对比（归一化 MFCC 欧氏距离、基频偏离率、综合打分与门禁判定）
4. 旧项目 voice.json 声学画像懒加载补全

CLI 用法：
  python acoustic_gate.py --sniff
  python acoustic_gate.py --extract 样本.mp3 --json
  python acoustic_gate.py --compare 候选.wav 参考.mp3
"""

import argparse
import json
import math
import os
import subprocess
import sys

# 默认门禁阈值
DEFAULT_MFCC_THRESHOLD = 0.20   # L2 归一化 MFCC 欧氏距离阈值（<=0.20 判定为音色良好）
DEFAULT_F0_THRESHOLD = 0.25     # F0 中位数相对偏离率阈值（<=25%）
# 语调动态：候选的 f0_std / rms_std 相对参考的比值下限。只提示，不拦截。
# 实测标定（5 段正常配音 vs 参考）：f0 比值 0.89–1.12、rms 比值 0.87–1.09，取 0.70 留出足够余量。
F0_DYNAMIC_MIN = 0.70
RMS_DYNAMIC_MIN = 0.70
MIN_VOICED_RATIO = 0.15         # 最小有声帧比例，防静音或空跑

# 探测"带 librosa 的 ComfyUI 嵌入式 Python"。
# 这里不能写死任何人的安装路径（早期版本写死过作者本机的绝对路径，分享给别人只会暴露用户名、
# 而且对别人完全无效）。做法：先看环境变量，再扫几个常见的安装根目录找 python_embeded/python.exe。
COMFYUI_SEARCH_ROOTS = [
    os.environ.get("COMFYUI_HOME"),
    os.path.expanduser("~/ComfyUI_windows_portable"),
    os.path.expanduser("~/Pictures/ComfyUI_windows_portable-G313"),
    os.path.expanduser("~/Downloads/ComfyUI_windows_portable"),
    os.path.expanduser("~/Desktop/ComfyUI_windows_portable"),
    r"C:\ComfyUI_windows_portable",
    r"D:\ComfyUI_windows_portable",
    "/opt/ComfyUI",
    os.path.expanduser("~/ComfyUI"),
]


def has_local_librosa():
    """检查当前 Python 进程是否能直接导入 librosa 与 scipy。"""
    try:
        import librosa  # noqa
        import scipy    # noqa
        return True
    except Exception:
        return False


def _iter_comfyui_pythons():
    """按常见安装位置找 python_embeded/python.exe 或 venv 里的 python。"""
    import glob
    for root in COMFYUI_SEARCH_ROOTS:
        if not root:
            continue
        # 直接就是 python_embeded 目录，或者下面还有一层（ComfyUI_windows_portable-X/）
        for pat in ("python_embeded/python.exe", "python_embeded/bin/python3",
                    "*/python_embeded/python.exe", "*/*/python_embeded/python.exe",
                    "venv/bin/python", ".venv/bin/python"):
            for hit in glob.glob(os.path.join(root, pat)):
                yield hit


def find_comfyui_python():
    """查找具备完整音频处理库（librosa）的 Python 解释器路径。

    优先环境变量（用户显式指定最可靠），其次扫常见安装位置。找不到就返回 None，
    调用方会退回"当前解释器"。
    """
    for p in (os.environ.get("COMFYUI_PYTHON"),
              os.environ.get("COMFYUI_PYTHON_PATH"),
              os.environ.get("JMCAI_PYTHON")):
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    for p in _iter_comfyui_pythons():
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None


def extract_profile_direct(audio_path, target_sr=16000):
    """在具备 librosa 的当前环境中直接提取声学特征画像。"""
    import librosa
    import numpy as np

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    # 读取并统一采样率
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    dur = float(len(y) / sr)
    if dur <= 0.05:
        raise ValueError(f"音频过短 ({dur:.2f}s)，无法提取有效声学特征")

    # 1. 能量与 RMS
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    rms_mean = float(np.mean(rms)) if len(rms) else 0.0
    rms_std = float(np.std(rms)) if len(rms) else 0.0

    # 2. 基频 F0 提取（基于概率 YIN 算法 pyin）
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=65,
        fmax=400,
        sr=sr,
        frame_length=1024,
        hop_length=256
    )
    voiced_mask = ~np.isnan(f0)
    voiced_f0 = f0[voiced_mask]
    total_frames = len(f0)
    voiced_frames = len(voiced_f0)

    f0_median = float(np.median(voiced_f0)) if voiced_frames else 0.0
    f0_mean = float(np.mean(voiced_f0)) if voiced_frames else 0.0
    f0_std = float(np.std(voiced_f0)) if voiced_frames else 0.0
    f0_voiced_ratio = float(voiced_frames / total_frames) if total_frames else 0.0

    # 3. MFCC 提取（13 维系数，全段均值池化）
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=256)
    mfcc_mean_arr = np.mean(mfcc, axis=1)

    norm = float(np.linalg.norm(mfcc_mean_arr))
    if norm > 1e-6:
        mfcc_norm_arr = mfcc_mean_arr / norm
    else:
        mfcc_norm_arr = mfcc_mean_arr

    profile = {
        "version": "1.0",
        "sample_rate": sr,
        "duration": round(dur, 3),
        "f0_median": round(f0_median, 2),
        "f0_mean": round(f0_mean, 2),
        "f0_std": round(f0_std, 2),
        "f0_voiced_ratio": round(f0_voiced_ratio, 4),
        "rms_mean": round(rms_mean, 5),
        "rms_std": round(rms_std, 5),
        "mfcc_mean": [round(float(v), 3) for v in mfcc_mean_arr],
        "mfcc_norm": [round(float(v), 5) for v in mfcc_norm_arr],
    }
    return profile


def extract_profile_subprocess(audio_path, python_exe=None):
    """通过子进程调用外部 Python 提取声学特征画像。"""
    exe = python_exe or find_comfyui_python()
    if not exe:
        raise RuntimeError("当前 Python 环境缺少 librosa，且未找到 ComfyUI 嵌入式 Python 解释器。")

    this_script = os.path.abspath(__file__)
    cmd = [exe, this_script, "--extract", os.path.abspath(audio_path), "--json"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"提取声学特征失败 (exit {proc.returncode}):\n{proc.stderr.strip()}")

    out = proc.stdout.strip()
    try:
        data = json.loads(out)
        return data
    except Exception as e:
        raise RuntimeError(f"解析声学特征 JSON 失败: {e}\n输出内容:\n{out}")


def extract_profile(audio_path):
    """自适应提取音频声学画像：优先本地 direct，无环境则子进程桥接。"""
    if has_local_librosa():
        return extract_profile_direct(audio_path)
    return extract_profile_subprocess(audio_path)


def mfcc_euclidean_distance(cand_norm, ref_norm):
    """计算两组 L2 归一化 MFCC 向量的欧氏距离。"""
    if not cand_norm or not ref_norm or len(cand_norm) != len(ref_norm):
        return 1.0
    dist_sq = sum((c - r) ** 2 for c, r in zip(cand_norm, ref_norm))
    return math.sqrt(dist_sq)


def compare_profiles(cand_profile, ref_profile,
                     mfcc_threshold=DEFAULT_MFCC_THRESHOLD,
                     f0_threshold=DEFAULT_F0_THRESHOLD):
    """对比候选音频与参考基准的声学画像。

    返回 dict:
      passed: bool
      mfcc_dist: float
      f0_cand_med: float
      f0_ref_med: float
      f0_diff_hz: float
      f0_ratio: float
      penalty: float
      score: float (0~100)
      reasons: list[str]
    """
    if not cand_profile or not ref_profile:
        return {
            "passed": False,
            "mfcc_dist": 1.0,
            "f0_cand_med": 0.0,
            "f0_ref_med": 0.0,
            "f0_diff_hz": 0.0,
            "f0_ratio": 1.0,
            "penalty": 2.0,
            "score": 0.0,
            "reasons": ["缺少声学画像数据"],
            "notes": [],
        }

    c_norm = cand_profile.get("mfcc_norm") or []
    r_norm = ref_profile.get("mfcc_norm") or []
    mfcc_dist = mfcc_euclidean_distance(c_norm, r_norm)

    f0_cand = cand_profile.get("f0_median") or 0.0
    f0_ref = ref_profile.get("f0_median") or 0.0
    f0_diff_hz = f0_cand - f0_ref

    if f0_ref > 10.0:
        f0_ratio = abs(f0_diff_hz) / f0_ref
    else:
        f0_ratio = 0.0

    voiced_ratio = cand_profile.get("f0_voiced_ratio", 0.0)

    # 综合惩罚分：MFCC 距离权重 1.0，F0 相对偏差权重 0.6
    penalty = mfcc_dist * 1.0 + f0_ratio * 0.6
    # 得分：基准 100 分，扣除 penalty * 150
    score = max(0.0, min(100.0, 100.0 - penalty * 150.0))

    reasons = []
    if mfcc_dist > mfcc_threshold:
        reasons.append(f"MFCC距离偏高({mfcc_dist:.4f} > {mfcc_threshold:.2f})")
    if f0_ratio > f0_threshold:
        sign = "+" if f0_diff_hz > 0 else ""
        reasons.append(f"F0基频偏离超标({sign}{f0_diff_hz:.1f}Hz, 偏离{f0_ratio*100:.1f}% > {f0_threshold*100:.0f}%)")
    if voiced_ratio < MIN_VOICED_RATIO:
        reasons.append(f"有效发音帧偏少({voiced_ratio*100:.1f}% < {MIN_VOICED_RATIO*100:.0f}%)")

    passed = (len(reasons) == 0)

    # ---- 语调动态（只提示，不参与 passed）
    # MFCC/F0 中位数只能说明"音色像不像"，测不出"语调是不是发僵"。
    # 发僵的段往往是中位数完全对、但基频几乎不动——所以看 f0_std 和 rms_std 的**相对值**。
    # 实测标定（20261019-incremental-intro，5 段正常配音 vs 参考）：f0 比值 0.89–1.12、rms 比值 0.87–1.09。
    notes, f0_dyn, rms_dyn = [], None, None
    c_std = cand_profile.get("f0_std")
    r_std = ref_profile.get("f0_std")
    if c_std and r_std:
        f0_dyn = round(float(c_std) / float(r_std), 3)
        if f0_dyn < F0_DYNAMIC_MIN:
            notes.append("语调偏平：基频起伏只有参考的 %.0f%%（正常 85%%–115%%）。"
                         "如果这段是在念清单／报参数，可以接受；否则换一张卡" % (f0_dyn * 100))
    c_rms = cand_profile.get("rms_std")
    r_rms = ref_profile.get("rms_std")
    if c_rms and r_rms:
        rms_dyn = round(float(c_rms) / float(r_rms), 3)
        if rms_dyn < RMS_DYNAMIC_MIN:
            notes.append("语气偏平：能量起伏只有参考的 %.0f%%（正常 85%%–115%%）" % (rms_dyn * 100))

    return {
        "passed": passed,
        "mfcc_dist": round(mfcc_dist, 4),
        "f0_cand_med": round(f0_cand, 1),
        "f0_ref_med": round(f0_ref, 1),
        "f0_diff_hz": round(f0_diff_hz, 1),
        "f0_ratio": round(f0_ratio, 4),
        "f0_dynamic_ratio": f0_dyn,
        "rms_dynamic_ratio": rms_dyn,
        "penalty": round(penalty, 4),
        "score": round(score, 1),
        "reasons": reasons,
        "notes": notes,
    }


def evaluate_audio(cand_audio_path, ref_audio_or_profile,
                   mfcc_threshold=DEFAULT_MFCC_THRESHOLD,
                   f0_threshold=DEFAULT_F0_THRESHOLD):
    """高阶便捷入口：评估候选音频是否通过参考音色的声学门禁。

    ref_audio_or_profile 可以是参考音频路径，也可以是已提取的 profile dict。
    """
    cand_profile = extract_profile(cand_audio_path)
    if isinstance(ref_audio_or_profile, dict):
        ref_profile = ref_audio_or_profile
    elif isinstance(ref_audio_or_profile, str) and os.path.isfile(ref_audio_or_profile):
        ref_profile = extract_profile(ref_audio_or_profile)
    else:
        raise ValueError("ref_audio_or_profile 必须是有效的 profile dict 或音频文件路径")

    res = compare_profiles(cand_profile, ref_profile, mfcc_threshold, f0_threshold)
    res["cand_profile"] = cand_profile
    return res


def ensure_voice_profile(voice_info, workdir=None, save_back=True):
    """确保 voice_info 具有 acoustic_profile；若缺失则懒加载提取并补全。"""
    if not isinstance(voice_info, dict):
        return None

    if voice_info.get("acoustic_profile"):
        prof = voice_info["acoustic_profile"]
        # 老画像没有 rms_std（这个字段是后加的）：如果参考音频还在，重算一次补上，
        # 否则"语气偏平"那一半会静默失效——宁可多花几秒，也不要让检查看起来做了其实没做。
        if prof.get("mfcc_norm") and prof.get("rms_std") is None:
            sample = voice_info.get("voice_id") or voice_info.get("sample")
            if sample and os.path.isfile(sample):
                try:
                    fresh = extract_profile(sample)
                    prof.update({"rms_std": fresh.get("rms_std"),
                                 "f0_std": prof.get("f0_std") or fresh.get("f0_std")})
                    voice_info["acoustic_profile"] = prof
                    if save_back and workdir:
                        vj = os.path.join(workdir, "voice.json")
                        if os.path.isfile(vj):
                            data = json.load(open(vj, encoding="utf-8"))
                            data["acoustic_profile"] = prof
                            json.dump(data, open(vj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                except Exception as e:
                    print(f"注意: 补算参考音色的 rms_std 失败（语气动态这项会跳过）: {e}", file=sys.stderr)
        return prof

    # 尝试寻找参考音频
    sample_path = voice_info.get("voice_id")
    if not sample_path or not os.path.isfile(sample_path):
        sample_path = voice_info.get("sample")

    if not sample_path or not os.path.isfile(sample_path):
        if workdir:
            for ext in [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"]:
                p = os.path.join(workdir, "voice_sample" + ext)
                if os.path.isfile(p):
                    sample_path = p
                    break

    if not sample_path or not os.path.isfile(sample_path):
        return None

    try:
        prof = extract_profile(sample_path)
        voice_info["acoustic_profile"] = prof
        if save_back and workdir:
            vj = os.path.join(workdir, "voice.json")
            if os.path.isfile(vj):
                try:
                    data = json.load(open(vj, encoding="utf-8"))
                    data["acoustic_profile"] = prof
                    json.dump(data, open(vj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                except Exception:
                    pass
        return prof
    except Exception as e:
        print(f"注意: 提取参考音色声学特征失败: {e}", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser(description="声学质量门禁与音色比对工具")
    ap.add_argument("--sniff", action="store_true", help="探测当前环境与 ComfyUI Python 解释器状态")
    ap.add_argument("--extract", help="提取指定音频的声学画像")
    ap.add_argument("--json", action="store_true", help="以纯 JSON 格式输出特征结果")
    ap.add_argument("--compare", help="候选音频路径，与 --ref 进行对比")
    ap.add_argument("--ref", help="参考音频路径或包含 acoustic_profile 的 voice.json")
    ap.add_argument("--mfcc-thresh", type=float, default=DEFAULT_MFCC_THRESHOLD)
    ap.add_argument("--f0-thresh", type=float, default=DEFAULT_F0_THRESHOLD)
    a = ap.parse_args()

    if a.sniff:
        has_lib = has_local_librosa()
        comfy_py = find_comfyui_python()
        print("=== 声学门禁环境探测 ===")
        print(f"当前 Python: {sys.executable}")
        print(f"当前环境自带 librosa: {'是 (可直接运行)' if has_lib else '否 (将使用子进程调度)'}")
        print(f"ComfyUI 嵌入式 Python: {comfy_py or '未找到'}")
        if not has_lib and not comfy_py:
            print("【警告】未找到可用音频环境，请配置环境变量 COMFYUI_PYTHON！")
            sys.exit(1)
        else:
            print("【状态】音频处理环境就绪。")
        return

    if a.extract:
        prof = extract_profile(a.extract)
        if a.json:
            print(json.dumps(prof, ensure_ascii=False))
        else:
            print(f"音频文件: {a.extract}")
            print(f"时长: {prof['duration']}s, 采样率: {prof['sample_rate']}Hz")
            print(f"F0 基频中位数: {prof['f0_median']}Hz (均值 {prof['f0_mean']}Hz, 标准差 {prof['f0_std']}Hz, 有声比例 {prof['f0_voiced_ratio']*100:.1f}%)")
            print(f"RMS 能量: {prof['rms_mean']}")
            print(f"13维归一化 MFCC: {prof['mfcc_norm']}")
        return

    if a.compare:
        if not a.ref:
            sys.exit("--compare 需要配合 --ref 指定参考文件")

        if a.ref.endswith(".json") and os.path.isfile(a.ref):
            ref_data = json.load(open(a.ref, encoding="utf-8"))
            ref_prof = ref_data.get("acoustic_profile") or extract_profile(ref_data.get("voice_id") or ref_data.get("sample"))
        else:
            ref_prof = extract_profile(a.ref)

        cand_prof = extract_profile(a.compare)
        res = compare_profiles(cand_prof, ref_prof, a.mfcc_thresh, a.f0_thresh)

        if a.json:
            print(json.dumps(res, ensure_ascii=False))
        else:
            print(f"候选音频: {a.compare}")
            print(f"参考基准: {a.ref}")
            print(f"MFCC 欧氏距离: {res['mfcc_dist']} (阈值 <= {a.mfcc_thresh})")
            print(f"F0 基频对比: 候选 {res['f0_cand_med']}Hz vs 参考 {res['f0_ref_med']}Hz (差值 {res['f0_diff_hz']:+}Hz, 偏离 {res['f0_ratio']*100:.1f}%)")
            print(f"综合评分: {res['score']} 分 (惩罚分 {res['penalty']})")
            print(f"门禁判定: {'[PASS 通过]' if res['passed'] else '[FAIL 未通过: ' + '; '.join(res['reasons']) + ']'}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
