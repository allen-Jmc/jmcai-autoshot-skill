"""把各段提交到本机 JMCAI Comfypet 桌面端的数字人工作流（minimax-h3-copy-json），带重试与断点续跑。

  python run_avatar.py --workdir <项目目录>                  # 只做检查和预览，不提交
  python run_avatar.py --workdir <项目目录> --yes            # 提交到桌面端算力（本地，无费用）
  python run_avatar.py --workdir <项目目录> --yes --only 2   # 只重跑第 2 段，其余沿用已有结果
  python run_avatar.py --workdir <项目目录> --yes --seed random   # 每段换一个随机种子

本地模式没有 API Key、没有会员档位、没有计费；算力来自桌面端里绑定的算力目标（本机 ComfyUI 或外部 4090）。
分段音频与底图直接以**本机绝对路径**提交（bridge 在 127.0.0.1 时会自动把文件送到算力目标），不需要上传。

重跑与补救：
  * **同样的输入重跑通常得到同样的结果**。某段效果不满意要重跑时，至少换一样东西：
    先用 `--seed random` 换种子，同一处还出问题再改提示词（prompt_src.json → build_prompts.py）。
  * **单段补救最多 2 次**（result.json 的 retakes 记录）。到上限会停下并说明，
    要突破得显式加 `--force-retake`——避免在同一个坑里反复烧时间。
  * 先遮盖，再重跑：画面某几秒有问题，优先交给包装阶段盖过去，比重新生成划算。
  * 旧片段不丢：重跑前把上一版收进 clips/takes/（见 --keep-take），随时可以换回来。

重试与续跑：
  * 提交：连接错误、队列/限流类拒绝退避 20 秒重试，最多 12 次。
  * 轮询：连接错误容忍若干次；超时（--timeout-min）后**不重新提交**，运行号保留在 result.json 的 pending 里，
    再次运行本脚本会继续等同一个运行。
  * 运行失败：参数/权限类直接停下并报告；其余退避 15 秒后重提交，最多 --retries 次（默认 1）。
  * 运行号一提交就写入 result.json 的 pending；脚本被中断后重跑，会先续上这些运行，不会全部重新提交。
"""
import argparse
import json
import os
import random
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# 单段补救上限：超过就停下，逼着换一样东西（种子／提示词），而不是原地反复重跑
MAX_RETAKES = 2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

lock = threading.Lock()
NON_RETRYABLE = re.compile(r"参数错误|invalid ?param|未授权|无权限|unauthorized|forbidden|没有启用|已禁用|不存在", re.I)
RATE_LIMIT_WORDS = ("并发", "排队", "队列", "限流", "concurr", "queue", "limit", "频繁", "busy")
# 桌面端等远端输出的上限（实测约 180 秒）；超时后任务往往还在目标机上跑，改用直连兜底
APP_TIMEOUT = re.compile(r"超时|timeout|等待.{0,8}输出", re.I)
VIDEO_EXT = (".mp4", ".mov", ".webm", ".mkv")


def log(msg):
    with lock:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


class State:
    """result.json：results（已完成）、errors、pending（已提交未完成的运行号）、retakes（单段补救次数）。
    每次变化立即落盘。"""

    def __init__(self, path):
        self.path, self.lk = path, threading.Lock()
        self.d = {"results": {}, "errors": {}, "pending": {}, "retakes": {}}
        if os.path.isfile(path):
            self.d.update(json.load(open(path, encoding="utf-8")))
            for k in ("results", "errors", "pending", "retakes"):
                self.d.setdefault(k, {})

    def _save(self):
        json.dump(self.d, open(self.path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    def set(self, section, i, value):
        with self.lk:
            self.d[section][str(i)] = value
            self._save()

    def drop(self, section, i):
        with self.lk:
            self.d[section].pop(str(i), None)
            self._save()

    def get(self, section, i):
        return self.d[section].get(str(i))


def with_retry(fn, what, tries=3, base=2.0):
    last = None
    for k in range(tries):
        try:
            return fn()
        except PermissionError:
            raise
        except Exception as e:
            last = e
            if k < tries - 1:
                wait = base * (2 ** k)
                log("%s失败（%s），%.0f 秒后重试（%d/%d）" % (what, str(e)[:80], wait, k + 1, tries - 1))
                time.sleep(wait)
    raise RuntimeError("%s重试 %d 次仍失败：%s" % (what, tries, last))


def submit(i, args):
    """提交一次运行，返回 run_id；多次被拒则抛 RuntimeError。"""
    bad = C.validate_args(C.jmcai.WF_AVATAR, args)
    if bad:
        raise RuntimeError("[不可重试] 参数与桌面端暴露的 schema 不符：\n    " + "\n    ".join(bad))
    for _ in range(12):
        r = C.jmcai.run(C.jmcai.WF_AVATAR, args)
        rid = r.get("id")
        if rid:
            return rid
        msg = (r.get("message") or json.dumps(r.get("details") or {}, ensure_ascii=False))[:300]
        if NON_RETRYABLE.search(msg):
            raise RuntimeError("[不可重试] 提交被拒：" + msg)
        if (r.get("http") or 0) in (0, 429) or any(k in msg for k in RATE_LIMIT_WORDS):
            log("[段%d] 暂时不能提交（%s），20 秒后重试" % (i, msg[:80]))
            time.sleep(20)
            continue
        raise RuntimeError("提交失败：" + msg)
    raise RuntimeError("多次提交都被拒绝（并发/排队）")


def poll(i, run_id, timeout_s, interval=8, max_conn_errors=20):
    """返回 ("SUCCESS"|"FAILED"|"TIMEOUT"|"STALE", 运行记录)。连接错误累计超限抛 RuntimeError（运行号保留）。"""
    t0, last, conn = time.time(), None, 0
    while time.time() - t0 < timeout_s:
        r = C.jmcai.status(run_id)
        if not r.get("id"):                                    # 非 200：连接层错误，或运行号不存在
            http = r.get("http") or 0
            if http in (400, 404, 410):
                return "STALE", r
            conn += 1
            if conn > max_conn_errors:
                raise RuntimeError("连续 %d 次查不到运行 %s，它可能还在桌面端队列里，重跑本脚本会继续查它" % (conn, run_id))
            time.sleep(interval)
            continue
        conn = 0
        s = r.get("status")
        if s != last:
            log("[段%d] 状态 %s" % (i, s))
            last = s
        if s == "success":
            return "SUCCESS", r
        if s == "error":
            # 桌面端只是「等输出」等超时了，任务通常还在目标机上跑 → 交给直连兜底
            if APP_TIMEOUT.search(C.jmcai.error_text(r)):
                return "APP_TIMEOUT", r
            return "FAILED", r
        time.sleep(interval)
    return "TIMEOUT", None


def fetch_from_target(record, i, tmp_dir, timeout_s, interval=10):
    """桌面端超时后，拿运行记录里的 promptId 直接问算力目标要视频，落到临时文件并返回路径。"""
    pid = record.get("promptId") or record.get("prompt_id")
    base = C.jmcai.target_url(record.get("target_id"))
    if not pid or not base:
        raise RuntimeError("[已保留] 桌面端等输出超时，但拿不到 promptId 或目标地址，无法直接取结果；run %s" % record.get("id"))
    log("[段%d] 桌面端等超时了，任务还在 %s 上跑，改为直接查 prompt %s" % (i, base, pid[:8]))
    t0, seen = time.time(), None
    while time.time() - t0 < timeout_s:
        rec = C.jmcai.comfy_history(base, pid)
        if rec:
            st = (rec.get("status") or {}).get("status_str")
            if st != seen:
                log("[段%d] 目标状态 %s" % (i, st))
                seen = st
            if st == "error":
                msgs = json.dumps((rec.get("status") or {}).get("messages"), ensure_ascii=False)
                raise RuntimeError("[不可重试] 目标机上执行失败：" + msgs[-500:])
            if st == "success":
                outs = [o for o in C.jmcai.comfy_outputs(rec)
                        if os.path.splitext(o["filename"])[1].lower() in VIDEO_EXT]
                if not outs:
                    raise RuntimeError("[不可重试] 目标机执行完成但没有视频输出："
                                       + json.dumps(rec.get("outputs"), ensure_ascii=False)[:300])
                dst = os.path.join(tmp_dir, "seg_%02d.from_target%s" % (i, os.path.splitext(outs[0]["filename"])[1]))
                with_retry(lambda: C.jmcai.comfy_download(base, outs[0], dst), "[段%d] 从目标机下载视频" % i, tries=4)
                log("[段%d] 已从目标机取回 %s" % (i, outs[0]["filename"]))
                return dst
        time.sleep(interval)
    raise RuntimeError("[已保留] 直接查目标机也超时（%d 秒），prompt %s 可能还在跑，重跑本脚本会继续等它" % (timeout_s, pid))


def save_clip(i, src, rid, out_dir, state, extra=None):
    dst = os.path.join(out_dir, "seg_%02d.mp4" % i)
    with_retry(lambda: shutil.copy2(src, dst), "[段%d] 保存视频" % i, tries=4)
    log("[段%d] 已保存 %s" % (i, dst))
    res = {"i": i, "run_id": rid, "file": dst, "source": src}
    if extra:
        res.update(extra)
    state.set("results", i, res)
    state.drop("pending", i)
    state.drop("errors", i)
    return res


def archive_take(od, i, src_file, take_no=None):
    """重跑前把上一版片段收进 clips/takes/，旧版本不丢，随时能换回来。返回归档路径或 None。"""
    if not src_file or not os.path.isfile(src_file):
        return None
    dst_dir = os.path.join(od, "clips", "takes")
    os.makedirs(dst_dir, exist_ok=True)
    ext = os.path.splitext(src_file)[1] or ".mp4"
    tag = ("take%02d." % take_no if take_no else "") + time.strftime("%Y%m%d-%H%M%S")
    dst = os.path.join(dst_dir, "seg_%02d.%s%s" % (i, tag, ext))
    k = 2
    while os.path.exists(dst):                             # 同一秒内多次重跑也不覆盖
        dst = os.path.join(dst_dir, "seg_%02d.%s-%d%s" % (i, tag, k, ext))
        k += 1
    try:
        shutil.copy2(src_file, dst)
        log("[段%d] 上一版片段已归档 → %s" % (i, dst))
        return dst
    except Exception as e:                                 # 归档失败不该挡住重跑
        log("[段%d] 旧片段归档失败（继续重跑）：%s" % (i, str(e)[:120]))
        return None


def run_one(seg, prompt, img_path, ratio_value, seed, megapixels, out_dir, timeout_min, tries, state, interval=8, backoff=15.0):
    i = seg["i"]
    if seed is None:                                       # --seed random：每段换一个随机种子
        seed = random.randint(1, 2 ** 31 - 1)
        log("[段%d] --seed random，本段用随机种子 %d" % (i, seed))
    for attempt in range(1, tries + 2):
        pend = state.get("pending", i)
        rid = (pend or {}).get("run_id")
        if rid:
            log("[段%d] 续上上次已提交的运行 %s" % (i, rid))
        else:
            args = {"audio": os.path.abspath(seg["audio"]), "image": img_path, "prompt": prompt,
                    "aspect_ratio": ratio_value, "seed": seed}
            if megapixels is not None:
                args["megapixels"] = float(megapixels)
            rid = submit(i, args)
            state.set("pending", i, {"run_id": rid, "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S")})
            log("[段%d] 已提交 run %s（第 %d 次，seed=%s，megapixels=%s）" % (i, rid, attempt, seed, megapixels if megapixels is not None else "默认(1.5)"))
        outcome, q = poll(i, rid, timeout_min * 60, interval)
        if outcome == "APP_TIMEOUT":
            src = fetch_from_target(q, i, out_dir, timeout_min * 60, interval)
            extra_info = {"from_target": True, "seed": seed}
            if megapixels is not None:
                extra_info["megapixels"] = megapixels
            return save_clip(i, src, rid, out_dir, state, extra_info)
        if outcome == "SUCCESS":
            src = (C.jmcai.first_output(q, media_kind="video")
                   or C.jmcai.first_output(q, suffixes=VIDEO_EXT))
            if not src or not os.path.isfile(src):
                state.drop("pending", i)
                raise RuntimeError(
                    "[不可重试] 运行报成功但没拿到视频文件，run %s。\n"
                    "    最常见的原因：算力目标上这个工作流的某个节点校验没过（比如某个枚举值不在候选列表里），\n"
                    "    ComfyUI 会把输出整个丢掉、但桌面端仍算成功。请在桌面端看一下这段的运行详情，\n"
                    "    或到算力目标的 ComfyUI 日志里搜 “Failed to validate prompt”。" % rid)
            extra_info = {"duration_ms": q.get("duration_ms"), "seed": seed}
            if megapixels is not None:
                extra_info["megapixels"] = megapixels
            return save_clip(i, src, rid, out_dir, state, extra_info)
        if outcome == "TIMEOUT":
            raise RuntimeError("[已保留] 超时（%d 分钟），run %s 可能还在跑，重跑本脚本会继续等它，不会重复提交"
                               % (timeout_min, rid))
        if outcome == "STALE":
            log("[段%d] 旧运行 %s 查询不到，丢弃后重新提交" % (i, rid))
            state.drop("pending", i)
            continue
        # FAILED
        state.drop("pending", i)
        reason = C.jmcai.error_text(q) or json.dumps(
            {k: q.get(k) for k in ("message", "error", "detail") if q.get(k)}, ensure_ascii=False)[:300] or "运行失败"
        log("[段%d] 运行失败：%s" % (i, reason[:200]))
        if NON_RETRYABLE.search(reason):
            raise RuntimeError("[不可重试] " + reason)
        if attempt <= tries:
            log("[段%d] %.0f 秒后重新提交" % (i, backoff))
            time.sleep(backoff)
            continue
        raise RuntimeError("运行失败，已重试 %d 次：%s" % (tries, reason))
    raise RuntimeError("段%d 没有得到结果" % i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--only", type=int, nargs="*")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--timeout-min", type=int, default=45)
    ap.add_argument("--seed", default=str(C.DEFAULT_SEED),
                    help="随机种子；填 random 则每段各用一个随机种子（默认 %d）" % C.DEFAULT_SEED)
    ap.add_argument("--megapixels", type=float, default=None, help="控制分辨率百万像素（例如 1.0 用于快速抽卡，2.0 用于高清大图，默认工作流 1.5）")
    ap.add_argument("--force-retake", action="store_true",
                    help="突破单段补救上限（默认 %d 次）" % MAX_RETAKES)
    ap.add_argument("--keep-take", dest="keep_take", action="store_true", default=True,
                    help="重跑前把上一版片段收进 clips/takes/（默认开）")
    ap.add_argument("--no-keep-take", dest="keep_take", action="store_false",
                    help="重跑时直接覆盖，不归档旧片段")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if isinstance(a.seed, str) and a.seed.strip().lower() == "random":
        seed = None                                        # 交给 run_one 每段现取随机种子
    else:
        try:
            seed = int(a.seed)
        except (TypeError, ValueError):
            sys.exit("--seed 只能是整数或 random，收到：%r" % a.seed)
    od = os.path.join(a.workdir, "avatar")
    plan = json.load(open(os.path.join(od, "plan.json"), encoding="utf-8"))
    info = json.load(open(os.path.join(od, "image_info.json"), encoding="utf-8"))
    segs = plan["segments"]
    prompts = {}
    for s in segs:
        p = os.path.join(od, "prompts", "segment_%02d.txt" % s["i"])
        if not os.path.isfile(p):
            sys.exit("缺少提示词：%s（先运行 build_prompts.py）" % p)
        prompts[s["i"]] = open(p, encoding="utf-8").read()
    img_path = os.path.abspath(info["uploaded_image"])
    if not os.path.isfile(img_path):
        sys.exit("找不到底图：%s（先运行 prepare_image.py）" % img_path)
    ratio_value = C.aspect_ratio_value(info.get("ratio"))
    conc = min(C.concurrency(a.concurrency), len(segs))
    state = State(os.path.join(od, "result.json"))
    retakes = {int(k): int(v) for k, v in (state.d.get("retakes") or {}).items()}
    # 指定重跑：先判断哪些段已到补救上限、哪些本次要 +1（预览时只提示，绝不动数据）
    blocked, retake_plan = [], []                          # blocked=[(i,已重跑次数)]  retake_plan=[(i,本次是第几次)]
    if a.only is not None:
        for i in a.only:
            had = state.get("results", i)
            if had and os.path.isfile(had.get("file", "")):
                n = retakes.get(i, 0)
                if n >= MAX_RETAKES and not a.force_retake:
                    blocked.append((i, n))
                else:
                    retake_plan.append((i, n + 1))
    if blocked:
        for i, n in blocked:
            print("段%d 已经重跑过 %d 次（上限 %d）。同样的输入重跑通常还是同样的结果——"
                  "先换一样东西：加 --seed random 换种子，或改提示词后重跑 build_prompts.py。"
                  "确实要再试就加 --force-retake。" % (i, n, MAX_RETAKES))
        sys.exit(3)
    retake_ids = {i for i, _ in retake_plan}
    done = {int(k): v for k, v in state.d["results"].items()
            if os.path.isfile(v["file"]) and int(k) not in retake_ids}
    todo = [s for s in segs if s["i"] not in done and (a.only is None or s["i"] in a.only)]
    resumable = [s["i"] for s in todo if state.get("pending", s["i"])]
    print("画幅 %s（%s），百万像素 %s，尺寸按工作流自动，seed %s" % (
        info.get("ratio"), ratio_value,
        ("%.1f" % a.megapixels) if a.megapixels is not None else "默认(1.5)",
        "random（每段现取）" if seed is None else seed))
    print("共 %d 段，本次处理 %d 段，沿用已有 %d 段，并发 %d" % (len(segs), len(todo), len(done), conc))
    if resumable:
        print("其中段%s 有上次已提交、未完成的运行，会先续上它们，不会重复提交。" % "、".join(map(str, resumable)))
    if retake_plan:
        print("重跑段：%s（将计入单段补救次数，旧片段%s）" % (
            "、".join("段%d→第%d次" % (i, n) for i, n in retake_plan),
            "归档进 clips/takes/" if a.keep_take else "直接覆盖"))
    for s in todo:
        print("  段%d：%.1f 秒  提示词 %d 字  %s" % (s["i"], s["duration"], len(prompts[s["i"]]), os.path.basename(s["audio"])))
    if not a.yes:
        print("[预览] 还没有提交。确认后加 --yes。")
        return
    if not todo:
        print("没有需要提交的段。")
        return
    for i, n in retake_plan:                               # 真正执行时才动数据：归档旧片段、记补救次数、清旧运行号
        had = state.get("results", i)
        if a.keep_take and had:
            archive_take(od, i, had.get("file"), take_no=n)
        retakes[i] = n
        state.set("retakes", i, n)
        state.drop("pending", i)                           # 必须清：否则 run_one 会去续上一次的旧运行，而不是重新生成
        state.drop("errors", i)
        # 旧结果刻意保留：重跑失败时老片段还在，不会把工程搞成「既没新片也没旧片」
    C.require_bridge()
    C.avatar_workflow()
    os.makedirs(os.path.join(od, "clips"), exist_ok=True)
    errors = {}
    with ThreadPoolExecutor(max_workers=max(1, conc)) as ex:
        futs = {ex.submit(run_one, s, prompts[s["i"]], img_path, ratio_value, seed, a.megapixels,
                          os.path.join(od, "clips"), a.timeout_min, a.retries, state): s["i"] for s in todo}
        for f, i in futs.items():
            try:
                f.result()
            except PermissionError as e:
                print(str(e))
                sys.exit(4)
            except Exception as e:
                errors[i] = str(e)
                state.set("errors", i, str(e))
    results = {int(k): v for k, v in state.d["results"].items()}
    todo_ids = {s["i"] for s in todo}
    print("\n=== 汇总 ===")
    for i in sorted(results):
        if i in todo_ids and i not in errors:              # 只报本次真正处理的段，避免把沿用的旧结果也当成「刚成功」
            print("段%d：成功 用时=%sms" % (i, results[i].get("duration_ms")))
    for i in sorted(errors):
        print("段%d：失败 → %s" % (i, errors[i][:300]))
    if retakes:
        print("单段补救次数：" + "，".join("段%d %d/%d" % (k, v, MAX_RETAKES) for k, v in sorted(retakes.items())))
    if errors:
        print("有失败的段，未拼接。排查后重跑本脚本即可：已成功的段不会重复提交，仍在跑的运行会被续上；"
              "想强制重做某段用 --only <段号>。")
        sys.exit(1)


if __name__ == "__main__":
    main()
