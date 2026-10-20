# -*- coding: utf-8 -*-
"""run_avatar.py 的离线回归测试：--seed random、单段补救上限、旧片段归档、预览只读。

自包含：临时造一个最小工程目录，不需要真实项目、不需要桌面端在线。
把 C.require_bridge 打成哨兵异常，保证 --yes 也在「提交前」停住，不会真的提交。

  python tests/avatar/test_retake_seed.py
"""
import io
import json
import os
import shutil
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(SKILL, "scripts", "avatar"))
import run_avatar as R                                          # noqa: E402


class Sentinel(Exception):
    """到「该提交了」的位置就抛出来，代表逻辑已经跑完。"""


def build(workdir):
    od = os.path.join(workdir, "avatar")
    os.makedirs(os.path.join(od, "prompts"))
    os.makedirs(os.path.join(od, "clips"))
    img = os.path.join(od, "image.png")
    open(img, "wb").write(b"fake-image")
    segs, res = [], {"results": {}, "errors": {}, "pending": {}, "retakes": {}}
    for i in (1, 2, 3):
        wav = os.path.join(od, "seg_%02d.wav" % i)
        open(wav, "wb").write(b"fake-audio")
        segs.append({"i": i, "duration": 8.0, "audio": wav})
        open(os.path.join(od, "prompts", "segment_%02d.txt" % i), "w", encoding="utf-8").write("测试提示词")
        clip = os.path.join(od, "clips", "seg_%02d.mp4" % i)
        open(clip, "wb").write(b"fake-%d" % i)
        res["results"][str(i)] = {"i": i, "run_id": "r%d" % i, "file": clip, "source": clip, "duration_ms": 1000, "seed": 42}
    json.dump({"segments": segs, "total_seconds": 24.0}, open(os.path.join(od, "plan.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump({"uploaded_image": img, "ratio": "16:9"}, open(os.path.join(od, "image_info.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(res, open(os.path.join(od, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return od


def run(args):
    old, buf, so = sys.argv, io.StringIO(), sys.stdout
    sys.argv = ["run_avatar.py"] + args
    sys.stdout, code = buf, "ok"
    try:
        R.main()
    except SystemExit as e:
        code = "exit:%s" % e.code
    except Sentinel:
        code = "SENTINEL"
    except Exception as e:
        code = "EXC:%s: %s" % (type(e).__name__, str(e)[:160])
        traceback.print_exc()
    finally:
        sys.stdout, sys.argv = so, old
    return buf.getvalue(), code


def load(od):
    return json.load(open(os.path.join(od, "result.json"), encoding="utf-8"))


def takes(od):
    d = os.path.join(od, "clips", "takes")
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


def main():
    workdir = tempfile.mkdtemp(prefix="ra_test_")
    try:
        od = build(workdir)
        R.C.require_bridge = lambda: (_ for _ in ()).throw(Sentinel())
        R.C.avatar_workflow = lambda: None
        wd = ["--workdir", workdir]
        checks = []

        # 1. 预览不改数据；默认 seed 显示 42
        before = load(od)
        out, code = run(wd)
        checks.append(("预览不改 result.json", load(od) == before))
        checks.append(("预览显示 seed 42", "seed 42" in out))
        checks.append(("预览不产生归档", not takes(od)))

        # 2. --seed random 预览
        out, code = run(wd + ["--seed", "random"])
        checks.append(("--seed random 生效", "random（每段现取）" in out))
        checks.append(("random 预览仍不改数据", load(od) == before))

        # 3. --only 2 预览：提示第 1 次重跑，但不落盘
        out, code = run(wd + ["--only", "2"])
        checks.append(("预览提示段2→第1次", "重跑段：段2→第1次" in out))
        checks.append(("--only 预览不改数据", load(od) == before))

        # 4. 非法 seed
        out, code = run(wd + ["--seed", "abc"])
        checks.append(("非法 seed 报错退出", code.startswith("exit:") and "random" in code))

        # 5. 第 1 次真重跑：retakes=1、归档 1 个、旧结果保留
        out, code = run(wd + ["--yes", "--only", "2"])
        st = load(od)
        checks.append(("重跑停在提交前", code == "SENTINEL"))
        checks.append(("retakes[2]==1", st["retakes"].get("2") == 1))
        checks.append(("重跑保留旧结果", "2" in st["results"]))
        checks.append(("其余段结果保留", "1" in st["results"] and "3" in st["results"]))
        checks.append(("归档 1 个 seg_02", len(takes(od)) == 1 and takes(od)[0].startswith("seg_02.")))

        # 6. 第 2 次重跑：retakes=2、归档 2 个（不互相覆盖）
        out, code = run(wd + ["--yes", "--only", "2"])
        checks.append(("retakes[2]==2", load(od)["retakes"].get("2") == 2))
        checks.append(("两次归档不覆盖", len(takes(od)) == 2))

        # 7. 第 3 次：到上限被拦，退出码 3，数据不动
        snap, ntk = load(od), len(takes(od))
        out, code = run(wd + ["--yes", "--only", "2"])
        checks.append(("到上限退出码 3", code == "exit:3"))
        checks.append(("到上限有明确提示", "已经重跑过 2 次" in out and "--force-retake" in out))
        checks.append(("被拦时不改数据", load(od) == snap and len(takes(od)) == ntk))

        # 8. --force-retake 突破上限
        out, code = run(wd + ["--yes", "--only", "2", "--force-retake"])
        checks.append(("--force-retake 突破上限", load(od)["retakes"].get("2") == 3))

        # 9. --no-keep-take 不归档
        ntk = len(takes(od))
        out, code = run(wd + ["--yes", "--only", "2", "--force-retake", "--no-keep-take"])
        checks.append(("--no-keep-take 不新增归档", len(takes(od)) == ntk))
        checks.append(("--no-keep-take 提示直接覆盖", "直接覆盖" in out))
        checks.append(("retakes 继续累加", load(od)["retakes"].get("2") == 4))

        # 10. 没有旧结果的段重跑不算补救（例如上次失败后重来）
        st = load(od)
        st["results"].pop("2", None)
        json.dump(st, open(os.path.join(od, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        prev = st["retakes"].get("2")
        out, code = run(wd + ["--yes", "--only", "2"])
        checks.append(("无结果段重跑不算补救", "重跑段" not in out and load(od)["retakes"].get("2") == prev))

        bad = [n for n, ok in checks if not ok]
        for n, ok in checks:
            print("%s %s" % ("✓" if ok else "✗", n))
        print("\n%d/%d 通过" % (len(checks) - len(bad), len(checks)))
        if bad:
            print("失败：" + "；".join(bad))
            sys.exit(1)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
