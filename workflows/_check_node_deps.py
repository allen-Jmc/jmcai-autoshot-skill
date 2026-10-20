# -*- coding: utf-8 -*-
"""核对工作流用到的节点分别由哪个包提供，并和 README 里的清单对账。

维护者工具。写节点依赖清单时**不要靠印象或仓库名猜**——实测踩过：社区里有几个名字
很像"这个工作流肯定需要它"的封装包，查下来发现它们提供的节点完全不同，一个都没用到；
还有几个"需要另装扩展包"的节点其实早已内置在 ComfyUI 本体里。两者都曾照着印象写进清单。

用法：
    python workflows/_check_node_deps.py                # 查每套工作流的外部依赖，并与 README 对账
    python workflows/_check_node_deps.py --print-table  # 直接打印可贴进 README 的表格
    python workflows/_check_node_deps.py --target-url https://<算力目标>:8188
                                                        # 对着**实际跑片子的那台机器**核对

**为什么一定要能查目标机**：开发机和跑片子的机器往往不是同一个 ComfyUI 版本，而内置节点
和外部节点的边界会随版本变。只有查目标机的 /object_info，才能确定"这份依赖清单在真实跑片子的
环境里成不成立"。

判定依据，从强到弱：
  1. ComfyUI 在跑 → 查 /object_info 的 python_module（最权威，能区分内置/外部）
  2. 扫 custom_nodes/ 下的 NODE_CLASS_MAPPINGS 键（定位到具体包名，只能查本机）
  3. 都没有 → 标为"未知"，不要猜
"""
import argparse
import json
import os
import pathlib
import re
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent

# 找本机 ComfyUI 安装目录（只用于扫 custom_nodes 定位包名）。
# 不能写死任何人的绝对路径——这里曾经硬编码过开发机的安装路径，
# 分享出去既暴露用户名、又对别人完全无效。做法：环境变量优先，再扫常见位置。
COMFYUI_CANDIDATES = [
    os.environ.get("COMFYUI_HOME"),
    os.path.expanduser("~/ComfyUI"),
    os.path.expanduser("~/ComfyUI_windows_portable"),
    os.path.expanduser("~/Pictures/ComfyUI_windows_portable-G313"),
    os.path.expanduser("~/Downloads/ComfyUI_windows_portable"),
    r"C:\ComfyUI_windows_portable",
    r"D:\ComfyUI_windows_portable",
    "/opt/ComfyUI",
]


def guess_comfyui():
    """返回第一个看起来像 ComfyUI 安装目录的路径，找不到就返回空串。"""
    import glob
    for root in COMFYUI_CANDIDATES:
        if not root:
            continue
        if (pathlib.Path(root) / "custom_nodes").is_dir():
            return root
        for hit in glob.glob(os.path.join(root, "*", "custom_nodes")):
            return str(pathlib.Path(hit).parent)
        for hit in glob.glob(os.path.join(root, "*", "*", "custom_nodes")):
            return str(pathlib.Path(hit).parent)
    return ""


DEFAULT_COMFYUI = guess_comfyui()


def workflows():
    for d in sorted(p for p in HERE.iterdir() if p.is_dir()):
        wf = d / "workflow.json"
        if wf.is_file():
            yield d.name, json.loads(wf.read_text(encoding="utf-8"))


def class_types(wf):
    return {v["class_type"] for v in wf.values() if isinstance(v, dict) and "class_type" in v}


def fetch_object_info(url_or_port, timeout=120):
    """从 ComfyUI 取 object_info。支持完整 URL（查远程算力目标）或端口号（查本机）。"""
    if isinstance(url_or_port, str) and url_or_port.startswith("http"):
        u = url_or_port.rstrip("/") + "/object_info"
    else:
        u = "http://127.0.0.1:%d/object_info" % int(url_or_port)
    try:
        return json.load(urllib.request.urlopen(u, timeout=timeout)), u
    except Exception as e:
        return None, "%s（%s）" % (u, type(e).__name__)


def fetch_version(url_or_port):
    """顺便报一下 ComfyUI 版本，便于确认两台机器是否一致。"""
    if isinstance(url_or_port, str) and url_or_port.startswith("http"):
        u = url_or_port.rstrip("/") + "/system_stats"
    else:
        u = "http://127.0.0.1:%d/system_stats" % int(url_or_port)
    try:
        d = json.load(urllib.request.urlopen(u, timeout=30))
        devs = ", ".join("%s %.0fGB" % (x.get("name", "?"), x.get("vram_total", 0) / 2 ** 30)
                         for x in d.get("devices", []))
        return "%s ｜ %s" % (d.get("system", {}).get("comfyui_version", "?"), devs or "?")
    except Exception:
        return "版本未知"


def try_object_info(port=8188):
    oi, _ = fetch_object_info(port)
    return oi


def scan_packages(comfy_root):
    """类名 → {包名}。只认 '类名': 这种映射键写法。"""
    cc = pathlib.Path(comfy_root) / "custom_nodes"
    if not cc.is_dir():
        return {}
    out = {}
    for pkg in cc.iterdir():
        if not pkg.is_dir() or pkg.name.startswith("."):
            continue
        for py in pkg.rglob("*.py"):
            if "__pycache__" in str(py):
                continue
            try:
                t = py.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for m in re.finditer(r'["\']([A-Za-z0-9_]{4,})["\']\s*:', t):
                out.setdefault(m.group(1), set()).add(pkg.name)
    return out


def classify(need, oi, fs):
    """返回 (外部依赖 {包: [节点]}, 内置 [节点], 未知 [节点])"""
    ext, builtin, unknown = {}, [], []
    for n in sorted(need):
        info = (oi or {}).get(n)
        mod = ""
        if info:
            mod = info.get("python_module") or info.get("module") or ""
        if mod and (mod == "nodes" or mod.startswith("comfy_extras")):
            builtin.append(n)
            continue
        owners = fs.get(n)
        if owners:
            for pkg in sorted(owners):
                ext.setdefault(pkg, []).append(n)
        elif info:
            unknown.append(n)          # ComfyUI 认得但没在 custom_nodes 找到，可能是站点包
        else:
            unknown.append(n)
    return ext, builtin, unknown


def doc_claims():
    """从 README 的表格里抽出「包 → 节点」。"""
    p = HERE / "README.md"
    if not p.is_file():
        return {}
    t = p.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"\|\s*\*\*([A-Za-z0-9_\-]+)\*\*\s*\|([^|]+)\|", t):
        nodes = re.findall(r"`([A-Za-z0-9_]+)`", m.group(2))
        if nodes:
            out[m.group(1)] = set(nodes)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfyui", default=os.environ.get("COMFYUI_HOME") or DEFAULT_COMFYUI,
                    help="本机 ComfyUI 安装目录（用于扫 custom_nodes）")
    ap.add_argument("--port", type=int, default=8188, help="本机 ComfyUI 端口")
    ap.add_argument("--target-url",
                    help="算力目标的 ComfyUI 地址，如 https://host:8188 —— 查它会同时覆盖 --port")
    ap.add_argument("--print-table", action="store_true", help="打印可贴进 README 的表格")
    a = ap.parse_args()

    src = a.target_url or a.port
    oi, where = fetch_object_info(src)
    fs = scan_packages(a.comfyui)
    print("数据源：%s %s" % (where, "✓" if oi else "✗ 连不上"))
    if oi:
        print("        ComfyUI %s，%d 个节点" % (fetch_version(src), len(oi)))
    print("        custom_nodes %s（本机扫到 %d 个类名，仅用于定位包名）"
          % ("✓" if fs else "✗ 没扫到", len(fs)))

    all_ext = {}
    for name, wf in workflows():
        need = class_types(wf)
        ext, builtin, unknown = classify(need, oi, fs)
        print("\n=== %s（%d 种节点）===" % (name, len(need)))
        print("  内置 %d 个，外部依赖 %d 个" % (len(builtin), sum(len(v) for v in ext.values())))
        for pkg, ns in sorted(ext.items()):
            print("  %-30s %s" % (pkg, ", ".join(ns)))
            all_ext.setdefault(pkg, set()).update(ns)
        if unknown:
            print("  ⚠ 未能定位（不要猜，需人工确认）：%s" % ", ".join(unknown))

    if a.print_table:
        print("\n=== 可贴进 README 的表格 ===\n")
        for pkg, ns in sorted(all_ext.items()):
            print("| **%s** | %s |  |" % (pkg, "、".join("`%s`" % n for n in sorted(ns))))

    # 与 README 对账
    claims = doc_claims()
    print("\n=== 与 README 清单对账 ===")
    problems = 0
    for pkg, ns in sorted(all_ext.items()):
        got = claims.get(pkg, set())
        miss = ns - got
        if miss:
            problems += 1
            print("  ✗ README 漏了 %s 的节点：%s" % (pkg, ", ".join(sorted(miss))))
    for pkg, ns in sorted(claims.items()):
        if pkg not in all_ext:
            problems += 1
            print("  ✗ README 列了 %s，但没有任何工作流用到它" % pkg)
        else:
            extra = ns - all_ext[pkg]
            if extra:
                problems += 1
                print("  ✗ README 说 %s 提供 %s，实际不是" % (pkg, ", ".join(sorted(extra))))
    print("  一致。" if not problems else "  共 %d 处不一致。" % problems)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
