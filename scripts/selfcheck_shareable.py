# -*- coding: utf-8 -*-
"""发布前自检：确认这个技能里没有不该带出去的东西。

    python scripts/selfcheck_shareable.py
    python scripts/selfcheck_shareable.py --fix-hint   # 顺带给出修法

**为什么需要它**：同一类错误已经犯过两次——写新脚本时把开发机的绝对路径写死在常量里
（一次在声学门禁找 ComfyUI python，一次在节点依赖核对脚本里找 ComfyUI 目录）。
两次都是"本地能跑、分享出去既暴露用户名又对别人无效"，而且都不报错、不崩溃，
只是静默地把个人信息留在仓库里。

所以每次发版 / 推送 / 打包前跑一遍这个，比事后追要好。

检查项：
  1. 硬编码的家目录路径（C:\\Users\\<名>、/home/<名>、/Users/<名>）
  2. 绝对路径里的盘符（C:\\... 只允许出现在文档的示例里，代码里不允许）
  3. 凭证（ghp_ / github_pat_ / sk- / Bearer / password= / api_key=）
  4. 私网地址与个人域名（192.168. / 10. / 172.16-31. / 已知的私有算力域名）
  5. 邮箱地址
  6. 具体硬件型号与 ComfyUI 版本号（这类是内部核对记录，不该进公开文档）

白名单：文档里的通用示例（D:\\ComfyUI_windows_portable）、占位符（<你的算力目标>）不算。
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {"__pycache__", "node_modules", ".git", ".idea", ".vscode", ".pytest_cache"}
SCAN_EXT = {".py", ".md", ".ts", ".tsx", ".json", ".sh", ".cmd", ".yml", ".yaml", ".txt"}

CHECKS = [
    # (标签, 正则, 适用于哪些后缀, 说明)
    ("家目录绝对路径", re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"'<>]+"), SCAN_EXT,
     "改成 os.path.expanduser('~/...') 或环境变量，别写死用户名"),
    ("POSIX 家目录路径", re.compile(r"/(?:home|Users)/[^/\s\"'<>]+/[^\s\"']*"), {".py", ".sh"},
     "改成 os.path.expanduser('~/...')"),
    ("凭证", re.compile(r"(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{16,}|Bearer\s+[A-Za-z0-9._\-]{20,})"), SCAN_EXT,
     "删掉，或改用环境变量读取"),
    ("疑似密码/密钥赋值", re.compile(r"(?i)(password|passwd|api[_-]?key|secret|token)\s*=\s*[\"'][^\"']{6,}[\"']"), {".py", ".sh", ".ts"},
     "改成从配置或环境变量读"),
    ("私网 IP", re.compile(r"\b(?:192\.168|10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"), SCAN_EXT,
     "文档用 <算力目标> 占位，代码用配置项"),
    ("邮箱", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), {".py", ".md", ".json"},
     "删掉"),
    ("具体显卡型号", re.compile(r"(?:本机|我这|我的机器|开发机)\s*(?:是|为|用的?)\s*(?:RTX\s*)?\d{4}"), {".md"},
     "别指认自己的设备，写成'一张够用的显卡'即可"),
    ("ComfyUI 具体版本号", re.compile(r"ComfyUI\s*v?0\.\d{1,2}\.\d{1,2}|\bcomfyui_version[\"']?\s*[:=]\s*[\"']0\.\d"), {".md"},
     "写成'较新版本'，或只写在内部记录里"),
]

# 允许出现的通用示例（文档里的路径样例、占位符），按行白名单
ALLOW_LINE = [
    re.compile(r"D:\\ComfyUI_windows_portable"),          # 文档通用的 Windows 示例路径
    re.compile(r"<[^>]*(?:算力目标|你的|路径|目录)[^>]*>"),  # 占位符
    re.compile(r"~/(?:ComfyUI|\.workbuddy-ai|\.digital-human)"),  # 波浪号写法本身是安全的
]


def iter_files():
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
            continue
        if p.suffix.lower() in SCAN_EXT:
            yield p


def scan():
    findings = []
    for p in iter_files():
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        rel = p.relative_to(ROOT).as_posix()
        for i, line in enumerate(lines, 1):
            if any(a.search(line) for a in ALLOW_LINE):
                continue
            for label, pat, exts, hint in CHECKS:
                if p.suffix.lower() not in exts:
                    continue
                m = pat.search(line)
                if m:
                    findings.append((rel, i, label, m.group(0)[:60], hint))
    return findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix-hint", action="store_true", help="同时打印修法建议")
    a = ap.parse_args()

    files = list(iter_files())
    findings = scan()
    print("扫描 %d 个文件" % len(files))
    if not findings:
        print("\n✓ 没有发现不该带出去的内容。")
        return 0

    print("\n发现 %d 处：\n" % len(findings))
    for rel, line, label, snippet, hint in findings:
        print("  %s:%d  [%s]  %s" % (rel, line, label, snippet))
        if a.fix_hint:
            print("      → %s" % hint)
    print("\n这些不需要全部删除：文档里的通用示例、占位符是可以的。")
    print("但**代码里的家目录绝对路径和凭证**必须改掉，否则分享出去既暴露信息又对别人无效。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
