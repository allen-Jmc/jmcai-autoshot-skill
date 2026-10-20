"""开工前检查：JMCAI Comfypet 桌面端是否在线、数字人工作流是否已启用、算力目标是否可用。

  python check_setup.py                      # 检查（只读，不产生任何费用）
  python check_setup.py --set-concurrency 2  # 记录并行段数（本地单张 4090 默认 1 路串行）

本地模式没有 API Key、没有会员档位、没有计费——算力全部来自桌面端里已配好的工作流。
退出码：0 齐了；3 桌面端没连上或工作流没启用。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-concurrency", type=int)
    a = ap.parse_args()

    cfg = C.load_config()
    if a.set_concurrency:
        cfg["concurrency"] = max(1, int(a.set_concurrency))
        C.save_config(cfg)
        print("已保存配置：", json.dumps(cfg, ensure_ascii=False))
        return

    h = C.require_bridge()                                   # 不在线会打印引导并退出 3
    wf = C.avatar_workflow()                                 # 没启用会退出 3
    w = next((x for x in C.jmcai.registry() if x.get("id") == wf), {})
    targets = w.get("available_targets") or []
    usable = [t for t in targets if t.get("available", True)]
    print(json.dumps({"ok": True,
                      "bridge": C.jmcai.bridge_url(),
                      "desktop_app": h.get("desktop_app"),
                      "bridge_version": h.get("bridge_version"),
                      "workflow": wf,
                      "workflow_name": w.get("name"),
                      "targets": [t.get("name") for t in usable],
                      "concurrency": C.concurrency()}, ensure_ascii=False))
    if not usable:
        print("提示：这个工作流没有可用的算力目标。请在桌面端里把工作流绑定到本机 ComfyUI 或外部 4090 服务器。")
        sys.exit(3)


if __name__ == "__main__":
    main()
