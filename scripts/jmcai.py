"""JMCAI Comfypet 桌面端 Workflow Bridge 客户端：把"本机已配置好的 ComfyUI 工作流"当成本地算力用。

这是配音、数字人两个阶段共用的底座，用来**平替阿里百炼与 RunningHub**：
  - 配音：qwen3-tts-_api（声音克隆）/ qwen3-tts-_api-2（预设音色）/ qwen3-tts-voicedesign-_api（声音设计）
  - 识别：qwen3-asr-_api（字级时间戳）
  - 数字人：minimax-h3-copy-json

前置：用户机器上必须**已启动 JMCAI Comfypet 桌面应用**（它内置 Workflow Bridge，默认监听 127.0.0.1:32100）。
桌面端里要已经配好上面这些工作流，并绑定好算力目标（本机 ComfyUI 或外部 4090 服务器）。

Bridge 契约（1.2.0）：
  GET  /api/v1/health                              桌面端与 Bridge 状态
  GET  /api/v1/workflows                           已启用工作流 + 暴露的 alias schema
  POST /api/v1/workflows/<id>/runs                 提交运行 {"args": {...}, "target_id": "..."}
  GET  /api/v1/runs/<run_id>                       查询单次运行（含 outputs）
  GET  /api/v1/workflows/<id>/history              历史运行
  POST /api/v1/uploads                             远程 bridge 场景上传本机资产

要点（都是实测确认的）：
  * **本机 bridge（127.0.0.1）时，image / audio / video / file 类参数直接传本机绝对路径**，
    桌面端自己负责把文件送到算力目标（本机 ComfyUI 或外部 4090），不需要我们上传、也不要用 upload: 前缀。
  * 运行状态：queued → running → success / error。终态是 success 与 error。
  * outputs[].path 在 loopback 场景下已经是**本机绝对路径**（桌面端已自动从算力目标下载回来），
    直接读即可。**但桌面端偶尔不拉**（status=success、远端跑完了、outputs 却是空数组）——
    outputs_of() / first_output() 已内置兜底：拿 promptId 直接去算力目标 /history + /view 取，
    落在系统临时目录 jmcai_fetch/<run_id>/，调用方无感。

环境变量：
  JMCAI_BRIDGE_URL      Bridge 地址，默认 http://127.0.0.1:32100
  JMCAI_TIMEOUT         单次 HTTP 超时（秒），默认 60
  JMCAI_POLL_INTERVAL   轮询间隔（秒），默认 5
  JMCAI_POLL_TIMEOUT    单次运行最长等待（秒），默认 3600
"""
import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_BRIDGE = "http://127.0.0.1:32100"
# 从 comfypet-jmcai-skill 的配置里兜底读一次 bridge 地址（用户改过端口时能自动跟上）
SKILL_CONFIG = os.path.join(os.path.expanduser("~"), ".workbuddy-ai", "skills", "comfypet-jmcai-skill", "config.json")

TERMINAL = ("success", "error")

# 各工作流的固定 id（桌面端里配好的）
WF_TTS_CLONE = "qwen3-tts-_api"              # 声音克隆：参考音频 + 台词
WF_TTS_PRESET = "qwen3-tts-_api-2"           # 预设音色：台词 + speaker + instruct
WF_TTS_DESIGN = "qwen3-tts-voicedesign-_api"  # 声音设计：台词 + instruct（自然语言描述）
WF_ASR = "qwen3-asr-_api"                    # 字级时间戳
WF_AVATAR = "minimax-h3-copy-json"           # 数字人：底图 + 音频 + 提示词


def bridge_url():
    u = os.environ.get("JMCAI_BRIDGE_URL", "").strip()
    if u:
        return u.rstrip("/")
    if os.path.isfile(SKILL_CONFIG):
        try:
            v = json.load(open(SKILL_CONFIG, encoding="utf-8")).get("bridge_url", "").strip()
            if v:
                return v.rstrip("/")
        except (OSError, ValueError):
            pass
    return DEFAULT_BRIDGE


def _is_loopback(url):
    host = urllib.parse.urlparse(url).hostname or ""
    return host in ("127.0.0.1", "localhost", "::1")


def is_local_bridge():
    return _is_loopback(bridge_url())


def _timeout():
    return float(os.environ.get("JMCAI_TIMEOUT", "60"))


# ---------------------------------------------------------------- HTTP
def _request_json(method, path, payload=None, timeout=None, tries=2):
    """返回 (status, data)。status=0 表示连接层失败（桌面端没启动/端口不通）。"""
    url = bridge_url() + path
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    last = None
    for k in range(tries):
        req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
        req.get_method = lambda: method
        try:
            with urllib.request.urlopen(req, timeout=timeout or _timeout()) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            try:
                return e.code, json.loads(raw)
            except ValueError:
                return e.code, {"message": raw[:400]}
        except Exception as e:                       # 断连、超时
            last = e
            if k < tries - 1:
                time.sleep(1.5)
    return 0, {"message": "%s: %s" % (type(last).__name__, last)}


# ---------------------------------------------------------------- 基础查询
def health():
    """桌面端 / Bridge 状态。连不上时返回 {'status': 'error', ...}。"""
    st, data = _request_json("GET", "/api/v1/health")
    if st != 200:
        return {"status": "error", "http": st, "message": data.get("message", ""), "bridge_url": bridge_url()}
    return {"status": "success", **data, "bridge_url": bridge_url()}


def require_bridge():
    """确认桌面端在线。不在线就打印引导并退出（退出码 3）。"""
    h = health()
    if h.get("status") != "success":
        print("本机没有连上 JMCAI Comfypet 桌面端（%s）。\n"
              "请先打开 JMCAI Comfypet 桌面应用并确认里面的 Workflow Bridge 已启动，然后重试。\n"
              "（端口不是默认的 32100 时，设置环境变量 JMCAI_BRIDGE_URL 指过去。）" % bridge_url())
        sys.exit(3)
    return h


def registry():
    st, data = _request_json("GET", "/api/v1/workflows")
    if st != 200:
        return []
    return data.get("workflows", [])


def workflow_schema(workflow_id):
    """取某个工作流暴露的 alias 参数表；找不到返回 None。"""
    for w in registry():
        if w.get("id") == workflow_id:
            return w.get("schema", [])
    return None


def pick_workflow(ids):
    """在候选 id 里挑第一个桌面端确实启用了的；都没有返回 None。"""
    have = {w.get("id") for w in registry()}
    for i in ids:
        if i in have:
            return i
    return None


# ---------------------------------------------------------------- 运行
def run(workflow_id, args, target_id=None):
    """提交一次运行。本机 bridge 时 args 里的资产参数直接传本机绝对路径。"""
    payload = {"args": args}
    if target_id:
        payload["target_id"] = target_id
    st, data = _request_json("POST", "/api/v1/workflows/%s/runs" % urllib.parse.quote(workflow_id), payload)
    # 提交成功是 202 Accepted（不是 200），2xx 一律算成功
    if not (200 <= st < 300):
        return {"status": "error", "http": st, "message": data.get("message", ""), "details": data}
    return data


def status(run_id):
    st, data = _request_json("GET", "/api/v1/runs/%s" % urllib.parse.quote(run_id))
    if st != 200:
        return {"status": "error", "http": st, "message": data.get("message", "")}
    return data


def wait(run_id, timeout_s=None, interval_s=None, quiet=False, label=""):
    """轮询到终态。返回最后一次的运行记录（含 outputs）。超时返回 status='timeout'。"""
    timeout_s = float(timeout_s or os.environ.get("JMCAI_POLL_TIMEOUT", "3600"))
    interval_s = float(interval_s or os.environ.get("JMCAI_POLL_INTERVAL", "5"))
    t0, last, shown = time.time(), None, None
    while time.time() - t0 < timeout_s:
        r = status(run_id)
        s = r.get("status")
        if s != shown:
            if not quiet:
                print("[%s] 状态 %s" % (label or run_id[:8], s), flush=True)
            shown = s
        last = r
        if s in TERMINAL:
            return r
        if s == "error":
            return r
        time.sleep(interval_s)
    if last:
        last = dict(last, status="timeout")
        return last
    return {"status": "timeout", "id": run_id}


def run_and_wait(workflow_id, args, target_id=None, timeout_s=None, interval_s=None, quiet=False, label=""):
    """提交并等到完成，一步到位。"""
    r = run(workflow_id, args, target_id)
    if r.get("status") == "error" or not r.get("id"):
        return r
    if not quiet:
        print("[%s] 已提交 run %s" % (label or workflow_id, r["id"]), flush=True)
    return wait(r["id"], timeout_s, interval_s, quiet=quiet, label=label)


# ---------------------------------------------------------------- 输出
def outputs_of(run_record, fetch=True):
    """规范化 outputs：返回 [{'path','media_kind','file_name'}]，只保留 path 存在的项。

    实测（2026-10-19）：桌面端有时**不把远端产物拉回来**——运行记录 status=success、
    远端 ComfyUI 也真的跑完了，但 outputs 是空数组、outputs/<工作流>/ 下没有任何新文件。
    这时（且只有这时）用运行记录里的 promptId 直接问算力目标要，见 outputs_from_target()。
    """
    out = []
    for o in run_record.get("outputs", []) or []:
        if isinstance(o, str):
            o = {"path": o, "media_kind": None, "file_name": os.path.basename(o)}
        p = o.get("path") or ""
        if not p:
            continue
        out.append({"path": p, "media_kind": o.get("media_kind"), "file_name": o.get("file_name") or os.path.basename(p),
                    "mime_type": o.get("mime_type")})
    if out or not fetch:
        return out
    return outputs_from_target(run_record)


def first_output(run_record, media_kind=None, suffixes=None, fetch=True):
    """按 media_kind 或后缀挑第一个输出，返回本机绝对路径；没有返回 None。"""
    for o in outputs_of(run_record, fetch=fetch):
        if media_kind and o["media_kind"] != media_kind:
            continue
        if suffixes and os.path.splitext(o["path"])[1].lower() not in suffixes:
            continue
        return o["path"]
    return None


def error_text(run_record):
    return (run_record.get("error_message") or "").strip()


# ---------------------------------------------------------------- 直接问算力目标（桌面端等不及时的兜底）
# 桌面端等远端输出有上限（实测约 180 秒），而数字人这种重任务常常要 4–5 分钟。
# 超时后**任务通常还在目标机上跑**，桌面端只是不等了；这时用运行记录里的 promptId
# 直接问目标 ComfyUI 要结果，就不用白等一轮。
def _appdata():
    """桌面端数据目录（Windows 的 %APPDATA%）。环境变量可能没导出，所以逐级兜底。"""
    for k in ("JMCAI_DATA_DIR", "APPDATA"):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    if os.name == "nt":
        p = os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
        if os.path.isdir(p):
            return p
    return os.path.join(os.path.expanduser("~"), ".config")


TARGETS_FILE = os.path.join(_appdata(), "JMCAI", "workflows", "targets.json")


def targets():
    """桌面端配置的算力目标（含 url）。"""
    if os.path.isfile(TARGETS_FILE):
        try:
            return json.load(open(TARGETS_FILE, encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return []


def target_url(target_id):
    for t in targets():
        if t.get("id") == target_id:
            return (t.get("url") or "").rstrip("/")
    return ""


def _comfy_json(base, path, timeout=30):
    req = urllib.request.Request(base + path)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def comfy_history(base, prompt_id):
    """目标 ComfyUI 上某个 prompt 的执行记录；没有就返回 None。"""
    try:
        d = _comfy_json(base, "/history/%s" % urllib.parse.quote(prompt_id))
    except Exception:
        return None
    return d.get(prompt_id)


def comfy_outputs(record):
    """从 ComfyUI 执行记录里挑出可下载的文件，返回 [{'filename','subfolder','type'}]。

    ComfyUI 把视频/图片放在 images / gifs / audio 等键下，形状一致。
    """
    out = []
    for _node, o in (record.get("outputs") or {}).items():
        if not isinstance(o, dict):
            continue
        for key in ("gifs", "videos", "images", "audio", "files"):
            for it in o.get(key) or []:
                if isinstance(it, dict) and it.get("filename"):
                    out.append({"filename": it["filename"], "subfolder": it.get("subfolder", ""),
                                "type": it.get("type", "output")})
    return out


def comfy_download(base, info, dest, timeout=600):
    """按 ComfyUI 的 /view 下载一个输出文件。"""
    q = urllib.parse.urlencode({"filename": info["filename"], "subfolder": info.get("subfolder", ""),
                                "type": info.get("type", "output")})
    req = urllib.request.Request("%s/view?%s" % (base, q))
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dest


KIND_BY_EXT = {".wav": "audio", ".mp3": "audio", ".flac": "audio", ".m4a": "audio", ".ogg": "audio",
               ".mp4": "video", ".webm": "video", ".mov": "video", ".mkv": "video",
               ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
               ".txt": "text", ".json": "text"}


def _guess_kind(path):
    return KIND_BY_EXT.get(os.path.splitext(path)[1].lower())


def _pick(rec, *names):
    """运行记录里同一个字段有两种写法：Bridge API 返回 snake_case（prompt_id），
    落盘的 runs/*.json 是 camelCase（promptId）。两个都认。"""
    for n in names:
        v = rec.get(n)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _fetch_dir(run_record):
    rid = _pick(run_record, "id") or _pick(run_record, "prompt_id", "promptId") or "run"
    base = os.environ.get("JMCAI_FETCH_DIR", "").strip() or os.path.join(tempfile.gettempdir(), "jmcai_fetch")
    return os.path.join(base, rid)


def outputs_from_target(run_record, dest_dir=None, tries=3):
    """桌面端没把远端产物拉回来时的兜底：拿 promptId 直接问算力目标，下载到本机临时目录。

    只读远端 /history 与 /view，不会重跑任务；文件已存在就跳过下载，可以反复调用。
    """
    pid = _pick(run_record, "prompt_id", "promptId")
    if not pid:
        return []
    base = target_url(_pick(run_record, "target_id", "targetId"))
    if not base:
        ts = [t for t in targets() if t.get("enabled", True)]
        if len(ts) == 1:
            base = (ts[0].get("url") or "").rstrip("/")
    if not base:
        return []
    rec = None
    for k in range(max(1, tries)):                  # 刚跑完时 history 可能还没落盘
        rec = comfy_history(base, pid)
        if rec:
            break
        time.sleep(1.5)
    if not rec:
        return []
    infos = comfy_outputs(rec)
    infos.sort(key=lambda x: 0 if x.get("type") == "output" else 1)   # 正式产物优先于 temp
    dest_dir = dest_dir or _fetch_dir(run_record)
    try:
        os.makedirs(dest_dir, exist_ok=True)
    except OSError:
        return []
    out = []
    for info in infos:
        dest = os.path.join(dest_dir, os.path.basename(info["filename"]))
        if not (os.path.isfile(dest) and os.path.getsize(dest) > 0):
            try:
                comfy_download(base, info, dest)
            except Exception:
                continue
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            out.append({"path": dest, "media_kind": _guess_kind(dest),
                        "file_name": info["filename"], "mime_type": None})
    return out


# ---------------------------------------------------------------- 命令行（调试用）
def _cmd_doctor(_a=None):
    print(json.dumps(health(), ensure_ascii=False, indent=1))


def _cmd_registry(_a=None):
    for w in registry():
        print("%-28s %-40s %s -> %s" % (w.get("id"), w.get("name"), w.get("input_modalities"), w.get("output_modalities")))


def _cmd_run(a):
    args = json.loads(a.args) if a.args else {}
    r = run(a.workflow, args, a.target)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    if a.wait and r.get("id"):
        print(json.dumps(wait(r["id"], a.timeout), ensure_ascii=False, indent=1))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="JMCAI Bridge 客户端")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("doctor")
    sp.add_parser("registry")
    r = sp.add_parser("run")
    r.add_argument("--workflow", required=True)
    r.add_argument("--args")
    r.add_argument("--target")
    r.add_argument("--wait", action="store_true")
    r.add_argument("--timeout", type=float)
    a = ap.parse_args()
    {"doctor": _cmd_doctor, "registry": _cmd_registry, "run": _cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
