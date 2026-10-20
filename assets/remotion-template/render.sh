#!/usr/bin/env bash
# 渲染全片（macOS / Linux / WSL 用这个；Windows 用同目录的 render.cmd）。
# 参数与 render.cmd 完全一致：GPU 浏览器渲染 + 多标签页并行 + 有 NVENC 就用显卡编码，失败退回 CPU。
#
# 用法：
#   cd <工程目录> && bash render.sh
#   素材或逐字时间改过？先跑 python scripts/packaging/remotion_new.py
set -u
cd "$(dirname "$0")" || exit 1

# 并行数取逻辑核心的一半，夹在 1..16
CORES=$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) )
CONC=$(( CORES / 2 ))
[ "$CONC" -lt 1 ] && CONC=1
[ "$CONC" -gt 16 ] && CONC=16

echo "并发 $CONC 路，优先显卡渲染 + 显卡编码"

# 和 render.cmd 同一个约定：remotion.config.ts 里写着「NVENC=1 时不要 CRF、改传码率」。
# 不设这个变量，配置会设 CRF，命令行再传 --video-bitrate 就会报
# "crf and videoBitrate can not both be set"。这是 render.sh 最容易踩的坑。
export NVENC=1

# 第一次尝试：硬件加速 + NVENC（在无独显的机器上会失败，然后退回 CPU）
if npx remotion render src/index.ts Main out/_raw.mp4 \
    --concurrency="$CONC" --hardware-acceleration=required \
    --video-bitrate=12M --audio-bitrate=192k; then
  :
else
  echo "显卡编码不可用，退回 CPU 编码（沿用 CRF 17，与显卡路径画质相当）..."
  unset NVENC
  if ! npx remotion render src/index.ts Main out/_raw.mp4 \
      --concurrency="$CONC" --audio-bitrate=192k; then
    echo "渲染失败" >&2
    exit 1
  fi
fi

[ -f out/_raw.mp4 ] || { echo "没有产出 out/_raw.mp4" >&2; exit 1; }

# 去掉容器元数据，不重编码：画面和声音一个字节都不动
if ! npx remotion ffmpeg -y -v error -i out/_raw.mp4 \
      -map_metadata -1 -map_chapters -1 -c copy -movflags +faststart out/main.mp4; then
  echo "封装失败" >&2
  exit 1
fi

rm -f out/_raw.mp4
echo "完成：out/main.mp4"
