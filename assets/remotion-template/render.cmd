@echo off
rem Render the full video: GPU browser rendering (ANGLE), parallel tabs, NVENC encoding.
rem Media or words changed? Run scripts/packaging/remotion_new.py on the project first.
cd /d "%~dp0"
set /a CONC=NUMBER_OF_PROCESSORS/2
if %CONC% LSS 1 set CONC=1
if %CONC% GTR 16 set CONC=16
set NVENC=1
call npx remotion render src/index.ts Main out/_raw.mp4 --concurrency=%CONC% --hardware-acceleration=required --video-bitrate=12M --audio-bitrate=192k
if errorlevel 1 (
  echo NVENC failed, falling back to CPU encoding...
  rem 退回 CPU 时不能再传码率：config 里 NVENC!=1 会设 CRF，两者同时给会报
  rem "crf and videoBitrate can not both be set"。CPU 走 CRF 17，画质与显卡路径相当。
  set NVENC=
  call npx remotion render src/index.ts Main out/_raw.mp4 --concurrency=%CONC% --audio-bitrate=192k
)
if not exist out\_raw.mp4 exit /b 1
rem Strip container metadata without re-encoding; picture and sound are untouched.
call npx remotion ffmpeg -y -v error -i out/_raw.mp4 -map_metadata -1 -map_chapters -1 -c copy -movflags +faststart out/main.mp4
if errorlevel 1 exit /b 1
del out\_raw.mp4
echo Done: out\main.mp4
