@echo off
rem 이 배치파일이 있는 폴더로 이동 후 앱 실행(콘솔 없이)
rem %~1: 더블클릭/파일 연결로 넘어온 파일 경로(있으면 함께 열기)
cd /d "%~dp0"
start "" pythonw "%~dp0main.py" "%~1"
