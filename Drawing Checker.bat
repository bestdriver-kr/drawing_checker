@echo off
rem 이 배치파일이 있는 폴더로 이동 후 앱 실행(콘솔 없이)
cd /d "%~dp0"
start "" pythonw "%~dp0main.py"
