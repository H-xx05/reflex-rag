@echo off
rem ReflexRAG Web 服务一键启动: 双击运行, 关闭窗口即停止服务
chcp 65001 >nul
cd /d "%~dp0"
echo [ReflexRAG] 启动中... 浏览器访问 http://127.0.0.1:8000
".venv\Scripts\python.exe" -m uvicorn reflex_rag.api.main:app --host 127.0.0.1 --port 8000
echo.
echo [ReflexRAG] 服务已退出。若上方有报错, 请检查 .env 配置与索引是否已建立。
pause
