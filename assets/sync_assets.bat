@echo off
setlocal

:: 1. 这里的路径必须指向你的 Isaac Sim 安装目录下的 python.bat
:: 根据你之前提供的路径信息，应为 D:\isaac-sim\python.bat
set "ISAAC_PYTHON=D:\isaac-sim\python.bat"
set "SYNC_SCRIPT=D:\rtx_lidar_dataset\assets\sync_assets.py"

echo [CHECK] Testing Isaac Sim Python environment...

if not exist "%ISAAC_PYTHON%" (
    echo [ERROR] 找不到 python.bat，请检查路径: %ISAAC_PYTHON%
    pause
    exit /b 1
)

echo [RUN] 正在启动同步脚本...
:: 使用 call 确保执行完毕后返回
call "%ISAAC_PYTHON%" "%SYNC_SCRIPT%"

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 脚本执行失败，错误代码: %ERRORLEVEL%
) else (
    echo [SUCCESS] 同步任务已尝试完成。
)

pause