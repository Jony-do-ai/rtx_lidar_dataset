@echo off
chcp 65001
setlocal

:: ---------------------------------------------------------
:: 配置区域：请根据你的实际安装路径修改下方路径
:: ---------------------------------------------------------
:: 常见的 Isaac Sim 安装路径如下，如果你安装在 D 盘或其他位置，请修改：
set "ISAAC_SIM_PATH=D:\isaac-sim"

:: 如果你知道具体的路径（比如你上次提到的 D:\isaac_sim），请取消下面这行的注释并修改：
:: set "ISAAC_SIM_PATH=D:\isaac_sim"

:: ---------------------------------------------------------
:: 自动检测与执行
:: ---------------------------------------------------------
if not exist "%ISAAC_SIM_PATH%\python.bat" (
    echo [ERROR] 找不到 Isaac Sim，请检查脚本中的 ISAAC_SIM_PATH 路径配置。
    echo 当前配置路径: "%ISAAC_SIM_PATH%"
    pause
    exit /b
)

echo [INFO] 正在使用 Isaac Sim 环境运行 run.py ...
echo [INFO] Isaac Sim 路径: %ISAAC_SIM_PATH%

:: 调用 Isaac Sim 的 python.bat 来运行当前目录下的 run.py
call "%ISAAC_SIM_PATH%\python.bat" run.py

:: 运行结束后暂停，方便查看报错信息
pause