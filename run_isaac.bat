@echo off
chcp 65001
setlocal

:: ---------------------------------------------------------
:: 配置区域：请根据你的实际安装路径修改下方路径
:: ---------------------------------------------------------
:: 常见的 Isaac Sim 安装路径如下，如果你安装在 D 盘或其他位置，请修改：
set "ISAAC_SIM_PATH=D:\isaac-sim"

:: 设置要清理的输出路径
set "OUTPUT_PATH=D:\rtx_lidar_dataset\output\dataset"

:: ---------------------------------------------------------
:: 自动检测与执行
:: ---------------------------------------------------------
:: 清理之前生成的数据集
if exist "%OUTPUT_PATH%" (
    echo [INFO] 正在清理旧数据: "%OUTPUT_PATH%"
    :: /s 表示删除所有子目录和文件，/q 表示安静模式（不确认）
    del /s /q "%OUTPUT_PATH%\*.*" >nul 2>nul
    for /d %%x in ("%OUTPUT_PATH%\*") do rd /s /q "%%x" >nul 2>nul
    echo [INFO] 清理完成。
) else (
    echo [INFO] 输出文件夹不存在，无需清理。
)

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