@echo off
title Simple Workflow Launcher
echo.
echo ========================================
echo     WINDSURF WORKFLOW LAUNCHER
echo ========================================
echo.
echo Choose your interface:
echo.
echo 1. Web Browser (Easiest)
echo 2. Desktop Panel
echo 3. Simple GUI
echo 4. Command Line
echo.
set /p choice="Enter number (1-4): "

if "%choice%"=="1" (
    echo.
    echo Starting Web Interface...
    start http://localhost:5000
    python ..\analysis_engine\windsurf_workflow_web.py --no-browser
    goto end
)

if "%choice%"=="2" (
    echo.
    echo Starting Desktop Panel...
    python ..\analysis_engine\windsurf_workflow_panel.py
    goto end
)

if "%choice%"=="3" (
    echo.
    echo Starting Simple GUI...
    python ..\analysis_engine\workflow_trigger_gui.py
    goto end
)

if "%choice%"=="4" (
    echo.
    echo Available workflows:
    python ..\analysis_engine\windsurf_trigger.py --list
    echo.
    set /p workflow="Enter workflow name: "
    python ..\analysis_engine\windsurf_trigger.py --workflow %workflow%
    goto end
)

echo Invalid choice. Please try again.
pause

:end
echo.
echo Done!
