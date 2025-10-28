@echo off
echo ========================================
echo 5G Network Simulation Enhanced Dashboard
echo ========================================
echo.
echo Choose dashboard mode:
echo 1. Enhanced (with network topology map)
echo 2. Basic (original version)
echo.
set /p choice="Enter choice (1 or 2, default=1): "
if "%choice%"=="" set choice=1
if "%choice%"=="1" (
    echo Starting Enhanced Dashboard...
    python web_dashboard_enhanced.py
) else (
    echo Starting Basic Dashboard...
    python web_dashboard.py
)
pause

