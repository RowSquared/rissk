@echo off
REM RISSK GUI launcher for Windows
cd /d "%~dp0"

python -c "import nicegui" 2>nul || pip install "nicegui>=1.4"

echo.
echo Starting RISSK GUI...
echo Open your browser at: http://localhost:8080
echo (Press Ctrl+C to stop)
echo.
python app\main.py
