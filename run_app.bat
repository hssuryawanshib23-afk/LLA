@echo off
echo Activating LLA conda environment...
call conda activate LLA
if errorlevel 1 (
    echo ERROR: Could not activate LLA environment
    pause
    exit /b 1
)

echo.
echo Python environment: 
python --version
echo.

echo Starting Streamlit app...
streamlit run app.py

pause
