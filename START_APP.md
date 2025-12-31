# START_APP.md - How to Run the App

## The Problem

The app needs to run in the **LLA conda environment** where all packages are installed.
If you run `streamlit run app.py` directly, it uses the base Python environment which doesn't have the required packages.

## ✅ Solution 1: Use the Batch File (Recommended)

**Double-click:** `run_app.bat`

Or from PowerShell:

```powershell
.\run_app.bat
```

This automatically:

1. Activates the LLA conda environment
2. Shows which Python is being used
3. Starts the Streamlit app

## ✅ Solution 2: Manual Activation

Open PowerShell/Command Prompt and run:

```bash
conda activate LLA
streamlit run app.py
```

**Important:** You must see `(LLA)` in your terminal prompt before running streamlit!

## ✅ Solution 3: Use Anaconda Prompt

1. Open **Anaconda Prompt** (not regular PowerShell)
2. Navigate to project: `cd C:\Users\Dell\Desktop\Harsh\Projects\LLA`
3. Activate environment: `conda activate LLA`
4. Run app: `streamlit run app.py`

## Verify Your Environment

Before running, check which Python you're using:

```bash
conda activate LLA
python -c "import sys; print(sys.executable)"
```

Should show: `C:\Users\Dell\Anaconda3\envs\LLA\python.exe`

## Troubleshooting

**Error: "ModuleNotFoundError: No module named 'faiss'"**

- You're in the wrong environment
- Solution: Make sure you see `(LLA)` in your prompt

**Error: "CondaError: Run 'conda init'"**

- Conda not initialized in your shell
- Solution: Use Anaconda Prompt or run `conda init powershell` once

**Streamlit won't stop**

- Press `Ctrl+C` in the terminal
- Or run: `Stop-Process -Name streamlit -Force`

## Your App URL

Once started, open: **http://localhost:8501**
