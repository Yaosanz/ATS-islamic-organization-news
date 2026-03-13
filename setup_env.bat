@echo off
echo ================================================
echo   SETUP LINGKUNGAN SKRIPSI IndoBERT Summarizer
echo ================================================

REM Buat virtual environment
python -m venv venv
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
pip install -r requirements.txt

REM Install ipykernel ke dalam venv supaya muncul di VS Code
python -m ipykernel install --user --name=indobert_skripsi --display-name "Python (IndoBERT Skripsi)"

REM Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

REM Buat folder yang diperlukan
if not exist "models" mkdir models
if not exist "data\processed" mkdir data\processed
if not exist "data\raw" mkdir data\raw
if not exist "logs" mkdir logs
if not exist "results" mkdir results

echo.
echo ================================================
echo   ✅ Setup selesai!
echo.
echo   Langkah selanjutnya:
echo   1. Buka VS Code
echo   2. Buka notebooks\skripsi_indobert.ipynb
echo   3. Pilih kernel "Python (IndoBERT Skripsi)"
echo   4. Jalankan sel satu per satu
echo.
echo   Untuk menjalankan Streamlit app:
echo   streamlit run app\streamlit_app.py
echo ================================================
pause
