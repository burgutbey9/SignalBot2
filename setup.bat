@echo off
echo Setting up SignalBot2...

REM .env yaratish
copy .env.example .env
echo .env fayli yaratildi!

REM Virtual environment
python -m venv venv
call venv\Scripts\activate

REM Paketlar o'rnatish
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Papkalar yaratish
mkdir logs data data\cache data\models data\backtest 2>nul

echo.
echo Setup tugadi! .env faylini tahrirlang:
echo notepad .env
echo.
echo Bot ishga tushirish: python src\main.py
pause