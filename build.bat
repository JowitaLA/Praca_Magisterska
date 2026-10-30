@echo off
REM Budowanie NuraFace.exe. Polozyc w glownym folderze projektu (obok main.py) i uruchomic.
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

echo === PyInstaller ===
pyinstaller --clean --noconfirm --onedir --noconsole --name NuraFace --icon=resources\app.ico --collect-all mediapipe --hidden-import extraction.extraction --hidden-import modeling.training main.py
if errorlevel 1 goto fail

set OUT=dist\NuraFace

echo === Kopiowanie folderow z zasobami ===
REM weights = modele (yolo_face.pt, arcface.onnx) - bez nich ekstrakcja nie dziala.
REM Pomijamy dane osobowe z maszyny deweloperskiej: database, modeling\outputs, logs, settings.ini
robocopy resources    %OUT%\resources    /E /XD __pycache__ >nul
robocopy extraction   %OUT%\extraction   /E /XD __pycache__ outputs >nul
robocopy modeling     %OUT%\modeling     /E /XD __pycache__ outputs >nul
robocopy verification %OUT%\verification /E /XD __pycache__ >nul
robocopy weights      %OUT%\weights      /E >nul

if not exist %OUT%\database        mkdir %OUT%\database
if not exist %OUT%\modeling\outputs mkdir %OUT%\modeling\outputs
if not exist %OUT%\logs            mkdir %OUT%\logs

copy Uninstall_NuraFace.bat %OUT%\Uninstall_NuraFace.bat >nul

REM Zbior impostorow (FairFace) potrzebny do wyznaczenia progu FAR/FRR.
REM Zawsze kopiujemy gotowe pliki csv z test_database; folder test_database\profile nie jest kopiowany.
if exist test_database\ff_train.csv (
    copy /y test_database\ff_train.csv %OUT%\database\ >nul
) else (
    echo UWAGA: brak test_database\ff_train.csv
)
if exist test_database\ff_test.csv (
    copy /y test_database\ff_test.csv %OUT%\database\ >nul
) else (
    echo UWAGA: brak test_database\ff_test.csv
)

echo.
echo Okrojona wersja FairFace (test_database\FairFace): rozmiar danych 4,64 GB, na dysku 5,37 GB.
choice /c TN /m "Skopiowac folder FairFace do dist (T = tak, N = nie)"
if errorlevel 2 goto skipff

if not exist test_database\FairFace (
    echo UWAGA: brak folderu test_database\FairFace - pomijam.
    goto skipff
)
echo Kopiowanie FairFace, to moze potrwac kilka minut...
robocopy test_database\FairFace %OUT%\database\FairFace /E /NFL /NDL
:skipff

echo.
echo Gotowe: %OUT%\NuraFace.exe
pause
exit /b 0

:fail
echo.
echo Budowanie nie powiodlo sie.
pause
exit /b 1