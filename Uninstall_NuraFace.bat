@echo off
setlocal EnableExtensions

rem ============================================================
rem  NuraFace - deinstalator
rem  Umiesc ten plik w folderze z NuraFace.exe i uruchom.
rem
rem  Jesli dane aplikacji (DATA_DIR z resources\config.py) leza
rem  POZA folderem aplikacji, wpisz tu ich sciezke, np.:
rem  set "EXTRA_DIR1=C:\Users\Jan\AppData\Local\NuraFaceData"
rem ============================================================
set "EXTRA_DIR1="

set "NF_DIR=%~dp0"
if "%NF_DIR:~-1%"=="\" set "NF_DIR=%NF_DIR:~0,-1%"

if "%NF_DIR:~2%"=="" (
    echo Blad: skrypt nie moze byc uruchomiony z glownego katalogu dysku.
    pause
    exit /b 2
)
if not exist "%NF_DIR%\NuraFace.exe" (
    echo Blad: w tym folderze nie znaleziono NuraFace.exe.
    echo Umiesc Uninstall.bat w folderze z aplikacja NuraFace.
    pause
    exit /b 2
)

echo.
echo Zostanie TRWALE usuniete:
echo   - folder aplikacji: %NF_DIR%
echo     (razem z modelami, profilami, logami i ustawieniami)
echo   - wpis autostartu NuraFace (jesli wskazuje na ten folder)
echo   - dodatkowe dane w AppData\NuraFace (jesli istnieja)
echo.
choice /C TN /N /M "Kontynuowac? [T/N]: "
if errorlevel 2 exit /b 0

cd /d "%TEMP%"

rem --- 1. Zatrzymanie procesow uruchomionych z tego folderu ---
set "NF_ROOT=%NF_DIR%\"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$root=[IO.Path]::GetFullPath($env:NF_ROOT).TrimEnd('\')+'\'; $all=@(Get-CimInstance Win32_Process); $ids=@{}; foreach($p in $all){ if($p.ExecutablePath -and $p.ExecutablePath.StartsWith($root,[StringComparison]::OrdinalIgnoreCase)){ $ids[[string]$p.ProcessId]=$true } }; $c=$true; while($c){ $c=$false; foreach($p in $all){ if($ids.ContainsKey([string]$p.ParentProcessId) -and -not $ids.ContainsKey([string]$p.ProcessId)){ $ids[[string]$p.ProcessId]=$true; $c=$true } } }; foreach($p in $all){ if($ids.ContainsKey([string]$p.ProcessId)){ Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue } }"

rem --- 2. Usuniecie autostartu (tylko jesli wskazuje na ten folder) ---
set "RUNKEY=HKCU\Software\Microsoft\Windows\CurrentVersion\Run"
reg query "%RUNKEY%" /v NuraFace 2>nul | find /I "%NF_DIR%\NuraFace.exe" >nul
if not errorlevel 1 reg delete "%RUNKEY%" /v NuraFace /f >nul

rem --- 3. Skrypt sprzatajacy (usuwa folder juz po zamknieciu tego okna) ---
set "CLEAN=%TEMP%\nuraface_cleanup_%RANDOM%.bat"
>"%CLEAN%" echo @echo off
>>"%CLEAN%" echo ping 127.0.0.1 -n 4 ^>nul
>>"%CLEAN%" echo rmdir /s /q "%NF_DIR%"
>>"%CLEAN%" echo if exist "%APPDATA%\NuraFace" rmdir /s /q "%APPDATA%\NuraFace"
>>"%CLEAN%" echo if exist "%LOCALAPPDATA%\NuraFace" rmdir /s /q "%LOCALAPPDATA%\NuraFace"
if defined EXTRA_DIR1 >>"%CLEAN%" echo if exist "%EXTRA_DIR1%" rmdir /s /q "%EXTRA_DIR1%"
>>"%CLEAN%" echo if exist "%NF_DIR%" ^(
>>"%CLEAN%" echo   echo Nie udalo sie usunac folderu. Zamknij programy uzywajace plikow NuraFace i uruchom skrypt ponownie.
>>"%CLEAN%" echo   pause
>>"%CLEAN%" echo ^) else ^(
>>"%CLEAN%" echo   echo Gotowe - NuraFace zostal usuniety.
>>"%CLEAN%" echo   pause
>>"%CLEAN%" echo ^)
>>"%CLEAN%" echo ^(goto^) 2^>nul ^& del "%%~f0"

start "NuraFace - usuwanie" "%CLEAN%"
exit /b 0
