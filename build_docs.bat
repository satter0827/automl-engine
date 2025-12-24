@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ============================================
REM AutoML Engine Sphinx API Build Script
REM ============================================

set DOCS_DIR=docs
set API_DIR=%DOCS_DIR%\api
set SRC_PKG_DIR=src\automl_engine
set BUILD_DIR=%DOCS_DIR%\_build\html

echo.
echo ============================================
echo  Sphinx API Documentation Build
echo ============================================

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    exit /b 1
)

python -m sphinx --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Sphinx is not installed in this environment.
    echo Please install requirements-docs.txt
    exit /b 1
)

REM --- Ensure conf.py exists where Sphinx expects it ---
if not exist "%DOCS_DIR%\conf.py" (
    echo [ERROR] conf.py not found: %DOCS_DIR%\conf.py
    echo Fix: place conf.py under "%DOCS_DIR%" or change DOCS_DIR to the correct conf directory.
    exit /b 1
)

REM --- clean api output to avoid stale modules.rst that keeps "src" at top ---
if exist "%API_DIR%" (
    rmdir /s /q "%API_DIR%"
)
mkdir "%API_DIR%"

echo Generating API docs from %SRC_PKG_DIR% ...
python -m sphinx.ext.apidoc -f -e -o "%API_DIR%" "%SRC_PKG_DIR%"
if errorlevel 1 (
    echo [ERROR] apidoc generation failed.
    exit /b 1
)

echo Building HTML documentation...
python -m sphinx -j auto -W --keep-going -b html "%DOCS_DIR%" "%BUILD_DIR%"
if errorlevel 1 (
    echo [ERROR] sphinx-build failed.
    exit /b 1
)

echo.
echo ============================================
echo  Build completed successfully
echo  Output: %BUILD_DIR%\index.html
echo ============================================

endlocal
