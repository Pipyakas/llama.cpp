@echo off
setlocal EnableDelayedExpansion
REM build-pipyakas.bat — universal 1-bin: AVX2/AVX512 + Turing/Ampere/Blackwell
REM GGML_CUDA=ON BACKEND_DL=ON ALL_VARIANTS=ON NATIVE=OFF 75;80;86;89;90;120a(HOP)
REM Run from anywhere; auto-detects source-pipyakas + bin-pipyakas per host.
set "SRC=%~dp0"
if not exist "%SRC%CMakeLists.txt" (
  if exist "D:\llama\source-pipyakas\CMakeLists.txt" set "SRC=D:\llama\source-pipyakas"
)
if not exist "%SRC%CMakeLists.txt" (
  if exist "C:\code\llama\source-pipyakas\CMakeLists.txt" set "SRC=C:\code\llama\source-pipyakas"
)
if not exist "%SRC%CMakeLists.txt" (
  echo [build-pipyakas] no source found from %~dp0
  exit /b 1
)
REM normalize trailing slash
if "%SRC:~-1%"=="\" set "SRC=%SRC:~0,-1%"

REM bin dest per host
set "BIN=D:\llama\bin-pipyakas"
if not exist "D:\llama" set "BIN=C:\code\llama\bin-pipyakas"
if "%BIN%"=="D:\llama\bin-pipyakas" if not exist "D:\llama" set "BIN=C:\code\llama\bin-pipyakas"

set "BUILD=%SRC%\build-universal"
set "TMPROOT=D:\llama\.tmp"
if not exist "D:\llama\.tmp" set "TMPROOT=C:\code\llama\.tmp"
if not exist "%TMPROOT%" mkdir "%TMPROOT%" 2>nul

set "CMAKE=C:\Program Files\CMake\bin\cmake.exe"
if not exist "%CMAKE%" set "CMAKE=cmake"

set "ARCHS=75-real;80-real;86-real;89-real;90-real;120a-real"
set "STATUS=%TMPROOT%\build-universal.status"
set "LOGCFG=%TMPROOT%\build-universal-config.log"
set "LOGBLD=%TMPROOT%\build-universal-build.log"

echo [build-pipyakas] %date% %time% SRC=%SRC% BIN=%BIN% ARCHS=%ARCHS% > "%STATUS%"
echo CUDA_ARCH=%ARCHS% GGML_NATIVE=OFF DL=ON ALL_VARIANTS=ON>> "%STATUS%"

REM --- CUDA VS props fix (L1/R2 BuildTools missing CUDA 13.3) ---
set "SRC_PROPS=C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations"
set "DST_PROPS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations"
if exist "%SRC_PROPS%\CUDA 13.3.props" if exist "%DST_PROPS%" (
  if not exist "%DST_PROPS%\CUDA 13.3.props" (
    echo [build-pipyakas] copying CUDA 13.3 props to BuildTools
    copy /Y "%SRC_PROPS%\CUDA 13.3.props" "%DST_PROPS%\" >nul 2>&1
    copy /Y "%SRC_PROPS%\CUDA 13.3.targets" "%DST_PROPS%\" >nul 2>&1
    copy /Y "%SRC_PROPS%\CUDA 13.3.xml" "%DST_PROPS%\" >nul 2>&1
    copy /Y "%SRC_PROPS%\CUDA 13.3.Version.props" "%DST_PROPS%\" >nul 2>&1
    copy /Y "%SRC_PROPS%\Nvda.Build.CudaTasks.v13.3.dll" "%DST_PROPS%\" >nul 2>&1
  )
)

REM also set CUDAToolkit env for R2
if not defined CUDAToolkit_ROOT if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3" set "CUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
if not defined CUDA_PATH if defined CUDAToolkit_ROOT set "CUDA_PATH=%CUDAToolkit_ROOT%"

if exist "%BUILD%" (
  echo [build-pipyakas] clean %BUILD%
  rmdir /s /q "%BUILD%" 2>nul
  timeout /t 1 >nul
)

echo [build-pipyakas] cmake configure
echo CONFIG-START %time%>> "%STATUS%"
"%CMAKE%" -S "%SRC%" -B "%BUILD%" -G "Visual Studio 17 2022" -T cuda=13.3 -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES="%ARCHS%" > "%LOGCFG%" 2>&1
set CFG_EXIT=%ERRORLEVEL%
if not "%CFG_EXIT%"=="0" (
  echo [build-pipyakas] -T cuda=13.3 failed %CFG_EXIT%, retry without -T
  rmdir /s /q "%BUILD%" 2>nul
  timeout /t 1 >nul
  "%CMAKE%" -S "%SRC%" -B "%BUILD%" -G "Visual Studio 17 2022" -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES="%ARCHS%" > "%LOGCFG%" 2>&1
  set CFG_EXIT=%ERRORLEVEL%
)
echo CONFIG-EXIT=%CFG_EXIT% %time%>> "%STATUS%"
if not "%CFG_EXIT%"=="0" (
  echo [build-pipyakas] CONFIG FAILED %CFG_EXIT%
  type "%LOGCFG%"
  exit /b %CFG_EXIT%
)

echo [build-pipyakas] cmake --build Release -j 12
echo BUILD-START %time%>> "%STATUS%"
"%CMAKE%" --build "%BUILD%" --config Release -j 12 > "%LOGBLD%" 2>&1
set BLD_EXIT=%ERRORLEVEL%
echo BUILD-EXIT=%BLD_EXIT% %time%>> "%STATUS%"
if not "%BLD_EXIT%"=="0" (
  echo [build-pipyakas] BUILD FAILED %BLD_EXIT%
  exit /b %BLD_EXIT%
)

REM stage: prefer bin/Release, fallback bin, fallback Release
set "STAGE=%BUILD%\bin\Release"
if not exist "%STAGE%" set "STAGE=%BUILD%\bin"
if not exist "%STAGE%" set "STAGE=%BUILD%\Release"
echo [build-pipyakas] staging from %STAGE% to %BIN%
if not exist "%BIN%" mkdir "%BIN%" 2>nul
set STAGED=0
for %%f in ("%STAGE%\*.exe" "%STAGE%\*.dll") do (
  if exist "%%f" (
    copy /Y "%%f" "%BIN%\" >nul 2>&1
    set /a STAGED+=1
  )
)
REM also any ggml-*.dll buried in build tree (cpu variants sometimes elsewhere)
for /r "%BUILD%" %%f in (ggml-*.dll) do (
  if not exist "%BIN%\%%~nxf" (
    copy /Y "%%f" "%BIN%\" >nul 2>&1
    set /a STAGED+=1
    echo [build-pipyakas] extra %%~nxf
  ) else (
    REM overwrite if size differs
    for %%a in ("%%f") do for %%b in ("%BIN%\%%~nxf") do if not "%%~za"=="%%~zb" copy /Y "%%f" "%BIN%\" >nul 2>&1
  )
)
echo STAGED %STAGED% files to %BIN% %time%>> "%STATUS%"

dir /b "%BIN%\*.dll" 2>nul >> "%STATUS%"

where cuobjdump >nul 2>&1
if not errorlevel 1 (
  echo [build-pipyakas] cuobjdump fatbin check
  cuobjdump --list-elf "%BIN%\ggml-cuda.dll" 2>&1 | findstr /i "sm_" | sort | uniq > "%TMPROOT%\fatbin.txt" 2>nul
  for /f "tokens=*" %%a in ('cuobjdump --list-elf "%BIN%\ggml-cuda.dll" 2^>^&1 ^| findstr /i "sm_"') do echo %%a>> "%STATUS%"
)

"%BIN%\llama-server.exe" --help >nul 2>&1
if errorlevel 1 (
  echo [build-pipyakas] smoke FAILED
  echo smoke FAIL>> "%STATUS%"
  exit /b 1
) else (
  echo HELP OK>> "%STATUS%"
)

echo BUILD_UNIVERSAL_DONE %date% %time%>> "%STATUS%"
echo [build-pipyakas] DONE
exit /b 0
