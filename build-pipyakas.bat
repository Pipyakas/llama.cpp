@echo off
setlocal EnableDelayedExpansion
REM build-pipyakas.bat - one host-optimized bin: exact CPU + NVIDIA GPU arch
REM GGML_CUDA=ON BACKEND_DL=ON CPU_ALL_VARIANTS=OFF NATIVE=OFF
REM Run from anywhere; detects the local CPU profile and CUDA compute capability.
REM Set CPU_PROFILE and/or GPU_ARCH before calling to override detection.
set "SRC=%~dp0"
if not exist "%SRC%CMakeLists.txt" if exist "%SRC%source-pipyakas\CMakeLists.txt" set "SRC=%SRC%source-pipyakas\"
if not exist "%SRC%CMakeLists.txt" if exist "D:\llama\source-pipyakas\CMakeLists.txt" set "SRC=D:\llama\source-pipyakas\"
if not exist "%SRC%CMakeLists.txt" if exist "C:\code\llama\source-pipyakas\CMakeLists.txt" set "SRC=C:\code\llama\source-pipyakas\"
if not exist "%SRC%CMakeLists.txt" (
  echo [build-pipyakas] no source found from %~dp0
  exit /b 1
)
REM normalize trailing slash
if "%SRC:~-1%"=="\" set "SRC=%SRC:~0,-1%"

REM bin dest per host
set "BIN=D:\llama\bin-pipyakas"
if not exist "D:\llama" set "BIN=C:\code\llama\bin-pipyakas"
if not exist "D:\llama" set "BIN=C:\code\llama\bin-pipyakas"

REM Detect the CPU profile used by this machine.
REM D1 has AVX512 + VBMI + VNNI512; L1 has AVX2; R2/R3 add AVX-VNNI256.
set "CPU_PROFILE=%CPU_PROFILE: =%"
set "GPU_ARCH=%GPU_ARCH: =%"
if not defined CPU_PROFILE (
  set "CPU_PROFILE=avx2"
  for /f "delims=" %%a in ('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Name" 2^>nul') do if not defined CPU_NAME set "CPU_NAME=%%a"
  if defined CPU_NAME echo !CPU_NAME! | findstr /i "Ryzen 7 7700 Ryzen 9 7900 Ryzen 9 7950 Ryzen 7 7800" >nul && set "CPU_PROFILE=avx512"
  if defined CPU_NAME echo !CPU_NAME! | findstr /i "5900 5800 5700 5600" >nul && set "CPU_PROFILE=avx2"
  if defined CPU_NAME echo !CPU_NAME! | findstr /i "14900 14700 14600 13900 13700 13600 12900 12700 12600" >nul && set "CPU_PROFILE=alderlake"
)
if /i "%CPU_PROFILE%"=="avx512" goto cpu_avx512
if /i "%CPU_PROFILE%"=="alderlake" goto cpu_alderlake
if /i "%CPU_PROFILE%"=="avx2" goto cpu_avx2
echo [build-pipyakas] invalid CPU_PROFILE=%CPU_PROFILE% (use avx2, alderlake, or avx512)
exit /b 1
:cpu_avx512
set "CPU_ARGS=-DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=OFF -DGGML_AVX2=ON -DGGML_AVX_VNNI=OFF -DGGML_BMI2=ON"
goto cpu_args_done
:cpu_alderlake
set "CPU_ARGS=-DGGML_AVX512=OFF -DGGML_AVX2=ON -DGGML_AVX_VNNI=ON -DGGML_BMI2=ON"
goto cpu_args_done
:cpu_avx2
set "CPU_ARGS=-DGGML_AVX512=OFF -DGGML_AVX2=ON -DGGML_AVX_VNNI=OFF -DGGML_BMI2=ON"
:cpu_args_done

REM Detect the first local NVIDIA compute capability, with override support.
if not defined GPU_ARCH (
  for /f "tokens=1,* delims=," %%a in ('nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2^>nul') do if not defined GPU_CC set "GPU_CC=%%a"
  set "GPU_CC=!GPU_CC: =!"
  if "!GPU_CC!"=="6.1" set "GPU_ARCH=61-real"
  if "!GPU_CC!"=="7.5" set "GPU_ARCH=75-real"
  if "!GPU_CC!"=="8.6" set "GPU_ARCH=86-real"
  if "!GPU_CC!"=="8.9" set "GPU_ARCH=89-real"
  if "!GPU_CC!"=="9.0" set "GPU_ARCH=90-real"
  if "!GPU_CC!"=="12.0" set "GPU_ARCH=120a-real"
)
if not defined GPU_ARCH (
  echo [build-pipyakas] no NVIDIA GPU detected; set GPU_ARCH explicitly
  exit /b 1
)
set "ARCHS=%GPU_ARCH%"
set "BUILD=%SRC%\build-host-%CPU_PROFILE%-%GPU_ARCH%"
set "TMPROOT=D:\llama\.tmp"
if not exist "D:\llama\.tmp" set "TMPROOT=C:\code\llama\.tmp"
if not exist "%TMPROOT%" mkdir "%TMPROOT%" 2>nul

set "CMAKE=C:\Program Files\CMake\bin\cmake.exe"
if not exist "%CMAKE%" set "CMAKE=cmake"

set "STATUS=%TMPROOT%\build-host-%CPU_PROFILE%-%GPU_ARCH%.status"
set "LOGCFG=%TMPROOT%\build-host-%CPU_PROFILE%-%GPU_ARCH%-config.log"
set "LOGBLD=%TMPROOT%\build-host-%CPU_PROFILE%-%GPU_ARCH%-build.log"

echo [build-pipyakas] %date% %time% SRC=%SRC% BIN=%BIN% CPU=%CPU_PROFILE% GPU=%GPU_ARCH% > "%STATUS%"
echo START_TIME=%date% %time%>> "%STATUS%"
echo CUDA_ARCH=%ARCHS% CPU_PROFILE=%CPU_PROFILE% GGML_NATIVE=OFF DL=ON CPU_ALL_VARIANTS=OFF>> "%STATUS%"
echo CPU_NAME=%CPU_NAME% GPU_CC=%GPU_CC%>> "%STATUS%"

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
"%CMAKE%" -S "%SRC%" -B "%BUILD%" -G "Visual Studio 17 2022" -T cuda=13.3 -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_NATIVE=OFF %CPU_ARGS% -DCMAKE_CUDA_ARCHITECTURES="%ARCHS%" > "%LOGCFG%" 2>&1
set CFG_EXIT=%ERRORLEVEL%
if not "%CFG_EXIT%"=="0" (
  echo [build-pipyakas] -T cuda=13.3 failed %CFG_EXIT%, retry without -T
  rmdir /s /q "%BUILD%" 2>nul
  timeout /t 1 >nul
  "%CMAKE%" -S "%SRC%" -B "%BUILD%" -G "Visual Studio 17 2022" -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_NATIVE=OFF %CPU_ARGS% -DCMAKE_CUDA_ARCHITECTURES="%ARCHS%" > "%LOGCFG%" 2>&1
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

REM Remove CPU backends from an older universal build.
for %%f in ("%BIN%\ggml-cpu-*.dll") do if exist "%%f" del /q "%%f" >nul 2>&1

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

echo BUILD_HOST_DONE %date% %time%>> "%STATUS%"
echo [build-pipyakas] DONE
exit /b 0
