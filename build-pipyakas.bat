@echo off
setlocal EnableDelayedExpansion
REM build-pipyakas.bat - universal bin-pipyakas, Zen4 AVX512 fixed, composable GPU backends
REM Usage: build-pipyakas.bat [cuda] [vulkan] [rocm]
REM   cuda   = CUDA sm_75 (RTX 2060), VS2026 generator
REM   vulkan = Vulkan (6700 XT), needs C:\VulkanSDK\1.4.350.0
REM   rocm   = HIP gfx1031 (6700 XT), needs rocm\.rocm-venv SDK (VS2022 BuildTools legacy)
REM Examples:
REM   build-pipyakas.bat cuda           CUDA-only (~8 min)
REM   build-pipyakas.bat cuda vulkan    CUDA + Vulkan (~10 min)
REM   build-pipyakas.bat cuda rocm      CUDA + ROCm HIP (~15 min, clang toolchain)
REM Env overrides: GPU_ARCH, VULKAN_SDK, ROCM_SDK
REM Always stages into C:\code\llama\bin-pipyakas (universal: keeps backends from prior builds).

set "WANT_CUDA=0"
set "WANT_VULKAN=0"
set "WANT_ROCM=0"
for %%a in (%*) do (
  if /i "%%a"=="cuda" set "WANT_CUDA=1"
  if /i "%%a"=="vulkan" set "WANT_VULKAN=1"
  if /i "%%a"=="rocm" set "WANT_ROCM=1"
  if /i not "%%a"=="cuda" if /i not "%%a"=="vulkan" if /i not "%%a"=="rocm" (
    echo [build-pipyakas] usage: build-pipyakas.bat [cuda] [vulkan] [rocm]
    exit /b 1
  )
)
if "%WANT_CUDA%%WANT_VULKAN%%WANT_ROCM%"=="000" (
  echo [build-pipyakas] usage: build-pipyakas.bat [cuda] [vulkan] [rocm]
  exit /b 1
)

set "SRC=%~dp0source-pipyakas"
if not exist "%SRC%\CMakeLists.txt" (
  echo [build-pipyakas] no source at %SRC%
  exit /b 1
)
set "BIN=%~dp0bin-pipyakas"
set "TMPROOT=%~dp0.tmp"
if not exist "%TMPROOT%" mkdir "%TMPROOT%" 2>nul
set "CMAKE=C:\Program Files\CMake\bin\cmake.exe"
if not exist "%CMAKE%" set "CMAKE=cmake"

REM --- Zen4 fixed ---
set "CPU_ARGS=-DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=OFF -DGGML_AVX2=ON -DGGML_AVX_VNNI=OFF -DGGML_BMI2=ON"
set "CPU_PROFILE=avx512"

REM --- GPU arch (NVIDIA, sm_75 default for D1 RTX 2060) ---
set "GPU_ARCH=%GPU_ARCH: =%"
if "%GPU_ARCH%"=="" set "GPU_ARCH=75-real"
if "%WANT_CUDA%"=="1" if "%GPU_ARCH%"=="" (
  echo [build-pipyakas] no NVIDIA GPU detected; set GPU_ARCH explicitly
  exit /b 1
)

if not defined CUDAToolkit_ROOT if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3" set "CUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
if not defined CUDA_PATH if defined CUDAToolkit_ROOT set "CUDA_PATH=%CUDAToolkit_ROOT%"

REM --- backend flags ---
set "BACKENDS=cuda=%WANT_CUDA% vulkan=%WANT_VULKAN% rocm=%WANT_ROCM%"
set "BUILD=%SRC%\build-host-avx512-%GPU_ARCH%"
set "STATUS=%TMPROOT%\build-%WANT_CUDA%%WANT_VULKAN%%WANT_ROCM%.status"
set "LOG=%TMPROOT%\build-%WANT_CUDA%%WANT_VULKAN%%WANT_ROCM%.log"

if "%WANT_ROCM%"=="1" goto build_ninja
if "%WANT_VULKAN%"=="1" goto build_ninja
goto build_vs

REM ================= VS2026 path (cuda-only) =================
:build_vs
if not defined VSGEN set "VSGEN=Visual Studio 18 2026"
echo [build-pipyakas] %BACKENDS% %date% %time% GEN=%VSGEN% > "%STATUS%"
if exist "%BUILD%" (
  echo [build-pipyakas] clean %BUILD%
  rmdir /s /q "%BUILD%" 2>nul
  timeout /t 1 >nul
)
echo [build-pipyakas] cmake configure
"%CMAKE%" -S "%SRC%" -B "%BUILD%" -G "%VSGEN%" -T cuda=13.3 -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_NATIVE=OFF %CPU_ARGS% -DCMAKE_CUDA_ARCHITECTURES="%GPU_ARCH%" > "%LOG%" 2>&1
if not "%ERRORLEVEL%"=="0" (
  echo [build-pipyakas] -T cuda=13.3 failed, retry without -T
  rmdir /s /q "%BUILD%" 2>nul
  timeout /t 1 >nul
  "%CMAKE%" -S "%SRC%" -B "%BUILD%" -G "%VSGEN%" -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_NATIVE=OFF %CPU_ARGS% -DCMAKE_CUDA_ARCHITECTURES="%GPU_ARCH%" > "%LOG%" 2>&1
  if not "!ERRORLEVEL!"=="0" (
    echo [build-pipyakas] CONFIG FAILED, see %LOG%
    exit /b 1
  )
)
echo [build-pipyakas] cmake --build Release -j 12
"%CMAKE%" --build "%BUILD%" --config Release -j 12 >> "%LOG%" 2>&1
if not "%ERRORLEVEL%"=="0" (
  echo [build-pipyakas] BUILD FAILED, see %LOG%
  exit /b 1
)
set "STAGE=%BUILD%\bin\Release"
goto stage

REM ================= Ninja path (vulkan and/or rocm) =================
:build_ninja
set "BUILD=%SRC%\build-multi"
set "CFG=-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_NATIVE=OFF %CPU_ARGS% -DGGML_RPC=ON -DGGML_IQK=ON -DGGML_MAPLE=ON -DGGML_MOE_CACHE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF"
if "%WANT_CUDA%"=="1" set "CFG=%CFG% -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75-real -DGGML_CUDA_FA_ALL_QUANTS=ON"
if "%WANT_VULKAN%"=="1" (
  if not defined VULKAN_SDK set "VULKAN_SDK=C:\VulkanSDK\1.4.350.0"
  if not exist "!VULKAN_SDK!" (
    echo [build-pipyakas] no Vulkan SDK at !VULKAN_SDK!
    exit /b 1
  )
  set "CFG=!CFG! -DVULKAN_SDK=!VULKAN_SDK! -DGGML_VULKAN=ON"
)
if "%WANT_ROCM%"=="1" (
  if not defined ROCM_SDK set "ROCM_SDK=%~dp0rocm\.rocm-venv\Lib\site-packages\_rocm_sdk_devel"
  if not exist "!ROCM_SDK!\lib\llvm\bin\clang++.exe" (
    echo [build-pipyakas] no ROCm SDK at !ROCM_SDK!
    exit /b 1
  )
  set "CFG=!CFG! -DHIP_PLATFORM=amd -DHIP_PATH=!ROCM_SDK! -DCMAKE_PREFIX_PATH=!ROCM_SDK! -DGGML_HIP=ON -DGPU_TARGETS=gfx1031 -DCMAKE_C_COMPILER=!ROCM_SDK!\lib\llvm\bin\clang.exe -DCMAKE_CXX_COMPILER=!ROCM_SDK!\lib\llvm\bin\clang++.exe -DCMAKE_HIP_COMPILER=!ROCM_SDK!\lib\llvm\bin\clang.exe -DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler"
)
echo [build-pipyakas] %BACKENDS% %date% %time% Ninja > "%STATUS%"
REM VS2022 BuildTools only for legacy ROCm (HIP headers break on MSVC 19.5x).
REM Override with VS18=0 to force BuildTools, VS18=1 to force VS2026.
if not defined VS18 (
  if "%WANT_ROCM%"=="1" ( set "VS18=0" ) else ( set "VS18=1" )
)
if "%VS18%"=="0" (
  call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
) else (
  call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
)
if exist "%BUILD%" (
  echo [build-pipyakas] clean %BUILD%
  rmdir /s /q "%BUILD%" 2>nul
  timeout /t 1 >nul
)
echo [build-pipyakas] cmake configure ^(Ninja^)
echo CFG=%CFG% > "%LOG%"
"%CMAKE%" -S "%SRC%" -B "%BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_RC_COMPILER="C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/rc.exe" !CFG! >> "%LOG%" 2>&1
if not "%ERRORLEVEL%"=="0" (
  echo [build-pipyakas] CONFIG FAILED, see %LOG%
  exit /b 1
)
echo [build-pipyakas] cmake --build -j 8
"%CMAKE%" --build "%BUILD%" -j 8 >> "%LOG%" 2>&1
if not "%ERRORLEVEL%"=="0" (
  echo [build-pipyakas] BUILD FAILED, see %LOG%
  exit /b 1
)
set "STAGE=%BUILD%\bin"
goto stage

REM ================= stage (universal bin) =================
:stage
REM additive: never delete backends from a previous wider build
echo [build-pipyakas] staging from %STAGE% to %BIN%
if not exist "%BIN%" mkdir "%BIN%" 2>nul
set STAGED=0
for %%f in ("%STAGE%\*.exe" "%STAGE%\*.dll") do (
  if exist "%%f" (
    copy /Y "%%f" "%BIN%\" >nul 2>&1
    set /a STAGED+=1
  )
)
for /r "%BUILD%" %%f in (ggml-*.dll) do (
  if not exist "%BIN%\%%~nxf" (
    copy /Y "%%f" "%BIN%\" >nul 2>&1
    set /a STAGED+=1
  ) else (
    for %%a in ("%%f") do for %%b in ("%BIN%\%%~nxf") do if not "%%~za"=="%%~zb" copy /Y "%%f" "%BIN%\" >nul 2>&1
  )
)
echo STAGED %STAGED% files to %BIN% %time%>> "%STATUS%"
"%BIN%\llama-server.exe" --help >nul 2>&1
if errorlevel 1 (
  echo [build-pipyakas] smoke FAILED
  exit /b 1
)
echo [build-pipyakas] DONE backends=%BACKENDS% staged=%STAGED%
exit /b 0
