@echo off
setlocal enableextensions
rem Build codemason and install it somewhere it can find its bundled ripgrep.
rem
rem WHY THIS EXISTS: `cargo build` alone is not an install. The binary looks for
rem vendor\ripgrep\<arch-os>\rg.exe next to itself, then one and two directories
rem up -- so it works from target\release inside this checkout (two up is the
rem repo root, where vendor\ lives), and stops working the moment you copy just
rem the exe somewhere else. Grep and Glob then fall back to a native directory
rem walk, which is correct but much slower, and nothing tells you it happened.
rem
rem So the exe, vendor\ and tokenizer\ travel together. This copies all three.
rem (Windows twin of setup.sh -- .bat runs in cmd.exe with no ExecutionPolicy
rem hurdle, matching the Python original's pair.)

cd /d "%~dp0"
set "REPO=%CD%"

if "%~1"=="" (set "PREFIX=%LOCALAPPDATA%\codemason") else (set "PREFIX=%~1")

rem rustup installs to %USERPROFILE%\.cargo\bin and puts it on the USER PATH,
rem but that entry goes missing often enough to matter -- a PATH edited by hand,
rem a profile copied between machines, an install done under another account.
rem When it does, `where cargo` fails and the old message here said "Install
rem Rust", which sends you to reinstall a toolchain that is already sitting on
rem the disk. Look there before saying that.
where cargo >nul 2>nul
if errorlevel 1 (
    if exist "%USERPROFILE%\.cargo\bin\cargo.exe" (
        echo [setup] cargo is not on PATH; using %USERPROFILE%\.cargo\bin
        set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
    ) else (
        echo [setup] cargo not found. Install Rust: https://rustup.rs
        exit /b 1
    )
)

echo [setup] building release binary
cargo build --release
if errorlevel 1 goto :fail

echo [setup] installing to %PREFIX%
if not exist "%PREFIX%" mkdir "%PREFIX%"
copy /y "%REPO%\target\release\codemason.exe" "%PREFIX%\codemason.exe" >nul
if errorlevel 1 goto :fail

rem vendor\ sits BESIDE the binary, which is the first place it looks.
if exist "%REPO%\vendor\ripgrep" (
    if exist "%PREFIX%\vendor" rmdir /s /q "%PREFIX%\vendor"
    mkdir "%PREFIX%\vendor"
    xcopy /e /i /q /y "%REPO%\vendor\ripgrep" "%PREFIX%\vendor\ripgrep" >nul
    if errorlevel 1 goto :fail
    echo [setup] bundled ripgrep installed alongside the binary
) else (
    echo [setup] WARNING: vendor\ripgrep is missing from this checkout.
    echo         Grep and Glob will fall back to a slower native walk unless
    echo         ripgrep is on PATH.
)

rem tokenizer\ resolves the same way vendor\ does — beside the binary, then one
rem and two directories up. It decides every token count in the process (the
rem auto-compact threshold, the context-window check, the usage readout), and a
rem missing one does NOT fail loudly: the process falls back to a character
rem heuristic. That is why this is copied here and not left to the checkout.
if exist "%REPO%\tokenizer" (
    if exist "%PREFIX%\tokenizer" rmdir /s /q "%PREFIX%\tokenizer"
    mkdir "%PREFIX%\tokenizer"
    xcopy /e /i /q /y "%REPO%\tokenizer" "%PREFIX%\tokenizer" >nul
    if errorlevel 1 goto :fail
    echo [setup] model tokenizers installed alongside the binary
) else (
    echo [setup] WARNING: tokenizer\ is missing from this checkout.
    echo         Token counts will fall back to a character heuristic, so
    echo         auto-compact and context-window checks become approximate.
)

call :make_cm2
if errorlevel 1 goto :fail
call :ensure_path

echo.
echo [setup] done.
echo         binary : %PREFIX%\codemason.exe
echo         command: %PREFIX%\cm2.exe
echo         vendor : %PREFIX%\vendor\ripgrep
echo         tokens : %PREFIX%\tokenizer
echo.
echo         Open a NEW terminal, then: cm2 -rt -dsp
exit /b 0

rem ---------------------------------------------------------------------------
rem `cm2` is the command the README documents and the only one anybody types.
rem setup.sh gives Linux users it as a symlink; this side used to give them
rem nothing, so on Windows `cm2` simply did not exist -- you installed, opened a
rem terminal, typed the documented command and got "not recognized".
rem
rem A hard link, not a .cmd shim: cmd.exe asks "Terminate batch job (Y/N)?" when
rem you Ctrl+C out of a batch wrapper, and this is an interactive REPL where
rem Ctrl+C is a normal keystroke -- a wrapper would put that prompt in the way
rem of every interrupt. A hard link is a second directory entry for the same
rem file, so cm2.exe IS the binary. It needs no elevation (only SYMlinks do) and
rem it must live in the same directory anyway, which is also where vendor\ is.
rem Copy is the fallback for a volume that cannot hard link.
:make_cm2
if exist "%PREFIX%\cm2.exe" del /q "%PREFIX%\cm2.exe"
mklink /h "%PREFIX%\cm2.exe" "%PREFIX%\codemason.exe" >nul 2>nul
if not errorlevel 1 (
    echo [setup] cm2 installed as a hard link to codemason.exe
    exit /b 0
)
copy /y "%PREFIX%\codemason.exe" "%PREFIX%\cm2.exe" >nul
if errorlevel 1 (
    echo [setup] FAILED to create cm2.exe
    exit /b 1
)
echo [setup] cm2 installed as a copy ^(this volume cannot hard link^)
exit /b 0

rem Put PREFIX on the user's PATH. Windows has no equivalent of ~/.local/bin
rem being on PATH already, so without this the install completes and the command
rem still does not resolve -- which is the whole failure this section prevents.
rem
rem Written through the registry with the value KIND preserved. The usual
rem shortcuts are both lossy: setx truncates a PATH over 1024 characters, and
rem [Environment]::SetEnvironmentVariable rewrites REG_EXPAND_SZ as REG_SZ,
rem which silently breaks any %VAR% already in the user's PATH. Idempotent:
rem re-running adds nothing.
:ensure_path
powershell -NoProfile -Command "$k='HKCU:\Environment'; $i=Get-Item $k; $v=[string]$i.GetValue('Path',$null,'DoNotExpandEnvironmentNames'); $add='%PREFIX%'; if (($v -split ';') -contains $add) { Write-Host '[setup] already on your user PATH' } else { Set-ItemProperty -Path $k -Name Path -Value ($v.TrimEnd(';') + ';' + $add) -Type ExpandString; Write-Host '[setup] added to your user PATH (new terminals only)' }"
exit /b 0

:fail
echo [setup] FAILED
exit /b 1
