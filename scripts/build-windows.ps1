# Builds Matita for Windows (x64): dist\Matita\ (the app folder) and dist\Matita-<version>-setup.exe.
# Runs on a GitHub windows-2022 runner, or any Windows PC with Visual Studio 2022 (C++), CMake, Git and Inno Setup 6.
#   pwsh scripts\build-windows.ps1
# Inside the app folder: Python (python-build-standalone), whisper-cli with its CPU and Vulkan back ends,
# FFmpeg (LGPL build), matita.py and its icon. Set VULKAN_SDK to build the GPU back end.
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$Root = (Get-Location).Path
$WhisperTag = if ($env:WHISPER_TAG) { $env:WHISPER_TAG } else { "v1.9.5" }
$PyMinor = if ($env:PY_MINOR) { $env:PY_MINOR } else { "3.12" }
$B = Join-Path $Root ".build-win"
$Dist = Join-Path $Root "dist"
New-Item -ItemType Directory -Force -Path $B, $Dist | Out-Null
$Version = (Select-String -Path matita.py -Pattern '^VERSION = "(.+)"').Matches[0].Groups[1].Value
Write-Host "== Matita $Version for Windows"
$Headers = @{ "Accept" = "application/vnd.github+json" }
if ($env:GITHUB_TOKEN) { $Headers["Authorization"] = "Bearer $env:GITHUB_TOKEN" }

function Invoke-Checked([string]$what, [scriptblock]$cmd) {
  & $cmd
  if ($LASTEXITCODE -ne 0) { throw "$what failed ($LASTEXITCODE)" }
}

# ---------------------------------------------------------------- whisper.cpp (CPU variants + Vulkan, loaded at run time)
$W = Join-Path $B "whisper"
if (-not (Test-Path (Join-Path $W "whisper-cli.exe"))) {
  Write-Host "== whisper.cpp $WhisperTag"
  $Src = Join-Path $B "whisper.cpp"
  if (Test-Path $Src) { Remove-Item -Recurse -Force $Src }
  Invoke-Checked "git clone" { git clone -q --depth 1 --branch $WhisperTag https://github.com/ggml-org/whisper.cpp $Src }
  $Vk = if ($env:VULKAN_SDK) { "ON" } else { "OFF" }
  Write-Host "   Vulkan: $Vk"
  # Each back end is its own DLL: a PC without a Vulkan driver simply skips that one, and the CPU
  # back end picks the best variant (AVX2, AVX-512...) for the processor it finds.
  Invoke-Checked "cmake configure" { cmake -S $Src -B "$Src\build" -A x64 `
    -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF `
    -DGGML_VULKAN=$Vk -DWHISPER_SDL2=OFF -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF }
  Invoke-Checked "cmake build" { cmake --build "$Src\build" --config Release -j $env:NUMBER_OF_PROCESSORS }
  New-Item -ItemType Directory -Force -Path $W | Out-Null
  $Out = Join-Path $Src "build\bin\Release"
  Copy-Item (Join-Path $Out "whisper-cli.exe") $W
  Copy-Item (Join-Path $Out "*.dll") $W
  Get-ChildItem $W | ForEach-Object { Write-Host "   $($_.Name)" }
}

# ---------------------------------------------------------------- FFmpeg (BtbN's LGPL build, static)
$F = Join-Path $B "ffmpeg"
if (-not (Test-Path (Join-Path $F "ffmpeg.exe"))) {
  Write-Host "== FFmpeg"
  $rel = Invoke-RestMethod -Headers $Headers "https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest"
  $rel.assets | Where-Object { $_.name -match 'win64-lgpl' } | ForEach-Object { Write-Host "   available: $($_.name)" }
  # the newest numbered release build (static, LGPL); the nightly "master" build only if there is none
  $asset = $rel.assets | Where-Object { $_.name -match '^ffmpeg-n(\d+)\.(\d+)-latest-win64-lgpl-\d+\.\d+\.zip$' } |
    Sort-Object { [version]($_.name -replace '^ffmpeg-n(\d+\.\d+)-.*$', '$1') } -Descending | Select-Object -First 1
  if (-not $asset) { $asset = $rel.assets | Where-Object { $_.name -eq 'ffmpeg-master-latest-win64-lgpl.zip' } | Select-Object -First 1 }
  if (-not $asset) { throw "No LGPL FFmpeg build for Windows found" }
  Write-Host "   $($asset.name)"
  $zip = Join-Path $B "ffmpeg.zip"
  Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip
  $x = Join-Path $B "ffmpeg-x"
  if (Test-Path $x) { Remove-Item -Recurse -Force $x }
  Expand-Archive $zip $x
  New-Item -ItemType Directory -Force -Path $F | Out-Null
  $bin = Get-ChildItem -Recurse -Path $x -Filter ffmpeg.exe | Select-Object -First 1
  Copy-Item (Join-Path $bin.DirectoryName "ffmpeg.exe"), (Join-Path $bin.DirectoryName "ffprobe.exe") $F
  Copy-Item (Get-ChildItem -Recurse -Path $x -Filter LICENSE.txt | Select-Object -First 1).FullName (Join-Path $F "FFMPEG-LICENSE.txt")
  $script:FfmpegName = $asset.name
}

# ---------------------------------------------------------------- Python (python-build-standalone)
$P = Join-Path $B "python"
if (-not (Test-Path (Join-Path $P "python.exe"))) {
  Write-Host "== Python $PyMinor"
  $rel = Invoke-RestMethod -Headers $Headers "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest"
  $re = '^cpython-' + [regex]::Escape($PyMinor) + '\.\d+\+\d+-x86_64-pc-windows-msvc-install_only_stripped\.tar\.gz$'
  $asset = $rel.assets | Where-Object { $_.name -match $re } | Select-Object -First 1
  if (-not $asset) { throw "No Python $PyMinor build for Windows found" }
  Write-Host "   $($asset.name)"
  $tgz = Join-Path $B "python.tar.gz"
  Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $tgz
  $x = Join-Path $B "pyx"
  if (Test-Path $x) { Remove-Item -Recurse -Force $x }
  New-Item -ItemType Directory -Force -Path $x | Out-Null
  Invoke-Checked "tar" { tar -xzf $tgz -C $x }
  Move-Item (Join-Path $x "python") $P
  # what Matita never needs (venv and pip stay: speaker names and Parakeet install into a venv)
  foreach ($d in "Lib\test", "Lib\idlelib", "Lib\tkinter", "Lib\turtledemo", "tcl") {
    $t = Join-Path $P $d
    if (Test-Path $t) { Remove-Item -Recurse -Force $t }
  }
}

# ---------------------------------------------------------------- the app folder
Write-Host "== app folder"
$App = Join-Path $Dist "Matita"
if (Test-Path $App) { Remove-Item -Recurse -Force $App }
New-Item -ItemType Directory -Force -Path (Join-Path $App "bin") | Out-Null
Copy-Item -Recurse $P (Join-Path $App "python")
Copy-Item (Join-Path $W "*") (Join-Path $App "bin")
Copy-Item (Join-Path $F "ffmpeg.exe"), (Join-Path $F "ffprobe.exe") (Join-Path $App "bin")
# the Visual C++ runtime, next to the programs that use it (not every PC has it installed)
foreach ($dll in "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll") {
  $src = Join-Path $env:SystemRoot "System32\$dll"
  foreach ($dir in "bin", "python") {
    if ((Test-Path $src) -and -not (Test-Path (Join-Path $App "$dir\$dll"))) { Copy-Item $src (Join-Path $App $dir) }
  }
}
Copy-Item matita.py (Join-Path $App "matita.py")
Copy-Item LICENSE (Join-Path $App "LICENSE.txt")
Copy-Item (Join-Path $F "FFMPEG-LICENSE.txt") $App
Get-ChildItem docs -Filter "*-OFL.txt" | ForEach-Object { Copy-Item $_.FullName $App }
@"
Matita for Windows includes:
- whisper.cpp $WhisperTag (MIT) https://github.com/ggml-org/whisper.cpp
- FFmpeg (LGPL 2.1, built by https://github.com/BtbN/FFmpeg-Builds without GPL parts) https://ffmpeg.org
- Python $PyMinor from python-build-standalone (PSF License and others) https://github.com/astral-sh/python-build-standalone
"@ | Set-Content -Encoding utf8 (Join-Path $App "THIRD-PARTY.txt")

# icon: the PNG inside matita.py, as a multi-size .ico
$png = Join-Path $B "icon.png"
Invoke-Checked "write icon" { & (Join-Path $App "python\python.exe") matita.py --write-icon $png }
Invoke-Checked "ico" { python scripts\make_ico.py $png (Join-Path $App "matita.ico") }

# every program inside must start without anything from the build machine
foreach ($exe in "bin\whisper-cli.exe", "bin\ffmpeg.exe", "python\python.exe") {
  $o = & (Join-Path $App $exe) --help 2>&1 | Out-String
  if ($LASTEXITCODE -ne 0 -and $exe -ne "bin\ffmpeg.exe") { throw "$exe doesn't start: $o" }
}
$size = "{0:N0} MB" -f ((Get-ChildItem -Recurse $App | Measure-Object -Sum Length).Sum / 1MB)
Write-Host "   $App ($size)"

# ---------------------------------------------------------------- installer
Write-Host "== installer"
$iscc = @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) {
  Invoke-Checked "choco install innosetup" { choco install innosetup -y --no-progress }
  $iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
}
Invoke-Checked "ISCC" { & $iscc /Qp "/DAppVersion=$Version" "/DAppDir=$App" "/DOutDir=$Dist" scripts\matita.iss }
Get-ChildItem $Dist -Filter "*.exe" | ForEach-Object { Write-Host ("   {0} ({1:N0} MB)" -f $_.Name, ($_.Length / 1MB)) }
