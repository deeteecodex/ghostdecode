# Builds GhostDecode.exe (Windows). Run from the repo folder, venv active:
#     .\build_exe.ps1
# Needs Tesseract installed at C:\Program Files\Tesseract-OCR (it is copied into the app).
$ErrorActionPreference = "Stop"

$tess = "C:\Program Files\Tesseract-OCR"
if (-not (Test-Path "$tess\tesseract.exe")) {
    throw "Tesseract not found at $tess. Install it first: https://github.com/UB-Mannheim/tesseract/wiki"
}

python -m pip install --upgrade pyinstaller
if ($LASTEXITCODE -ne 0) { throw "pip install pyinstaller failed" }

# Stage just what Tesseract needs (exe, DLLs, English data) so the app stays small.
$stage = "vendor\tesseract"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory "$stage\tessdata" -Force | Out-Null
Copy-Item "$tess\*.exe" $stage
Copy-Item "$tess\*.dll" $stage
Copy-Item "$tess\tessdata\eng.traineddata" "$stage\tessdata\"
foreach ($d in "configs", "tessconfigs") {
    if (Test-Path "$tess\tessdata\$d") { Copy-Item "$tess\tessdata\$d" "$stage\tessdata\$d" -Recurse }
}

# Build a folder-style app (starts faster and triggers fewer antivirus false alarms
# than a single-file exe).
pyinstaller --noconfirm --clean --windowed --name GhostDecode `
    --add-data "vendor\tesseract;tesseract" `
    ghostdecode_app.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

Copy-Item "README.md" "dist\GhostDecode\README.md" -ErrorAction SilentlyContinue
Copy-Item "THIRD_PARTY_NOTICES.md" "dist\GhostDecode\THIRD_PARTY_NOTICES.md" -ErrorAction SilentlyContinue

Compress-Archive -Path "dist\GhostDecode\*" -DestinationPath "dist\GhostDecode-windows.zip" -Force
Write-Host ""
Write-Host "Done. Test:  dist\GhostDecode\GhostDecode.exe"
Write-Host "Share:       dist\GhostDecode-windows.zip"
