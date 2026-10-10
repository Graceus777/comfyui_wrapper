param([switch]$SkipModelDownload)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskRuntime = Join-Path $taskRoot '.local\tts'
$taskCache = Join-Path $taskRuntime 'uv-cache'
$taskPythonDir = Join-Path $taskRuntime 'python'
$taskVenv = Join-Path $taskRuntime 'venv'
$taskPython = Join-Path $taskVenv 'Scripts\python.exe'

function Invoke-TtsUv {
    & uv @args --cache-dir $taskCache --no-progress
    if ($LASTEXITCODE -ne 0) { throw "TTS setup failed: uv $args" }
}

if (-not (Test-Path -LiteralPath $taskPython)) {
    Invoke-TtsUv python install 3.12 --install-dir $taskPythonDir --no-bin --no-registry
    $taskManagedPython = Join-Path $taskPythonDir 'cpython-3.12-windows-x86_64-none\python.exe'
    Invoke-TtsUv venv $taskVenv --python $taskManagedPython --no-python-downloads
}
Invoke-TtsUv pip install --python $taskPython --index-url 'https://download.pytorch.org/whl/cu128' 'torch==2.8.0' 'torchaudio==2.8.0'
Invoke-TtsUv pip install --python $taskPython --no-deps --link-mode copy 'dots.tts==0.3.1'
Invoke-TtsUv pip install --python $taskPython -r (Join-Path $PSScriptRoot 'requirements-voice-clone.txt')
& $taskPython (Join-Path $PSScriptRoot 'dots_tts_windows_compat.py')
if ($LASTEXITCODE -ne 0) { throw 'Windows compatibility setup failed.' }
$env:HF_HOME = Join-Path $taskRuntime 'hf-cache'
if (-not $SkipModelDownload) {
    & $taskPython (Join-Path $PSScriptRoot 'download_voice_clone.py')
    if ($LASTEXITCODE -ne 0) { throw 'Model download or verification failed.' }
}
Write-Output "Voice Clone runtime installed. Launch the wrapper: python '$taskRoot\webui.py'"
