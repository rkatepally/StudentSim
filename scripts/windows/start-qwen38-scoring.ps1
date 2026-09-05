param(
    [Parameter(Mandatory = $true)]
    [string]$LlamaCppDir,

    [Parameter(Mandatory = $true)]
    [string]$ModelDir,

    [int]$Context = 32768,
    [int]$Port = 8081
)

$ErrorActionPreference = "Stop"

$Server = Join-Path $LlamaCppDir "llama-server.exe"
$Model = Join-Path $ModelDir "Qwen3.8-27B-Q4_K_M.gguf"
$Mmproj = Join-Path $ModelDir "mmproj-Qwen3.8-27B-Q8_0.gguf"

foreach ($Path in @($Server, $Model, $Mmproj)) {
    if (-not (Test-Path $Path)) {
        throw "Required file not found: $Path"
    }
}

Write-Host "Starting Qwen3.8 scoring profile on http://127.0.0.1:$Port"
Write-Host "Context: $Context | MTP/speculative decoding: OFF"
Write-Host "Use this profile for fidelity runs that request top_logprobs."

& $Server `
    -m $Model `
    --mmproj $Mmproj `
    --alias qwen38-code `
    -ngl 99 `
    -fa on `
    -c $Context `
    --host 127.0.0.1 `
    --port $Port
