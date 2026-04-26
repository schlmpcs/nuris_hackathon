param(
    [string]$EnvFile = "distributed.env",
    [string]$Config = "configs/landcover_ai_training.yaml"
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $repoRoot $EnvFile
$configPath = Join-Path $repoRoot $Config
$srcPath = Join-Path $repoRoot "src"

if (-not (Test-Path $envPath)) {
    throw "Env file not found: $envPath"
}

if (-not (Test-Path $configPath)) {
    throw "Config file not found: $configPath"
}

if (-not (Test-Path $srcPath)) {
    throw "Source directory not found: $srcPath"
}

Get-Content $envPath | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith("#")) {
        return
    }

    $parts = $line -split "=", 2
    if ($parts.Count -ne 2) {
        throw "Invalid env line: $line"
    }

    $name = $parts[0].Trim()
    $value = $parts[1].Trim()
    Set-Item -Path "Env:$name" -Value $value
}

Set-Location $repoRoot
$env:PYTHONPATH = $srcPath
$env:GLOO_SOCKET_IFNAME = "Ethernet"
py -3.11 -m nuris_pipeline.cli train-segmentation --config $Config
