# Mosaic flow-optimisation ablation: YOLO11n Mosaic-ON vs Mosaic-OFF.
# Single variable: only `mosaic` differs between the two arms.
# ASCII-only on purpose: PowerShell 5.1 misreads Chinese .ps1 without a UTF-8 BOM.
# Paths are resolved from $PSScriptRoot so no literal CJK directory name is needed.

$ErrorActionPreference = 'Continue'

$py   = "D:\anaconda\envs\test\python.exe"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$logs = Join-Path $root "runs\mosaic_ablation_logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$master = Join-Path $logs "master.log"

"[mosaic-ablation] $(Get-Date -Format o) LAUNCH cwd=$root" | Tee-Object -FilePath $master -Append

foreach ($arm in @("on", "off")) {
    $start = Get-Date
    "$(Get-Date -Format o) arm=$arm START" | Tee-Object -FilePath $master -Append

    & $py "tools\train_mosaic_ablation.py" --arm $arm `
        1> (Join-Path $logs "$arm.stdout.log") `
        2> (Join-Path $logs "$arm.stderr.log")
    $code = $LASTEXITCODE

    $dur = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
    "$(Get-Date -Format o) arm=$arm END exit=$code dur_min=$dur" | Tee-Object -FilePath $master -Append
}

"[mosaic-ablation] $(Get-Date -Format o) ALL_DONE" | Tee-Object -FilePath $master -Append
