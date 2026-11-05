# Detached trigger: wait for master.log "ALL_ARM_DONE" (pruned 4 arms done), then
# Start-Process Hidden the unpruned-side 4-arm ablation. Run THIS via Start-Process Hidden
# so it survives session idle (a plain run_in_background task gets reaped on idle).
$ErrorActionPreference = 'Continue'
$proj = "D:\个人文件整理\02_开发项目\ultralytics-8.4.94"
$master = "$proj\runs\prune_finetune_e100\logs\master.log"
$unprunedPs1 = "$proj\tools\run_unpruned_ablations_e100.ps1"
$logs = "$proj\runs\prune_finetune_e100\logs"

# wait for ALL_ARM_DONE (line is UTF-16LE in master.log; read via Get-Content which decodes)
while (-not (Test-Path $master)) { Start-Sleep -Seconds 20 }
$done = $false
while (-not $done) {
    $content = Get-Content $master -ErrorAction SilentlyContinue | Out-String
    if ($content -match "ALL_ARM_DONE") { $done = $true }
    else { Start-Sleep -Seconds 30 }
}

"[trigger-unpruned] $(Get-Date -Format o) ALL_ARM_DONE detected -> launching unpruned side" | Add-Content -Path "$master" -Encoding UTF8
Start-Process -FilePath "powershell.exe" -WindowStyle Hidden `
  -ArgumentList @("-NoProfile","-ExecutionPolicy","Bypass","-File", $unprunedPs1) `
  -WorkingDirectory $proj -PassThru |
  ForEach-Object { "[trigger-unpruned] launched unpruned scheduler PID $($_.Id)" | Add-Content -Path "$master" -Encoding UTF8; $_ } | Select-Object -ExpandProperty Id
