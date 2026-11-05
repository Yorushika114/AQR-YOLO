# Unpruned control arm: same 100-epoch protocol as the four pruned arms, but the weights are NOT pruned.
# Source = plain yolo11n (nc=3, 2,590,876 params, Detect head, no router) = the desktop unpruned baseline best.pt.
# Run AFTER the four pruned arms finish (see master.log "ALL_ARM_DONE").
# Launch detached via Start-Process so it survives session idle.
$ErrorActionPreference = 'Continue'
$py = "D:\anaconda\envs\test\python.exe"
$dir = "runs/prune_finetune_e100"
$logs = "$dir/logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$Name = "unpruned"
$Src = "C:\Users\23223\Desktop\新论文数据保存\基线测试数据\weights\best.pt"
$start = Get-Date
"[arm-$Name] $(Get-Date -Format o) START (src=$Src, unpruned)" | Tee-Object -FilePath "$logs/master.log" -Append
$cmdLine = @("tools/finetune_pruned_mcsr.py","--weights",$Src,"--unpruned","--epochs","100","--close-mosaic","10","--name",$Name,"--project",$dir,"--warmup-epochs","1.0","--sa-size","40","--seed","0","--no-mcsr","--no-sa-tal")
& $py @cmdLine 1> "$logs/$Name.stdout.log" 2> "$logs/$Name.stderr.log"
$code = $LASTEXITCODE
$dur = [math]::Round(((Get-Date) - $start).TotalMinutes,1)
"[arm-$Name] $(Get-Date -Format o) END exit=$code dur_min=$dur" | Tee-Object -FilePath "$logs/master.log" -Append
"[run-unpruned] $(Get-Date -Format o) DONE" | Tee-Object -FilePath "$logs/master.log" -Append
