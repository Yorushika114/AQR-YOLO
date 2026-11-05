# Unpruned-side 2x4 ablation arms: same 100-epoch protocol as the pruned arms, but NO pruning.
# Arms: unpruned_none / unpruned_mcsr / unpruned_satal / unpruned_both.
# Sources (all unpruned, BEFORE == AFTER, no parameter reduction):
#   unpruned_none / unpruned_satal  <- desktop baseline best.pt  (plain Detect, 2,590,876)
#   unpruned_mcsr / unpruned_both   <- desktop both best.pt      (AdaptiveDefectDetect+router, 2,877,957)
# Each is tagged --unpruned so load_and_audit treats it as unpruned (mcsr derived from head).
# Run AFTER the pruned four arms finish (see master.log "ALL_ARM_DONE").
$ErrorActionPreference = 'Continue'
$py = "D:\anaconda\envs\test\python.exe"
$dir = "runs/prune_finetune_e100"
$logs = "$dir/logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$BASE = "C:\Users\23223\Desktop\新论文数据保存\基线测试数据\weights\best.pt"
$BOTH = "C:\Users\23223\Desktop\新论文数据保存\MCSR_SA-TAL消融实验_75ep\both\weights\best.pt"

function Run-Arm {
  param([string]$Name, [string]$Src, [string[]]$Flags)
  $start = Get-Date
  "[arm-$Name] $(Get-Date -Format o) START (src=$Src, flags=$($Flags -join ' '))" | Tee-Object -FilePath "$logs/master.log" -Append
  $cmdLine = @("tools/finetune_pruned_mcsr.py","--weights",$Src,"--unpruned","--epochs","100","--close-mosaic","10","--name",$Name,"--project",$dir,"--warmup-epochs","1.0","--sa-size","40","--seed","0") + $Flags
  & $py @cmdLine 1> "$logs/$Name.stdout.log" 2> "$logs/$Name.stderr.log"
  $code = $LASTEXITCODE
  $dur = [math]::Round(((Get-Date) - $start).TotalMinutes,1)
  "[arm-$Name] $(Get-Date -Format o) END exit=$code dur_min=$dur" | Tee-Object -FilePath "$logs/master.log" -Append
}

Run-Arm -Name "unpruned_none"   -Src $BASE -Flags @("--no-mcsr","--no-sa-tal")
Run-Arm -Name "unpruned_mcsr"   -Src $BOTH -Flags @("--mcsr","--no-sa-tal")
Run-Arm -Name "unpruned_satal"  -Src $BASE -Flags @("--no-mcsr","--sa-tal")
Run-Arm -Name "unpruned_both"   -Src $BOTH -Flags @("--mcsr","--sa-tal")

"[run-unpruned] $(Get-Date -Format o) ALL_UNPRUNED_DONE" | Tee-Object -FilePath "$logs/master.log" -Append
