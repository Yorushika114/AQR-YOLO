$ErrorActionPreference = "Continue"
$root = "D:\个人文件整理\02_开发项目\ultralytics-8.4.94"
$py   = "D:/anaconda/envs/test/python.exe"
$out  = Join-Path $root "runs\detect\_logs_both_mosaic_off"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Set-Location $root
$p = Start-Process -FilePath $py -ArgumentList "tools/train_msa_mosaic_off.py" -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $out "stdout.log") -RedirectStandardError (Join-Path $out "stderr.log")
$p.Id | Out-File -Encoding ascii (Join-Path $out "pid.txt")
Write-Output ("STARTED PID=" + $p.Id)
