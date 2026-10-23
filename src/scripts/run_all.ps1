<#
.SYNOPSIS
    Launch the full experiment matrix as detached background processes.

.DESCRIPTION
    The Windows equivalent of run_all.sh.  This machine has no WSL (installing
    it needs administrator rights), so there is no tmux; Start-Process with
    -WindowStyle Hidden plays the same role.  The processes are detached from
    this shell and keep running after it closes.

    Each run already tees its own stdout to runs/<name>/train.log, so there is
    nothing to attach to -- use status.py or tail the log.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File src\scripts\run_all.ps1
    python src\scripts\status.py --watch 30
#>
param(
    [string]$Python = $null,
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$src  = Split-Path -Parent $here
$root = Split-Path -Parent $src
if (-not $Python) { $Python = Join-Path $root ".venv\Scripts\python.exe" }
if (-not (Test-Path $Python)) { throw "python not found at $Python (pass -Python <path>)" }

$runs = @(
    @{ Name = "MsPacman_dqn";      Args = "--env ALE/MsPacman-v5 --variant dqn     --training-steps 20000000 --seed 0" },
    @{ Name = "Enduro_dqn";        Args = "--env ALE/Enduro-v5   --variant dqn     --training-steps 10000000 --seed 0" },
    @{ Name = "MsPacman_ddqn";     Args = "--env ALE/MsPacman-v5 --variant ddqn    --training-steps 10000000 --seed 0" },
    @{ Name = "MsPacman_dueling";  Args = "--env ALE/MsPacman-v5 --variant dueling --training-steps 10000000 --seed 0" },
    @{ Name = "MsPacman_parallel"; Args = "--env ALE/MsPacman-v5 --variant dqn     --training-steps 10000000 --seed 0 --num-envs 8" }
)

$pidFile = Join-Path $root "runs\RUNNING_PIDS.txt"
New-Item -ItemType Directory -Force -Path (Join-Path $root "runs") | Out-Null
if (Test-Path $pidFile) { Remove-Item $pidFile }

foreach ($r in $runs) {
    $argList = "main.py $($r.Args) --run-name $($r.Name)"
    if ($WhatIf) {
        Write-Host "would launch: $Python $argList"
        continue
    }
    $p = Start-Process -FilePath $Python -ArgumentList $argList `
                       -WorkingDirectory $src -WindowStyle Hidden -PassThru
    "$($r.Name) $($p.Id)" | Add-Content -Encoding utf8 $pidFile
    Write-Host ("launched {0,-20} pid {1}" -f $r.Name, $p.Id)
    Start-Sleep -Seconds 5   # stagger: five simultaneous ALE/CUDA inits thrash the disk
}

if (-not $WhatIf) {
    Write-Host ""
    Write-Host "pids:        $pidFile"
    Write-Host "status:      $Python $src\scripts\status.py --watch 30"
    Write-Host "tensorboard: $root\.venv\Scripts\tensorboard.exe --logdir $root\runs"
    Write-Host "stop all:    Get-Content '$pidFile' | ForEach-Object { Stop-Process -Id `$_.Split(' ')[1] }"
}
