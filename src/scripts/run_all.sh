#!/usr/bin/env bash
# Launch the full experiment matrix, one tmux window per run.
#
#   bash src/scripts/run_all.sh
#   tmux attach -t rl-lab2
#
# Detaching (Ctrl-b d) or dropping the SSH connection leaves everything
# running; each window keeps its shell open after the run ends so a crash
# stays visible instead of closing the window.
#
# On the Windows host used for this lab there is no WSL, so the equivalent
# launcher is run_all.ps1 -- same commands, PowerShell background processes
# instead of tmux.  See DECISIONS.md.
set -euo pipefail

SESSION="${SESSION:-rl-lab2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(dirname "$HERE")"
ROOT="$(dirname "$SRC")"
PY="${PY:-python}"

cd "$SRC"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
    exit 1
fi

# name | args
RUNS=(
  "MsPacman_dqn|--env ALE/MsPacman-v5 --variant dqn      --training-steps 20000000 --seed 0"
  "Enduro_dqn|--env ALE/Enduro-v5   --variant dqn      --training-steps 10000000 --seed 0"
  "MsPacman_ddqn|--env ALE/MsPacman-v5 --variant ddqn     --training-steps 10000000 --seed 0"
  "MsPacman_dueling|--env ALE/MsPacman-v5 --variant dueling  --training-steps 10000000 --seed 0"
  "MsPacman_parallel|--env ALE/MsPacman-v5 --variant dqn      --training-steps 10000000 --seed 0 --num-envs 8"
)

first=1
for entry in "${RUNS[@]}"; do
    name="${entry%%|*}"
    args="${entry#*|}"
    cmd="$PY main.py $args --run-name ${name}; echo; echo '=== ${name} exited ==='; exec bash"
    if [[ $first -eq 1 ]]; then
        tmux new-session -d -s "$SESSION" -n "$name" -c "$SRC" "$cmd"
        first=0
    else
        tmux new-window -t "$SESSION" -n "$name" -c "$SRC" "$cmd"
    fi
    echo "launched: $name"
    sleep 3   # stagger: five simultaneous ALE/CUDA inits thrash the disk
done

echo
echo "tmux attach -t $SESSION"
echo "python $SRC/scripts/status.py --watch 30"
echo "tensorboard --logdir $ROOT/runs"
