# Lab 2 — Deep Q-Network (MsPacman-v5 / Enduro-v5)

DQN, Double DQN, Dueling DQN and DQN with parallelized rollout, built on the TA
sample code (`base_agent.py`, `dqn_agent_atari.py`, `models/atari_model.py`,
`replay_buffer/replay_buffer.py`).

```
src/
├── base_agent.py            DQNBaseAgent: training loops, logging, checkpoints, evaluation
├── dqn_agent_atari.py       AtariDQNAgent: epsilon-greedy + the TD target for each variant
├── atari_wrappers.py        env construction (grayscale, 84x84, frame stack, frame skip)
├── models/atari_model.py    Nature DQN net + dueling head
├── replay_buffer/           uint8 frame-indexed replay memory
├── main.py                  training entry point
├── test.py                  evaluate a checkpoint, print/save the score table
├── find_best_seed.py        scan seeds to pick a reproducible demo game
├── utils.py                 run naming, stdout teeing, RNG state
├── tests/test_core.py       unit tests
└── scripts/                 run_all, status, export_curves, make_zip
```

## 助教範例程式碼 TODO 對照表

助教範例中的 `### TODO ###` 都在 `dqn_agent_atari.py`，程式碼裡以
`# ===== [TODO n] ... =====` 開頭、`# ===== [TODO n] END =====` 結尾標出
（可直接搜尋 `[TODO`）。`base_agent.py` 中範例的兩個 TODO 位於 abstract
method，實作在子類別，即下表的 TODO 3、4。

| # | 助教範例的 TODO | 實作位置 |
|---|---|---|
| 1 | initialize env (`self.env = ???`) | `dqn_agent_atari.py` 第 41–52 行，`__init__` |
| 2 | initialize test_env (`self.test_env = ???`) | `dqn_agent_atari.py` 第 54–58 行，`__init__` |
| 3 | get action from behavior net, with epsilon-greedy selection | `dqn_agent_atari.py` 第 84–110 行，`decide_agent_actions` |
| 4 | calculate the loss and update the behavior network（步驟 1–5） | `dqn_agent_atari.py` 第 119–153 行，`update_behavior_network`；步驟 4 的 `criterion` 在第 75–77 行 |

超出 TODO 的部分（報告第 3–5 項）標為 `[EXTRA: ...]`：

| 項目 | 位置 |
|---|---|
| DDQN target | `dqn_agent_atari.py` `update_behavior_network` 中的 `if self.double:` 分支 |
| Dueling DQN | `models/atari_model.py` `AtariNetDuelingDQN` |
| Parallelized rollout | `base_agent.py` `_train_vector`；`dqn_agent_atari.py` 的 vector env 與批次 epsilon-greedy |

`models/atari_model.py` 的 `AtariNetDQN` 與助教範例相同；`replay_buffer.py`
與 `base_agent.py` 的訓練迴圈以範例為基礎改寫（各檔開頭 docstring 有說明改了什麼）。

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "gymnasium[atari,other]" ale-py opencv-python tensorboard numpy \
            moviepy imageio-ffmpeg pandas matplotlib psutil pytest
```

Verified with Python 3.11, torch 2.6.0+cu124, gymnasium 1.4.0, ale-py 0.12.1,
numpy 2.2.6 on an RTX 2080 Ti.

> **Windows without admin rights:** if `import torch` or `import ale_py` fails
> with a missing `msvcp140.dll`, install the MSVC runtime system-wide
> (`winget install Microsoft.VCRedist.2015+.x64`, needs one UAC approval), or
> `pip install msvc-runtime` and add the venv's `Scripts` directory via
> `os.add_dll_directory` in a `sitecustomize.py`. See `DECISIONS.md`.

## Train

```bash
cd src
python main.py --env ALE/MsPacman-v5 --variant dqn --training-steps 20000000
```

All five experiments at once:

```bash
bash scripts/run_all.sh                                          # tmux (Linux/WSL)
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1     # Windows
```

Outputs land in `runs/<run_name>/` (TensorBoard events, `config.json`,
`eval.csv`, `train.log`) and `checkpoints/<run_name>/` (`latest.pt`, `best.pt`).
`run_name` defaults to `<env>_<variant>_s<seed>_<MMDD-HHMM>`.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--env` | `ALE/MsPacman-v5` | gymnasium env id |
| `--variant` | `dqn` | `dqn`, `ddqn`, `dueling`, `dueling_ddqn` |
| `--seed` | `0` | seeds python/numpy/torch and the envs |
| `--run-name` | auto | log/checkpoint directory name |
| `--training-steps` | `10000000` | total env steps |
| `--warmup-steps` | `20000` | random-action steps before the first gradient step |
| `--eps-decay` | `1000000` | env steps for epsilon 1.0 → `--eps-min` |
| `--eps-min` | `0.1` | final exploration rate |
| `--update-freq` | `4` | env steps per gradient step |
| `--update-target-freq` | `10000` | env steps per target-network sync |
| `--batch-size` | `32` | |
| `--learning-rate` | `0.0000625` | Adam, `eps=1.5e-4` |
| `--gamma` | `0.99` | |
| `--loss` | `mse` | `mse` or `huber` |
| `--max-grad-norm` | `10.0` | `0` disables gradient clipping |
| `--replay-buffer-capacity` | `100000` | transitions |
| `--num-envs` | `1` | `>1` enables parallelized rollout |
| `--sync-envs` | off | `SyncVectorEnv` instead of `AsyncVectorEnv` |
| `--no-clip-reward` | off | train on raw rewards (raw score is logged either way) |
| `--life-loss-terminal` | off | a lost life is terminal **in the TD target only** |
| `--eval-interval` | `100000` | env steps between evaluations |
| `--eval-episode` | `5` | games per evaluation |
| `--eval-epsilon` | `0.01` | |
| `--eval-seed-base` | `10000` | evaluation uses seeds `base..base+n-1` every time |
| `--log-interval` | `100` | gradient steps per `Train/Loss` / `Train/Q Mean` write |
| `--resume` | — | continue from a checkpoint |
| `--cpu` | off | force CPU |

### Resuming

```bash
python main.py --run-name MsPacman_dqn --resume ../checkpoints/MsPacman_dqn/latest.pt \
               --env ALE/MsPacman-v5 --variant dqn --training-steps 20000000
```

The replay buffer is **not** checkpointed (it would be a ~700 MB file per
save), so a resumed run refills it for 20k steps before taking gradient steps
again — using the current epsilon, not 1.0. The log marks this with `RESUMED`.

## Test

```bash
python test.py --checkpoint ../checkpoints/MsPacman_dqn/best.pt --episodes 5 --seed 0
```

```
Episode: 1     Length: 13301     Total reward: 1040.00
...
average score: 1037.6
```

Every run is saved to `test_logs/<run_name>_<timestamp>.txt` (for the report
screenshot) and `.json` (seeds, per-episode scores and lengths).

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | required | `best.pt`, `latest.pt`, or a bare state_dict |
| `--env` / `--variant` | from checkpoint | override if the checkpoint has no config |
| `--episodes` | `5` | |
| `--seed` | `0` | episode *i* uses `seed + i` |
| `--seeds` | — | explicit seed list, overrides `--seed` |
| `--epsilon` | `0.01` | |
| `--record-video` | off | mp4 per episode into `videos/<run_name>/` |
| `--render` | off | live window (needs a display) |
| `--no-log` | off | print only |

### Demo

```bash
python find_best_seed.py --checkpoint ../checkpoints/MsPacman_dqn/best.pt --max-seed 50
python test.py --checkpoint ../checkpoints/MsPacman_dqn/best.pt \
               --seeds <best> --episodes 1 --record-video
```

A given checkpoint + seed reproduces the same game byte for byte: the ALE (with
sticky actions and random no-ops), the action-space sampler and the
epsilon-greedy coin flips are all derived from that one seed.

`--render` needs an X display. Headless WSL2 has none, so record the mp4 and
play that instead — it satisfies the handout's "show the video while testing".

## Monitoring and figures

```bash
python scripts/status.py --watch 30
tensorboard --logdir ../runs

python scripts/export_curves.py                                   # per-run PNG + CSV
python scripts/export_curves.py --compare A B --label dqn_vs_ddqn # overlay
python scripts/export_curves.py --compare A B --x wall            # wall-clock x-axis
```

TensorBoard tags (identical across every variant, so runs overlay cleanly):
`Train/Episode Reward`, `Evaluate/Episode Reward`, `Train/Episode Length`,
`Train/Loss`, `Train/Q Mean`, `Train/Epsilon`, `Perf/Steps Per Second`. The
x-axis is always cumulative env steps, summed over all envs for parallel runs.

`Train/Episode Reward` is the **raw** game score; reward clipping to [-1, 1]
applies only to what goes into the replay buffer.

## Tests

```bash
cd src && python -m pytest tests -q
```

Covers buffer shape/dtype, frame-stack reconstruction and its episode-boundary
padding, pad entries never being sampled, truncation still bootstrapping,
per-stream isolation under parallel rollout, wraparound safety, the DQN vs DDQN
target formulas, the dueling mean-centring identity, the epsilon schedule, and
target-sync timing (including that a parallel rollout's large step jumps never
skip a sync).

## Packaging

```bash
python scripts/make_zip.py --student-id <ID> --name <Name>
```

Builds `RL_LAB2_<ID>_<Name>.zip` with `src/` and `report.pdf`, excludes weights,
logs, videos and caches, then re-opens the archive and fails if anything is
wrong.
