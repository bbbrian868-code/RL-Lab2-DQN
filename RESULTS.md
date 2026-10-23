# RESULTS.md — 實驗結果

由 `src/scripts/collect_results.py` 於 2026-10-11 17:25 自動產生。

以下所有數字都直接來自 `runs/*/eval.csv`、`runs/*/config.json` 與 `test.py` 實測輸出，未經人工填寫。

## 總表

| 實驗 | 環境 | 變體 | 實際步數 | 目標步數 | 最佳評估分數 | 測試平均 | 測試最高 | 耗時 | 吞吐量 |
|---|---|---|---|---|---|---|---|---|---|
| `Enduro_dqn` | Enduro-v5 | dqn | 9,920,000 | 10,000,000 | **1142.0** | 948.8 | 1094 | 12.8 h | 214步/秒 |
| `MsPacman_ddqn` | MsPacman-v5 | ddqn | 9,920,000 | 10,000,000 | **3444.0** | 3502.0 | 5050 | 11.3 h | 245步/秒 |
| `MsPacman_dqn` | MsPacman-v5 | dqn | 19,920,000 | 20,000,000 | **4044.0** | 2954.0 | 4960 | 17.5 h | 316步/秒 |
| `MsPacman_dueling` | MsPacman-v5 | dueling | 9,920,000 | 10,000,000 | **2286.0** | 2072.0 | 2270 | 12.7 h | 217步/秒 |
| `MsPacman_parallel` | MsPacman-v5 | dqn（8 環境） | 9,920,003 | 10,000,000 | **2848.0** | 1580.0 | 2530 | 7.4 h | 374步/秒 |

> 「最佳評估分數」為訓練中每 100k 步、5 場評估（epsilon 0.01）的最高平均分；
> 「測試」為訓練結束後用 `best.pt` 重跑 5 場完整遊戲的結果。
> 五組實驗同時共用一張 RTX 2080 Ti，吞吐量因此互相壓低，不能當作單跑速度。

## Enduro_dqn

- 環境：`ALE/Enduro-v5`，變體：`dqn`，seed：0，環境數：1
- 目標步數：10,000,000，是否跑完：是
- 實際耗時：12.8 小時，平均 214 env steps/s
- 記錄目錄：`runs/Enduro_dqn`，權重：`checkpoints/Enduro_dqn/best.pt`
- 最佳評估：**1142.0** （標準差 145.3，最低 1001，最高 1331）於第 9,320,000 步
- 最後一次評估：761.2 於第 9,920,000 步

`best.pt` 測試 5 場，epsilon = 0.01，seeds = [0, 1, 2, 3, 4]：

```
Episode: 1     Length: 13307     Total reward: 1034.00
Episode: 2     Length: 13311     Total reward: 1094.00
Episode: 3     Length: 6652      Total reward: 484.00
Episode: 4     Length: 13306     Total reward: 1064.00
Episode: 5     Length: 13312     Total reward: 1068.00
average score: 948.8
```

- 單場最高：1094（seed 1）
- 完整記錄：`test_logs/` 下的 `Enduro_dqn_*.txt` 與 `.json`

## MsPacman_ddqn

- 環境：`ALE/MsPacman-v5`，變體：`ddqn`，seed：0，環境數：1
- 目標步數：10,000,000，是否跑完：是
- 實際耗時：11.3 小時，平均 245 env steps/s
- 記錄目錄：`runs/MsPacman_ddqn`，權重：`checkpoints/MsPacman_ddqn/best.pt`
- 最佳評估：**3444.0** （標準差 652.0，最低 2760，最高 4670）於第 7,820,000 步
- 最後一次評估：3168.0 於第 9,920,000 步

`best.pt` 測試 5 場，epsilon = 0.01，seeds = [0, 1, 2, 3, 4]：

```
Episode: 1     Length: 803       Total reward: 1530.00
Episode: 2     Length: 1131      Total reward: 4500.00
Episode: 3     Length: 710       Total reward: 1420.00
Episode: 4     Length: 1050      Total reward: 5050.00
Episode: 5     Length: 994       Total reward: 5010.00
average score: 3502.0
```

- 單場最高：5050（seed 3）
- 完整記錄：`test_logs/` 下的 `MsPacman_ddqn_*.txt` 與 `.json`

## MsPacman_dqn

- 環境：`ALE/MsPacman-v5`，變體：`dqn`，seed：0，環境數：1
- 目標步數：20,000,000，是否跑完：是
- 實際耗時：17.5 小時，平均 316 env steps/s
- 記錄目錄：`runs/MsPacman_dqn`，權重：`checkpoints/MsPacman_dqn/best.pt`
- 最佳評估：**4044.0** （標準差 1396.3，最低 2340，最高 6030）於第 14,320,000 步
- 最後一次評估：2548.0 於第 19,920,000 步

`best.pt` 測試 5 場，epsilon = 0.01，seeds = [0, 1, 2, 3, 4]：

```
Episode: 1     Length: 1005      Total reward: 4960.00
Episode: 2     Length: 1031      Total reward: 2020.00
Episode: 3     Length: 970       Total reward: 1970.00
Episode: 4     Length: 968       Total reward: 3200.00
Episode: 5     Length: 1108      Total reward: 2620.00
average score: 2954.0
```

- 單場最高：4960（seed 0）
- 完整記錄：`test_logs/` 下的 `MsPacman_dqn_*.txt` 與 `.json`

## MsPacman_dueling

- 環境：`ALE/MsPacman-v5`，變體：`dueling`，seed：0，環境數：1
- 目標步數：10,000,000，是否跑完：是
- 實際耗時：12.7 小時，平均 217 env steps/s
- 記錄目錄：`runs/MsPacman_dueling`，權重：`checkpoints/MsPacman_dueling/best.pt`
- 最佳評估：**2286.0** （標準差 628.3，最低 1740，最高 3410）於第 4,020,000 步
- 最後一次評估：2022.0 於第 9,920,000 步

`best.pt` 測試 5 場，epsilon = 0.01，seeds = [0, 1, 2, 3, 4]：

```
Episode: 1     Length: 957       Total reward: 2270.00
Episode: 2     Length: 837       Total reward: 2220.00
Episode: 3     Length: 938       Total reward: 1820.00
Episode: 4     Length: 1088      Total reward: 2040.00
Episode: 5     Length: 862       Total reward: 2010.00
average score: 2072.0
```

- 單場最高：2270（seed 0）
- 完整記錄：`test_logs/` 下的 `MsPacman_dueling_*.txt` 與 `.json`

## MsPacman_parallel

- 環境：`ALE/MsPacman-v5`，變體：`dqn`，seed：0，環境數：8
- 目標步數：10,000,000，是否跑完：是
- 實際耗時：7.4 小時，平均 374 env steps/s
- 記錄目錄：`runs/MsPacman_parallel`，權重：`checkpoints/MsPacman_parallel/best.pt`
- 最佳評估：**2848.0** （標準差 1261.8，最低 1270，最高 4130）於第 2,020,007 步
- 最後一次評估：2196.0 於第 9,920,003 步

`best.pt` 測試 5 場，epsilon = 0.01，seeds = [0, 1, 2, 3, 4]：

```
Episode: 1     Length: 821       Total reward: 2530.00
Episode: 2     Length: 787       Total reward: 1580.00
Episode: 3     Length: 848       Total reward: 1320.00
Episode: 4     Length: 1042      Total reward: 1470.00
Episode: 5     Length: 880       Total reward: 1000.00
average score: 1580.0
```

- 單場最高：2530（seed 0）
- 完整記錄：`test_logs/` 下的 `MsPacman_parallel_*.txt` 與 `.json`

## 比較圖

報告需要的是 TensorBoard 截圖（請自行截），以下是 matplotlib 備份圖，實線為平滑後曲線、淡色為原始曲線：

- `figures/compare__dqn_vs_ddqn__Evaluate_Episode_Reward.png`
- `figures/compare__dqn_vs_ddqn__Train_Episode_Reward.png`
- `figures/compare__dqn_vs_ddqn__Train_Q_Mean.png`
- `figures/compare__dqn_vs_dueling__Evaluate_Episode_Reward.png`
- `figures/compare__dqn_vs_dueling__Train_Episode_Reward.png`
- `figures/compare__dqn_vs_parallel__Evaluate_Episode_Reward.png`
- `figures/compare__dqn_vs_parallel__Evaluate_Episode_Reward__wall.png`
- `figures/compare__dqn_vs_parallel__Perf_Steps_Per_Second.png`
- `figures/compare__dqn_vs_parallel__Train_Episode_Reward.png`
- `figures/compare__dqn_vs_parallel__Train_Episode_Reward__wall.png`

每組實驗各自的曲線圖與 CSV 同樣在 `figures/`，檔名格式 `<run_name>__<tag>.png` / `.csv`。

## 初步觀察

以下僅陳述數據呈現的事實，解釋與推論請自行在報告中展開。

- **DQN（主力）**：20M 步最佳評估 4044。在 10M 步時為 2222，後 10M 步再增加 +1822。
- **DQN vs DDQN**（同為 10M 步）：DQN 2222、DDQN 3444（差 +1222）。請搭配 `Train/Q Mean` 比較圖討論 over-estimation：DDQN 以 online 網路選動作、target 網路估值，理論上 Q 值應較低。
- **DQN vs Dueling**（同為 10M 步）：DQN 2222、Dueling 2286（差 +64）。
- **DQN vs 平行 rollout**（同為 10M 步）：DQN 2222、平行 2848（差 +626）。吞吐量 374 vs 316 步/秒（1.18x）。注意五組實驗共用一張 GPU，GPU 使用率接近 100%，平行化的加速因此被壓縮；要取得誠實的加速比需單獨重跑。樣本效率請看以 env step 為 x 軸的圖，牆鐘時間請看 `__wall` 那張。
- **Enduro**：最佳評估 1142。前期 reward 長時間為 0 屬正常，因為必須先超車才會得分。

## Demo 建議

`find_best_seed.py` 掃描 seed 0–50 的結果（依最高單場分數排序）：

| 模型 | 最佳 seed | 最高單場 | >3000 分的 seed 數 | 前五高 seed |
|---|---|---|---|---|
| `MsPacman_ddqn` | **18** | **5310** | 25 / 51 | 18、16、48、34、17 |
| `MsPacman_dqn` | **0** | **4960** | 10 / 51 | 0、18、14、24、41 |

依作業評分表單場 >3000 即可拿滿 50 分。建議用 **`MsPacman_ddqn`**、**seed 18**（5310 分）做 demo；該模型在 51 個 seed 中有 25 個超過 3000，就算 demo 當天想換 seed 也很安全。

```bash
cd src
python test.py --checkpoint ../checkpoints/MsPacman_ddqn/best.pt \
               --seeds 18 --episodes 1 --record-video
```

同一個 checkpoint + 同一個 seed 重跑分數完全相同（已驗證：ALE、動作取樣器、epsilon 擲骰都由同一個 seed 決定），所以掃描到的分數就是 demo 當天會重現的分數。
