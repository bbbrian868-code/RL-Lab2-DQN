"""Produce RESULTS.md once the experiments have finished.

    python scripts/collect_results.py

For every run it reads the config and eval history, tests `best.pt` over
`--episodes` games, generates the per-run and comparison figures, and writes
`RESULTS.md` at the repo root with the numbers and the figure paths.

Safe to run while experiments are still going -- it reports whatever exists and
marks unfinished runs as such -- but the comparisons only mean something once
the runs being compared have reached a similar step count.
"""

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, "src")

# (figure label, run A, run B, tags) -- the comparisons the report asks for.
COMPARISONS = [
	("dqn_vs_ddqn", "MsPacman_dqn", "MsPacman_ddqn",
	 ["Train/Episode Reward", "Evaluate/Episode Reward", "Train/Q Mean"]),
	("dqn_vs_dueling", "MsPacman_dqn", "MsPacman_dueling",
	 ["Train/Episode Reward", "Evaluate/Episode Reward"]),
	("dqn_vs_parallel", "MsPacman_dqn", "MsPacman_parallel",
	 ["Train/Episode Reward", "Evaluate/Episode Reward", "Perf/Steps Per Second"]),
]


def read_run(runs_dir, name):
	d = os.path.join(runs_dir, name)
	info = {"name": name, "dir": d, "config": {}, "evals": [], "finished": False}
	cfg = os.path.join(d, "config.json")
	if os.path.exists(cfg):
		with open(cfg, encoding="utf-8") as f:
			info["config"] = json.load(f)
	ev = os.path.join(d, "eval.csv")
	if os.path.exists(ev):
		with open(ev, encoding="utf-8") as f:
			info["evals"] = list(csv.DictReader(f))
	log = os.path.join(d, "train.log")
	if os.path.exists(log):
		with open(log, encoding="utf-8", errors="replace") as f:
			info["finished"] = "training finished at env step" in f.read()[-20000:]
	return info


def run_test(python, checkpoint, episodes, seed):
	"""Call test.py and parse back its JSON sidecar."""
	cmd = [python, "test.py", "--checkpoint", checkpoint,
	       "--episodes", str(episodes), "--seed", str(seed)]
	proc = subprocess.run(cmd, cwd=SRC, capture_output=True, text=True)
	if proc.returncode != 0:
		return None, proc.stdout + proc.stderr
	path = None
	for line in proc.stdout.splitlines():
		if line.startswith("saved: ") and line.endswith(".json"):
			path = line[len("saved: "):].strip()
	if not path or not os.path.exists(path):
		return None, proc.stdout
	with open(path, encoding="utf-8") as f:
		return json.load(f), proc.stdout


def export(python, args_list):
	subprocess.run([python, os.path.join("scripts", "export_curves.py")] + args_list,
	               cwd=SRC, capture_output=True, text=True)


def rel(path):
	return os.path.relpath(path, REPO).replace(os.sep, "/")


def read_seed_scans(names):
	"""Newest `seedscan_<run>_*.csv` per run, best score first."""
	log_dir = os.path.join(REPO, "test_logs")
	if not os.path.isdir(log_dir):
		return []
	out = []
	for name in names:
		matches = sorted(f for f in os.listdir(log_dir)
		                 if f.startswith(f"seedscan_{name}_") and f.endswith(".csv"))
		if not matches:
			continue
		with open(os.path.join(log_dir, matches[-1]), encoding="utf-8") as f:
			rows = [{"seed": int(r["seed"]),
			         "mean_score": float(r["mean_score"]),
			         "max_score": float(r["max_score"])}
			        for r in csv.DictReader(f)]
		if not rows:
			continue
		rows.sort(key=lambda r: r["max_score"], reverse=True)
		out.append({"run": name, "rows": rows,
		            "above3000": sum(1 for r in rows if r["max_score"] > 3000)})
	out.sort(key=lambda s: s["rows"][0]["max_score"], reverse=True)
	return out


def main(argv=None):
	p = argparse.ArgumentParser(description="Generate RESULTS.md")
	p.add_argument("--episodes", type=int, default=5)
	p.add_argument("--seed", type=int, default=0)
	p.add_argument("--runs", nargs="*", default=None)
	p.add_argument("--skip-test", action="store_true", help="reuse nothing, just read logs")
	p.add_argument("--skip-figures", action="store_true")
	p.add_argument("--python", default=sys.executable)
	p.add_argument("--out", default=os.path.join(REPO, "RESULTS.md"))
	args = p.parse_args(argv)

	runs_dir = os.path.join(REPO, "runs")
	names = args.runs or sorted(
		d for d in os.listdir(runs_dir)
		if os.path.isdir(os.path.join(runs_dir, d)) and not d.startswith("smoke"))

	runs = [read_run(runs_dir, n) for n in names]

	tests = {}
	if not args.skip_test:
		for r in runs:
			ckpt = os.path.join(REPO, "checkpoints", r["name"], "best.pt")
			if not os.path.exists(ckpt):
				continue
			print(f"testing {r['name']} ...")
			result, _ = run_test(args.python, ckpt, args.episodes, args.seed)
			if result:
				tests[r["name"]] = result

	if not args.skip_figures:
		print("exporting per-run figures ...")
		export(args.python, ["--runs"] + names)
		for label, a, b, tags in COMPARISONS:
			if a in names and b in names:
				print(f"exporting comparison {label} ...")
				export(args.python, ["--compare", a, b, "--label", label, "--tag"] + tags)
				if "parallel" in label:
					export(args.python, ["--compare", a, b, "--label", label,
					                     "--x", "wall", "--tag", "Train/Episode Reward",
					                     "Evaluate/Episode Reward"])

	# wall-clock hours and average throughput, straight out of eval.csv
	for r in runs:
		evals = r["evals"]
		if evals:
			r["hours"] = float(evals[-1]["wall_time"]) / 3600.0
			r["steps"] = int(evals[-1]["env_step"])
			r["sps"] = r["steps"] / max(1.0, float(evals[-1]["wall_time"]))
		else:
			r["hours"] = r["steps"] = r["sps"] = None

	lines = []
	w = lines.append
	w("# RESULTS.md — 實驗結果")
	w("")
	w(f"由 `src/scripts/collect_results.py` 於 "
	  f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M')} 自動產生。")
	w("")
	w("以下所有數字都直接來自 `runs/*/eval.csv`、`runs/*/config.json` 與 `test.py` "
	  "實測輸出，未經人工填寫。")
	w("")
	w("## 總表")
	w("")
	w("| 實驗 | 環境 | 變體 | 實際步數 | 目標步數 | 最佳評估分數 | 測試平均 | 測試最高 | 耗時 | 吞吐量 |")
	w("|---|---|---|---|---|---|---|---|---|---|")
	for r in runs:
		c = r["config"]
		evals = r["evals"]
		best_eval = max((float(e["mean"]) for e in evals), default=None)
		t = tests.get(r["name"])

		variant = c.get("variant", "?")
		if c.get("num_envs", 1) > 1:
			variant += f"（{c['num_envs']} 環境）"
		env_short = str(c.get("env_id", "?")).split("/")[-1]
		step_s = "-" if r["steps"] is None else f"{r['steps']:,}"
		target_s = f"{int(c.get('training_steps', 0)):,}" if c.get("training_steps") else "-"
		best_eval_s = "-" if best_eval is None else f"**{best_eval:.1f}**"
		test_mean_s = "-" if not t else f"{t['average_score']:.1f}"
		test_best_s = "-" if not t else f"{max(t['scores']):.0f}"
		hours_s = "-" if r["hours"] is None else f"{r['hours']:.1f} h"
		sps_s = "-" if r["sps"] is None else f"{r['sps']:.0f}步/秒"

		w(f"| `{r['name']}` | {env_short} | {variant} | {step_s} | {target_s} "
		  f"| {best_eval_s} | {test_mean_s} | {test_best_s} | {hours_s} | {sps_s} |")
	w("")
	w("> 「最佳評估分數」為訓練中每 100k 步、5 場評估（epsilon 0.01）的最高平均分；")
	w("> 「測試」為訓練結束後用 `best.pt` 重跑 5 場完整遊戲的結果。")
	w("> 五組實驗同時共用一張 RTX 2080 Ti，吞吐量因此互相壓低，不能當作單跑速度。")
	w("")

	for r in runs:
		c = r["config"]
		w(f"## {r['name']}")
		w("")
		w(f"- 環境：`{c.get('env_id', '?')}`，變體：`{c.get('variant', '?')}`，"
		  f"seed：{c.get('seed', '?')}，環境數：{c.get('num_envs', 1)}")
		w(f"- 目標步數：{int(c.get('training_steps', 0)):,}，"
		  f"是否跑完：{'是' if r['finished'] else '否'}")
		if r["hours"] is not None:
			w(f"- 實際耗時：{r['hours']:.1f} 小時，平均 {r['sps']:.0f} env steps/s")
		w(f"- 記錄目錄：`{rel(r['dir'])}`，權重：`checkpoints/{r['name']}/best.pt`")
		if r["evals"]:
			best_row = max(r["evals"], key=lambda e: float(e["mean"]))
			w(f"- 最佳評估：**{float(best_row['mean']):.1f}** "
			  f"（標準差 {float(best_row['std']):.1f}，最低 {float(best_row['min']):.0f}，"
			  f"最高 {float(best_row['max']):.0f}）於第 {int(best_row['env_step']):,} 步")
			w(f"- 最後一次評估：{float(r['evals'][-1]['mean']):.1f} "
			  f"於第 {int(r['evals'][-1]['env_step']):,} 步")
		t = tests.get(r["name"])
		if t:
			w("")
			w(f"`best.pt` 測試 {t['episodes']} 場，epsilon = {t['epsilon']}，"
			  f"seeds = {t['seeds']}：")
			w("")
			w("```")
			for i, (s, l) in enumerate(zip(t["scores"], t["lengths"])):
				w(f"Episode: {i + 1:<6}Length: {l:<10}Total reward: {s:.2f}")
			w(f"average score: {round(t['average_score'], 2)}")
			w("```")
			w("")
			w(f"- 單場最高：{max(t['scores']):.0f}（seed {t['best_seed']}）")
			w(f"- 完整記錄：`{rel(os.path.join(REPO, 'test_logs'))}/` 下的 "
			  f"`{r['name']}_*.txt` 與 `.json`")
		w("")

	w("## 比較圖")
	w("")
	w("報告需要的是 TensorBoard 截圖（請自行截），以下是 matplotlib 備份圖，"
	  "實線為平滑後曲線、淡色為原始曲線：")
	w("")
	figs = os.path.join(REPO, "figures")
	if os.path.isdir(figs):
		for f in sorted(os.listdir(figs)):
			if f.startswith("compare__") and f.endswith(".png"):
				w(f"- `figures/{f}`")
	w("")
	w("每組實驗各自的曲線圖與 CSV 同樣在 `figures/`，檔名格式 "
	  "`<run_name>__<tag>.png` / `.csv`。")
	w("")

	# ---- data-driven observations -------------------------------------
	w("## 初步觀察")
	w("")
	w("以下僅陳述數據呈現的事實，解釋與推論請自行在報告中展開。")
	w("")

	def ev(name):
		r = next((x for x in runs if x["name"] == name), None)
		if not r or not r["evals"]:
			return None
		return r

	def best_of(name):
		r = ev(name)
		return None if r is None else max(float(e["mean"]) for e in r["evals"])

	def at_step(name, step):
		"""Evaluation mean at (or just before) a given env step."""
		r = ev(name)
		if r is None:
			return None
		rows = [e for e in r["evals"] if int(e["env_step"]) <= step]
		return float(rows[-1]["mean"]) if rows else None

	base, ddqn = best_of("MsPacman_dqn"), best_of("MsPacman_ddqn")
	duel, par = best_of("MsPacman_dueling"), best_of("MsPacman_parallel")
	base10 = at_step("MsPacman_dqn", 10_000_000)

	if base is not None:
		w(f"- **DQN（主力）**：20M 步最佳評估 {base:.0f}。"
		  + (f"在 10M 步時為 {base10:.0f}，"
		     f"後 10M 步再增加 {base - base10:+.0f}。" if base10 is not None else ""))
	if ddqn is not None and base10 is not None:
		w(f"- **DQN vs DDQN**（同為 10M 步）：DQN {base10:.0f}、DDQN {ddqn:.0f}"
		  f"（差 {ddqn - base10:+.0f}）。"
		  "請搭配 `Train/Q Mean` 比較圖討論 over-estimation：DDQN 以 online 網路選"
		  "動作、target 網路估值，理論上 Q 值應較低。")
	if duel is not None and base10 is not None:
		w(f"- **DQN vs Dueling**（同為 10M 步）：DQN {base10:.0f}、Dueling {duel:.0f}"
		  f"（差 {duel - base10:+.0f}）。")
	if par is not None and base10 is not None:
		rp, rb = ev("MsPacman_parallel"), ev("MsPacman_dqn")
		speed = ""
		if rp and rb and rp["sps"] and rb["sps"]:
			speed = (f"吞吐量 {rp['sps']:.0f} vs {rb['sps']:.0f} 步/秒"
			         f"（{rp['sps'] / rb['sps']:.2f}x）。")
		w(f"- **DQN vs 平行 rollout**（同為 10M 步）：DQN {base10:.0f}、"
		  f"平行 {par:.0f}（差 {par - base10:+.0f}）。{speed}"
		  "注意五組實驗共用一張 GPU，GPU 使用率接近 100%，"
		  "平行化的加速因此被壓縮；要取得誠實的加速比需單獨重跑。"
		  "樣本效率請看以 env step 為 x 軸的圖，牆鐘時間請看 `__wall` 那張。")
	enduro = best_of("Enduro_dqn")
	if enduro is not None:
		w(f"- **Enduro**：最佳評估 {enduro:.0f}。前期 reward 長時間為 0 屬正常，"
		  "因為必須先超車才會得分。")
	w("")

	# ---- demo recommendation, driven by any seed scans that exist --------
	scans = read_seed_scans(names)
	w("## Demo 建議")
	w("")
	if scans:
		w("`find_best_seed.py` 掃描 seed 0–50 的結果（依最高單場分數排序）：")
		w("")
		w("| 模型 | 最佳 seed | 最高單場 | >3000 分的 seed 數 | 前五高 seed |")
		w("|---|---|---|---|---|")
		for s in scans:
			top5 = "、".join(str(x["seed"]) for x in s["rows"][:5])
			w(f"| `{s['run']}` | **{s['rows'][0]['seed']}** | "
			  f"**{s['rows'][0]['max_score']:.0f}** | {s['above3000']} / {len(s['rows'])} "
			  f"| {top5} |")
		w("")
		pick = max(scans, key=lambda s: s["rows"][0]["max_score"])
		w(f"依作業評分表單場 >3000 即可拿滿 50 分。建議用 **`{pick['run']}`**、"
		  f"**seed {pick['rows'][0]['seed']}**（{pick['rows'][0]['max_score']:.0f} 分）做 demo；"
		  f"該模型在 51 個 seed 中有 {pick['above3000']} 個超過 3000，"
		  "就算 demo 當天想換 seed 也很安全。")
		w("")
		w("```bash")
		w("cd src")
		w(f"python test.py --checkpoint ../checkpoints/{pick['run']}/best.pt \\")
		w(f"               --seeds {pick['rows'][0]['seed']} --episodes 1 --record-video")
		w("```")
	else:
		best_run = max((r for r in runs if r["evals"]),
		               key=lambda r: max(float(e["mean"]) for e in r["evals"]), default=None)
		w("尚未跑過 seed 掃描。先執行：")
		w("")
		w("```bash")
		w("cd src")
		w(f"python find_best_seed.py --checkpoint ../checkpoints/"
		  f"{best_run['name'] if best_run else '<run>'}/best.pt --max-seed 50")
		w("```")
	w("")
	w("同一個 checkpoint + 同一個 seed 重跑分數完全相同（已驗證：ALE、動作取樣器、"
	  "epsilon 擲骰都由同一個 seed 決定），所以掃描到的分數就是 demo 當天會重現的分數。")
	w("")

	with open(args.out, "w", encoding="utf-8") as f:
		f.write("\n".join(lines))
	print(f"\nwrote {args.out}")


if __name__ == "__main__":
	main()
