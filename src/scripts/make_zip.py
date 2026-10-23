"""Build the submission archive and verify it before declaring success.

    python scripts/make_zip.py --student-id 313551072 --name 郭建廷

Produces `RL_LAB2_<ID>_<Name>.zip` containing exactly:

    RL_LAB2_<ID>_<Name>.zip
    |-- src/...        all source
    `-- report.pdf

Model weights, TensorBoard logs, videos and caches are excluded, and the
archive is re-opened and checked afterwards -- a wrong format costs 5 points
per violation, so this fails loudly rather than shipping a bad zip.
"""

import argparse
import fnmatch
import os
import sys
import zipfile

# Never ship these.  The handout is explicit about model weights.
EXCLUDE_PATTERNS = [
	"*.pt", "*.pth", "*.ckpt", "*.ckpt.*", "*.h5", "*.npz", "*.pkl",
	"*.tfevents*", "events.out.*",
	"*.mp4", "*.avi", "*.gif", "*.json.gz",
	"*.pyc", "*.pyo", "*.so", "*.dll", "*.log",
	".DS_Store", "Thumbs.db",
]
EXCLUDE_DIRS = {
	"__pycache__", ".pytest_cache", ".git", ".ipynb_checkpoints", ".venv", "venv",
	"runs", "checkpoints", "videos", "test_logs", "figures", "wandb",
}
WEIGHT_SUFFIXES = (".pt", ".pth", ".ckpt")


def excluded(name):
	return any(fnmatch.fnmatch(name, pat) for pat in EXCLUDE_PATTERNS)


def collect(src_dir):
	"""(absolute path, path inside the zip) for every file to include."""
	out = []
	for root, dirs, files in os.walk(src_dir):
		dirs[:] = sorted(d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith("."))
		for f in sorted(files):
			if excluded(f):
				continue
			abs_path = os.path.join(root, f)
			rel = os.path.relpath(abs_path, os.path.dirname(src_dir))
			out.append((abs_path, rel.replace(os.sep, "/")))
	return out


def verify(zip_path):
	"""Re-open the archive and assert the handout's structure. Returns problems."""
	problems = []
	with zipfile.ZipFile(zip_path) as z:
		names = z.namelist()

	weights = [n for n in names if n.lower().endswith(WEIGHT_SUFFIXES)]
	if weights:
		problems.append(f"model weights present: {weights[:5]}")

	if "report.pdf" not in names:
		problems.append("report.pdf is missing from the archive root")

	srcs = [n for n in names if n.startswith("src/")]
	if not srcs:
		problems.append("no src/ directory in the archive")
	if not any(n == "src/base_agent.py" for n in names):
		problems.append("src/base_agent.py is missing")

	stray = [n for n in names
	         if not n.startswith("src/") and n != "report.pdf" and not n.endswith("/")]
	if stray:
		problems.append(f"unexpected top-level entries: {stray[:5]}")

	for n in names:
		if any(part in EXCLUDE_DIRS for part in n.split("/")[:-1]):
			problems.append(f"excluded directory leaked into the archive: {n}")
			break
	return problems, names


def main(argv=None):
	p = argparse.ArgumentParser(description="Package the Lab 2 submission")
	p.add_argument("--student-id", required=True)
	p.add_argument("--name", required=True)
	p.add_argument("--repo-root", default=None)
	p.add_argument("--report", default=None, help="path to report.pdf")
	p.add_argument("--out", default=None)
	p.add_argument("--allow-missing-report", action="store_true",
	               help="build anyway (the archive will NOT be submittable)")
	args = p.parse_args(argv)

	repo_root = args.repo_root or os.path.dirname(
		os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	src_dir = os.path.join(repo_root, "src")
	report = args.report or os.path.join(repo_root, "report.pdf")
	out = args.out or os.path.join(repo_root, f"RL_LAB2_{args.student_id}_{args.name}.zip")

	if not os.path.isdir(src_dir):
		sys.exit(f"ERROR: {src_dir} does not exist")
	have_report = os.path.exists(report)
	if not have_report and not args.allow_missing_report:
		sys.exit(f"ERROR: {report} not found. Write the report first, or pass "
		         f"--allow-missing-report to build an incomplete archive.")

	files = collect(src_dir)
	if os.path.exists(out):
		os.remove(out)
	with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
		for abs_path, rel in files:
			z.write(abs_path, rel)
		if have_report:
			z.write(report, "report.pdf")

	problems, names = verify(out)

	print(f"archive: {out}")
	print(f"{len(names)} entries, {os.path.getsize(out) / 1e6:.2f} MB\n")
	for n in names:
		print(f"  {n}")

	if not have_report:
		problems.append("report.pdf was not included (--allow-missing-report)")

	print("")
	if problems:
		print("FAILED verification:")
		for prob in problems:
			print(f"  - {prob}")
		sys.exit(1)
	print("OK: no weight files, report.pdf present, structure matches the handout.")


if __name__ == "__main__":
	main()
