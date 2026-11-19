"""Scheduler for training jobs: keeps N `src.train` processes running (2 threads each).

Job file: one job per line = arguments to `python -m src.train` (lines starting with # are
skipped). The file is re-read every poll, so jobs can be appended or reordered while it runs.
A job is skipped when its results/<exp>/<test>/metrics.json exists or its .running lock is held.
Usage: python scripts/run_jobs.py <jobfile> [parallel=4]
Log: results/jobs.log
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LOG = RESULTS / "jobs.log"
PY = ROOT / ".venv" / "bin" / "python"


def log(msg: str):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%F %T')} {msg}\n")


def job_dir(job: str) -> Path:
    a = job.split()
    return RESULTS / a[a.index("--exp") + 1] / a[a.index("--test") + 1]


def running_count() -> int:
    out = subprocess.run(["pgrep", "-f", "python -m src.train"], capture_output=True, text=True).stdout
    return len([l for l in out.split() if l.strip()])


def lock_alive(d: Path) -> bool:
    lock = d / ".running"
    if not lock.exists():
        return False
    try:
        pid = int(lock.read_text().strip()); os.kill(pid, 0); return True
    except Exception:
        lock.unlink(missing_ok=True); return False


def launch(job: str):
    d = job_dir(job); d.mkdir(parents=True, exist_ok=True)
    f = open(LOG, "a")
    p = subprocess.Popen([str(PY), "-m", "src.train", *job.split(), "--threads", "2"],
                         cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    (d / ".running").write_text(str(p.pid))
    log(f"start pid {p.pid}: {job}")


def main():
    jobfile = Path(sys.argv[1]); par = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    RESULTS.mkdir(exist_ok=True)
    log(f"scheduler on {jobfile} with {par} slots")
    idle_polls = 0
    while True:
        jobs = [l.strip() for l in jobfile.read_text().splitlines() if l.strip() and not l.startswith("#")]
        pending = [j for j in jobs if not (job_dir(j) / "metrics.json").exists() and not lock_alive(job_dir(j))]
        free = par - running_count()
        if not pending:
            if running_count() == 0:
                idle_polls += 1
                if idle_polls >= 3:
                    log("queue empty and nothing running; scheduler exiting"); return
            time.sleep(20); continue
        idle_polls = 0
        for j in pending[:max(free, 0)]:
            launch(j)
        time.sleep(30)


if __name__ == "__main__":
    main()
