"""Real-crash resume check: run a config uninterrupted, run it again and SIGKILL the process in the middle of a
generation, resume, and compare state digests. Also reports log growth per generation. Prints JSON.

usage: python scripts/kill_resume_check.py configs/dryrun.yaml runs/dry 100"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def launch(cfg: str, out: Path) -> subprocess.Popen:
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=str(ROOT / "src"))
    return subprocess.Popen([sys.executable, "-m", "culture.run", "--config", cfg, "--out", str(out), "--quiet"],
                            env=env, cwd=ROOT, start_new_session=True)


def lines(p: Path) -> int:
    return sum(1 for _ in open(p)) if p.exists() else 0


def main() -> None:
    cfg, base, kill_gen = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
    ref_dir, crash_dir = base / "ref", base / "crash"
    t0 = time.time()
    ref = launch(cfg, ref_dir)
    crash = launch(cfg, crash_dir)
    gens = crash_dir / "generations.jsonl"
    while lines(gens) < kill_gen + 1:  # generation kill_gen has started once kill_gen rows exist... wait one more row
        if crash.poll() is not None:
            raise SystemExit("crash run ended before the kill point")
        time.sleep(0.2)
    # generation kill_gen+1 is now running; wait about half a generation, then kill the whole process group
    time.sleep(1.5)
    ck_before = json.loads((crash_dir / "checkpoint.json").read_text())["state"]["generation"]
    os.killpg(crash.pid, signal.SIGKILL)
    crash.wait()
    partial = {n: (crash_dir / f"{n}.jsonl").stat().st_size for n in ("generations", "ledger", "messages", "artifacts")}
    resumed = launch(cfg, crash_dir)
    resumed.wait()
    ref.wait()
    sys.path.insert(0, str(ROOT / "src"))
    from culture.run.runner import state_digest

    da, db = state_digest(ref_dir), state_digest(crash_dir)
    sizes = [len(x) for x in (ref_dir / "generations.jsonl").read_text().splitlines()]
    out = {
        "config": cfg, "killed_with": "SIGKILL to the process group", "checkpoint_generation_at_kill": ck_before,
        "log_bytes_at_kill": partial, "ref_returncode": ref.returncode, "resume_returncode": resumed.returncode,
        "identical": da == db, "digest_ref": da, "digest_resumed": db,
        "generations": len(sizes),
        "generation_record_bytes": {"first10_mean": sum(sizes[1:11]) / 10, "last10_mean": sum(sizes[-10:]) / 10,
                                    "max": max(sizes)},
        "run_dir_bytes": {p.name: sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
                          for p in sorted(ref_dir.iterdir())},
        "wall_seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(out, indent=1))
    (base / "kill_resume_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
