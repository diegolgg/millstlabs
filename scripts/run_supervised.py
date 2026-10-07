"""Run a long command unattended with crash-resume against the local MLX server (Overnight 2).

The command must be resumable on its own (the single-student driver skips replays already in replays.jsonl and
serves finished LLM calls from its cache). Before every attempt the supervisor checks the server's /v1/models; if it
does not answer, it kills any mlx_lm.server process and restarts it with the command in CLAUDE.md, then waits for
/v1/models. A failed attempt (non-zero exit) is retried up to --max-attempts times. Everything goes to --log.

    python scripts/run_supervised.py --log runs/c1-full/progress.log -- \
        .venv/bin/python -m culture.run.single_student --spec configs/c1_full_mlx.yaml --out runs/c1-full
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = [str(ROOT / ".venv" / "bin" / "python"), "-m", "mlx_lm.server", "--model",
          "mlx-community/Qwen3.6-35B-A3B-4bit", "--host", "127.0.0.1", "--port", "8080", "--max-tokens", "8192",
          "--chat-template-args", '{"enable_thinking": false}']


def log(path: Path, msg: str) -> None:
    line = f"[{dt.datetime.now().isoformat(timespec='seconds')}] supervisor: {msg}"
    with open(path, "a") as f:
        f.write(line + "\n")


def healthy(url: str, timeout: float = 15.0) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=timeout) as r:  # noqa: S310 - local URL
            return r.status == 200
    except Exception:
        return False


def restart_server(path: Path, url: str, wait_s: float = 900.0) -> bool:
    subprocess.run(["pkill", "-f", "mlx_lm.server"], check=False)
    time.sleep(10)
    with open(path.parent / "mlx_server.log", "a") as f:
        subprocess.Popen(SERVER, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if healthy(url):
            log(path, f"server up after {time.time() - t0:.0f} s")
            return True
        time.sleep(10)
    log(path, f"server not up after {wait_s:.0f} s")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--max-attempts", type=int, default=8)
    ap.add_argument("--no-server", action="store_true", help="the command makes no model calls (stub runs)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    path = Path(a.log)
    path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    for attempt in range(1, a.max_attempts + 1):
        if not a.no_server and not healthy(a.url):
            log(path, "server not answering; restarting it")
            if not restart_server(path, a.url):
                time.sleep(60)
                continue
        log(path, f"attempt {attempt}: {' '.join(cmd)}")
        t0 = time.time()
        with open(path, "a") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env).returncode
        log(path, f"attempt {attempt} exited {rc} after {time.time() - t0:.0f} s")
        if rc == 0:
            log(path, "done")
            return 0
        time.sleep(30)
    log(path, "giving up")
    return 1


if __name__ == "__main__":
    sys.exit(main())
