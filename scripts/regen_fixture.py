"""Re-record the stub LLM cache fixture used by tests/test_runner.py (run after changing prompts or the stub)."""

import shutil
import tempfile
from pathlib import Path

from culture.run.config import load_config
from culture.run.runner import run_config

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "llm_cache_smoke"

if __name__ == "__main__":
    if FIX.exists():
        shutil.rmtree(FIX)
    cfg = load_config(ROOT / "configs" / "smoke.yaml", {"name": "smoke-fixture", "llm": {"cache_mode": "record", "cache_dir": str(FIX)}})
    with tempfile.TemporaryDirectory() as d:
        run_config(cfg, Path(d) / "run")
    print(f"recorded {sum(1 for _ in FIX.glob('*/*.json'))} responses into {FIX}")
