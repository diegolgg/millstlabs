"""usage: python scripts/run_experiment.py configs/stage1_transfer.yaml runs/stage1 [processes]"""

import sys

from culture.run.experiment import run_experiment

if __name__ == "__main__":
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else None
    for p in run_experiment(sys.argv[1], sys.argv[2], procs):
        print(p)
