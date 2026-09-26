"""Run the whole pipeline end to end.

  python run_pipeline.py                 # all steps
  python run_pipeline.py --from train    # resume from a step (scrape, clean, label, train, analyze, report)

Extra arguments after -- are passed to the labelling step, e.g.
  python run_pipeline.py --from label -- --model claude-haiku-4-5
"""
import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
STEPS = ["scrape", "clean", "label", "train", "analyze", "report"]
SCRIPTS = {"scrape": "scrape.py", "clean": "clean.py", "label": "label_llm.py", "train": "train_classifier.py",
           "analyze": "analyze.py", "report": "report.py"}


def main() -> None:
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="start", choices=STEPS, default="scrape")
    args = p.parse_args(argv)

    for step in STEPS[STEPS.index(args.start):]:
        print(f"\n=== {step} ===", flush=True)
        cmd = [sys.executable, str(SRC / SCRIPTS[step])] + (extra if step == "label" else [])
        subprocess.run(cmd, check=True, cwd=SRC.parent)


if __name__ == "__main__":
    main()
