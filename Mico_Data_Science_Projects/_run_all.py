"""
_run_all.py - Run every analysis script end-to-end and report the key metrics
======================================================================
Used to validate the whole pipeline from a clean state (no data/ or outputs/).

Run: python _run_all.py
"""

import os
import re
import subprocess
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    r"project_1_eda_customer_behavior\eda_analysis.py",
    r"project_2_regression_house_prices\regression_model.py",
    r"project_3_classification_churn\classification_model.py",
]

KEY_LINES = re.compile(
    r"(ANOVA F-statistic|Correlation \(Age|R-squared|Adjusted R-squared|"
    r"MAE:|MSE:|RMSE:|ROC-AUC:|accuracy|complete\.)"
)


def main():
    failures = 0
    for script in SCRIPTS:
        print("\n" + "=" * 62)
        print("RUNNING:", script)
        print("=" * 62)

        result = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, script)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=BASE_DIR,
        )

        for line in result.stdout.splitlines():
            if KEY_LINES.search(line):
                print("   ", line.strip())

        if result.returncode != 0:
            failures += 1
            print("    FAILED (exit", result.returncode, ")")
            print(result.stderr[-1500:])
        else:
            print("    OK")

    print("\n" + "=" * 62)
    print(f"RESULT: {len(SCRIPTS) - failures}/{len(SCRIPTS)} scripts succeeded")
    print("=" * 62)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
