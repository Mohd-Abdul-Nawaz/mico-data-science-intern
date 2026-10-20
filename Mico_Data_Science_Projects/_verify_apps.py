"""
_verify_apps.py - Smoke test for all three Streamlit apps
=========================================================
Runs each Streamlit app through Streamlit's AppTest harness and asserts
that the script executes end-to-end with no uncaught exceptions.

Run: python _verify_apps.py
"""

import os
import sys
import traceback

from streamlit.testing.v1 import AppTest

# Windows consoles default to cp1252, which cannot encode the rupee sign.
# Force UTF-8 so metric values containing ₹ print correctly.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

APPS = [
    ("Project 1 - EDA",       "project_1_eda_customer_behavior",  "streamlit_app.py"),
    ("Project 2 - Regression", "project_2_regression_house_prices", "streamlit_app.py"),
    ("Project 3 - Classification", "project_3_classification_churn", "streamlit_app.py"),
]


def check(label, project_folder, script):
    app_path = os.path.join(BASE_DIR, project_folder, script)
    print(f"\n{'=' * 60}")
    print(f"Testing {label}  ->  {project_folder}/{script}")
    print("=" * 60)

    try:
        at = AppTest.from_file(app_path, default_timeout=180)
        at.run()
    except Exception:
        print("FAIL: app raised an exception during execution")
        traceback.print_exc()
        return False

    if at.exception:
        print("FAIL: uncaught exception in app")
        for exc in at.exception:
            print(f"  - {getattr(exc, 'value', exc)}")
        return False

    titles = [t.value for t in at.title]
    subheaders = [s.value for s in at.subheader]
    metrics = [(m.label, m.value) for m in at.metric]

    print(f"  Titles    : {titles}")
    print(f"  Subheaders: {subheaders}")
    print(f"  Metrics   : {metrics}")
    print(f"  Errors    : {len(at.error)}")
    print(f"  Warnings  : {len(at.warning)}")
    print("  RESULT    : PASS")
    return True


if __name__ == "__main__":
    results = [check(*a) for a in APPS]
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {sum(results)}/{len(results)} apps passed")
    print("=" * 60)
    sys.exit(0 if all(results) else 1)
