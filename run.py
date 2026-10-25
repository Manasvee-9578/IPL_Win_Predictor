"""
ChaseIQ - one-command launcher for demos.

    python run.py                 start API (8000) + dashboard (8501)
    python run.py --pipeline      rebuild data -> features -> model -> database first
    python run.py --test          run the test-suite
"""

import argparse
import os
import subprocess
import sys
import time
import webbrowser


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pipeline", action="store_true", help="re-run the offline pipeline first")
    ap.add_argument("--test", action="store_true", help="run pytest and exit")
    ap.add_argument("--api-port", type=int, default=8000)
    ap.add_argument("--ui-port", type=int, default=8501)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    if args.test:
        return subprocess.call([py, "-m", "pytest", "-q"])
    if args.pipeline:
        if subprocess.call([py, "-m", "src.run_pipeline"]) != 0:
            return 1

    env = {**os.environ, "API_BASE": f"http://127.0.0.1:{args.api_port}"}
    api = subprocess.Popen([py, "-m", "uvicorn", "api.main:app", "--port", str(args.api_port)], env=env)
    ui = subprocess.Popen([py, "-m", "streamlit", "run", "dashboard/app.py",
                           "--server.port", str(args.ui_port), "--server.headless", "true"], env=env)
    print(f"\n  API docs  : http://127.0.0.1:{args.api_port}/docs"
          f"\n  Dashboard : http://127.0.0.1:{args.ui_port}\n  (Ctrl+C to stop)\n")
    if not args.no_browser:
        time.sleep(4)
        webbrowser.open(f"http://127.0.0.1:{args.ui_port}")
    try:
        while api.poll() is None and ui.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for proc in (api, ui):
            if proc.poll() is None:
                proc.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
