#!/usr/bin/env python3
"""Launch JFAA probe training through the V-JEPA2 eval runner."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/jfaa_vjepa2_vitG_train.yaml")
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--vjepa2-root", help="Optional path to a local facebookresearch/vjepa2 checkout.")
    parser.add_argument("--debugmode", action="store_true")
    parser.add_argument("--master-addr", default=os.environ.get("MASTER_ADDR", "127.0.0.1"))
    parser.add_argument("--master-port", default=os.environ.get("MASTER_PORT", "auto"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    env = os.environ.copy()
    pythonpath = [str(repo_root)]
    if args.vjepa2_root:
        pythonpath.append(str(Path(args.vjepa2_root).resolve()))
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    cmd = [
        sys.executable,
        "-m",
        "evals.main",
        "--fname",
        args.config,
        "--devices",
        *args.devices,
        "--master_addr",
        args.master_addr,
        "--master_port",
        args.master_port,
    ]
    if args.debugmode:
        cmd.extend(["--debugmode", "true"])
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)


if __name__ == "__main__":
    main()
