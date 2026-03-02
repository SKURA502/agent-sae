#!/usr/bin/env python3
"""Simple SwanLab sync loop for a single offline run directory.

Example:
  python scripts/swanlab_sync_loop.py \
    --run-dir /data/agent_tool_use/Agent-Tool-Use-MI/swanlog/run-20260302_131903-cjnyta1safqaehr63rbi8 \
    --interval 10
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


RUN_ID_IN_URL = re.compile(r"/runs/([A-Za-z0-9]{21})")


def read_cloud_run_id(run_dir: Path) -> str | None:
    id_file = run_dir / ".cloud_run_id"
    if not id_file.exists():
        return None
    run_id = id_file.read_text(encoding="utf-8").strip()
    if re.fullmatch(r"[A-Za-z0-9]{21}", run_id):
        return run_id
    return None


def write_cloud_run_id(run_dir: Path, run_id: str) -> None:
    id_file = run_dir / ".cloud_run_id"
    id_file.write_text(run_id + "\n", encoding="utf-8")


def extract_cloud_run_id(stdout: str, stderr: str) -> str | None:
    merged = f"{stdout}\n{stderr}"
    matches = RUN_ID_IN_URL.findall(merged)
    return matches[-1] if matches else None


def build_sync_cmd(args: argparse.Namespace, cloud_run_id: str | None) -> list[str]:
    cmd = ["swanlab", "sync", str(args.run_dir)]

    if args.workspace:
        cmd.extend(["--workspace", args.workspace])
    if args.project:
        cmd.extend(["--project", args.project])

    if cloud_run_id:
        cmd.extend(["--id", cloud_run_id])

    return cmd


def is_backup_stable(run_dir: Path, settle_seconds: float = 1.0) -> bool:
    backup_file = run_dir / "backup.swanlab"
    if not backup_file.exists() or not backup_file.is_file():
        print(f"[WARN] backup file not found: {backup_file}")
        return False

    first = backup_file.stat()
    if first.st_size < 16:
        print(f"[INFO] backup file too small ({first.st_size} bytes), wait next round")
        return False

    time.sleep(settle_seconds)
    second = backup_file.stat()
    stable = (first.st_size == second.st_size) and (first.st_mtime_ns == second.st_mtime_ns)
    if not stable:
        print("[INFO] backup.swanlab is being written, skip this round")
    return stable


def get_backup_signature(run_dir: Path) -> tuple[int, int] | None:
    backup_file = run_dir / "backup.swanlab"
    if not backup_file.exists() or not backup_file.is_file():
        return None
    stat = backup_file.stat()
    return stat.st_size, stat.st_mtime_ns


def get_backup_idle_seconds(run_dir: Path) -> float | None:
    backup_file = run_dir / "backup.swanlab"
    if not backup_file.exists() or not backup_file.is_file():
        return None
    stat = backup_file.stat()
    return max(0.0, time.time() - stat.st_mtime)


def run_once(args: argparse.Namespace) -> int:
    cloud_run_id = read_cloud_run_id(args.run_dir)
    cmd = build_sync_cmd(args, cloud_run_id)

    if cloud_run_id:
        print(f"[INFO] resume cloud run id: {cloud_run_id}")

    print(f"[SYNC] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n", file=sys.stderr)

    # 首次成功同步后，记录云端 run id，后续用 --id 避免重复创建 experiment
    if proc.returncode == 0 and cloud_run_id is None:
        parsed_id = extract_cloud_run_id(proc.stdout, proc.stderr)
        if parsed_id:
            write_cloud_run_id(args.run_dir, parsed_id)
            print(f"[INFO] saved cloud run id: {parsed_id}")

    if proc.returncode != 0:
        merged = f"{proc.stdout}\n{proc.stderr}"
        if "LEVELDB_HEADER_LEN" in merged or "header is" in merged:
            print("[WARN] log file is mid-write/corrupted for this instant; will retry next interval")
        print(f"[WARN] sync failed with exit code {proc.returncode}")
    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Loop swanlab sync for a single run directory")
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest run-* directory under --swanlog-dir; if set, --run-dir is optional",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=False,
        help="Path to one SwanLab offline run directory (run-xxxx)",
    )
    parser.add_argument(
        "--swanlog-dir",
        type=Path,
        default=Path("./swanlog"),
        help="Root swanlog directory used by --latest (default: ./swanlog)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Sync interval in seconds (default: 10s, must be > 0)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Override workspace when syncing",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Override project when syncing",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one sync and exit",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help="How long to wait when checking backup.swanlab stability (default: 1.0)",
    )
    parser.add_argument(
        "--stop-stable-rounds",
        type=int,
        default=3,
        help="Auto-stop after backup.swanlab is unchanged for N rounds (default: 3)",
    )
    parser.add_argument(
        "--stop-idle-seconds",
        type=int,
        default=600,
        help="Auto-stop only when backup.swanlab has also been idle for at least this many seconds (default: 600)",
    )
    return parser.parse_args()


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.latest:
        if not args.swanlog_dir.exists() or not args.swanlog_dir.is_dir():
            raise FileNotFoundError(f"Swanlog directory not found: {args.swanlog_dir}")

        candidates = [
            path for path in args.swanlog_dir.glob("run-*")
            if path.is_dir()
        ]
        if not candidates:
            raise FileNotFoundError(f"No run-* directory found under: {args.swanlog_dir}")

        # 优先按目录名中的时间戳排序（run-YYYYMMDD_HHMMSS-xxxx）
        candidates.sort(key=lambda p: p.name)
        selected = candidates[-1]
        print(f"[INFO] auto selected latest run_dir: {selected}")
        return selected

    if args.run_dir is None:
        raise ValueError("--run-dir is required when --latest is not set")

    return args.run_dir


def main() -> None:
    args = parse_args()

    if args.interval <= 0:
        raise ValueError("--interval must be > 0")
    if args.stop_stable_rounds <= 0:
        raise ValueError("--stop-stable-rounds must be > 0")
    if args.stop_idle_seconds <= 0:
        raise ValueError("--stop-idle-seconds must be > 0")

    args.run_dir = resolve_run_dir(args)

    if not args.run_dir.exists() or not args.run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {args.run_dir}")

    if args.once:
        if is_backup_stable(args.run_dir, args.settle_seconds):
            run_once(args)
        else:
            print("[INFO] skip one-shot sync because backup file is not stable")
        return

    print(f"[INFO] run_dir={args.run_dir}")
    print(f"[INFO] interval={args.interval}s")
    print(f"[INFO] settle_seconds={args.settle_seconds}")
    print(f"[INFO] stop_stable_rounds={args.stop_stable_rounds}")
    print(f"[INFO] stop_idle_seconds={args.stop_idle_seconds}")

    last_signature: tuple[int, int] | None = None
    unchanged_rounds = 0
    synced_once = False

    while True:
        print(f"[INFO] {time.strftime('%F %T')} syncing...")
        if is_backup_stable(args.run_dir, args.settle_seconds):
            signature = get_backup_signature(args.run_dir)
            ret = run_once(args)
            if ret == 0:
                synced_once = True

            if signature is not None and signature == last_signature:
                unchanged_rounds += 1
            else:
                unchanged_rounds = 0
            last_signature = signature

            idle_seconds = get_backup_idle_seconds(args.run_dir)
            idle_ok = idle_seconds is not None and idle_seconds >= args.stop_idle_seconds

            if synced_once and unchanged_rounds >= args.stop_stable_rounds and idle_ok:
                print(
                    "[INFO] backup.swanlab has been unchanged for "
                    f"{unchanged_rounds} rounds and idle for {idle_seconds:.1f}s; "
                    "assume training finished, exiting sync loop"
                )
                break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
