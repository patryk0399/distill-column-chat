from __future__ import annotations

from pathlib import Path

from .config import load_config

"""
Using this to test configs on different computers.
"""
def main() -> None:
    cfg = load_config()
    data_dir = Path(cfg.data_dir)
    print("[check_env] ENV:", cfg.env)
    print("[check_env] Data dir:", data_dir, "exists:", data_dir.exists())


if __name__ == "__main__":
    main()