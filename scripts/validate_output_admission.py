"""Compare local Output files with git staging representation."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "Output"


def main() -> int:
    local = set()
    for dp, _, fns in os.walk(OUTPUT):
        for fn in fns:
            rel = str((Path(dp) / fn).relative_to(OUTPUT)).replace("\\", "/")
            local.add(rel)

    proc = subprocess.run(
        ["git", "add", "-n", "Output"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    would_add = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("add "):
            continue
        if "'" in line:
            p = line.split("'", 2)[1]
            if p.startswith("Output/"):
                would_add.add(p[len("Output/") :])

    tracked = set(
        subprocess.run(
            ["git", "ls-files", "Output"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.splitlines()
    )
    for t in tracked:
        if t.startswith("Output/"):
            tracked.add(t[len("Output/") :])
    tracked_norm = {t.replace("Output/", "") if t.startswith("Output/") else t for t in tracked}

    represented = would_add | tracked_norm
    missing = sorted(local - represented)
    extra = sorted(represented - local)

    print(f"local_files={len(local)}")
    print(f"would_stage={len(would_add)}")
    print(f"already_tracked={len(tracked_norm)}")
    print(f"represented={len(represented)}")
    print(f"missing_from_git={len(missing)}")
    if missing:
        print("missing_sample:")
        for m in missing[:30]:
            print(f"  {m}")
    if extra:
        print(f"extra_in_git={len(extra)}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
