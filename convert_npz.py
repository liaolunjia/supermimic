"""Convert LAFAN1 retargeting CSVs (30 fps) to the 50 fps npz files that
mjlab's MotionLoader expects, without uploading to the W&B artifact registry.

Upstream `mjlab/src/mjlab/scripts/csv_to_npz.py` always logs the motion as a
W&B artifact (needs an account/network). This wrapper reuses its MotionLoader
math (30 -> 50 fps interpolation + finite-difference velocities) and writes the
npz locally.

Usage (inside the container, from the supermimic dir):
  uv run python convert_npz.py --all
  uv run python convert_npz.py --clips run1_subject5 walk3_subject2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mjlab.scripts.csv_to_npz import MotionLoader

REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR.parent / "LAFAN1_Retargeting_Dataset" / "g1"
OUT_DIR = REPO_DIR.parent / "LAFAN1_motions"


def convert_one(csv_path: Path, out_path: Path, device: str) -> None:
  motion = MotionLoader(
    motion_file=str(csv_path),
    input_fps=30.0,
    output_fps=50.0,
    device=device,
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(out_path, **motion.arrays())
  print(f"[OK] {csv_path.name} -> {out_path} ({motion.output_frames} frames @ 50 fps)")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--clips", nargs="*", default=[], help="clip names without .csv")
  parser.add_argument("--all", action="store_true", help="convert every CSV in the dataset dir")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
  parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
  args = parser.parse_args()

  if args.all:
    clips = sorted(p.stem for p in args.data_dir.glob("*.csv"))
  else:
    clips = args.clips
  if not clips:
    parser.error("pass --all or --clips <name> [...]")

  for clip in clips:
    csv_path = args.data_dir / f"{clip}.csv"
    if not csv_path.is_file():
      print(f"[SKIP] missing {csv_path}")
      continue
    convert_one(csv_path, args.out_dir / f"{clip}.npz", args.device)


if __name__ == "__main__":
  main()
