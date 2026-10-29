#!/usr/bin/env python3
"""
compute_board_center_R.py - MultiSense S27 depth validation (Take2)

Reads the depth CSV files written by chessboard_depth_v2.py and computes, for each
valid capture, the straight-line range R from the left camera optical centre to the
board centre.

Steps for each capture
  1. Board-centre pixel (u_c, v_c) = centre of the ROI in the CSV header:
        u_c = round((top_left_x + bottom_right_x) / 2)
        v_c = round((top_left_y + bottom_right_y) / 2)
  2. Z_c = mean of the valid depth values in the 5 x 5 window centred on (u_c, v_c)
     (valid = finite and 0 < Z < 30 m, same rule as the acquisition script)
  3. R = Z_c * sqrt(1 + ((u_c - cx)/fx)^2 + ((v_c - cy)/fy)^2)  ( = Z_c / cos(theta) )

Camera intrinsics (pixels, 960 x 600 rectified image) - defaults until the P matrix
is available:
  fx = fy = 600   (from the data: fx * B = 162.0 px*m, B = 0.27 m)
  cx = 480, cy = 300  (assumed image centre)

Usage
  python compute_board_center_R.py <csv_folder> <output_csv>
  python compute_board_center_R.py data/Take2_CSV output/take2_board_center_R.csv
  python compute_board_center_R.py data/Take2_CSV out.csv --fx 600 --fy 600 --cx 480 --cy 300

Requires: numpy   (pip install numpy)
"""
import argparse
import csv
import glob
import math
import os
import re

import numpy as np

# Captures that are NOT valid (false chessboard detection -> ROI includes background)
EXCLUDED = {"Take2_20m_left_bottom", "Take2_20m_center_bottom"}

HALF = 2  # 5 x 5 window = centre pixel +/- 2


def read_depth_csv(path):
    """Return (header dict, {(x, y): depth}) from one chessboard_depth_v2.py CSV."""
    header, depth = {}, {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r"^(\d+),(\d+),([-\d.eE+]+)$", line)
            if m:  # per-pixel row: pixel_x, pixel_y, depth_m
                depth[(int(m.group(1)), int(m.group(2)))] = float(m.group(3))
            elif "," in line and not line.startswith(("#", "pixel_x")):  # header row
                key, value = line.split(",", 1)
                header[key] = value
    return header, depth


def main():
    ap = argparse.ArgumentParser(description="Compute R at the board centre for each capture")
    ap.add_argument("csv_folder", help="folder with *_depth_*.csv files")
    ap.add_argument("output_csv", help="output CSV file")
    ap.add_argument("--fx", type=float, default=600.0)
    ap.add_argument("--fy", type=float, default=600.0)
    ap.add_argument("--cx", type=float, default=480.0)
    ap.add_argument("--cy", type=float, default=300.0)
    ap.add_argument("--prefix", default="Take2_",
                    help="only use files whose name starts with this (default: Take2_)")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.csv_folder, f"{a.prefix}*_depth_*.csv")))
    if not files:
        raise SystemExit(f"No *_depth_*.csv files found in {a.csv_folder}")

    rows = []
    for path in files:
        label = os.path.basename(path).split("_depth_")[0]   # e.g. Take2_10m_left_top
        if label in EXCLUDED:
            print(f"skip  {label} (excluded)")
            continue

        header, depth = read_depth_csv(path)
        x0, y0 = int(header["top_left_x"]), int(header["top_left_y"])
        x1, y1 = int(header["bottom_right_x"]), int(header["bottom_right_y"])
        u_c = int(math.floor((x0 + x1) / 2 + 0.5))   # round half up
        v_c = int(math.floor((y0 + y1) / 2 + 0.5))

        window = []
        for v in range(v_c - HALF, v_c + HALF + 1):
            for u in range(u_c - HALF, u_c + HALF + 1):
                z = depth.get((u, v))
                if z is not None and np.isfinite(z) and 0 < z < 30:
                    window.append(z)
        if not window:
            print(f"WARN  {label}: no valid depth in the 5x5 window")
            continue
        Z_c = float(np.mean(window))

        k = math.sqrt(1 + ((u_c - a.cx) / a.fx) ** 2 + ((v_c - a.cy) / a.fy) ** 2)
        theta_deg = math.degrees(math.acos(1 / k))
        R = Z_c * k

        rows.append({
            "label": label,
            "roi_x0": x0, "roi_y0": y0, "roi_x1": x1, "roi_y1": y1,
            "u_c": u_c, "v_c": v_c,
            "n_valid_5x5": len(window),
            "Z_center_mean_m": round(Z_c, 4),
            "Z_5x5_std_m": round(float(np.std(window)), 4),
            "theta_deg": round(theta_deg, 2),
            "k_ray_factor": round(k, 5),
            "R_center_m": round(R, 4),
            "fx": a.fx, "fy": a.fy, "cx": a.cx, "cy": a.cy,
        })
        print(f"ok    {label:26s} (u,v)=({u_c},{v_c})  Z={Z_c:7.3f} m  "
              f"theta={theta_deg:5.1f} deg  R={R:7.3f} m")

    os.makedirs(os.path.dirname(os.path.abspath(a.output_csv)), exist_ok=True)
    with open(a.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} captures written to {a.output_csv}")


if __name__ == "__main__":
    main()
