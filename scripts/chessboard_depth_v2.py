import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
import cv2
import csv
import os
import sys

if len(sys.argv) < 2:
    print("ERROR: Please provide a label for this measurement")
    print("Usage: python3 chessboard_depth_v2.py <label>")
    print("Example: python3 chessboard_depth_v2.py 5m_center")
    sys.exit(1)

LABEL       = sys.argv[1]
BOARD_COLS  = 4
BOARD_ROWS  = 4
OUTPUT_DIR  = "/ws/scripts/data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

class ChessboardDepthSaver(Node):
    def __init__(self):
        super().__init__('chessboard_depth_saver')
        self.img_sub   = Subscriber(self, Image, '/multisense/left/image_rect')
        self.depth_sub = Subscriber(self, Image, '/multisense/left/depth')
        self.sync = ApproximateTimeSynchronizer(
            [self.img_sub, self.depth_sub],
            queue_size=10, slop=0.05)
        self.sync.registerCallback(self.callback)
        print(f"Label      : {LABEL}")
        print(f"Waiting for chessboard — hold it in front of the camera...")

    def callback(self, img_msg, depth_msg):
        img   = np.frombuffer(img_msg.data,   dtype=np.uint8  ).reshape(img_msg.height,   img_msg.width)
        depth = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width)
        timestamp = f"{img_msg.header.stamp.sec}_{img_msg.header.stamp.nanosec}"

        clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img)

        found   = False
        corners = None
        for (src, method, flags) in [
            (img,      "original", cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE),
            (enhanced, "enhanced", cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE),
            (img,      "SB",       cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE),
            (enhanced, "SB",       cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE),
        ]:
            if method == "SB":
                ret, c = cv2.findChessboardCornersSB(src, (BOARD_COLS, BOARD_ROWS), flags=flags)
            else:
                ret, c = cv2.findChessboardCorners(src, (BOARD_COLS, BOARD_ROWS), flags=flags)
            if ret:
                pts = c.reshape(-1, 2)
                w = pts[:,0].max() - pts[:,0].min()
                h = pts[:,1].max() - pts[:,1].min()
                if w < 10 or h < 10:
                    continue
                found   = True
                corners = c
                print(f"Board found [{method}]")
                break

        if not found:
            print("Not detected — retrying...")
            return

        pts = corners.reshape(-1, 2)
        x0, y0 = int(pts[:,0].min()), int(pts[:,1].min())
        x1, y1 = int(pts[:,0].max()), int(pts[:,1].max())

        depth_region = depth[y0:y1, x0:x1]
        valid = depth_region[np.isfinite(depth_region) & (depth_region > 0) & (depth_region < 30)]

        if len(valid) == 0:
            print("No valid depth in board region — retrying...")
            return

        mean_depth   = float(np.mean(valid))
        std_depth    = float(np.std(valid))
        min_depth    = float(valid.min())
        max_depth    = float(valid.max())
        median_depth = float(np.median(valid))

        print(f"\n--- BOARD FOUND ---")
        print(f"  Label        : {LABEL}")
        print(f"  Timestamp    : {timestamp}")
        print(f"  Top-left     : ({x0}, {y0})")
        print(f"  Bottom-right : ({x1}, {y1})")
        print(f"  Size         : {x1-x0} x {y1-y0} pixels")
        print(f"\n--- DEPTH STATS ---")
        print(f"  Valid pixels : {len(valid)} / {depth_region.size}")
        print(f"  Mean         : {mean_depth:.4f} m")
        print(f"  Std dev      : {std_depth:.4f} m")
        print(f"  Median       : {median_depth:.4f} m")
        print(f"  Min          : {min_depth:.4f} m")
        print(f"  Max          : {max_depth:.4f} m")

        # ── save CSV ─────────────────────────────────────────────
        csv_path = os.path.join(OUTPUT_DIR, f"{LABEL}_depth_{timestamp}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['# MEASUREMENT SUMMARY'])
            writer.writerow(['label',          LABEL])
            writer.writerow(['timestamp',      timestamp])
            writer.writerow(['top_left_x',     x0])
            writer.writerow(['top_left_y',     y0])
            writer.writerow(['bottom_right_x', x1])
            writer.writerow(['bottom_right_y', y1])
            writer.writerow(['valid_pixels',   len(valid)])
            writer.writerow(['mean_m',         f"{mean_depth:.4f}"])
            writer.writerow(['std_m',          f"{std_depth:.4f}"])
            writer.writerow(['median_m',       f"{median_depth:.4f}"])
            writer.writerow(['min_m',          f"{min_depth:.4f}"])
            writer.writerow(['max_m',          f"{max_depth:.4f}"])
            writer.writerow([])
            writer.writerow(['pixel_x', 'pixel_y', 'depth_m'])
            for row in range(depth_region.shape[0]):
                for col in range(depth_region.shape[1]):
                    d = depth_region[row, col]
                    if np.isfinite(d) and d > 0:
                        writer.writerow([x0+col, y0+row, f"{d:.4f}"])

        # ── save annotated image ─────────────────────────────────
        img_path = os.path.join(OUTPUT_DIR, f"{LABEL}_frame_{timestamp}.png")
        annotated = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(annotated, (BOARD_COLS, BOARD_ROWS), corners, True)
        cv2.rectangle(annotated, (x0,y0), (x1,y1), (0,255,0), 2)
        cv2.circle(annotated, (x0,y0), 5, (255,0,0), -1)
        cv2.putText(annotated, f"TL({x0},{y0})",
                    (x0+5, y0-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        cv2.circle(annotated, (x1,y1), 5, (255,0,0), -1)
        cv2.putText(annotated, f"BR({x1},{y1})",
                    (x1-80, y1+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        cv2.putText(annotated, f"Mean:{mean_depth:.3f}m Std:{std_depth:.3f}m",
                    (x0, y0-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(annotated, f"Label: {LABEL}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imwrite(img_path, annotated)

        print(f"\n--- SAVED ---")
        print(f"  CSV   : {csv_path}")
        print(f"  Image : {img_path}")

        rclpy.shutdown()

def main():
    rclpy.init()
    node = ChessboardDepthSaver()
    rclpy.spin(node)

main()
