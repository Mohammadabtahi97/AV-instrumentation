import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
import cv2
import csv
import os

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
        print(f"Waiting for chessboard...")
        print(f"Hold the board in front of the camera")

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

        print(f"\n--- BOARD FOUND ---")
        print(f"  Timestamp    : {timestamp}")
        print(f"  Top-left     : ({x0}, {y0})")
        print(f"  Bottom-right : ({x1}, {y1})")
        print(f"  Size         : {x1-x0} x {y1-y0} pixels")

        print(f"\n--- CORNERS ---")
        for i, corner in enumerate(corners):
            cx, cy = int(corner[0][0]), int(corner[0][1])
            print(f"  corner {i:>2} : ({cx}, {cy})")

        depth_region = depth[y0:y1, x0:x1]
        valid = depth_region[np.isfinite(depth_region) & (depth_region > 0) & (depth_region < 30)]

        print(f"\n--- DEPTH STATS ---")
        if len(valid) > 0:
            print(f"  Valid pixels : {len(valid)} / {depth_region.size}")
            print(f"  Distance     : {np.median(valid):.3f} m")
            print(f"  Min          : {valid.min():.3f} m")
            print(f"  Max          : {valid.max():.3f} m")
            print(f"  Std dev      : {valid.std():.3f} m")
        else:
            print(f"  No valid depth in board region")

        csv_path = os.path.join(OUTPUT_DIR, f"depth_{timestamp}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['pixel_x', 'pixel_y', 'depth_m'])
            for row in range(depth_region.shape[0]):
                for col in range(depth_region.shape[1]):
                    d = depth_region[row, col]
                    if np.isfinite(d) and d > 0:
                        writer.writerow([x0+col, y0+row, f"{d:.4f}"])

        img_path = os.path.join(OUTPUT_DIR, f"frame_{timestamp}.png")
        annotated = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(annotated, (BOARD_COLS, BOARD_ROWS), corners, True)
        cv2.rectangle(annotated, (x0,y0), (x1,y1), (0,255,0), 2)
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
