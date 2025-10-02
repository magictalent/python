# src/calibration.py
import json
import numpy as np
import cv2

class FieldCalibrator:
    def __init__(self, points_json="data/field_points.json"):
        with open(points_json, "r") as f:
            data = json.load(f)
        self.image_points = np.array(data["image_points"], dtype=np.float32)
        self.world_points = np.array(data["world_points"], dtype=np.float32)
        # Compute homography from image -> world (meters)
        self.h, _ = cv2.findHomography(self.image_points, self.world_points)
        # Also compute inverse if needed
        self.h_inv, _ = cv2.findHomography(self.world_points, self.image_points)

    def pixel_to_world(self, x, y):
        """
        Convert pixel coordinates (x,y) to world coords in meters using homography.
        Returns (wx, wy)
        """
        pt = np.array([ [x, y] ], dtype=np.float32)
        res = cv2.perspectiveTransform(np.array([pt]), self.h)[0][0]
        return float(res[0]), float(res[1])

    def world_to_pixel(self, wx, wy):
        pt = np.array([ [wx, wy] ], dtype=np.float32)
        res = cv2.perspectiveTransform(np.array([pt]), self.h_inv)[0][0]
        return int(res[0]), int(res[1])
