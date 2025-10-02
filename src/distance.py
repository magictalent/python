# src/distance.py
import numpy as np
from collections import defaultdict
import math

class DistanceCalculator:
    def __init__(self, calibrator):
        """
        calibrator: FieldCalibrator instance that provides pixel_to_world
        """
        self.calibrator = calibrator
        # store last known world positions per player id
        self.last_pos = {}
        # store total distance (meters)
        self.total_distance = defaultdict(float)

    def update_distance(self, track_id, center_pixel):
        """
        track_id: unique id
        center_pixel: (cx, cy) center of bbox in pixel coords
        returns: cumulative distance in meters for this track_id (float)
        """
        cx, cy = center_pixel
        wx, wy = self.calibrator.pixel_to_world(cx, cy)

        if track_id in self.last_pos:
            lx, ly = self.last_pos[track_id]
            # Euclidean distance in meters
            d = math.hypot(wx - lx, wy - ly)
            # small jitter protection: ignore extremely large jumps (camera cut)
            if d < 20:  # 20 meters per frame is unreasonable; adjust as needed
                self.total_distance[track_id] += d
        else:
            d = 0.0

        self.last_pos[track_id] = (wx, wy)
        return self.total_distance[track_id]
