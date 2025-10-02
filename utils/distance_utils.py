import numpy as np

# Example calibration: 1 pixel ≈ 0.02 meters
PIXEL_TO_METER = 0.02

def calculate_distance(path_points):
    distance = 0
    for i in range(1, len(path_points)):
        x1, y1 = path_points[i - 1]
        x2, y2 = path_points[i]
        distance += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance * PIXEL_TO_METER
