import numpy as np

class Track:
    def __init__(self, detection, track_id):
        self.track_id = track_id
        self.detections = [detection]
        self.hits = 1
        self.missed = 0
        self._confirmed = False

    def predict(self):
        pass  # Kalman filter normally goes here

    def update(self, detection):
        self.detections.append(detection)
        self.hits += 1
        self.missed = 0
        if self.hits >= 3:
            self._confirmed = True

    def mark_missed(self):
        self.missed += 1

    def is_confirmed(self):
        return self._confirmed

    def is_deleted(self):
        return self.missed > 30

    def to_ltrb(self):
        return self.detections[-1].to_tlbr()
