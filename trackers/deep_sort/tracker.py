from .track import Track
from . import linear_assignment
import numpy as np

class Tracker:
    def __init__(self, metric):
        self.metric = metric
        self.tracks = []
        self._next_id = 1

    def predict(self):
        for track in self.tracks:
            track.predict()

    def update(self, detections):
        matches, unmatched_tracks, unmatched_dets = linear_assignment.match(self.tracks, detections, self.metric)

        for t, d in matches:
            self.tracks[t].update(detections[d])

        for t in unmatched_tracks:
            self.tracks[t].mark_missed()

        for d in unmatched_dets:
            self._initiate_track(detections[d])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def _initiate_track(self, detection):
        track = Track(detection, self._next_id)
        self.tracks.append(track)
        self._next_id += 1
