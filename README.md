# Football Player Distance Tracking

## Overview
Detects and tracks players in a prerecorded football match video and computes the distance each player runs (in meters) using homography calibration.

## Project structure
- data/: put `football_match.mp4` and `field_points.json` here
- models/: put `yolov8n.pt` (or other YOLOv8) here
- src/: source files (detector, tracker, calibration, distance, visualizer, main)
- output/: tracked_video.mp4 and player_distances.csv will be saved here

## Setup
1. Create a virtual env:

2. Download a YOLOv8 model:
- Put `yolov8n.pt` or similar in `models/` (you can use ultralytics to download automatically too).

3. Prepare `data/field_points.json`
- Open a representative frame of your video, and annotate the four corners of the pitch visible in the frame.
- Replace the image_points in the JSON accordingly.
- world_points are in meters (example uses standard 105x68 pitch).

4. Put your prerecorded video at `data/football_match.mp4`.

## Run

## Notes & Tips
- If the camera is moving a lot (panning), consider performing frame-to-frame stabilization or estimating a global homography per frame.
- If the full pitch is not visible, you can still calibrate using visible field markings (penalty area corners etc) but accuracy will drop.
- For live camera feeds you may need a more advanced calibration pipeline.
