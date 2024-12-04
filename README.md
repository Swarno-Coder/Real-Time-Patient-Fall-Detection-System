# Real-Time-Patient-Monitoring-Dashboard

This repo is for Real-Time Patient Monitoring Dashboard paper submission in INDICON 2024.

## Pose Estimation

This project runs pose estimation on a video source using command-line arguments.

### Requirements

- Python 3.x
- Required Python packages (install using `pip install -r requirements.txt`)

### Usage

To run the pose estimation, use the following command:

```
python main.py --source <video_source> [options]

Arguments:
--source, -s (required): Path to the video source.
--flip: Flip the video horizontally.
--use_popup: Use a popup window for display.
--draw: Draw the pose estimation on the video.
--fps: Frames per second for the output video (default: 60).
--out: Output the processed video.
--output_video_path: Path to save the output video.

Example:
python main.py --source test/test4.mp4 --use_popup --fps 60 --out --output_video_path demo/output2.avi

or

python main.py -s test/test4.mp4 --use_popup --fps 60 --out --output_video_path demo/output2.avi
```

### Function

The main function used for pose estimation is `run_pose_estimation`, which is imported from `mainalgo`.

```python
from mainalgo import run_pose_estimation
```

### License

This project is licensed under the MIT License.