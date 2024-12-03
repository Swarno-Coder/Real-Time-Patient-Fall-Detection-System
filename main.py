from mainalgo import run_pose_estimation
# argparser for source selection or webcam and flp option

run_pose_estimation(source='test/test4.mp4', flip=False, use_popup=True, draw=False, fps=60, out=True, output_video_path="demo/output2.avi")