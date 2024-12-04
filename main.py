import argparse
from mainalgo import run_pose_estimation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run pose estimation on a video source.')
    parser.add_argument('--source', "-s", type=str, required=True, help='Path to the video source')
    parser.add_argument('--flip', action='store_true', help='Flip the video horizontally')
    parser.add_argument('--use_popup', action='store_true', help='Use popup window for display')
    parser.add_argument('--draw', action='store_true', help='Draw the pose estimation on the video')
    parser.add_argument('--fps', type=int, default=60, help='Frames per second for the output video')
    parser.add_argument('--out', action='store_true', help='Output the processed video')
    parser.add_argument('--output_video_path', type=str, help='Path to save the output video')

    args = parser.parse_args()

    run_pose_estimation(
        source=args.source,
        flip=args.flip,
        use_popup=args.use_popup,
        draw=args.draw,
        fps=args.fps,
        out=args.out,
        output_video_path=args.output_video_path
    )