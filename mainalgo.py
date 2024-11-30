import notebook_utils as utils
from openposedecoder import OpenPoseDecoder

decoder = OpenPoseDecoder()

import numpy as np
from numpy.lib.stride_tricks import as_strided
# 2D pooling in numpy (from: https://stackoverflow.com/a/54966908/1624463)
def pool2d(A, kernel_size, stride, padding, pool_mode="max"):
    """
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    """
    # Padding
    A = np.pad(A, padding, mode="constant")

    # Window view of A
    output_shape = (
        (A.shape[0] - kernel_size) // stride + 1,
        (A.shape[1] - kernel_size) // stride + 1,
    )
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(
        A,
        shape=output_shape + kernel_size,
        strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides,
    )
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling.
    if pool_mode == "max":
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == "avg":
        return A_w.mean(axis=(1, 2)).reshape(output_shape)


# non maximum suppression
def heatmap_nms(heatmaps, pooled_heatmaps):
    return heatmaps * (heatmaps == pooled_heatmaps)

import openvino.properties.hint as hints
import openvino as ov

model_path = "model\intel\human-pose-estimation-0001\FP16-INT8\human-pose-estimation-0001.xml"  # Replace with the path to your OpenVINO model

# Initialize OpenVINO Runtime
core = ov.Core()
# Read the network from a file.
model = core.read_model(model_path)
# Let the AUTO device decide where to load the model (you can use CPU, GPU as well).
compiled_model = core.compile_model(model=model, device_name='CPU', config={hints.performance_mode(): hints.PerformanceMode.LATENCY})

# Get the input and output names of nodes.
input_layer = compiled_model.input(0)
output_layers = compiled_model.outputs

# Get the input size.
height, width = list(input_layer.shape)[2:]

# Get poses from results.
def process_results(img, pafs, heatmaps):
    # This processing comes from
    # https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/python/models/open_pose.py
    pooled_heatmaps = np.array([[pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") for h in heatmaps[0]]])
    nms_heatmaps = heatmap_nms(heatmaps, pooled_heatmaps)

    # Decode poses.
    poses, scores = decoder(heatmaps, nms_heatmaps, pafs)
    output_shape = list(compiled_model.output(index=0).partial_shape)
    output_scale = (
        img.shape[1] / output_shape[3].get_length(),
        img.shape[0] / output_shape[2].get_length(),
    )
    # Multiply coordinates by a scaling factor.
    poses[:, :, :2] *= output_scale
    return poses, scores

colors = (
    (255, 0, 0),
    (255, 0, 255),
    (170, 0, 255),
    (255, 0, 85),
    (255, 0, 170),
    (85, 255, 0),
    (255, 170, 0),
    (0, 255, 0),
    (255, 255, 0),
    (0, 255, 85),
    (170, 255, 0),
    (0, 85, 255),
    (0, 255, 170),
    (0, 0, 255),
    (0, 255, 255),
    (85, 0, 255),
    (0, 170, 255),
)

default_skeleton = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)

import cv2
def draw_poses(img, poses, point_score_threshold, skeleton=default_skeleton):
    if poses.size == 0:
        return img

    img_limbs = np.copy(img)
    for pose in poses:
        points = pose[:, :2].astype(np.int32)
        points_scores = pose[:, 2]
        # Draw joints.
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img, tuple(p), 1, colors[i], 2)
        # Draw limbs.
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                cv2.line(
                    img_limbs,
                    tuple(points[i]),
                    tuple(points[j]),
                    color=colors[j],
                    thickness=4,
                )
    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    return img

def process_poses(poses, img_width, img_height, point_score_threshold=0.1):
    """
    Process raw pose data to extract and analyze keypoints for fall detection.

    Args:
        poses: Detected poses in float16 format, shape (N, 17, 3).
        img_height: Height of the frame for normalization.
        point_score_threshold: Minimum score to consider a keypoint valid.

    Returns:
        Keypoint dictionary containing normalized and filtered keypoints.
    """
    keypoints_dict = {}
    for pose in poses:
        for idx, (x, y, score) in enumerate(pose):
            if score > point_score_threshold:
                # Normalize y-coordinates relative to image height
                keypoints_dict[idx] = {"x": x/img_width, "y": y, "y_norm": y / img_height, "score": score}
    return keypoints_dict
import numpy as np
import cv2

def detect_fall_from_keypoints(keypoints_dict):
    """
    Detect fall based on processed keypoints.

    Args:
        keypoints_dict: Dictionary of valid keypoints.

    Returns:
        Boolean indicating if a fall is detected.
    """
    if not keypoints_dict:
        return False

    # Extract relevant keypoints (use COCO index for body parts)
    head = keypoints_dict.get(0)
    left_hip = keypoints_dict.get(11)
    right_hip = keypoints_dict.get(12)
    left_knee = keypoints_dict.get(13)
    right_knee = keypoints_dict.get(14)

    if all([head, left_hip, right_hip, left_knee, right_knee]):
        # Fall detection conditions:
        # 1. Head close to ground
        if head["y_norm"] > 0.7:
            # 2. Hips aligned horizontally
            if abs(left_hip["y"] - right_hip["y"]) < 20:
                # 3. Knees close to hips vertically
                if abs(left_knee["y_norm"] - left_hip["y_norm"]) < 0.2 and abs(right_knee["y_norm"] - right_hip["y_norm"]) < 0.2:
                    print("Alert: Fall Detected!")
                    return True

    return False


import collections, time
# Main processing function to run pose estimation.
def run_pose_estimation(source=0, flip=False, fps=30, use_popup=False, skip_first_frames=0,w=1280,h=720):
    pafs_output_key = compiled_model.output("Mconv7_stage2_L1")
    heatmaps_output_key = compiled_model.output("Mconv7_stage2_L2")
    player = None
    try:
        # Create a video player to play with target fps.
        player = utils.VideoPlayer(source, flip=flip, fps=fps, skip_first_frames=skip_first_frames,width=w, height=h)
        # Start capturing.
        player.start()
        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

        processing_times = collections.deque()

        while True:
            # Grab the frame.
            frame = player.next()
            if frame is None:
                print("Source ended")
                break
            # If the frame is larger than full HD, reduce size to improve the performance.
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # Resize the image and change dims to fit neural network input.
            # (see https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/human-pose-estimation-0001)
            input_img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            # Create a batch of images (size = 1).
            input_img = input_img.transpose((2, 0, 1))[np.newaxis, ...]

            # Measure processing time.
            start_time = time.time()
            # Get results.
            results = compiled_model([input_img])
            stop_time = time.time()

            pafs = results[pafs_output_key]
            heatmaps = results[heatmaps_output_key]
            # Get poses from network results.
            poses, scores = process_results(frame, pafs, heatmaps)
            # print(poses)
            # Draw poses on a frame.
            frame = draw_poses(frame, poses, 0.1)
            kp = process_poses(poses,width,height)
            detect_fall_from_keypoints(kp)
            
            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # mean processing time [ms]
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            # print(fps, processing_time)
            cv2.putText(
                frame,
                f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                (20, 40),
                cv2.FONT_HERSHEY_COMPLEX,
                f_width / 1000,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

            # Use this workaround if there is flickering.
            if use_popup:
                cv2.imshow(title, frame)
                key = cv2.waitKey(1)
                # escape = 27
                if key == 27:
                    break
            else:
                # Encode numpy array to jpg.
                # _, encoded_img = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 90])
                cv2.imshow("processd image", frame)
                
    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            # Stop capturing.
            player.stop()
        if use_popup:
            cv2.destroyAllWindows()
            
            
if __name__ == "__main__":
    
    run_pose_estimation(source='test/test4.mp4', flip=False, use_popup=True, fps=60)