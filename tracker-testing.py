import torch
import torchvision
import cv2
import numpy as np
from pathlib import Path
from boxmot import BotSort
import pyzed.sl as sl  # Import the ZED SDK
import argparse

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

def get_detector(detector_type, device):
    if detector_type == 'fasterrcnn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif detector_type == 'maskrcnn':
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    elif detector_type == 'retinanet':
        model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")
    model.eval().to(device)
    return model

def main(args):
    # Load the detector
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    detector = get_detector(args.detector, device)

    # Initialize the tracker
    tracker = BotSort(
        reid_weights=Path('osnet_x0_25_msmt17.pt'),  # Path to ReID model
        device=device,  # Use GPU for inference
        half=False
    )
    print('Detector initialized', detector)
    print('Tracker initialized', tracker)

    # Initialize the ZED camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Set resolution (HD720, HD1080, etc.)
    init_params.camera_fps = 60  # Set FPS
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.coordinate_units = sl.UNIT.METER

    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening ZED camera: {status}")
        exit(1)

    runtime_params = sl.RuntimeParameters()
    left_image = sl.Mat()
    right_image = sl.Mat()

    # Initialize variables for FPS calculation
    prev_time = cv2.getTickCount()
    fps = 0

    while True:
        # Grab a frame from the ZED camera
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left and right images in BGR format
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            zed.retrieve_image(right_image, sl.VIEW.RIGHT)
            left_frame = left_image.get_data()
            right_frame = right_image.get_data()
            left_frame = cv2.cvtColor(left_frame, cv2.COLOR_RGBA2RGB)  # Convert to RGB format as OpenCV expects BGR
            right_frame = cv2.cvtColor(right_frame, cv2.COLOR_RGBA2RGB)  # Convert to RGB format as OpenCV expects BGR

            # Convert frame to tensor and move to device
            frame_tensor = torchvision.transforms.functional.to_tensor(left_frame).to(device)

            # Perform detection
            with torch.no_grad():
                detections = detector([frame_tensor])[0]

            # Filter the detections (e.g., based on confidence threshold)
            confidence_threshold = 0.5
            dets = []
            for i, score in enumerate(detections['scores']):
                if score >= confidence_threshold:
                    bbox = detections['boxes'][i].cpu().numpy()
                    label = detections['labels'][i].item()
                    conf = score.item()
                    dets.append([*bbox, conf, label])

            # Convert detections to numpy array (N X (x, y, x, y, conf, cls))
            dets = np.array(dets)

            # Update the tracker
            res = tracker.update(dets, left_frame)  # --> M X (x, y, x, y, id, conf, cls, ind)

            # Plot tracking results on the left image
            tracker.plot_results(left_frame, show_trajectories=False)

            # Calculate FPS
            current_time = cv2.getTickCount()
            time_diff = (current_time - prev_time) / cv2.getTickFrequency()
            fps = 1 / time_diff
            prev_time = current_time

            # Display FPS on the left frame
            cv2.putText(left_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Concatenate left and right frames horizontally
            combined_frame = np.hstack((left_frame, right_frame))

            # Display the combined frame
            cv2.imshow('BoXMOT + Torchvision', combined_frame)

            # Simulate wait for key press to continue, press 'q' to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    # Close the ZED camera
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', type=str, default='fasterrcnn', help='Type of detector to use: fasterrcnn, maskrcnn, retinanet')
    args = parser.parse_args()
    main(args)