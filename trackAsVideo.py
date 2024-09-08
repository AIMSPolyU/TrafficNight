import argparse
import csv
import cv2
import numpy as np
from ultralytics import YOLO

def adjust_video_frame_rate(input_video_path, output_video_path, target_fps=24):
    """
        Adjusts the frame rate of the input video and saves the result to the output path.
    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the output video file with the adjusted frame rate.
        target_fps (int, optional): Desired frames per second for the output video. Defaults to 24.
    """
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (width, height))
    
    frame_interval = original_fps / target_fps
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval < 1:
            out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    print(f"Save VIDEO ON: {output_video_path}")

def process_video(input_video_path, output_video_path, model_path, output_csv_path):
    """
    Processes a video by applying a YOLOv8 model for object tracking, 
    annotating the video with bounding boxes, and saving the results to a CSV file.

    Args:
        input_video_path (str): Path to the input video file to be processed.
        output_video_path (str): Path to save the annotated video with bounding boxes.
        model_path (str): Path to the YOLOv8 model used for object tracking.
        output_csv_path (str): Path to save the CSV file containing tracking data.
    """

    # Load the YOLOv8 model
    yolo_model = YOLO(model_path)

    # Open the video file
    video_capture = cv2.VideoCapture(input_video_path)

    # Determine the video's width, height, and frame rate
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object to write our output video
    codec = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    video_writer = cv2.VideoWriter(output_video_path, codec, frame_rate, (frame_width, frame_height))

    # Prepare to write CSV output
    csv_header = ['agent_id', 'class', 'scene_ts', 'u', 'v', 'width', 'height', 'rotation']
    data_rows = []

    frame_index = 0

    # Open the CSV file for writing
    with open(output_csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_header)  # Write the CSV header

        # Loop through the video frames
        while video_capture.isOpened():
            # Read a frame from the video
            success, frame = video_capture.read()

            if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = yolo_model.track(frame, persist=True,
                                           conf=0.8,
                                           device=0,
                                           retina_masks=True,
                                           tracker="bytetrack.yaml")

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Write the annotated frame to the output video file
                video_writer.write(annotated_frame)

                # Extract tracking data and append to CSV file
                boxes = results[0].obb  # Extract bounding box results
                # print('boxes:', boxes)
                try:
                    cls = boxes.cls.numpy().astype(int)  # Convert tensor to numpy array for class IDs
                    angtid = boxes.id.numpy().astype(int)  # Convert tensor to numpy array for agent IDs
                    xywhr = boxes.xywhr.numpy()  # Convert tensor to numpy array for bounding boxes (x, y, w, h, r)
                    scene_timestep = np.full(cls.shape, frame_index / frame_rate)  # Create a full array with the scene timestep

                    # Combine all data into a single numpy array for this frame
                    frame_data = np.column_stack((angtid, cls, scene_timestep, xywhr[:, 0], xywhr[:, 1], xywhr[:, 2], xywhr[:, 3], xywhr[:, 4]))
                    csv_writer.writerows(frame_data.tolist())  # Write the data for this frame to the CSV file
                except Exception as e:
                    print(f"Error processing frame {frame_index}: {e}")

                # Increment the frame index after processing each frame
                frame_index += 1
            else:
                # Break the loop if the end of the video is reached
                break

    # Release the video capture, video writer object, and close all windows
    video_capture.release()
    video_writer.release()

if __name__ == "__main__":
    '''
    python process_video.py --input_video /path/to/input/video.mp4 --output_video /path/to/output/video.mp4 --model /usr/src/TrafficNight/weights/yolov8m-obb/best.pt --output_csv /usr/src/TrafficNight/trackRes/TN03_DJI_20231028195825_0001_T_24hz.csv

    model='/usr/src/TrafficNight/weights/yolov8m-obb/best.pt' 
    input_video= '/usr/src/TrafficNight/TN03_DJI_20231028195825_0001_T_24hz.MP4'
    output_video =  '/usr/src/TrafficNight/TN03_DJI_20231028195825_0001_T_24hz_trackvis.MP4' 
    output_csv = '/usr/src/TrafficNight/trackRes/TN03_DJI_20231028195825_0001_T_24hz.csv'
    process_video2(input_video, output_video, model, output_csv)
    '''

    parser = argparse.ArgumentParser(description="Process video with YOLOv8 tracking and save output.")
    parser.add_argument('--input_video', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output_video', type=str, required=True, help='Path to save the output video file.')
    parser.add_argument('--model', type=str, default='/usr/src/TrafficNight/weights/yolov8m-obb/best.pt', help='Path to the YOLOv8 model file.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save the output video file.')
    parser.add_argument('--trackfps', type=str, required=False)
    args = parser.parse_args()
    process_video(args.input_video, args.output_video, args.model, args.output_csv)
