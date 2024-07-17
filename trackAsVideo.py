import cv2
from ultralytics import YOLO
import argparse

def process_video(input_video_path, output_video_path, model_path):
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

    # Loop through the video frames
    while video_capture.isOpened():
        # Read a frame from the video
        success, frame = video_capture.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = yolo_model.track(frame, persist=True,
                                       conf=0.9,
                                       device=0,
                                       retina_masks=True,
                                       tracker="bytetrack.yaml")

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Write the annotated frame to the output video file
            video_writer.write(annotated_frame)
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture, video writer object, and close all windows
    video_capture.release()
    video_writer.release()

if __name__ == "__main__":
    '''
    python process_video.py --input_video /path/to/input/video.mp4 --output_video /path/to/output/video.mp4 --model
    '''

    parser = argparse.ArgumentParser(description="Process video with YOLOv8 tracking and save output.")
    parser.add_argument('--input_video', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output_video', type=str, required=True, help='Path to save the output video file.')
    parser.add_argument('--model', type=str, default='/usr/src/TrafficNight/weights/yolov8m-obb/best.pt', help='Path to the YOLOv8 model file.')

    args = parser.parse_args()

    process_video(args.input_video, args.output_video, args.model)
