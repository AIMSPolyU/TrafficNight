from ultralytics import YOLO
import cv2, csv
import os
from glob import glob
import numpy as np

class infor_cfg:
    model_path = '/usr/src/ultralytics/runs/obb/train/weights/best.pt'
    videos_dir = '/usr/src/TrafficNight'
    output_folder = '/usr/src/dataset/trafficnight/trackres'

class DeepTrafficInfer():
    def __init__(self) -> None:
        self.cfg = infor_cfg()
        self.model = YOLO(self.cfg.model_path)  # Load the best Detect model
        self.data_rows = []
        
        pass
    
    def trackAllvideos(self):
        '''
        track all videos
        '''
        video_paths = glob(os.path.join(self.cfg.videos_dir, '*.MP4'))  # 视频文件是mp4格式
            
        for video_path in video_paths:
            video_name = os.path.basename(video_path)
            print("starc track video : %s"% video_name)
            csv_name = os.path.splitext(video_name)[0] + '.csv'
            res_csv_path = os.path.join(self.cfg.output_folder, csv_name)
            if not os.path.exists(self.cfg.output_folder):
                os.makedirs(self.cfg.output_folder) 
            self.trackOffline(video_path)
            self.dumpCsv(res_csv_path)
            print("track res dump at : %s"% res_csv_path)
        

    def trackOffline(self, video_path):
        '''
        track offline and save resulat as csv
        csv: agent
        '''
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        cap.release()  # Close the video file as it's no longer needed here

        # Initialize a list to hold all rows of data to be written to the CSV
        self.data_rows = []
        # Perform tracking with the model
        results = self.model.track(source=video_path, persist=True,
                                   conf=0.5, iou=0.5,
                                    stream=True,
                                    device=0,
                                    stream_buffer=True,
                                    retina_masks=True,
                                    tracker="bytetrack.yaml")

        # Process results generator
        frame_index = 0
        for result in results:
            boxes = result.obb  # Boxes object for bbox outputs
            try:
                cls = boxes.cls.numpy().astype(int)  # Convert tensor to numpy array
                # TODO if agentID is None please use -1;
                angtid = boxes.id.numpy().astype(int)  # Convert tensor to numpy array
                xywhr = boxes.xywhr.numpy()  # Convert tensor to numpy array
                scene_timestep = np.full(cls.shape, frame_index / frame_rate)  # Create a full array with the scene timestep

                # Combine all data into a single numpy array for this frame
                frame_data = np.column_stack((angtid, cls, scene_timestep, xywhr[:, 0], xywhr[:, 1], xywhr[:, 2], xywhr[:, 3], xywhr[:,4]))
                self.data_rows.extend(frame_data.tolist())  # Add the data for this frame to our list
            except:
                print("find error in track task")
            # Increment the frame index after processing each frame
            frame_index += 1
    
    def dumpCsv(self, dump_path):
        '''
        Dump track result as csv file
        '''
        # Now write all the collected data to the CSV file
        # '/usr/src/DeepTraffic/track_results.csv'
        with open(dump_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['agent_id', 'class', 'scene_ts', 'u', 'v','w', 'h', 'r'])  # Write the header
            writer.writerows(self.data_rows)  # Write all data rows
        return dump_path
    

if __name__ == '__main__':
    dfi = DeepTrafficInfer()
    dfi.trackAllvideos()