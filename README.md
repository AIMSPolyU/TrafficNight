<p align="center">
  <img src="assets\websiteH.jpg" alt="trafficnight Logo", width="94%">
</p>

<h1 align="center">TrafficNight: An Aerial Multimodal Benchmark For Nighttime Vehicle Surveillance</h1>

## What's New

To address these limitations and explore the relationship between human driving behavior and traffic conditions, we present the TrafficNight dataset. 
This comprehensive dataset, collected using drones, includes:

## Data
1. [ Video ] -- Aerial footage from a vertical perspective. (mp4)
2. [ Image&Label ] -- Images & annotations for training vehicle object detection model. (jpg, json)
3. [ OpenDriver File ] -- High-definition map data of the captured area. (.xord)

## Toolkit
5. Vehicle object detection model we have trained.
6. Convert coordinates on the image to longitude and latitude coordinates.
7. Temperature color palette and label mapping tool.


<!-- <p align="center">
  <a >
      <img src="Source/WorkFlow.png" width="74%" height="12%">
  </a>
</p> -->


# Manually Download
### (1) Image and Json Label Data
- [ FIR OBB Detection](https://drive.google.com/file/d/1ge9FXrhaSszlEOVQdR1CQxc0nDVh1QJ5/view?usp=sharing)
- [ sRGB OBB Detection](https://drive.google.com/file/d/1CfLUlaJ9N2HdfEi3qEZ8Sfa0aUJw9rR9/view?usp=sharing)

### (2) Raw Videos
Access Password
`trafficnight`
- [TN1 <34.7G>](https://gofile.me/7rhQP/GXYC18Riy) 
- [TN2 <21.9G>](https://gofile.me/7rhQP/x4S5pXuCv)
- [TN3 <24.8G>](https://gofile.me/7rhQP/MS5okJAtz)
- [TN4 <24.7G>](https://gofile.me/7rhQP/ccX8kmyp0)
- [TN5 <27.8G>](https://gofile.me/7rhQP/qKsOkQgDe)
- [TN6 <23.6G>](https://gofile.me/7rhQP/0cfDnwweS)
- [TN7 <25.7G>](https://gofile.me/7rhQP/nYblEYIoa)


### (3) HD-MAP files (opendrive.xord/apollo.bin):
- [HDMAP Files](https://drive.google.com/drive/folders/1AfnrxAYxN7FFATbxkeWEHJ_hQYc3aKI3?usp=sharing)

### (4) Pre-Train Model:
- [Yolov8m-FIR](https://drive.google.com/file/d/1NIM3Dma9rouvxNsTV2d5865xmIJLEFfg/view?usp=drive_link)
- [Yolov8m-sRGB](https://drive.google.com/file/d/1tYXZneBFwLfrHk2JLmTZi7RaJum61JnG/view?usp=drive_link)

# Tookit Env
Recommend running the following command script for automatic data download and extraction into the current project. 

- Clone Git Project
```
git clone https://github.com/AIMSPolyU/TrafficNight.git
```
- Build Docker Image
```
cd trafficnight
docker build -t aims/trafficnight .
```

- Create Container
```
docker run -it --ipc=host -v $(PWD)\TrafficNight:/usr/src/TrafficNight --gpus all aims/trafficnight:latest

```


<!-- - Download Data By Shell (you must have more than 150Gb Free space in your disk)
```
# download video
bash download_data.sh -n video

# download obb labels
bash download_data.sh -n obb

# download hdmap data
bash download_data.sh -n hdmap
``` -->

# Tutorial
## (1) Tracking on Video
we use [ultralytics](https://github.com/ultralytics/ultralytics) to train yolov8 (object detection and tracking)

download [demo video](https://drive.google.com/file/d/1t_AkifkttbO8gXXFTGW0FZWin8TJOUk9/view?usp=drive_link) 
```
python trackAsVideo.py --input_video /usr/src/TrafficNight/DJI_20231026220911_0002_T.MP4 --output_video /usr/src/TrafficNight/track_output.mp4 --model /usr/src/ultralytics/runs/obb/train/weights/best.pt
```

## (2) Training Object Detection
we use [ultralytics](https://github.com/ultralytics/ultralytics) to train yolov8 (object detection and tracking)
```
python train.py
```

## (3) Track on video
```
python trackAsVideo.py --input_video <path_to_video> --output_video <path_to_output_video>  --model /usr/src/TrafficNight/weights/yolov8m-obb/best.pt --output_csv <save_track_result>
```

- **--input_video**:
  - Path to the input video file that will be processed for object tracking.
  
  - **Example**:  
    `/path/to/input/video.mp4`

- **--output_video**:
  - Path to save the output video file, which will contain object annotations (bounding boxes) based on YOLOv8 tracking results.
  
  - **Example**:  
    `/path/to/output/video.mp4`

- **--model**:
  - Path to the YOLOv8 model file used for object detection and tracking. The default path is `/usr/src/TrafficNight/weights/yolov8m-obb/best.pt`, but it can be replaced with any compatible YOLOv8 model.
  
  - **Example**:  
    `/usr/src/TrafficNight/weights/yolov8m-obb/best.pt`

- **--output_csv**:
  - Path to save the CSV file that contains tracking data such as object IDs, class labels, and bounding box coordinates for each frame.
  
  - **Example**:  
    `/usr/src/TrafficNight/trackRes/TN03_DJI_20231028195825_0001_T_24hz.csv`


## (4) Mapping Toolkit
<!-- ### [Temperature mapping tool](Doc/TrackingYolo8.md) -->

### 4.1 Convert UV-Coordinates (image) To Physic World
```
python toolkit/Get3DPose.py --video_json <path_to_video_json> --dsm_path <path_to_dsm_file> --ort_path <path_to_ortho_image> --ort_json <path_to_ortho_json> --track_csv <path_to_trackAsVideo_result>
```
- **--video_json**:
  - Path to the JSON file containing metadata for the input video. This file typically includes reference point UV coordinates. It is used to match the reference points in `ort_json` and calculate the camera's external pose parameters.

  - Example:
  `'/usr/src/TrafficNight/TN_RawVedio/TN03/TN10281958.json'`

- **--dsm_path**:
  - Path to the DSM (Digital Surface Model) file, which contains elevation data for the terrain in the video. This file is used to improve the accuracy of 3D pose calculations by incorporating terrain height information.

  - Example:
  `'/usr/src/TrafficNight/Physics/TN03_dsm_20cm.tif'`

- **--ort_path**:
  - Path to the orthophoto (ortho image) file, a georeferenced aerial image that provides a 2D spatial view of the video area, aligned with geographic features.

  - Example:
  `'/usr/src/TrafficNight/Physics/TN03_ort.tif'`

- **--ort_json**:
  - Path to the JSON metadata file for the orthophoto. This file contains the UV coordinates of the ground reference point, and the script will obtain the longitude and latitude information based on this UV coordinate.

  - Example:
  `'/usr/src/TrafficNight/Physics/TN03_ort.json'`

- **--track_csv**:
  - Path to the csv file containing yolo track result. 

  - Example:
  `'/usr/src/TrafficNight/trackRes/TN03_DJI_20231028195825_0001_T_24hz_with_enu.csv'`



<p align="center">
  <img src="assets\Track3D.gif" alt="Mapping 2D UV to 3D", width="80%">
</p>


# Citation
<!-- **Download the Paper**: [Download PDF](link_to_your_paper) -->

If you use DeepTraffic Dataset in your research , please use the following BibTeX entry.
```BibTeX
@InProceedings{Zhang24,
      title = {TrafficNight: An Aerial Multimodal Benchmark For Nighttime Vehicle Surveillance},
      author = {Zhang, Guoxing and Liu, Yiming and Yang, Xiaoyu and Huang, Chao and  Huang, Hailong},
      booktitile = {ECCV},
      year = {2024}

}
```

<!-- # New Project 

For related papers and resources, please refer to the [References section](link_to_references) in the repository. -->



# License
This dataset is released under the TrafficNight. Please review the [LICENSE file](http://www.apache.org/licenses/LICENSE-2.0) in the repository for details.