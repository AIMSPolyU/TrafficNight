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
<!-- - [ sRGB OBB Detection](https://drive.google.com/file/d/1ge9FXrhaSszlEOVQdR1CQxc0nDVh1QJ5/view?usp=sharing) -->

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
- [Yolov8n](https://drive.google.com/file/d/1NIM3Dma9rouvxNsTV2d5865xmIJLEFfg/view?usp=drive_link)
- [Yolov8m](https://drive.google.com/file/d/1NIM3Dma9rouvxNsTV2d5865xmIJLEFfg/view?usp=drive_link) 


# Tookit
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

## (3) Prediction on image or dir
```
python predict.py
```
<!-- - [Convert UV-Coordinates (image) To Physic World](Doc/TrainingYolo8Seg.md) -->

<!-- ## (3) Mapping Toolkit

### [Temperature mapping tool](Doc/TrackingYolo8.md)

### [Convert UV-Coordinates (image) To Physic World](Doc/TrackingYolo8.md) -->


# Citation
<!-- **Download the Paper**: [Download PDF](link_to_your_paper) -->

If you use DeepTraffic Dataset in your research , please use the following BibTeX entry.
<!-- ```BibTeX
@InProceedings{Zhang24,
      title = {TrafficNight: An Aerial Multimodal Benchmark For Nighttime Vehicle Surveillance},
      author = {Zhang, Guoxing and Liu, Yiming and Yang, Xiaoyu and Huang, Chao and  Huang, Hailong},
      booktitile = {ECCV},
      year = {2024}

}
``` -->

<!-- # New Project 

For related papers and resources, please refer to the [References section](link_to_references) in the repository. -->



# License
This dataset is released under the TrafficNight. Please review the [LICENSE file](http://www.apache.org/licenses/LICENSE-2.0) in the repository for details.