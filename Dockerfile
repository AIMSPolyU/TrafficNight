# docker file for ultralytics yolov8
FROM ultralytics/ultralytics:latest

# Set the working directory
WORKDIR /usr/src/

# Install 
RUN apt-get install -y p7zip-full exiftool\
    && pip install gdown rasterio affine pyproj