# Yolov8 for Oriented Object Detection

# Build

t=ultralytics/yolov8_obb:latest && sudo docker build -t $t .



# Pull and Run

sudo docker run -it -v ./imgs:/usr/src/app/imgs -p 9000:9000 --ipc=host  $t


