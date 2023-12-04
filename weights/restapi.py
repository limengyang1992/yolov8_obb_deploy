"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
import io
import json

import time
from PIL import Image
from flask import Flask, request
from detect import DetectMultiBackend,select_device,run


imgsz=(1280, 1280)
device = select_device("cpu")
model = DetectMultiBackend('weights/best.pt', device=device)
model.model.float()
model.warmup(imgsz=(1, 3, *imgsz), half=False) 

app = Flask(__name__)
DETECTION_URL = "/api/v1.0/detect"

@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        start = time.time()
        image_file = request.files["image"]
        image_bytes = image_file.read()
        source = "imgs/image.jpg"
        Image.open(io.BytesIO(image_bytes)).save(source)
        results = run(model,source,device)
        print(time.time()-start)
        return json.dumps(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=9000, type=int, help="port number")
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
