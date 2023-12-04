# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import (non_max_suppression_obb,scale_polys)
from utils.torch_utils import select_device
from utils.rboxs_utils import rbox2poly


def cut(poly,im0s):
    corners = np.array([[poly[0],poly[1]],[poly[2],poly[3]],[poly[4],poly[5]],[poly[6],poly[7]]])
    # 获取旋转矩形的四个角点
    box = np.int0(corners)
    src_pts = box.astype(np.float32)
    w = np.sqrt(np.sum((src_pts[0] - src_pts[1]) ** 2))
    h = np.sqrt(np.sum((src_pts[1] - src_pts[2]) ** 2))
    size = (int(w), int(h))
    # 计算透视变换矩阵
    dst_pts = np.array([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(im0s, M, (size[0], size[1]))
    return result

@torch.no_grad()
def run(model=None,  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        device=None,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        imgsz=(1280, 1280),  # inference size (height, width)
        conf_thres=0.56,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        half=False,  # use FP16 half-precision inference
        ):
    
    source = str(source)
    # Load model
    
    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=model.stride, auto=model.pt)
    
    
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference
        pred = model(im, augment=augment, visualize=False)
        # NMS
        pred = non_max_suppression_obb(pred, conf_thres, iou_thres, classes, agnostic_nms, multi_label=True, max_det=max_det)
        # pred_poly = rbox2poly(det[:, :5],det[:, 6:7]) # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
        pred_poly = rbox2poly(pred[0][:, :5]) # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
        
        det = pred[0]

        print(len(det))
        # Rescale polys from img_size to im0 size
        pred_poly = scale_polys(im.shape[2:], pred_poly, im0s.shape)
        
        det = torch.cat((pred_poly, det[:, -2:]), dim=1) # (n, [poly conf cls])\
        # 过滤置信度低的框
        det = det[det[:, 8] > conf_thres]
        
        # 转化为 json 格式
        json_data = []
        for i,(*xyxy, conf, cls) in enumerate(reversed(det)):
            poly = [x.item() for x in xyxy]
            
            # poly剪裁
            # cut_img = cut(poly,im0s)
            # cv2.imwrite(f'imgs_cut/im_{i}_{conf}.jpg', cut_img)
            
            json_data.append({
                "poly": poly,
                "conf": conf.item(),
                "cls": int(cls.item())
            })
            
        return json_data
                



if __name__ == "__main__":
    imgsz=(1280, 1280)
    source = 'imgs/1.jpg'
    device = select_device("cpu")
    
    model = DetectMultiBackend('weights/best.pt', device=device)
    model.model.float()
    model.warmup(imgsz=(1, 3, *imgsz), half=False) 
    
    json_data = run(model,source,device)
    print(json_data)

