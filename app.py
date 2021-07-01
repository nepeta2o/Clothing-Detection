"""
modified from new_image_demo.py
"""
import torch
import os
import cv2
from yolo.utils.utils import *
from predictors.YOLOv3 import YOLOv3Predictor
# from predictors.DetectronModels import Predictor
import glob
from tqdm import tqdm
import sys

import base64
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


#YOLO PARAMS
yolo_df2_params = {   "model_def" : "yolo/df2cfg/yolov3-df2.cfg",
"weights_path" : "yolo/weights/yolov3-df2_15000.weights",
"class_path":"yolo/df2cfg/df2.names",
"conf_thres" : 0.5,
"nms_thres" :0.4,
"img_size" : 416,
"device" : device}

yolo_modanet_params = {   "model_def" : "yolo/modanetcfg/yolov3-modanet.cfg",
"weights_path" : "yolo/weights/yolov3-modanet_last.weights",
"class_path":"yolo/modanetcfg/modanet.names",
"conf_thres" : 0.5,
"nms_thres" :0.4,
"img_size" : 416,
"device" : device}


#DATASET
dataset = 'df2'


if dataset == 'df2': #deepfashion2
    yolo_params = yolo_df2_params

if dataset == 'modanet':
    yolo_params = yolo_modanet_params


#Classes
classes = load_classes(yolo_params["class_path"])

#Colors
cmap = plt.get_cmap("rainbow")
colors = np.array([cmap(i) for i in np.linspace(0, 1, 13)])
#np.random.shuffle(colors)


model = 'yolo'

if model == 'yolo':
    detectron = YOLOv3Predictor(params=yolo_params)
# else:
#     detectron = Predictor(model=model,dataset= dataset, CATEGORIES = classes)

#Faster RCNN / RetinaNet / Mask RCNN


##### Flask #####
app = Flask(__name__)

@app.route('/deepfashion2', methods=['POST'])
def classification():
    response = {}
    if 'application/json' in request.content_type:
        request_json = request.get_json()
        if request_json and 'image' in request_json:
            img = cv2.imdecode(np.frombuffer(base64.b64decode(request_json['image']), np.uint8), cv2.IMREAD_UNCHANGED)
            response = predict(img)
        else:
            response = {'error': {'message': "JSON is invalid, or missing a 'image' property"}}
    else:
        response = {'error': {'message': 'Invalid request'}}

    return jsonify(response)


def predict(img):
    result = {"result": []}
    detections = detectron.get_detections(img)
    if len(detections) != 0:
        detections.sort(reverse=False ,key = lambda x:x[4])
        for x1, y1, x2, y2, cls_conf, cls_pred in detections:
            result["result"].append({
                "bbox": [x1, y1, x2, y2],
                "cls_conf": cls_conf,
                "cls_pred": classes[int(cls_pred)]
            })

            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf))
            color = colors[int(cls_pred)]
            color = tuple(c*255 for c in color)
            color = (.7*color[2],.7*color[1],.7*color[0])
            font = cv2.FONT_HERSHEY_SIMPLEX

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            text =  "%s conf: %.3f" % (classes[int(cls_pred)] ,cls_conf)

            cv2.rectangle(img,(x1,y1) , (x2,y2) , color,3)
            y1 = 0 if y1<0 else y1
            y1_rect = y1-25
            y1_text = y1-5

            if y1_rect<0:
                y1_rect = y1+27
                y1_text = y1+20
            cv2.rectangle(img,(x1-2,y1_rect) , (x1 + int(8.5*len(text)),y1) , color,-1)
            cv2.putText(img,text,(x1,y1_text), font, 0.5,(255,255,255),1,cv2.LINE_AA)

    # cv2.imwrite('output/output-test.jpg', img)
    return result
