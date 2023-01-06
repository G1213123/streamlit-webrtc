# backend/inference.py

import uuid

import os
import config
import cv2
from YOLOv7 import YOLOv7



def init(model, conf_thres=0.3, iou_thres=0.5):
    model_name = os.path.join(os.path.abspath(config.MODEL_PATH),f"{model}.onnx")

    # Initialize YOLOv7 object detector
    yolov7_detector = YOLOv7(model_name, conf_thres, iou_thres)

    return yolov7_detector

def inference_image(detector, image):
    boxes, scores, class_ids = detector(image)
    output = detector.draw_detections(image)

    return output

def inference_video(detector, image):
    cap = cv2.VideoCapture(image)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path =  os.path.join(os.path.dirname(__file__),f"storage/{str(uuid.uuid4())}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                break
        except Exception as e:
            print(e)
            continue

        # Update object localizer
        combined_img = inference_image(detector, frame)

        out.write(combined_img)

    cap.release()
    out.release()

    output_path_h264 = output_path.replace('.mp4', '_h264.mp4')

    # Encode video streams into the H.264
    os.system('{} -i {} -vcodec libx264 {}'.format(config.FFMPEG_PATH, output_path, output_path_h264))
    os.remove(output_path)

    return output_path_h264, width, height