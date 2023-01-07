from pathlib import Path

HERE = Path(__file__).parent

MODEL_PATH = HERE / "./models/"

# https://github.com/PINTO0309/PINTO_model_zoo
MODEL_URL = "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/307_YOLOv7/no-postprocess/resources.tar.gz"

# This is only needed when hosting in local
# Download from: https://www.gyan.dev/ffmpeg/builds/
FFMPEG_PATH = HERE / "./ffmpeg/ffmpeg.exe"

DEBUG = False

STYLES = {
    "yolov7_256x320": "yolov7_256x320",
    "yolov7_256x480": "yolov7_256x480",
    "yolov7_256x640": "yolov7_256x640",
    "yolov7_384x640": "yolov7_384x640",
    "yolov7_480x640": "yolov7_480x640",
    "yolov7_640x640": "yolov7_640x640",
    "yolov7_736x1280": "yolov7_736x1280",
    "yolov7-tiny_256x320": "yolov7-tiny_256x320",
    "yolov7-tiny_256x480": "yolov7-tiny_256x480",
    "yolov7-tiny_256x640": "yolov7-tiny_256x640",
    "yolov7-tiny_384x640": "yolov7-tiny_384x640",
    "yolov7-tiny_480x640": "yolov7-tiny_480x640",
    "yolov7-tiny_640x640": "yolov7-tiny_640x640",
    "yolov7-tiny_736x1280": "yolov7-tiny_736x1280",
}

