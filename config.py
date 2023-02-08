from pathlib import Path

HERE = Path(__file__).parent

MODEL_PATH = HERE / "./models/"


# This is only needed when hosting in local
# Download from: https://www.gyan.dev/ffmpeg/builds/
FFMPEG_PATH = HERE / "./ffmpeg/ffmpeg.exe"

DEBUG = False

STYLES = {
    "yolov7": "yolov7.pt",
    "yolov7x": "yolov7x.pt",
}

