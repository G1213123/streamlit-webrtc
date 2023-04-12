from helpers.bridge_wrapper import  YOLOv7_Byte
import  helpers.detection_helpers as detection_helpers
import ffmpeg
import config
import  subprocess
from pathlib import Path
import os

BASE = Path(__file__).parent.parent
print(BASE)

# init object detector and tracker
detector = detection_helpers.Detector( 0.7 )
detector.load_model( BASE.joinpath ('weights/' + "yolov7-tiny.pt"), trace=False )
deepsort_tracker = YOLOv7_Byte( detector)

in_path= r"New video.mp4"
output_path = r"test1.mp4"

# encode cv2 output into h264
# https://stackoverflow.com/questions/30509573/writing-an-mp4-video-using-python-opencv
args = (ffmpeg
        .input( 'pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format( 1920, 1080 ) )
        .output( output_path, pix_fmt='yuv420p', vcodec='libx264', r=30, crf=12 )
        .overwrite_output()
        .get_args()
        )

# check if deployed on cloud or local host
ffmpeg_source = config.FFMPEG_PATH
process = subprocess.Popen( [ffmpeg_source] + args, stdin=subprocess.PIPE )

deepsort_tracker.track_video(in_path, output=process, show_live=True)