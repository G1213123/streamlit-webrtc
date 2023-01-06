import YOLOv7.utils
import config
import inference

import logging
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple
import requests
import tarfile

import pandas as pd

from sort import Sort
import tempfile
import uuid
import asyncio

import av
import cv2
import numpy as np

import streamlit as st

from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    webrtc_streamer,
)

logger = logging.getLogger(__name__)

CLASSES = YOLOv7.utils.class_names


@st.experimental_singleton
def generate_label_colors():
    return np.random.uniform( 0, 255, size=(65536, 3) )


COLORS = generate_label_colors()
DEFAULT_CONFIDENCE_THRESHOLD =0.5

#TODO:update model download function
#url = "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/307_YOLOv7/no-postprocess/resources.tar.gz"
#with open(local_filename, 'wb') as f:
#    r = requests.get(url, stream=True)
#    for chunk in r.raw.stream(1024, decode_content=False):
#        if chunk:
#            f.write(chunk)
#            f.flush()
##file = tarfile.open(fileobj=response.raw, mode="r|gz")
#file.extractall(path=".")

# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

async def main():
    st.header("Object Tracking demo")

    pages = {
        "Real time object detection (sendrecv)": live_object_detection,
        "Upload Video for detection": video_object_detection,
    }
    page_titles = pages.keys()

    page_title = st.sidebar.selectbox(
        "Choose the app mode",
        page_titles,
    )
    st.subheader(page_title)

    page_func = pages[page_title]
    await page_func()

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")

def variables_container():
    with st.container():
        confidence_threshold = st.slider(
            "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05, key='confidence_threshold'
        )
        st.caption( 'SORT Tracking Algorithm see: https://github.com/abewley/sort' )
        track_age = st.slider(
            "Tracking Age (frames)", 0, 20, 10, 1, key='track_age'
        )
        track_hits = st.slider(
            "Tracking hits", 0, st.session_state.track_age, 6, 1, key='track_age'
        )
        iou_thres = st.slider(
            "IOU threshold", 0.0, 1.0, 0.7, 0.1, key='iou_thres'
        )
    return confidence_threshold, track_age, track_hits, iou_thres

def app_loopback():
    """Simple video loopback"""
    webrtc_streamer(key="loopback")

async def video_object_detection():
    #usable video for detection
    #https://www.pexels.com/video/aerial-footage-of-vehicular-traffic-of-a-busy-street-intersection-at-night-3048225/

    confidence_threshold, track_age, track_hits, iou_thres = variables_container()

    track_list = []
    result_list = []
    style = st.selectbox( 'Choose the model', [i for i in config.STYLES.keys()] )
    file = st.file_uploader('Choose a video', type=['avi', 'mp4', 'mov'])
    if st.button( 'Detect' ):
        if file is not None:
            progress_bar = st.progress(0)
            progress=0

            tfile = tempfile.NamedTemporaryFile( delete=False )
            tfile.write( file.read() )
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            width, height = int( cap.get( cv2.CAP_PROP_FRAME_WIDTH ) ), int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
            fps = cap.get( cv2.CAP_PROP_FPS )

            if not os.path.exists( os.path.join( config.HERE,'storage') ):
                os.makedirs(os.path.join( config.HERE,'storage') )
            output_path = os.path.join( config.HERE, f"storage\\{str( uuid.uuid4() )}.mp4" )
            fourcc = cv2.VideoWriter_fourcc( *'mp4v' )
            out = cv2.VideoWriter( output_path, fourcc, fps, (width, height) )
            detector =  model_init(style, confidence_threshold)
            sort_tracker = app_object_track(track_age, track_hits,iou_thres)
            while cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if not ret:
                        break
                except Exception as e:
                    print( e )
                    continue
                detections =  detector(frame)
                # Update object localizer
                image, result,track_str = annotate_image( frame, detections,sort_tracker,progress )
                out.write( image )
                result_list.append( result )
                track_list.append( track_str )

                #progress of analysis
                progress +=1
                progress_bar.progress(progress/int(cap.get(cv2.CAP_PROP_FRAME_COUNT))*0.8)

            cap.release()
            out.release()

            output_path_h264 = output_path.replace( '.mp4', '_h264.mp4' )

            # Encode video streams into the H.264
            os.system( '{} -i {} -vcodec libx264 {}'.format( config.FFMPEG_PATH, output_path, output_path_h264 ) )
            tfile.close()
            st.video( output_path_h264 )
            os.remove( output_path )
            progress_bar.progress(100)


                # labels_placeholder = st.empty()
            try:
                track_list= [item.strip() for sublist in [element.split( "\n" ) for element in track_list] for item in sublist]
                track_list = [e.split() for e in track_list if e != '']
                track_table = pd.DataFrame( track_list,
                                            columns=['frame', 'id', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'xmid',
                                                     'ymid'] )
                # track_table.insert(loc=2, column='type', value=track_table['class'].apply(lambda x:CLASSES[x]))
                # labels_placeholder.table( result_list )
                st.table( track_table )
            except:
                'No tracking data found'


async def live_object_detection():
    confidence_threshold, track_age, track_hits, iou_thres = variables_container()

    #public-stun-list.txt
    #https://gist.github.com/mondain/b0ec1cf5f60ae726202e
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.ucsb.edu:3478"]}]}
    )

    style = st.selectbox( 'Choose the model', [i for i in config.STYLES.keys()] )

    detector = model_init(style, confidence_threshold)
    sort_tracker = app_object_track(track_age, track_hits, iou_thres)

    def frame_callback(frame: av.VideoFrame, ) -> av.VideoFrame:
        image = frame.to_ndarray( format="bgr24" )
        #blob = cv2.dnn.blobFromImage(
        #    cv2.resize( image, (size, size) ), 0.007843, (size, size), 127.5
        #)
        detections =  detector(image)
        annotated_image, result, track_str = annotate_image( image, detections, sort_tracker )

        # NOTE: This `recv` method is called in another thread,
        # so it must be thread-safe.
        result_queue.put( result )
        track_queue.put( track_str )

        return av.VideoFrame.from_ndarray( annotated_image, format="bgr24" )


    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        #rtc_configuration=RTC_CONFIGURATION, #when deploy on remote host need stun server for camera connection
        video_frame_callback=frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if st.checkbox( "Show the detected labels", value=True ):
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
            track_placeholder = st.empty()
            # NOTE: The video transformation with object detection and
            # this loop displaying the result labels are running
            # in different threads asynchronously.
            # Then the rendered video frames and the labels displayed here
            # are not strictly synchronized.
            while True:
                try:
                    result = result_queue.get( timeout=1.0 )
                    track_list = track_queue.get( timeout=1.0 )
                except queue.Empty:
                    result = None
                    track_list = None
                labels_placeholder.table( result )
                track_placeholder.table( None if track_list is None else track_list.split() )

    st.markdown(
        "This demo uses a model and code from "
        "https://github.com/robmarkcole/object-detection-app. "
        "Many thanks to the project."
    )


def model_init(model='yolov7-nms-640', confidence_threshold = 0.5):
    """Object detection demo with YOLO v7.
    This model and code are based on
    https://github.com/WongKinYiu/yolov7/releases
    """

    MODEL_LOCAL_PATH = config.MODEL_PATH / f'{config.STYLES[model]}.onnx'
    if not Path(MODEL_LOCAL_PATH).exists():
        download_file(config.MODEL_URL_ROOT + config.STYLES[model], MODEL_LOCAL_PATH, expected_size=721000)

    # Session-specific caching
    cache_key = "object_detection_dnn"
    if cache_key in st.session_state:
        detector = st.session_state[cache_key]
    else:
        detector = inference.init(model, conf_thres=confidence_threshold)
        st.session_state[cache_key] = detector
    print(st.session_state[cache_key])
    return detector

def app_object_track(sort_max_age =5, sort_min_hits = 2, sort_iou_thresh = 0.9):
    # .... Initialize SORT ....
    # .........................
    sort_tracker = Sort( max_age=sort_max_age,
                         min_hits=sort_min_hits,
                         iou_threshold=sort_iou_thresh )

    return sort_tracker

class Detection(NamedTuple):
    name: str
    prob: float

def annotate_image(image, detections,  sort_tracker = None, frame = None ):

    # loop over the detections
    (h, w) = image.shape[:2]
    result: List[Detection] = []
    track_result = []
    txt_str = ""

    if frame:
        cv2.putText( image, f'frame:{frame}', (40,40), cv2.FONT_HERSHEY_SIMPLEX,
                     1, (240,240,240), 2 )

    boxes, confidences, ids = detections
    if len(detections[0]) > 0:
        for box, confidence, idx in zip(boxes, confidences, ids):

            (startX, startY, endX, endY) = box.astype("int")

            name = CLASSES[idx]
            result.append(Detection(name=name, prob=float(confidence)))

            try:
               _ = dets_to_sort
            except UnboundLocalError:
               dets_to_sort = np.empty( (0, 6) )
            # NOTE: We send in detected object class too
            dets_to_sort = np.vstack( (dets_to_sort,
                                       np.array( [startX, startY, endX, endY, confidence, idx] )) )

            # Run SORT
            if sort_tracker is not None:
                tracked_dets = sort_tracker.update( dets_to_sort )
                tracks = sort_tracker.getTrackers()

            # loop over tracks
            for track in tracks:
                # color = compute_color_for_labels(id)
                # draw colored tracks
                drawn_track = [cv2.line( image, (int( track.centroidarr[i][0] ),
                                 int( track.centroidarr[i][1] )),
                           (int( track.centroidarr[i + 1][0] ),
                            int( track.centroidarr[i + 1][1] )),
                           COLORS[track.id], thickness=2 )
                 for i, _ in enumerate( track.centroidarr )
                 if i < len( track.centroidarr ) - 1]

            # draw boxes for visualization
            if len( tracked_dets ) > 0:
                bbox_xyxy = tracked_dets[:, :4]
                identities = tracked_dets[:, 8]
                categories = tracked_dets[:, 4]

                for i, box in enumerate( bbox_xyxy ):
                    x1, y1, x2, y2 = [int( i ) for i in box]
                    cat = int( categories[i] ) if categories is not None else 0
                    id = int( identities[i] ) if identities is not None else 0
                    data = (int( (box[0] + box[2]) / 2 ), (int( (box[1] + box[3]) / 2 )))
                    label = str( id ) + ":" + CLASSES[cat] + "-" + str(confidence)
                    (w, h), _ = cv2.getTextSize( label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2 )
                    cv2.rectangle( image, (x1, y1), (x2, y2), COLORS[track.id], 2 )
                    #cv2.rectangle( image, (x1, y1 - 20), (x1 + w, y1), COLORS[track.id], -1 )
                    cv2.putText( image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                 1, COLORS[track.id], 2 )
                    # cv2.circle(img, data, 6, color,-1)   #centroid of box

                    txt_str += "%s%i %i %f %f %f %f %f %f" % (str(frame) + ' ' if frame is not None else '',
                        id, cat, int( box[0] ) , int( box[1] ) ,
                        int( box[2] ) ,
                        int( box[3] ) , int( box[0] + (box[2] * 0.5) ) ,
                        int( box[1] + (
                                box[3] * 0.5) ) )
                    txt_str += "\n"
                    track_result.append(txt_str)




    return image, result, txt_str

result_queue = (
    queue.Queue()
)  # TODO: A general-purpose shared state object may be more useful.
track_queue = (
    queue.Queue()
)



if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    loop = asyncio.new_event_loop()
    loop.run_until_complete( main() )
