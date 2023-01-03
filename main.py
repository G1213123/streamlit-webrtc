
import logging
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple, Optional
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
    WebRtcStreamerContext,
    webrtc_streamer,
)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

FFMPEG_PATH = r"C:\software\ffmpeg\bin\ffmpeg"

CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


@st.experimental_singleton
def generate_label_colors():
    return np.random.uniform( 0, 255, size=(255, 3) )


COLORS = generate_label_colors()
DEFAULT_CONFIDENCE_THRESHOLD =0.5
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
    st.header("WebRTC demo")

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


def app_loopback():
    """Simple video loopback"""
    webrtc_streamer(key="loopback")

async def video_object_detection():
    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05, key='confidence_threshold'
    )
    file = st.file_uploader('Choose a video', type=['avi', 'mp4', 'mov'])
    if st.button( 'Detect' ):
        if file is not None:
            try:
                tfile = tempfile.NamedTemporaryFile( delete=False )
                tfile.write( file.read() )
                tfile.close()

                cap = cv2.VideoCapture(tfile.name)
                width, height = int( cap.get( cv2.CAP_PROP_FRAME_WIDTH ) ), int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
                fps = cap.get( cv2.CAP_PROP_FPS )

                if not os.path.exists( os.path.join( HERE,'storage') ):
                    os.makedirs(os.path.join( HERE,'storage') )
                output_path = os.path.join( HERE, f"storage\\{str( uuid.uuid4() )}.mp4" )
                fourcc = cv2.VideoWriter_fourcc( *'mp4v' )
                out = cv2.VideoWriter( output_path, fourcc, fps, (width, height) )
                net =  model_init()
                sort_tracker = app_object_track()
                while cap.isOpened():
                    try:
                        ret, frame = cap.read()
                        if not ret:
                            break
                    except Exception as e:
                        print( e )
                        continue

                    blob = cv2.dnn.blobFromImage(
                        cv2.resize( frame, (300, 300) ), 0.007843, (300, 300), 127.5
                    )
                    net.setInput( blob )
                    detections = net.forward()
                    # Update object localizer
                    annotated_image, result, track_str = annotate_image( frame, detections, confidence_threshold,sort_tracker )
                    out.write( annotated_image )
                    result_queue.put( result )
                    track_queue.put( track_str )

                cap.release()
                out.release()

                output_path_h264 = output_path.replace( '.mp4', '_h264.mp4' )

                # Encode video streams into the H.264
                os.system( '{} -i {} -vcodec libx264 {}'.format( FFMPEG_PATH, output_path, output_path_h264 ) )
                os.remove( output_path )

                tfile.close()
                st.video( output_path_h264 )

            except Exception as e:
                return {"message": "There was an error processing the file\n" + str( e )}

    if st.checkbox( "Show the detected labels", value=True ):
        labels_placeholder = st.empty()
        track_placeholder = st.empty()

        while True:
            try:
                result = result_queue.get( timeout=1.0 )
                track_list = track_queue.get( timeout=1.0 )
            except queue.Empty:
                result = None
                track_list = None
            labels_placeholder.table( result )
            track_placeholder.table( None if track_list is None else track_list.split() )


async def live_object_detection():
    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05, key = 'confidence_threshold'
    )

    #public-stun-list.txt
    #https://gist.github.com/mondain/b0ec1cf5f60ae726202e
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.ucsb.edu:3478"]}]}
    )

    net = model_init()
    sort_tracker = app_object_track()

    def frame_callback(frame: av.VideoFrame, ) -> av.VideoFrame:
        image = frame.to_ndarray( format="bgr24" )
        blob = cv2.dnn.blobFromImage(
            cv2.resize( image, (300, 300) ), 0.007843, (300, 300), 127.5
        )
        net.setInput( blob )
        detections = net.forward()
        annotated_image, result, track_str = annotate_image( image, detections,  confidence_threshold, sort_tracker )

        # NOTE: This `recv` method is called in another thread,
        # so it must be thread-safe.
        result_queue.put( result )
        track_queue.put( track_str )

        return av.VideoFrame.from_ndarray( annotated_image, format="bgr24" )


    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
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


def model_init():
    """Object detection demo with MobileNet SSD.
    This model and code are based on
    https://github.com/robmarkcole/object-detection-app
    """
    MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
    MODEL_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.caffemodel"
    PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
    PROTOTXT_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.prototxt.txt"

    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
    download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)

    # Session-specific caching
    cache_key = "object_detection_dnn"
    if cache_key in st.session_state:
        net = st.session_state[cache_key]
    else:
        net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH))
        st.session_state[cache_key] = net
    print(st.session_state[cache_key])
    return net

def app_object_track():
    # .... Initialize SORT ....
    # .........................
    sort_max_age = 5
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort( max_age=sort_max_age,
                         min_hits=sort_min_hits,
                         iou_threshold=sort_iou_thresh )

    return sort_tracker

class Detection(NamedTuple):
    name: str
    prob: float

def annotate_image(image, detections, confidence_threshold, sort_tracker = None ):

    # loop over the detections
    (h, w) = image.shape[:2]
    result: List[Detection] = []
    track_result = []
    txt_str = ""

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            name = CLASSES[idx]
            result.append(Detection(name=name, prob=float(confidence)))

            try:
               dets_to_sort
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
                [cv2.line( image, (int( track.centroidarr[i][0] ),
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
                    (w, h), _ = cv2.getTextSize( label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1 )
                    cv2.rectangle( image, (x1, y1), (x2, y2), COLORS[track.id], 2 )
                    #cv2.rectangle( image, (x1, y1 - 20), (x1 + w, y1), (255, 144, 30), -1 )
                    cv2.putText( image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                 0.6, COLORS[track.id], 1 )
                    # cv2.circle(img, data, 6, color,-1)   #centroid of box

                    txt_str += "%i %i %f %f %f %f %f %f" % (
                        id, cat, int( box[0] ) / image.shape[1], int( box[1] ) / image.shape[0],
                        int( box[2] ) / image.shape[1],
                        int( box[3] ) / image.shape[0], int( box[0] + (box[2] * 0.5) ) / image.shape[1],
                        int( box[1] + (
                                box[3] * 0.5) ) / image.shape[0])
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
