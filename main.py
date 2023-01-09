import config
import inference
from YOLOv7.utils import class_names

import platform
import logging
import queue
import urllib.request
from pathlib import Path
from typing import List, NamedTuple
import tarfile
import ffmpeg
import subprocess
import pandas as pd
from sort import Sort
import tempfile
import uuid
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    webrtc_streamer,
)

st.set_page_config( layout="wide" )
logger = logging.getLogger( __name__ )


@st.experimental_singleton
def generate_label_colors():
    return np.random.uniform( 0, 255, size=(65536, 3) )


COLORS = generate_label_colors()


def color_row(s):
    return COLORS[s.name]


# Dump queue for real time detection result
result_queue = (queue.Queue())


class Detection( NamedTuple ):
    # Store detected object
    frame: int
    id: int
    type: str
    prob: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    xmid: float
    ymid: float


class frame_counter_class():
    # Use for counting elapsed frame in the live stream
    def __init__(self):
        self.frame = 0

    def __call__(self, count=0):
        self.frame += count
        return self.frame


class st_variables_container:
    # Container for the variable sliders listening user input
    def __init__(self):
        with st.container():
            st.selectbox( 'Choose the [detection model](https://github.com/WongKinYiu/yolov7)',
                          list( config.STYLES.keys() ), key='model_style' )
            st.slider(
                "Confidence threshold", 0.0, 1.0, 0.5, 0.05, key='confidence_threshold'
            )
            st.caption( '[SORT](https://github.com/abewley/sort) Tracking Algorithm' )
            st.slider(
                "Tracking Age (frames)", 0, 20, 10, 1, key='track_age'
            )
            st.slider(
                "Tracking hits", 0, st.session_state.track_age, 3, 1, key='tracking_hits'
            )
            st.slider(
                "IOU threshold", 0.0, 1.0, 0.7, 0.1, key='iou_thres'
            )

    def get_var(self):
        return st.session_state['model_style'], \
            st.session_state['confidence_threshold'], st.session_state['track_age'], \
            st.session_state['tracking_hits'], st.session_state['iou_thres']

@st.experimental_memo
def gcd(a, b):
    if b == 0:
        return a
    return gcd( b, a % b )

@st.experimental_memo
def best_match_ratio(w, h, style_list):
    diff = float( 'inf' )
    match = ''
    for s in style_list:
        s_h, s_w = list( map( int, s.split( '_' )[1].split( 'x' ) ) )
        if abs( w / s_w - h / s_h ) < diff:
            diff = abs( w / s_w - h / s_h )
            match = s
    return match

@st.experimental_memo
def download_file(url, download_to: Path, expected_size=None):
    """
    This code is based on
    https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
    """
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info( f"{url} is already downloaded." )
            if not st.button( "Download again?" ):
                return

    download_to.parent.mkdir( parents=True, exist_ok=True )

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning( "Downloading %s..." % url )
        progress_bar = st.progress( 0 )

        with open( download_to, "wb" ) as output_file:
            with urllib.request.urlopen( url ) as response:
                length = int( response.info()["Content-Length"] )
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read( 131072 )
                    if not data:
                        break
                    counter += len( data )
                    output_file.write( data )

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress( min( counter / length, 1.0 ) )
        file = tarfile.open( name=output_file.name, mode="r|gz" )
        file.extractall( path=download_to.parent )
        file.close()
        os.remove( output_file.name )
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

@st.experimental_singleton
def model_init(model, confidence_threshold=0.5):
    """Object detection demo with YOLO v7.
    This model and code are based on
    https://github.com/WongKinYiu/yolov7/releases
    """

    MODEL_LOCAL_PATH = config.MODEL_PATH / f'{config.STYLES[model]}.onnx'
    if not Path( MODEL_LOCAL_PATH ).exists():
        download_file( config.MODEL_URL, Path( MODEL_LOCAL_PATH ).parent / "resources.tar.gz",
                       expected_size=1007618059 )

    # Session-specific caching
    cache_key = "object_detection_dnn"
    if cache_key in st.session_state:
        detector = st.session_state[cache_key]
    else:
        detector = inference.init( model, conf_thres=confidence_threshold )
        st.session_state[cache_key] = detector
    print( st.session_state[cache_key] )
    return detector


def annotate_image(image, detections, sort_tracker, frame=None):
    # loop over the detections
    (h, w) = image.shape[:2]
    result: List[Detection] = []

    if frame:
        cv2.putText( image, f'frame:{frame}', (40, 40), cv2.FONT_HERSHEY_SIMPLEX,
                     1, (240, 240, 240), 2 )

    boxes, confidences, ids = detections

    dets_to_sort = np.empty( (0, 6) )

    for box, confidence, idx in zip( boxes, confidences, ids ):
        (startX, startY, endX, endY) = box.astype( "int" )

        # NOTE: We send in detected object class too
        dets_to_sort = np.vstack( (dets_to_sort,
                                   np.array( [startX, startY, endX, endY, confidence, idx] )) )

    # Run SORT

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
                                 COLORS[track.id + 1], thickness=2 )
                       for i, _ in enumerate( track.centroidarr )
                       if i < len( track.centroidarr ) - 1]

    # draw boxes for visualization
    if len( tracked_dets ) > 0:
        bbox_xyxy = tracked_dets[:, :4]
        identities = tracked_dets[:, 9]
        conf = tracked_dets[:, 5]
        categories = tracked_dets[:, 4]

        for i, box in enumerate( bbox_xyxy ):
            x1, y1, x2, y2 = [int( i ) for i in box]
            cat = int( categories[i] ) if categories is not None else 0
            id = int( identities[i] ) if identities is not None else 0
            data = (int( (box[0] + box[2]) / 2 ), (int( (box[1] + box[3]) / 2 )))
            label = str( id ) + ":" + class_names[cat] + "-" + "%.2f" % conf[i]
            (w, h), _ = cv2.getTextSize( label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2 )
            cv2.rectangle( image, (x1, y1), (x2, y2), COLORS[id], 2 )
            # cv2.rectangle( image, (x1, y1 - 20), (x1 + w, y1), COLORS[track.id], -1 )
            cv2.putText( image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                         1, COLORS[id], 2 )
            # cv2.circle(img, data, 6, color,-1)   #centroid of box

            result.append( Detection( frame=frame if frame is not None else 0, id=id, type=class_names[cat],
                                      prob=float( conf[i] ), xmin=box[0],
                                      ymin=box[1], xmax=box[2], ymax=box[3], xmid=box[0] + (box[2] * 0.5),
                                      ymid=box[1] + (box[3] * 0.5) ) )

    return image, result


def video_object_detection(variables):
    """
    Static detection method on the uploaded video
    https://github.com/yeha98555/object-detection-web-app
    """
    # test video for detection
    # https://www.pexels.com/video/aerial-footage-of-vehicular-traffic-of-a-busy-street-intersection-at-night-3048225/

    style, confidence_threshold, track_age, track_hits, iou_thres = variables.get_var()

    result_list = []

    file = st.file_uploader( 'Choose a video', type=['avi', 'mp4', 'mov'] )
    if file is not None:
        tfile = tempfile.NamedTemporaryFile( delete=True )
        tfile.write( file.read() )

        cap = cv2.VideoCapture( tfile.name )
        tfile.close()
        width, height = int( cap.get( cv2.CAP_PROP_FRAME_WIDTH ) ), int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
        fps = cap.get( cv2.CAP_PROP_FPS )
        total_frame = int( cap.get( cv2.CAP_PROP_FRAME_COUNT ) )

        # size limited by streamlit cloud service
        if width > 1920 or height > 1080:
            st.warning( f"File resolution [{width}x{height}] exceeded limit [1920x1080], "
                        f"please consider scale down the video", icon="⚠️" )
        else:
            gcd_wh = gcd( width, height )
            st.info( f"Uploaded video has aspect ratio of [{width // gcd_wh}:{height // gcd_wh}], "
                     f"best detection with model {best_match_ratio(width,height,config.STYLES)}"
                     )
            if st.button( 'Detect' ):
                progress_txt = st.caption( f'Analysing Video: 0 out of {total_frame} frames' )
                progress_bar = st.progress( 0 )
                progress = frame_counter_class()
                # temp dir for saving the video to be processed by opencv
                if not os.path.exists( os.path.join( config.HERE, 'storage' ) ):
                    os.makedirs( os.path.join( config.HERE, 'storage' ) )
                output_path = os.path.join( config.HERE, f"storage\\{str( uuid.uuid4() )}.mp4" )

                # encode cv2 output into h264
                # https://stackoverflow.com/questions/30509573/writing-an-mp4-video-using-python-opencv
                args = (ffmpeg
                        .input( 'pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format( width, height ) )
                        .output( output_path, pix_fmt='yuv420p', vcodec='libx264', r=fps, crf=37 )
                        .overwrite_output()
                        .get_args()
                        )
                # check if deployed at cloud or local host
                ffmpeg_source = config.FFMPEG_PATH if platform.processor() else 'ffmpeg'
                process = subprocess.Popen( [ffmpeg_source] + args, stdin=subprocess.PIPE )

                # init object detector and tracker
                detector = model_init( style, confidence_threshold )
                sort_tracker = Sort( track_age, track_hits, iou_thres )
                while cap.isOpened():
                    try:
                        ret, frame = cap.read()
                        if not ret:
                            break
                    except Exception as e:
                        print( e )
                        continue
                    detections = detector( frame )
                    # Update object localizer
                    image, result = annotate_image( frame, detections, sort_tracker, progress( 0 ) )
                    process.stdin.write( cv2.cvtColor( image, cv2.COLOR_BGR2RGB ).astype( np.uint8 ).tobytes() )
                    result_list.append( result )

                    # progress of analysis
                    progress_bar.progress( progress( 1 ) / total_frame )
                    progress_txt.caption( f'Analysing Video: {progress( 0 )} out of {total_frame} frames' )

                process.stdin.close()
                process.wait()
                process.kill()
                cap.release()
                tfile.close()

                st.video( output_path )
                os.remove( output_path )

                progress_bar.progress( 100 )
                progress_txt.empty()

                # Dumping analysis result into table
                try:
                    st.dataframe( pd.DataFrame.from_records( [item for sublist in result_list for item in sublist],
                                                             columns=Detection._fields ),
                                  # .style.apply(color_row, axis=1), TODO: add color to df by row index
                                  use_container_width=True )
                except ValueError as e:
                    'No tracking data found'
                    e


def live_object_detection(variables):
    """
    #This component was originated from https://github.com/whitphx/streamlit-webrtc
    """
    style, confidence_threshold, track_age, track_hits, iou_thres = variables.get_var()

    # public-stun-list.txt
    # https://gist.github.com/mondain/b0ec1cf5f60ae726202e
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # init frame counter, object detector and tracker
    frame_counter = frame_counter_class()
    detector = model_init( style, confidence_threshold )
    sort_tracker = Sort( track_age, track_hits, iou_thres )

    # reading each frame of live stream and passing to backend processing
    def frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray( format="bgr24" )
        detections = detector( image )
        counter = frame_counter
        annotated_image, result = annotate_image( image, detections, sort_tracker, counter() )
        counter( 1 )

        # NOTE: This `recv` method is called in another thread,
        # so it must be thread-safe.
        result_queue.put( result )

        return av.VideoFrame.from_ndarray( annotated_image, format="bgr24" )

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,  # when deploy on remote host need stun server for camera connection
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
                except queue.Empty:
                    result = None
                labels_placeholder.dataframe( result )


def main():
    st.header( "Object Detecting and Tracking demo" )

    pages = {
        "Real time object detection (sendrecv)": live_object_detection,
        "Upload Video for detection": video_object_detection,
    }
    page_titles = pages.keys()

    my_sidebar = st.sidebar
    page_title = my_sidebar.selectbox(
        "Choose the app mode",
        page_titles,
    )
    with my_sidebar:
        variables = st_variables_container()
    st.subheader( page_title )

    page_func = pages[page_title]
    page_func( variables )

    # logger.debug( "=== Alive threads ===" )
    # for thread in threading.enumerate():
    #    if thread.is_alive():
    #        logger.debug( f"  {thread.name} ({thread.ident})" )


if __name__ == "__main__":
    import os

    DEBUG = config.DEBUG

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
               "%(message)s",
        force=True,
    )

    logger.setLevel( level=logging.DEBUG if DEBUG else logging.INFO )

    st_webrtc_logger = logging.getLogger( "streamlit_webrtc" )
    st_webrtc_logger.setLevel( logging.DEBUG )

    fsevents_logger = logging.getLogger( "fsevents" )
    fsevents_logger.setLevel( logging.WARNING )

    main()
