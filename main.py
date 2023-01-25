import config
import inference
from YOLOv7.utils import class_names

import copy
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
from streamlit_drawable_canvas import st_canvas
from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    webrtc_streamer,
)
from PIL import Image

st.set_page_config( layout="wide" )
logger = logging.getLogger( __name__ )

@st.experimental_singleton
def generate_label_colors():
    return np.random.uniform( 0, 255, size=(65536, 3) )


COLORS = generate_label_colors()


def color_row(s):
    return COLORS[s.name]


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


class st_counter_setup_container:
    def __init__(self, image, width, height, screen_width=640):
        self.display_scale = width / screen_width
        self.counters_df_display = None
        self.counters_table = None
        self.counters = []
        self.counters_num = 0
        self.wrapper = st.expander( "Setup Counter" )
        self.option = 'Empty'
        with self.wrapper:
            canvas_result = st_canvas(
                width=screen_width,
                height=height // self.display_scale,
                background_image=Image.fromarray( cv2.cvtColor( image, cv2.COLOR_BGR2RGB ) ),
                stroke_width=1,
                drawing_mode='line', key="canvas"
            )

            if canvas_result.json_data is not None:
                if len( canvas_result.json_data['objects'] ) > 0:
                    self.canvas_result = canvas_result.json_data['objects']
                    self.format_counters_display()
                    self.counters_df_display = st.dataframe( self.counters_table.style.set_precision( 1 ) )

    def generate_counters(self):
        for ind in range( self.counters_num ):
            centroid = point( self.counters_table['left'][ind], self.counters_table['top'][ind] )
            xoffset = self.counters_table['x1'][ind]
            yoffset = self.counters_table['y1'][ind]
            counter = passing_object_counter.init_centroid( centroid, xoffset, yoffset, ind, self.display_scale )
            self.counters.append( counter )
            yield counter

    def counter_results_select(self):
        if len( self.counters ) > 0:
            if 'index' not in st.session_state:
                st.session_state.counter_option_index = 0
            if 'counter_result_options' not in st.session_state:
                st.session_state.counter_result_options = [d.id for d in self.counters]
            with self.wrapper:
                if st.session_state.counter_option_index == 0:
                    st.session_state.counter_option_index = st.selectbox(
                        'Counter Result',
                        options=st.session_state.counter_result_options,
                    )
                    selected_counter = [x for x in self.counters if x.id == st.session_state.counter_result_options[st.session_state.counter_option_index]][0]
                    st.table( pd.DataFrame( selected_counter.counted_objects ) )

    def format_counters_display(self, results=None):
        self.counters_table=pd.json_normalize( self.canvas_result )
        show_columns = ['type', 'left', 'top', 'x1', 'x2', 'y1', 'y2', 'width', 'height']
        self.counters_num = len( self.counters_table.index )
        if self.counters_table is not None:
            self.counters_table = self.counters_table[show_columns]
            #self.counters_table.style.format(precision=1)
        else:
            return None
        if results is not None and len( results ) == len( self.counters_table.index ):
            self.counters_table['count'] = [r.count for r in results]

        return self.counters_table


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


# 進階
# 建立平面座標點
class point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __mul__(self, other: float):
        self.x *= other
        self.y *= other
        return self

    __rmul__ = __mul__

    def __add__(self, other):
        self.x += other.x
        self.y += other.y
        return self

    def __iter__(self):
        for i in [self.x, self.y]:
            yield i


# https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
def ccw(A, B, C):
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)


class passing_object_counter():
    def __init__(self, vertices: List[point], id, scale: float = 1.0):
        self.vertices = [v * scale for v in vertices]
        self.count = 0
        self.counted_objects = []
        self.id = int( id )

    @classmethod
    def init_centroid(cls, centroid, x_offset, y_offset, id, scale=1.0):
        p1, p2 = copy.deepcopy( centroid ), copy.deepcopy( centroid )
        vertices = [p1 + point( x_offset, y_offset ), p2 + point( -x_offset, -y_offset )]
        return cls( vertices, id, scale )

    def check_intersect(self, path_vertices: List[point], object):
        A, B, C, D = self.vertices[0], self.vertices[1], path_vertices[0], path_vertices[1]
        if ccw( A, C, D ) != ccw( B, C, D ) and ccw( A, B, C ) != ccw( A, B, D ):
            self.count += 1
            self.counted_objects.append( object )
        return self.count


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


def track_and_annotate_detections(image, detections, sort_tracker, passing_counters=None, frame=None):
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

    # Run SORT and get tracked objects
    tracked_dets = sort_tracker.update( dets_to_sort )
    tracks = sort_tracker.getTrackers()

    # loop over tracks
    for track in tracks:
        # draw colored tracks
        if len( track.centroidarr ) > 1:
            track_last_path = [point( *track.centroidarr[-1] ), point( *track.centroidarr[-2] )]

        drawn_track = [cv2.line( image, (int( track.centroidarr[i][0] ),
                                         int( track.centroidarr[i][1] )),
                                 (int( track.centroidarr[i + 1][0] ),
                                  int( track.centroidarr[i + 1][1] )),
                                 COLORS[track.id + 1], thickness=2 )
                       for i, _ in enumerate( track.centroidarr )
                       if i < len( track.centroidarr ) - 1]

        # draw boxes for visualization
        bbox_xyxy = track.bbox_history[-1][:4]
        identities = track.id + 1
        conf = track.conf
        categories = int( track.detclass )
        x1, y1, x2, y2 = [int( i ) for i in bbox_xyxy]
        label = str( identities ) + ":" + class_names[categories] + "-" + "%.2f" % conf
        cv2.rectangle( image, (x1, y1), (x2, y2), COLORS[identities], 2 )
        cv2.putText( image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                     1, COLORS[identities], 2 )

        # dump the tracked object to the result queue
        detected_obj = Detection( frame=frame if frame is not None else 0, id=identities, type=class_names[categories],
                                  prob=float( conf ), xmin=bbox_xyxy[0],
                                  ymin=bbox_xyxy[1], xmax=bbox_xyxy[2], ymax=bbox_xyxy[3],
                                  xmid=bbox_xyxy[0] + (bbox_xyxy[2] * 0.5),
                                  ymid=bbox_xyxy[1] + (bbox_xyxy[3] * 0.5) )
        result.append( detected_obj )

        # update passing counter with the latest tracked objects path
        if passing_counters is not None:
            for p in passing_counters:
                if len( track.centroidarr ) > 1:
                    p.check_intersect( track_last_path, detected_obj )
                cv2.line( image, tuple( map( int, p.vertices[0] ) ), tuple( map( int, p.vertices[1] ) ),
                          COLORS[p.id], thickness=2 )
                label = f'Counter_{p.id}: {p.count}'
                cv2.putText( image, label, tuple( map( int, copy.deepcopy( p.vertices[0] ) + point( 5, 5 ) ) ),
                             cv2.FONT_HERSHEY_SIMPLEX,
                             1, COLORS[p.id], 2 )

    return image, result


def video_object_detection(variables):
    """
    Static detection method on the uploaded video
    https://github.com/yeha98555/object-detection-web-app
    """
    # test video for detection
    # https://www.pexels.com/video/aerial-footage-of-vehicular-traffic-of-a-busy-street-intersection-at-night-3048225/

    style, confidence_threshold, track_age, track_hits, iou_thres = variables.get_var()

    if 'detect' not in st.session_state:
        st.session_state.detect = False
    st.session_state['result_list'] = []
    passing_object_counter_list = []

    file = st.file_uploader( 'Choose a video', type=['avi', 'mp4', 'mov'] )
    if file is not None:
        # save the uploaded file to a temporary location
        tfile = tempfile.NamedTemporaryFile( delete=True )
        tfile.write( file.read() )
        cap = cv2.VideoCapture( tfile.name )
        tfile.close()

        width, height = int( cap.get( cv2.CAP_PROP_FRAME_WIDTH ) ), int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
        fps = cap.get( cv2.CAP_PROP_FPS )
        total_frame = int( cap.get( cv2.CAP_PROP_FRAME_COUNT ) )
        success, image = cap.read()

        passing_counter = st_counter_setup_container( image, width, height )
        passing_object_counter_list = list( passing_counter.generate_counters() )

        # size limited by streamlit cloud service
        if width > 1920 or height > 1080:
            st.warning( f"File resolution [{width}x{height}] exceeded limit [1920x1080], "
                        f"please consider scale down the video", icon="⚠️" )
        else:
            gcd_wh = gcd( width, height )
            st.info( f"Uploaded video has aspect ratio of [{width // gcd_wh}:{height // gcd_wh}], "
                     f"best detection with model {best_match_ratio( width, height, config.STYLES )}"
                     )
            detect = st.button('Detect')
            if detect or st.session_state.detect:
                st.session_state.detect=True
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
                    image, result = track_and_annotate_detections( frame, detections, sort_tracker,
                                                                   passing_object_counter_list, progress( 0 ) )
                    process.stdin.write( cv2.cvtColor( image, cv2.COLOR_BGR2RGB ).astype( np.uint8 ).tobytes() )
                    st.session_state['result_list'].append( result )

                    # progress of analysis
                    progress_bar.progress( progress( 1 ) / total_frame )
                    progress_txt.caption( f'Analysing Video: {progress( 0 )} out of {total_frame} frames' )

                process.stdin.close()
                process.wait()
                process.kill()
                cap.release()
                tfile.close()

                st.video( output_path)
                os.remove( output_path )

                progress_bar.progress( 100 )
                progress_txt.empty()

                # Dumping analysis result into table
                try:
                    st.dataframe( pd.DataFrame.from_records( [item for sublist in st.session_state['result_list'] for item in sublist],
                                                             columns=Detection._fields ),
                                  # .style.apply(color_row, axis=1), TODO: add color to df by row index
                                  use_container_width=True )

                except ValueError as e:
                    'No tracking data found'
                    e
                passing_counter.counters_df_display.dataframe(
                    passing_counter.format_counters_display( passing_object_counter_list ) )
                passing_counter.counter_results_select()


def live_object_detection(variables):
    """
    #This component was originated from https://github.com/whitphx/streamlit-webrtc
    """
    style, confidence_threshold, track_age, track_hits, iou_thres = variables.get_var()

    # public-stun-list.txt
    # https://gist.github.com/mondain/b0ec1cf5f60ae726202e
    # turn server
    # https://www.metered.ca/tools/openrelay/
    turn_server = {"urls": st.secrets['URL'],
                   "username": st.secrets['USERNAME'],
                   "credential": st.secrets['CREDENTIAL'],
                   } if st.secrets['URL'] is not None else None
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"url": "stun:stun3.l.google.com:19302"},
                        {"url": "stun:stun.l.google.com:19302"},
                        turn_server]}
    )

    # init frame counter, object detector, tracker and passing object counter
    frame_counter = frame_counter_class()
    detector = model_init( style, confidence_threshold )
    sort_tracker = Sort( track_age, track_hits, iou_thres )
    counter_setup_container = None
    passing_object_counters = None

    # Dump queue for real time detection result
    result_queue = (queue.Queue())
    frame_queue = (queue.Queue( maxsize=1 ))

    # reading each frame of live stream and passing to backend processing
    def frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray( format="bgr24" )
        detections = detector( image )
        counter = frame_counter
        annotated_image, result = track_and_annotate_detections( image, detections, sort_tracker,
                                                                 passing_object_counters,
                                                                 counter() )
        counter( 1 )

        # NOTE: This `recv` method is called in another thread,
        # so it must be thread-safe.
        result_queue.put( result )
        if not frame_queue.full():
            frame_queue.put( image )

        return av.VideoFrame.from_ndarray( annotated_image, format="bgr24" )

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,  # when deploy on remote host need stun server for camera connection
        video_frame_callback=frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # capture image for the counter setup container
    if webrtc_ctx.state.playing:
        image = frame_queue.get()
        if counter_setup_container is None:
            counter_setup_container = st_counter_setup_container( image, image.shape[1], image.shape[0] )
    try:
        passing_object_counters = list( counter_setup_container.generate_counters() )
    except:
        passing_object_counters = None

    if st.checkbox( "Show the detected labels", value=True ):
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
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
