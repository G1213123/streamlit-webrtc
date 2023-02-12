import bridge_wrapper
import config
import detection_helpers
from components import intersect_counter as ic, frame_counter as fc, variables_panel as vp, session_result

import platform
import logging
import queue
import ffmpeg
import subprocess
import tempfile
import uuid
import av
import cv2
import numpy as np
from io import BytesIO
import streamlit as st

from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    webrtc_streamer,
)

st.set_page_config( layout="wide" )
logger = logging.getLogger( __name__ )



def video_object_detection(variables):
    """
    Static detection method on the uploaded video
    https://github.com/yeha98555/object-detection-web-app
    """
    # test video for detection
    # https://www.pexels.com/video/aerial-footage-of-vehicular-traffic-of-a-busy-street-intersection-at-night-3048225/

    weight, confidence_threshold, track_age, track_hits, iou_thres = variables.get_var()

    if 'result_list' not in st.session_state:
        st.session_state.result_list = []
    if 'video' not in st.session_state:
        st.session_state.video = None
    if 'file' not in st.session_state:
        st.session_state.file = None

    file = st.file_uploader( 'Choose a video', type=['avi', 'mp4', 'mov'] )
    if file is not None:
        if file != st.session_state.file:
            st.session_state.file = file
            st.session_state.video = None
            st.session_state.counters = []
            st.session_state.counters_table = []
            st.session_state.counted = False
            st.session_state.result_list = []

        # save the uploaded file to a temporary location
        tfile = tempfile.NamedTemporaryFile( delete=True )
        tfile.write( file.read() )
        cap = cv2.VideoCapture( tfile.name )
        tfile.close()

        width, height = int( cap.get( cv2.CAP_PROP_FRAME_WIDTH ) ), int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
        fps = cap.get( cv2.CAP_PROP_FPS )
        total_frame = int( cap.get( cv2.CAP_PROP_FRAME_COUNT ) )
        success, image = cap.read()

        # setup expander for user to draw line counters
        icounter = ic.st_IntersectCounter( image, width, height )

        # size limited by streamlit cloud service (superseded)
        if max(width, height) > 1920 or min(width, height) > 1080:
            st.warning( f"File resolution [{width}x{height}] exceeded limit [1920x1080], "
                        f"please consider scale down the video", icon="⚠️" )
        else:
            detect = st.button( 'Detect' )
            if detect:
                # show analysis progress
                progress_txt = st.caption( f'Analysing Video: 0 out of {total_frame} frames' )
                progress_bar = st.progress( 0 )
                progress = fc.FrameCounter()

                for c in st.session_state.counters:
                    c.reset()

                # temp dir for saving the video to be processed by opencv
                if not os.path.exists( os.path.join( config.HERE, 'storage' ) ):
                    os.makedirs( os.path.join( config.HERE, 'storage' ) )
                output_path = os.path.join( config.HERE, f"storage\\{str( uuid.uuid4() )}.mp4" )

                # encode cv2 output into h264
                # https://stackoverflow.com/questions/30509573/writing-an-mp4-video-using-python-opencv
                args = (ffmpeg
                        .input( 'pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format( width, height ) )
                        .output( output_path, pix_fmt='yuv420p', vcodec='libx264', r=fps, crf=12 )
                        .overwrite_output()
                        .get_args()
                        )

                # check if deployed on cloud or local host
                ffmpeg_source = config.FFMPEG_PATH if platform.processor() else 'ffmpeg'
                process = subprocess.Popen( [ffmpeg_source] + args, stdin=subprocess.PIPE )

                # init object detector and tracker
                detector = detection_helpers.Detector( confidence_threshold )
                detector.load_model( 'weights/' + config.STYLES[weight], trace=False )
                deepsort_tracker = bridge_wrapper.YOLOv7_DeepSORT(
                    reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector,
                    max_iou_distance=iou_thres, max_age=track_age, n_init=track_hits )
                frame_num = fc.FrameCounter()

                # analysis per frame here
                while cap.isOpened():
                    try:
                        ret, frame = cap.read()
                        if not ret:
                            break
                    except Exception as e:
                        print( e )
                        continue
                    image, result = deepsort_tracker.track_video_stream( frame, frame_num( 1 ), verbose=1 )
                    image = icounter.update_counters( deepsort_tracker.tracker.tracks, image )
                    # Update object localizer
                    # image, result = track_and_annotate_detections( frame, detections, sort_tracker,
                    #                                               st.session_state.counters, progress( 0 ) )
                    process.stdin.write( cv2.cvtColor( image, cv2.COLOR_BGR2RGB ).astype( np.uint8 ).tobytes() )
                    st.session_state['result_list'].extend( result )

                    # progress of analysis
                    progress_bar.progress( progress( 1 ) / total_frame )
                    progress_txt.caption( f'Analysing Video: {progress( 0 )} out of {total_frame} frames' )

                process.stdin.close()
                process.wait()
                process.kill()
                cap.release()
                tfile.close()

                # TODO: skip writing the analysis result into a temp file and read into memory
                with open( output_path, "rb" ) as fh:
                    buf = BytesIO( fh.read() )
                st.session_state.video = buf
                os.remove( output_path )

                progress_bar.progress( 100 )
                progress_txt.empty()
                st.session_state.counted = True

            if st.session_state.video is not None:
                st.video( st.session_state.video )

            # Dumping analysis result into table
            if st.session_state.counted:
                if st.checkbox( "Show all detection results" ):
                    if len( st.session_state.result_list ) > 0:
                        result_df = session_result.result_to_df( st.session_state.result_list )
                        st.dataframe( result_df, use_container_width=True )
                icounter.show_counter_results()


def live_object_detection(variables):
    """
    #This component was originated from https://github.com/whitphx/streamlit-webrtc
    """
    weight, confidence_threshold, track_age, track_hits, iou_thres = variables.get_var()

    # init frame counter, object detector, tracker and passing object counter
    frame_num = fc.FrameCounter()
    detector = detection_helpers.Detector( confidence_threshold )
    detector.load_model( 'weights/' + config.STYLES[weight], trace=False )
    deepsort_tracker = bridge_wrapper.YOLOv7_DeepSORT(
        reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector,
        max_iou_distance=iou_thres, max_age=track_age, n_init=track_hits )

    if 'counters' not in st.session_state:
        st.session_state.counters = []
    icounter = st.session_state.counters

    # Dump queue for real time detection result
    result_queue = (queue.Queue())
    frame_queue = (queue.Queue( maxsize=1 ))

    # reading each frame of live stream and passing to backend processing
    def frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        frame = frame.to_ndarray( format="bgr24" )

        # Detect, track and counter the intersect of objects here
        image, result = deepsort_tracker.track_video_stream( frame, frame_num( 1 ))
        if icounter is not None:
            if len(icounter)>0:
                image = st_icounter.update_counters( deepsort_tracker.tracker.tracks, image, icounter )

        # NOTE: This `recv` method is called in another thread,
        # so it must be thread-safe.
        result_queue.put( result )
        if not frame_queue.full():
            frame_queue.put( image )

        return av.VideoFrame.from_ndarray( image, format="bgr24" )

    # public-stun-list.txt
    # https://gist.github.com/mondain/b0ec1cf5f60ae726202e
    servers = [{"url": "stun:stun.l.google.com:19302"}]
    if 'URL' in st.secrets:
        servers.append( {"urls": st.secrets['URL'],
                         "username": st.secrets['USERNAME'],
                         "credential": st.secrets['CREDENTIAL'],                         } )
    RTC_CONFIGURATION = RTCConfiguration( {"iceServers": servers} )

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
        st_icounter = ic.st_IntersectCounter( image, image.shape[1], image.shape[0] )
        icounter = st.session_state.counters
        if len( st.session_state.counters ) > 0:
            st.session_state.counted = True
        labels_placeholder = st.empty()

        # NOTE: The video transformation with object detection and
        # this loop displaying the result labels are running
        # in different threads asynchronously.
        # Then the rendered video frames and the labels displayed here
        # are not strictly synchronized.
        while True:
            try:
                result = result_queue.get( timeout=1.0 )
                labels_placeholder.dataframe( session_result.result_to_df(result), use_container_width=True )
            except queue.Empty:
                result = None
            if st_icounter is not None:
                st_icounter.show_counter_results()

    else:
        st.session_state.counters = []
        st.session_state.counters_table = []
        st.session_state.counted = False
        st.session_state.result_list = []


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
        variables = vp.st_VariablesPanel()
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
