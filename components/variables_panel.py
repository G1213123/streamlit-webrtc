import streamlit as st
import config

class st_VariablesPanel:
    # Container for the variable sliders listening user input
    def __init__(self):
        with st.container():
            st.subheader("Detection")
            st.selectbox( 'Choose the [detection model](https://github.com/WongKinYiu/yolov7)',
                          list( config.STYLES.keys() ), key='model_style' )
            st.slider(
                "Confidence threshold", 0.0, 1.0, 0.5, 0.05, key='confidence_threshold'
            )
            st.subheader( "Tracking" )
            st.caption( '[DEEPSORT](https://github.com/deshwalmahesh/yolov7-deepsort-tracking) Tracking Algorithm' )
            st.slider(
                "Tracking Age (frames)", 0, 20, 10, 1, key='track_age'
            )
            st.slider(
                "Tracking hits", 0, st.session_state.track_age, 3, 1, key='tracking_hits'
            )
            st.slider(
                "IOU threshold", 0.0, 1.0, 0.5, 0.1, key='iou_thres'
            )

    def get_var(self):
        return st.session_state['model_style'], \
            st.session_state['confidence_threshold'], st.session_state['track_age'], \
            st.session_state['tracking_hits'], st.session_state['iou_thres']