from ultralytics.yolo.engine.model import YOLO
import streamlit as st
import tempfile
import cv2
from tracker import Tracker
import torch
import supervision as sv
from supervision import Detections, BoxAnnotator
import numpy as np
import math
@st.cache_resource
def loadYoloModel():
 return YOLO("models/yolov8m.pt")

model = loadYoloModel()
all_classes = model.names

def loadTracker():
    return Tracker()

tracker = loadTracker()

st.title("Traffic Monitoring")

APP_MODE = st.sidebar.selectbox("Choose the app mode",["Speed Estimation","Road Heatmap"])

if APP_MODE == "Speed Estimation":
    st.subheader("Speed Estimation")
    
    videoFile = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi", "asf", "m4v"])

    if torch.cuda.is_available():
        enable_gpu = st.sidebar.checkbox("Enable GPU")
    else:
        enable_gpu = False

    if enable_gpu:
        device = 0
    else:
        device = 'cpu'

    tmpfile = tempfile.NamedTemporaryFile(delete=False)

    distance_unit = st.sidebar.number_input("Enter the distance unit in meters",value=30.0)

    required_classes = ['car', 'bus', 'truck', 'motorcycle', 'bicycle']
    annotator = BoxAnnotator()
    required_classes = st.sidebar.multiselect("Select the classes to track", required_classes,required_classes[0])
    if videoFile is not None:
       
        if st.sidebar.button("Start") or True:
           
            tmpfile.write(videoFile.read())

            cap = cv2.VideoCapture(tmpfile.name)

            stframe = st.empty()

            required_classes_ids = [idx for idx,cls in all_classes.items() if cls in required_classes]

            curr_center = dict()
            prev_center = dict()
            speeds = dict()

            while True:
                ret, frame = cap.read()
                  
                if not ret:
                    st.error("Video file not supported!")
                    break

                results = model(frame,device=device,classes=required_classes_ids)
                detections = []
                names = []

                for result in results:
                    for object in result.boxes.data.tolist():
                        
                        x1, y1, x2, y2, conf, cls = object
                        if conf < 0.5:
                            continue
                        x1 = int(x1)
                        y1 = int(y1)
                        x2 = int(x2)
                        y2 = int(y2)
                        
                        detections.append([x1, y1, x2, y2, conf])
                        names.append(int(cls))
                    if len(detections) > 0:
                        tracker.update(frame,detections,names)

                    
                    tracks = {track.track_id : {"bbox":track.bbox,"cls":track.cls} for track in tracker.tracks}
                    bboxes = np.array([track['bbox'] for track in  tracks.values()],dtype=int)

                    labels = np.array([track['cls'] for track in tracks.values()])
                    tracker_ids = np.array([track_id for track_id in tracks.keys()])

                    prev_center.update(curr_center)
                    curr_center = {track_id : ((bbox[0]+bbox[2])//2,(bbox[1]+bbox[3])//2) for track_id,bbox in zip(tracks.keys(),bboxes)}
                    
                    for track_id,center in curr_center.items():
                        if track_id in prev_center:
                            distance = math.hypot(center[0]-prev_center[track_id][0],center[1]-prev_center[track_id][1])
                            ratio = (frame.shape[0]-center[1])/frame.shape[0]
                            speed =  distance*distance_unit*ratio
                            speeds[track_id].append(speed)
                        else:
                            speeds[track_id] = [0]
                    
                    if bboxes.shape[-1] == 4:
                        detections = Detections(bboxes,class_id=labels,tracker_id=tracker_ids)
                        labels = [
                                f"{all_classes[class_id]} {track_id} {sum(speeds[track_id])//len(speeds[track_id])}km/h "
                                for class_id,track_id
                                in zip(detections.class_id, detections.tracker_id)
                            ]
                        
                        frame = annotator.annotate(frame,detections,labels=labels)

                stframe.image(frame, channels="BGR")


if  APP_MODE == "Road Heatmap":
    st.subheader("Road Heatmap")
    
    video_file_buffer = st.sidebar.file_uploader("Select Video", type=["mp4", "mov", "avi", "asf", "m4v"])
    required_classes = ['car', 'bus', 'truck', 'motorcycle', 'bicycle']

    required_classes = st.sidebar.multiselect("Select the classes to track", required_classes,required_classes[0])
    draw_bbox = st.sidebar.checkbox("Draw Bounding Boxes")
    if video_file_buffer is not None:
        
        if st.sidebar.button("start"):

            tffile = tempfile.NamedTemporaryFile(delete=False)
            tffile.write(video_file_buffer.read())

            cap = cv2.VideoCapture(tffile.name)

            stframe = st.empty()
            required_classes_ids = [idx for idx,cls in all_classes.items() if cls in required_classes]
           
            heat_map_norm = None
            heat_map_array = np.ones((1080,1920),dtype='uint32')
            while True:
                ret,frame = cap.read()

                results = model(frame, device="0",classes = required_classes_ids)[0]

                for object in results.boxes.data.tolist():
                    x1, y1, x2, y2, conf, cls = object
                    if conf < 0.5:
                        continue
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    heat_map_array[y1:y2,x1:x2] += 1

                    if draw_bbox:
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),2)
                
                heat_map_norm = (heat_map_array - heat_map_array.min())/(heat_map_array.max()-heat_map_array.min()) * 255
                heat_map_norm = heat_map_norm.astype("uint8")
                heat_map_norm = cv2.GaussianBlur(heat_map_norm, (9,9), 0)
                heat_map_img = cv2.applyColorMap(heat_map_norm,cv2.COLORMAP_JET)
                frame = cv2.addWeighted(heat_map_img,0.5,frame,0.5,0)

                stframe.image(frame,channels="BGR")


