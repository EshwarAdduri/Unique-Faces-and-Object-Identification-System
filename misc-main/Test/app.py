import streamlit as st
import cv2
import numpy as np
import pickle
import face_recognition
from sklearn.cluster import DBSCAN
from imutils import build_montages
import os
import io
from PIL import Image 
 
# Function to handle face encoding for images
def encode_faces_from_image(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="cnn")
    encodings = face_recognition.face_encodings(rgb, boxes)
    return encodings, boxes

# Function to handle face encoding from video
def encode_faces_from_video(video_file):
    video_capture = cv2.VideoCapture(video_file)
    data = []
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="cnn")
        encodings = face_recognition.face_encodings(rgb, boxes)
        for (box, encoding) in zip(boxes, encodings):
            data.append({"frame": frame_count, "loc": box, "encoding": encoding})
    video_capture.release()
    return data

# Function to perform DBSCAN clustering
def cluster_faces(encodings):
    clt = DBSCAN(metric="euclidean", n_jobs=-1)
    clt.fit(encodings)
    return clt

# Function to visualize face clusters
def display_clustered_faces(data, labels):
    labelIDs = np.unique(labels)
    numUniqueFaces = len(np.where(labelIDs > -1)[0])
    st.write(f"[INFO] Number of unique faces: {numUniqueFaces}")
    
    for labelID in labelIDs:
        idxs = np.where(labels == labelID)[0]
        idxs = np.random.choice(idxs, size=min(25, len(idxs)), replace=False)

        faces = []
        for i in idxs:
            (top, right, bottom, left) = data[i]["loc"]
            face = data[i]["frame"][top:bottom, left:right]
            face = cv2.resize(face, (96, 96))
            faces.append(face)

        montage = build_montages(faces, (96, 96), (5, 5))[0]
        title = f"Face ID #{labelID}" if labelID != -1 else "Unknown Faces"
        st.image(montage, caption=title)

# Streamlit UI
st.title("Face Detection, Encoding, and Clustering")

# Option to upload an image
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image = np.array(image)
    
    # Perform face encoding on the image
    encodings, boxes = encode_faces_from_image(image)

    # Cluster the faces and display the results
    if len(encodings) > 0:
        labels = cluster_faces(encodings).labels_
        display_clustered_faces(boxes, labels)
    else:
        st.write("No faces found in the image.")

# Option to upload a video
uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    video_bytes = uploaded_video.read()
    st.video(video_bytes)

    # Convert video file to bytes-like object for processing
    video_data = io.BytesIO(video_bytes)
    encodings = encode_faces_from_video(video_data)

    # Perform DBSCAN clustering on the video faces
    if len(encodings) > 0:
        labels = cluster_faces([e["encoding"] for e in encodings]).labels_
        display_clustered_faces(encodings, labels)
    else:
        st.write("No faces found in the video.")
