import cv2
import face_recognition
import numpy as np
import pickle
import os
import torch
import torch.cuda
from tqdm import tqdm
import dlib
import time
import urllib.request
import bz2

def download_and_extract_model(url, filename):
    """Download and extract the model file if it doesn't exist"""
    if not os.path.exists(filename):
        print(f"[INFO] Downloading {filename}...")
        compressed_file = filename + '.bz2'
        
        # Download the compressed file
        urllib.request.urlretrieve(url, compressed_file)
         
        # Extract the compressed file
        print(f"[INFO] Extracting {filename}...")
        with bz2.BZ2File(compressed_file) as fr, open(filename, 'wb') as fw:
            fw.write(fr.read())
        
        # Remove the compressed file
        os.remove(compressed_file)
        print(f"[INFO] {filename} ready!")

def ensure_models_exist():
    """Ensure all required models are downloaded"""
    models = {
        'mmod_human_face_detector.dat': 'http://dlib.net/files/mmod_human_face_detector.dat.bz2',
        'shape_predictor_68_face_landmarks.dat': 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
        'dlib_face_recognition_resnet_model_v1.dat': 'http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2'
    }
    
    for model_file, url in models.items():
        download_and_extract_model(url, model_file)

def get_gpu_info():
    """Print GPU information"""
    if torch.cuda.is_available():
        print(f"[INFO] Using device: cuda")
        print(f"[INFO] GPU Name: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[INFO] Available GPU memory: {gpu_memory:.2f} GB")
        print(f"[INFO] CUDA version: {torch.version.cuda}")
    else:
        print("[WARNING] CUDA is not available. Using CPU instead.")


def process_frame(frame, cnn_face_detector, face_encoder, scale_factor=0.5):
    """Process a single frame"""
    start_time = time.time()
    
    # Resize frame for faster face detection
    small_frame = cv2.resize(frame, (640, 480))  # Resize for better detection
    
    # Convert to RGB
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces using dlib's CNN face detector
    detected_faces = cnn_face_detector(rgb_small_frame)
    print(f"[INFO] Detected {len(detected_faces)} faces")  # Debugging: output face detection results
    
    # Adjust face locations back to original frame size
    boxes = []
    for face in detected_faces:
        rect = face.rect
        left = int(rect.left() / scale_factor)
        top = int(rect.top() / scale_factor)
        right = int(rect.right() / scale_factor)
        bottom = int(rect.bottom() / scale_factor)
        boxes.append((top, right, bottom, left))
    
    # Get face encodings
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_frame, boxes)
    
    process_time = time.time() - start_time
    
    # Debugging: Check how many encodings were found
    print(f"[INFO] Found {len(encodings)} encodings")
    
    return boxes, encodings, process_time

def main():
    # First ensure we have all required models
    print("[INFO] Checking and downloading required models...")
    ensure_models_exist()
    
    # Print GPU information
    get_gpu_info()
     
    # Initialize dlib's CNN face detector
    print("[INFO] Loading face detection model...")
    cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
    
    if torch.cuda.is_available():
        print("[INFO] Using CUDA for face detection")
        dlib.DLIB_USE_CUDA = True
    
    # Initialize face encoder
    print("[INFO] Loading face recognition model...")
    face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
    
    # Define your paths
    video_folder = "videos/"
    encodings_file = "encodings_video2.pickle"
    
    # Create video folder if it doesn't exist
    os.makedirs(video_folder, exist_ok=True)
    
    # Load existing encodings
    if os.path.exists(encodings_file):
        print("[INFO] Loading existing encodings...")
        with open(encodings_file, "rb") as f:
            data = pickle.load(f)
    else:
        data = []
    
    # Get video files
    video_paths = [os.path.join(video_folder, f) 
                  for f in os.listdir(video_folder) 
                  if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_paths:
        print("[ERROR] No video files found in the videos folder!")
        return
    
    # Process each video
    for video_path in video_paths:
        print(f"[INFO] Processing video: {video_path}")
        video_capture = cv2.VideoCapture(video_path)
        
        if not video_capture.isOpened():
            print(f"[ERROR] Could not open video file: {video_path}")
            continue
        
        # Get total frames for progress bar
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        # Performance tracking
        times = []
        
        # Create progress bar
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break
                
                frame_count += 1
                
                try:
                    # Process frame
                    boxes, encodings, process_time = process_frame(
                        frame, 
                        cnn_face_detector, 
                        face_encoder,
                        scale_factor=0.5  # Adjusted scale factor
                    )
                    times.append(process_time)
                    
                    # Store results
                    for box, encoding in zip(boxes, encodings):
                        data.append({
                            "videoPath": video_path,
                            "frame": frame_count,
                            "loc": box,
                            "encoding": encoding
                        })
                    
                    # Update progress bar with performance info
                    if len(times) > 0:
                        avg_time = sum(times[-100:]) / min(len(times), 100)
                        pbar.set_postfix({'fps': 1/avg_time if avg_time > 0 else 0})
                    
                except Exception as e:
                    print(f"\n[ERROR] Error processing frame {frame_count}: {str(e)}")
                    continue
                
                pbar.update(1)
                
                # Clear GPU cache periodically
                if frame_count % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        video_capture.release()
        
        # Print performance summary
        if times:
            avg_time = sum(times) / len(times)
            print(f"\n[INFO] Average processing time per frame: {avg_time:.3f} seconds")
            print(f"[INFO] Average FPS: {1/avg_time:.2f}")
    
    # Save encodings
    print("[INFO] Serializing encodings...")
    with open(encodings_file, "wb") as f:
        f.write(pickle.dumps(data))
    print("[INFO] Encodings saved to encodings.pickle")

if __name__ == "__main__":
    main()

