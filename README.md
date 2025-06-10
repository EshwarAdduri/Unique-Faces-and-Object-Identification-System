## Unique Faces and Object Identification Project

### Project Overview

This project aims to provide a comprehensive solution for identifying and indexing unique faces and objects in videos and images. The system is designed to handle large datasets efficiently, ensuring scalability and confidentiality by processing data locally.

### Project Structure

- **face_and_object_detection/**
  - **Object_detection/**: Contains files and models for object detection.
  - **dataset/**: Includes training images.
  - **videos/**: Contains training videos.

- **Files:**
  - `.gitignore`: Specifies files and directories to ignore in version control.
  - `image_processing.ipynb`: Jupyter notebook for face detection, clustering, and searching in images.
  - `video_processing.ipynb`: Jupyter notebook for face detection, clustering, and searching in videos.
  - `requirements.txt`: Lists all Python dependencies.
  - `wasabi.ipynb` & `wasabi_interface.py`: Basic scripts for integrating with Wasabi storage.

### How to Run and Use

1. **Setup Environment:**
   - Ensure Python is installed on your machine.
   - Install required packages using: `pip install -r requirements.txt`.

2. **Prepare Data:**
   - Place your images in the `dataset/` folder.
   - Place your videos in the `videos/` folder.

3. **Run Image Processing:**
   - Open `image_processing.ipynb` in Jupyter Notebook.
   - Execute the cells sequentially to process images, perform clustering, and search for faces.

4. **Run Video Processing:**
   - Open `video_processing.ipynb` in Jupyter Notebook.
   - Execute the cells to process videos, perform clustering, and search for faces.

5. **Object Detection:**
   - First download yolov3.weights form chrome, next place this file in *\Object_detection\model\
   - Run the first two cells in `Object_detection.ipynb` to save YOLOv3 model.
   - Use `detection.py` to detect objects in images using the YOLOv3 model.
   - Run the script and input the path to the image you want to analyze.

6. **View Results:**
   - Processed data and metadata are stored in the Qdrant vector database.
   - Use the provided scripts to visualize and debug the results.