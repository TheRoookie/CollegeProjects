# 🚗 Car Parking Space Counter

## 📋 Project Overview
The **Car Parking Space Counter** is an intelligent computer vision application designed to monitor a parking lot via a video feed. It automatically detects, tracks, and classifies parking spaces as either "Empty" or "Occupied" in real-time. The system utilizes OpenCV for core image processing, integrates modern machine learning techniques, and features a user-friendly interface for inputting video feeds and visualizing the parking status.

---

## 🎯 Syllabus Topic Integration
This project is strategically designed to fulfill all required syllabus topics, demonstrating a comprehensive understanding of computer vision principles:

1. **Video Processing**
   * **Implementation:** The system uses `cv2.VideoCapture` to read video feeds (either live RTSP streams or pre-recorded `.mp4` files) frame-by-frame. It manages frame rates, processes the video stream in real-time, and outputs a continuous annotated video feed.
2. **Image Filtering**
   * **Implementation:** Before analyzing a frame, preprocessing techniques are applied to reduce noise and enhance features. This includes **Gaussian Blurring** to remove high-frequency noise and **Adaptive Thresholding** or **Canny Edge Detection** to highlight vehicle outlines and parking lines.
3. **Image Segmentation**
   * **Implementation:** The project allows users to define Regions of Interest (ROIs) for each parking space. Using bitwise operations and masking (`cv2.fillPoly` and `cv2.bitwise_and`), the system segments individual parking spots from the rest of the image so they can be analyzed independently.
4. **Object Detection**
   * **Implementation:** A pre-trained **YOLOv8 nano (`yolov8n.pt`)** model is used to detect the presence of vehicles within the broader video frame. YOLOv8n is chosen for its minimal footprint and high inference speed, making it suitable for real-time CPU-bound processing.
5. **Object Tracking**
   * **Implementation:** As cars move through the parking lot, an object tracker (e.g., Centroid Tracking or DeepSORT) assigns unique IDs to detected vehicles. This ensures the system understands trajectory and doesn't double-count a moving car before it parks.
6. **Image Classification**
   * **Implementation:** Once a parking space is segmented, the cropped ROI is passed to a classification algorithm (a lightweight CNN or an SVM). The classifier analyzes the visual features of the specific slot and outputs a binary classification: `Class 0 (Empty)` or `Class 1 (Occupied)`.

---

## 🛠️ Technology Stack
* **Core Language:** Python 3.9+
* **Computer Vision:** OpenCV (`cv2`)
* **Machine Learning/Deep Learning:** PyTorch + **Ultralytics YOLOv8 nano** (for vehicle detection), Scikit-Learn (for rule-based classification).
* **User Interface:** PyQt5 or Streamlit (for a clean, interactive desktop or web UI).
* **Data Manipulation:** NumPy, Pandas.

---

## 🖥️ User Interface (UI) Features
The application will feature a clear, intuitive UI divided into three main sections:
* **Input Panel:** Buttons to upload a video file or enter an IP camera URL. A tool to manually draw or upload bounding boxes (ROIs) for the parking spots.
* **Real-Time Dashboard:** A large central video player displaying the processed feed. Parking spots will be overlaid with colored bounding boxes (🟩 Green for Empty, 🟥 Red for Occupied).
* **Analytics Sidebar:** Live counters showing "Total Spaces," "Available Spaces," and "Occupied Spaces."

---

## 🚀 Implementation Plan

### Phase 1: Setup and Environment (Week 1)
* [ ] Initialize a Git repository for version control.
* [ ] Create a virtual environment (`venv` or `conda`).
* [ ] Install required dependencies (`opencv-python`, `numpy`, `PyQt5`/`streamlit`, etc.).
* [ ] Gather test data (overhead parking lot videos and corresponding coordinate files for the parking slots).

### Phase 2: Core Computer Vision Pipeline (Week 2)
* [ ] **Video Processing Module:** Write the main loop to read and display video frames using OpenCV.
* [ ] **ROI & Segmentation Module:** Implement a script to parse parking spot coordinates and apply masks to extract individual slots from the main frame.
* [ ] **Filtering Module:** Apply grayscale conversion, Gaussian blur, and thresholding to the segmented ROIs to prepare them for classification.

### Phase 3: Detection, Tracking & Classification (Week 3)
* [ ] **Detection Module:** Integrate **YOLOv8 nano (`yolov8n.pt`)** via the Ultralytics library to locate cars in the frame. Filter detections to vehicle classes (car, truck, bus) using COCO class IDs.
* [ ] **Tracking Module:** Implement a centroid tracking algorithm to assign IDs to moving vehicles, ensuring they are only processed when stationary in a spot.
* [ ] **Classification Module:** Train or implement a binary classifier. Pass the filtered, segmented ROIs through the classifier to determine if a car is currently taking up the space.

### Phase 4: UI Development and Integration (Week 4)
* [ ] Build the frontend using the chosen UI framework.
* [ ] Connect the backend CV pipeline to the UI so the video processes in a background thread and updates the UI in real-time.
* [ ] Implement the live counter logic (updating variables based on classification outputs).

### Phase 5: Testing, Optimization, and Documentation (Week 5)
* [ ] **Testing:** Run the system on various lighting conditions (day/night videos) to test the robustness of the Image Filtering and Classification.
* [ ] **Optimization:** Resize frames and optimize the inference loop to maintain a minimum of 24 FPS.
* [ ] **Documentation:** Finalize code comments, type hinting, and this README file.

---

## ⭐ Best Practices Followed
* **Modular Architecture:** The codebase will be split into logical modules (`video_handler.py`, `detector.py`, `classifier.py`, `ui.py`) rather than a single monolithic script.
* **Configuration Management:** Use a `config.yaml` or `.env` file for hardcoded variables (model paths, video paths, UI colors, threshold values).
* **Type Hinting & Docstrings:** All Python functions will use PEP-484 type hints and docstrings for readability and easier debugging.
* **Error Handling:** Implement `try-except` blocks around video stream reading and model inference to prevent the UI from crashing if a frame drops.
* **Non-Blocking UI:** The heavy OpenCV/ML processing will run on a separate worker thread to keep the user interface responsive.