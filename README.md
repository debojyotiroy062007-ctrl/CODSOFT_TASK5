# Real-Time Face Detection System

A computer vision application using OpenCV's Haar Feature-based Cascade Classifiers to detect and track human faces via live webcam feed.

## Features
- **Real-Time Detection:** Live frontal face localization using pre-trained Haar Cascades.
- **Dynamic Overlays:** Real-time bounding boxes with dimension telemetry ( \times H$).
- **Live Counter:** Persistent overlay displaying the count of detected faces in the current frame.
- **Grayscale Processing:** Optimized frame conversion for high-FPS performance.
- **Clean Resource Handling:** Safe release of video capture devices and GUI windows on exit.

## Tech Stack
- Python 3.x
- OpenCV (\opencv-python\)
- NumPy

## Installation & Setup

1. **Clone the repository:**
   \\\ash
   git clone https://github.com/debojyotiroy062007-ctrl/CODSOFT_TASK5.git
   cd CODSOFT_TASK5
   \\\

2. **Install dependencies:**
   \\\ash
   pip install opencv-python numpy
   \\\

3. **Run the script:**
   \\\ash
   python face_detection_recognition.py
   \\\

## Controls
- Press **\q\** on the preview window to exit cleanly.

## Author
- **Debojyoti Roy**
