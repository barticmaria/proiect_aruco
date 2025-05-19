# ArUco & Visual Odometry Demo Application

This project is a real-time computer vision application built with Python and OpenCV, focused on:

- Detecting **ArUco markers** in one or two camera streams
- Estimating **camera inclination** based on marker orientation
- Drawing 3D **axis overlays** on detected markers
- Performing **monocular visual odometry** (VO) with ORB feature tracking
- Visualizing the **2D trajectory** of the camera in real time
- Displaying a **GUI menu (Tkinter)** to switch between functionalities

---

##  Functionalities

-  **Single-camera ArUco Detection**  
  Detects and displays marker ID and its 3D position (X, Y, Z).

-  **Dual-camera View with ArUco Overlay & Inclination**  
  Displays markers from both cameras, estimates pitch angle, and overlays 3D axes.

-  **Real-time Visual Odometry**  
  Tracks camera movement using ORB + Essential Matrix. Displays the 2D motion trajectory.

-  **Combined Feature Matching + Trajectory View**  
  Shows live ORB feature matching between frames along with real-time trajectory.

-  **Multithreaded GUI Menu**  
  Tkinter-based menu with threaded execution to avoid freezing the UI.

---

##  Known Limitations

-  No stereo depth estimation (despite dual camera availability)
-  No SLAM integration (only Visual Odometry is implemented)
-  Trajectory can drift due to lack of scale or loop closure correction
-  No automatic camera calibration — user must provide `.npy` calibration files
-  Only tested on Windows with OpenCV + numpy

---

##  How to Run

1. **Install dependencies**  
   (You should use a virtual environment)
   ```bash
   pip install opencv-python numpy

2. **Calibrate your camera(s) and save the following files in the project root:**
   ```bash
   cam_matrix.npy
   dist_coeffs.npy
   (optional for dual cam) dist_coeffs1.npy
3. **Run the application:**
    ```bash
   python meniu_1.py
    
This project is the result of laboratory work for the "Proiectarea Algoritmilor" (PA) course at the
Faculty of Automation, Computers and Electronics (FACE), University of Craiova (http://ace.ucv.ro/).

It is intended for educational use only and not as a production-grade solution.
