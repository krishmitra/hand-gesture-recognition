# 🖐️ Real-Time Hand Gesture Recognition

> A beginner-friendly Computer Vision project using Python, OpenCV, and MediaPipe.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

This project uses your **webcam** to detect hands in real time and draw the **21 hand landmark points** tracked by Google's [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html) model.

### How does hand tracking work?

1. **Palm Detection** — A fast machine-learning model scans the image and finds bounding boxes around any hands present.
2. **Landmark Regression** — A second model zooms into each bounding box and predicts the exact 3D position of **21 keypoints** on the hand.
3. **Temporal Smoothing** — In video mode, MediaPipe reuses the previous frame's landmarks to guide the next prediction, keeping tracking smooth and fast.

### The 21 MediaPipe Landmarks

```
         8   12  16  20
         |   |   |   |
         7   11  15  19
    4    |   |   |   |
    |    6   10  14  18
    3    |   |   |   |
    |    5---9--13--17
    2    |
     \   |
      1  |
       \ |
        [0]  ← WRIST
```

| ID | Landmark        | ID | Landmark        |
|----|-----------------|-----|-----------------|
| 0  | WRIST           | 11 | MIDDLE_PIP      |
| 1  | THUMB_CMC       | 12 | MIDDLE_DIP      |
| 2  | THUMB_MCP       | 13 | RING_MCP        |
| 3  | THUMB_IP        | 14 | RING_PIP        |
| 4  | THUMB_TIP       | 15 | RING_DIP        |
| 5  | INDEX_MCP       | 16 | RING_TIP        |
| 6  | INDEX_PIP       | 17 | PINKY_MCP       |
| 7  | INDEX_DIP       | 18 | PINKY_PIP       |
| 8  | INDEX_TIP  ✋   | 19 | PINKY_DIP       |
| 9  | MIDDLE_MCP      | 20 | PINKY_TIP       |
| 10 | MIDDLE_PIP      |    |                 |

---

## 📁 Project Structure

```
hand-gesture-recognition/
│
├── main.py            # Entry point — opens webcam, runs the main loop
├── hand_tracker.py    # HandTracker class (reusable MediaPipe wrapper)
├── requirements.txt   # Python dependencies
├── README.md          # You are here!
└── assets/            # Screenshots / demo GIFs (optional)
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or newer — [Download here](https://www.python.org/downloads/)
- A working webcam

### Step-by-step setup

```bash
# 1. Clone this repository
git clone https://github.com/YOUR_USERNAME/hand-gesture-recognition.git
cd hand-gesture-recognition

# 2. (Recommended) Create a virtual environment
python -m venv venv

# Activate it:
#   Windows:
venv\Scripts\activate
#   macOS / Linux:
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt
```

> **Tip:** If `mediapipe` fails to install, try upgrading pip first:
> ```bash
> pip install --upgrade pip
> ```

---

## ▶️ How to Run

```bash
python main.py
```

- **Hold your hand** in front of the webcam — green dots and white lines will appear on your hand.
- The **info panel** (top-left) shows FPS, number of hands, and Left/Right detection.
- The **cyan circle** highlights your index fingertip with live coordinates.
- Press **`Q`** or **`Esc`** to quit.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📷 Real-time webcam feed | Smooth 30+ FPS processing |
| 🖐 21-point landmark detection | Full hand skeleton via MediaPipe |
| 🦴 Skeleton connections | Bones drawn between each joint |
| 📊 Live FPS counter | Green = good, Red = slow |
| 🤚 Hand count & Left/Right label | Detect up to 2 hands simultaneously |
| 👆 Index fingertip highlight | Live (x, y) pixel coordinates shown |
| 🪞 Mirrored feed | Natural selfie-camera feel |

---

## 🚀 Future Improvements

Here are some exciting upgrades you can build on top of this project:

### 1. 🔢 Finger Counter
Compare each fingertip landmark's y-position to its lower knuckle. If the tip is **above** the knuckle, the finger is **up** → count it!

### 2. 🖱️ Virtual Mouse Control
Map your index fingertip coordinates to your screen resolution using `pyautogui` to control the mouse cursor with your hand.

### 3. 🔊 Volume Control
Calculate the distance between your **thumb tip (4)** and **index fingertip (8)** using NumPy. Map that distance to system volume using the `pycaw` library (Windows) or `osascript` (macOS).

### 4. 🤙 Gesture Recognition
Encode the state of all 5 fingers as a binary tuple (e.g., `(1,0,0,0,0)` = thumbs up) and map each tuple to a named gesture like ✌️ Peace, 👌 OK, 🤘 Rock.

### 5. 🎮 Hand-Controlled Game
Build a simple Pygame application (e.g., Pong, Snake) controlled entirely by hand gestures.

---

## 📚 Learning Resources

- [MediaPipe Hands documentation](https://google.github.io/mediapipe/solutions/hands.html)
- [OpenCV Python tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html)

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and share.

---

## 🙋 Author

Made with ❤️ as a first AI/ML portfolio project.
If you found this helpful, please ⭐ the repository!
