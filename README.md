<div align="center">

<img src="veiledguard%20logo.jpg" alt="VeiledGuard Logo" width="250" />

# VeiledGuard: Edge Vision Analytics 🛡️

**A privacy-first edge AI agent that monitors screen presence and autonomously blocks distractions using real-time computer vision.**

<p align="center">
  <img src="https://img.shields.io/badge/Python_3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=yolo&logoColor=black" alt="YOLOv8" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
</p>

</div>

<br/>

## 📌 Overview

**VeiledGuard** is a lightweight, locally-hosted edge vision system designed to enhance digital privacy and focus. By leveraging the `YOLOv8` nano model (`yolov8n.pt`), the system actively monitors the user's environment through a smart camera feed. If an unauthorized person approaches the screen or the user walks away, the system autonomously triggers a screen-blocking mechanism to protect sensitive information. 

Everything runs **100% locally on the edge**—no video feeds or images are ever sent to the cloud, ensuring absolute privacy.

<br/>

## 🚀 Core Features

- 👁️ **Real-Time Vision Analytics (`vision.py`):** Utilizes lightweight Convolutional Neural Networks (CNNs) to detect human presence and spatial proximity with near-zero latency.
- 📸 **Smart Camera Module (`SmartCam.py`):** Efficiently manages the webcam stream, optimizing frames for the YOLO model to prevent CPU/GPU bottlenecking.
- 🛑 **Autonomous Screen Blocker (`ScreenBlocker.py`):** Acts as the enforcement agent. When the vision module detects a privacy breach (e.g., someone looking over your shoulder), it instantly overlays a protective block on the display.

<br/>

## 🛠️ System Architecture

1. **Input:** The `SmartCam` captures local video frames.
2. **Processing:** The frames are passed to the `vision` script, which runs inference using the pre-trained `yolov8n.pt` weights.
3. **Logic:** The bounding box outputs are analyzed for specific conditions (number of people, distance, eye-gaze direction).
4. **Action:** If a threat condition is met, the `ScreenBlocker` is deployed.

<br/>


## 👥 Development Team

This research and engineering project was developed by:

| Name | Institution | GitHub |
| :--- | :--- | :--- |
| **Osama Ibn Mahfuz** | Shanghai University of Engineering Science | [![GitHub](https://img.shields.io/badge/GitHub-OsamaIM-111111?style=flat-square&logo=github&logoColor=white)](https://github.com/OsamaIM) |

<br/>

## 💻 Installation & Setup

### Prerequisites
Ensure you have Python 3.8+ installed on your machine along with a functioning webcam.

### 1. Clone the Repository
```bash
git clone [https://github.com/OsamaIM/edge-vision-analytics.git](https://github.com/OsamaIM/edge-vision-analytics.git)
cd edge-vision-analytics
