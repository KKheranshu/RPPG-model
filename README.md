<div align="center">

# 🫀 Contactless Heart Rate Monitor
### Remote Photoplethysmography (rPPG) using Webcam

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange?style=for-the-badge)](https://mediapipe.dev)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

**Estimate your heart rate in real time — just using your webcam. No wearable. No sensor. No contact.**

![Demo](demo.gif)

</div>

---

## 📌 About The Project

Built a contactless pulse detection system using Python and OpenCV that estimates heart rate from subtle skin color changes captured by a standard webcam. Implemented the **POS (Plane-Orthogonal-to-Skin) algorithm** for motion artifact cancellation, **Butterworth bandpass filtering**, **FFT-based BPM extraction**, and a **sliding window BPM timeline** — achieving real-time heart rate monitoring without any physical sensor.

> ✅ Validated against **Fossil Gen 6 smartwatch** readings — achieved a mean absolute error of **±4–5 BPM** under controlled conditions.

---

## 🎯 Key Features

- 🟢 **Real-time BPM** displayed live on webcam feed
- 🧠 **MediaPipe FaceMesh** — 468 landmarks for precise forehead tracking
- 🔬 **POS Algorithm** — uses all 3 RGB channels to cancel motion noise
- 〰️ **Butterworth Bandpass Filter** — isolates 50–150 BPM frequency range
- 📊 **FFT-based BPM** — more robust than simple peak counting
- 🛡️ **Motion Artifact Rejection** — flags bad frames automatically
- 🪟 **Sliding Window BPM** — timeline of heart rate over recording
- 💾 **Auto-saves** results to CSV and PNG plot

---

## ⚙️ How It Works

```
📷 Webcam Input
      │
      ▼
🧠 MediaPipe FaceMesh (468 landmarks)
      │
      ▼
🟩 Forehead ROI Extraction (mean R, G, B per frame)
      │
      ▼
🛡️ Motion Artifact Rejection (>15px shift → flag frame)
      │
      ▼
📉 Detrending (remove slow lighting drift)
      │
      ▼
〰️ Butterworth Bandpass Filter (50–150 BPM)
      │
      ▼
🔬 POS Algorithm (multi-channel motion cancellation)
      │
      ▼
🌊 Gaussian Smoothing (σ=2)
      │
      ▼
📊 FFT BPM (primary) + 🔺 Peak BPM (cross-check)
      │
      ▼
🪟 Sliding Window BPM Timeline
      │
      ▼
🫀 Final Heart Rate Estimate (BPM)
```

| Step | Method | Purpose |
|------|--------|---------|
| 1 | Camera Capture | Raw frames at ~30 FPS with real timestamps |
| 2 | FaceMesh | Locate forehead via 468 3D landmarks |
| 3 | ROI Extraction | Mean R,G,B from forehead region per frame |
| 4 | Motion Rejection | Skip frames with large ROI position shifts |
| 5 | Detrending | Subtract mean → remove DC illumination offset |
| 6 | Bandpass Filter | Butterworth order 3, keep 50–150 BPM only |
| 7 | POS Algorithm | Project RGB onto skin-orthogonal plane |
| 8 | Smoothing | Gaussian filter to remove frame-level jitter |
| 9 | FFT BPM | Dominant frequency in HR band = BPM |
| 10 | Peak BPM | RR-interval counting as secondary check |
| 11 | Sliding Window | 10s rolling FFT for BPM over time |

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Validation Device | Fossil Gen 6 Smartwatch |
| Mean Absolute Error | ±4–5 BPM |
| Recommended Capture Duration | 15+ seconds |
| Working Condition | Good indoor lighting, minimal movement |
| FPS (typical webcam) | ~25–30 FPS |

---

## 🛠️ Tech Stack

| Library | Version | Used For |
|---------|---------|----------|
| Python | 3.8+ | Core language |
| OpenCV | 4.9 | Camera capture, frame processing |
| MediaPipe | 0.10 | Face landmark detection |
| NumPy | 1.26 | Signal arrays, FFT |
| SciPy | 1.12 | Bandpass filter, peak detection |
| Pandas | 2.2 | Data logging to CSV |
| Matplotlib | 3.8 | Plots and visualization |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Webcam
- Good lighting (avoid backlight)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/KKheranshu/rppg-heart-rate-monitor.git

# 2. Navigate into the folder
cd rppg-heart-rate-monitor

# 3. (Recommended) Create a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Run


```

** — Jupyter Notebook:**
```bash
jupyter notebook rppg_notebook.ipynb
```

---

## 📖 Usage Guide

1. Run the script and **allow webcam access**
2. Sit with your **face well-lit** (face a window or lamp)
3. **Stay still** — minimize head movement
4. Wait at least **15 seconds** for an accurate reading
5. Press **Q** to stop recording
6. View the BPM result and signal plots
7. Check `rppg_data.csv` for raw data and `rppg_results.png` for plots

> ⚠️ **Tips for best accuracy:**
> - Use good frontal lighting
> - Avoid moving your head during capture
> - Capture for 20–30 seconds for most stable result
> - Avoid strong backlight behind you

---

## 📁 Project Structure

```
rppg-heart-rate-monitor/
│
├── rppg_pipeline.py        # Main Python script
├── rppg_notebook.ipynb     # Jupyter Notebook version
├── requirements.txt        # All dependencies
├── .gitignore              # Files to exclude from git
├── README.md               # This file
├── demo.gif                # Demo recording
│
└── assets/                 # (optional) screenshots, plots
    └── pipeline.png
```

---

## 🧪 Algorithm Detail — POS

The **POS (Plane-Orthogonal-to-Skin)** algorithm by de Haan & Jeanne (2013) is the core of this project:

```
# Normalize each channel by its mean
R_n = R / mean(R)
G_n = G / mean(G)
B_n = B / mean(B)

# Two projection vectors
S1 =  G_n - B_n           → tangent to skin tone
S2 = -2·R_n + G_n + B_n   → orthogonal axis

# Scale factor to minimize noise
α = std(S1) / std(S2)

# Final pulse signal
POS = S1 - α · S2
```

This projects the RGB signal onto a plane perpendicular to skin-tone variation — separating heartbeat signal from motion and lighting noise.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👤 Author

Kheranshu Kashyap 
[![LinkedIn](https://www.linkedin.com/in/kheranshu-kashyap-796a94310/)
[![GitHub](https://github.com/KKheranshu)

---

## 🙏 References

- de Haan, G., & Jeanne, V. (2013). *Robust pulse rate from chrominance-based rPPG.* IEEE Transactions on Biomedical Engineering.
- MediaPipe FaceMesh — [Google MediaPipe](https://mediapipe.dev)

---

<div align="center">
⭐ If you found this useful, please star the repo!
</div>
