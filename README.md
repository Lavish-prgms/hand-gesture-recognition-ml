
# Hand Gesture Recognition and PC Control Using Machine Learning

A webcam-based Machine Learning project that recognizes predefined hand gestures and supports hands-free desktop interaction.

## Project Overview

This project combines computer vision, hand landmark detection, machine learning, and desktop automation. It uses MediaPipe to detect hand landmarks and a trained Random Forest classifier to recognize predefined gestures.

## Features

- Real-time hand detection using a webcam
- Detection of 21 hand landmarks
- Collection and analysis of a labeled gesture dataset
- Gesture classification using Random Forest
- Live gesture recognition with a confidence score
- Experimental mouse movement, clicking, and scrolling
- Gesture-based desktop and PowerPoint control
- Streamlit dashboard for project information and dataset exploration

Some desktop-control features are experimental and may require further tuning for reliable operation.

## Technologies Used

- Python
- OpenCV
- MediaPipe
- Scikit-learn
- NumPy and Pandas
- PyAutoGUI
- Streamlit
- PyWin32 for Windows PowerPoint automation

## Recognized Gestures

The dataset contains these five gesture classes:

- Open Palm
- Fist
- Peace
- Thumbs Up
- Pointing

The actions associated with each gesture depend on the selected control program.

## Project Workflow

1. Capture webcam frames.
2. Detect hand landmarks using MediaPipe.
3. Extract numerical features from the landmarks.
4. Train a Random Forest classifier using labeled gesture samples.
5. Predict gestures from live webcam input.
6. Map recognized gestures or hand movements to supported computer actions.

## Project Structure

```text
Hand_Gesture_Recognition/
├── data/
│   └── hand_gestures.csv
├── model/
│   └── gesture_model.pkl
├── hand_detection.py
├── collect_data.py
├── add_samples.py
├── train_model.py
├── live_gesture_recognition.py
├── desktop_gesture_control.py
├── gesture_ppt_control.py
├── dashboard.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Hand_Gesture_Recognition
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the Project

### Launch the dashboard

```bash
python -m streamlit run dashboard.py
```

### Run live gesture recognition

```bash
python live_gesture_recognition.py
```

### Run desktop gesture control

```bash
python desktop_gesture_control.py
```

### Run PowerPoint gesture control

```bash
python gesture_ppt_control.py
```

Close other webcam-based programs before launching another to avoid camera conflicts.

## Applications

- Hands-free desktop interaction
- Presentation control
- Assistive computer interaction research
- Contactless user interfaces
- Human-computer interaction prototypes

## Limitations

Recognition performance can vary with hand position, lighting, camera quality, and gesture similarity. Independent testing is required to measure real-world accuracy. This prototype recognizes predefined gestures and is not a complete sign-language translation system.

## Future Improvements

- Improve recognition on unseen recordings
- Add more gestures
- Improve cursor smoothing and scroll control
- Integrate additional desktop applications
- Develop a unified voice-and-gesture assistant

## Project Type

B.Tech Mini-Project | Machine Learning | Computer Vision
