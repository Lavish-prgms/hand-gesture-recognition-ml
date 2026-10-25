
# Hand Gesture Recognition and PC Control Using Machine Learning

A webcam-based Machine Learning project that recognizes predefined hand gestures and enables hands-free desktop interaction.

## Live Dashboard

Explore the project dashboard here:

**[Hand Gesture AI Dashboard](https://hand-gesture-recognition-ml.streamlit.app/)**

The dashboard displays project information, dataset statistics, model details, and fresh-session evaluation results.

## Project Overview

This project integrates computer vision, hand landmark detection, machine learning, and desktop automation.

MediaPipe detects 21 hand landmarks from webcam frames. The landmarks are converted into numerical features, which are processed by a trained Random Forest classifier to recognize predefined gestures.

The recognized gestures and hand movements can then be mapped to supported computer actions.

## Features

- Real-time hand detection using a webcam
- Detection of 21 hand landmarks
- Collection of a labeled gesture dataset
- Gesture classification using Random Forest
- Live gesture recognition with confidence scores
- Experimental cursor movement, clicking, and scrolling
- Gesture-based desktop and PowerPoint control
- Interactive Streamlit dashboard
- Model evaluation with accuracy, precision, recall, F1-score, and a confusion matrix

Desktop-control features are experimental and may require further tuning for reliable operation.

## Technologies Used

- Python
- OpenCV
- MediaPipe
- Scikit-learn
- NumPy and Pandas
- Joblib
- PyAutoGUI
- Streamlit
- PyWin32 for Windows PowerPoint automation

## Recognized Gestures

The dataset includes five gesture classes:

1. Open Palm
2. Fist
3. Peace
4. Thumbs Up
5. Pointing

The actions associated with each gesture depend on the selected control program.

## Project Workflow

1. Capture frames from the webcam.
2. Detect hand landmarks using MediaPipe.
3. Extract numerical features from the landmarks.
4. Train a Random Forest classifier using labeled examples.
5. Predict gestures from live webcam input.
6. Map recognized gestures or hand movements to supported computer actions.
7. Evaluate model performance using a separate test dataset.

## Project Structure

```text
hand-gesture-recognition-ml/
├── data/
│   ├── hand_gestures.csv
│   └── hand_gestures_test.csv
├── model/
│   ├── gesture_model.pkl
│   └── evaluation_metrics.json
├── hand_detection.py
├── collect_data.py
├── add_samples.py
├── train_model.py
├── evaluate_model.py
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
git clone https://github.com/Lavish-prgms/hand-gesture-recognition-ml.git
cd hand-gesture-recognition-ml
```

### 2. Create a virtual environment

Python 3.10 is recommended for the local Windows setup.

```powershell
python -m venv .venv
```

Activate the environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## Run the Project

### Launch the dashboard

```powershell
python -m streamlit run dashboard.py
```

### Run hand detection

```powershell
python hand_detection.py
```

### Run live gesture recognition

```powershell
python live_gesture_recognition.py
```

### Run desktop gesture control

```powershell
python desktop_gesture_control.py
```

This program provides experimental cursor movement, pinch-based double-clicking, and scrolling controls.

### Run PowerPoint gesture control

```powershell
python gesture_ppt_control.py
```

PowerPoint control requires Microsoft PowerPoint to be installed and open.

Close other webcam-based programs before starting another to avoid camera conflicts.

## Model Evaluation

Run:

```powershell
python evaluate_model.py
```

The evaluation script collects a fresh webcam test session for the five predefined gestures. It then calculates:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The results are saved to:

`model/evaluation_metrics.json`

Evaluation results describe the recorded test session. They do not guarantee the same performance for different users, cameras, lighting conditions, or hand positions.

## Dashboard

The Streamlit dashboard contains five sections:

- **Overview:** Project introduction and workflow
- **Dataset Explorer:** Sample counts and dataset preview
- **Model Details:** Trained model information
- **Performance:** Evaluation metrics and confusion matrix
- **Demo Guide:** Instructions for running the project components

The online dashboard presents project information and evaluation results. Webcam-based recognition and desktop-control programs run locally on the user's computer.

## Applications

- Hands-free desktop interaction
- Presentation control
- Assistive computer interaction research
- Contactless user interfaces
- Human-computer interaction prototypes

These are potential applications; the current prototype has not been validated for every environment.

## Limitations

Recognition performance can vary with hand position, lighting, camera quality, and gesture similarity.

A high score on one test session does not establish universal real-world accuracy. Additional testing with independent users and recording conditions is required.

The system recognizes predefined gestures and is not a complete sign-language translation system.

## Future Improvements

- Evaluate recognition across different users and environments
- Improve cursor smoothing and scroll control
- Reduce accidental gesture activations
- Add more desktop applications
- Integrate voice commands with gesture control

## Project Type

B.Tech Mini-Project | Machine Learning | Computer Vision
