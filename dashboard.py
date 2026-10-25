
from pathlib import Path
import json
import pandas as pd
import streamlit as st

# ---------------- PROJECT PATHS ----------------

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "hand_gestures.csv"
MODEL_FILE = BASE_DIR / "model" / "gesture_model.pkl"
METRICS_FILE = BASE_DIR / "model" / "evaluation_metrics.json"

PROJECT_FILES = [
    "hand_detection.py",
    "collect_data.py",
    "add_samples.py",
    "train_model.py",
    "live_gesture_recognition.py",
    "desktop_gesture_control.py",
    "gesture_ppt_control.py",
    "dashboard.py",
]

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Hand Gesture AI | Project Dashboard",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- CUSTOM STYLING ----------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 1.7rem;
        border-radius: 18px;
        background: linear-gradient(125deg, #172554, #164e63);
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        margin-bottom: 0.5rem;
        font-size: 2.2rem;
    }

    .hero p {
        color: #dbeafe;
        font-size: 1rem;
        margin-bottom: 0.3rem;
    }

    .section-note {
        color: #64748b;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- HEADER ----------------

st.markdown(
    """
    <div class="hero">
        <h1>✋ Hand Gesture AI</h1>
        <p>Real-Time Hand Gesture Recognition Using Machine Learning</p>
        <p>Computer vision · Gesture classification · Desktop interaction</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "An interactive project dashboard for exploring the gesture dataset, "
    "checking the trained model, and understanding the desktop-control system."
)

# ---------------- LOAD DATASET ----------------

dataset = None
dataset_error = None

if DATA_FILE.exists():
    try:
        dataset = pd.read_csv(DATA_FILE)

        if "label" not in dataset.columns:
            dataset_error = "The CSV does not contain a 'label' column."
            dataset = None

    except Exception as error:
        dataset_error = str(error)

model_exists = MODEL_FILE.exists()

# ---------------- LOAD MODEL INFORMATION ----------------

model_name = "Not available"
model_features = "—"
model_classes = []

if model_exists:
    try:
        import joblib

        saved_model = joblib.load(MODEL_FILE)

        if isinstance(saved_model, dict):
            classifier = saved_model.get("model")
            saved_features = saved_model.get("feature_names")
            saved_classes = saved_model.get("classes")
        else:
            classifier = saved_model
            saved_features = None
            saved_classes = None

        if classifier is not None:
            model_name = type(classifier).__name__

            if saved_features is not None:
                model_features = len(saved_features)
            else:
                model_features = getattr(
                    classifier, "n_features_in_", "—"
                )

            if saved_classes is not None:
                model_classes = list(saved_classes)
            elif hasattr(classifier, "classes_"):
                model_classes = list(classifier.classes_)

    except Exception as error:
        model_name = "Could not load"
        st.warning(
            "The saved model file exists, but its details could not "
            f"be loaded: {error}"
        )

# ---------------- SUMMARY METRICS ----------------

total_samples = len(dataset) if dataset is not None else 0

gesture_count = (
    dataset["label"].nunique()
    if dataset is not None
    else 0
)

total_features = (
    len(dataset.columns) - 1
    if dataset is not None
    else 0
)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Dataset Samples", f"{total_samples:,}")
m2.metric("Gesture Classes", gesture_count)
m3.metric(
    "Trained Model",
    "Available" if model_exists else "Not found",
)
m4.metric("Input Features", total_features if dataset is not None else "—")

st.divider()

# ---------------- NAVIGATION TABS ----------------

overview_tab, dataset_tab, model_tab, performance_tab, demo_tab = st.tabs(
    [
        "🏠 Overview",
        "📊 Dataset Explorer",
        "🧠 Model Details",
        "📈 Performance",
        "🎮 Demo Guide",
    ]
)

# ================= OVERVIEW =================

with overview_tab:
    st.subheader("Project Overview")

    st.write(
        "The system uses a webcam to capture hand movements. "
        "MediaPipe extracts 21 hand landmarks, which are converted "
        "into numerical features. A trained machine learning model "
        "classifies predefined hand gestures."
    )

    left, right = st.columns(2)

    with left:
        st.markdown("#### Technologies Used")
        st.markdown(
            """
            - **Python:** Main programming language
            - **OpenCV:** Webcam image processing
            - **MediaPipe:** Hand landmark detection
            - **Scikit-learn:** Random Forest classification
            - **Streamlit:** Interactive project dashboard
            - **PyAutoGUI:** Mouse movement, clicks and scrolling
            """
        )

    with right:
        st.markdown("#### Project Pipeline")

        st.markdown(
            """
            1. Capture a hand using the webcam.
            2. Detect hand landmarks.
            3. Extract numerical features.
            4. Predict the gesture using the trained model.
            5. Map supported gestures to computer actions.
            """
        )

    st.info(
        "This dashboard is a project overview and analysis interface. "
        "Live webcam recognition and desktop control run as separate "
        "Python programs."
    )

    st.markdown("#### Project Files")

    file_status = pd.DataFrame(
        [
            {
                "File": filename,
                "Status": (
                    "Available"
                    if (BASE_DIR / filename).is_file()
                    else "Missing"
                ),
            }
            for filename in PROJECT_FILES
        ]
    )

    st.dataframe(
        file_status,
        use_container_width=True,
        hide_index=True,
    )

# ================= DATASET =================

with dataset_tab:
    st.subheader("Gesture Dataset Explorer")

    if dataset is None:
        st.warning(
            "Dataset could not be loaded. Check that "
            "data/hand_gestures.csv exists and contains a 'label' column."
        )

        if dataset_error:
            st.caption(dataset_error)

    else:
        st.write(
            f"The dataset contains **{total_samples:,} samples** "
            f"and **{total_features} numerical features** per sample."
        )

        counts = dataset["label"].value_counts().rename_axis(
            "Gesture"
        ).reset_index(name="Samples")

        left, right = st.columns([1.3, 1])

        with left:
            st.markdown("#### Samples per Gesture")
            st.bar_chart(
                counts.set_index("Gesture"),
                y="Samples",
            )

        with right:
            st.markdown("#### Sample Distribution")
            st.dataframe(
                counts,
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Preview the first 10 dataset rows"):
            st.dataframe(
                dataset.head(10),
                use_container_width=True,
                hide_index=True,
            )

        st.download_button(
            "Download Dataset CSV",
            data=DATA_FILE.read_bytes(),
            file_name="hand_gestures.csv",
            mime="text/csv",
        )

# ================= MODEL DETAILS =================

with model_tab:
    st.subheader("Machine Learning Model")

    st.write("**Algorithm:** Random Forest Classifier")
    st.write(f"**Saved model status:** {'Available' if model_exists else 'Not found'}")
    st.write(f"**Loaded classifier:** {model_name}")
    st.write(f"**Input feature count:** {model_features}")

    if model_classes:
        st.write(
            "**Recognized classes:** "
            + ", ".join(str(value) for value in model_classes)
        )

    st.markdown("#### How the Model Works")

    st.markdown(
        """
        - A labeled dataset provides examples of hand gestures.
        - The model learns patterns from the training samples.
        - New landmark features are passed to the trained classifier.
        - The classifier predicts one of the learned gesture categories.
        """
    )

    st.warning(
        "Model confidence and test accuracy are different. "
        "The dashboard does not claim a verified accuracy percentage "
        "because the current training script does not save independent "
        "evaluation results for this dashboard."
    )

# ================= MODEL PERFORMANCE =================

with performance_tab:
    st.subheader("Model Performance — Fresh Test Session")

    if not METRICS_FILE.exists():
        st.info(
            "Evaluation results not found. Run evaluate_model.py "
            "locally and save model/evaluation_metrics.json first."
        )
    else:
        try:
            with open(METRICS_FILE, "r", encoding="utf-8") as file:
                results = json.load(file)

            st.caption(results.get("evaluation_type", "Evaluation results"))

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Test Accuracy",
                f"{results['accuracy'] * 100:.2f}%"
            )
            col2.metric(
                "Test Samples",
                results["test_samples"]
            )
            col3.metric(
                "Gesture Classes",
                len(results["classes"])
            )

            report = pd.DataFrame.from_dict(
                results["per_class"],
                orient="index"
            )
            report.index.name = "Gesture"

            st.subheader("Gesture-wise Performance")

            st.dataframe(
                report.style.format({
                    "precision": "{:.2%}",
                    "recall": "{:.2%}",
                    "f1-score": "{:.2%}",
                    "support": "{:.0f}",
                }),
                use_container_width=True
            )

            st.subheader("Precision, Recall and F1-score")

            st.bar_chart(
                report[["precision", "recall", "f1-score"]]
            )

            st.subheader("Confusion Matrix")

            matrix = pd.DataFrame(
                results["confusion_matrix"],
                index=results["classes"],
                columns=results["classes"]
            )

            matrix.index.name = "Actual Gesture"
            matrix.columns.name = "Predicted Gesture"

            st.dataframe(matrix, use_container_width=True)

            st.caption(
                "Rows represent actual gestures; columns represent "
                "predicted gestures. Off-diagonal values indicate "
                "misclassifications."
            )

            st.warning(
                "These results apply to the recorded test session "
                "and do not guarantee the same accuracy in all conditions."
            )

        except Exception as error:
            st.error(f"Could not load evaluation results: {error}")


# ================= DEMONSTRATION GUIDE =================

with demo_tab:
    st.subheader("How to Demonstrate the Project")

    st.markdown(
        """
        **1. Hand detection**

        Run `python hand_detection.py` to demonstrate webcam-based
        hand landmark detection.

        **2. Live gesture recognition**

        Run `python live_gesture_recognition.py` to show the trained
        classifier recognising supported gestures.

        **3. Desktop interaction**

        Run `python desktop_gesture_control.py` to demonstrate cursor
        movement, pinch-based double-clicking and scrolling.

        **4. PowerPoint control**

        Run `python gesture_ppt_control.py` to test the configured
        PowerPoint actions separately.
        """
    )

    st.info(
        "Only one webcam-based program should run at a time. "
        "Close an active camera program before starting another."
    )

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Hand Gesture Recognition using Machine Learning | "
    "Educational mini-project"
)
