# Tricking Motion Classification System

A computer vision and machine learning pipeline that classifies aerial martial arts tricks from video using pose estimation and engineered biomechanical features.

The current system recognizes:

- **540 kick**
- **Cheat 720**
- **Cheat 900**

It processes each clip with YOLO Pose, extracts a compact set of biomechanical summary features, filters unreliable pose detections, and classifies the movement with a Random Forest model.

---

## Results

Current prototype performance on the pose-validated dataset:

- **84.3% mean accuracy** across 100 repeated stratified cross-validation folds
- **86% out-of-fold accuracy**
- **0.86 macro F1**
- **57 pose-validated clips**
- **78% clip acceptance rate** after quality filtering

The reported accuracy applies only to clips that pass pose-quality validation. The dataset is still small, so these results should be treated as prototype performance rather than universal real-world accuracy.

---

## Demo Clips

Three example clips are included in the repository.

Suggested structure:

```text
├── input1.mp4 (cheat 900)
├── input2.mp4 (540 kick)
└── input3.mp4 (cheat 720)
```

---

## How It Works

```text
Input video
    ↓
Frame capture at 30 FPS
    ↓
YOLO pose estimation
    ↓
Takeoff, airborne, landing, and rotation analysis
    ↓
Pose-quality filtering
    ↓
Clip-level feature extraction
    ↓
Random Forest classification
```

The classifier does not receive raw frames or full landmark sequences. Each clip is summarized into a small set of interpretable biomechanical features.

---

## Features

The current classifier uses:

| Feature | Description |
|---|---|
| `in_air_frames` | Number of analyzed frames between takeoff and landing |
| `max_hip_rotation` | Maximum observed in-air hip rotation |
| `takeoff_foot` | Detected takeoff foot |
| `landing_foot` | Detected landing foot |
| `left_ft_ht` | Left foot vertical position at landing, normalized by body height |

Foot values use this encoding:

| Value | Meaning |
|---:|---|
| `0` | Left foot |
| `1` | Right foot |
| `2` | Both feet |
| `-1` | Not detected |

Distances are normalized relative to estimated body height. Hip-rotation geometry is calculated after converting normalized landmark coordinates into pixel coordinates so that different video aspect ratios do not distort angles.

---

## Pose-Quality Filtering

Extreme tricking movements are difficult for generic pose-estimation models because of:

- Motion blur
- Crossed limbs
- Occlusion
- Back-facing poses
- Rapid twisting
- Unusual airborne body positions
- Legs overlapping or collapsing into the same predicted location

The pipeline automatically flags clips when:

```python
red_flag = (
    in_air_frames < 8
    or in_air_frames > 35
    or max_hip_rotation > 600
)
```

Clips are also manually rejected during dataset validation when the pose output is clearly unusable, such as:

- Both legs collapsing into the same location
- Major lower-body landmark hallucinations
- Incorrect landing geometry
- Implausible hip-rotation spikes
- Failed airborne triggering

Rejected clips are treated as failed measurements, not classifier mistakes.

---

## Project Structure

A typical repository layout is:

```text
tricking-cv-yolo/
├── examples/                    # Example clips
├── input_videos/                # Raw input clips
├── output_videos/               # Annotated output clips
├── capture.py                   # Video capture and FPS handling
├── pose.py                      # YOLO pose-estimation wrapper
├── analysis.py                  # Biomechanical feature extraction
├── overlay.py                   # Skeleton and metric visualization
├── app.py                       # Video-processing entry point
├── train.py                     # Model training and evaluation
├── dataset.csv                  # Extracted clip-level features
├── tricking_classifier.joblib   # Saved classifier artifact
├── requirements.txt
└── README.md
```

Your exact filenames may differ.

---

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install ultralytics opencv-python numpy pandas scikit-learn joblib
```

Or, if a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

---

## Running Video Analysis

Run the project entry point:

```bash
python3 app.py
```

The processing pipeline should:

1. Load a video clip.
2. Sample frames at the target frame rate.
3. Run YOLO pose estimation.
4. Detect takeoff and landing.
5. Estimate in-air hip rotation.
6. Extract clip-level features.
7. Save an annotated output video.
8. Return the extracted metrics and quality flag.

Example output:

```python
{
    "in_air_frames": 18,
    "max_hip_rotation": 372.4,
    "takeoff_foot": 1,
    "landing_foot": 0,
    "left_ft_ht": 0.41,
    "red_flag": False
}
```

---

## Training the Classifier

Run:

```bash
python3 train.py
```

The training script should:

- Load the feature dataset
- Exclude red-flagged clips
- Split features and class labels
- Train a Random Forest classifier
- Evaluate a held-out test set
- Run repeated stratified cross-validation
- Print precision, recall, F1, and confusion matrices

Because the dataset is small, repeated cross-validation is more reliable than reporting only one train/test split.

---

## Saving the Model

Save the trained model and feature order together:

```python
import joblib

FEATURES = [
    "in_air_frames",
    "max_hip_rotation",
    "takeoff_foot",
    "landing_foot",
    "left_ft_ht",
]

artifact = {
    "model": model,
    "features": FEATURES,
    "version": "1.0",
}

joblib.dump(artifact, "tricking_classifier.joblib")
```

Saving the feature order matters because the Random Forest expects every new input in the same column order used during training.

---

## Loading and Using the Model

```python
import joblib
import pandas as pd

artifact = joblib.load("tricking_classifier.joblib")

model = artifact["model"]
features = artifact["features"]

clip_metrics = {
    "in_air_frames": 18,
    "max_hip_rotation": 372.4,
    "takeoff_foot": 1,
    "landing_foot": 0,
    "left_ft_ht": 0.41,
}

X = pd.DataFrame(
    [[clip_metrics[name] for name in features]],
    columns=features,
)

prediction = model.predict(X)[0]
probabilities = model.predict_proba(X)[0]

print("Prediction:", prediction)

for class_name, probability in zip(model.classes_, probabilities):
    print(f"{class_name}: {probability:.1%}")
```

The inference pipeline should match training as closely as possible, including:

- YOLO model and weights
- Target frame rate
- Inference resolution
- Coordinate conversion
- Body-height normalization
- Takeoff and landing thresholds
- Pose-quality rules
- Feature names and order

---

## Data Visualization

This project includes a visualization component implemented in `overlay.py`, which plots points for each detected landmark and connects them to form a skeleton overlay on the video frames. Additionally, it displays key metrics such as hip rotation in degrees and airborne status directly on the video. This visualization is effective for real-time or frame-by-frame analysis, allowing users to verify the accuracy of pose estimation and observe how the system interprets the athlete's movements.

### Why This Is Effective

- **Immediate Feedback**: The overlay provides a clear, visual representation of the detected skeleton and metrics, making it easier to understand how the system processes the video.
- **Debugging**: It helps identify specific frames where pose estimation might fail, such as incorrect limb positions or missing landmarks.
- **Interpretability**: Displaying metrics like hip rotation in degrees directly on the video makes the system's calculations more transparent.

### Why This Isn't Effective

While the overlay is useful for analyzing individual clips, it does not address the challenge of analyzing a large batch of clips to pinpoint where pose estimation errors are occurring. For example, if the pose model consistently fails under certain conditions (e.g., motion blur, occlusion, or extreme body positions), manually reviewing each clip with the overlay is time-consuming and inefficient. A more scalable solution, such as automated error detection or batch-level data visualization, would be needed to identify systemic issues in pose estimation across a dataset.

---

## Current Limitations

- The validated dataset contains only 57 clips.
- Results may not generalize to new athletes, camera angles, environments, or recording styles.
- Generic pose estimators can hallucinate landmarks during extreme tricking movements.
- The system currently recognizes only three classes.
- `max_hip_rotation` is a 2D observable estimate, not true 3D rotation.
- Pose-quality validation still includes manual review.
- The classifier depends on the feature extractor producing reliable measurements.
- Real-time mobile deployment will likely require a smaller pose model or model optimization.

---

## Roadmap

- Add more athletes, sessions, and camera angles
- Add more trick classes, including tornado kick
- Fine-tune YOLO Pose on tricking-specific frames to reduce bad pose estimations
- Automate pose-quality and anomaly detection
- Improve temporal landmark consistency
- Detect and repair short pose-estimation failures
- Add confidence-based prediction rejection
- Benchmark smaller YOLO Pose models for mobile use
- Export the classifier for mobile or edge deployment
- Separate real-time feedback from deeper offline analysis

---

## Motivation

Most pose-estimation systems are designed for ordinary movement, fitness exercises, or common sports positions. Tricking introduces extreme rotation, crossed limbs, rapid kicks, unusual silhouettes, and heavy occlusion.

This project explores how far an interpretable and lightweight pipeline can go by combining:

- Pose estimation
- Biomechanical feature engineering
- Quality filtering
- Anomaly detection
- Classical machine learning

The long-term goal is to support automated trick recognition and useful coaching feedback from ordinary video.

---

## Tech Stack

- Python
- OpenCV
- Ultralytics YOLO Pose
- NumPy
- pandas
- scikit-learn
- joblib

---

