# Smart Guard AI

Smart Guard AI is a real-time AI-based face surveillance and attendance monitoring system. It uses YOLOv8 for face detection, FaceNet for face embedding generation, and an SVM classifier for known/unknown face identification. The system can detect authorized users, identify unknown/intruder faces, send Telegram alerts, store attendance data in SQL Server, and display live monitoring data through a Flask web dashboard.

---

## Features

- Real-time face detection using YOLOv8-face
- Face recognition using FaceNet embeddings
- SVM-based known person classification
- Unknown/intruder detection using confidence and centroid-based verification
- Attendance logging with arrival and departure time
- SQL Server database integration
- Telegram alert support for intruder detection
- Flask-based web dashboard
- Real-time performance logging
- Training and real-time analysis reports
- Graph generation for accuracy, confidence, FPS, latency, and predictions

---

## Project Workflow

```text
Camera/Webcam Input
        ↓
YOLOv8 Face Detection
        ↓
Face Crop and Preprocessing
        ↓
FaceNet Embedding Extraction
        ↓
SVM Classification
        ↓
Centroid Verification
        ↓
Known / Unknown / Intruder Decision
        ↓
Database Logging + Dashboard + Telegram Alert
```

---

## Technologies Used

- Python
- OpenCV
- YOLOv8
- FaceNet
- Scikit-learn
- Flask
- SQL Server
- PyODBC
- NumPy
- Pandas
- Matplotlib
- Telegram Bot API

---

## Project Structure

```text
Smart-Guard-Ai/
│
├── analysis_report/          # Training and real-time analysis reports
├── dataset/                  # Face dataset folder
├── intruder_images/          # Captured intruder images
├── logs/                     # Detection, performance, and frame logs
├── model/                    # Trained model files
├── templates/                # Flask dashboard HTML files
├── utils/                    # Utility files
│
├── analysis_security.py      # Generates analysis reports and graphs
├── clear_logs.py             # Clears old log files
├── collect_dataset.py        # Captures face images for dataset creation
├── config.py                 # Project configuration
├── database.py               # SQL Server database operations
├── dbtest.py                 # Database connection testing
├── flask_app.py              # Flask dashboard application
├── generate_centroid.py      # Generates centroid data for verification
├── retrain_job.py            # Model retraining script
├── security_system.py        # Main real-time AI security system
├── telegram_send.py          # Telegram alert handling
├── train_model.py            # Model training script
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Burribhanutejareddy/Smart-Guard-Ai.git
cd Smart-Guard-Ai
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Virtual Environment

For Windows PowerShell:

```bash
.venv\Scripts\activate
```

For Linux/Mac:

```bash
source .venv/bin/activate
```

### 4. Install Requirements

```bash
pip install -r requirements.txt
```

If training gives missing package errors, install these also:

```bash
pip install albumentations tqdm scipy pillow torchvision
```

---

## Dataset Preparation

Create a `dataset` folder with one subfolder for each known person.

Example:

```text
dataset/
├── Sasidhar/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Mohan/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Suresh/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

You can collect images using:

```bash
python collect_dataset.py
```

Recommended: capture 30 to 50 clear images per person under different lighting, angles, and facial positions.

---

## Model Training

Train the face recognition model using:

```bash
python train_model.py
```

This script:

- Detects faces using YOLOv8-face
- Extracts FaceNet embeddings
- Trains an SVM classifier
- Saves trained model files inside the `model/` folder
- Generates training evaluation reports inside `analysis_report/`

Generated model files:

```text
model/face_classifier.pkl
model/scaler.pkl
model/labels.pkl
```

---

## Generate Face Centroids

After training the model, generate centroid data:

```bash
python generate_centroid.py
```

This creates:

```text
model/centroids.pkl
```

Centroids are used to improve known/unknown verification and reduce false recognition.

---

## Database Configuration

This project supports SQL Server for attendance logging.

Default database configuration is available in `config.py`:

```python
DB_CONFIG = {
    "driver": "ODBC Driver 17 for SQL Server",
    "server": "localhost\\SQLEXPRESS",
    "database": "AI_SECURITY",
    "schema": "dbo",
    "persons_table": "Persons",
    "attendance_table": "Attendance"
}
```

You can also configure database values using environment variables:

```bash
DB_DRIVER
DB_SERVER
DB_NAME
DB_SCHEMA
DB_PERSONS_TABLE
DB_ATTENDANCE_TABLE
DB_USERNAME
DB_PASSWORD
```

Test database connection:

```bash
python dbtest.py
```

---

## Running the Real-Time Security System

Run the main system:

```bash
python security_system.py
```

To run with test recording details:

```bash
python security_system.py --test-recording --true-people "Sasidhar,Mohan" --true-label authorized
```

Press `q` to stop the camera window.

---

## Running the Dashboard

The Flask dashboard starts automatically when running the main security system. You can also run it separately:

```bash
python flask_app.py
```

Open the dashboard in your browser:

```text
http://localhost:5000
```

---

## Analysis and Reports

Generate training and real-time analysis reports:

```bash
python analysis_security.py --mode both
```

Generate only real-time analysis:

```bash
python analysis_security.py --mode realtime
```

Generate only training analysis:

```bash
python analysis_security.py --mode training
```

Reports and graphs are saved in:

```text
analysis_report/
analysis_report/graphs/
```

---

## Performance Summary

Based on the included training report:

```text
Training Accuracy      : 95.58%
Macro Precision        : 96.11%
Macro Recall           : 95.59%
Macro F1-score         : 95.59%
Weighted Precision     : 96.10%
Weighted Recall        : 95.58%
Weighted F1-score      : 95.58%
```

Real-time analysis includes:

- Face detection logs
- Prediction confidence
- FPS and latency
- Known/unknown decisions
- FAR and FRR calculation
- Frame-level summary
- Prediction distribution graphs

---

## Telegram Alert Setup

The system supports Telegram alerts for intruder detection.

For safety, do not hard-code your bot token and chat ID in public repositories. Use environment variables or a private config file.

Example environment variables:

```bash
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## Important Security Note

Before pushing this project to a public GitHub repository:

- Remove Telegram bot tokens from `config.py`
- Do not upload private datasets
- Do not upload trained model files if they contain private identity data
- Do not upload database passwords
- Add sensitive folders/files to `.gitignore`

Recommended `.gitignore` entries:

```text
.venv/
__pycache__/
dataset/
intruder_images/
model/
logs/
security.log
*.pkl
*.pt
.env
```

---

## Future Improvements

- Add a proper login system for the dashboard
- Add role-based admin access
- Improve anti-spoofing using liveness detection
- Add multi-camera support
- Add automatic model retraining from dashboard
- Add email alerts along with Telegram alerts
- Improve UI design for analytics dashboard
- Package the project as a desktop application

---

## Author

Developed as part of an AI-based face recognition security system project.

---

## Disclaimer

This project is intended for academic and research purposes. For real-world deployment, proper privacy, security, consent, and legal guidelines must be followed.
