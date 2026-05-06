import os, cv2, pickle, random
import numpy as np
from tqdm import tqdm
import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import albumentations as A

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo = YOLO("yolov8n-face.pt")
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

dataset_path = "dataset"
report_dir = "analysis_report"
embeddings, labels = [], []

# Augmentations (applied to face crops)
aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=10, p=0.3)
])

def align_and_crop(img, box, margin=0.2):
    h, w = img.shape[:2]
    x1,y1,x2,y2 = box
    dx = int((x2-x1)*margin)
    dy = int((y2-y1)*margin)
    x1, y1 = max(0, x1-dx), max(0, y1-dy)
    x2, y2 = min(w, x2+dx), min(h, y2+dy)
    return img[y1:y2, x1:x2]

with torch.no_grad():
    for person in os.listdir(dataset_path):
        ppath = os.path.join(dataset_path, person)
        if not os.path.isdir(ppath): continue
        for img_name in os.listdir(ppath):
            img_path = os.path.join(ppath, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
            res = yolo(img)[0]
            if len(res.boxes) == 0: continue
            # choose largest face box
            boxes = [list(map(int, b)) for b in res.boxes.xyxy.cpu().numpy()]
            boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
            face = align_and_crop(img, boxes[0])
            if face.size == 0: continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (160,160))
            # apply augmentation randomly
            if random.random() < 0.6:
                face = aug(image=face)['image']
            face_t = torch.tensor(face).permute(2,0,1).float()
            face_t = fixed_image_standardization(face_t).unsqueeze(0).to(device)
            emb = facenet(face_t).cpu().numpy()[0]
            embeddings.append(emb)
            labels.append(person)

# Convert and normalize
X = np.array(embeddings)
y = np.array(labels)

if len(X) == 0:
    raise RuntimeError("No face embeddings were created. Check dataset images and YOLO face detections.")

classes, class_counts = np.unique(y, return_counts=True)
if len(classes) < 2:
    raise RuntimeError("Training needs at least 2 people/classes in the dataset.")
if np.min(class_counts) < 2:
    raise RuntimeError("Each class needs at least 2 detected faces for a stratified train/validation split.")

scaler = StandardScaler()
X = scaler.fit_transform(X)  # helps SVM/cosine

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.2,
    random_state=RANDOM_STATE,
)

# Hyperparameter tune SVM
param_grid = {'C':[0.1,1,10], 'kernel':['linear']}
cv_folds = min(5, int(np.min(np.unique(y_train, return_counts=True)[1])))
if cv_folds >= 2:
    svc = GridSearchCV(SVC(probability=True), param_grid, cv=cv_folds, n_jobs=-1, verbose=1)
else:
    svc = SVC(C=1, kernel="linear", probability=True)
svc.fit(X_train, y_train)
best_params = getattr(svc, "best_params_", {"C": 1, "kernel": "linear"})

print("Best params:", best_params)
print("Val score:", svc.score(X_val, y_val))

# Evaluation report
y_pred = svc.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    y_val, y_pred, average="macro", zero_division=0
)
precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
    y_val, y_pred, average="weighted", zero_division=0
)
report = classification_report(y_val, y_pred, labels=classes, zero_division=0)
matrix = confusion_matrix(y_val, y_pred, labels=classes)

print("\nEvaluation Report")
print("=================")
print(f"Accuracy          : {accuracy:.4f}")
print(f"Macro Precision   : {precision_macro:.4f}")
print(f"Macro Recall      : {recall_macro:.4f}")
print(f"Macro F1-score    : {f1_macro:.4f}")
print(f"Weighted Precision: {precision_weighted:.4f}")
print(f"Weighted Recall   : {recall_weighted:.4f}")
print(f"Weighted F1-score : {f1_weighted:.4f}")
print("\nPer-class report:")
print(report)
print("Confusion matrix labels:", list(classes))
print(matrix)

# Save artifacts
os.makedirs("model", exist_ok=True)
with open("model/face_classifier.pkl","wb") as f:
    pickle.dump(svc, f)
with open("model/scaler.pkl","wb") as f:
    pickle.dump(scaler, f)
with open("model/labels.pkl","wb") as f:
    pickle.dump(list(np.unique(y)), f)

# Save evaluation artifacts
os.makedirs(report_dir, exist_ok=True)
report_path = os.path.join(report_dir, "training_evaluation_report.txt")
matrix_path = os.path.join(report_dir, "training_confusion_matrix.csv")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("Training Evaluation Report\n")
    f.write("==========================\n\n")
    f.write(f"Device            : {device}\n")
    f.write(f"Dataset path      : {dataset_path}\n")
    f.write(f"Total embeddings  : {len(X)}\n")
    f.write(f"Classes           : {', '.join(classes)}\n")
    f.write(f"Train samples     : {len(X_train)}\n")
    f.write(f"Validation samples: {len(X_val)}\n")
    f.write(f"Best params       : {best_params}\n")
    f.write(f"CV folds          : {cv_folds}\n\n")
    f.write("Summary Metrics\n")
    f.write("---------------\n")
    f.write(f"Accuracy          : {accuracy:.4f}\n")
    f.write(f"Macro Precision   : {precision_macro:.4f}\n")
    f.write(f"Macro Recall      : {recall_macro:.4f}\n")
    f.write(f"Macro F1-score    : {f1_macro:.4f}\n")
    f.write(f"Weighted Precision: {precision_weighted:.4f}\n")
    f.write(f"Weighted Recall   : {recall_weighted:.4f}\n")
    f.write(f"Weighted F1-score : {f1_weighted:.4f}\n\n")
    f.write("Per-class Report\n")
    f.write("----------------\n")
    f.write(report)
    f.write("\nConfusion Matrix\n")
    f.write("----------------\n")
    f.write("Labels: " + ", ".join(classes) + "\n")
    f.write(str(matrix))
    f.write("\n")

header = "," + ",".join(classes)
matrix_rows = [
    f"{label}," + ",".join(str(value) for value in row)
    for label, row in zip(classes, matrix)
]
with open(matrix_path, "w", encoding="utf-8") as f:
    f.write(header + "\n")
    f.write("\n".join(matrix_rows))
    f.write("\n")

print(f"\nSaved evaluation report: {report_path}")
print(f"Saved confusion matrix : {matrix_path}")
