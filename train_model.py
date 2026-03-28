import os, cv2, pickle, random
import numpy as np
from tqdm import tqdm
import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
import albumentations as A

device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo = YOLO("yolov8n-face.pt")
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

dataset_path = "dataset"
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
scaler = StandardScaler()
X = scaler.fit_transform(X)  # helps SVM/cosine

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Hyperparameter tune SVM
param_grid = {'C':[0.1,1,10], 'kernel':['linear']}
svc = GridSearchCV(SVC(probability=True), param_grid, cv=5, n_jobs=-1, verbose=1)
svc.fit(X_train, y_train)

print("Best params:", svc.best_params_)
print("Val score:", svc.score(X_val, y_val))

# Save artifacts
os.makedirs("model", exist_ok=True)
with open("model/face_classifier.pkl","wb") as f:
    pickle.dump(svc, f)
with open("model/scaler.pkl","wb") as f:
    pickle.dump(scaler, f)
with open("model/labels.pkl","wb") as f:
    pickle.dump(list(np.unique(y)), f)
