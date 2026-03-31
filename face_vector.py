import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization

# -------------------------------
# 1. Load FaceNet Model
# -------------------------------
model = InceptionResnetV1(pretrained='vggface2').eval()

# -------------------------------
# 2. Load Image
# -------------------------------
img_path = r"S:\final_year_project\ai_security-main\dataset\Sasidhar\Sasidhar_p0_0006.jpg"
img = cv2.imread(img_path)

# Convert BGR → RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -------------------------------
# 3. Use YOUR YOLO Bounding Box
# -------------------------------
# Example (from your calculation)
x1, y1 = 306, 262
x2, y2 = 440, 467

face = img_rgb[y1:y2, x1:x2]

# -------------------------------
# 4. Resize to FaceNet size
# -------------------------------
face = cv2.resize(face, (160, 160))

# Convert to PIL Image
face_pil = Image.fromarray(face)

# -------------------------------
# 5. Convert to Tensor
# -------------------------------
face_tensor = torch.tensor(np.array(face_pil)).float()

# Shape: (160,160,3) → (3,160,160)
face_tensor = face_tensor.permute(2, 0, 1)

# Add batch dimension
face_tensor = face_tensor.unsqueeze(0)

# -------------------------------
# 6. Normalize (IMPORTANT)
# -------------------------------
face_tensor = fixed_image_standardization(face_tensor)

# -------------------------------
# 7. Generate Embedding
# -------------------------------
with torch.no_grad():
    embedding = model(face_tensor)

# Convert to numpy
embedding = embedding.numpy()[0]

# -------------------------------
# 8. L2 Normalize
# -------------------------------
embedding = embedding / np.linalg.norm(embedding)

# -------------------------------
# 9. Output
# -------------------------------
print("Embedding Shape:", embedding.shape)
print("First 100 values:", embedding[:100])