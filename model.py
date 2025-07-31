# import os
# import cv2
# import numpy as np
# from PIL import Image
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from numpy.linalg import norm

# DB_PATH = "face_db"
# attendance = set()
# THRESHOLD = 0.75  # You may tune this for your setup

# # Load models
# mtcnn = MTCNN(keep_all=True, device='cpu')  # Use device='cuda' for GPU
# resnet = InceptionResnetV1(pretrained='vggface2').eval()

# # 1. Prepare reference embeddings
# db_embeddings = []
# db_labels = []
# for fname in os.listdir(DB_PATH):
#     if fname.lower().endswith((".jpg", ".png")):
#         img = Image.open(os.path.join(DB_PATH, fname)).convert("RGB")
#         face = mtcnn(img)
#         if face is not None:
#             if face.ndim == 4:
#                 face = face[0]  # Only the first face if batch
#             db_embeddings.append(resnet(face.unsqueeze(0)).detach().cpu().numpy()[0])
#             db_labels.append(os.path.splitext(fname)[0])
# db_embeddings = np.stack(db_embeddings)

# # 2. Start video capture
# cap = cv2.VideoCapture(0)  # Use filename for video file
# print("Press 'q' to quit attendance system.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     pil_img = Image.fromarray(rgb)

#     # Detect all faces and their bounding boxes
#     boxes, probs = mtcnn.detect(pil_img)
#     if boxes is not None:
#         faces_aligned = mtcnn.extract(pil_img, boxes, save_path=None)
#         for box, face in zip(boxes, faces_aligned):
#             if face.ndim == 3:
#                 face = face.unsqueeze(0)
#             emb = resnet(face).detach().cpu().numpy()[0]
#             # Cosine similarity to all reference embeddings
#             sims = [np.dot(emb, db_emb) / (norm(emb) * norm(db_emb)) for db_emb in db_embeddings]
#             if len(sims) > 0:
#                 idx = np.argmax(sims)
#                 label = db_labels[idx]
#                 conf = sims[idx]
#                 (x1, y1, x2, y2) = [int(v) for v in box]
#                 if conf > THRESHOLD:
#                     attendance.add(label)
#                     name_disp = f"{label} ({conf:.2f})"
#                     color = (0,255,0)
#                 else:
#                     name_disp = "Unknown"
#                     color = (0,0,255)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, name_disp, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     cv2.imshow("FaceNet Multi-Attendance", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

# print("Attendance List:")
# print("\n".join(sorted(attendance)))

import os
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from numpy.linalg import norm

DB_PATH = "dataset"
attendance = set()
THRESHOLD = 0.75  # Tune if needed

# Load models
mtcnn = MTCNN(keep_all=True, device='cpu')  # Change to 'cuda' if GPU is available
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# 1. Prepare reference embeddings using folder names as labels
db_embeddings = []
db_labels = []

for person_name in os.listdir(DB_PATH):
    person_dir = os.path.join(DB_PATH, person_name)
    if os.path.isdir(person_dir):
        print(f"🔍 Processing folder: {person_name}")
        for fname in os.listdir(person_dir):
            if fname.lower().endswith((".jpg", ".png")):
                img_path = os.path.join(person_dir, fname)
                print(f"  🖼️ Loading image: {img_path}")
                img = Image.open(img_path).convert("RGB")
                face = mtcnn(img)
                if face is not None:
                    if face.ndim == 4:
                        face = face[0]  # Use first face if batch
                    embedding = resnet(face.unsqueeze(0)).detach().cpu().numpy()[0]
                    db_embeddings.append(embedding)
                    db_labels.append(person_name)  # Use folder name
                else:
                    print(f"  ❌ No face detected in {img_path}")

# Handle empty embedding list
if len(db_embeddings) == 0:
    raise ValueError("No embeddings found! Make sure 'face_db' has folders with valid face images.")

db_embeddings = np.stack(db_embeddings)

# 2. Start video capture
cap = cv2.VideoCapture(0)  # Use a filename for video file if needed
print("\n📷 Camera started. Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # Detect all faces and bounding boxes
    boxes, probs = mtcnn.detect(pil_img)
    if boxes is not None:
        faces_aligned = mtcnn.extract(pil_img, boxes, save_path=None)
        for box, face in zip(boxes, faces_aligned):
            if face.ndim == 3:
                face = face.unsqueeze(0)
            emb = resnet(face).detach().cpu().numpy()[0]
            # Cosine similarity
            sims = [np.dot(emb, db_emb) / (norm(emb) * norm(db_emb)) for db_emb in db_embeddings]
            if len(sims) > 0:
                idx = np.argmax(sims)
                label = db_labels[idx]
                conf = sims[idx]
                (x1, y1, x2, y2) = [int(v) for v in box]
                if conf > THRESHOLD:
                    attendance.add(label)
                    name_disp = f"{label} ({conf:.2f})"
                    color = (0, 255, 0)
                else:
                    name_disp = "Unknown"
                    color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name_disp, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("FaceNet Multi-Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 3. Print and Save attendance list
print("\n📋 Attendance List:")
print("\n".join(sorted(attendance)))

with open("attendance_list.txt", "w") as f:
    for name in sorted(attendance):
        f.write(name + "\n")

print("\n✅ Attendance saved to 'attendance_list.txt'")
