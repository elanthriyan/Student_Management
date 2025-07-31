import os
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from numpy.linalg import norm
import pickle

DB_PATH = "dataset"
CACHE_FILE = "db_faces.pkl"
attendance = set()
THRESHOLD = 0.75  # Adjust as needed
device = 'cpu'    # Use 'cuda' if available

mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def process_reference_frames(img):
    face = mtcnn(img)
    embeddings = []
    if face is not None:
        if face.ndim == 4:
            for f in face:
                emb = resnet(f.unsqueeze(0)).detach().cpu().numpy()[0]
                embeddings.append(emb)
        else:
            emb = resnet(face.unsqueeze(0)).detach().cpu().numpy()[0]
            embeddings.append(emb)
    return embeddings

if os.path.exists(CACHE_FILE):
    print("✅ Loading cached face embeddings ...")
    with open(CACHE_FILE, 'rb') as f:
        data = pickle.load(f)
    db_embeddings = data['embeddings']
    db_labels = data['labels']
else:
    print("🔁 No cache found. Processing images and videos to extract embeddings ...")
    db_embeddings = []
    db_labels = []
    for person_name in os.listdir(DB_PATH):
        person_dir = os.path.join(DB_PATH, person_name)
        if os.path.isdir(person_dir):
            print(f"🔍 Processing: {person_name}")
            for fname in os.listdir(person_dir):
                fpath = os.path.join(person_dir, fname)
                ext = fname.lower().split('.')[-1]
                # Process images
                if ext in ("jpg", "jpeg", "png"):
                    img = Image.open(fpath).convert("RGB")
                    emb_list = process_reference_frames(img)
                    for emb in emb_list:
                        db_embeddings.append(emb)
                        db_labels.append(person_name)
                # Process videos
                elif ext in ("mp4", "avi", "mov", "mkv"):
                    cap = cv2.VideoCapture(fpath)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    N = max(1, frame_count // 10)  # Sample 10 frames per video
                    idx = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if idx % N == 0:
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            img = Image.fromarray(rgb)
                            emb_list = process_reference_frames(img)
                            for emb in emb_list:
                                db_embeddings.append(emb)
                                db_labels.append(person_name)
                        idx += 1
                    cap.release()
    if len(db_embeddings) == 0:
        raise ValueError("No embeddings found! Ensure dataset has valid image/video files.")
    db_embeddings = np.stack(db_embeddings)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump({'embeddings': db_embeddings, 'labels': db_labels}, f)
    print(f"✅ Cached {len(db_embeddings)} embeddings to '{CACHE_FILE}'.")

# --- User Selection: Live or Recorded Video ---
print("Select input method:")
print("1. Go Live (Use Webcam)")
print("2. Use Recorded Video")
choice = input("Enter 1 or 2: ").strip()

if choice == '1':
    video_source = 0  # Webcam
elif choice == '2':
    video_path = input("Enter the path to the recorded video file: ").strip()
    if not os.path.exists(video_path):
        print(f"Error: File '{video_path}' does not exist.")
        exit(1)
    video_source = video_path
else:
    print("Invalid option. Exiting.")
    exit(1)

cap = cv2.VideoCapture(video_source)
print("\n📷 Processing video stream. Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    boxes, probs = mtcnn.detect(pil_img)
    if boxes is not None:
        faces_aligned = mtcnn.extract(pil_img, boxes, save_path=None)
        for box, face in zip(boxes, faces_aligned):
            if face.ndim == 3:
                face = face.unsqueeze(0)
            emb = resnet(face).detach().cpu().numpy()[0]
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

# Print and save attendance
print("\n📋 Attendance List:")
print("\n".join(sorted(attendance)))

with open("attendance_list.txt", "w") as f:
    for name in sorted(attendance):
        f.write(name + "\n")

print("\n✅ Attendance saved to 'attendance_list.txt'")
