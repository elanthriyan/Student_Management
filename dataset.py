import cv2
import os

DATASET_PATH = "dataset"
NUM_IMAGES = 50  # Images per person

person_name = input("Enter the name of the person: ").strip().lower()
save_dir = os.path.join(DATASET_PATH, person_name)
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img_count = 0
print(f"Saving images to {save_dir}. Press 'q' to quit early.")

while img_count < NUM_IMAGES:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Expand the face rectangle a bit for better cropping (optional)
        pad = int(0.1 * w)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
        face_img = frame[y1:y2, x1:x2]
        img_filename = os.path.join(save_dir, f"{person_name}_{img_count+1:03d}.jpg")
        cv2.imwrite(img_filename, face_img)
        img_count += 1
        print(f"Saved {img_filename}")
        if img_count >= NUM_IMAGES:
            break

    cv2.imshow("Face Dataset Capture", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

print(f"Collected {img_count} images for {person_name}.")
cap.release()
cv2.destroyAllWindows()
