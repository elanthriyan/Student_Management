# Face Recognition Attendance System using FaceNet and MTCNN

![Python](https://img.shields.io/badge/Python-3.6%2B-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-%3E%3D4.0-green.svg)
![facenet-pytorch](https://img.shields.io/badge/facenet--pytorch-%E2%9C%94%EF%B8%8F-orange.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

This project implements a real-time face recognition-based attendance system using the `facenet-pytorch` library with pre-trained FaceNet and MTCNN models. The system includes:

- **Dataset Creation** from webcam
- **Face Embedding Extraction** using FaceNet
- **Face Detection** using MTCNN
- **Face Recognition** by cosine similarity
- **Attendance Recording** in a text file

---

## 🔧 Requirements

- Python 3.7+
- OpenCV
- NumPy
- PIL (Pillow)
- facenet-pytorch
- torch
- torchvision

You can install the required packages using:

```bash
pip install opencv-python numpy pillow facenet-pytorch torch torchvision
```

## 📁 Folder Structure
```bash
.
├── dataset/               # Stores collected face images
│   ├── person1/
│   ├── person2/
│   └── ...
├── db_faces.pkl           # Cached face embeddings and labels
├── model.py               # Main face recognition and attendance system
├── create_dataset.py      # Script to collect face images per person
├── attendance_list.txt    # Attendance output file
```
## 🖼️ Step 1: Create Dataset

Run the following script to capture 50 face images from your webcam:
```bash
python dataset.py
```
- Enter the name of the person when prompted.
- It will capture and save the cropped face images in the dataset/ folder under a subfolder named after the person.
- Press q to quit early if needed.
## 🧠 Step 2: Run Face Recognition Model
```bash
python model.py
```
You'll be prompted to choose the video input method:

1. Webcam - for live face detection and recognition.
2. Recorded Video - for face recognition on a video file.
- Detected faces will be matched with your dataset, and the system will display:
- Green box with name and confidence if the face is recognized.
- Red box with "Unknown" if the face is not in the database.
## ✅ Attendance Recording

- Recognized names will be recorded and deduplicated.
- At the end of the session, the list will be printed and saved in attendance_list.txt.
## 📦 Caching

- The face embeddings are stored in a cache file (db_faces.pkl) after first run.
- This avoids redundant processing of the dataset on subsequent runs.
## ⚙️ Threshold

The similarity threshold for recognition is defined as:
```python
THRESHOLD = 0.75  # You can fine-tune this for better accuracy
```
You can experiment with values between 0.6 to 0.9 depending on lighting, resolution, and dataset quality.

## 📌 Notes

- Use clear and well-lit facial images for best results.
- This implementation works on CPU by default. To enable GPU, change the device to 'cuda' in the code:
```bash
device = 'cuda'
```

## Demo
![Demo](demo.gif)
