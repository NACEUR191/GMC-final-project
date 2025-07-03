import streamlit as st
import os
import cv2
from PIL import Image
import numpy as np
from pathlib import Path
from keras_facenet import FaceNet
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from zipfile import ZipFile

# App setup
st.set_page_config(page_title="Face Recognition Classifier", layout="centered")
st.title("🤖 Face Recognition with FaceNet & SVM")

# Load HAAR cascade model
cascade_path = st.text_input("Paste the HAAR cascade XML file path:", "haarcascade_frontalface_default.xml")

@st.cache_resource
def load_cascade(xml_path):
    if os.path.exists(xml_path):
        return cv2.CascadeClassifier(xml_path)
    return None

face_cascade = load_cascade(cascade_path)

# Load FaceNet embedding model
@st.cache_resource
def load_facenet_model():
    return FaceNet()

embedder = load_facenet_model()

# Functions
def extract_face(filename, required_size=(160, 160)):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, required_size)
    return face

def load_faces(directory):
    faces = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        face = extract_face(path)
        if face is not None:
            faces.append(face)
    return faces

def load_dataset(parent_folder):
    X, y = [], []
    for person_dir in os.listdir(parent_folder):
        path = os.path.join(parent_folder, person_dir)
        if os.path.isdir(path):
            faces = load_faces(path)
            labels = [person_dir] * len(faces)
            X.extend(faces)
            y.extend(labels)
    return np.array(X), np.array(y)

def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = model.embeddings(samples)
    return yhat[0]

# User Input: Directory Uploads
st.subheader("📁 Upload Train & Test Image Folders")
train_dir = st.text_input("Path to Training Images Directory")
test_dir = st.text_input("Path to Testing Images Directory")

if st.button("🧠 Train Recognition Model"):
    if not face_cascade:
        st.error("Invalid HAAR cascade path. Please check and try again.")
    else:
        st.info("Loading and embedding training data...")
        trainX, trainy = load_dataset(train_dir)
        trainX_embed = np.array([get_embedding(embedder, face) for face in trainX])

        st.info("Loading and embedding test data...")
        testX, testy = load_dataset(test_dir)
        testX_embed = np.array([get_embedding(embedder, face) for face in testX])

        # Train SVM
        if len(trainX_embed) == 0 or len(trainy) == 0:
            st.error("Training data is empty. Please check that valid face images exist in the training directory.")
        elif len(trainX_embed) != len(trainy):
            st.error(f"Inconsistent training data: found {len(trainX_embed)} embeddings and {len(trainy)} labels.")
        else:
            clf = SVC(kernel='linear', probability=True)
            clf.fit(trainX_embed, trainy)


        # Predict and evaluate
        preds = clf.predict(testX_embed)
        acc = accuracy_score(testy, preds)
        report = classification_report(testy, preds)

        st.success(f"✅ Model trained. Accuracy on test set: {acc:.2f}")
        st.text("\nClassification Report:")
        st.code(report)

        # Save model and embeddings
        np.savez_compressed("face_recognition_embeddings.npz", 
                            trainX=trainX_embed, trainy=trainy, 
                            testX=testX_embed, testy=testy)

        st.download_button("📥 Download Dataset (.npz)", 
                           data=open("face_recognition_embeddings.npz", "rb"), 
                           file_name="face_recognition_embeddings.npz")

st.markdown("---")
st.markdown("**Instructions**")
st.markdown("""
1. Paste the path to the HAAR cascade XML file.
2. Provide local paths to training and test image folders structured like: `dataset/train/person1`, `dataset/train/person2`, etc.
3. Click the button to process, embed, and train the SVM classifier.
4. Download the dataset with embedded vectors.
5. Ensure `keras-facenet` and `scikit-learn` are installed.
""")
