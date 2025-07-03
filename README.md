# GMC-final-project
Face Recognition with FaceNet and SVM
 Face Recognition App using FaceNet and SVM
This Streamlit app provides an end-to-end pipeline for face recognition using the FaceNet embedding model and an SVM classifier. It guides users through face detection using the HAAR cascade, embedding face images, and training a classifier—all through an intuitive interface.

🚀 Features
Upload and process structured training and test face image folders.

Use OpenCV’s HAAR cascade for face detection.

Generate face embeddings using the FaceNet model (keras-facenet).

Train a support vector machine (SVM) on embedded vectors.

Evaluate performance with accuracy and classification report.

Download the processed dataset as a .npz file for future use.

📂 Folder Structure Example
bash
Copier
Modifier
dataset/
├── train/
│   ├── person1/
│   ├── person2/
├── test/
│   ├── person1/
│   ├── person2/
🛠️ Requirements
Python

Streamlit

OpenCV (opencv-python)

scikit-learn

keras-facenet

📋 How to Use
Paste the path to the HAAR cascade XML file.

Enter paths to your training and testing directories.

Click "Train Recognition Model".

View performance metrics and download the .npz result file.
