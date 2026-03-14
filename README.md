# 🔍 Criminal Identification System using Face Recognition

An end‑to‑end **Machine Learning + Computer Vision project** that
identifies criminals from images using **Face Detection and Face
Recognition techniques**.

The system detects faces from images, extracts facial embeddings, trains
a machine learning classifier, and predicts whether the person matches
an existing criminal record.

------------------------------------------------------------------------

# 📌 Table of Contents

-   Overview
-   Problem Statement
-   Technologies Used
-   Algorithms Used
-   Project Workflow
-   Modules
-   Project Structure
-   Dataset
-   Model Performance
-   How to Run the Project
-   Future Improvements
-   Author

------------------------------------------------------------------------

# 🚀 Overview

The **Criminal Identification System** is designed to assist law
enforcement in identifying criminals using facial recognition
techniques.

The system: - Detects faces from images - Extracts facial features -
Trains a machine learning model - Identifies criminals from uploaded
test images

It uses **pre‑trained deep learning models for face embedding** and
**Support Vector Machine (SVM)** for classification.

------------------------------------------------------------------------

# 🧩 Problem Statement

Identifying criminals manually from large datasets is difficult and
time‑consuming.

This project automates the process by: - Detecting faces automatically -
Learning facial features - Classifying criminals using machine learning

------------------------------------------------------------------------

# 🛠 Technologies Used

  Category             Tools
  -------------------- ---------------------
  Programming          Python
  Computer Vision      OpenCV
  Deep Learning        Keras
  Face Detection       MTCNN
  Feature Extraction   FaceNet
  Machine Learning     Scikit‑learn
  GUI                  Tkinter
  Visualization        Matplotlib, Seaborn
  Version Control      Git & GitHub

------------------------------------------------------------------------

# 🤖 Algorithms Used

### 1️⃣ MTCNN

Used for **face detection** in images.

### 2️⃣ FaceNet

Used to generate **facial embeddings (feature vectors)** from detected
faces.

### 3️⃣ Support Vector Machine (SVM)

Used to **classify faces and identify criminals**.

------------------------------------------------------------------------

# 🔄 Project Workflow

1.  Upload criminal dataset
2.  Detect faces using MTCNN
3.  Extract face embeddings using FaceNet
4.  Normalize feature vectors
5.  Split dataset into train and test sets
6.  Train SVM classifier
7.  Evaluate performance using accuracy, precision, recall and F1 score
8.  Upload test image for criminal identification

------------------------------------------------------------------------

# 🧩 System Modules

### Upload Criminal Dataset

Loads the dataset and initializes **MTCNN** and **FaceNet** models.

### Preprocess Dataset

-   Detect faces
-   Extract embeddings
-   Normalize features
-   Split data into training and testing sets.

### Train SVM Model

Trains the **Support Vector Machine classifier** using facial
embeddings.

### Performance Graph

Displays model performance metrics such as: - Accuracy - Precision -
Recall - F1 Score

### Criminal Identification

Uploads a test image and predicts the matching criminal along with
probability.

------------------------------------------------------------------------

# 📊 Dataset

The dataset contains **multiple images of criminals organized in
folders**.

Each folder represents a unique criminal identity and contains several
face images used for training.

Example:

    dataset
     ├── criminal_1
     │   ├── img1.jpg
     │   ├── img2.jpg
     │
     ├── criminal_2
     │   ├── img1.jpg
     │   ├── img2.jpg

------------------------------------------------------------------------

# 📈 Model Performance

The trained model achieves approximately:

-   Accuracy: \~97%
-   Precision: High precision classification
-   Recall: Effective detection of criminals
-   F1 Score: Balanced model performance

Performance is visualized using: - Confusion Matrix - Performance Graph

------------------------------------------------------------------------

# ▶️ How to Run the Project

## 1. Clone Repository

    git clone https://github.com/sravalika-gudimalla/criminal-identification-system.git
    cd criminal-identification-system

## 2. Install Dependencies

    pip install -r requirements.txt

## 3. Run Application

    python CriminalIdentification.py

## 4. Steps inside Application

1.  Upload Criminal Dataset
2.  Preprocess Dataset
3.  Train SVM Model
4.  View Comparison Graph
5.  Upload test image for criminal identification

------------------------------------------------------------------------

# 🚀 Future Improvements

-   Real‑time CCTV criminal detection
-   Deep learning based classification models
-   Web‑based interface using Flask or Streamlit
-   Larger dataset for improved accuracy
-   Integration with law enforcement databases

------------------------------------------------------------------------

# ✨ Author

**Sravalika Gudimalla**\
B.Tech -- Computer Science & Engineering

Machine Learning \| Data Science \| AI Enthusiast

GitHub: https://github.com/sravalika-gudimalla
