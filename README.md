# Face Recognition Using Machine Learning

## Overview

This project focuses on facial recognition using machine learning techniques for identification and authentication. It applies Principal Component Analysis (PCA) for dimensionality reduction and uses multiple classification models to achieve high accuracy in recognizing faces.
Face recognition is an advanced biometric technology that identifies and verifies individuals based on their facial features. This project implements facial recognition using machine learning techniques such as Principal Component Analysis (PCA) for feature extraction and Linear Discriminant Analysis (LDA), Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and Decision Trees for classification. The Olivetti Faces Dataset is used to train and evaluate the model.

## Features

Face Identification: Recognizes and classifies facial images.
Access Control: Enables secure authentication through facial recognition.
Surveillance: Identifies individuals in public spaces.
Identity Verification: Securely validates users in online services.
Attendance Monitoring: Automates attendance tracking.

## Technologies Used

Python
OpenCV (for image processing)
NumPy, Pandas(for data handling)
Scikit-learn(for machine learning models)
Matplotlib, Seaborn (for visualization)

## Dataset
Olivetti Faces Dataset: Contains 400 grayscale images of 40 distinct individuals.
Each person has 10 images captured under different lighting and facial expressions.
Datasets :
olivetti_faces_target.npy
olivetti_faces.npy


 1. **Data Preprocessing**

- Loaded the dataset and converted images into numerical arrays.
- Applied normalization and grayscale conversion for consistency.

2. **Feature Extraction Using PCA**

- Reduced dimensionality while retaining essential features.
- Selected 40 principal components for optimal performance.

3. **Model Training & Classification**

Applied various classifiers to recognize faces:

- **Linear Discriminant Analysis (LDA)**
- **Logistic Regression**
- **Gaussian Naive Bayes (NB)**
- **K-Nearest Neighbors (KNN)**
- **Decision Trees**
- **Support Vector Machines (SVM)**

## Results

- Achieved **98% accuracy** using **LDA**.
- PCA improved feature extraction and reduced computation time.
- **Confusion Matrix**: Demonstrated high precision and recall.
- **ROC Curves**: Showed strong classification performance.





  
