﻿# 🐾 Animal Image Classifier

A Streamlit web app that classifies animal images using a pre-trained VGG16-based deep learning model.

live app => https://animal-classifier-yuvraj.streamlit.app/

model download => https://drive.google.com/file/d/1KyP5xHClTGxZRCHm0Ha7OvT1zVD533E-/view?usp=drive_link

---

## 🔍 Features

- Upload any animal image (JPG, PNG, BMP, GIF, etc.)
- Predicts the animal class with confidence
- Displays a bar chart of class probabilities
- Downloads the model file from Google Drive automatically (no local model needed)

---

## 🧠 Model

- Based on **VGG16** pre-trained on ImageNet
- Fine-tuned for **15 animal categories**
- Model saved as `best_model.keras` and hosted on Google Drive
- https://drive.google.com/file/d/1KyP5xHClTGxZRCHm0Ha7OvT1zVD533E-/view?usp=drive_link

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yuvraj8433/Animal-Classifier.git
cd Animal-Classifier
