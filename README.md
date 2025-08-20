![Background](https://github.com/user-attachments/assets/5dc6e3e3-0dd8-4368-a25e-c96bf54ecc35)



# 🧠 Brain Tumor Detection & Segmentation

This project focuses on **Brain Tumor Detection** and **Tumor Segmentation** using Deep Learning techniques.  
The system includes **two models**:

1. **Classification (Detection)** – Identifies whether an MRI image contains a **Tumor** or **No Tumor** using **ResNet50**.
   - Achieved Accuracy: **92.3%**

2. **Segmentation** – Locates and segments the tumor region from the MRI using **U-Net**.
   - Achieved Accuracy: **99.5%**

---

## 📂 Project Structure

```bash
├── models/              # Saved trained models
├── Models Notebooks/   # Jupyter Notebooks used for training
|                 ├── Detection/Brain_Tumor_Classifier_92_3_
|                 └──Segmentation/Brain_Tumor_Segmentation_Unet_99_5_
|
├── static/              # Static files (CSS , images)
├── templates/           # HTML templates for web app
├── Test/                # Test images
├── app.py               # Flask application
├── Background.jpg       # Background 
└── requirements.txt     # Required dependencies
``` 



## 🚀 Features
- **Tumor Detection:** Classifies MRI images as `Tumor` or `No Tumor`.
- **Tumor Segmentation:** Highlights the exact tumor region in MRI scans.
- **Web Application:** User-friendly interface built with **Flask**.
- **Deep Learning Models:**  
  - ResNet50 for classification.  
  - U-Net for segmentation.

---

## 📊 Results
- **ResNet50 (Detection):** `92.3% Accuracy`
- **U-Net (Segmentation):** `99.5% Accuracy`

---
## 📸 Example Outputs  

### 🔹 Detection (ResNet50)  
- **Input MRI → Tumor / No Tumor Prediction**
<img width="1920" height="1020" alt="Screenshot 2025-08-19 220503" src="https://github.com/user-attachments/assets/a0a562d2-a50a-4ce7-a656-c78f65d4e4c9" />

### 🔹 Segmentation (U-Net)  
- **Input MRI → Segmented Tumor Region**
<img width="1920" height="1020" alt="Screenshot 2025-08-19 220530" src="https://github.com/user-attachments/assets/8df45fc4-6f62-4087-b731-58cad2f67c6f" />


---

## 🛠️ Tech Stack  
- **Python**, **Flask**  
- **TensorFlow / Keras**  
- **ResNet50**, **U-Net**  
- **OpenCV**, **NumPy**, **Matplotlib**  

---

## 🙌 Acknowledgements  
- Thanks to **Allah** for making this project possible.  
- **Dataset:** Publicly available Brain MRI dataset from *Kaggle* & other medical sources.  
- Research inspiration from various deep learning papers on medical imaging.  

--- 
## ⚡ How It Works  
Simple steps of the process:  
1. **Preprocessing MRI images**  
2. **Classification** using ResNet50 (Tumor / No Tumor)  
3. **Segmentation** of tumor region using U-Net  
4. **Display results** in a user-friendly Flask web app  

---

## 🚧 Future Work / Improvements  
- Improve model accuracy with more training data  
- Experiment with other architectures (e.g., EfficientNet, Vision Transformers)  
- Deploy as a mobile or cloud-based app for doctors to use in real-time  

---

## 🏷️ Badges   

![Python](https://img.shields.io/badge/Python-3.9-blue)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0-orange)  
![Keras](https://img.shields.io/badge/Keras-DeepLearning-red)  
![Flask](https://img.shields.io/badge/Flask-WebApp-green)  
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)  


